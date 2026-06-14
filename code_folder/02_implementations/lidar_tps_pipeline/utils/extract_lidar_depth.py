#!/usr/bin/env python3
"""Extract LiDAR depth maps from AV2 tar and project onto each ring camera.

For each frame in frames.json:
  1. Find the nearest LiDAR sweep by timestamp (nanoseconds).
  2. Project ego-frame LiDAR points onto each camera using calibration.
  3. Save as uint16 PNG (1 unit = 4 mm, 0 = no return).

Output:
  <out_root>/sensors/depth/<cam_name>/<timestamp>.png
  <out_root>/depth_frames.json   -- same structure as frames.json, depth paths

Usage:
    python extract_lidar_depth.py \\
        --tar  ~/Downloads/train-000.tar \\
        --frames /home/Erik/mThesis/argo2_data/extracted/frames.json \\
        --calib  /home/Erik/mThesis/argo2_data/extracted/calibration.json \\
        --out    /home/Erik/mThesis/argo2_data/extracted
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import cv2

# 1 PNG unit = DEPTH_SCALE metres  (uint16 -> metres: val * DEPTH_SCALE)
DEPTH_SCALE = 0.004         # 4 mm per unit -> max 262 m

# Sparse disparity PNG: stores 1/depth_m scaled to uint16.
# 0 = no LiDAR return (sky, glass, large occlusions).
# DISP_SCALE chosen so max measurable disparity (0.5 m -> 2.0 m⁻¹) fits in uint16.
#   disp_u16 = round(disp_m_inv / DISP_SCALE)
#   disp_m_inv = disp_u16 * DISP_SCALE
#   depth_m = 1.0 / disp_m_inv  (when disp_u16 > 0)
DISP_SCALE = 2.0 / 65535   # ~ 3.05e-5  m⁻¹ per unit

# Hardware range limit of the AV2 Luminar Iris 128-ch LiDAR.
LIDAR_MAX_RANGE_M = 120.0   # conservative spec range for Luminar Iris

# Kept as an alias so callers that imported MAX_DEPTH_M still work.
MAX_DEPTH_M = LIDAR_MAX_RANGE_M


def quat_to_mat(qw, qx, qy, qz):
    """Unit quaternion -> 3x3 rotation (camera->ego)."""
    n = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    return np.array([
        [1-2*(qy**2+qz**2),   2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [  2*(qx*qy+qz*qw), 1-2*(qx**2+qz**2),   2*(qy*qz-qx*qw)],
        [  2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw), 1-2*(qx**2+qy**2)],
    ])


def project_lidar_to_depth(pts_ego: np.ndarray, cam: dict,
                            max_depth: float = LIDAR_MAX_RANGE_M) -> np.ndarray:
    """
    Project ego-frame LiDAR points onto a camera and return a uint16 depth map.

    Convention:
      R = camera->ego rotation  (from calibration egovehicle_SE3_sensor)
      t = camera position in ego frame

    The AV2 ring-camera JPEGs are stored already undistorted (distortion-corrected
    at capture time), so pure pin-hole projection is used -- no distortion polynomial.

    Args:
        pts_ego: (N, 3) float32 xyz in ego frame
        cam:     calibration dict with R, t, fx, fy, cx, cy, W, H
    Returns:
        depth_u16: (H, W) uint16, value = depth_m / DEPTH_SCALE, 0 = invalid
    """
    R  = cam['R']                        # (3,3) camera->ego
    t  = cam['t']                        # (3,)  camera origin in ego frame
    fx, fy = cam['fx'], cam['fy']
    cx, cy = cam['cx'], cam['cy']
    W, H   = cam['W'], cam['H']

    # -- 1. Transform to camera frame: p_cam = R^T @ (p_ego - t) ----------
    pts_c = (pts_ego - t) @ R            # R^T via right-multiply

    # -- 2. Depth filter: in front of camera, within range -----------------
    z = pts_c[:, 2]
    keep = (z > 0.5) & (z <= max_depth)
    pts_c, z = pts_c[keep], z[keep]
    if len(z) == 0:
        return np.zeros((H, W), dtype=np.uint16)

    # -- 3. Pin-hole projection --------------------------------------------
    # AV2 ring-camera JPEGs are stored already undistorted (distortion-corrected
    # at capture time).  Pure pin-hole projection is therefore correct; we do
    # NOT apply the Brown–Conrady distortion polynomial here.
    xn = pts_c[:, 0] / z
    yn = pts_c[:, 1] / z

    # -- 4. Pixel coords ---------------------------------------------------
    u_f = fx * xn + cx
    v_f = fy * yn + cy
    in_img = (u_f >= 0.0) & (u_f < W) & (v_f >= 0.0) & (v_f < H)
    u_f, v_f, z = u_f[in_img], v_f[in_img], z[in_img]
    if len(z) == 0:
        return np.zeros((H, W), dtype=np.uint16)

    ui = np.round(u_f).astype(np.int32)
    vi = np.round(v_f).astype(np.int32)
    # Clamp: rounding can push 0 -> -1 or W-1 -> W by at most 0.5
    ui = np.clip(ui, 0, W - 1)
    vi = np.clip(vi, 0, H - 1)

    # -- 6. Scatter depth -- nearest (smallest z) wins ----------------------
    order = np.argsort(z)[::-1]          # descending: near overwrites far
    flat  = np.zeros(H * W, dtype=np.float32)
    flat[vi[order] * W + ui[order]] = z[order]

    return np.clip(flat / DEPTH_SCALE, 0, 65535).astype(np.uint16).reshape(H, W)


def make_dense_depth(sparse_u16: np.ndarray,
                     max_depth_m: float = LIDAR_MAX_RANGE_M) -> np.ndarray:
    """Convert a sparse LiDAR depth map to a fully-dense depth map.

    Nearest-neighbour fill in disparity (1/depth) space.

    Args:
        sparse_u16:  (H, W) uint16, 0 = no return, value = depth_m / DEPTH_SCALE
        max_depth_m: fallback depth when the input is completely empty

    Returns:
        dense_u16:  (H, W) uint16, fully dense, same depth encoding
    """
    from scipy.ndimage import distance_transform_edt

    H, W    = sparse_u16.shape
    valid_b = sparse_u16 > 0

    if not valid_b.any():
        max_u16 = int(max_depth_m / DEPTH_SCALE)
        return np.full((H, W), max_u16, dtype=np.uint16)

    # Work in disparity (1/z) space so close objects dominate at boundaries
    d_m  = sparse_u16.astype(np.float32) * DEPTH_SCALE
    disp = np.where(valid_b, 1.0 / np.maximum(d_m, 0.1), 0.0)

    # NN fill: each invalid pixel copies the disparity of its nearest LiDAR hit
    _, nn_idx   = distance_transform_edt(~valid_b, return_indices=True)
    dense_disp  = disp[nn_idx[0], nn_idx[1]]

    # Convert back to depth uint16
    dense_m   = 1.0 / np.maximum(dense_disp, 1.0 / max_depth_m)
    dense_u16 = np.clip(dense_m / DEPTH_SCALE, 0, 65535).astype(np.uint16)

    # Restore original LiDAR hits exactly (no rounding drift)
    dense_u16[valid_b] = sparse_u16[valid_b]
    return dense_u16


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sensor-root', default='/home/Erik/mThesis/argo2_data/sensor',
                        help='Path to extracted sensor root (contains train/<log_id>/)')
    parser.add_argument('--frames', required=True,
                        help='Path to frames.json')
    parser.add_argument('--calib',  required=True,
                        help='Path to calibration.json')
    parser.add_argument('--out',    required=True,
                        help='Output root (same as frames.json directory)')
    args = parser.parse_args()

    out_root    = Path(args.out)
    sensor_root = Path(args.sensor_root)

    # -- Load calibration --------------------------------------------------
    with open(args.calib) as f:
        raw_calib = json.load(f)

    cams = {}
    for name, v in raw_calib.items():
        if 'ring_' not in name:
            continue
        cams[name] = {
            'R':  quat_to_mat(v['qw'], v['qx'], v['qy'], v['qz']),
            't':  np.array([v['tx_m'], v['ty_m'], v['tz_m']]),
            'fx': v['fx'], 'fy': v['fy'],
            'cx': v['cx'], 'cy': v['cy'],
            'k1': v.get('k1', 0.0), 'k2': v.get('k2', 0.0), 'k3': v.get('k3', 0.0),
            'W':  v['width'], 'H': v['height'],
        }
    cam_names = list(cams.keys())
    print(f'Cameras: {cam_names}')

    # -- Load frames -------------------------------------------------------
    with open(args.frames) as f:
        frames = json.load(f)
    print(f'Frames: {len(frames)}')

    # -- Index LiDAR sweeps from filesystem -------------------------------
    print(f'Indexing LiDAR sweeps in {sensor_root} ...')
    lidar_files: dict[int, Path] = {}   # timestamp_ns -> feather path

    for feather in sensor_root.glob('train/*/sensors/lidar/*.feather'):
        try:
            lidar_files[int(feather.stem)] = feather
        except ValueError:
            pass

    lidar_ts = np.array(sorted(lidar_files.keys()), dtype=np.int64)
    print(f'Found {len(lidar_ts)} LiDAR sweeps')

    # -- Create output directories -----------------------------------------
    for name in cam_names:
        (out_root / 'sensors' / 'disp' / name).mkdir(parents=True, exist_ok=True)

    # -- Process frames ----------------------------------------------------
    depth_frames = []

    for fi, frame in enumerate(frames):
        # Get camera timestamp from the first camera's image path
        first_cam = next(iter(frame))
        img_ts = int(Path(frame[first_cam]).stem)

        # Find nearest LiDAR sweep
        idx            = int(np.argmin(np.abs(lidar_ts - img_ts)))
        lidar_ts_match = lidar_ts[idx]
        dt_ms          = abs(int(lidar_ts_match) - img_ts) / 1e6

        if dt_ms > 55:
            print(f'  Frame {fi}: no nearby LiDAR sweep ({dt_ms:.0f} ms gap) -- skipping')
            depth_frames.append({})   # empty placeholder keeps indices aligned with frames.json
            continue

        # Load LiDAR sweep directly from filesystem
        df  = pd.read_feather(lidar_files[int(lidar_ts_match)])
        pts = df[['x', 'y', 'z']].values.astype(np.float32)

        # Project onto each camera and save dense disparity (NN-in-disparity fill)
        depth_entry: dict[str, str] = {}
        for cam_name in cam_names:
            if cam_name not in frame:
                continue

            # Sparse depth (uint16, 0 = no return)
            depth_u16 = project_lidar_to_depth(pts, cams[cam_name])

            # Dense fill in disparity space: NN propagates each hit to nearby pixels.
            # Result: every pixel has a depth value (no zeros except empty frame).
            dense_u16 = make_dense_depth(depth_u16)

            # Convert dense depth -> disparity encoding for the CNN channel:
            #   disp_u16 = round((1/depth_m) / DISP_SCALE)
            # Close objects -> high values, far -> low, encoding is monotone in 1/z.
            d_m  = dense_u16.astype(np.float32) * DEPTH_SCALE
            disp = np.clip(
                np.round((1.0 / np.maximum(d_m, 0.001)) / DISP_SCALE), 1, 65535
            ).astype(np.uint16)

            cam_ts   = int(Path(frame[cam_name]).stem)
            out_path = out_root / 'sensors' / 'disp' / cam_name / f'{cam_ts}.png'
            cv2.imwrite(str(out_path), disp)
            depth_entry[cam_name] = str(out_path)

        depth_frames.append(depth_entry)

        if fi % 20 == 0 or fi == len(frames) - 1:
            print(f'  [{fi+1:3d}/{len(frames)}]  '
                  f'lidar_ts={lidar_ts_match}  dt={dt_ms:.1f}ms  '
                  f'pts={len(pts)}')

    # -- Write depth_frames.json -------------------------------------------
    out_json = out_root / 'depth_frames.json'
    with open(out_json, 'w') as f:
        json.dump(depth_frames, f)
    print(f'\nSaved {len(depth_frames)} disparity frame entries -> {out_json}')
    print(f'Disparity scale: {DISP_SCALE:.2e} m⁻¹/unit')
    print(f'Read back: disp_m_inv = uint16 * {DISP_SCALE:.2e}, depth_m = 1/disp_m_inv (when nonzero)')


if __name__ == '__main__':
    main()
