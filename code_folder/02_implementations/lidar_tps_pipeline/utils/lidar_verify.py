#!/usr/bin/env python3
"""LiDAR projection verification -- 1px dots on all ring cameras.

Reads LiDAR and images from the extracted filesystem (argo2_data/sensor/).
Outputs a stacked panel of all ring cameras with depth-coloured 1px dots.

Usage:
    python lidar_verify.py
    python lidar_verify.py --sensor-root ~/mThesis/argo2_data/sensor \
                           --calib ~/mThesis/argo2_data/extracted/calibration.json \
                           --out output/lidar_verify.png
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from extract_lidar_depth import LIDAR_MAX_RANGE_M, quat_to_mat

SENSOR_ROOT = Path.home() / 'mThesis/argo2_data/sensor'
CALIB_JSON  = Path.home() / 'mThesis/argo2_data/extracted/calibration.json'
OUT         = Path(__file__).parent.parent / 'output/lidar_verify.png'
TARGET_W    = 1280   # output width per camera panel


def load_calib(path):
    with open(path) as f:
        raw = json.load(f)
    cams = {}
    for name, v in raw.items():
        if 'ring_' not in name:
            continue
        cams[name] = {
            'R':  quat_to_mat(v['qw'], v['qx'], v['qy'], v['qz']),
            't':  np.array([v['tx_m'], v['ty_m'], v['tz_m']]),
            'fx': v['fx'], 'fy': v['fy'],
            'cx': v['cx'], 'cy': v['cy'],
            'W':  v['width'], 'H': v['height'],
        }
    return cams


def project(pts_ego, cam):
    pts_c = (pts_ego - cam['t']) @ cam['R']
    z = pts_c[:, 2]
    keep = (z > 0.5) & (z <= LIDAR_MAX_RANGE_M)
    pts_c, z = pts_c[keep], z[keep]
    if len(z) == 0:
        return np.empty((0, 3))
    u = cam['fx'] * pts_c[:, 0] / z + cam['cx']
    v = cam['fy'] * pts_c[:, 1] / z + cam['cy']
    W, H = cam['W'], cam['H']
    m = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return np.column_stack([u[m], v[m], z[m]])


def depth_color(z, lo=0.5, hi=60.0):
    """Jet-like: red=near, blue=far."""
    t = np.clip((z - lo) / (hi - lo), 0, 1)
    t = 1 - t  # close -> 1
    r = np.clip(1.5 - np.abs(4*t - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4*t - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4*t - 1), 0, 1)
    return (np.stack([b, g, r], axis=1) * 255).astype(np.uint8)


def overlay_dots(img, uvz, radius=3):
    """Draw filled circles per LiDAR hit; far-to-near so near overwrites."""
    out = img.copy()
    if len(uvz) == 0:
        return out
    order  = np.argsort(uvz[:, 2])[::-1]   # far first
    colors = depth_color(uvz[order, 2])
    H, W = out.shape[:2]
    for (u, v, _), c in zip(uvz[order], colors):
        px, py = int(round(u)), int(round(v))
        if 0 <= px < W and 0 <= py < H:
            cv2.circle(out, (px, py), radius, (int(c[0]), int(c[1]), int(c[2])), -1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sensor-root', default=str(SENSOR_ROOT))
    ap.add_argument('--calib',       default=str(CALIB_JSON))
    ap.add_argument('--out',         default=str(OUT))
    args = ap.parse_args()

    sensor_root = Path(args.sensor_root)
    cams = load_calib(args.calib)

    # -- Index LiDAR sweeps ------------------------------------------------
    lidar_files = {}
    for f in sensor_root.glob('train/*/sensors/lidar/*.feather'):
        try:
            lidar_files[int(f.stem)] = f
        except ValueError:
            pass
    lidar_ts = np.array(sorted(lidar_files.keys()), dtype=np.int64)
    print(f'LiDAR sweeps indexed: {len(lidar_ts)}')

    # -- Index camera images (first image per camera) ----------------------
    cam_order = [
        'ring_front_left', 'ring_front_center', 'ring_front_right',
        'ring_side_left', 'ring_side_right', 'ring_rear_left', 'ring_rear_right',
    ]
    first_imgs = {}
    for cam_name in cam_order:
        imgs = sorted(sensor_root.glob(f'train/*/sensors/cameras/{cam_name}/*.jpg'))
        if imgs:
            first_imgs[cam_name] = imgs[0]

    if not first_imgs:
        print('No images found'); return

    # Use timestamp from first available camera to find matching LiDAR
    ref_cam = next(iter(first_imgs))
    img_ts  = int(first_imgs[ref_cam].stem)
    idx     = int(np.argmin(np.abs(lidar_ts - img_ts)))
    best_ts = int(lidar_ts[idx])
    dt_ms   = abs(best_ts - img_ts) / 1e6
    print(f'Image ts={img_ts}  LiDAR ts={best_ts}  dt={dt_ms:.1f}ms')

    # -- Load LiDAR --------------------------------------------------------
    df  = pd.read_feather(lidar_files[best_ts])
    pts = df[['x', 'y', 'z']].values.astype(np.float32)
    print(f'LiDAR pts: {len(pts)}  z∈[{pts[:,2].min():.1f}, {pts[:,2].max():.1f}]')

    # -- Per-camera panels -------------------------------------------------
    panels = []
    for cam_name in cam_order:
        if cam_name not in first_imgs or cam_name not in cams:
            continue

        img = cv2.imread(str(first_imgs[cam_name]))
        if img is None:
            continue

        uvz  = project(pts, cams[cam_name])
        panel = overlay_dots(img, uvz, radius=3)

        # Resize to TARGET_W
        H0, W0 = panel.shape[:2]
        sc  = TARGET_W / W0
        panel = cv2.resize(panel, (TARGET_W, int(H0 * sc)))

        cv2.putText(panel, f'{cam_name}  {len(uvz)} hits  dt={dt_ms:.0f}ms',
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(panel, f'{cam_name}  {len(uvz)} hits  dt={dt_ms:.0f}ms',
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        panels.append(panel)
        print(f'  {cam_name}: {len(uvz)} hits')

    if not panels:
        print('No panels generated'); return

    out_img = np.vstack(panels)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out_img)
    print(f'\nSaved -> {out_path}  ({out_img.shape[1]}x{out_img.shape[0]})')


if __name__ == '__main__':
    main()
