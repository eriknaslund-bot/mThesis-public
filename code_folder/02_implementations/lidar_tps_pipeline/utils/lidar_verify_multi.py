#!/usr/bin/env python3
"""Multi-sample LiDAR projection + dense depth verification.

Panel A (top rows): projection dots for N samples across 3 front cameras
Panel B (bottom rows): sparse depth -> dense depth comparison for 2 cameras,
         highlighting ground-vs-sky fill correctness.

Usage:
    python lidar_verify_multi.py
    python lidar_verify_multi.py --n-samples 6
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from extract_lidar_depth import (
    LIDAR_MAX_RANGE_M, DEPTH_SCALE, quat_to_mat,
    project_lidar_to_depth, make_dense_depth,
)

SENSOR_ROOT = Path.home() / 'mThesis/argo2_data/sensor'
CALIB_JSON  = Path.home() / 'mThesis/argo2_data/extracted/calibration.json'
OUT_DIR     = Path(__file__).parent.parent / 'output'
TARGET_W    = 900


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
    t = np.clip((z - lo) / (hi - lo), 0, 1)
    t = 1 - t
    r = np.clip(1.5 - np.abs(4*t - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4*t - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4*t - 1), 0, 1)
    return (np.stack([b, g, r], axis=1) * 255).astype(np.uint8)


def overlay_dots(img, uvz, radius=2):
    out = img.copy()
    if len(uvz) == 0:
        return out
    order = np.argsort(uvz[:, 2])[::-1]
    colors = depth_color(uvz[order, 2])
    H, W = out.shape[:2]
    for (u, v, _), c in zip(uvz[order], colors):
        px, py = int(round(u)), int(round(v))
        if 0 <= px < W and 0 <= py < H:
            cv2.circle(out, (px, py), radius, (int(c[0]), int(c[1]), int(c[2])), -1)
    return out


def depth_to_color(depth_u16, max_val=None):
    """uint16 depth -> coloured BGR image. 0=black, low=red, high=blue."""
    d = depth_u16.astype(np.float32) * DEPTH_SCALE
    if max_val is None:
        max_val = LIDAR_MAX_RANGE_M
    # Normalize: 0->black, near->red/warm, far->blue/cool, max->purple
    valid = d > 0
    norm = np.zeros_like(d)
    norm[valid] = np.clip(d[valid] / max_val, 0, 1)
    # Use TURBO colormap (better perceptual uniformity)
    u8 = (norm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    # Black where no data (sparse) or truly zero
    colored[~valid] = 0
    return colored


def label(img, text, y=22, color=(255, 255, 255)):
    cv2.putText(img, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
    cv2.putText(img, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def resize_w(img, target_w):
    H, W = img.shape[:2]
    sc = target_w / W
    return cv2.resize(img, (target_w, int(H * sc)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sensor-root', default=str(SENSOR_ROOT))
    ap.add_argument('--calib',       default=str(CALIB_JSON))
    ap.add_argument('--n-samples',   type=int, default=6)
    args = ap.parse_args()

    sensor_root = Path(args.sensor_root)
    cams = load_calib(args.calib)
    N = args.n_samples

    # -- Index LiDAR sweeps ------------------------------------------------
    lidar_files = {}
    for f in sensor_root.glob('train/*/sensors/lidar/*.feather'):
        try:
            lidar_files[int(f.stem)] = f
        except ValueError:
            pass
    lidar_ts = np.array(sorted(lidar_files.keys()), dtype=np.int64)
    print(f'LiDAR sweeps: {len(lidar_ts)}')

    # -- Index camera images -----------------------------------------------
    proj_cams  = ['ring_front_left', 'ring_front_center', 'ring_front_right']
    depth_cams = ['ring_front_center', 'ring_side_left']

    cam_imgs = {}
    for cam_name in set(proj_cams + depth_cams):
        imgs = sorted(sensor_root.glob(f'train/*/sensors/cameras/{cam_name}/*.jpg'))
        cam_imgs[cam_name] = imgs
    n_imgs = len(cam_imgs[proj_cams[0]])

    # Pick N evenly-spaced samples
    indices = np.linspace(0, n_imgs - 1, N, dtype=int)
    print(f'Using {N} samples from {n_imgs} images: indices={list(indices)}')

    # ═══════════════════════════════════════════════════════════════════════
    # PART A: Projection dot overlay (3 front cameras x N samples)
    # ═══════════════════════════════════════════════════════════════════════
    print('\n=== Part A: Projection overlay ===')
    proj_rows = []

    for si, idx in enumerate(indices):
        img_path = cam_imgs[proj_cams[0]][idx]
        img_ts   = int(img_path.stem)
        li       = int(np.argmin(np.abs(lidar_ts - img_ts)))
        best_ts  = int(lidar_ts[li])
        dt_ms    = abs(best_ts - img_ts) / 1e6

        if dt_ms > 100:
            print(f'  Sample {si}: dt={dt_ms:.0f}ms -- skip')
            continue

        df  = pd.read_feather(lidar_files[best_ts])
        pts = df[['x', 'y', 'z']].values.astype(np.float32)

        panels = []
        for cam_name in proj_cams:
            # Find the image with closest timestamp for this camera
            cam_path = cam_imgs[cam_name][idx]
            img = cv2.imread(str(cam_path))
            if img is None:
                continue

            uvz   = project(pts, cams[cam_name])
            panel = overlay_dots(img, uvz, radius=2)
            panel = resize_w(panel, TARGET_W)
            label(panel, f'S{si} {cam_name} {len(uvz)}pts dt={dt_ms:.0f}ms')
            panels.append(panel)

        if panels:
            # Pad to same height and hstack
            max_h = max(p.shape[0] for p in panels)
            padded = []
            for p in panels:
                if p.shape[0] < max_h:
                    pad = np.zeros((max_h - p.shape[0], p.shape[1], 3), np.uint8)
                    p = np.vstack([p, pad])
                padded.append(p)
            proj_rows.append(np.hstack(padded))
            print(f'  Sample {si}: idx={idx} dt={dt_ms:.1f}ms pts={len(pts)}')

    # ═══════════════════════════════════════════════════════════════════════
    # PART B: Sparse vs Dense depth maps (check fill quality)
    # ═══════════════════════════════════════════════════════════════════════
    print('\n=== Part B: Sparse vs Dense depth ===')
    depth_rows = []

    # Use 4 spread-out samples for depth analysis
    depth_indices = np.linspace(0, n_imgs - 1, 4, dtype=int)

    for si, idx in enumerate(depth_indices):
        img_path = cam_imgs[depth_cams[0]][idx]
        img_ts   = int(img_path.stem)
        li       = int(np.argmin(np.abs(lidar_ts - img_ts)))
        best_ts  = int(lidar_ts[li])
        dt_ms    = abs(best_ts - img_ts) / 1e6

        if dt_ms > 100:
            continue

        df  = pd.read_feather(lidar_files[best_ts])
        pts = df[['x', 'y', 'z']].values.astype(np.float32)

        panels = []
        for cam_name in depth_cams:
            cam = cams[cam_name]
            cam_path = cam_imgs[cam_name][idx]
            img = cv2.imread(str(cam_path))
            if img is None:
                continue

            # Sparse depth
            sparse = project_lidar_to_depth(pts, cam)
            n_hits = (sparse > 0).sum()

            # Dense depth
            dense = make_dense_depth(sparse)

            # -- Analysis: check ground vs sky fill ------------------------
            H, W = sparse.shape
            # Bottom 30% = ground region
            ground_rows = slice(int(H * 0.7), H)
            # Top 20% = sky region
            sky_rows = slice(0, int(H * 0.2))

            ground_sparse_valid = (sparse[ground_rows] > 0).sum()
            ground_dense  = dense[ground_rows].astype(np.float32) * DEPTH_SCALE
            ground_mean   = ground_dense.mean()
            ground_max_pct = (ground_dense > 80).sum() / ground_dense.size * 100

            sky_dense = dense[sky_rows].astype(np.float32) * DEPTH_SCALE
            sky_mean  = sky_dense.mean()
            sky_max_pct = (sky_dense > 100).sum() / sky_dense.size * 100

            print(f'  {cam_name} S{si}: hits={n_hits} '
                  f'ground(mean={ground_mean:.1f}m, >{80}m:{ground_max_pct:.1f}%) '
                  f'sky(mean={sky_mean:.1f}m, >{100}m:{sky_max_pct:.1f}%)')

            # -- Visualize -------------------------------------------------
            # 1. Original image (small)
            p_img = resize_w(img, TARGET_W // 3)

            # 2. Sparse depth (colored, black=no data)
            p_sparse = resize_w(depth_to_color(sparse, max_val=80), TARGET_W // 3)
            label(p_sparse, f'sparse {n_hits}pts')

            # 3. Dense depth (colored)
            p_dense = resize_w(depth_to_color(dense, max_val=80), TARGET_W // 3)

            # Mark regions where dense depth > 80m in ground zone as red overlay
            dense_full_color = depth_to_color(dense, max_val=80)
            bad_ground = np.zeros_like(dense_full_color)
            gslice = slice(int(H * 0.7), H)
            d_m = dense[gslice].astype(np.float32) * DEPTH_SCALE
            bad_mask = d_m > 80
            bad_ground[gslice][bad_mask] = [0, 0, 255]  # red
            p_dense_check = resize_w(
                cv2.addWeighted(dense_full_color, 1.0, bad_ground, 0.5, 0), TARGET_W // 3)
            label(p_dense_check,
                  f'dense gnd_mean={ground_mean:.0f}m sky_mean={sky_mean:.0f}m')

            # Pad heights to match
            max_h = max(p_img.shape[0], p_sparse.shape[0], p_dense_check.shape[0])
            for p in [p_img, p_sparse, p_dense_check]:
                if p.shape[0] < max_h:
                    pad = np.zeros((max_h - p.shape[0], p.shape[1], 3), np.uint8)
                    p = np.vstack([p, pad])

            row = np.hstack([
                p_img if p_img.shape[0] == max_h else np.vstack([p_img, np.zeros((max_h - p_img.shape[0], p_img.shape[1], 3), np.uint8)]),
                p_sparse if p_sparse.shape[0] == max_h else np.vstack([p_sparse, np.zeros((max_h - p_sparse.shape[0], p_sparse.shape[1], 3), np.uint8)]),
                p_dense_check if p_dense_check.shape[0] == max_h else np.vstack([p_dense_check, np.zeros((max_h - p_dense_check.shape[0], p_dense_check.shape[1], 3), np.uint8)]),
            ])
            panels.append(row)

        if panels:
            # Stack cameras for this sample vertically
            max_w = max(p.shape[1] for p in panels)
            padded = []
            for p in panels:
                if p.shape[1] < max_w:
                    pad = np.zeros((p.shape[0], max_w - p.shape[1], 3), np.uint8)
                    p = np.hstack([p, pad])
                padded.append(p)
            depth_rows.append(np.vstack(padded))

    # ═══════════════════════════════════════════════════════════════════════
    # Save Part A
    # ═══════════════════════════════════════════════════════════════════════
    if proj_rows:
        out_a = np.vstack(proj_rows)
        path_a = OUT_DIR / 'lidar_verify_multi.png'
        path_a.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path_a), out_a)
        print(f'\nPart A -> {path_a}  ({out_a.shape[1]}x{out_a.shape[0]})')

    # ═══════════════════════════════════════════════════════════════════════
    # Save Part B
    # ═══════════════════════════════════════════════════════════════════════
    if depth_rows:
        max_w = max(r.shape[1] for r in depth_rows)
        padded = []
        for r in depth_rows:
            if r.shape[1] < max_w:
                pad = np.zeros((r.shape[0], max_w - r.shape[1], 3), np.uint8)
                r = np.hstack([r, pad])
            padded.append(r)
        out_b = np.vstack(padded)
        path_b = OUT_DIR / 'depth_sparse_vs_dense.png'
        cv2.imwrite(str(path_b), out_b)
        print(f'Part B -> {path_b}  ({out_b.shape[1]}x{out_b.shape[0]})')


if __name__ == '__main__':
    main()
