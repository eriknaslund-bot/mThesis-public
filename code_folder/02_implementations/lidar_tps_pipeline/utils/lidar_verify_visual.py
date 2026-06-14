#!/usr/bin/env python3
"""Visual verification: source | LiDAR overlay | dense depth  (one row per sample/camera).

Usage:
    python lidar_verify_visual.py
    python lidar_verify_visual.py --n-samples 4 --cam ring_front_center
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
PANEL_W     = 860   # width of each of the 3 panels


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


def overlay_dots(img, uvz, radius=3):
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


def depth_to_colormap(depth_u16):
    """Dense uint16 depth -> coloured BGR. Near=red/warm, far=blue/cool."""
    d_m = depth_u16.astype(np.float32) * DEPTH_SCALE
    # Invert: near->1->red, far->0->blue in TURBO
    norm = 1.0 - np.clip(d_m / LIDAR_MAX_RANGE_M, 0, 1)
    u8   = (norm * 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)


def resize_w(img, w):
    H, W = img.shape[:2]
    return cv2.resize(img, (w, int(H * w / W)))


def label(img, text, y=26):
    cv2.putText(img, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(img, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sensor-root', default=str(SENSOR_ROOT))
    ap.add_argument('--calib',       default=str(CALIB_JSON))
    ap.add_argument('--cam',         default='ring_front_center')
    ap.add_argument('--n-samples',   type=int, default=5)
    ap.add_argument('--out',         default=str(OUT_DIR / 'lidar_visual.png'))
    args = ap.parse_args()

    sensor_root = Path(args.sensor_root)
    cams = load_calib(args.calib)
    cam  = cams[args.cam]

    # Index LiDAR
    lidar_files = {}
    for f in sensor_root.glob('train/*/sensors/lidar/*.feather'):
        try:
            lidar_files[int(f.stem)] = f
        except ValueError:
            pass
    lidar_ts = np.array(sorted(lidar_files.keys()), dtype=np.int64)

    # Index images
    imgs = sorted(sensor_root.glob(f'train/*/sensors/cameras/{args.cam}/*.jpg'))
    n = len(imgs)
    indices = np.linspace(0, n - 1, args.n_samples, dtype=int)
    print(f'{args.cam}: {n} images, showing {args.n_samples} samples')

    rows = []
    for idx in indices:
        img_path = imgs[idx]
        img_ts   = int(img_path.stem)
        li       = int(np.argmin(np.abs(lidar_ts - img_ts)))
        best_ts  = int(lidar_ts[li])
        dt_ms    = abs(best_ts - img_ts) / 1e6

        if dt_ms > 100:
            print(f'  idx={idx}: dt={dt_ms:.0f}ms -- skip')
            continue

        df  = pd.read_feather(lidar_files[best_ts])
        pts = df[['x', 'y', 'z']].values.astype(np.float32)

        img = cv2.imread(str(img_path))

        # 1. Source
        p1 = resize_w(img.copy(), PANEL_W)
        label(p1, f'{args.cam}  sample {idx}')

        # 2. LiDAR overlay
        uvz = project(pts, cam)
        p2  = resize_w(overlay_dots(img, uvz, radius=3), PANEL_W)
        label(p2, f'LiDAR projection  {len(uvz)} hits  dt={dt_ms:.0f}ms')

        # 3. Dense depth map
        sparse = project_lidar_to_depth(pts, cam)
        dense  = make_dense_depth(sparse)
        p3     = resize_w(depth_to_colormap(dense), PANEL_W)
        label(p3, f'Dense depth  0m(red) -> {LIDAR_MAX_RANGE_M:.0f}m(blue/dark)')

        # Pad to same height and hstack
        target_h = max(p1.shape[0], p2.shape[0], p3.shape[0])
        def pad_h(p):
            if p.shape[0] < target_h:
                pad = np.zeros((target_h - p.shape[0], p.shape[1], 3), np.uint8)
                return np.vstack([p, pad])
            return p

        rows.append(np.hstack([pad_h(p1), pad_h(p2), pad_h(p3)]))
        print(f'  idx={idx}  dt={dt_ms:.1f}ms  hits={len(uvz)}')

    out = np.vstack(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out)
    print(f'\nSaved -> {out_path}  ({out.shape[1]}x{out.shape[0]})')


if __name__ == '__main__':
    main()
