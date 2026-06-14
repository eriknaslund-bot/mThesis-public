#!/usr/bin/env python3
"""Verify LiDAR cross-camera alignment for a handful of samples.

Produces a side-by-side image for each sample:
  Left panel:  img_l  with LiDAR dots (magenta)
  Middle panel: img_r (original) with LiDAR dots (cyan)
  Right panel: img_r after LiDAR TPS warp -- should align with img_l

Usage:
    python verify_lidar_align.py \
        --frames /home/Erik/mThesis/argo2_data/training/combined_frames.json \
        --calib  /home/Erik/mThesis/argo2_data/training/calibration.json \
        --pair 1 --n_samples 4 --out output/lidar_align_verify
"""
import argparse
import json
import os
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import AV2PairDataset, ADJACENT_PAIRS
from compute_overlaps import load_calib


def draw_pts(img_bgr, u, v, color, radius=3):
    for x, y in zip(u.astype(int), v.astype(int)):
        cv2.circle(img_bgr, (x, y), radius, color, -1)
    return img_bgr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames',    required=True)
    parser.add_argument('--calib',     required=True)
    parser.add_argument('--pair',      type=int, default=1,
                        help='Pair index 0-4 (default 1 = side_left<->front_left)')
    parser.add_argument('--n_samples', type=int, default=4)
    parser.add_argument('--img_h',     type=int, default=512)
    parser.add_argument('--img_w',     type=int, default=704)
    parser.add_argument('--out',       default='output/lidar_align_verify')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    net_h, net_w = args.img_h, args.img_w

    with open(args.frames) as f:
        frames = json.load(f)
    cams = load_calib(args.calib)

    left_cam, right_cam = ADJACENT_PAIRS[args.pair]
    print(f'Pair {args.pair}: {left_cam} <-> {right_cam}')

    # Build LiDAR index
    lidar_paths = AV2PairDataset._index_lidar_paths(frames)
    n_found = sum(1 for p in lidar_paths if p is not None)
    print(f'LiDAR sweeps indexed: {n_found}/{len(frames)}')

    cam_l = cams[left_cam]
    cam_r = cams[right_cam]

    count = 0
    for fi, frame in enumerate(frames):
        if count >= args.n_samples:
            break
        if left_cam not in frame or right_cam not in frame:
            continue
        if lidar_paths[fi] is None:
            continue

        import pandas as pd
        from PIL import Image
        import torchvision.transforms.functional as TF

        df      = pd.read_feather(lidar_paths[fi])
        pts_ego = df[['x', 'y', 'z']].values.astype(np.float32)

        # Load and resize images
        img_l = np.array(Image.open(frame[left_cam]).convert('RGB')
                         .resize((net_w, net_h), Image.BILINEAR))
        img_r = np.array(Image.open(frame[right_cam]).convert('RGB')
                         .resize((net_w, net_h), Image.BILINEAR))

        # Project LiDAR into both cameras at original resolution, then scale
        u_l, v_l, mask_l = AV2PairDataset._project_to_cam(pts_ego, cam_l)
        u_r, v_r, mask_r = AV2PairDataset._project_to_cam(pts_ego, cam_r)
        both = mask_l & mask_r

        u_l_sc = u_l[both] * (net_w / cam_l['W'])
        v_l_sc = v_l[both] * (net_h / cam_l['H'])
        u_r_sc = u_r[both] * (net_w / cam_r['W'])
        v_r_sc = v_r[both] * (net_h / cam_r['H'])

        print(f'  Frame {fi}: {int(both.sum())} cross-camera LiDAR points')

        # TPS warp
        warped, n_pts = AV2PairDataset._lidar_tps_warp(
            img_r, pts_ego, cam_l, cam_r, net_h, net_w)

        # Draw dots on BGR copies
        il_bgr = cv2.cvtColor(img_l, cv2.COLOR_RGB2BGR)
        ir_bgr = cv2.cvtColor(img_r, cv2.COLOR_RGB2BGR)
        draw_pts(il_bgr, u_l_sc, v_l_sc, (255, 0, 255))   # magenta on img_l
        draw_pts(ir_bgr, u_r_sc, v_r_sc, (255, 255, 0))   # cyan on img_r

        if warped is not None:
            iw_bgr = cv2.cvtColor(warped, cv2.COLOR_RGB2BGR)
            cv2.putText(iw_bgr, f'Calib H ({n_pts} cross-pts)',
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv2.putText(iw_bgr, f'Calib H ({n_pts} cross-pts)',
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        else:
            iw_bgr = np.zeros_like(il_bgr)
            cv2.putText(iw_bgr, 'TPS FAILED',
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(il_bgr, f'img_l  {left_cam}',
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(il_bgr, f'img_l  {left_cam}',
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(ir_bgr, f'img_r  {right_cam}',
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(ir_bgr, f'img_r  {right_cam}',
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        panel = np.hstack([il_bgr, ir_bgr, iw_bgr])
        out_path = os.path.join(args.out, f'pair{args.pair}_frame{fi:04d}.jpg')
        cv2.imwrite(out_path, panel)
        print(f'  Saved -> {out_path}')
        count += 1

    print(f'\nDone. {count} samples saved to {args.out}/')


if __name__ == '__main__':
    main()
