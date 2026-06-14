#!/usr/bin/env python3
"""Generate stitched panorama for every sweep config x 3 frames.

Output: output/sweep_visuals/frame_{N:04d}/cfg_{i:03d}_mcr{mcr}_mdp{mdp}_ga{ga}_sm{sm}.jpg
        (91 configs x 3 frames = 273 JPEGs)

Usage
-----
    python gen_sweep_visuals.py
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from lidar_ring_stitch import (
    FRONT_CAMS, paste_with_feather, FEATHER_HALF,
)
from sweep_params import (
    load_frame_data, CONFIGS, SENSOR_ROOT, CALIB_JSON, FRAMES_JSON,
)
from lidar_ring_stitch import build_tps_remap

FRAMES_TO_RENDER = [0, 100, 200]
# The deployed pipeline is asymmetric (FC at rotation baseline, FL/FR warped).
# The historical sym/mid_alpha sweep was removed when the symmetric path was
# deleted; the keys remain in each cfg for back-compat with build_tps_remap.
_SYM_CONFIGS = [
    dict(min_ctrl_range=12, tps_smoothing=0, real_overlap_mask=True, symmetric=False, mid_alpha=1.0),
]
OUT_ROOT = Path(__file__).parent.parent / 'output/sweep_visuals'
JPEG_QUALITY = 90


def warp_and_stitch(cfg: dict, fd: dict) -> np.ndarray:
    """Warp all three cameras and composite with DP seam. Returns BGR uint8."""
    cams       = fd['cams']
    images     = fd['images']
    f_cyl      = fd['f_cyl']
    Z_REF      = fd['Z_REF']
    W_canvas   = fd['W_canvas']
    H_canvas   = fd['H_canvas']
    cx_canvas  = fd['cx_canvas']
    cy_canvas  = fd['cy_canvas']
    rot_remaps = fd['rot_remaps']
    rot_valid  = fd['rot_valid']
    proj       = fd['proj']
    FL, FC, FR = fd['FL'], fd['FC'], fd['FR']

    overlap_masks = (fd['real_overlap_masks'] if cfg['real_overlap_mask']
                     else fd['strip_overlap_masks'])

    from lidar_ring_stitch import find_shared_ctrl_pts, cam_pixel_to_canvas_rot
    sh_px_FL, sh_px_FC_l, sh_cvs_FL_FC = find_shared_ctrl_pts(
        *proj[FL], *proj[FC],
        f_cyl, cx_canvas, cy_canvas, Z_REF,
        min_range_m=cfg['min_ctrl_range'],
        ref_cam=cams[FC], ref_side='right')
    sh_px_FC_r, sh_px_FR, sh_cvs_FC_FR = find_shared_ctrl_pts(
        *proj[FC], *proj[FR],
        f_cyl, cx_canvas, cy_canvas, Z_REF,
        min_range_m=cfg['min_ctrl_range'],
        ref_cam=cams[FC], ref_side='left')

    symmetric = cfg.get('symmetric', False)
    mid_alpha  = cfg.get('mid_alpha', 0.5)

    if symmetric:
        alpha = mid_alpha
        if len(sh_px_FL):
            u, v = cam_pixel_to_canvas_rot(
                sh_px_FL[:, 0], sh_px_FL[:, 1], cams[FL], f_cyl, cx_canvas, cy_canvas)
            cvs_fl = np.column_stack([u, v]).astype(np.float32)
            sh_cvs_FL_FC = alpha * sh_cvs_FL_FC + (1 - alpha) * cvs_fl
        sh_cvs_FC_l = sh_cvs_FL_FC.copy()
        if len(sh_px_FR):
            u, v = cam_pixel_to_canvas_rot(
                sh_px_FR[:, 0], sh_px_FR[:, 1], cams[FR], f_cyl, cx_canvas, cy_canvas)
            cvs_fr = np.column_stack([u, v]).astype(np.float32)
            sh_cvs_FC_FR = alpha * sh_cvs_FC_FR + (1 - alpha) * cvs_fr
        sh_cvs_FC_r = sh_cvs_FC_FR.copy()
        fc_src = (np.vstack([sh_px_FC_l, sh_px_FC_r])
                  if len(sh_px_FC_l) and len(sh_px_FC_r)
                  else (sh_px_FC_l if len(sh_px_FC_l) else sh_px_FC_r))
        fc_dst = (np.vstack([sh_cvs_FC_l, sh_cvs_FC_r])
                  if len(sh_cvs_FC_l) and len(sh_cvs_FC_r)
                  else (sh_cvs_FC_l if len(sh_cvs_FC_l) else sh_cvs_FC_r))
        shared_src = {FL: sh_px_FL, FC: fc_src, FR: sh_px_FR}
        shared_dst = {FL: sh_cvs_FL_FC, FC: fc_dst, FR: sh_cvs_FC_FR}
    else:
        shared_src = {FL: sh_px_FL, FC: np.empty((0, 2), np.float32), FR: sh_px_FR}
        shared_dst = {FL: sh_cvs_FL_FC, FC: np.empty((0, 2), np.float32), FR: sh_cvs_FC_FR}

    warped: dict[str, np.ndarray] = {}
    valid:  dict[str, np.ndarray] = {}

    for name in FRONT_CAMS:
        cam = cams[name]
        W_cam, H_cam = cam['W'], cam['H']
        img = images[name]

        if name == FC and not symmetric:
            mx, my = rot_remaps[name]
        else:
            src_pts = shared_src[name].copy()
            dst_pts = shared_dst[name].copy()

            if len(src_pts) < 4:
                mx, my = rot_remaps[name]
            else:
                mx, my = build_tps_remap(
                    src_pts, dst_pts, W_cam, H_cam, W_canvas, H_canvas,
                    cam=cam, f_cyl=f_cyl,
                    cx_canvas=cx_canvas, cy_canvas=cy_canvas,
                    smoothing=cfg['tps_smoothing'], remap_scale=0.1,
                    overlap_mask=overlap_masks[name])

        warped[name] = cv2.remap(img, mx, my, cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        valid[name]  = rot_valid[name]

    # Composite: paste each camera in order with feather blend (same as
    # lidar_ring_stitch.py default -- no seam DP to avoid black-strip artefacts)
    canvas = np.zeros((H_canvas, W_canvas, 3), dtype=np.uint8)
    for name in FRONT_CAMS:
        paste_with_feather(canvas, warped[name], valid[name],
                           feather_half=FEATHER_HALF)
    return canvas


def cfg_label(i: int, cfg: dict) -> str:
    rom = int(cfg['real_overlap_mask'])
    sym = f"_sym{int(cfg.get('symmetric', False))}_a{int(cfg.get('mid_alpha', 0.5) * 100):02d}"
    return (f"cfg_{i:03d}"
            f"_mcr{int(cfg['min_ctrl_range']):02d}"
            f"_sm{int(cfg['tps_smoothing']):02d}"
            f"_rom{rom}"
            f"{sym}")


def main():
    n_cfg    = len(_SYM_CONFIGS)
    n_frames = len(FRAMES_TO_RENDER)
    total    = n_cfg * n_frames
    done     = 0

    for frame_idx in FRAMES_TO_RENDER:
        out_dir = OUT_ROOT / f'frame_{frame_idx:04d}'
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f'\n{"="*60}')
        print(f'Loading frame {frame_idx}…')
        fd = load_frame_data(frame_idx, SENSOR_ROOT, str(CALIB_JSON),
                             str(FRAMES_JSON))

        for i, cfg in enumerate(_SYM_CONFIGS):
            label    = cfg_label(i, cfg)
            out_path = out_dir / f'{label}.jpg'

            if out_path.exists():
                done += 1
                print(f'  [{done:3d}/{total}] {label}  (cached)')
                continue

            canvas = warp_and_stitch(cfg, fd)
            cv2.imwrite(str(out_path), canvas,
                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            done += 1
            print(f'  [{done:3d}/{total}] {label}  '
                  f'{canvas.shape[1]}x{canvas.shape[0]}')

    print(f'\nAll {total} images saved under {OUT_ROOT}')


if __name__ == '__main__':
    main()
