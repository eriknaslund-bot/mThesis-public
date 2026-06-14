#!/usr/bin/env python3
"""Per-step debug dump for the LiDAR-guided stitching pipeline.

Saves one image per step to output/lidar_ring_stitch/debug/frame_XXXX/
so every intermediate result can be inspected individually.

Steps saved
-----------
01_source_{cam}.jpg             Raw camera image
02_lidar_all_{cam}.jpg          All LiDAR hits on camera (depth-coloured)
03_lidar_far_{cam}.jpg          Only hits >= MIN_CTRL_RANGE_M (used for TPS)
04_ctrl_pts_{cam}.jpg           Grid-subsampled control points + correction arrows
05_canvas_ctrl_pts.jpg          All cameras' control points on blank canvas
06_overlap_mask_{cam}.jpg       Soft overlap mask (heatmap, 0=black 1=white)
07_rotation_{cam}.jpg           Camera warped with rotation-only remap
08_correction_dx_{cam}.jpg      Δx correction field (blue=neg red=pos)
08_correction_dy_{cam}.jpg      Δy correction field
09_tps_{cam}.jpg                Camera warped with TPS remap
10_canvas_rotation.jpg          Full composite -- rotation-only
11_canvas_tps.jpg               Full composite -- TPS
12_seam_FL_FC.jpg               Zoomed seam strip FL|FC  (rotation vs TPS side-by-side)
12_seam_FC_FR.jpg               Zoomed seam strip FC|FR
13_displacement_{cam}.jpg       |TPS − rotation| displacement heatmap per camera

Usage
-----
    python debug_lidar_stitch.py
    python debug_lidar_stitch.py --frame 5
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from lidar_ring_stitch import (
    FRONT_CAMS, LIDAR_MAX_RANGE_M, LIDAR_MIN_CTRL_RANGE_M,
    CANVAS_MARGIN_FRAC, GRID_COLS, GRID_ROWS, SEAM_HALF_WIDTH_PX,
    TPS_SMOOTHING, LOCAL_SIGMA, FEATHER_HALF,
    load_calib, project_with_ego, ego_to_canvas, cam_pixel_to_canvas_rot,
    find_shared_ctrl_pts,
    build_tps_remap, build_locally_weighted_remap, build_rotation_remap,
    find_seam_dp, blend_with_seam,
    paste_with_feather, depth_color,
)

SENSOR_ROOT = Path.home() / 'mThesis/argo2_data/sensor'
CALIB_JSON  = Path.home() / 'mThesis/argo2_data/extracted/calibration.json'
FRAMES_JSON = Path.home() / 'mThesis/argo2_data/extracted/frames.json'
OUT_ROOT    = Path(__file__).parent.parent / 'output/lidar_ring_stitch/debug'

FONT       = cv2.FONT_HERSHEY_SIMPLEX
CAM_SHORT  = {
    'ring_front_left':   'FL',
    'ring_front_center': 'FC',
    'ring_front_right':  'FR',
}


# -- Helpers -------------------------------------------------------------------

def label(img: np.ndarray, text: str, scale: float = 0.7) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (10, 28), FONT, scale, (0, 0, 0),   3, cv2.LINE_AA)
    cv2.putText(out, text, (10, 28), FONT, scale, (255,255,255), 1, cv2.LINE_AA)
    return out


def save(path: Path, img: np.ndarray, quality: int = 90):
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    print(f'  -> {path.name}')


def diverging_heatmap(field: np.ndarray, cap: float = 80.0,
                      valid: np.ndarray = None) -> np.ndarray:
    """Blue (negative) -- black (zero) -- red (positive), capped at ±cap px."""
    out = np.zeros((*field.shape, 3), np.uint8)
    pos = np.clip( field / cap, 0, 1)
    neg = np.clip(-field / cap, 0, 1)
    out[:, :, 2] = (pos * 255).astype(np.uint8)   # R
    out[:, :, 0] = (neg * 255).astype(np.uint8)   # B
    if valid is not None:
        out[~valid] = 30   # dark grey for invalid
    return out


def magnitude_heatmap(field: np.ndarray, cap: float = 100.0,
                      valid: np.ndarray = None) -> np.ndarray:
    norm = np.clip(field / cap, 0, 1)
    heat = (norm * 255).astype(np.uint8)
    bgr  = cv2.applyColorMap(heat, cv2.COLORMAP_INFERNO)
    if valid is not None:
        bgr[~valid] = 0
    return bgr


def seam_strip(canvas_rot: np.ndarray, canvas_tps: np.ndarray,
               col: int, strip_w: int = 250, gap: int = 6) -> np.ndarray:
    """Side-by-side seam comparison strip centred on canvas column `col`.

    Left half: rotation-only.  Right half: TPS.
    A white vertical line marks the exact seam column in each half.
    """
    H, W = canvas_rot.shape[:2]
    c0 = max(0, col - strip_w)
    c1 = min(W, col + strip_w)
    rot_crop = canvas_rot[:, c0:c1].copy()
    tps_crop = canvas_tps[:, c0:c1].copy()

    rel = col - c0
    cv2.line(rot_crop, (rel, 0), (rel, H - 1), (255, 255, 255), 1)
    cv2.line(tps_crop, (rel, 0), (rel, H - 1), (255, 255, 255), 1)

    divider = np.full((H, gap, 3), 40, np.uint8)
    side_by_side = np.hstack([rot_crop, divider, tps_crop])

    cv2.putText(side_by_side, 'rotation-only', (8, 24),
                FONT, 0.65, (0,0,0),     2, cv2.LINE_AA)
    cv2.putText(side_by_side, 'rotation-only', (8, 24),
                FONT, 0.65, (200,200,200), 1, cv2.LINE_AA)
    tx = rot_crop.shape[1] + gap + 8
    cv2.putText(side_by_side, 'corrected', (tx, 24),
                FONT, 0.65, (0,0,0),     2, cv2.LINE_AA)
    cv2.putText(side_by_side, 'corrected', (tx, 24),
                FONT, 0.65, (200,200,200), 1, cv2.LINE_AA)
    return side_by_side


# -- Main ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frame',       type=int, default=0)
    ap.add_argument('--method',      default='tps',
                    choices=['rot', 'tps', 'local', 'tps_local'],
                    help='Warp method for the corrected canvas')
    ap.add_argument('--local-sigma',  type=float, default=LOCAL_SIGMA)
    ap.add_argument('--feather-half', type=int,   default=FEATHER_HALF)
    ap.add_argument('--seam-dp',      action='store_true')
    ap.add_argument('--sensor-root',  default=str(SENSOR_ROOT))
    ap.add_argument('--calib',       default=str(CALIB_JSON))
    ap.add_argument('--frames',      default=str(FRAMES_JSON))
    ap.add_argument('--real-overlap-mask', action='store_true',
                    help='Use actual pixel-overlap mask instead of seam strip')
    ap.add_argument('--min-ctrl-range',   type=float, default=LIDAR_MIN_CTRL_RANGE_M,
                    help='Min LiDAR range (m) for TPS control points (default: %(default)s)')
    ap.add_argument('--tps-smoothing',    type=float, default=TPS_SMOOTHING,
                    help='RBF smoothing factor (0=exact interp, higher=smoother, default: %(default)s)')
    args = ap.parse_args()
    # Compatibility shim: the deployed pipeline is always asymmetric (FC at
    # rotation baseline, FL/FR carry the warp). The historical --symmetric /
    # --mid-alpha flags were removed when the symmetric path was deleted.
    args.symmetric = False
    args.mid_alpha = 1.0

    out_dir = OUT_ROOT / f'frame_{args.frame:04d}'
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- Load ------------------------------------------------------------------
    cams = load_calib(args.calib)
    with open(args.frames) as f:
        frames = json.load(f)
    frame = frames[args.frame]

    sensor_root = Path(args.sensor_root)
    lidar_files = {}
    for fp in sensor_root.glob('train/*/sensors/lidar/*.feather'):
        try: lidar_files[int(fp.stem)] = fp
        except ValueError: pass
    lidar_ts = np.array(sorted(lidar_files.keys()), np.int64)

    ref_cam = 'ring_front_center'
    img_ts  = int(Path(frame[ref_cam]).stem)
    best_ts = int(lidar_ts[np.argmin(np.abs(lidar_ts - img_ts))])
    dt_ms   = abs(best_ts - img_ts) / 1e6
    df   = pd.read_feather(lidar_files[best_ts])
    pts  = df[['x', 'y', 'z']].values.astype(np.float32)
    print(f'Frame {args.frame}  LiDAR dt={dt_ms:.1f}ms  {len(pts)} pts')

    images = {}
    for name in FRONT_CAMS:
        img = cv2.imread(frame[name])
        if img is None:
            raise FileNotFoundError(frame[name])
        images[name] = img

    # -- Canvas geometry -------------------------------------------------------
    f_cyl = float(cams[ref_cam]['fx'])
    Z_REF = float(np.mean([cams[n]['t'][2] for n in FRONT_CAMS]))
    all_az, all_el = [], []
    for name in FRONT_CAMS:
        _, pts_v = project_with_ego(pts, cams[name])
        if len(pts_v) == 0: continue
        az   = np.arctan2(pts_v[:, 1], pts_v[:, 0])
        r_xy = np.sqrt(pts_v[:, 0]**2 + pts_v[:, 1]**2)
        el   = np.arctan2(pts_v[:, 2] - Z_REF, r_xy)
        all_az.append(az); all_el.append(el)
    az_all = np.concatenate(all_az); el_all = np.concatenate(all_el)
    az_min, az_max = float(az_all.min()), float(az_all.max())
    el_min, el_max = float(el_all.min()), float(el_all.max())
    mg = CANVAS_MARGIN_FRAC
    az_min -= mg*(az_max-az_min); az_max += mg*(az_max-az_min)
    el_min -= mg*(el_max-el_min); el_max += mg*(el_max-el_min)
    W_canvas = max(1, int(np.ceil(f_cyl*(az_max-az_min))))
    H_canvas = max(1, int(np.ceil(f_cyl*(el_max-el_min))))
    cx_canvas = float(az_max * f_cyl)    # az_max (FL, leftmost) -> u=0
    cy_canvas = float(el_max * f_cyl)
    print(f'Canvas {W_canvas}x{H_canvas}  Z_REF={Z_REF:.3f}m')

    # -- Rotation remaps + overlap masks ---------------------------------------
    rot_remaps, rot_valid = {}, {}
    for name in FRONT_CAMS:
        mx, my = build_rotation_remap(cams[name], f_cyl, cx_canvas, cy_canvas,
                                      W_canvas, H_canvas)
        rot_remaps[name] = (mx, my)
        rot_valid[name]  = ((mx >= 0) & (my >= 0) &
                            (mx < cams[name]['W']) & (my < cams[name]['H']))

    FL, FC, FR = FRONT_CAMS

    ov_FL_FC = rot_valid[FL] & rot_valid[FC]
    ov_FC_FR = rot_valid[FC] & rot_valid[FR]
    seam_FL_FC = (int(np.where(ov_FL_FC.any(axis=0))[0].mean())
                  if ov_FL_FC.any() else W_canvas // 3)
    seam_FC_FR = (int(np.where(ov_FC_FR.any(axis=0))[0].mean())
                  if ov_FC_FR.any() else 2 * W_canvas // 3)

    def seam_strip_mask(seam_col, half_width=SEAM_HALF_WIDTH_PX):
        cols = np.arange(W_canvas, dtype=np.float32)
        prof = np.maximum(0.0, 1.0 - np.abs(cols - seam_col) / half_width)
        return np.tile(prof[np.newaxis, :], (H_canvas, 1)).astype(np.float32)

    strip_FL_FC = seam_strip_mask(seam_FL_FC)
    strip_FC_FR = seam_strip_mask(seam_FC_FR)
    if args.real_overlap_mask:
        ov_lc = cv2.GaussianBlur(
            (rot_valid[FL] & rot_valid[FC]).astype(np.float32), (0, 0), 40)
        ov_cr = cv2.GaussianBlur(
            (rot_valid[FC] & rot_valid[FR]).astype(np.float32), (0, 0), 40)
        overlap_masks = {
            FL: ov_lc,
            FC: np.maximum(ov_lc, ov_cr),
            FR: ov_cr,
        }
    else:
        overlap_masks = {
            FL: strip_FL_FC,
            FC: np.maximum(strip_FL_FC, strip_FC_FR),
            FR: strip_FC_FR,
        }

    # -- Project LiDAR into all cameras ONCE -----------------------------------
    proj: dict = {}
    for name in FRONT_CAMS:
        px, ego, idx = project_with_ego(pts, cams[name], return_indices=True)
        proj[name] = (px, ego, idx)

    # -- Shared control points (index set-intersection, no reprojection) -------
    sh_px_FL, sh_px_FC_l, sh_cvs_FL_FC = find_shared_ctrl_pts(
        *proj[FL], *proj[FC],
        f_cyl, cx_canvas, cy_canvas, Z_REF,
        min_range_m=args.min_ctrl_range,
        ref_cam=cams[FC], ref_side='right')
    sh_px_FC_r, sh_px_FR, sh_cvs_FC_FR = find_shared_ctrl_pts(
        *proj[FC], *proj[FR],
        f_cyl, cx_canvas, cy_canvas, Z_REF,
        min_range_m=args.min_ctrl_range,
        ref_cam=cams[FC], ref_side='left')

    # Save FC-only canvas targets before any blending (used in Step 12c vis).
    orig_cvs_FL_FC = sh_cvs_FL_FC.copy()
    orig_cvs_FC_FR = sh_cvs_FC_FR.copy()

    # Symmetric "meet-in-middle" mode: blend canvas targets toward midpoint.
    if args.symmetric:
        alpha = args.mid_alpha
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
        # FC stays rotation-only -- no ctrl pts needed.
        shared_src = {FL: sh_px_FL, FC: np.empty((0, 2), np.float32), FR: sh_px_FR}
        shared_dst = {FL: sh_cvs_FL_FC, FC: np.empty((0, 2), np.float32), FR: sh_cvs_FC_FR}
    print(f'Shared pts  FL<->FC={len(sh_px_FL)}  FC<->FR={len(sh_px_FR)}')

    # -- Per-camera data -------------------------------------------------------
    cam_data = {}
    for name in FRONT_CAMS:
        cam    = cams[name]
        W_cam, H_cam = cam['W'], cam['H']

        cam_px_all, pts_all, _ = proj[name]
        range_all = np.linalg.norm(pts_all, axis=1) if len(pts_all) else np.array([])

        # Far-only (used for TPS)
        far   = range_all >= LIDAR_MIN_CTRL_RANGE_M
        cam_px_far = cam_px_all[far]
        pts_far    = pts_all[far]

        # Control pts: only shared LiDAR points visible in both cameras
        sh_src = shared_src[name]
        sh_dst = shared_dst[name]
        src_pts = sh_src
        dst_pts = sh_dst

        mx_rot, my_rot = rot_remaps[name]

        # Corrected remap -- FC is the reference camera, kept as rotation-only
        method = args.method

        if (name == FC and not args.symmetric) or len(src_pts) < 4 or method == 'rot':
            mx_tps, my_tps = mx_rot.copy(), my_rot.copy()
        elif method == 'tps':
            mx_tps, my_tps = build_tps_remap(
                src_pts, dst_pts, W_cam, H_cam, W_canvas, H_canvas,
                cam=cam, f_cyl=f_cyl,
                cx_canvas=cx_canvas, cy_canvas=cy_canvas,
                smoothing=args.tps_smoothing, remap_scale=0.5,
                overlap_mask=overlap_masks[name])
        elif method == 'local':
            mx_tps, my_tps = build_locally_weighted_remap(
                src_pts, dst_pts,
                cam=cam, f_cyl=f_cyl,
                cx_canvas=cx_canvas, cy_canvas=cy_canvas,
                W_canvas=W_canvas, H_canvas=H_canvas,
                sigma=args.local_sigma,
                overlap_mask=overlap_masks[name])
        elif method == 'tps_local':
            _mx_tps, _my_tps = build_tps_remap(
                src_pts, dst_pts, W_cam, H_cam, W_canvas, H_canvas,
                cam=cam, f_cyl=f_cyl,
                cx_canvas=cx_canvas, cy_canvas=cy_canvas,
                smoothing=args.tps_smoothing, remap_scale=0.5,
                overlap_mask=overlap_masks[name])
            mx_tps, my_tps = build_locally_weighted_remap(
                src_pts, dst_pts,
                cam=cam, f_cyl=f_cyl,
                cx_canvas=cx_canvas, cy_canvas=cy_canvas,
                W_canvas=W_canvas, H_canvas=H_canvas,
                sigma=args.local_sigma,
                overlap_mask=overlap_masks[name],
                base_map_x=_mx_tps, base_map_y=_my_tps)

        # Correction fields
        v_rot = (mx_rot >= 0) & (my_rot >= 0) & (mx_rot < W_cam) & (my_rot < H_cam)
        dx = np.where(v_rot, mx_tps - mx_rot, 0.0).astype(np.float32)
        dy = np.where(v_rot, my_tps - my_rot, 0.0).astype(np.float32)

        cam_data[name] = dict(
            W_cam=W_cam, H_cam=H_cam,
            cam_px_all=cam_px_all, range_all=range_all,
            cam_px_far=cam_px_far, range_far=range_all[far],
            src_pts=src_pts, dst_pts=dst_pts,
            mx_rot=mx_rot, my_rot=my_rot,
            mx_tps=mx_tps, my_tps=my_tps,
            dx=dx, dy=dy, v_rot=v_rot,
        )

    # -- Composites ------------------------------------------------------------
    # Collect warped images for both rotation and corrected composites
    w_rots, v_rots, w_tpss, v_tpss = {}, {}, {}, {}
    for name in FRONT_CAMS:
        d   = cam_data[name]
        img = images[name]
        w_rots[name] = cv2.remap(img, d['mx_rot'], d['my_rot'], cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        w_tpss[name] = cv2.remap(img, d['mx_tps'], d['my_tps'], cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        v_rots[name] = d['v_rot']
        v_tpss[name] = ((d['mx_tps'] >= 0) & (d['my_tps'] >= 0) &
                        (d['mx_tps'] < d['W_cam']) & (d['my_tps'] < d['H_cam']))

    canvas_rot = np.zeros((H_canvas, W_canvas, 3), np.uint8)
    canvas_tps = np.zeros((H_canvas, W_canvas, 3), np.uint8)

    seam_dp_paths = {}   # stored for visualisation below
    if args.seam_dp:
        for canvas_out, w_imgs, v_imgs in [
                (canvas_rot, w_rots, v_rots),
                (canvas_tps, w_tpss, v_tpss)]:
            paste_with_feather(canvas_out, w_imgs[FL], v_imgs[FL], feather_half=0)
            seam_lc = find_seam_dp(w_imgs[FL], w_imgs[FC], v_imgs[FL], v_imgs[FC])
            tmp = blend_with_seam(canvas_out, w_imgs[FC], seam_lc,
                                  feather_half=args.feather_half,
                                  valid_left=v_imgs[FL], valid_right=v_imgs[FC])
            canvas_out[:] = tmp
            seam_cr = find_seam_dp(w_imgs[FC], w_imgs[FR], v_imgs[FC], v_imgs[FR])
            tmp = blend_with_seam(canvas_out, w_imgs[FR], seam_cr,
                                  feather_half=args.feather_half,
                                  valid_left=v_imgs[FC], valid_right=v_imgs[FR])
            canvas_out[:] = tmp
            seam_dp_paths['FL_FC'] = seam_lc
            seam_dp_paths['FC_FR'] = seam_cr
    else:
        for name in FRONT_CAMS:
            paste_with_feather(canvas_rot, w_rots[name], v_rots[name],
                               feather_half=args.feather_half)
            paste_with_feather(canvas_tps, w_tpss[name], v_tpss[name],
                               feather_half=args.feather_half)

    # -- Find seam columns -----------------------------------------------------
    # Use the centre of the overlap zone column range
    ov_FL_FC = rot_valid[FL] & rot_valid[FC]
    ov_FC_FR = rot_valid[FC] & rot_valid[FR]
    seam_FL_FC = int(np.where(ov_FL_FC.any(axis=0))[0].mean()) if ov_FL_FC.any() else W_canvas // 3
    seam_FC_FR = int(np.where(ov_FC_FR.any(axis=0))[0].mean()) if ov_FC_FR.any() else 2 * W_canvas // 3

    # ══════════════════════════════════════════════════════════════════════════
    # Save all debug images
    # ══════════════════════════════════════════════════════════════════════════
    print(f'\nSaving debug images to {out_dir}/')

    # -- Step 01: source images ------------------------------------------------
    for name in FRONT_CAMS:
        s = CAM_SHORT[name]
        save(out_dir / f'01_source_{s}.jpg',
             label(images[name], f'01 source -- {s}'))

    # -- Step 02: all LiDAR hits on camera -------------------------------------
    for name in FRONT_CAMS:
        d   = cam_data[name]
        img = images[name].copy()
        s   = CAM_SHORT[name]
        if len(d['cam_px_all']) > 0:
            order  = np.argsort(d['range_all'])[::-1]
            colors = depth_color(d['range_all'][order])
            for (u, v), c in zip(d['cam_px_all'][order], colors):
                cv2.circle(img, (int(round(float(u))), int(round(float(v)))),
                           3, (int(c[0]), int(c[1]), int(c[2])), -1)
        save(out_dir / f'02_lidar_all_{s}.jpg',
             label(img, f'02 all LiDAR hits -- {s}  ({len(d["cam_px_all"])} pts, red=near blue=far)'))

    # -- Step 03: far-only LiDAR hits (used for TPS) ---------------------------
    for name in FRONT_CAMS:
        d   = cam_data[name]
        img = images[name].copy()
        s   = CAM_SHORT[name]
        # Grey-out close points
        if len(d['cam_px_all']) > 0:
            close = d['range_all'] < LIDAR_MIN_CTRL_RANGE_M
            for (u, v) in d['cam_px_all'][close]:
                cv2.circle(img, (int(round(float(u))), int(round(float(v)))),
                           3, (60, 60, 60), -1)
        if len(d['cam_px_far']) > 0:
            order  = np.argsort(d['range_far'])[::-1]
            colors = depth_color(d['range_far'][order])
            for (u, v), c in zip(d['cam_px_far'][order], colors):
                cv2.circle(img, (int(round(float(u))), int(round(float(v)))),
                           4, (int(c[0]), int(c[1]), int(c[2])), -1)
        save(out_dir / f'03_lidar_far_{s}.jpg',
             label(img, f'03 far LiDAR (>={LIDAR_MIN_CTRL_RANGE_M:.0f}m) -- {s}  '
                        f'({len(d["cam_px_far"])} kept, grey=discarded)'))

    # -- Step 04: control points -- shared LiDAR seam pts only -----------------
    # These are LiDAR points visible in BOTH this camera and its neighbour.
    # Same physical 3D point -> must land at the same canvas pixel from both cameras.
    for name in FRONT_CAMS:
        d    = cam_data[name]
        img  = images[name].copy()
        s    = CAM_SHORT[name]
        H_cam, W_cam = img.shape[:2]

        if name == FC:
            # FC is rotation-only -- show shared pts from both seam pairs
            # yellow = FL<->FC side, cyan = FC<->FR side
            for pts_arr, color in [(sh_px_FC_l, (0, 220, 255)),
                                   (sh_px_FC_r, (255, 220, 0))]:
                for (u, v) in pts_arr:
                    pu, pv = int(round(float(u))), int(round(float(v)))
                    if 0 <= pu < W_cam and 0 <= pv < H_cam:
                        cv2.circle(img, (pu, pv), 7, color, -1)
            n_fc = len(sh_px_FC_l) + len(sh_px_FC_r)
            save(out_dir / f'04_ctrl_pts_{s}.jpg',
                 label(img, f'04 shared LiDAR ctrl pts -- {s}  '
                            f'({n_fc} pts, ref-only: yellow=FL side  cyan=FR side)'))
        else:
            for (u, v) in d['src_pts']:
                pu, pv = int(round(float(u))), int(round(float(v)))
                if 0 <= pu < W_cam and 0 <= pv < H_cam:
                    cv2.circle(img, (pu, pv), 7, (0, 220, 255), -1)
            save(out_dir / f'04_ctrl_pts_{s}.jpg',
                 label(img, f'04 shared LiDAR ctrl pts -- {s}  ({len(d["src_pts"])} pts)'))

    # -- Step 05: control points on canvas -------------------------------------
    # All control pts are shared -- same canvas px used by both adjacent cameras.
    # FL<->FC seam pts in yellow, FC<->FR in cyan; they overlap at the canvas target.
    cvs_ctrl = np.zeros((H_canvas, W_canvas, 3), np.uint8)

    for pts_arr, color in [(sh_cvs_FL_FC, (0, 220, 255)), (sh_cvs_FC_FR, (255, 220, 0))]:
        for (u, v) in pts_arr:
            pu, pv = int(round(float(u))), int(round(float(v)))
            if 0 <= pu < W_canvas and 0 <= pv < H_canvas:
                cv2.circle(cvs_ctrl, (pu, pv), 5, color, -1)

    n_shared = len(sh_cvs_FL_FC) + len(sh_cvs_FC_FR)
    save(out_dir / '05_canvas_ctrl_pts.jpg',
         label(cvs_ctrl,
               f'05 canvas ctrl pts  yellow=FL|FC({len(sh_cvs_FL_FC)})  '
               f'cyan=FC|FR({len(sh_cvs_FC_FR)})'))

    # -- Step 06: overlap masks -------------------------------------------------
    for name in FRONT_CAMS:
        s    = CAM_SHORT[name]
        mask = overlap_masks[name]
        vis  = (mask * 255).astype(np.uint8)
        vis_bgr = cv2.applyColorMap(vis, cv2.COLORMAP_HOT)
        save(out_dir / f'06_overlap_mask_{s}.jpg',
             label(vis_bgr, f'06 overlap mask -- {s}  (white=full TPS  black=rotation only)'))

    # -- Step 07: rotation warp per camera -------------------------------------
    for name in FRONT_CAMS:
        d   = cam_data[name]
        s   = CAM_SHORT[name]
        img = images[name]
        W_cam, H_cam = d['W_cam'], d['H_cam']
        w = cv2.remap(img, d['mx_rot'], d['my_rot'], cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        save(out_dir / f'07_rotation_{s}.jpg',
             label(w, f'07 rotation warp -- {s}'))

    # -- Steps 08a/b: correction fields (dx, dy) --------------------------------
    for name in FRONT_CAMS:
        d   = cam_data[name]
        s   = CAM_SHORT[name]
        cap = float(np.abs(d['dx'][d['v_rot']]).max()) if d['v_rot'].any() else 80.0
        cap = max(cap, 1.0)
        dx_vis = diverging_heatmap(d['dx'], cap=cap, valid=d['v_rot'])
        dy_vis = diverging_heatmap(d['dy'], cap=cap, valid=d['v_rot'])
        save(out_dir / f'08_correction_dx_{s}.jpg',
             label(dx_vis, f'08 Δx correction -- {s}  (blue=left  red=right  cap=±{cap:.0f}px)'))
        save(out_dir / f'08_correction_dy_{s}.jpg',
             label(dy_vis, f'08 Δy correction -- {s}  (blue=up  red=down  cap=±{cap:.0f}px)'))

    # -- Step 09: TPS warp per camera ------------------------------------------
    for name in FRONT_CAMS:
        d   = cam_data[name]
        s   = CAM_SHORT[name]
        img = images[name]
        W_cam, H_cam = d['W_cam'], d['H_cam']
        w = cv2.remap(img, d['mx_tps'], d['my_tps'], cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        save(out_dir / f'09_tps_{s}.jpg',
             label(w, f'09 TPS warp -- {s}'))

    # -- Step 10/11: composites ------------------------------------------------
    save(out_dir / '10_canvas_rotation.jpg',
         label(canvas_rot, '10 final composite -- rotation-only'))
    save(out_dir / '11_canvas_tps.jpg',
         label(canvas_tps, '11 final composite -- TPS (LiDAR-guided)'))

    # -- Step 12: seam zoom strips ---------------------------------------------
    save(out_dir / '12_seam_FL_FC.jpg',
         label(seam_strip(canvas_rot, canvas_tps, seam_FL_FC),
               f'12 seam FL|FC  (col~{seam_FL_FC})  rotation-only | TPS'))
    save(out_dir / '12_seam_FC_FR.jpg',
         label(seam_strip(canvas_rot, canvas_tps, seam_FC_FR),
               f'12 seam FC|FR  (col~{seam_FC_FR})  rotation-only | TPS'))

    # -- Step 12a: DP seam path overlay ----------------------------------------
    if args.seam_dp and seam_dp_paths:
        seam_vis = canvas_tps.copy()
        for pair_name, seam_arr in seam_dp_paths.items():
            color = (0, 255, 255) if pair_name == 'FL_FC' else (255, 0, 255)
            for row, col in enumerate(seam_arr):
                cv2.circle(seam_vis, (int(col), row), 0, color, -1)
        save(out_dir / '12a_seam_dp_paths.jpg',
             label(seam_vis, '12a DP seam paths  cyan=FL|FC  magenta=FC|FR'))

    # -- Step 12b: red/cyan alignment overlay ---------------------------------
    # FL warped (red) vs FC (cyan) in the overlap zone.
    # Perfect alignment -> grey.  Misalignment -> colour fringe.
    # Crops the actual column range where both cameras have valid content.
    for pair_name, ov_mask, seam_col, left_cam, right_cam in [
            ('FL_FC', ov_FL_FC, seam_FL_FC, FL, FC),
            ('FC_FR', ov_FC_FR, seam_FC_FR, FC, FR)]:
        ov_cols = np.where(ov_mask.any(axis=0))[0]
        if len(ov_cols) == 0:
            print(f'  skip 12b {pair_name} -- no overlap columns')
            continue
        c0 = int(ov_cols[0])
        c1 = int(ov_cols[-1]) + 1

        for tag in ('rot', 'tps'):
            mx_l = cam_data[left_cam]['mx_rot' if tag == 'rot' else 'mx_tps']
            my_l = cam_data[left_cam]['my_rot' if tag == 'rot' else 'my_tps']

            w_left = cv2.remap(images[left_cam], mx_l, my_l,
                               cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            w_right = cv2.remap(images[right_cam],
                                cam_data[right_cam]['mx_rot'],
                                cam_data[right_cam]['my_rot'],
                                cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

            crop_l = w_left[:, c0:c1].astype(np.float32)
            crop_r = w_right[:, c0:c1].astype(np.float32)

            gray_l = cv2.cvtColor(crop_l.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
            gray_r = cv2.cvtColor(crop_r.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)

            # Red = left camera, Cyan (G+B) = right camera; grey = aligned
            overlay = np.zeros((*gray_l.shape, 3), np.float32)
            overlay[:, :, 2] = gray_l          # R channel <- left cam
            overlay[:, :, 1] = gray_r          # G channel <- right cam
            overlay[:, :, 0] = gray_r          # B channel <- right cam
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)

            # White line at seam column
            rel = seam_col - c0
            if 0 <= rel < overlay.shape[1]:
                cv2.line(overlay, (rel, 0), (rel, overlay.shape[0] - 1), (255, 255, 255), 1)

            fname = f'12b_align_{pair_name}_{tag}.jpg'
            save(out_dir / fname,
                 label(overlay, f'12b {pair_name} {tag}  red={CAM_SHORT[left_cam]}  '
                                f'cyan={CAM_SHORT[right_cam]}  grey=aligned  '
                                f'overlap_cols=[{c0},{c1}]'))

    # -- Step 12c: control-point alignment scatter -----------------------------
    # For each shared LiDAR control point draw:
    #   Green circle  = FC-only rotation-model target (original, pre-blend)
    #   White circle  = symmetric midpoint target (only shown when --symmetric)
    #   Red   circle + line = FL/FR rotation-model canvas position -> target
    #   Blue  circle + line = TPS residual: where TPS remap looks up at target
    #                         position vs where the point actually is in FL camera
    for pair_name, src_px, tgt_px, fc_tgt_px, side_cam in [
            ('FL_FC', sh_px_FL,  sh_cvs_FL_FC, orig_cvs_FL_FC, FL),
            ('FC_FR', sh_px_FR,  sh_cvs_FC_FR, orig_cvs_FC_FR, FR)]:

        overlay = canvas_rot.copy()

        # FL/FR canvas positions via rotation model (forward project camera pixel)
        u_rot, v_rot = cam_pixel_to_canvas_rot(
            src_px[:, 0], src_px[:, 1],
            cams[side_cam], f_cyl, cx_canvas, cy_canvas)

        # TPS residual: look up FL/FR TPS remap at target canvas position,
        # compare to actual camera pixel -> residual in camera px -> convert
        mx_tps = cam_data[side_cam]['mx_tps']
        my_tps = cam_data[side_cam]['my_tps']
        f_cam  = (cams[side_cam]['fx'] + cams[side_cam]['fy']) / 2.0

        # Collect all canvas positions for zoom bounds
        all_u, all_v = [], []
        pt_data = []
        for i in range(len(src_px)):
            # Green = FC-only target; white = midpoint (same as green if not symmetric)
            ug, vg = float(fc_tgt_px[i, 0]), float(fc_tgt_px[i, 1])
            ut, vt = float(tgt_px[i, 0]), float(tgt_px[i, 1])
            ur, vr = float(u_rot[i]), float(v_rot[i])

            # TPS residual: what cam pixel does TPS look up at target canvas pos?
            ui, vi = int(round(ut)), int(round(vt))
            ui = max(0, min(ui, W_canvas - 1))
            vi = max(0, min(vi, H_canvas - 1))
            tps_cam_x = float(mx_tps[vi, ui])
            tps_cam_y = float(my_tps[vi, ui])
            scale = f_cyl / f_cam
            utps = ut + (tps_cam_x - src_px[i, 0]) * scale
            vtps = vt + (tps_cam_y - src_px[i, 1]) * scale

            pt_data.append((ug, vg, ut, vt, ur, vr, utps, vtps))
            for u, v in [(ug, vg), (ut, vt), (ur, vr), (utps, vtps)]:
                all_u.append(u); all_v.append(v)

        # Full-canvas overlay (small dots)
        def draw_pts(img, r_dot=4, lw=1):
            for ug, vg, ut, vt, ur, vr, utps, vtps in pt_data:
                pt_g  = (int(round(ug)),   int(round(vg)))   # FC-only target (green)
                pt_t  = (int(round(ut)),   int(round(vt)))   # midpoint target (white)
                pt_r  = (int(round(ur)),   int(round(vr)))   # rot-model (red)
                pt_tp = (int(round(utps)), int(round(vtps))) # TPS residual (blue)
                cv2.line(img, pt_r,  pt_t,  (0,   0, 220), lw, cv2.LINE_AA)
                cv2.line(img, pt_tp, pt_t,  (220, 0,   0), lw, cv2.LINE_AA)
                cv2.circle(img, pt_g,  r_dot,     (0, 220,   0), -1, cv2.LINE_AA)
                if args.symmetric:
                    cv2.circle(img, pt_t, r_dot - 1, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(img, pt_r,  r_dot - 1, (0,   0, 220), -1, cv2.LINE_AA)
                cv2.circle(img, pt_tp, r_dot - 2, (220,  0,   0), -1, cv2.LINE_AA)

        sym_legend = '  white=mid_target' if args.symmetric else ''
        draw_pts(overlay)
        save(out_dir / f'12c_ctrl_align_{pair_name}.jpg',
             label(overlay,
                   f'12c {pair_name}  green=FC_target{sym_legend}  red=rot_pos  '
                   f'blue=TPS_residual  lines=error_to_target'))

        # Zoomed crop around the control-point cluster (+200px margin)
        if all_u:
            pad = 200
            zu0 = max(0, int(min(all_u)) - pad)
            zu1 = min(W_canvas, int(max(all_u)) + pad)
            zv0 = max(0, int(min(all_v)) - pad)
            zv1 = min(H_canvas, int(max(all_v)) + pad)

            zoom_base = canvas_rot[zv0:zv1, zu0:zu1].copy()
            # Shift all coordinates by crop origin and draw bigger
            pt_data_z = [(ug-zu0, vg-zv0, ut-zu0, vt-zv0,
                          ur-zu0, vr-zv0, utps-zu0, vtps-zv0)
                         for ug, vg, ut, vt, ur, vr, utps, vtps in pt_data]
            orig_data, pt_data = pt_data, pt_data_z
            draw_pts(zoom_base, r_dot=8, lw=2)
            pt_data = orig_data
            save(out_dir / f'12c_ctrl_align_{pair_name}_zoom.jpg',
                 label(zoom_base,
                       f'12c {pair_name} ZOOM  green=FC_target{sym_legend}  '
                       f'red=rot_pos  blue=TPS_residual'))

    # -- Step 13: displacement heatmaps ----------------------------------------
    for name in FRONT_CAMS:
        d   = cam_data[name]
        s   = CAM_SHORT[name]
        mag = np.sqrt(d['dx']**2 + d['dy']**2)
        # p95 over pixels where correction is non-trivial (> 0.5 px).
        # Median over ALL valid pixels is always 0 because correction is zero
        # outside the narrow overlap strip.
        nonzero = mag[d['v_rot']] > 0.5
        p95 = float(np.percentile(mag[d['v_rot']][nonzero], 95)) if nonzero.any() else 0.0
        vis = magnitude_heatmap(mag, cap=100.0, valid=d['v_rot'])
        save(out_dir / f'13_displacement_{s}.jpg',
             label(vis, f'13 |correction| -- {s}  p95={p95:.1f}px  scale=0–100px'))

    print(f'\nDone -- {len(list(out_dir.glob("*.jpg")))} images in {out_dir}')


if __name__ == '__main__':
    main()
