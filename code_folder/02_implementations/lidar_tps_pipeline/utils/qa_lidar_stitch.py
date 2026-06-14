#!/usr/bin/env python3
"""QA/debug script for the LiDAR-guided panoramic stitching pipeline.

Produces a single tall composite JPEG with 5 diagnostic panels:

  A  Numbered correspondences -- camera vs canvas, 20 pts spread across depth range
  B  Correction arrows -- rotation model vs LiDAR truth, on canvas
  C  Warp grid -- deformed 12x9 grid, rotation (green) vs LiDAR-corrected (orange)
  D  Seam alignment -- FL/FC and FC/FR seam strip comparison
  E  Correction field -- Δx and Δy diverging heatmaps

Usage
-----
    python qa_lidar_stitch.py --frame 0
    python qa_lidar_stitch.py --frame 3 --calib /path/to/calibration.json
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# -- Import pipeline functions -------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from lidar_ring_stitch import (
    FRONT_CAMS, LIDAR_MAX_RANGE_M,
    CANVAS_MARGIN_FRAC, TPS_SMOOTHING,
    load_calib, project_with_ego, ego_to_canvas,
    grid_subsample, build_ghost_anchors,
    build_tps_remap, build_rotation_remap,
    depth_color,
)

# -- Default paths -------------------------------------------------------------
SENSOR_ROOT = Path.home() / 'mThesis/argo2_data/sensor'
CALIB_JSON  = Path.home() / 'mThesis/argo2_data/extracted/calibration.json'
FRAMES_JSON = Path.home() / 'mThesis/argo2_data/extracted/frames.json'
OUT_DIR     = Path(__file__).parent.parent / 'output/lidar_ring_stitch'

# -- Layout constants ----------------------------------------------------------
OUTPUT_W    = 3000
LABEL_H     = 30
FONT        = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE  = 0.65
FONT_THICK  = 1
BG_COLOR    = (25, 25, 25)
LABEL_BG    = (40, 40, 40)
LABEL_FG    = (220, 220, 220)

# Camera label colours: FL=red, FC=green, FR=blue
CAM_LABEL_COLORS = {
    'ring_front_left':   (80, 80, 220),   # BGR: red-tinted
    'ring_front_center': (80, 200, 80),   # green
    'ring_front_right':  (220, 80, 80),   # blue-tinted
}
CAM_SHORT = {
    'ring_front_left':   'FL',
    'ring_front_center': 'FC',
    'ring_front_right':  'FR',
}

# Panel A: number of control points shown per camera
N_QA_POINTS = 20

# Panel C: warp grid dimensions
GRID_COLS_VIZ = 12
GRID_ROWS_VIZ = 9


# -- Drawing helpers -----------------------------------------------------------

def label_bar(text: str, width: int, h: int = LABEL_H) -> np.ndarray:
    bar = np.full((h, width, 3), LABEL_BG, dtype=np.uint8)
    cv2.putText(bar, text, (8, h - 8), FONT, FONT_SCALE, LABEL_FG, FONT_THICK,
                cv2.LINE_AA)
    return bar


def separator(width: int, h: int = 6) -> np.ndarray:
    return np.full((h, width, 3), BG_COLOR, dtype=np.uint8)


def fit_to_width(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    new_h = max(1, int(round(h * width / w)))
    return cv2.resize(img, (width, new_h), interpolation=cv2.INTER_LINEAR)


def fit_to_height(img: np.ndarray, height: int) -> np.ndarray:
    h, w = img.shape[:2]
    new_w = max(1, int(round(w * height / h)))
    return cv2.resize(img, (new_w, height), interpolation=cv2.INTER_LINEAR)


def hstack_equal_height(imgs: list, target_h: int, total_w: int,
                        gap: int = 4) -> np.ndarray:
    """Scale images to target_h, hstack with gap, resize row to total_w."""
    scaled = [fit_to_height(im, target_h) for im in imgs]
    row_w = sum(im.shape[1] for im in scaled) + gap * (len(scaled) - 1)
    row = np.full((target_h, row_w, 3), BG_COLOR, dtype=np.uint8)
    x = 0
    for im in scaled:
        row[:, x:x + im.shape[1]] = im
        x += im.shape[1] + gap
    return cv2.resize(row, (total_w, target_h), interpolation=cv2.INTER_LINEAR)


def jet_scalar(t: float) -> tuple:
    """Jet colormap scalar t∈[0,1] -> BGR tuple."""
    t = float(np.clip(t, 0, 1))
    r = float(np.clip(1.5 - abs(4*t - 3), 0, 1))
    g = float(np.clip(1.5 - abs(4*t - 2), 0, 1))
    b = float(np.clip(1.5 - abs(4*t - 1), 0, 1))
    return (int(b*255), int(g*255), int(r*255))


def diverging_colormap(val: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Blue=negative, black=zero, red=positive diverging colormap -> BGR uint8."""
    t = np.clip((val - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    # 0=pure blue, 0.5=black, 1=pure red
    out = np.zeros((*val.shape, 3), dtype=np.uint8)
    pos_mask = t >= 0.5
    neg_mask = t < 0.5
    # Positive: black -> red
    s_pos = (t[pos_mask] - 0.5) * 2
    out[pos_mask, 2] = (s_pos * 255).astype(np.uint8)  # R channel
    # Negative: blue -> black
    s_neg = (0.5 - t[neg_mask]) * 2
    out[neg_mask, 0] = (s_neg * 255).astype(np.uint8)  # B channel
    return out


# -- Canvas setup (shared across panels) --------------------------------------

def compute_canvas_geometry(pts: np.ndarray, cams: dict, ref_cam: str):
    """Replicate lidar_ring_stitch.py canvas geometry computation."""
    f_cyl = float(cams[ref_cam]['fx'])
    Z_REF = float(np.mean([cams[n]['t'][2] for n in FRONT_CAMS]))
    all_az, all_el = [], []
    for name in FRONT_CAMS:
        _, pts_v = project_with_ego(pts, cams[name])
        if len(pts_v) == 0:
            continue
        az   = np.arctan2(pts_v[:, 1], pts_v[:, 0])
        r_xy = np.sqrt(pts_v[:, 0]**2 + pts_v[:, 1]**2)
        el   = np.arctan2(pts_v[:, 2] - Z_REF, r_xy)   # elevation from camera height
        all_az.append(az)
        all_el.append(el)
    if not all_az:
        raise RuntimeError('No LiDAR points visible in any front camera')
    az_all = np.concatenate(all_az)
    el_all = np.concatenate(all_el)
    az_min, az_max = float(az_all.min()), float(az_all.max())
    el_min, el_max = float(el_all.min()), float(el_all.max())
    az_mg = CANVAS_MARGIN_FRAC * (az_max - az_min)
    el_mg = CANVAS_MARGIN_FRAC * (el_max - el_min)
    az_min -= az_mg;  az_max += az_mg
    el_min -= el_mg;  el_max += el_mg
    W_canvas = max(1, int(np.ceil(f_cyl * (az_max - az_min))))
    H_canvas = max(1, int(np.ceil(f_cyl * (el_max - el_min))))
    cx_canvas = float(-az_min * f_cyl)
    cy_canvas = float(el_max  * f_cyl)   # el_max -> v=0 (sky at top)
    return f_cyl, cx_canvas, cy_canvas, W_canvas, H_canvas, Z_REF


# -- Per-camera data computation ------------------------------------------------

def compute_camera_data(pts: np.ndarray, cams: dict,
                        f_cyl: float, cx_canvas: float, cy_canvas: float,
                        W_canvas: int, H_canvas: int, z_ref: float = 0.0):
    """For each camera compute all intermediate data needed by QA panels.

    Returns dict keyed by camera name:
      cam_px_all   (N,2) all LiDAR hits in camera
      pts_valid    (N,3) ego-frame points for those hits
      z_all        (N,)  depth values
      src_pts      (M,2) subsampled camera control pts
      dst_pts      (M,2) subsampled canvas control pts
      ctrl_z       (M,)  depths for subsampled pts
      map_x_rot    (H_canvas,W_canvas) rotation remap
      map_y_rot    (H_canvas,W_canvas)
      map_x_tps    (H_canvas,W_canvas) LiDAR-corrected remap (or rotation if <4 pts)
      map_y_tps    (H_canvas,W_canvas)
      tps_valid    bool: True if TPS was actually computed
    """
    from scipy.spatial import cKDTree

    result = {}
    for name in FRONT_CAMS:
        cam = cams[name]
        W_cam, H_cam = cam['W'], cam['H']

        cam_px_all, pts_valid = project_with_ego(pts, cam)
        z_all = (np.linalg.norm(pts_valid, axis=1).astype(np.float32)
                 if len(pts_valid) > 0 else np.array([], np.float32))

        if len(cam_px_all) > 0:
            canvas_px_all = ego_to_canvas(pts_valid, f_cyl, cx_canvas, cy_canvas, z_ref)
            src_pts, dst_pts = grid_subsample(cam_px_all, canvas_px_all, W_cam, H_cam)
        else:
            canvas_px_all = np.empty((0, 2), np.float32)
            src_pts = np.empty((0, 2), np.float32)
            dst_pts = np.empty((0, 2), np.float32)

        if len(src_pts) > 0 and len(cam_px_all) > 0:
            tree = cKDTree(cam_px_all)
            _, idxs = tree.query(src_pts, k=1)
            ctrl_z = z_all[idxs]
        else:
            ctrl_z = np.array([], np.float32)

        mx_rot, my_rot = build_rotation_remap(
            cam, f_cyl, cx_canvas, cy_canvas, W_canvas, H_canvas)

        tps_valid = len(src_pts) >= 4
        if tps_valid:
            mx_tps, my_tps = build_tps_remap(
                src_pts, dst_pts, W_cam, H_cam, W_canvas, H_canvas,
                cam=cam, f_cyl=f_cyl,
                cx_canvas=cx_canvas, cy_canvas=cy_canvas,
                smoothing=TPS_SMOOTHING, remap_scale=0.5)
        else:
            mx_tps, my_tps = mx_rot.copy(), my_rot.copy()

        result[name] = dict(
            cam_px_all=cam_px_all,
            pts_valid=pts_valid,
            canvas_px_all=canvas_px_all,
            z_all=z_all,
            src_pts=src_pts,
            dst_pts=dst_pts,
            ctrl_z=ctrl_z,
            map_x_rot=mx_rot, map_y_rot=my_rot,
            map_x_tps=mx_tps, map_y_tps=my_tps,
            tps_valid=tps_valid,
        )
        n_ctrl = len(src_pts)
        print(f'  [{CAM_SHORT[name]}]  LiDAR hits={len(cam_px_all)}  '
              f'ctrl_pts={n_ctrl}  tps={tps_valid}')
    return result


# -- Panel A -------------------------------------------------------------------

def panel_A(images: dict, cams: dict, cam_data: dict,
            f_cyl: float, cx_canvas: float, cy_canvas: float,
            W_canvas: int, H_canvas: int) -> np.ndarray:
    """Numbered correspondences -- camera image vs canvas thumbnail, per camera."""

    TARGET_H = 400   # height for each sub-row (cam and canvas side)
    RADIUS   = 9
    NUMBER_FONT_SCALE = 0.45
    NUMBER_THICK = 1

    # For each camera produce [cam_annotated | canvas_thumb_annotated]
    pair_panels = []

    for name in FRONT_CAMS:
        cam  = cams[name]
        data = cam_data[name]
        img  = images[name].copy()
        clr  = CAM_LABEL_COLORS[name]
        short = CAM_SHORT[name]

        cam_px_all  = data['cam_px_all']     # (N,2)
        canvas_px_all = data['canvas_px_all'] # (N,2)
        z_all       = data['z_all']           # (N,)

        if len(cam_px_all) == 0:
            h, w = cam['H'], cam['W']
            blank = np.zeros((h, w, 3), np.uint8)
            cv2.putText(blank, f'{short}: no LiDAR hits', (10, 30),
                        FONT, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
            pair_panels.append(fit_to_height(blank, TARGET_H))
            pair_panels.append(fit_to_height(blank, TARGET_H))
            continue

        # Bin by depth into N_QA_POINTS equal-count bins; pick one per bin
        order      = np.argsort(z_all)
        n          = len(order)
        bin_size   = max(1, n // N_QA_POINTS)
        chosen_idx = []
        for b in range(N_QA_POINTS):
            lo = b * bin_size
            hi = lo + bin_size if b < N_QA_POINTS - 1 else n
            segment = order[lo:hi]
            if len(segment) == 0:
                continue
            # Pick the one closest to the centre of its depth bin
            z_mid = (z_all[segment[0]] + z_all[segment[-1]]) / 2.0
            best  = segment[np.argmin(np.abs(z_all[segment] - z_mid))]
            chosen_idx.append(int(best))

        sel_cam    = cam_px_all[chosen_idx]    # (M,2)
        sel_cvs    = canvas_px_all[chosen_idx] # (M,2)
        sel_z      = z_all[chosen_idx]         # (M,)
        M          = len(chosen_idx)

        # Depth normalised 0->1 (near=1 -> red end of jet)
        z_lo, z_hi = float(sel_z.min()), max(float(sel_z.max()), float(sel_z.min()) + 1.0)
        t_vals     = 1.0 - (sel_z - z_lo) / (z_hi - z_lo)  # 1=near(red), 0=far(blue)

        # -- Camera image annotation -----------------------------------------
        cam_ann = img.copy()
        W_cam, H_cam = cam['W'], cam['H']
        for i in range(M):
            u_c, v_c = float(sel_cam[i, 0]), float(sel_cam[i, 1])
            bgr = jet_scalar(float(t_vals[i]))
            px, py = int(round(u_c)), int(round(v_c))
            if 0 <= px < W_cam and 0 <= py < H_cam:
                cv2.circle(cam_ann, (px, py), RADIUS, bgr, -1)
                cv2.circle(cam_ann, (px, py), RADIUS, (255,255,255), 1, cv2.LINE_AA)
                cv2.putText(cam_ann, str(i+1), (px + RADIUS + 2, py + 4),
                            FONT, NUMBER_FONT_SCALE, (255,255,255),
                            NUMBER_THICK, cv2.LINE_AA)
        cv2.putText(cam_ann, short, (8, 24), FONT, 0.75, clr, 2, cv2.LINE_AA)

        # -- Canvas thumbnail annotation -------------------------------------
        cvs_thumb = np.zeros((H_canvas, W_canvas, 3), np.uint8)
        # Light grey grid lines for context
        for col in range(0, W_canvas, max(1, W_canvas // 12)):
            cv2.line(cvs_thumb, (col, 0), (col, H_canvas-1), (40, 40, 40), 1)
        for row in range(0, H_canvas, max(1, H_canvas // 8)):
            cv2.line(cvs_thumb, (0, row), (W_canvas-1, row), (40, 40, 40), 1)

        for i in range(M):
            u_cv, v_cv = float(sel_cvs[i, 0]), float(sel_cvs[i, 1])
            bgr = jet_scalar(float(t_vals[i]))
            px, py = int(round(u_cv)), int(round(v_cv))
            if 0 <= px < W_canvas and 0 <= py < H_canvas:
                cv2.circle(cvs_thumb, (px, py), RADIUS, bgr, -1)
                cv2.circle(cvs_thumb, (px, py), RADIUS, (255,255,255), 1, cv2.LINE_AA)
                cv2.putText(cvs_thumb, str(i+1), (px + RADIUS + 2, py + 4),
                            FONT, NUMBER_FONT_SCALE, (255,255,255),
                            NUMBER_THICK, cv2.LINE_AA)
        cv2.putText(cvs_thumb, f'{short} canvas', (8, 24), FONT, 0.75, clr, 2, cv2.LINE_AA)

        pair_panels.append(fit_to_height(cam_ann, TARGET_H))
        pair_panels.append(fit_to_height(cvs_thumb, TARGET_H))

    return hstack_equal_height(pair_panels, TARGET_H, OUTPUT_W, gap=4)


# -- Panel B -------------------------------------------------------------------

def panel_B(cams: dict, cam_data: dict,
            f_cyl: float, cx_canvas: float, cy_canvas: float,
            W_canvas: int, H_canvas: int) -> np.ndarray:
    """Correction arrows: rotation prediction (yellow) vs LiDAR truth (cyan).

    One full-width canvas per camera, stacked vertically.
    """
    # Scale factor to render canvas at OUTPUT_W
    scale = OUTPUT_W / W_canvas
    H_render = max(1, int(round(H_canvas * scale)))

    sub_rows = []
    for name in FRONT_CAMS:
        cam   = cams[name]
        data  = cam_data[name]
        short = CAM_SHORT[name]
        clr   = CAM_LABEL_COLORS[name]

        src_pts = data['src_pts']   # (M,2) cam px
        dst_pts = data['dst_pts']   # (M,2) canvas px (LiDAR truth)

        cvs_img = np.zeros((H_canvas, W_canvas, 3), np.uint8)
        # Faint grid
        for col in range(0, W_canvas, max(1, W_canvas // 16)):
            cv2.line(cvs_img, (col, 0), (col, H_canvas-1), (35, 35, 35), 1)
        for row in range(0, H_canvas, max(1, H_canvas // 8)):
            cv2.line(cvs_img, (0, row), (W_canvas-1, row), (35, 35, 35), 1)

        if len(src_pts) > 0:
            # For each control pt: compute rotation-predicted canvas position
            u_cam, v_cam = src_pts[:, 0], src_pts[:, 1]
            u_rot, v_rot = _cam_px_to_canvas_rot(
                u_cam, v_cam, cam, f_cyl, cx_canvas, cy_canvas)

            for i in range(len(src_pts)):
                u_r = int(round(float(u_rot[i])))
                v_r = int(round(float(v_rot[i])))
                u_l = int(round(float(dst_pts[i, 0])))
                v_l = int(round(float(dst_pts[i, 1])))

                # Clamp to canvas bounds
                u_r = np.clip(u_r, 0, W_canvas - 1)
                v_r = np.clip(v_r, 0, H_canvas - 1)
                u_l = np.clip(u_l, 0, W_canvas - 1)
                v_l = np.clip(v_l, 0, H_canvas - 1)

                # Arrow from rotation (yellow) to LiDAR (cyan)
                cv2.arrowedLine(cvs_img, (u_r, v_r), (u_l, v_l),
                                (0, 180, 180), 1, cv2.LINE_AA, tipLength=0.3)
                cv2.circle(cvs_img, (u_r, v_r), 5, (0, 220, 220), -1)   # yellow
                cv2.circle(cvs_img, (u_l, v_l), 5, (220, 220, 0), -1)   # cyan

        # Legend
        cv2.circle(cvs_img, (8, 14),  5, (0, 220, 220), -1)
        cv2.putText(cvs_img, 'rotation predicted', (16, 18),
                    FONT, 0.45, (0,220,220), 1, cv2.LINE_AA)
        cv2.circle(cvs_img, (8, 30),  5, (220, 220, 0), -1)
        cv2.putText(cvs_img, 'LiDAR truth', (16, 34),
                    FONT, 0.45, (220,220,0), 1, cv2.LINE_AA)
        cv2.putText(cvs_img, short, (W_canvas - 50, 18),
                    FONT, 0.65, clr, 2, cv2.LINE_AA)

        sub_rows.append(fit_to_width(cvs_img, OUTPUT_W))

    return np.vstack(sub_rows)


def _cam_px_to_canvas_rot(u_arr, v_arr, cam, f_cyl, cx_canvas, cy_canvas):
    """Forward: camera pixel -> canvas pixel via rotation-only model."""
    px = (u_arr - cam['cx']) / cam['fx']
    py = (v_arr - cam['cy']) / cam['fy']
    pz = np.ones_like(px)
    p_cam = np.column_stack([px, py, pz])
    p_ego = p_cam @ cam['R'].T
    az   = np.arctan2(p_ego[:, 1], p_ego[:, 0])
    r_xy = np.sqrt(p_ego[:, 0]**2 + p_ego[:, 1]**2)
    el   = np.arctan2(p_ego[:, 2], r_xy)
    u_c  = (f_cyl * az  + cx_canvas).astype(np.float32)
    v_c  = (-f_cyl * el + cy_canvas).astype(np.float32)
    return u_c, v_c


# -- Panel C -------------------------------------------------------------------

def panel_C(images: dict, cams: dict, cam_data: dict,
            f_cyl: float, cx_canvas: float, cy_canvas: float,
            W_canvas: int, H_canvas: int) -> np.ndarray:
    """Warp grid: rotation remap (green) vs LiDAR-corrected remap (orange).

    Method: create a white-on-black grid image in camera space, remap it with
    both remaps, display side-by-side on a canvas background.
    """
    TARGET_H = int(round(H_canvas * OUTPUT_W / (W_canvas * 2 + 8)))
    TARGET_H = max(80, min(TARGET_H, 400))

    pair_panels = []

    for name in FRONT_CAMS:
        cam  = cams[name]
        data = cam_data[name]
        W_cam, H_cam = cam['W'], cam['H']
        short = CAM_SHORT[name]
        clr = CAM_LABEL_COLORS[name]

        # -- Build grid image in camera space --------------------------------
        grid_cam = np.zeros((H_cam, W_cam), dtype=np.uint8)
        for c in range(GRID_COLS_VIZ + 1):
            x = int(round(c * (W_cam - 1) / GRID_COLS_VIZ))
            cv2.line(grid_cam, (x, 0), (x, H_cam - 1), 255, 1)
        for r in range(GRID_ROWS_VIZ + 1):
            y = int(round(r * (H_cam - 1) / GRID_ROWS_VIZ))
            cv2.line(grid_cam, (0, y), (W_cam - 1, y), 255, 1)
        grid_cam_bgr = cv2.cvtColor(grid_cam, cv2.COLOR_GRAY2BGR)

        mx_rot = data['map_x_rot']
        my_rot = data['map_y_rot']
        mx_tps = data['map_x_tps']
        my_tps = data['map_y_tps']

        # -- Remap grid images ------------------------------------------------
        warped_rot = cv2.remap(grid_cam_bgr, mx_rot, my_rot,
                               cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        warped_tps = cv2.remap(grid_cam_bgr, mx_tps, my_tps,
                               cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # -- Render on dark canvas, colorise: rot=green, tps=orange ----------
        cvs_rot = np.zeros((H_canvas, W_canvas, 3), np.uint8)
        cvs_tps = np.zeros((H_canvas, W_canvas, 3), np.uint8)

        mask_rot = warped_rot[:, :, 0] > 128
        mask_tps = warped_tps[:, :, 0] > 128
        cvs_rot[mask_rot] = (0, 200, 0)      # green
        cvs_tps[mask_tps] = (0, 140, 255)    # orange (BGR)

        # Camera source image as dim background
        src_img = images[name]
        src_warped_rot = cv2.remap(src_img, mx_rot, my_rot,
                                   cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        bg_dim = (src_warped_rot.astype(np.float32) * 0.25).astype(np.uint8)
        cvs_rot = np.maximum(cvs_rot, bg_dim)
        src_warped_tps = cv2.remap(src_img, mx_tps, my_tps,
                                   cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        bg_dim2 = (src_warped_tps.astype(np.float32) * 0.25).astype(np.uint8)
        cvs_tps = np.maximum(cvs_tps, bg_dim2)

        cv2.putText(cvs_rot, f'{short} rotation', (8, 22),
                    FONT, 0.6, (0, 200, 0), 1, cv2.LINE_AA)
        cv2.putText(cvs_tps, f'{short} LiDAR-corrected', (8, 22),
                    FONT, 0.6, (0, 140, 255), 1, cv2.LINE_AA)

        pair_panels.append(fit_to_height(cvs_rot, TARGET_H))
        pair_panels.append(fit_to_height(cvs_tps, TARGET_H))

    return hstack_equal_height(pair_panels, TARGET_H, OUTPUT_W, gap=4)


# -- Panel D -------------------------------------------------------------------

def panel_D(images: dict, cams: dict, cam_data: dict,
            f_cyl: float, cx_canvas: float, cy_canvas: float,
            W_canvas: int, H_canvas: int) -> np.ndarray:
    """Seam alignment: FL/FC and FC/FR seam strip comparison.

    For each seam:
      [FL_rot_strip | FC_rot_strip]  vs  [FL_tps_strip | FC_tps_strip]
    Plus LiDAR ring elevation overlay lines.
    """
    STRIP_W = 60   # pixels wide per strip
    TARGET_H = 320

    # Find seam column for each pair: x where one camera's valid mask ends / another begins
    def find_seam_col(valid_left: np.ndarray, valid_right: np.ndarray) -> int:
        """Find approximate seam x-column between left/right camera footprints."""
        # Find where left camera's rightmost valid column is
        left_rbound = np.max(np.where(valid_left.any(axis=0))[0]) if valid_left.any() else W_canvas // 2
        right_lbound = np.min(np.where(valid_right.any(axis=0))[0]) if valid_right.any() else W_canvas // 2
        return int((left_rbound + right_lbound) // 2)

    def extract_seam_strips(seam_x: int, warp_key: str) -> tuple:
        """Extract LEFT strip from left camera and RIGHT strip from right camera."""
        mx_l = cam_data[FRONT_CAMS[0]][warp_key + '_rot'] if warp_key == 'map_x' else cam_data[FRONT_CAMS[0]]['map_x_' + warp_key]
        my_l = cam_data[FRONT_CAMS[0]]['map_y_' + warp_key]
        return mx_l, my_l

    # Compute warped images for rotation and TPS
    warped_imgs_rot = {}
    warped_imgs_tps = {}
    valid_rot_masks = {}
    valid_tps_masks = {}
    for name in FRONT_CAMS:
        cam = cams[name]
        W_cam, H_cam = cam['W'], cam['H']
        img = images[name]
        mx_r = cam_data[name]['map_x_rot']
        my_r = cam_data[name]['map_y_rot']
        mx_t = cam_data[name]['map_x_tps']
        my_t = cam_data[name]['map_y_tps']
        warped_imgs_rot[name] = cv2.remap(img, mx_r, my_r,
                                          cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        warped_imgs_tps[name] = cv2.remap(img, mx_t, my_t,
                                          cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        valid_rot_masks[name] = ((mx_r >= 0) & (my_r >= 0) &
                                 (mx_r < W_cam) & (my_r < H_cam))
        valid_tps_masks[name] = ((mx_t >= 0) & (my_t >= 0) &
                                 (mx_t < W_cam) & (my_t < H_cam))

    # LiDAR ring elevation angles (AV2 sensor, approximate)
    RING_EL_DEG = [-15, -10, -7, -5, -3, -1, 0, 1, 3, 5, 7, 10, 15]

    def ring_rows(el_degrees):
        """Convert elevation angles to canvas row indices."""
        rows = []
        for el_deg in el_degrees:
            el_rad = np.deg2rad(el_deg)
            v = -f_cyl * el_rad + cy_canvas
            r = int(round(v))
            if 0 <= r < H_canvas:
                rows.append(r)
        return rows

    ring_row_list = ring_rows(RING_EL_DEG)

    def make_seam_comparison(left_name: str, right_name: str,
                              label_str: str) -> np.ndarray:
        vl_rot = valid_rot_masks[left_name]
        vr_rot = valid_rot_masks[right_name]

        seam_x = find_seam_col(vl_rot, vr_rot)

        # Extract strips from both warps
        x0_l = max(0, seam_x - STRIP_W)
        x1_l = seam_x
        x0_r = seam_x
        x1_r = min(W_canvas, seam_x + STRIP_W)

        def strip_pair(wl, wr, vl, vr):
            sl = wl[:, x0_l:x1_l].copy()
            sr = wr[:, x0_r:x1_r].copy()
            # Darken invalid pixels
            sl[~vl[:, x0_l:x1_l]] = (30, 30, 30)
            sr[~vr[:, x0_r:x1_r]] = (30, 30, 30)
            # Add ring elevation lines
            for row in ring_row_list:
                cv2.line(sl, (0, row), (sl.shape[1]-1, row), (0, 80, 180), 1)
                cv2.line(sr, (0, row), (sr.shape[1]-1, row), (0, 80, 180), 1)
            return np.hstack([sl, np.full((H_canvas, 2, 3), (80,80,80), np.uint8), sr])

        strip_rot = strip_pair(
            warped_imgs_rot[left_name], warped_imgs_rot[right_name],
            valid_rot_masks[left_name], valid_rot_masks[right_name])
        strip_tps = strip_pair(
            warped_imgs_tps[left_name], warped_imgs_tps[right_name],
            valid_tps_masks[left_name], valid_tps_masks[right_name])

        divider = np.full((H_canvas, 6, 3), (120, 120, 0), np.uint8)
        combined = np.hstack([strip_rot, divider, strip_tps])

        # Add labels
        cv2.putText(combined, f'{label_str}  rotation', (4, 20),
                    FONT, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        w_rot_strip = strip_rot.shape[1] + 6
        cv2.putText(combined, f'{label_str}  LiDAR-corrected',
                    (w_rot_strip + 4, 20),
                    FONT, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(combined, f'seam x~{seam_x}', (4, 40),
                    FONT, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

        return combined

    seam_fl_fc = make_seam_comparison(
        'ring_front_left', 'ring_front_center', 'FL/FC seam')
    seam_fc_fr = make_seam_comparison(
        'ring_front_center', 'ring_front_right', 'FC/FR seam')

    row1 = fit_to_height(seam_fl_fc, TARGET_H)
    row2 = fit_to_height(seam_fc_fr, TARGET_H)
    combined = hstack_equal_height([row1, row2], TARGET_H, OUTPUT_W, gap=8)
    return combined


# -- Panel E -------------------------------------------------------------------

def panel_E(cam_data: dict, W_canvas: int, H_canvas: int) -> np.ndarray:
    """Correction field Δx and Δy heatmaps (TPS − rotation).

    Blue=negative, black=zero, red=positive. Three cameras side by side.
    """
    TARGET_H = max(80, int(round(H_canvas * OUTPUT_W / (W_canvas * 3 + 16))))
    TARGET_H = min(TARGET_H, 300)

    all_panels = []
    for name in FRONT_CAMS:
        data  = cam_data[name]
        short = CAM_SHORT[name]
        mx_r  = data['map_x_rot']
        my_r  = data['map_y_rot']
        mx_t  = data['map_x_tps']
        my_t  = data['map_y_tps']

        # Only compute within rotation-valid FOV
        valid = (mx_r >= 0) & (my_r >= 0)

        dx = mx_t - mx_r
        dy = my_t - my_r
        dx[~valid] = 0.0
        dy[~valid] = 0.0

        if valid.any():
            dx_vals = dx[valid]
            dy_vals = dy[valid]
            cap = max(float(np.percentile(np.abs(dx_vals), 95)), 1.0)
            cap_y = max(float(np.percentile(np.abs(dy_vals), 95)), 1.0)
        else:
            cap = 10.0
            cap_y = 10.0

        dx_heat = diverging_colormap(dx, -cap, cap)
        dy_heat = diverging_colormap(dy, -cap_y, cap_y)

        dx_heat[~valid] = [20, 20, 20]
        dy_heat[~valid] = [20, 20, 20]

        # Add labels
        cv2.putText(dx_heat, f'{short} Dx  (p95={cap:.1f}px)', (6, 20),
                    FONT, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(dy_heat, f'{short} Dy  (p95={cap_y:.1f}px)', (6, 20),
                    FONT, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        # Put Dx and Dy side by side for this camera
        pair = np.hstack([dx_heat,
                          np.full((H_canvas, 4, 3), BG_COLOR, np.uint8),
                          dy_heat])
        all_panels.append(fit_to_height(pair, TARGET_H))

    return hstack_equal_height(all_panels, TARGET_H, OUTPUT_W, gap=4)


# -- Main ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description='QA/debug script for LiDAR-guided panoramic stitching')
    ap.add_argument('--frame',       type=int, default=0,
                    help='Frame index in frames.json')
    ap.add_argument('--sensor-root', default=str(SENSOR_ROOT))
    ap.add_argument('--calib',       default=str(CALIB_JSON))
    ap.add_argument('--frames',      default=str(FRAMES_JSON))
    ap.add_argument('--out',         default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sensor_root = Path(args.sensor_root)

    # -- Load calibration ------------------------------------------------------
    cams = load_calib(args.calib)
    print(f'Calibration: {len(cams)} cameras loaded')

    # -- Load frame ------------------------------------------------------------
    with open(args.frames) as fh:
        frames = json.load(fh)
    frame = frames[args.frame]

    # -- Find and load nearest LiDAR sweep ------------------------------------
    lidar_files: dict[int, Path] = {}
    for fp in sensor_root.glob('train/*/sensors/lidar/*.feather'):
        try:
            lidar_files[int(fp.stem)] = fp
        except ValueError:
            pass
    if not lidar_files:
        raise RuntimeError(f'No LiDAR feather files found under {sensor_root}')
    lidar_ts  = np.array(sorted(lidar_files.keys()), dtype=np.int64)
    ref_cam   = 'ring_front_center'
    img_ts    = int(Path(frame[ref_cam]).stem)
    idx       = int(np.argmin(np.abs(lidar_ts - img_ts)))
    best_ts   = int(lidar_ts[idx])
    dt_ms     = abs(best_ts - img_ts) / 1e6
    print(f'Frame {args.frame}  img_ts={img_ts}  lidar_ts={best_ts}  dt={dt_ms:.1f}ms')

    df  = pd.read_feather(lidar_files[best_ts])
    pts = df[['x', 'y', 'z']].values.astype(np.float32)
    print(f'LiDAR points: {len(pts)}')

    # -- Load camera images ----------------------------------------------------
    images: dict[str, np.ndarray] = {}
    for name in FRONT_CAMS:
        path = frame.get(name)
        if path is None:
            raise KeyError(f'{name} not in frame {args.frame}')
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(path)
        images[name] = img
        print(f'  {name}: {img.shape[1]}x{img.shape[0]}')

    # -- Canvas geometry -------------------------------------------------------
    f_cyl, cx_canvas, cy_canvas, W_canvas, H_canvas, Z_REF = compute_canvas_geometry(
        pts, cams, ref_cam)
    print(f'Canvas: {W_canvas}x{H_canvas}  f_cyl={f_cyl:.0f}  '
          f'cx={cx_canvas:.0f}  cy={cy_canvas:.0f}  Z_REF={Z_REF:.3f}m')

    # -- Per-camera data -------------------------------------------------------
    print('\nComputing per-camera remap data...')
    cam_data = compute_camera_data(
        pts, cams, f_cyl, cx_canvas, cy_canvas, W_canvas, H_canvas, z_ref=Z_REF)

    # -- Build panels ----------------------------------------------------------
    W = OUTPUT_W
    rows = []

    def add(label: str, img: np.ndarray):
        rows.append(label_bar(f'  {label}', W))
        rows.append(img)
        rows.append(separator(W))

    print('\nBuilding Panel A...')
    add('A | LiDAR correspondences: camera vs canvas  (numbers 1–20, red=near  blue=far)',
        panel_A(images, cams, cam_data, f_cyl, cx_canvas, cy_canvas,
                W_canvas, H_canvas))

    print('Building Panel B...')
    add('B | Correction arrows: yellow=rotation model  cyan=LiDAR truth  '
        '(coherent = LiDAR working)',
        panel_B(cams, cam_data, f_cyl, cx_canvas, cy_canvas,
                W_canvas, H_canvas))

    print('Building Panel C...')
    add('C | Warp grid: green=rotation  orange=LiDAR-corrected  '
        '(smooth deformation = good, folds = bad)',
        panel_C(images, cams, cam_data, f_cyl, cx_canvas, cy_canvas,
                W_canvas, H_canvas))

    print('Building Panel D...')
    add('D | Seam alignment: [rotation] vs [LiDAR-corrected]  '
        '(blue lines = LiDAR ring elevations)',
        panel_D(images, cams, cam_data, f_cyl, cx_canvas, cy_canvas,
                W_canvas, H_canvas))

    print('Building Panel E...')
    add('E | Correction field heatmaps: Dx (horizontal) and Dy (vertical)  '
        'blue=neg  black=0  red=pos',
        panel_E(cam_data, W_canvas, H_canvas))

    # -- Write output ----------------------------------------------------------
    out_img  = np.vstack(rows)
    out_path = out_dir / f'qa_frame_{args.frame:04d}.jpg'
    cv2.imwrite(str(out_path), out_img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f'\nSaved -> {out_path}  ({out_img.shape[1]}x{out_img.shape[0]})')


if __name__ == '__main__':
    main()
