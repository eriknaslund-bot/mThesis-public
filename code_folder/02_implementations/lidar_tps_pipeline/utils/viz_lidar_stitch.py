#!/usr/bin/env python3
"""End-to-end visualization of the LiDAR-guided stitching pipeline.

Produces a single tall PNG with annotated panels:

  1. Source images
  2. LiDAR projected onto cameras (depth-coloured dots)
  3. TPS control points  (LiDAR grid pts + ghost anchors)
  4. Camera footprints -- rotation-only  (colour-coded per camera)
  5. Camera footprints -- TPS            (colour-coded per camera)
  6. Final composite -- rotation-only
  7. Final composite -- TPS
  8. Warp displacement heatmap (|TPS − rotation| in pixels)

Usage:
    python viz_lidar_stitch.py
    python viz_lidar_stitch.py --frame 3
    python viz_lidar_stitch.py --frame 0 --out output/viz
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# Import pipeline functions from sibling script
sys.path.insert(0, str(Path(__file__).parent))
from lidar_ring_stitch import (
    FRONT_CAMS, LIDAR_MAX_RANGE_M, LIDAR_MIN_CTRL_RANGE_M, CANVAS_MARGIN_FRAC,
    SEAM_HALF_WIDTH_PX, TPS_SMOOTHING,
    load_calib, project_with_ego, ego_to_canvas, cam_pixel_to_canvas_rot,
    find_shared_ctrl_pts,
    build_tps_remap, build_rotation_remap,
    paste_with_feather, depth_color,
)

# -- Layout constants ----------------------------------------------------------
OUTPUT_W      = 3000     # width of each panel in the final image
CAM_PANEL_H   = 380      # height of camera-image panels (1–3)
LABEL_H       = 28       # height of per-panel text label bar
FONT          = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE    = 0.65
FONT_THICK    = 1
BG_COLOR      = (25, 25, 25)   # dark background between panels
LABEL_BG      = (45, 45, 45)
LABEL_FG      = (220, 220, 220)

# Per-camera BGR tints used in footprint panels
# FL=red  FC=green  FR=blue  -> overlaps show yellow/cyan (additive max-blend)
CAM_TINTS = {
    'ring_front_left':   2,   # BGR channel index -> red
    'ring_front_center': 1,   # green
    'ring_front_right':  0,   # blue
}

# Default paths (same as lidar_ring_stitch.py)
SENSOR_ROOT = Path.home() / 'mThesis/argo2_data/sensor'
CALIB_JSON  = Path.home() / 'mThesis/argo2_data/extracted/calibration.json'
FRAMES_JSON = Path.home() / 'mThesis/argo2_data/extracted/frames.json'
OUT_DIR     = Path(__file__).parent.parent / 'output/lidar_ring_stitch'


# -- Drawing helpers -----------------------------------------------------------

def label_bar(text: str, width: int, h: int = LABEL_H) -> np.ndarray:
    bar = np.full((h, width, 3), LABEL_BG, dtype=np.uint8)
    cv2.putText(bar, text, (8, h - 7), FONT, FONT_SCALE, LABEL_FG, FONT_THICK,
                cv2.LINE_AA)
    return bar


def separator(width: int, h: int = 4) -> np.ndarray:
    return np.full((h, width, 3), BG_COLOR, dtype=np.uint8)


def fit_to_width(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    new_h = int(round(h * width / w))
    return cv2.resize(img, (width, new_h), interpolation=cv2.INTER_LINEAR)


def fit_to_height(img: np.ndarray, height: int) -> np.ndarray:
    h, w = img.shape[:2]
    new_w = int(round(w * height / h))
    return cv2.resize(img, (new_w, height), interpolation=cv2.INTER_LINEAR)


def hstack_to_width(imgs: list[np.ndarray], target_w: int,
                    target_h: int, gap: int = 4) -> np.ndarray:
    """Scale images to the same height, hstack, then fit to target_w."""
    scaled = [fit_to_height(im, target_h) for im in imgs]
    # Fill any width gap with bg color to reach at least target_w
    total_w = sum(im.shape[1] for im in scaled) + gap * (len(scaled) - 1)
    row = np.full((target_h, total_w, 3), BG_COLOR, dtype=np.uint8)
    x = 0
    for im in scaled:
        row[:, x:x + im.shape[1]] = im
        x += im.shape[1] + gap
    # Resize row to exact output width
    return cv2.resize(row, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def draw_lidar_dots(img: np.ndarray, cam_px: np.ndarray,
                    z_vals: np.ndarray, radius: int = 3) -> np.ndarray:
    out = img.copy()
    if len(cam_px) == 0:
        return out
    order  = np.argsort(z_vals)[::-1]
    colors = depth_color(z_vals[order])
    H, W = out.shape[:2]
    for (u, v), c in zip(cam_px[order], colors):
        px, py = int(round(float(u))), int(round(float(v)))
        if 0 <= px < W and 0 <= py < H:
            cv2.circle(out, (px, py), radius, (int(c[0]), int(c[1]), int(c[2])), -1)
    return out


def draw_control_pts(img: np.ndarray,
                     lidar_pts: np.ndarray, lidar_z: np.ndarray) -> np.ndarray:
    """Draw shared LiDAR control pts -- depth-coloured filled circles."""
    out = img.copy()
    H, W = out.shape[:2]
    if len(lidar_pts) > 0:
        order  = np.argsort(lidar_z)[::-1]
        colors = depth_color(lidar_z[order])
        for (u, v), c in zip(lidar_pts[order], colors):
            px, py = int(round(float(u))), int(round(float(v)))
            if 0 <= px < W and 0 <= py < H:
                cv2.circle(out, (px, py), 5, (int(c[0]), int(c[1]), int(c[2])), -1)
                cv2.circle(out, (px, py), 5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def tint_warp(warped: np.ndarray, valid: np.ndarray,
              channel: int, canvas_shape: tuple) -> np.ndarray:
    """Convert warped camera image to a single-channel tinted contribution map.

    Returns an (H, W, 3) uint8 image where only `channel` (0=B,1=G,2=R) is set.
    Pixels outside the camera FOV are black.
    """
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY).astype(np.float32)
    out  = np.zeros(canvas_shape, dtype=np.float32)
    out[:, :, channel] = gray
    # Blank out invalid (outside FOV) pixels
    inv = ~valid
    out[inv] = 0
    return out.clip(0, 255).astype(np.uint8)


DISP_CAP_PX = 100.0   # heatmap saturates at this displacement (px)


def displacement_heatmap(map_x_tps: np.ndarray, map_y_tps: np.ndarray,
                          map_x_rot: np.ndarray, map_y_rot: np.ndarray,
                          W_cam: int, H_cam: int) -> tuple:
    """Pixel-wise displacement magnitude |TPS − rotation| -> colourised heatmap.

    Normalises to DISP_CAP_PX so the interesting interior corrections are visible
    (the TPS tail at FOV boundaries is clipped/saturated).
    """
    valid = ((map_x_tps >= 0) & (map_y_tps >= 0) &
             (map_x_rot >= 0) & (map_y_rot >= 0) &
             (map_x_tps < W_cam) & (map_y_tps < H_cam) &
             (map_x_rot < W_cam) & (map_y_rot < H_cam))
    disp = np.zeros(map_x_tps.shape, np.float32)
    disp[valid] = np.sqrt((map_x_tps[valid] - map_x_rot[valid])**2 +
                          (map_y_tps[valid] - map_y_rot[valid])**2)
    median_px = float(np.median(disp[valid])) if valid.any() else 0.0
    norm   = np.clip(disp / DISP_CAP_PX, 0, 1)
    heat_u8 = (norm * 255).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_INFERNO)
    heat_bgr[~valid] = 0
    return heat_bgr, median_px


# -- Panel builders ------------------------------------------------------------

def panel_source(images: dict, width: int) -> np.ndarray:
    imgs = [images[n] for n in FRONT_CAMS]
    return hstack_to_width(imgs, width, CAM_PANEL_H)


def panel_lidar(images: dict, cam_hits: dict, width: int) -> np.ndarray:
    panels = []
    for name in FRONT_CAMS:
        cam_px, z_v = cam_hits[name]
        overlaid = draw_lidar_dots(images[name], cam_px, z_v)
        panels.append(overlaid)
    return hstack_to_width(panels, width, CAM_PANEL_H)


def panel_ctrl_pts(images: dict, ctrl_data: dict, width: int) -> np.ndarray:
    panels = []
    for name in FRONT_CAMS:
        src_pts, z_v = ctrl_data[name]
        drawn = draw_control_pts(images[name], src_pts, z_v)
        panels.append(drawn)
    return hstack_to_width(panels, width, CAM_PANEL_H)


def panel_footprints(warped_per_cam: dict, valid_per_cam: dict,
                     W_canvas: int, H_canvas: int, width: int) -> np.ndarray:
    """Colour-coded per-camera footprints using additive max-blend."""
    canvas = np.zeros((H_canvas, W_canvas, 3), np.uint8)
    for name in FRONT_CAMS:
        ch = CAM_TINTS[name]
        tinted = tint_warp(warped_per_cam[name], valid_per_cam[name], ch,
                           (H_canvas, W_canvas, 3))
        canvas = np.maximum(canvas, tinted)

    # Draw camera-name colour legend
    legend_colors = {'ring_front_left':   (0, 0, 200),
                     'ring_front_center': (0, 180, 0),
                     'ring_front_right':  (200, 0, 0)}
    x = 12
    for name in FRONT_CAMS:
        short = name.replace('ring_front_', '')
        c = legend_colors[name]
        cv2.rectangle(canvas, (x, 8), (x + 14, 22), c, -1)
        cv2.putText(canvas, short, (x + 18, 21), FONT, 0.5, (200, 200, 200),
                    1, cv2.LINE_AA)
        x += 90
    note = 'yellow=FL+FC overlap  cyan=FC+FR overlap'
    cv2.putText(canvas, note, (12, H_canvas - 8), FONT, 0.45, (150, 150, 150),
                1, cv2.LINE_AA)
    return fit_to_width(canvas, width)


def panel_composite(canvas_img: np.ndarray, width: int) -> np.ndarray:
    return fit_to_width(canvas_img, width)


def panel_displacement(disps: dict, p95s: dict,
                       W_canvas: int, H_canvas: int, width: int) -> np.ndarray:
    """Side-by-side displacement maps for the 3 cameras."""
    panels = []
    for name in FRONT_CAMS:
        heat = disps[name]
        p95  = p95s[name]
        short = name.replace('ring_front_', '')
        cv2.putText(heat, f'{short}  median={p95:.0f}px  scale=0–{DISP_CAP_PX:.0f}px', (6, 20),
                    FONT, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
        panels.append(heat)
    row = hstack_to_width(panels, width, int(round(H_canvas * width / (W_canvas * 3 + 8))))
    return row


# -- Main ----------------------------------------------------------------------

def main():
    import json

    ap = argparse.ArgumentParser(description='LiDAR stitch pipeline visualizer')
    ap.add_argument('--frame',       type=int, default=0)
    ap.add_argument('--sensor-root', default=str(SENSOR_ROOT))
    ap.add_argument('--calib',       default=str(CALIB_JSON))
    ap.add_argument('--frames',      default=str(FRAMES_JSON))
    ap.add_argument('--out',         default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sensor_root = Path(args.sensor_root)

    # -- Load calibration + frame ----------------------------------------------
    cams = load_calib(args.calib)
    with open(args.frames) as f:
        frames = json.load(f)
    frame = frames[args.frame]

    # -- Load LiDAR ------------------------------------------------------------
    lidar_files: dict[int, Path] = {}
    for fp in sensor_root.glob('train/*/sensors/lidar/*.feather'):
        try:
            lidar_files[int(fp.stem)] = fp
        except ValueError:
            pass
    lidar_ts = np.array(sorted(lidar_files.keys()), dtype=np.int64)

    ref_cam = 'ring_front_center'
    img_ts  = int(Path(frame[ref_cam]).stem)
    idx     = int(np.argmin(np.abs(lidar_ts - img_ts)))
    best_ts = int(lidar_ts[idx])
    df      = pd.read_feather(lidar_files[best_ts])
    pts     = df[['x', 'y', 'z']].values.astype(np.float32)
    dt_ms   = abs(best_ts - img_ts) / 1e6
    print(f'Frame {args.frame}  LiDAR dt={dt_ms:.1f}ms  pts={len(pts)}')

    # -- Load images -----------------------------------------------------------
    images: dict[str, np.ndarray] = {}
    for name in FRONT_CAMS:
        img = cv2.imread(frame[name])
        if img is None:
            raise FileNotFoundError(frame[name])
        images[name] = img
        print(f'  {name}: {img.shape[1]}x{img.shape[0]}')

    # -- Canvas geometry -------------------------------------------------------
    f_cyl = float(cams[ref_cam]['fx'])
    # Reference height: mean camera z -- elevations measured from here so that
    # ego_to_canvas and rotation-model rays agree (minimises TPS corrections).
    Z_REF = float(np.mean([cams[n]['t'][2] for n in FRONT_CAMS]))

    # -- Project LiDAR into all cameras ONCE ----------------------------------
    # Each point is projected once per camera; shared-point extraction is then
    # a cheap index set-intersection -- no redundant reprojection.
    proj: dict[str, tuple] = {}
    for name in FRONT_CAMS:
        px, ego, idx = project_with_ego(pts, cams[name], return_indices=True)
        proj[name] = (px, ego, idx)
        print(f'  {name}: {len(px)} hits')

    all_az, all_el = [], []
    for name in FRONT_CAMS:
        _, pts_v, _ = proj[name]
        if len(pts_v) == 0:
            continue
        az   = np.arctan2(pts_v[:, 1], pts_v[:, 0])
        r_xy = np.sqrt(pts_v[:, 0]**2 + pts_v[:, 1]**2)
        el   = np.arctan2(pts_v[:, 2] - Z_REF, r_xy)   # from camera height
        all_az.append(az);  all_el.append(el)
    az_all = np.concatenate(all_az);  el_all = np.concatenate(all_el)
    az_min, az_max = float(az_all.min()), float(az_all.max())
    el_min, el_max = float(el_all.min()), float(el_all.max())
    mg = 0.05
    az_min -= mg*(az_max-az_min);  az_max += mg*(az_max-az_min)
    el_min -= mg*(el_max-el_min);  el_max += mg*(el_max-el_min)
    W_canvas = max(1, int(np.ceil(f_cyl * (az_max - az_min))))
    H_canvas = max(1, int(np.ceil(f_cyl * (el_max - el_min))))
    cx_canvas = float(az_max * f_cyl)    # az_max (FL, leftmost) -> u=0
    cy_canvas = float(el_max * f_cyl)    # el_max -> v=0 (top = sky)
    print(f'Canvas {W_canvas}x{H_canvas}  f_cyl={f_cyl:.0f}  Z_REF={Z_REF:.3f}m')

    # -- Rotation remaps + overlap masks --------------------------------------
    rot_remaps, rot_valid = {}, {}
    for name in FRONT_CAMS:
        mx, my = build_rotation_remap(cams[name], f_cyl, cx_canvas, cy_canvas,
                                      W_canvas, H_canvas)
        rot_remaps[name] = (mx, my)
        rot_valid[name]  = (mx >= 0) & (my >= 0)

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
    overlap_masks = {
        FL: strip_FL_FC,
        FC: np.maximum(strip_FL_FC, strip_FC_FR),
        FR: strip_FC_FR,
    }

    # -- Shared control points -------------------------------------------------
    # Canvas targets use FC's rotation model -- consistent with main pipeline.
    sh_px_FL, sh_px_FC_l, sh_cvs_FL_FC = find_shared_ctrl_pts(
        *proj[FL], *proj[FC],
        f_cyl, cx_canvas, cy_canvas, Z_REF,
        ref_cam=cams[FC], ref_side='right')
    sh_px_FC_r, sh_px_FR, sh_cvs_FC_FR = find_shared_ctrl_pts(
        *proj[FC], *proj[FR],
        f_cyl, cx_canvas, cy_canvas, Z_REF,
        ref_cam=cams[FC], ref_side='left')
    shared_src = {FL: sh_px_FL, FC: np.empty((0, 2), np.float32), FR: sh_px_FR}
    shared_dst = {FL: sh_cvs_FL_FC, FC: np.empty((0, 2), np.float32), FR: sh_cvs_FC_FR}

    # -- Compute all per-camera intermediate data ------------------------------
    cam_hits:     dict = {}   # name -> (cam_px, z_vals)   [all LiDAR hits]
    ctrl_data:    dict = {}   # name -> (shared cam_px, z_vals)
    warped_rot:   dict = {}   # name -> warped image (rotation)
    warped_tps:   dict = {}   # name -> warped image (TPS)
    valid_rot:    dict = {}   # name -> bool mask
    valid_tps:    dict = {}
    disps:        dict = {}   # name -> heatmap image
    p95s:         dict = {}   # name -> 95th percentile displacement

    for name in FRONT_CAMS:
        cam    = cams[name]
        W_cam, H_cam = cam['W'], cam['H']
        img    = images[name]
        print(f'\n  [{name}]')

        # All LiDAR hits for dot overlay (from pre-computed upfront projection)
        cam_px_all, pts_valid_all, _ = proj[name]
        z_all = np.linalg.norm(pts_valid_all, axis=1) if len(pts_valid_all) else np.array([])
        cam_hits[name] = (cam_px_all, z_all)

        # Control pts: only shared LiDAR points visible in both cameras
        sh_src = shared_src[name]
        sh_dst = shared_dst[name]
        src_pts = sh_src
        dst_pts = sh_dst

        # z values for visualisation (seam pts only -- ghost anchors have no range)
        if len(sh_src) > 0 and len(cam_px_all) > 0:
            from scipy.spatial import cKDTree
            tree = cKDTree(cam_px_all)
            _, idxs = tree.query(sh_src, k=1)
            ctrl_z = z_all[idxs]
        else:
            ctrl_z = np.array([])
        ctrl_data[name] = (sh_src, ctrl_z)

        # Rotation remap
        mx_rot, my_rot = rot_remaps[name]
        w_rot = cv2.remap(img, mx_rot, my_rot, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        v_rot = (mx_rot >= 0) & (my_rot >= 0) & (mx_rot < W_cam) & (my_rot < H_cam)
        warped_rot[name] = w_rot
        valid_rot[name]  = v_rot

        # TPS remap -- FC is reference, kept as rotation-only
        if name == FC or len(src_pts) < 4:
            mx_tps, my_tps = mx_rot.copy(), my_rot.copy()
            if name == FC:
                print(f'    FC: rotation-only (reference camera)')
            else:
                print(f'    Insufficient LiDAR pts -- TPS = rotation for this camera')
        else:
            mx_tps, my_tps = build_tps_remap(src_pts, dst_pts, W_cam, H_cam,
                                              W_canvas, H_canvas,
                                              cam=cam, f_cyl=f_cyl,
                                              cx_canvas=cx_canvas,
                                              cy_canvas=cy_canvas,
                                              smoothing=TPS_SMOOTHING, remap_scale=0.5,
                                              overlap_mask=overlap_masks[name])
        w_tps = cv2.remap(img, mx_tps, my_tps, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        v_tps = (mx_tps >= 0) & (my_tps >= 0) & (mx_tps < W_cam) & (my_tps < H_cam)
        warped_tps[name] = w_tps
        valid_tps[name]  = v_tps

        # Displacement heatmap
        heat, p95 = displacement_heatmap(mx_tps, my_tps, mx_rot, my_rot, W_cam, H_cam)
        disps[name] = heat
        p95s[name]  = p95
        print(f'    Disp median = {p95:.1f}px  (scale 0–{DISP_CAP_PX:.0f}px)')

    # -- Build canvases --------------------------------------------------------
    canvas_rot = np.zeros((H_canvas, W_canvas, 3), np.uint8)
    canvas_tps = np.zeros((H_canvas, W_canvas, 3), np.uint8)
    for name in FRONT_CAMS:
        paste_with_feather(canvas_rot, warped_rot[name], valid_rot[name])
        paste_with_feather(canvas_tps, warped_tps[name], valid_tps[name])

    # -- Assemble visualization ------------------------------------------------
    W = OUTPUT_W
    rows = []

    def add(label: str, img: np.ndarray):
        rows.append(label_bar(f'  {label}', W))
        rows.append(img)
        rows.append(separator(W))

    add('1 | Source images  (FL -- FC -- FR)',
        panel_source(images, W))

    add('2 | LiDAR projected onto cameras  (red=near  blue=far)',
        panel_lidar(images, cam_hits, W))

    add('3 | TPS control points  (● shared LiDAR pts in both cameras, depth-coloured)',
        panel_ctrl_pts(images, ctrl_data, W))

    add('4 | Camera footprints -- rotation-only warp  (red=FL  green=FC  blue=FR)',
        panel_footprints(warped_rot, valid_rot, W_canvas, H_canvas, W))

    add('5 | Camera footprints -- TPS warp  (red=FL  green=FC  blue=FR)',
        panel_footprints(warped_tps, valid_tps, W_canvas, H_canvas, W))

    add('6 | Final composite -- rotation-only',
        panel_composite(canvas_rot, W))

    add('7 | Final composite -- TPS (LiDAR-guided)',
        panel_composite(canvas_tps, W))

    add(f'8 | Warp displacement |TPS − rotation|  (black=0px  white={DISP_CAP_PX:.0f}px, capped)',
        panel_displacement(disps, p95s, W_canvas, H_canvas, W))

    out_img = np.vstack(rows)
    out_path = out_dir / f'frame_{args.frame:04d}_viz.jpg'
    cv2.imwrite(str(out_path), out_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f'\nSaved -> {out_path}  ({out_img.shape[1]}x{out_img.shape[0]})')


if __name__ == '__main__':
    main()
