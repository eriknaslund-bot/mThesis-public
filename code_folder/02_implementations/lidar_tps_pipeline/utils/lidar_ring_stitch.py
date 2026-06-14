#!/usr/bin/env python3
"""LiDAR-guided geometric stitcher for 3 front cameras (FL, FC, FR).

Each camera is independently warped to a shared cylindrical canvas using
LiDAR points as correspondences: each 3D ego-frame point projects to a pixel
in the camera image AND to a canvas pixel via spherical math.  TPS (thin-plate
spline) interpolates the full dense warp from ~300 subsampled control points.
Ghost anchor points at image corners (from rotation homography) keep the
extrapolation stable outside the LiDAR convex hull.

Canvas convention
-----------------
  u = f_cyl * atan2(P.y, P.x) + cx_canvas   (azimuth -> x)
  v = f_cyl * atan2(P.z, r_xy)  + cy_canvas  (elevation -> y, positive = up)

with cx_canvas = -az_min * f_cyl and cy_canvas = -el_min * f_cyl so that
the minimum azimuth/elevation maps to u=0, v=0.

Usage
-----
    python lidar_ring_stitch.py
    python lidar_ring_stitch.py --frame 5
    python lidar_ring_stitch.py --frame 0 --debug      # LiDAR ring dot overlay
    python lidar_ring_stitch.py --frame 0 --no-tps     # rotation homography only
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# -- Default paths ------------------------------------------------------------
SENSOR_ROOT = Path.home() / 'mThesis/argo2_data/sensor'
CALIB_JSON  = Path.home() / 'mThesis/argo2_data/extracted/calibration.json'
FRAMES_JSON = Path.home() / 'mThesis/argo2_data/extracted/frames.json'
OUT_DIR     = Path(__file__).parent.parent / 'output/lidar_ring_stitch'

LIDAR_MAX_RANGE_M = 120.0
# Minimum range for TPS control points.  Below this threshold parallax grows
# large enough (>70px) to destabilise the RBF -- these objects are typically
# large close-range obstacles that span both cameras anyway.
LIDAR_MIN_CTRL_RANGE_M = 8.0
FRONT_CAMS = ['ring_front_left', 'ring_front_center', 'ring_front_right']

# TPS control-point grid dimensions (cols x rows in camera image space)
GRID_COLS = 20
GRID_ROWS = 15

# Ghost anchors: N points per image edge (4 edges x N)
GHOST_PER_EDGE = 5

# Canvas azimuth/elevation margin beyond the LiDAR point cloud extent
CANVAS_MARGIN_FRAC = 0.05

# Half-width (in canvas pixels) of the correction strip around each seam.
# TPS corrections fade linearly from 1.0 at the seam centre to 0.0 at
# ±SEAM_HALF_WIDTH_PX.  Keeping this narrow prevents seam-zone corrections
# from bleeding into the camera interiors and distorting the whole image.
SEAM_HALF_WIDTH_PX = 800

# RBF smoothing parameter for TPS correction field.  0 = exact interpolation.
# With mcr>=12 filtering noisy close-range points, exact interpolation is safe.
TPS_SMOOTHING  = 0.0
LOCAL_SIGMA    = 150.0   # Gaussian radius (canvas px) for locally-weighted correction
FEATHER_HALF   = 40      # ±px narrow feather band around overlap midpoint (0 = full overlap)


# -- Geometry helpers ---------------------------------------------------------

def quat_to_mat(qw, qx, qy, qz):
    """Unit quaternion -> 3x3 rotation matrix (camera -> ego frame)."""
    n = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    return np.array([
        [1-2*(qy**2+qz**2),   2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [  2*(qx*qy+qz*qw), 1-2*(qx**2+qz**2),   2*(qy*qz-qx*qw)],
        [  2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw), 1-2*(qx**2+qy**2)],
    ])


def load_calib(path: str) -> dict:
    """Load ring-camera calibration from JSON -> dict keyed by camera name."""
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


def project_with_ego(pts_ego: np.ndarray, cam: dict,
                     return_indices: bool = False):
    """Project ego-frame LiDAR points into camera; return both pixel coords and ego pts.

    Returns
    -------
    cam_px : (N, 2) float32  -- (u, v) in camera image for valid hits
    valid_ego : (N, 3) float32  -- ego-frame coordinates of those same points
    indices : (N,) int  -- original indices into pts_ego (only when return_indices=True)
    """
    pts_c = (pts_ego - cam['t']) @ cam['R']   # (n,3) camera-frame coords
    z = pts_c[:, 2]
    keep = (z > 0.5) & (z <= LIDAR_MAX_RANGE_M)
    if not keep.any():
        if return_indices:
            return (np.empty((0, 2), np.float32),
                    np.empty((0, 3), np.float32),
                    np.empty(0, np.int64))
        return np.empty((0, 2), np.float32), np.empty((0, 3), np.float32)
    keep_idx  = np.where(keep)[0]
    pts_c_v   = pts_c[keep]
    pts_ego_v = pts_ego[keep]
    z_v       = z[keep]
    u = cam['fx'] * pts_c_v[:, 0] / z_v + cam['cx']
    v = cam['fy'] * pts_c_v[:, 1] / z_v + cam['cy']
    W, H = cam['W'], cam['H']
    m = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if return_indices:
        return (np.column_stack([u[m], v[m]]).astype(np.float32),
                pts_ego_v[m].astype(np.float32),
                keep_idx[m])
    return (np.column_stack([u[m], v[m]]).astype(np.float32),
            pts_ego_v[m].astype(np.float32))


def find_shared_ctrl_pts(px_l: np.ndarray, ego_l: np.ndarray, idx_l: np.ndarray,
                         px_r: np.ndarray, ego_r: np.ndarray, idx_r: np.ndarray,
                         f_cyl: float, cx_canvas: float, cy_canvas: float,
                         z_ref: float = 0.0,
                         min_range_m: float = LIDAR_MIN_CTRL_RANGE_M,
                         ref_cam: dict = None,
                         ref_side: str = None):
    """Find shared control pts from pre-computed per-camera projections.

    Takes the output of project_with_ego(..., return_indices=True) for two
    adjacent cameras -- no redundant reprojection.  Points visible in both
    cameras are the same physical 3D point; both cameras must map them to the
    same canvas pixel, which is the sole alignment constraint.

    Parameters
    ----------
    px_l, ego_l, idx_l : output of project_with_ego for left camera
    px_r, ego_r, idx_r : output of project_with_ego for right camera
    ref_cam  : when provided, canvas targets are computed via the reference
               camera's rotation model (cam_pixel_to_canvas_rot) so that the
               warped camera lands exactly where FC places the same 3-D point.
               Use ref_cam=cams[FC] with ref_side='right' for the FL<->FC pair
               and ref_side='left' for the FC<->FR pair.
    ref_side : 'right' to use px_r (FC pixels in FL<->FC) or 'left' to use px_l
               (FC pixels in FC<->FR).

    Returns
    -------
    cam_px_l  : (N, 2) pixel coords in left camera
    cam_px_r  : (N, 2) pixel coords in right camera
    canvas_px : (N, 2) canvas coords -- identical target for both cameras
    """
    empty = np.empty((0, 2), np.float32)
    if len(idx_l) == 0 or len(idx_r) == 0:
        return empty, empty, empty

    # Set intersection on original LiDAR point indices
    map_l = {int(i): j for j, i in enumerate(idx_l)}
    map_r = {int(i): j for j, i in enumerate(idx_r)}
    shared_orig = sorted(set(map_l.keys()) & set(map_r.keys()))

    if not shared_orig:
        return empty, empty, empty

    jl = np.array([map_l[i] for i in shared_orig], dtype=np.int64)
    jr = np.array([map_r[i] for i in shared_orig], dtype=np.int64)

    # Minimum range filter
    ego_shared = ego_l[jl]
    range_m = np.linalg.norm(ego_shared, axis=1)
    far = range_m >= min_range_m
    jl, jr, ego_shared = jl[far], jr[far], ego_shared[far]

    if len(jl) == 0:
        return empty, empty, empty

    if ref_cam is not None and ref_side is not None:
        # Use the reference camera's rotation model as the canvas target.
        # This guarantees that the warped camera lands exactly where FC's
        # rotation model places the same physical point -- preventing ghosting
        # caused by parallax disagreement between ego_to_canvas and the FC
        # rotation projection for near objects.
        ref_px = px_r[jr] if ref_side == 'right' else px_l[jl]
        u_tgt, v_tgt = cam_pixel_to_canvas_rot(
            ref_px[:, 0], ref_px[:, 1], ref_cam, f_cyl, cx_canvas, cy_canvas)
        canvas_px = np.column_stack([u_tgt, v_tgt]).astype(np.float32)
    else:
        canvas_px = ego_to_canvas(ego_shared, f_cyl, cx_canvas, cy_canvas, z_ref)

    # Range-stratified sampling: 80% near (<30 m), 20% far (>=30 m).
    # Near objects have 10-100x more parallax and are the primary alignment
    # signal; sampling them preferentially produces larger, more visible TPS
    # corrections.
    range_m = np.linalg.norm(ego_shared, axis=1)
    N = len(jl)
    MAX_CTRL = 200
    if N > MAX_CTRL:
        rng   = np.random.default_rng(seed=0)
        near  = range_m < 30.0
        far   = ~near
        n_near = min(int(near.sum()), 160)
        n_far  = min(int(far.sum()),   40)
        near_idx = np.where(near)[0]
        far_idx  = np.where(far)[0]
        sel_near = rng.choice(near_idx, n_near, replace=False) if n_near > 0 else np.empty(0, dtype=np.int64)
        sel_far  = rng.choice(far_idx,  n_far,  replace=False) if n_far  > 0 else np.empty(0, dtype=np.int64)
        sel = np.concatenate([sel_near, sel_far]).astype(np.int64)
        return px_l[jl][sel], px_r[jr][sel], canvas_px[sel]

    return px_l[jl], px_r[jr], canvas_px


def _grid_subsample_with_idx(cam_px: np.ndarray, W: int, H: int,
                              n_cols: int = GRID_COLS, n_rows: int = GRID_ROWS):
    """Like grid_subsample but returns the surviving indices into cam_px."""
    if len(cam_px) == 0:
        return cam_px, np.empty(0, dtype=np.int64)
    cell_w = W / n_cols
    cell_h = H / n_rows
    out_idx = []
    for r in range(n_rows):
        v0, v1 = r * cell_h, (r + 1) * cell_h
        cy_cell = (v0 + v1) / 2
        row_m = (cam_px[:, 1] >= v0) & (cam_px[:, 1] < v1)
        if not row_m.any():
            continue
        for c in range(n_cols):
            u0, u1 = c * cell_w, (c + 1) * cell_w
            cx_cell = (u0 + u1) / 2
            m = row_m & (cam_px[:, 0] >= u0) & (cam_px[:, 0] < u1)
            if not m.any():
                continue
            indices = np.where(m)[0]
            d2 = ((cam_px[indices, 0] - cx_cell)**2 +
                  (cam_px[indices, 1] - cy_cell)**2)
            out_idx.append(indices[np.argmin(d2)])
    if not out_idx:
        return cam_px[:0], np.empty(0, dtype=np.int64)
    idx = np.array(out_idx, dtype=np.int64)
    return cam_px[idx], idx


def ego_to_canvas(pts_ego: np.ndarray, f_cyl: float,
                  cx_canvas: float, cy_canvas: float,
                  z_ref: float = 0.0) -> np.ndarray:
    """Cylindrical canvas placement: ego 3D points -> (N, 2) canvas pixels.

    Convention: u = -f*az + cx_canvas,  v = -f*el + cy_canvas
    Azimuth is negated so that left (positive y in AV2 ego frame) maps to
    the LEFT side of the canvas -- ring_front_left appears on the left.
    cx_canvas = f_cyl * az_max  so that az_max (leftmost) -> u=0.

    z_ref: camera height above ego origin (metres).  Elevation is measured
    relative to z_ref so that the rotation baseline (ray from camera) and
    this LiDAR formula agree for far objects, minimising TPS corrections.
    """
    az = np.arctan2(pts_ego[:, 1], pts_ego[:, 0])
    r_xy = np.sqrt(pts_ego[:, 0]**2 + pts_ego[:, 1]**2)
    el = np.arctan2(pts_ego[:, 2] - z_ref, r_xy)   # elevation from camera height
    u = -f_cyl * az + cx_canvas      # negated: left (positive az) -> small u (left)
    v = -f_cyl * el + cy_canvas       # negated: high el -> small v (top)
    return np.column_stack([u, v]).astype(np.float32)


def cam_pixel_to_canvas_rot(u_arr: np.ndarray, v_arr: np.ndarray,
                             cam: dict, f_cyl: float,
                             cx_canvas: float, cy_canvas: float):
    """Map camera pixels -> canvas pixels via rotation-only (far-field) model.

    Treats each pixel as a ray direction in ego frame; baseline ignored.

    Returns (u_canvas, v_canvas) arrays.
    """
    # Normalised camera-frame direction (not unit, but correct direction)
    px = (u_arr - cam['cx']) / cam['fx']
    py = (v_arr - cam['cy']) / cam['fy']
    pz = np.ones_like(px)
    p_cam = np.column_stack([px, py, pz])  # (N, 3)
    # Ego-frame direction: R (cam->ego) maps camera coords to ego coords
    p_ego = p_cam @ cam['R'].T             # (N, 3)  equivalent to (R @ p_cam.T).T
    az = np.arctan2(p_ego[:, 1], p_ego[:, 0])
    r_xy = np.sqrt(p_ego[:, 0]**2 + p_ego[:, 1]**2)
    el = np.arctan2(p_ego[:, 2], r_xy)
    return (-f_cyl * az + cx_canvas).astype(np.float32), \
           (-f_cyl * el + cy_canvas).astype(np.float32)   # az negated: left -> left


# -- Control-point sampling ----------------------------------------------------

def grid_subsample(cam_px: np.ndarray, canvas_px: np.ndarray,
                   W: int, H: int,
                   n_cols: int = GRID_COLS, n_rows: int = GRID_ROWS):
    """Pick one LiDAR hit per image grid cell (closest to cell centre).

    Returns
    -------
    src_pts : (M, 2) -- camera pixels for selected hits
    dst_pts : (M, 2) -- corresponding canvas pixels
    """
    if len(cam_px) == 0:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)

    cell_w = W / n_cols
    cell_h = H / n_rows
    src_out, dst_out = [], []

    for r in range(n_rows):
        v0, v1 = r * cell_h, (r + 1) * cell_h
        cy_cell = (v0 + v1) / 2
        row_m = (cam_px[:, 1] >= v0) & (cam_px[:, 1] < v1)
        if not row_m.any():
            continue
        for c in range(n_cols):
            u0, u1 = c * cell_w, (c + 1) * cell_w
            cx_cell = (u0 + u1) / 2
            m = row_m & (cam_px[:, 0] >= u0) & (cam_px[:, 0] < u1)
            if not m.any():
                continue
            indices = np.where(m)[0]
            d2 = ((cam_px[indices, 0] - cx_cell)**2 +
                  (cam_px[indices, 1] - cy_cell)**2)
            best = indices[np.argmin(d2)]
            src_out.append(cam_px[best])
            dst_out.append(canvas_px[best])

    if not src_out:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
    return np.array(src_out, np.float32), np.array(dst_out, np.float32)


def build_ghost_anchors(cam: dict, f_cyl: float,
                        cx_canvas: float, cy_canvas: float,
                        n_per_edge: int = GHOST_PER_EDGE,
                        exclude_right: bool = False,
                        exclude_left: bool = False):
    """Generate border anchor points using rotation-only canvas mapping.

    These constrain TPS extrapolation at image edges where LiDAR is sparse.

    Parameters
    ----------
    exclude_right : omit the right-edge anchors (use for FL, whose right edge
                    is the seam side with dense LiDAR coverage -- anchoring it
                    to δ=0 fights the genuine parallax corrections there).
    exclude_left  : omit the left-edge anchors (use for FR, same reason).

    Returns
    -------
    src_pts : (K*n, 2) camera pixels along selected edges
    dst_pts : (K*n, 2) corresponding canvas pixels (rotation-only model)
    """
    W, H = cam['W'], cam['H']
    ns = np.linspace(0, 1, n_per_edge)

    u_parts, v_parts = [], []
    # top edge
    u_parts.append(ns * (W - 1));  v_parts.append(np.zeros(n_per_edge))
    # bottom edge
    u_parts.append(ns * (W - 1));  v_parts.append(np.full(n_per_edge, H - 1))
    # left edge (skip if overlaps seam for FR)
    if not exclude_left:
        u_parts.append(np.zeros(n_per_edge));  v_parts.append(ns * (H - 1))
    # right edge (skip if overlaps seam for FL)
    if not exclude_right:
        u_parts.append(np.full(n_per_edge, W - 1));  v_parts.append(ns * (H - 1))

    us = np.concatenate(u_parts)
    vs = np.concatenate(v_parts)
    cu, cv = cam_pixel_to_canvas_rot(us, vs, cam, f_cyl, cx_canvas, cy_canvas)
    return (np.column_stack([us, vs]).astype(np.float32),
            np.column_stack([cu, cv]).astype(np.float32))


# -- Warp builders -------------------------------------------------------------

def build_tps_remap(src_pts: np.ndarray, dst_pts: np.ndarray,
                    W_cam: int, H_cam: int,
                    W_canvas: int, H_canvas: int,
                    cam: dict = None,
                    f_cyl: float = None,
                    cx_canvas: float = None,
                    cy_canvas: float = None,
                    smoothing: float = 0.5,
                    remap_scale: float = 0.5,
                    overlap_mask: np.ndarray = None):
    """Build a LiDAR-guided remap as rotation baseline + interpolated correction.

    Method
    ------
    1. Build rotation baseline remap (map_x_rot, map_y_rot).
    2. At each LiDAR control point, compute delta = true_cam_px − rotation_prediction.
    3. Fit a local RBF on (canvas_dst -> [Δx, Δy]).
    4. Apply correction ONLY inside `overlap_mask` (canvas pixels shared with an
       adjacent camera).  Everywhere else the rotation baseline is kept unchanged --
       correcting non-overlap regions would distort the image with no alignment benefit.

    Parameters
    ----------
    src_pts, dst_pts : LiDAR control points (camera px and canvas px)
    cam, f_cyl, cx_canvas, cy_canvas : needed for rotation baseline
    smoothing : RBFInterpolator smoothing (0 = exact interpolation)
    remap_scale : canvas sub-sampling fraction for RBF evaluation (upsampled afterwards)
    overlap_mask : bool (H_canvas, W_canvas) -- where to apply the correction.
                   None means apply inside convex hull of dst_pts (legacy behaviour).

    Returns
    -------
    map_x, map_y : (H_canvas, W_canvas) float32 arrays for cv2.remap
    """
    from scipy.interpolate import RBFInterpolator

    # -- Step 1: rotation baseline ------------------------------------------
    map_x_rot, map_y_rot = build_rotation_remap(
        cam, f_cyl, cx_canvas, cy_canvas, W_canvas, H_canvas)

    # -- Step 2: correction at each control point ---------------------------
    # Sample rotation remap at the control-point canvas positions (bilinear)
    cx_int = np.clip(dst_pts[:, 0], 0, W_canvas - 1.001)
    cy_int = np.clip(dst_pts[:, 1], 0, H_canvas - 1.001)
    cx0 = cx_int.astype(int);  cx1 = np.minimum(cx0 + 1, W_canvas - 1)
    cy0 = cy_int.astype(int);  cy1 = np.minimum(cy0 + 1, H_canvas - 1)
    fx  = (cx_int - cx0).astype(np.float32)
    fy  = (cy_int - cy0).astype(np.float32)

    def bilinear(field):
        return ((1-fy)*((1-fx)*field[cy0, cx0] + fx*field[cy0, cx1]) +
                   fy *((1-fx)*field[cy1, cx0] + fx*field[cy1, cx1]))

    rot_x_at_ctrl = bilinear(map_x_rot)
    rot_y_at_ctrl = bilinear(map_y_rot)

    delta_x = src_pts[:, 0] - rot_x_at_ctrl
    delta_y = src_pts[:, 1] - rot_y_at_ctrl

    # No outlier rejection: shared LiDAR points are the same physical 3D point
    # seen from both cameras, so large corrections (near objects, 30-100 px)
    # are geometrically valid and are exactly the signal we want to capture.
    deltas  = np.column_stack([delta_x, delta_y])

    # -- Step 3: fit local RBF on the correction field ---------------------
    # smoothing=0.5: small residual budget damps oscillation from noisy
    # near-range points while still producing ~10-80 px corrections.
    # kernel='thin_plate_spline': minimises bending energy -- the physically
    # correct smoothness prior for 2D warp fields.
    N = len(dst_pts)
    k = min(max(8, N // 8), 40)
    rbf = RBFInterpolator(dst_pts, deltas,
                          kernel='thin_plate_spline',
                          neighbors=k,
                          smoothing=smoothing)

    # -- Step 4: evaluate corrections on downsampled canvas grid -----------
    W_small = max(4, int(round(W_canvas * remap_scale)))
    H_small = max(4, int(round(H_canvas * remap_scale)))
    uu = np.linspace(0, W_canvas - 1, W_small, dtype=np.float32)
    vv = np.linspace(0, H_canvas - 1, H_small, dtype=np.float32)
    UU, VV = np.meshgrid(uu, vv)
    grid_pts = np.column_stack([UU.ravel(), VV.ravel()])

    corr = rbf(grid_pts)
    dx_small = corr[:, 0].reshape(H_small, W_small).astype(np.float32)
    dy_small = corr[:, 1].reshape(H_small, W_small).astype(np.float32)

    dx = cv2.resize(dx_small, (W_canvas, H_canvas), interpolation=cv2.INTER_CUBIC)
    dy = cv2.resize(dy_small, (W_canvas, H_canvas), interpolation=cv2.INTER_CUBIC)

    # -- Step 5: apply correction only where we have overlap evidence -------
    # overlap_mask is float [0,1]: 1.0 deep in overlap zone, fades to 0 at
    # the boundary so there is no hard discontinuity in the warp field.
    if overlap_mask is not None:
        dx = (dx * overlap_mask).astype(np.float32)
        dy = (dy * overlap_mask).astype(np.float32)

    map_x = (map_x_rot + dx).astype(np.float32)
    map_y = (map_y_rot + dy).astype(np.float32)

    # Where rotation baseline is invalid (outside camera FOV), keep as invalid
    rot_invalid = (map_x_rot < 0) | (map_y_rot < 0)
    map_x[rot_invalid] = -1.0
    map_y[rot_invalid] = -1.0

    return map_x, map_y


def build_locally_weighted_remap(
        src_pts: np.ndarray, dst_pts: np.ndarray,
        cam: dict, f_cyl: float,
        cx_canvas: float, cy_canvas: float,
        W_canvas: int, H_canvas: int,
        sigma: float = 150.0,
        overlap_mask: np.ndarray = None,
        remap_scale: float = 0.05,
        base_map_x: np.ndarray = None,
        base_map_y: np.ndarray = None):
    """Locally-weighted scattered correction on top of a baseline remap.

    At every canvas pixel the correction is a Gaussian-weighted average of the
    per-control-point delta vectors.  Support radius `sigma` (canvas pixels)
    keeps nearby corrections independent: ground rows get their parallax
    correction, distant buildings get near-zero, and they don't fight.

    Parameters
    ----------
    src_pts, dst_pts : LiDAR control points (camera px and canvas px)
    base_map_x/y     : baseline remap to correct on top of.
                       None -> use rotation baseline (mode='local').
                       Pass TPS maps for mode='tps_local'.
    sigma            : Gaussian support radius in canvas pixels.
    """
    # Rotation baseline (always needed as starting point)
    map_x_rot, map_y_rot = build_rotation_remap(
        cam, f_cyl, cx_canvas, cy_canvas, W_canvas, H_canvas)

    map_x_base = base_map_x if base_map_x is not None else map_x_rot
    map_y_base = base_map_y if base_map_y is not None else map_y_rot

    # -- Correction deltas at control points (base remap -> desired cam pixel) --
    cx_int = np.clip(dst_pts[:, 0], 0, W_canvas - 1.001)
    cy_int = np.clip(dst_pts[:, 1], 0, H_canvas - 1.001)
    cx0 = cx_int.astype(int);  cx1 = np.minimum(cx0 + 1, W_canvas - 1)
    cy0 = cy_int.astype(int);  cy1 = np.minimum(cy0 + 1, H_canvas - 1)
    fx  = (cx_int - cx0).astype(np.float32)
    fy  = (cy_int - cy0).astype(np.float32)

    def bilinear(field):
        return ((1-fy)*((1-fx)*field[cy0, cx0] + fx*field[cy0, cx1]) +
                   fy *((1-fx)*field[cy1, cx0] + fx*field[cy1, cx1]))

    base_x_at_ctrl = bilinear(map_x_base)
    base_y_at_ctrl = bilinear(map_y_base)

    delta_x = (src_pts[:, 0] - base_x_at_ctrl).astype(np.float32)  # (N,)
    delta_y = (src_pts[:, 1] - base_y_at_ctrl).astype(np.float32)  # (N,)

    # -- Evaluate Gaussian-weighted field on downsampled grid ------------------
    W_small = max(4, int(round(W_canvas * remap_scale)))
    H_small = max(4, int(round(H_canvas * remap_scale)))
    uu = np.linspace(0, W_canvas - 1, W_small, dtype=np.float32)
    vv = np.linspace(0, H_canvas - 1, H_small, dtype=np.float32)
    UU, VV = np.meshgrid(uu, vv)
    grid_pts = np.column_stack([UU.ravel(), VV.ravel()])  # (M, 2)

    # (M, N) Gaussian weights
    diff = grid_pts[:, np.newaxis, :] - dst_pts[np.newaxis, :, :]   # (M,N,2)
    dist2 = (diff ** 2).sum(axis=2)                                  # (M,N)
    W_mat = np.exp(-dist2 / (2.0 * sigma * sigma))                  # (M,N)
    W_sum = W_mat.sum(axis=1)                                        # (M,)

    # Pixels with negligible total weight keep zero correction
    valid = W_sum > 1e-6
    W_norm = np.where(valid[:, np.newaxis], W_mat / (W_sum[:, np.newaxis] + 1e-12), 0.0)

    dx_small = (W_norm @ delta_x).reshape(H_small, W_small).astype(np.float32)
    dy_small = (W_norm @ delta_y).reshape(H_small, W_small).astype(np.float32)

    dx = cv2.resize(dx_small, (W_canvas, H_canvas), interpolation=cv2.INTER_CUBIC)
    dy = cv2.resize(dy_small, (W_canvas, H_canvas), interpolation=cv2.INTER_CUBIC)

    if overlap_mask is not None:
        dx = (dx * overlap_mask).astype(np.float32)
        dy = (dy * overlap_mask).astype(np.float32)

    map_x = (map_x_base + dx).astype(np.float32)
    map_y = (map_y_base + dy).astype(np.float32)

    rot_invalid = (map_x_rot < 0) | (map_y_rot < 0)
    map_x[rot_invalid] = -1.0
    map_y[rot_invalid] = -1.0

    return map_x, map_y


def build_rotation_remap(cam: dict, f_cyl: float,
                         cx_canvas: float, cy_canvas: float,
                         W_canvas: int, H_canvas: int):
    """Rotation-only remap: for each canvas pixel, find the source camera pixel.

    Returns
    -------
    map_x, map_y : (H_canvas, W_canvas) float32 arrays for cv2.remap
                   Values outside the camera image are set to -1.
    """
    uu = np.arange(W_canvas, dtype=np.float64)
    vv = np.arange(H_canvas, dtype=np.float64)
    UU, VV = np.meshgrid(uu, vv)

    # Canvas pixel -> spherical angles
    # u = -f*az + cx  ->  az = (cx - u) / f  (negated convention)
    az = (cx_canvas - UU) / f_cyl
    el = (cy_canvas - VV) / f_cyl     # negated: small v (top) = high elevation

    # Spherical -> ego-frame unit direction
    cos_el = np.cos(el)
    px_ego = cos_el * np.cos(az)
    py_ego = cos_el * np.sin(az)
    pz_ego = np.sin(el)

    # Ego -> camera frame: R^T (R is cam->ego, R^T is ego->cam)
    R = cam['R']
    px_cam = R[0, 0]*px_ego + R[1, 0]*py_ego + R[2, 0]*pz_ego
    py_cam = R[0, 1]*px_ego + R[1, 1]*py_ego + R[2, 1]*pz_ego
    pz_cam = R[0, 2]*px_ego + R[1, 2]*py_ego + R[2, 2]*pz_ego

    # Pinhole projection
    valid = pz_cam > 0
    safe_z = np.where(valid, pz_cam, 1.0)
    map_x = np.where(valid, cam['fx'] * px_cam / safe_z + cam['cx'], -1.0)
    map_y = np.where(valid, cam['fy'] * py_cam / safe_z + cam['cy'], -1.0)

    return map_x.astype(np.float32), map_y.astype(np.float32)


# -- SeamDP --------------------------------------------------------------------

def find_seam_dp(warped_left: np.ndarray, warped_right: np.ndarray,
                 valid_left: np.ndarray, valid_right: np.ndarray,
                 grad_weight: float = 2.0) -> np.ndarray:
    """Find the optimal vertical seam between two warped canvas images using DP.

    Seam is constrained to the actual overlap zone (both cameras valid).  For
    rows outside the overlap (e.g. FC taller than FL/FR) the seam is
    interpolated from the nearest row that does have overlap, so it never
    snaps to the canvas edge.

    Cost = |ΔL| + grad_weight * (|∇L_left| + |∇L_right|)
    The gradient term penalises cutting through strong edges in either image,
    steering the seam through smooth / low-contrast regions.

    Returns
    -------
    seam : (H_canvas,) int32 array -- best seam column for each row.
    """
    H, W = warped_left.shape[:2]

    gray_l = cv2.cvtColor(warped_left,  cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_r = cv2.cvtColor(warped_right, cv2.COLOR_BGR2GRAY).astype(np.float32)

    overlap     = valid_left & valid_right
    has_overlap = overlap.any(axis=1)          # (H,) -- rows with mutual content

    # Colour-difference term
    color_cost = np.abs(gray_l - gray_r)

    # Gradient magnitude of each image (Sobel, normalised 0–1)
    sobel_l = (np.abs(cv2.Sobel(gray_l, cv2.CV_32F, 1, 0, ksize=3)) +
               np.abs(cv2.Sobel(gray_l, cv2.CV_32F, 0, 1, ksize=3))) / (4.0 * 255.0)
    sobel_r = (np.abs(cv2.Sobel(gray_r, cv2.CV_32F, 1, 0, ksize=3)) +
               np.abs(cv2.Sobel(gray_r, cv2.CV_32F, 0, 1, ksize=3))) / (4.0 * 255.0)
    grad_cost = grad_weight * (sobel_l + sobel_r) * 255.0   # back to ~[0,255] scale

    cost = color_cost + grad_cost
    cost[~overlap] = 1e9   # steer seam away from non-overlap

    # Forward DP pass (vectorised per row)
    dp = cost.copy()
    for row in range(1, H):
        prev   = dp[row - 1]
        prev_l = np.empty_like(prev); prev_l[0]  = prev[0];  prev_l[1:] = prev[:-1]
        prev_r = np.empty_like(prev); prev_r[-1] = prev[-1]; prev_r[:-1] = prev[1:]
        dp[row] += np.minimum(np.minimum(prev_l, prev), prev_r)

    # Backtrack
    seam = np.empty(H, dtype=np.int32)
    seam[-1] = int(dp[-1].argmin())
    for row in range(H - 2, -1, -1):
        c = seam[row + 1]
        c0 = max(0, c - 1);  c1 = min(W - 1, c + 1)
        seam[row] = c0 + int(dp[row, c0:c1 + 1].argmin())

    # Fix rows outside the overlap: propagate from nearest overlap row
    if not has_overlap.all():
        ov_idx = np.where(has_overlap)[0]
        if len(ov_idx) > 0:
            for row in np.where(~has_overlap)[0]:
                nearest = ov_idx[int(np.argmin(np.abs(ov_idx - row)))]
                seam[row] = seam[nearest]
        else:
            seam[:] = W // 2

    return seam


def blend_with_seam(left: np.ndarray, right: np.ndarray,
                    seam: np.ndarray, feather_half: int = 40,
                    valid_left: np.ndarray = None,
                    valid_right: np.ndarray = None) -> np.ndarray:
    """Blend two full-canvas images using a per-row DP seam + narrow feather.

    Pixels well left of seam come from `left`; right of seam from `right`;
    ±feather_half columns around the seam are blended linearly.

    valid_left / valid_right (optional): where only one camera has content the
    other camera's black pixels are not mixed in -- prevents darkening at the
    top/bottom rows where cameras have different vertical extents.
    """
    H, W = left.shape[:2]
    cols = np.arange(W, dtype=np.float32)
    s = seam[:, np.newaxis].astype(np.float32)   # (H, 1)

    if feather_half > 0:
        t = np.clip((s + feather_half - cols) / (2.0 * feather_half), 0.0, 1.0)
        alpha = t * t * (3.0 - 2.0 * t)   # smoothstep: zero derivative at both ends
    else:
        alpha = (cols <= s).astype(np.float32)

    a3 = alpha[:, :, np.newaxis]
    result = np.clip(
        a3         * left.astype(np.float32) +
        (1.0 - a3) * right.astype(np.float32),
        0, 255).astype(np.uint8)

    # Where only one camera has valid content, use it directly (no mixing)
    if valid_left is not None and valid_right is not None:
        only_right = valid_right & ~valid_left
        only_left  = valid_left  & ~valid_right
        result[only_right] = right[only_right]
        result[only_left]  = left[only_left]

    return result


# -- Canvas compositing --------------------------------------------------------

def paste_with_feather(canvas: np.ndarray, warped: np.ndarray,
                       valid: np.ndarray,
                       feather_half: int = 0) -> None:
    """Paste warped image onto canvas in-place with per-row feather blending.

    feather_half=0  : blend linearly across the full overlap per row (original).
    feather_half>0  : blend only in a ±feather_half band around the overlap
                      midpoint; outside that band keep existing canvas (left)
                      or use new image fully (right).  Eliminates ghosting from
                      wide geometric overlaps where one camera is effectively black.
    """
    has_canvas = canvas.max(axis=2) > 0
    has_new    = valid > 0

    overlap  = has_canvas & has_new
    only_new = (~has_canvas) & has_new

    if overlap.any():
        ov_rows = np.where(overlap.any(axis=1))[0]
        alpha   = np.zeros(canvas.shape[:2], np.float32)

        for row in ov_rows:
            cols = np.where(overlap[row])[0]
            c0, c1 = int(cols[0]), int(cols[-1]) + 1
            if feather_half > 0:
                mid = (c0 + c1) // 2
                f0  = max(c0, mid - feather_half)
                f1  = min(c1, mid + feather_half)
                # Left of feather: keep existing canvas fully (alpha=1)
                alpha[row, c0:f0] = 1.0
                # Feather zone: linear 1->0
                n = f1 - f0
                if n > 0:
                    alpha[row, f0:f1] = np.linspace(1.0, 0.0, n, dtype=np.float32)
                # Right of feather: use new image fully (alpha=0, already 0)
            else:
                alpha[row, c0:c1] = np.linspace(1.0, 0.0, c1 - c0, dtype=np.float32)

        a3 = alpha[:, :, np.newaxis]
        blended = (a3 * canvas.astype(np.float32) +
                   (1 - a3) * warped.astype(np.float32)).clip(0, 255).astype(np.uint8)
        canvas[overlap] = blended[overlap]

    if only_new.any():
        canvas[only_new] = warped[only_new]


# -- Debug overlay -------------------------------------------------------------

def depth_color(z: np.ndarray, lo: float = 0.5, hi: float = 60.0) -> np.ndarray:
    """Jet-like colormap: red = near, blue = far."""
    t = np.clip((z - lo) / (hi - lo), 0, 1)
    t = 1 - t
    r = np.clip(1.5 - np.abs(4*t - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4*t - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4*t - 1), 0, 1)
    return (np.stack([b, g, r], axis=1) * 255).astype(np.uint8)


# -- Main ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='LiDAR-guided geometric stitcher')
    ap.add_argument('--frame',       type=int, default=0,
                    help='Frame index in frames.json')
    ap.add_argument('--debug',       action='store_true',
                    help='Overlay LiDAR ring dots on canvas (depth-coloured)')
    ap.add_argument('--method',      default='tps',
                    choices=['rot', 'tps', 'local', 'tps_local'],
                    help='Warp method: rot=rotation-only, tps=global TPS, '
                         'local=locally-weighted only, tps_local=TPS then local residual')
    ap.add_argument('--local-sigma',  type=float, default=LOCAL_SIGMA,
                    help='Gaussian support radius in canvas px for local methods')
    ap.add_argument('--feather-half', type=int,   default=FEATHER_HALF,
                    help='Narrow feather half-width in px (0=full overlap blend)')
    ap.add_argument('--seam-dp',      action='store_true', default=True,
                    help='Use DP seam finder for blending (overrides feather midpoint)')
    ap.add_argument('--sensor-root', default=str(SENSOR_ROOT))
    ap.add_argument('--calib',       default=str(CALIB_JSON))
    ap.add_argument('--frames',      default=str(FRAMES_JSON))
    ap.add_argument('--out',         default=str(OUT_DIR))
    ap.add_argument('--dump-warps',  action='store_true',
                    help='Also save per-camera warped canvases and valid masks')
    ap.add_argument('--real-overlap-mask', action='store_true', default=True,
                    help='Use actual pixel-overlap mask instead of seam strip')
    ap.add_argument('--min-ctrl-range',   type=float, default=LIDAR_MIN_CTRL_RANGE_M,
                    help='Min LiDAR range (m) for TPS control points (default: %(default)s)')
    ap.add_argument('--tps-smoothing',    type=float, default=TPS_SMOOTHING,
                    help='RBF smoothing factor (0=exact interp, higher=smoother, default: %(default)s)')
    args = ap.parse_args()
    # Compatibility shim: the deployed pipeline always holds FC at its
    # rotation baseline and warps FL/FR. The historical --symmetric /
    # --mid-alpha flags were removed when the symmetric path was deleted
    # from the C++ pipeline; this prototype now always runs asymmetric.
    args.symmetric = False
    args.mid_alpha = 1.0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sensor_root = Path(args.sensor_root)

    # -- Load calibration ------------------------------------------------------
    cams = load_calib(args.calib)
    print(f'Calibration loaded: {len(cams)} cameras')

    # -- Load frame record -----------------------------------------------------
    with open(args.frames) as f:
        frames = json.load(f)
    frame = frames[args.frame]
    print(f'Frame {args.frame}: {list(frame.keys())}')

    # -- Index and match LiDAR sweep -------------------------------------------
    lidar_files: dict[int, Path] = {}
    for fp in sensor_root.glob('train/*/sensors/lidar/*.feather'):
        try:
            lidar_files[int(fp.stem)] = fp
        except ValueError:
            pass
    if not lidar_files:
        raise RuntimeError(f'No LiDAR feather files found under {sensor_root}')
    lidar_ts = np.array(sorted(lidar_files.keys()), dtype=np.int64)
    print(f'LiDAR sweeps indexed: {len(lidar_ts)}')

    # Use ring_front_center timestamp as reference
    ref_cam = 'ring_front_center'
    img_ts  = int(Path(frame[ref_cam]).stem)
    idx     = int(np.argmin(np.abs(lidar_ts - img_ts)))
    best_ts = int(lidar_ts[idx])
    dt_ms   = abs(best_ts - img_ts) / 1e6
    print(f'Image ts={img_ts}  LiDAR ts={best_ts}  dt={dt_ms:.1f} ms')

    df  = pd.read_feather(lidar_files[best_ts])
    pts = df[['x', 'y', 'z']].values.astype(np.float32)
    print(f'LiDAR points: {len(pts)}  z∈[{pts[:,2].min():.1f}, {pts[:,2].max():.1f}]')

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

    # -- Canvas setup ----------------------------------------------------------
    # Cylindrical focal length = FC's fx
    f_cyl = float(cams[ref_cam]['fx'])

    # Reference height: mean camera z in ego frame.
    # Elevation is measured from this height so that the rotation baseline
    # (camera rays) and ego_to_canvas agree, keeping TPS corrections small.
    Z_REF = float(np.mean([cams[n]['t'][2] for n in FRONT_CAMS]))
    print(f'Camera height Z_REF = {Z_REF:.3f} m')

    # Collect azimuths/elevations for all LiDAR pts visible in any front camera
    all_az: list[np.ndarray] = []
    all_el: list[np.ndarray] = []
    for name in FRONT_CAMS:
        _, pts_v = project_with_ego(pts, cams[name])
        if len(pts_v) == 0:
            continue
        az = np.arctan2(pts_v[:, 1], pts_v[:, 0])
        r_xy = np.sqrt(pts_v[:, 0]**2 + pts_v[:, 1]**2)
        el = np.arctan2(pts_v[:, 2] - Z_REF, r_xy)   # from camera height
        all_az.append(az)
        all_el.append(el)

    if not all_az:
        raise RuntimeError('No LiDAR points visible in any front camera')

    az_all = np.concatenate(all_az)
    el_all = np.concatenate(all_el)
    az_min, az_max = float(az_all.min()), float(az_all.max())
    el_min, el_max = float(el_all.min()), float(el_all.max())

    # Add margin
    az_mg = CANVAS_MARGIN_FRAC * (az_max - az_min)
    el_mg = CANVAS_MARGIN_FRAC * (el_max - el_min)
    az_min -= az_mg;  az_max += az_mg
    el_min -= el_mg;  el_max += el_mg

    # Canvas size and origin offsets
    W_canvas = max(1, int(np.ceil(f_cyl * (az_max - az_min))))
    H_canvas = max(1, int(np.ceil(f_cyl * (el_max - el_min))))
    cx_canvas = float(az_max * f_cyl)    # az_max (FL, leftmost) -> u=0 (left edge)
    cy_canvas = float(el_max * f_cyl)    # el_max  -> v=0  (top = sky)

    print(f'Canvas: {W_canvas}x{H_canvas}  '
          f'az=[{np.degrees(az_min):.1f}°, {np.degrees(az_max):.1f}°]  '
          f'el=[{np.degrees(el_min):.1f}°, {np.degrees(el_max):.1f}°]  '
          f'f_cyl={f_cyl:.0f}')

    # -- Pre-compute rotation remaps + overlap masks ---------------------------
    # Overlap mask for each camera = canvas pixels where this camera AND its
    # neighbour(s) both have valid rotation coverage.  TPS correction is only
    # applied inside this mask; everywhere else the rotation baseline is kept.
    print('\nPre-computing rotation remaps…')
    rot_remaps: dict[str, tuple] = {}
    rot_valid:  dict[str, np.ndarray] = {}
    for name in FRONT_CAMS:
        cam = cams[name]
        mx, my = build_rotation_remap(cam, f_cyl, cx_canvas, cy_canvas,
                                      W_canvas, H_canvas)
        rot_remaps[name] = (mx, my)
        rot_valid[name]  = ((mx >= 0) & (my >= 0) &
                            (mx < cam['W']) & (my < cam['H']))

    FL, FC, FR = FRONT_CAMS

    # -- Seam columns: centre of the rotation-overlap strip --------------------
    # The TPS correction is ONLY applied in a narrow ±SEAM_HALF_WIDTH_PX strip
    # around each seam.  The earlier approach (full rotation-overlap mask) was
    # too wide: seam-zone corrections bled into camera interiors and distorted
    # the whole image due to the lens-distortion component of the LiDAR delta.
    ov_FL_FC = rot_valid[FL] & rot_valid[FC]
    ov_FC_FR = rot_valid[FC] & rot_valid[FR]
    seam_FL_FC = (int(np.where(ov_FL_FC.any(axis=0))[0].mean())
                  if ov_FL_FC.any() else W_canvas // 3)
    seam_FC_FR = (int(np.where(ov_FC_FR.any(axis=0))[0].mean())
                  if ov_FC_FR.any() else 2 * W_canvas // 3)
    print(f'  Seam columns: FL<->FC={seam_FL_FC}  FC<->FR={seam_FC_FR}')

    def seam_strip_mask(seam_col: int, half_width: int = SEAM_HALF_WIDTH_PX
                        ) -> np.ndarray:
        """Float [0,1] mask -- 1.0 at seam_col, linear falloff to 0 at ±half_width."""
        cols   = np.arange(W_canvas, dtype=np.float32)
        prof   = np.maximum(0.0, 1.0 - np.abs(cols - seam_col) / half_width)
        return np.tile(prof[np.newaxis, :], (H_canvas, 1)).astype(np.float32)

    strip_FL_FC = seam_strip_mask(seam_FL_FC)
    strip_FC_FR = seam_strip_mask(seam_FC_FR)

    if args.real_overlap_mask:
        # Fix 1: mask built from actual pixels seen by both cameras, blurred for
        # smooth falloff.  Replaces the wide ±SEAM_HALF_WIDTH_PX strip which
        # covers non-overlap regions and causes extrapolation artifacts.
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
    for name in FRONT_CAMS:
        n_ov = int((overlap_masks[name] > 0).sum())
        print(f'  {name}: correction strip = {n_ov:,} canvas px')

    # -- Project LiDAR into all cameras ONCE -----------------------------------
    # Each point is projected once per camera; shared-point extraction is then
    # a cheap index set-intersection -- no redundant reprojection.
    print('\nProjecting LiDAR into cameras…')
    proj: dict[str, tuple] = {}   # name -> (cam_px, ego_pts, orig_indices)
    for name in FRONT_CAMS:
        px, ego, idx = project_with_ego(pts, cams[name], return_indices=True)
        proj[name] = (px, ego, idx)
        print(f'  {name}: {len(px)} hits')

    # -- Shared control points between adjacent camera pairs -------------------
    # Canvas targets use FC's rotation model so FL/FR land exactly where FC
    # projects the same 3-D point -- eliminating parallax-induced ghosting.
    print('Finding shared control points…')
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

    # Symmetric "meet-in-middle" mode: blend canvas targets toward midpoint
    # between FC's and the side camera's own rotation projections.
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
        # FC stays rotation-only (reference camera) -- no ctrl pts needed for it.
        shared_src = {FL: sh_px_FL, FC: np.empty((0, 2), np.float32), FR: sh_px_FR}
        shared_dst = {FL: sh_cvs_FL_FC, FC: np.empty((0, 2), np.float32), FR: sh_cvs_FC_FR}

    for name in FRONT_CAMS:
        n_sh = len(shared_src[name])
        print(f'  {name}: {n_sh} shared ctrl pts in overlap zone')

    # -- Per-camera warp -------------------------------------------------------
    warped_imgs: dict[str, np.ndarray] = {}
    valid_masks: dict[str, np.ndarray] = {}
    last_maps: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # For debug: canvas-space LiDAR pts with depth per camera
    debug_hits: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for name in FRONT_CAMS:
        cam = cams[name]
        W_cam, H_cam = cam['W'], cam['H']
        img = images[name]
        print(f'\n  [{name}]')

        if args.method == 'rot' or (name == FC and not args.symmetric):
            map_x, map_y = rot_remaps[name]
            if name == FC and args.method != 'rot':
                print(f'    FC: rotation-only (reference camera)')
        else:
            src_pts = shared_src[name]
            dst_pts = shared_dst[name]
            print(f'    Control pts: {len(src_pts)} shared seam pts  method={args.method}')

            if args.debug:
                _, pts_dbg, _ = proj[name]
                if len(pts_dbg) > 0:
                    canvas_px_dbg = ego_to_canvas(
                        pts_dbg, f_cyl, cx_canvas, cy_canvas, Z_REF)
                    z_vals = np.linalg.norm(pts_dbg, axis=1)
                    debug_hits[name] = (canvas_px_dbg, z_vals)

            if len(src_pts) < 4:
                print(f'    < 4 LiDAR pts -- falling back to rotation warp')
                map_x, map_y = rot_remaps[name]
            elif args.method == 'tps':
                print(f'    Fitting TPS remap…')
                map_x, map_y = build_tps_remap(
                    src_pts, dst_pts, W_cam, H_cam, W_canvas, H_canvas,
                    cam=cam, f_cyl=f_cyl, cx_canvas=cx_canvas, cy_canvas=cy_canvas,
                    smoothing=args.tps_smoothing, remap_scale=0.5,
                    overlap_mask=overlap_masks[name])
            elif args.method == 'local':
                print(f'    Fitting local-weighted remap (σ={args.local_sigma:.0f}px)…')
                map_x, map_y = build_locally_weighted_remap(
                    src_pts, dst_pts,
                    cam=cam, f_cyl=f_cyl, cx_canvas=cx_canvas, cy_canvas=cy_canvas,
                    W_canvas=W_canvas, H_canvas=H_canvas,
                    sigma=args.local_sigma, overlap_mask=overlap_masks[name])
            elif args.method == 'tps_local':
                print(f'    Fitting TPS remap then local residual (σ={args.local_sigma:.0f}px)…')
                mx_tps, my_tps = build_tps_remap(
                    src_pts, dst_pts, W_cam, H_cam, W_canvas, H_canvas,
                    cam=cam, f_cyl=f_cyl, cx_canvas=cx_canvas, cy_canvas=cy_canvas,
                    smoothing=args.tps_smoothing, remap_scale=0.5,
                    overlap_mask=overlap_masks[name])
                map_x, map_y = build_locally_weighted_remap(
                    src_pts, dst_pts,
                    cam=cam, f_cyl=f_cyl, cx_canvas=cx_canvas, cy_canvas=cy_canvas,
                    W_canvas=W_canvas, H_canvas=H_canvas,
                    sigma=args.local_sigma, overlap_mask=overlap_masks[name],
                    base_map_x=mx_tps, base_map_y=my_tps)

        # -- Remap ---------------------------------------------------------
        warped = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        valid_mask = ((map_x >= 0) & (map_y >= 0) &
                      (map_x < W_cam) & (map_y < H_cam))
        print(f'    Valid canvas pixels: {valid_mask.sum():,}')
        warped_imgs[name]  = warped
        valid_masks[name]  = valid_mask
        last_maps[name]    = (map_x, map_y)

    # -- Composite -------------------------------------------------------------
    canvas = np.zeros((H_canvas, W_canvas, 3), dtype=np.uint8)

    if args.seam_dp:
        print('\nCompositing with SeamDP…')
        # FL only -- no overlap yet
        paste_with_feather(canvas, warped_imgs[FL], valid_masks[FL], feather_half=0)
        # FL <-> FC seam
        seam_fl_fc = find_seam_dp(warped_imgs[FL], warped_imgs[FC],
                                   valid_masks[FL], valid_masks[FC])
        canvas = blend_with_seam(canvas, warped_imgs[FC],
                                  seam_fl_fc, feather_half=args.feather_half,
                                  valid_left=valid_masks[FL],
                                  valid_right=valid_masks[FC])
        # FC <-> FR seam
        seam_fc_fr = find_seam_dp(warped_imgs[FC], warped_imgs[FR],
                                   valid_masks[FC], valid_masks[FR])
        canvas = blend_with_seam(canvas, warped_imgs[FR],
                                  seam_fc_fr, feather_half=args.feather_half,
                                  valid_left=valid_masks[FC],
                                  valid_right=valid_masks[FR])
        print(f'  FL<->FC seam col range: [{seam_fl_fc.min()}, {seam_fl_fc.max()}]')
        print(f'  FC<->FR seam col range: [{seam_fc_fr.min()}, {seam_fc_fr.max()}]')
    else:
        for name in FRONT_CAMS:
            paste_with_feather(canvas, warped_imgs[name], valid_masks[name],
                               feather_half=args.feather_half)

    # -- Debug overlay ---------------------------------------------------------
    if args.debug and debug_hits:
        debug_canvas = canvas.copy()
        for name, (cvs_pts, z_vals) in debug_hits.items():
            step = max(1, len(cvs_pts) // 3000)
            pts_show = cvs_pts[::step]
            z_show   = z_vals[::step]
            colors   = depth_color(z_show)
            for (u, v), c in zip(pts_show, colors):
                px, py = int(round(float(u))), int(round(float(v)))
                if 0 <= px < W_canvas and 0 <= py < H_canvas:
                    cv2.circle(debug_canvas, (px, py), 2,
                               (int(c[0]), int(c[1]), int(c[2])), -1)
        dbg_path = out_dir / f'frame_{args.frame:04d}_debug.png'
        cv2.imwrite(str(dbg_path), debug_canvas)
        print(f'\nDebug saved -> {dbg_path}')

    # -- Save main output ------------------------------------------------------
    suffix    = '' if args.method == 'tps' else f'_{args.method}'
    out_path  = out_dir / f'frame_{args.frame:04d}{suffix}.png'
    cv2.imwrite(str(out_path), canvas)
    print(f'\nSaved -> {out_path}  ({canvas.shape[1]}x{canvas.shape[0]})')

    # -- Dump per-camera warped canvases + valid masks (for downstream analyses)
    if args.dump_warps:
        for name in FRONT_CAMS:
            cv2.imwrite(str(out_dir / f'frame_{args.frame:04d}{suffix}_warp_{name}.png'),
                        warped_imgs[name])
            cv2.imwrite(str(out_dir / f'frame_{args.frame:04d}{suffix}_mask_{name}.png'),
                        (valid_masks[name].astype(np.uint8)) * 255)
        # Also save the source camera images and inverse maps (canvas -> source)
        for name in FRONT_CAMS:
            cv2.imwrite(str(out_dir / f'frame_{args.frame:04d}_source_{name}.png'),
                        images[name])
        np.savez_compressed(
            str(out_dir / f'frame_{args.frame:04d}{suffix}_maps.npz'),
            **{f'{name}_map_x': last_maps[name][0] for name in FRONT_CAMS},
            **{f'{name}_map_y': last_maps[name][1] for name in FRONT_CAMS},
        )
        print(f'Dumped per-camera warps, masks, sources and maps to {out_dir}')


if __name__ == '__main__':
    main()
