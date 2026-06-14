"""Cylindrical projection utilities for AV2 camera-pair stitching.

For adjacent ring cameras (~60° angular separation), planar rotation
homographies create extreme triangular distortion.  Cylindrical projection
eliminates this: both images become horizontal strips differing only by a
small horizontal shift.  UDIS++ then corrects residual misalignment.

Public API
----------
build_cylindrical_remap(cam, f_cyl=None)
    -> (map_x, map_y)  float32 arrays (H, W) for cv2.remap
compute_cylindrical_overlap(cam_l, cam_r, f_cyl=None)
    -> (l_crop, r_crop) | (None, None)   (u0,u1,v0,v1) in cylindrical pixels
project_to_cylinder(img_bgr, map_x, map_y)
    -> img_cyl  BGR ndarray
cylindrical_canvas_offset(cam_l, cam_r, f_cyl=None)
    -> int   x_offset to place right cylindrical image in left camera's panorama
"""

from __future__ import annotations
import numpy as np


def build_cylindrical_remap(cam: dict, f_cyl: float | None = None):
    """Precompute cv2.remap maps: cylindrical pixels -> source pixels.

    For each cylindrical pixel (u_cyl, v_cyl):
        theta = (u_cyl - cx) / f_cyl      # horizontal angle
        h     = (v_cyl - cy) / f_cyl      # normalised height on cylinder
        u_src = fx * tan(theta) + cx
        v_src = fy * h / cos(theta) + cy

    Args:
        cam:   camera dict with keys fx, fy, cx, cy, W, H
        f_cyl: cylindrical focal length (defaults to cam['fx'])

    Returns:
        map_x, map_y: float32 arrays of shape (H, W).
        Out-of-bounds pixels get -1 (cv2.remap fills them with border value 0).
    """
    if f_cyl is None:
        f_cyl = cam['fx']

    H, W   = cam['H'], cam['W']
    cx, cy = cam['cx'], cam['cy']
    fx, fy = cam['fx'], cam['fy']

    u_cyl = np.arange(W, dtype=np.float32)
    v_cyl = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u_cyl, v_cyl)   # (H, W)

    theta = (uu - cx) / f_cyl            # horizontal angle (rad)
    h_cyl = (vv - cy) / f_cyl            # normalised height on cylinder

    cos_t = np.cos(theta)
    u_src = fx * np.tan(theta) + cx
    v_src = fy * h_cyl / cos_t + cy      # h / cos(theta) = h * sec(theta)

    valid = (u_src >= 0) & (u_src < W) & (v_src >= 0) & (v_src < H)
    u_src = np.where(valid, u_src, -1.0).astype(np.float32)
    v_src = np.where(valid, v_src, -1.0).astype(np.float32)

    return u_src, v_src   # map_x, map_y


def _cam_yaw(cam: dict) -> float:
    """Yaw of the camera's forward direction in the ego XY plane (radians).

    Camera +Z in ego frame = third column of the cam->ego rotation matrix R.
    """
    fwd_ego = cam['R'][:, 2]   # (3,)
    return float(np.arctan2(fwd_ego[1], fwd_ego[0]))


def compute_cylindrical_overlap(
    cam_l: dict,
    cam_r: dict,
    f_cyl: float | None = None,
    n_samples: int = 200,
):
    """Compute overlap crop bounds in cylindrical pixel space.

    Uses ray-tracing to find the planar overlap (correct for all camera
    orientations including side / rear cameras whose image-x axis does NOT
    align with the increasing-yaw direction in the ego frame), then converts
    the planar u bounds to cylindrical u via the inverse cylindrical formula:

        u_cyl = f_cyl * arctan((u_src - cx) / fx) + cx

    Returns (l_crop, r_crop) where each crop is (u0, u1, v0, v1) in
    **cylindrical pixel coordinates**.  Returns (None, None) if no overlap.

    Args:
        cam_l, cam_r: camera dicts (keys: R, fx, fy, cx, cy, W, H)
        f_cyl:        cylindrical focal length (defaults to cam_l['fx'])
        n_samples:    ray-tracing grid density (passed to compute_overlap)
    """
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from compute_overlaps import compute_overlap

    if f_cyl is None:
        f_cyl = cam_l['fx']

    # Planar overlap: ray-tracing guarantees correctness for any camera orientation
    l_crop_p, r_crop_p = compute_overlap(cam_l, cam_r, n_samples)
    if l_crop_p is None:
        return None, None

    lu0_p, lu1_p, lv0_p, lv1_p = l_crop_p
    ru0_p, ru1_p, rv0_p, rv1_p = r_crop_p

    def src_u_to_cyl_u(u_src: float, cam: dict) -> float:
        """Inverse cylindrical remap: source pixel u -> cylindrical u."""
        alpha = np.arctan2(u_src - cam['cx'], cam['fx'])   # camera-frame angle
        return f_cyl * alpha + cam['cx']

    # Convert left-camera planar u range -> cylindrical u range
    lu0_c = int(np.floor(min(src_u_to_cyl_u(lu0_p, cam_l),
                             src_u_to_cyl_u(lu1_p, cam_l))))
    lu1_c = int(np.ceil( max(src_u_to_cyl_u(lu0_p, cam_l),
                             src_u_to_cyl_u(lu1_p, cam_l))))
    lu0_c = max(0, lu0_c)
    lu1_c = min(cam_l['W'], lu1_c)

    # Convert right-camera planar u range -> cylindrical u range
    ru0_c = int(np.floor(min(src_u_to_cyl_u(ru0_p, cam_r),
                             src_u_to_cyl_u(ru1_p, cam_r))))
    ru1_c = int(np.ceil( max(src_u_to_cyl_u(ru0_p, cam_r),
                             src_u_to_cyl_u(ru1_p, cam_r))))
    ru0_c = max(0, ru0_c)
    ru1_c = min(cam_r['W'], ru1_c)

    if lu0_c >= lu1_c or ru0_c >= ru1_c:
        return None, None

    l_crop = (lu0_c, lu1_c, 0, cam_l['H'])
    r_crop = (ru0_c, ru1_c, 0, cam_r['H'])
    return l_crop, r_crop


def project_to_cylinder(img_bgr, map_x: np.ndarray, map_y: np.ndarray):
    """Apply cylindrical remap to a BGR image.

    Out-of-bounds pixels are filled with black (border_value=0).
    """
    import cv2
    return cv2.remap(img_bgr, map_x, map_y,
                     cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT,
                     borderValue=0)


def cylindrical_canvas_offset(cam_l: dict, cam_r: dict,
                               f_cyl: float | None = None) -> int:
    """X offset to place the right cylindrical image in the left camera's panorama.

    When building a two-camera cylindrical panorama with cam_l at x=0:
        pano[:, offset : offset + cam_r['W']] <- cyl_r

    The formula is: offset = f_cyl * (yaw_r - yaw_l) + cx_l - cx_r
    which is derived from requiring the same world angle to map to the same
    panorama x in both cylindrical images.
    """
    if f_cyl is None:
        f_cyl = cam_l['fx']
    yaw_l = _cam_yaw(cam_l)
    yaw_r = _cam_yaw(cam_r)
    # Normalise yaw difference (handle rear-camera wraparound)
    diff = yaw_r - yaw_l
    if diff >  np.pi:
        diff -= 2 * np.pi
    elif diff < -np.pi:
        diff += 2 * np.pi
    return int(round(f_cyl * diff + cam_l['cx'] - cam_r['cx']))
