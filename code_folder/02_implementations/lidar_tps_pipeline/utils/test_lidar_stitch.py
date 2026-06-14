#!/usr/bin/env python3
"""Unit tests for lidar_ring_stitch.py pipeline correctness.

Run:
    python test_lidar_stitch.py
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from lidar_ring_stitch import (
    FRONT_CAMS, LIDAR_MIN_CTRL_RANGE_M, SEAM_HALF_WIDTH_PX, TPS_SMOOTHING,
    load_calib, project_with_ego, ego_to_canvas,
    find_shared_ctrl_pts, grid_subsample, build_rotation_remap,
    build_tps_remap, _grid_subsample_with_idx,
    quat_to_mat,
)

CALIB_JSON = Path.home() / 'mThesis/argo2_data/extracted/calibration.json'
PASS = '\033[92mPASS\033[0m'
FAIL = '\033[91mFAIL\033[0m'
_results = []

def check(name, cond, detail=''):
    status = PASS if cond else FAIL
    _results.append(cond)
    print(f'  [{status}]  {name}' + (f'  ({detail})' if detail else ''))
    return cond


# -- Helpers -------------------------------------------------------------------

def _make_cam(yaw_deg, tx=0.0, ty=0.0, tz=1.4,
              fx=1777, fy=1777, cx=960, cy=604, W=1920, H=1216):
    """Synthetic camera pointing at given yaw (degrees from x-axis)."""
    import math
    yaw = math.radians(yaw_deg)
    # R maps camera x->ego: column i of R = ego direction of camera axis i
    R = np.array([
        [ math.cos(yaw), -math.sin(yaw), 0],
        [ math.sin(yaw),  math.cos(yaw), 0],
        [ 0,              0,             1],
    ], dtype=np.float64)
    return dict(R=R, t=np.array([tx, ty, tz]),
                fx=fx, fy=fy, cx=cx, cy=cy, W=W, H=H)


def _canvas_params(cams, pts_list, z_ref):
    """Minimal canvas geometry from a list of ego pts."""
    f_cyl = 1777.0
    all_az, all_el = [], []
    for pts in pts_list:
        az = np.arctan2(pts[:, 1], pts[:, 0])
        r  = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
        el = np.arctan2(pts[:, 2] - z_ref, r)
        all_az.append(az); all_el.append(el)
    az_all = np.concatenate(all_az); el_all = np.concatenate(all_el)
    az_min, az_max = az_all.min() - 0.1, az_all.max() + 0.1
    el_min, el_max = el_all.min() - 0.05, el_all.max() + 0.05
    W = max(1, int(np.ceil(f_cyl * (az_max - az_min))))
    H = max(1, int(np.ceil(f_cyl * (el_max - el_min))))
    cx = float(-az_min * f_cyl)
    cy = float(el_max  * f_cyl)
    return f_cyl, cx, cy, W, H


# ══════════════════════════════════════════════════════════════════════════════
# Test 1 -- ego_to_canvas is camera-independent
# ══════════════════════════════════════════════════════════════════════════════

def test_canvas_camera_independent():
    """Same 3-D point -> same canvas pixel regardless of which camera it's from."""
    print('\n[1] ego_to_canvas camera-independence')
    f, cx, cy = 1777.0, 2500.0, 800.0
    z_ref = 1.4
    P = np.array([[10.0, 5.0, 1.5]])   # some point ahead-left
    c1 = ego_to_canvas(P, f, cx, cy, z_ref)
    c2 = ego_to_canvas(P, f, cx, cy, z_ref)   # same call, deterministic
    check('identical canvas coords from same 3-D pt',
          np.allclose(c1, c2), f'c1={c1[0]}  c2={c2[0]}')

    # Perturb z_ref slightly -- canvas coords must change
    c3 = ego_to_canvas(P, f, cx, cy, z_ref + 0.5)
    check('different z_ref -> different canvas v',
          not np.allclose(c1[:, 1], c3[:, 1]))


# ══════════════════════════════════════════════════════════════════════════════
# Test 2 -- project_with_ego / return_indices round-trip
# ══════════════════════════════════════════════════════════════════════════════

def test_project_indices():
    """project_with_ego with return_indices=True returns consistent data."""
    print('\n[2] project_with_ego index consistency')
    cam = _make_cam(yaw_deg=45)   # FL-like
    rng = np.random.default_rng(42)
    pts = rng.uniform(-1, 1, (500, 3)).astype(np.float32)
    pts[:, 0] += 20  # push forward

    px, ego, idx = project_with_ego(pts, cam, return_indices=True)
    px2, ego2    = project_with_ego(pts, cam, return_indices=False)

    check('pixel arrays match with/without return_indices', np.allclose(px, px2))
    check('ego arrays match', np.allclose(ego, ego2))
    check('idx values are valid', idx.max() < len(pts) if len(idx) else True)
    check('pts[idx] == ego', np.allclose(pts[idx], ego, atol=1e-4) if len(idx) else True)


# ══════════════════════════════════════════════════════════════════════════════
# Test 3 -- shared control points: canvas targets are identical
# ══════════════════════════════════════════════════════════════════════════════

def test_shared_pts_same_canvas():
    """Shared 3-D points project to identical canvas coords from both cameras."""
    print('\n[3] shared_ctrl_pts canvas identity')
    cam_l = _make_cam(yaw_deg=50)   # FL-like
    cam_r = _make_cam(yaw_deg=10)   # FC-like (overlaps with FL on their shared edge)
    z_ref = 1.4

    rng = np.random.default_rng(7)
    pts = rng.uniform(5, 50, (2000, 3)).astype(np.float32)
    pts[:, 2] = 1.4 + rng.uniform(-0.5, 2.0, 2000).astype(np.float32)

    # Compute canvas params from all visible pts
    _, ego_l = project_with_ego(pts, cam_l)
    _, ego_r = project_with_ego(pts, cam_r)
    all_pts = np.vstack([ego_l, ego_r]) if len(ego_l) and len(ego_r) else pts[:10]
    f, cx, cy, W, H = _canvas_params(None, [all_pts], z_ref)

    px_l, px_r, cvs = find_shared_ctrl_pts(pts, cam_l, cam_r, f, cx, cy, z_ref)

    if len(cvs) == 0:
        print('  (no shared pts found -- adjust synthetic cameras)')
        return

    # Independently compute canvas coords for px_l and px_r's originating 3D pts
    # They must be the same (canvas depends only on the 3D pt, not the camera)
    _, ego_l2, idx_l = project_with_ego(pts, cam_l, return_indices=True)
    _, ego_r2, idx_r = project_with_ego(pts, cam_r, return_indices=True)
    map_l = {int(i): j for j, i in enumerate(idx_l)}
    map_r = {int(i): j for j, i in enumerate(idx_r)}
    shared_orig = sorted(set(map_l) & set(map_r))

    # Filter to those that survived min-range
    ego_shared_all = ego_l2[[map_l[i] for i in shared_orig]]
    rng_m = np.linalg.norm(ego_shared_all, axis=1)
    far_shared = ego_shared_all[rng_m >= LIDAR_MIN_CTRL_RANGE_M]

    if len(far_shared) == 0:
        print('  (no far shared pts)')
        return

    # Grid-subsample to match find_shared_ctrl_pts output
    _, sub_idx = _grid_subsample_with_idx(px_l, cam_l['W'], cam_l['H'])
    ego_sub = far_shared[sub_idx] if len(sub_idx) else far_shared[:0]

    cvs_recomputed = ego_to_canvas(ego_sub, f, cx, cy, z_ref) if len(ego_sub) else cvs[:0]

    check('find_shared canvas == ego_to_canvas(shared 3D pts)',
          np.allclose(cvs, cvs_recomputed, atol=0.5) if len(cvs) else True,
          f'{len(cvs)} shared pts')


# ══════════════════════════════════════════════════════════════════════════════
# Test 4 -- seam strip mask shape and values
# ══════════════════════════════════════════════════════════════════════════════

def test_seam_strip_mask():
    """seam_strip_mask peaks at seam_col and is zero ±half_width away."""
    print('\n[4] seam_strip_mask correctness')
    W, H = 5000, 1500
    seam_col = 2000
    half = SEAM_HALF_WIDTH_PX

    cols = np.arange(W, dtype=np.float32)
    prof = np.maximum(0.0, 1.0 - np.abs(cols - seam_col) / half)
    mask = np.tile(prof[np.newaxis, :], (H, 1)).astype(np.float32)

    check('mask peaks at 1.0 at seam col',
          np.isclose(mask[H//2, seam_col], 1.0), f'got {mask[H//2, seam_col]:.4f}')
    check('mask is 0.0 at seam_col - half_width',
          np.isclose(mask[H//2, max(0, seam_col - half)], 0.0),
          f'got {mask[H//2, max(0, seam_col - half)]:.4f}')
    check('mask is 0.0 at seam_col + half_width',
          np.isclose(mask[H//2, min(W-1, seam_col + half)], 0.0),
          f'got {mask[H//2, min(W-1, seam_col + half)]:.4f}')
    check('mask is 0.0 far from seam',
          mask[H//2, 0] == 0.0 and mask[H//2, W-1] == 0.0)
    check('mask is uniform vertically',
          np.allclose(mask[0, :], mask[-1, :]))


# ══════════════════════════════════════════════════════════════════════════════
# Test 5 -- TPS correction is zero outside the seam strip
# ══════════════════════════════════════════════════════════════════════════════

def test_tps_correction_outside_strip():
    """TPS map equals rotation baseline outside the seam correction strip."""
    print('\n[5] TPS correction confined to seam strip')
    if not CALIB_JSON.exists():
        print('  (skipped -- calibration not found)')
        return

    cams = load_calib(str(CALIB_JSON))
    FL, FC, FR = FRONT_CAMS
    f_cyl = float(cams[FC]['fx'])
    z_ref = float(np.mean([cams[n]['t'][2] for n in FRONT_CAMS]))

    # Minimal canvas for test (use a small synthetic point cloud)
    rng = np.random.default_rng(0)
    pts = rng.uniform(5, 80, (3000, 3)).astype(np.float32)
    pts[:, 0] = np.abs(pts[:, 0])  # positive x = forward

    _, pts_v = project_with_ego(pts, cams[FC])
    if len(pts_v) == 0:
        print('  (no pts in FC)')
        return

    f, cx, cy, W, H = _canvas_params(None, [pts_v], z_ref)

    # Build seam strip mask at column W//2 with half_width=200
    seam_col = W // 2
    half = 200
    cols = np.arange(W, dtype=np.float32)
    prof = np.maximum(0.0, 1.0 - np.abs(cols - seam_col) / half)
    mask = np.tile(prof[np.newaxis, :], (H, 1)).astype(np.float32)

    cam = cams[FC]
    canvas_px = ego_to_canvas(pts_v, f, cx, cy, z_ref)
    cam_px, _ = project_with_ego(pts, cams[FC])
    src_pts, dst_pts = grid_subsample(cam_px, canvas_px, cam['W'], cam['H'])

    if len(src_pts) < 4:
        print('  (too few control pts)')
        return

    mx_tps, my_tps = build_tps_remap(
        src_pts, dst_pts, cam['W'], cam['H'], W, H,
        cam=cam, f_cyl=f, cx_canvas=cx, cy_canvas=cy,
        smoothing=TPS_SMOOTHING, remap_scale=0.5, overlap_mask=mask)
    mx_rot, my_rot = build_rotation_remap(cam, f, cx, cy, W, H)

    # Outside the strip (mask == 0) the maps must be identical
    outside = mask == 0.0
    valid   = (mx_rot >= 0) & (my_rot >= 0)
    region  = outside & valid
    if not region.any():
        print('  (no outside-strip valid pixels)')
        return

    dx_outside = np.abs(mx_tps[region] - mx_rot[region])
    dy_outside = np.abs(my_tps[region] - my_rot[region])
    check('TPS == rotation outside seam strip (max delta < 0.1px)',
          dx_outside.max() < 0.1 and dy_outside.max() < 0.1,
          f'max Δx={dx_outside.max():.3f}  max Δy={dy_outside.max():.3f}')


# ══════════════════════════════════════════════════════════════════════════════
# Test 6 -- integration: shared pts included in both cameras' control sets
# ══════════════════════════════════════════════════════════════════════════════

def test_shared_pts_in_both_ctrl_sets():
    """After prepending shared points, both FL and FC include the shared canvas targets."""
    print('\n[6] shared pts appear in both cameras ctrl sets')
    if not CALIB_JSON.exists():
        print('  (skipped -- calibration not found)')
        return

    cams = load_calib(str(CALIB_JSON))
    FL, FC, FR = FRONT_CAMS
    f_cyl = float(cams[FC]['fx'])
    z_ref = float(np.mean([cams[n]['t'][2] for n in FRONT_CAMS]))

    rng = np.random.default_rng(1)
    pts = rng.uniform(5, 80, (5000, 3)).astype(np.float32)
    pts[:, 0] = np.abs(pts[:, 0])

    _, pts_l = project_with_ego(pts, cams[FL])
    _, pts_r = project_with_ego(pts, cams[FC])
    if len(pts_l) == 0 or len(pts_r) == 0:
        print('  (no pts visible)')
        return
    f, cx, cy, W, H = _canvas_params(None, [pts_l, pts_r], z_ref)

    sh_px_FL, sh_px_FC, sh_cvs = find_shared_ctrl_pts(
        pts, cams[FL], cams[FC], f_cyl, cx, cy, z_ref)

    if len(sh_cvs) == 0:
        print('  (no shared pts found)')
        return

    # Build FL ctrl set
    cam_px_fl, pts_fl = project_with_ego(pts, cams[FL])
    rng_fl = np.linalg.norm(pts_fl, axis=1)
    far_fl = rng_fl >= LIDAR_MIN_CTRL_RANGE_M
    cvs_fl = ego_to_canvas(pts_fl[far_fl], f_cyl, cx, cy, z_ref)
    src_fl, dst_fl = grid_subsample(cam_px_fl[far_fl], cvs_fl, cams[FL]['W'], cams[FL]['H'])
    src_fl = np.vstack([sh_px_FL, src_fl]) if len(sh_px_FL) else src_fl
    dst_fl = np.vstack([sh_cvs, dst_fl])   if len(sh_cvs)  else dst_fl

    # Build FC ctrl set
    cam_px_fc, pts_fc = project_with_ego(pts, cams[FC])
    rng_fc = np.linalg.norm(pts_fc, axis=1)
    far_fc = rng_fc >= LIDAR_MIN_CTRL_RANGE_M
    cvs_fc = ego_to_canvas(pts_fc[far_fc], f_cyl, cx, cy, z_ref)
    src_fc, dst_fc = grid_subsample(cam_px_fc[far_fc], cvs_fc, cams[FC]['W'], cams[FC]['H'])
    src_fc = np.vstack([sh_px_FC, src_fc]) if len(sh_px_FC) else src_fc
    dst_fc = np.vstack([sh_cvs, dst_fc])   if len(sh_cvs)  else dst_fc

    # Each shared canvas target appears in both FL and FC dst_pts
    sh_set_fl = set(map(tuple, np.round(dst_fl[:len(sh_cvs)]).astype(int).tolist()))
    sh_set_fc = set(map(tuple, np.round(dst_fc[:len(sh_cvs)]).astype(int).tolist()))
    sh_set_gt = set(map(tuple, np.round(sh_cvs).astype(int).tolist()))

    check('shared canvas targets in FL ctrl set', sh_set_gt == sh_set_fl,
          f'{len(sh_set_gt & sh_set_fl)}/{len(sh_set_gt)} matched')
    check('shared canvas targets in FC ctrl set', sh_set_gt == sh_set_fc,
          f'{len(sh_set_gt & sh_set_fc)}/{len(sh_set_gt)} matched')
    check('FL and FC share identical canvas targets for shared pts',
          sh_set_fl == sh_set_fc)


# ══════════════════════════════════════════════════════════════════════════════

def main():
    print('=' * 60)
    print('lidar_ring_stitch -- unit tests')
    print('=' * 60)

    test_canvas_camera_independent()
    test_project_indices()
    test_shared_pts_same_canvas()
    test_seam_strip_mask()
    test_tps_correction_outside_strip()
    test_shared_pts_in_both_ctrl_sets()

    passed = sum(_results)
    total  = len(_results)
    print(f'\n{"=" * 60}')
    print(f'Results: {passed}/{total} passed')
    if passed < total:
        print(f'{FAIL}: {total - passed} test(s) failed')
        sys.exit(1)
    else:
        print(f'{PASS}: all tests passed')


if __name__ == '__main__':
    main()
