#!/usr/bin/env python3
"""Parameter sweep to find optimal LiDAR-TPS stitching configuration.

Metrics per config per frame:
  p95_FR     -- 95th-percentile TPS displacement magnitude in FR non-overlap zone
                (lower = less extrapolation distortion outside overlap)
  seam_l1    -- mean L1 colour diff at the FC<->FR DP seam
                (lower = better seam alignment)
  seam_std   -- std of seam column positions across rows
                (lower = smoother / more stable seam path)

Composite score (all metrics normalised to [0,1] over all configs):
  score = 0.4 * p95_FR_norm + 0.4 * seam_l1_norm + 0.2 * seam_std_norm

Usage
-----
    cd code_folder/02_implementations/udis_pp_lidarcustom
    python sweep_params.py
"""

import itertools
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# -- Import helpers from the main script --------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from lidar_ring_stitch import (
    FRONT_CAMS, LIDAR_MIN_CTRL_RANGE_M, TPS_SMOOTHING,
    CANVAS_MARGIN_FRAC, SEAM_HALF_WIDTH_PX,
    SENSOR_ROOT, CALIB_JSON, FRAMES_JSON,
    load_calib, project_with_ego, ego_to_canvas,
    find_shared_ctrl_pts, build_tps_remap, build_rotation_remap, find_seam_dp,
)

OUT_DIR = Path(__file__).parent.parent / 'output/lidar_ring_stitch'
FRAMES_TO_TEST = [0, 50, 200]

# -- Sweep grid ----------------------------------------------------------------
# real_overlap_mask is always ON (clearly the dominant fix).
# One extra baseline row (all defaults, mask OFF) for reference.
_swept = [
    dict(min_ctrl_range=mcr, tps_smoothing=sm, real_overlap_mask=True,
         symmetric=False, mid_alpha=1.0)
    for mcr in [12]
    for sm  in [0]
]
_baseline = [dict(min_ctrl_range=5, tps_smoothing=10, real_overlap_mask=False,
                  symmetric=False, mid_alpha=1.0)]
CONFIGS = _swept + _baseline   # 90 + 1 = 91 configs


# -- Frame data loading --------------------------------------------------------

def load_frame_data(frame_idx: int, sensor_root: Path, calib_path: str,
                    frames_path: str) -> dict:
    """Load images, LiDAR, calib and pre-compute per-camera rotation remaps.

    Returns a dict with everything needed to run all configs against this frame
    without re-loading heavy data.
    """
    cams = load_calib(calib_path)
    with open(frames_path) as f:
        frames = json.load(f)
    frame = frames[frame_idx]

    # -- Find closest LiDAR sweep ----------------------------------------------
    lidar_files: dict[int, Path] = {}
    for fp in sensor_root.glob('train/*/sensors/lidar/*.feather'):
        try:
            lidar_files[int(fp.stem)] = fp
        except ValueError:
            pass
    if not lidar_files:
        raise RuntimeError(f'No LiDAR feather files under {sensor_root}')

    ref_cam = 'ring_front_center'
    img_ts  = int(Path(frame[ref_cam]).stem)
    lidar_ts = np.array(sorted(lidar_files.keys()), dtype=np.int64)
    best_ts  = int(lidar_ts[int(np.argmin(np.abs(lidar_ts - img_ts)))])
    df  = pd.read_feather(lidar_files[best_ts])
    pts = df[['x', 'y', 'z']].values.astype(np.float32)

    # -- Load images -----------------------------------------------------------
    images: dict[str, np.ndarray] = {}
    for name in FRONT_CAMS:
        path = frame.get(name)
        if path is None:
            raise KeyError(f'{name} not in frame {frame_idx}')
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(path)
        images[name] = img

    # -- Canvas geometry (identical to lidar_ring_stitch.py) ------------------
    f_cyl = float(cams[ref_cam]['fx'])
    Z_REF = float(np.mean([cams[n]['t'][2] for n in FRONT_CAMS]))

    all_az, all_el = [], []
    for name in FRONT_CAMS:
        _, pts_v = project_with_ego(pts, cams[name])
        if len(pts_v) == 0:
            continue
        az = np.arctan2(pts_v[:, 1], pts_v[:, 0])
        r_xy = np.sqrt(pts_v[:, 0]**2 + pts_v[:, 1]**2)
        el = np.arctan2(pts_v[:, 2] - Z_REF, r_xy)
        all_az.append(az); all_el.append(el)

    if not all_az:
        raise RuntimeError('No LiDAR points visible in any front camera')

    az_all = np.concatenate(all_az)
    el_all = np.concatenate(all_el)
    az_min, az_max = float(az_all.min()), float(az_all.max())
    el_min, el_max = float(el_all.min()), float(el_all.max())
    az_mg = CANVAS_MARGIN_FRAC * (az_max - az_min)
    el_mg = CANVAS_MARGIN_FRAC * (el_max - el_min)
    az_min -= az_mg; az_max += az_mg
    el_min -= el_mg; el_max += el_mg

    W_canvas = max(1, int(np.ceil(f_cyl * (az_max - az_min))))
    H_canvas = max(1, int(np.ceil(f_cyl * (el_max - el_min))))
    cx_canvas = float(az_max * f_cyl)
    cy_canvas = float(el_max * f_cyl)

    # -- Rotation remaps + overlap geometry ------------------------------------
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

    # Seam strip masks (used when real_overlap_mask=False)
    ov_FL_FC = rot_valid[FL] & rot_valid[FC]
    ov_FC_FR = rot_valid[FC] & rot_valid[FR]
    seam_FL_FC = (int(np.where(ov_FL_FC.any(axis=0))[0].mean())
                  if ov_FL_FC.any() else W_canvas // 3)
    seam_FC_FR = (int(np.where(ov_FC_FR.any(axis=0))[0].mean())
                  if ov_FC_FR.any() else 2 * W_canvas // 3)

    def seam_strip_mask(seam_col: int) -> np.ndarray:
        cols = np.arange(W_canvas, dtype=np.float32)
        prof = np.maximum(0.0, 1.0 - np.abs(cols - seam_col) / SEAM_HALF_WIDTH_PX)
        return np.tile(prof[np.newaxis, :], (H_canvas, 1)).astype(np.float32)

    strip_FL_FC = seam_strip_mask(seam_FL_FC)
    strip_FC_FR = seam_strip_mask(seam_FC_FR)

    # Real overlap masks (blur for smooth falloff)
    ov_lc = cv2.GaussianBlur((rot_valid[FL] & rot_valid[FC]).astype(np.float32),
                              (0, 0), 40)
    ov_cr = cv2.GaussianBlur((rot_valid[FC] & rot_valid[FR]).astype(np.float32),
                              (0, 0), 40)
    real_overlap_masks = {
        FL: ov_lc,
        FC: np.maximum(ov_lc, ov_cr),
        FR: ov_cr,
    }
    strip_overlap_masks = {
        FL: strip_FL_FC,
        FC: np.maximum(strip_FL_FC, strip_FC_FR),
        FR: strip_FC_FR,
    }

    # -- Project LiDAR once ----------------------------------------------------
    proj: dict[str, tuple] = {}
    for name in FRONT_CAMS:
        px, ego, idx = project_with_ego(pts, cams[name], return_indices=True)
        proj[name] = (px, ego, idx)

    # -- FR rotation remap for displacement metric -----------------------------
    # Valid FR pixels that are NOT in the FC<->FR overlap zone (non-overlap region)
    fr_nonov_valid = rot_valid[FR] & ~(rot_valid[FC] & rot_valid[FR])
    rot_mx_FR, rot_my_FR = rot_remaps[FR]

    return dict(
        cams=cams, images=images, pts=pts,
        f_cyl=f_cyl, Z_REF=Z_REF,
        W_canvas=W_canvas, H_canvas=H_canvas,
        cx_canvas=cx_canvas, cy_canvas=cy_canvas,
        rot_remaps=rot_remaps, rot_valid=rot_valid,
        proj=proj,
        real_overlap_masks=real_overlap_masks,
        strip_overlap_masks=strip_overlap_masks,
        rot_mx_FR=rot_mx_FR, rot_my_FR=rot_my_FR,
        fr_nonov_valid=fr_nonov_valid,
        ov_FC_FR=ov_FC_FR,
        FL=FL, FC=FC, FR=FR,
    )


# -- Per-config evaluation -----------------------------------------------------

def run_config(cfg: dict, fd: dict) -> dict:
    """Run one config on pre-loaded frame data; return metric dict."""
    cams         = fd['cams']
    images       = fd['images']
    f_cyl        = fd['f_cyl']
    Z_REF        = fd['Z_REF']
    W_canvas     = fd['W_canvas']
    H_canvas     = fd['H_canvas']
    cx_canvas    = fd['cx_canvas']
    cy_canvas    = fd['cy_canvas']
    rot_remaps   = fd['rot_remaps']
    rot_valid    = fd['rot_valid']
    proj         = fd['proj']
    FL, FC, FR   = fd['FL'], fd['FC'], fd['FR']

    overlap_masks = (fd['real_overlap_masks'] if cfg['real_overlap_mask']
                     else fd['strip_overlap_masks'])

    # -- Shared control points -------------------------------------------------
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

    shared_src = {FL: sh_px_FL,
                  FC: np.empty((0, 2), np.float32),
                  FR: sh_px_FR}
    shared_dst = {FL: sh_cvs_FL_FC,
                  FC: np.empty((0, 2), np.float32),
                  FR: sh_cvs_FC_FR}

    # -- Warp each camera ------------------------------------------------------
    warped: dict[str, np.ndarray] = {}
    valid:  dict[str, np.ndarray] = {}
    tps_remaps: dict[str, tuple]  = {}

    for name in FRONT_CAMS:
        cam = cams[name]
        W_cam, H_cam = cam['W'], cam['H']
        img = images[name]

        if name == FC:
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

        tps_remaps[name] = (mx, my)
        warped_img = cv2.remap(img, mx, my, cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        warped[name] = warped_img
        valid[name]  = rot_valid[name]

    # -- Metric 1: p95 FR displacement in non-overlap zone --------------------
    rot_mx_FR, rot_my_FR = rot_remaps[FR]
    tps_mx_FR, tps_my_FR = tps_remaps[FR]
    dx = tps_mx_FR - rot_mx_FR
    dy = tps_my_FR - rot_my_FR
    mag = np.sqrt(dx**2 + dy**2)
    nonov = fd['fr_nonov_valid']
    vals = mag[nonov]
    nonzero = vals > 0.5
    p95_FR = float(np.percentile(vals[nonzero], 95)) if nonzero.any() else 0.0

    # -- Metric 2+3: seam L1 and seam std at FC<->FR ----------------------------
    seam = find_seam_dp(warped[FC], warped[FR], valid[FC], valid[FR])
    H = H_canvas
    diffs = []
    for r in range(H):
        c = seam[r]
        if (0 <= c < warped[FC].shape[1]
                and valid[FC][r, c] and valid[FR][r, c]):
            diffs.append(float(np.mean(
                np.abs(warped[FC][r, c].astype(float)
                       - warped[FR][r, c].astype(float)))))
    seam_l1  = float(np.mean(diffs)) if diffs else 0.0
    seam_std = float(np.std(seam))

    return dict(p95_FR=p95_FR, seam_l1=seam_l1, seam_std=seam_std)


# -- Main ----------------------------------------------------------------------

def main():
    sensor_root = SENSOR_ROOT
    calib_path  = str(CALIB_JSON)
    frames_path = str(FRAMES_JSON)

    results = []
    for frame_idx in FRAMES_TO_TEST:
        print(f'\n{"="*60}')
        print(f'Loading frame {frame_idx}…')
        fd = load_frame_data(frame_idx, sensor_root, calib_path, frames_path)
        n_cfg = len(CONFIGS)
        for i, cfg in enumerate(CONFIGS):
            label = ('BASELINE' if not cfg['real_overlap_mask']
                     else f'mcr={cfg["min_ctrl_range"]} sm={cfg["tps_smoothing"]}')
            print(f'  [{i+1:3d}/{n_cfg}] {label}', end='', flush=True)
            try:
                metrics = run_config(cfg, fd)
                print(f'  p95={metrics["p95_FR"]:5.1f} l1={metrics["seam_l1"]:5.1f}'
                      f' std={metrics["seam_std"]:5.1f}')
            except Exception as e:
                print(f'  ERROR: {e}')
                metrics = dict(p95_FR=np.nan, seam_l1=np.nan, seam_std=np.nan)
            results.append({'frame': frame_idx, **cfg, **metrics})

    df = pd.DataFrame(results)

    # -- Normalise and score ---------------------------------------------------
    # Per frame, normalise each metric to [0,1] over all configs, then average
    # across frames.
    for metric in ['p95_FR', 'seam_l1', 'seam_std']:
        df[f'{metric}_norm'] = np.nan
        for fi in FRAMES_TO_TEST:
            mask = df['frame'] == fi
            col  = df.loc[mask, metric]
            mn, mx_v = col.min(), col.max()
            rng = mx_v - mn
            df.loc[mask, f'{metric}_norm'] = (col - mn) / rng if rng > 0 else 0.0

    df['score_per_frame'] = (0.4 * df['p95_FR_norm']
                             + 0.4 * df['seam_l1_norm']
                             + 0.2 * df['seam_std_norm'])

    # Average score across frames
    cfg_cols = ['min_ctrl_range', 'tps_smoothing', 'real_overlap_mask']
    agg = (df.groupby(cfg_cols, dropna=False)
             .agg(score=('score_per_frame', 'mean'),
                  p95_FR=('p95_FR', 'mean'),
                  seam_l1=('seam_l1', 'mean'),
                  seam_std=('seam_std', 'mean'))
             .reset_index()
             .sort_values('score'))

    # -- Save CSV --------------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / 'sweep_results.csv'
    df.to_csv(csv_path, index=False)
    print(f'\nFull results saved to {csv_path}')

    # -- Print top-10 ---------------------------------------------------------
    print('\n-- Top-10 configs (lower score = better) --------------------------')
    top10 = agg.head(10).reset_index(drop=True)
    print(top10.to_string(
        columns=['min_ctrl_range', 'tps_smoothing', 'real_overlap_mask',
                 'score', 'p95_FR', 'seam_l1', 'seam_std'],
        float_format='%.2f', index=True))

    # -- Per-frame best per metric ---------------------------------------------
    for fi in FRAMES_TO_TEST:
        sub = df[df['frame'] == fi].sort_values('score_per_frame')
        print(f'\n-- Frame {fi} best config --')
        row = sub.iloc[0]
        print(f'  score={row["score_per_frame"]:.3f}  '
              f'p95={row["p95_FR"]:.1f}  l1={row["seam_l1"]:.1f}  '
              f'std={row["seam_std"]:.1f}')
        print(f'  min_ctrl_range={row["min_ctrl_range"]}  '
              f'tps_smoothing={row["tps_smoothing"]}  '
              f'real_overlap_mask={row["real_overlap_mask"]}')

    print('\n-- Baseline (no real_overlap_mask) -------------------------------')
    baseline = agg[~agg['real_overlap_mask']]
    if not baseline.empty:
        row = baseline.iloc[0]
        print(f'  score={row["score"]:.3f}  '
              f'p95={row["p95_FR"]:.1f}  l1={row["seam_l1"]:.1f}  '
              f'std={row["seam_std"]:.1f}')

    print('\nDone.')


if __name__ == '__main__':
    main()
