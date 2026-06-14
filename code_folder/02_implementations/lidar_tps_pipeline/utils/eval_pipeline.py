#!/usr/bin/env python3
# eval_pipeline.py -- full evaluation harness for the CUDA lidar_tps_pipeline.
#
# Pass 0  (smoke):  4-frame sanity check on the deployed default + one extra log,
#                   to fail fast before the long passes if a flag or path is wrong.
# Pass 1  (cheap):  318 frames x 2 configs, run lidartps --metrics, aggregate
#                   per-stage timings, shared-pt counts, seam-L1, seam-std, PSNR.
# Pass 2  (rich):   32 frames x 2 configs with --eval-dump + --holdout-frac.
#                   Held-out LiDAR reprojection, ORB residual, SSIM, LPIPS.
# Pass 3  (λ):      318 frames x 4 λ values (§4.6).
# Pass 4  (M):      318 frames x 7 ctrl-pt-density values (§4.7).
# Pass 5  (α):      318-frame CPU-H.264 MP4 x 5 α values, file-size as
#                   compressibility proxy (§4.8). CPU libx264 is intentional --
#                   NVENC defaults to CBR and would flatten the signal.
# Pass 6  (multi):  319 frames x 3 extra AV2 logs, generalisation check (§4.9).
# Pass 7  (codec):  CPU H.264 vs NVENC HEVC end-to-end walltime, 64 frames each
#                   (§4.9 tab:results_video_encoder).
#
# Outputs (next to this script in a sibling `eval/` dir):
#   eval/results_{cheap,rich,lambda,m,alpha,multilog,encoder}.csv
#   eval/results_summary_*.csv  (per-table aggregates)
#   eval/results_summary.csv    (combined one-pager)

import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    print("ERROR: needs opencv-python -- pip install opencv-python", file=sys.stderr)
    sys.exit(1)

try:
    from skimage.metrics import structural_similarity as _ssim
    _HAVE_SSIM = True
except ImportError:
    _HAVE_SSIM = False
    print("[warn] scikit-image missing -- SSIM will be omitted (pip install scikit-image)")

try:
    import torch
    import lpips as _lpips_mod
    _HAVE_LPIPS = True
    _LPIPS_NET = None  # lazy init
    _LPIPS_DEV = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    _HAVE_LPIPS = False
    print("[warn] lpips/torch missing -- LPIPS will be omitted (pip install lpips)")

# -- Paths ---------------------------------------------------------------------

THIS  = Path(__file__).resolve()
PIPE  = THIS.parent.parent                   # lidar_tps_pipeline/
REPO  = PIPE.parent.parent.parent            # mThesis repo root
BIN   = PIPE / "build" / "lidartps"
EVAL  = PIPE / "eval"
EVAL.mkdir(exist_ok=True)

N_FRAMES_CHEAP   = 318    # one full pass through the primary log (no repetition)
N_FRAMES_RICH    = 32     # disk-bounded: each frame dumps ~108 MB of warped PNGs
N_FRAMES_SWEEP   = 318    # full primary log per λ / M value
N_FRAMES_ALPHA   = 318    # matches §4.8 caption for the temporal-stability MP4
N_FRAMES_MULTI   = 319    # per-log frame count; each extra log has 319 frames

LAMBDA_SWEEP = [0.0, 0.1, 1.0, 10.0]               # §4.6 TPS regularisation sweep
M_SWEEP      = [10, 50, 100, 200, 400, 600, 1000]  # §4.7 ctrl-pt density sweep
ALPHA_SWEEP  = [1.0, 0.6, 0.3, 0.15, 0.05]         # §4.8 video IIR sweep

# §4.9 multi-log generalisation: three additional AV2 sensor-split logs sitting
# alongside the primary extracted/ in argo2_data/. Same per-log directory
# layout: calibration.json + frames.json + sensors/lidar/*.bin.
MULTILOG_DIRS = [
    # First 3: original 2026-05-25 extras (bright daytime, MIA/WDC).
    "argo2_data/extra_extracted/01bb304d-7bd8-35f8-bbef-7086b688e35e",
    "argo2_data/extra_extracted/022af476-9937-3e70-be52-f65420d52703",
    "argo2_data/extra_extracted/04973bcf-fc64-367c-9642-6d6c5f363b61",
    # Next 6: diversity extension 2026-05-26 (PIT/WDC/DTW; HDR, wet, fog,
    # bright suburb, depth-variance, overcast).
    "argo2_data/extra_extracted/087695bd-c662-3e86-83b4-aedc3b8eec36",  # PIT bridge underpass / HDR
    "argo2_data/extra_extracted/05853f69-f948-3d04-8d64-d4e721c0e1a5",  # PIT wet pavement
    "argo2_data/extra_extracted/12071817-ba53-35a4-bf6c-a8e8e7ad8969",  # PIT fog / haze
    "argo2_data/extra_extracted/12c3c14b-9cf2-3434-9a5d-e0bfa332f6ce",  # WDC bright suburban arterial
    "argo2_data/extra_extracted/0a524e66-ee33-3b6c-89ef-eac1985316db",  # PIT depth-variance downtown
    "argo2_data/extra_extracted/072c8e90-a51c-3429-9cdf-4dababb4e9d8",  # DTW overcast far-scene
]
# For the 2026-05-26 delta run only, restrict to the 6 new logs to avoid
# re-running the existing 3. The previously-saved CSVs are concatenated
# back in by the post-run merge step.
MULTILOG_DIRS_NEW6 = MULTILOG_DIRS[3:]

CONFIGS_CHEAP = [
    # (tag, extra cli flags)
    # The binary default (no extra flags) is the deployed configuration:
    # FC at rotation baseline, FL/FR carry the TPS warp, DP seam, gain
    # compensation on.
    ("seamdp",        []),                            # deployed default
    ("seamdp_nogain", ["--no-gain"]),                 # gain ablation
]
RICH_CONFIG_TAG   = "seamdp"
RICH_CONFIG_FLAGS = []  # recommended (deployed default -- no extra flags needed)

# Pass 2 runs the same configs so SSIM/LPIPS and held-out reprojection
# are directly comparable to the cheap-pass numbers per config.
CONFIGS_RICH = CONFIGS_CHEAP

# -- Helpers -------------------------------------------------------------------

def run_metrics(tag, extra_flags, n_frames):
    """Run lidartps --benchmark n_frames --metrics, return list of CSV dict rows."""
    cmd = [
        str(BIN),
        "--benchmark", str(n_frames),
        "--metrics",
        "--config-tag", tag,
        *extra_flags,
    ]
    print(f"[pass1] {' '.join(cmd[1:])}")
    p = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True, check=False)
    if p.returncode != 0:
        print(p.stdout)
        print(p.stderr, file=sys.stderr)
        raise RuntimeError(f"lidartps failed (config={tag})")
    header = None
    rows = []
    for line in p.stdout.splitlines():
        if not line.startswith("METRIC,"):
            continue
        fields = line[len("METRIC,"):].split(",")
        if header is None:
            header = fields
        else:
            rows.append(dict(zip(header, fields)))
    return rows


def write_csv(rows, path, fieldnames=None):
    if not rows:
        print(f"[warn] nothing to write to {path}")
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path} ({len(rows)} rows)")


# -- Pass 0: smoke -------------------------------------------------------------

def pass0_smoke():
    """4-frame sanity run on the deployed default + one extra-log path.

    Catches stale CLI flags, missing data dirs, and binary-load failures
    before the long passes burn 30 minutes."""
    print("\n--- Pass 0: smoke (4 frames x deployed default + 1 extra log) ---")

    rows = run_metrics("smoke_default", [], 4)
    if not rows:
        raise RuntimeError("smoke: deployed default emitted no METRIC rows")
    print(f"  default OK: {len(rows)} frames, first t_total={rows[0].get('t_total','?')} ms")

    for log_path in MULTILOG_DIRS:
        log_dir = REPO / log_path
        if log_dir.exists():
            extra = [
                "--calib",  str(log_dir / "calibration.json"),
                "--frames", str(log_dir / "frames.json"),
                "--lidar",  str(log_dir / "sensors" / "lidar"),
            ]
            rows = run_metrics(f"smoke_log_{log_dir.name.split('-')[0]}", extra, 4)
            if not rows:
                raise RuntimeError(f"smoke: extra log {log_dir.name} emitted no METRIC rows")
            print(f"  log {log_dir.name.split('-')[0]} OK: {len(rows)} frames")
            return
    print("  [warn] no extra log dirs present -- multi-log pass will be skipped")


# -- Pass 1: cheap -------------------------------------------------------------

def pass1_cheap():
    print(f"\n--- Pass 1: cheap metrics ({N_FRAMES_CHEAP} frames x {len(CONFIGS_CHEAP)} configs) ---")
    all_rows = []
    for tag, flags in CONFIGS_CHEAP:
        rows = run_metrics(tag, flags, N_FRAMES_CHEAP)
        print(f"  {tag}: {len(rows)} frames")
        all_rows.extend(rows)
    path = EVAL / "results_cheap.csv"
    write_csv(all_rows, path)
    return all_rows


# -- Pass 2: rich --------------------------------------------------------------

def _orb_residual(img_a, img_b, mask):
    """ORB keypoint match residual in overlap.

    Returns (median, p95, n) of matched-pair pixel distances. The median
    is used instead of RMSE because ORB latches onto any strong texture
    -- including parallax-prone objects at different depths -- so the
    residual distribution has a heavy right tail dominated by outliers
    that RMSE over-weights.
    """
    orb = cv2.ORB_create(nfeatures=2000)
    ga = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
    kpa, da = orb.detectAndCompute(ga, mask)
    kpb, db = orb.detectAndCompute(gb, mask)
    if da is None or db is None or len(kpa) < 10 or len(kpb) < 10:
        return np.nan, np.nan, 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(da, db, k=2)
    good = []
    for m in knn:
        if len(m) < 2: continue
        if m[0].distance < 0.75 * m[1].distance:
            good.append(m[0])
    if len(good) < 5:
        return np.nan, np.nan, len(good)
    pa = np.array([kpa[mm.queryIdx].pt for mm in good])
    pb = np.array([kpb[mm.trainIdx].pt for mm in good])
    d = np.linalg.norm(pa - pb, axis=1)
    return float(np.median(d)), float(np.percentile(d, 95)), len(good)


def _bgr_to_y_bt709(img_bgr):
    """BT.709 BGR->Y (luma), uint8 -> uint8. img_bgr layout: [..., 3] BGR."""
    b = img_bgr[..., 0].astype(np.float32)
    g = img_bgr[..., 1].astype(np.float32)
    r = img_bgr[..., 2].astype(np.float32)
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return np.clip(y, 0.0, 255.0).astype(np.uint8)


def _masked_ssim(img_a, img_b, mask):
    """Y-channel (BT.709 luma) masked SSIM. Reports the canonical Y-SSIM
    used in video-quality literature; reads the same mask the C++
    cheap-pass uses for its hull-restricted PSNR so the two metrics
    agree on evaluation region."""
    if not _HAVE_SSIM:
        return float('nan')
    # Crop to tight bbox of the mask to speed up
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return float('nan')
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    a = _bgr_to_y_bt709(img_a[y0:y1, x0:x1])
    b = _bgr_to_y_bt709(img_b[y0:y1, x0:x1])
    m = mask[y0:y1, x0:x1]
    ssim_full, ssim_map = _ssim(a, b, data_range=255, full=True, win_size=7)
    valid = m[3:-3, 3:-3] > 0  # SSIM drops a border of (win_size-1)/2
    smap_inner = ssim_map[3:-3, 3:-3]
    if valid.sum() == 0:
        return float(ssim_full)
    return float(smap_inner[valid].mean())


def _masked_lpips(img_a, img_b, mask):
    """LPIPS (AlexNet) computed on the tight mask bbox. Whole-patch score is
    fine as a proxy because the mask is the principal overlap band and the
    background is zero-padded identically in both inputs, contributing near-zero
    perceptual residual."""
    if not _HAVE_LPIPS:
        return float('nan')
    global _LPIPS_NET
    if _LPIPS_NET is None:
        _LPIPS_NET = _lpips_mod.LPIPS(net='alex', verbose=False).to(_LPIPS_DEV).eval()
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return float('nan')
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    a = img_a[y0:y1, x0:x1]
    b = img_b[y0:y1, x0:x1]
    m = (mask[y0:y1, x0:x1] > 0).astype(np.float32)
    # Apply mask before the perceptual pass so out-of-overlap pixels are zeroed
    # in both inputs -> contribute ~0 to the score.
    a = a.astype(np.float32) * m[..., None]
    b = b.astype(np.float32) * m[..., None]
    # BGR->RGB, HxWxC -> 1xCxHxW, [0,255] -> [-1,1]
    def _to_t(x):
        x = cv2.cvtColor(x.astype(np.uint8), cv2.COLOR_BGR2RGB).astype(np.float32)
        x = (x / 127.5) - 1.0
        t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(_LPIPS_DEV)
        return t
    with torch.no_grad():
        d = _LPIPS_NET(_to_t(a), _to_t(b))
    return float(d.squeeze().cpu().numpy())


def _holdout_error(csv_path):
    """Cross-camera held-out LiDAR reprojection error, in canvas pixels.

    Returns a dict with both the rotation-baseline residual (`rot_*`) and
    the post-TPS residual (`tps_*`). The post-TPS fields are computed by
    inverting the warp per camera and report the distance between canvas
    positions of the same 3-D point as seen by each camera after the TPS
    correction is applied -- so they directly measure the pipeline's output
    alignment at held-out points.
    """
    import pandas as pd
    df = pd.read_csv(csv_path)

    def pair_stats(rows, cu_a, cv_a, cu_b, cv_b):
        sel = (rows[cu_a] >= 0) & (rows[cu_b] >= 0)
        r = rows[sel]
        if len(r) == 0:
            return float('nan'), float('nan'), 0
        d = np.sqrt((r[cu_a] - r[cu_b])**2 + (r[cv_a] - r[cv_b])**2)
        return float(np.mean(d)), float(np.percentile(d, 95)), int(len(d))

    lc = df[df.overlap == 0]
    cr = df[df.overlap == 1]

    m_rot_lc, p_rot_lc, n_lc = pair_stats(lc, "cu_FL", "cv_FL", "cu_FC", "cv_FC")
    m_rot_cr, p_rot_cr, n_cr = pair_stats(cr, "cu_FC", "cv_FC", "cu_FR", "cv_FR")

    if "cu_tps_FL" in df.columns:
        m_tps_lc, p_tps_lc, _ = pair_stats(lc, "cu_tps_FL", "cv_tps_FL", "cu_tps_FC", "cv_tps_FC")
        m_tps_cr, p_tps_cr, _ = pair_stats(cr, "cu_tps_FC", "cv_tps_FC", "cu_tps_FR", "cv_tps_FR")
    else:
        m_tps_lc = p_tps_lc = m_tps_cr = p_tps_cr = float('nan')

    return dict(
        rot_mean_FL_FC=m_rot_lc, rot_p95_FL_FC=p_rot_lc,
        rot_mean_FC_FR=m_rot_cr, rot_p95_FC_FR=p_rot_cr,
        tps_mean_FL_FC=m_tps_lc, tps_p95_FL_FC=p_tps_lc,
        tps_mean_FC_FR=m_tps_cr, tps_p95_FC_FR=p_tps_cr,
        n_FL_FC=n_lc, n_FC_FR=n_cr,
    )


def pass2_rich():
    print(f"\n--- Pass 2: rich metrics ({N_FRAMES_RICH} frames x {len(CONFIGS_RICH)} configs) ---")
    all_rows = []
    for tag, flags in CONFIGS_RICH:
        all_rows.extend(_rich_one_config(tag, flags))
    path = EVAL / "results_rich.csv"
    write_csv(all_rows, path)
    return all_rows


def _rich_one_config(tag, flags):
    print(f"\n  [pass2:{tag}]")
    dump_root = EVAL / "dump" / tag
    dump_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(BIN),
        "--benchmark", str(N_FRAMES_RICH),
        "--eval-dump", str(dump_root),
        "--holdout-frac", "0.2",
        *flags,
    ]
    print(f"  [pass2:{tag}] {' '.join(cmd[1:])}")
    p = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True, check=False)
    if p.returncode != 0:
        print(p.stdout)
        print(p.stderr, file=sys.stderr)
        raise RuntimeError(f"lidartps --eval-dump failed (config={tag})")

    rows = []
    for fi in range(N_FRAMES_RICH):
        d = dump_root / f"frame_{fi:04d}"
        if not d.exists():
            print(f"  missing {d}; skipping")
            continue
        fl = cv2.imread(str(d / "warped_FL.png"))
        fc = cv2.imread(str(d / "warped_FC.png"))
        fr = cv2.imread(str(d / "warped_FR.png"))
        mlc = cv2.imread(str(d / "mask_FL_FC.png"), cv2.IMREAD_GRAYSCALE)
        mcr = cv2.imread(str(d / "mask_FC_FR.png"), cv2.IMREAD_GRAYSCALE)
        if fl is None or fc is None or fr is None or mlc is None or mcr is None:
            print(f"  frame {fi}: missing intermediates, skipping")
            continue

        timings = json.loads((d / "timings.json").read_text())

        orb_lc_med, orb_lc_p95, orb_lc_n = _orb_residual(fl, fc, mlc)
        orb_cr_med, orb_cr_p95, orb_cr_n = _orb_residual(fc, fr, mcr)

        ssim_lc = _masked_ssim(fl, fc, mlc)
        ssim_cr = _masked_ssim(fc, fr, mcr)
        lpips_lc = _masked_lpips(fl, fc, mlc)
        lpips_cr = _masked_lpips(fc, fr, mcr)

        ho = _holdout_error(d / "lidar_holdout.csv")

        rows.append(dict(
            config=tag,
            frame=fi,
            n_shared_FL_FC=timings["n_shared_FL_FC"],
            n_shared_FC_FR=timings["n_shared_FC_FR"],
            holdout_n_FL_FC=ho["n_FL_FC"],
            holdout_rot_mean_FL_FC=ho["rot_mean_FL_FC"],
            holdout_rot_p95_FL_FC=ho["rot_p95_FL_FC"],
            holdout_tps_mean_FL_FC=ho["tps_mean_FL_FC"],
            holdout_tps_p95_FL_FC=ho["tps_p95_FL_FC"],
            holdout_n_FC_FR=ho["n_FC_FR"],
            holdout_rot_mean_FC_FR=ho["rot_mean_FC_FR"],
            holdout_rot_p95_FC_FR=ho["rot_p95_FC_FR"],
            holdout_tps_mean_FC_FR=ho["tps_mean_FC_FR"],
            holdout_tps_p95_FC_FR=ho["tps_p95_FC_FR"],
            orb_med_FL_FC=orb_lc_med, orb_p95_FL_FC=orb_lc_p95, orb_n_FL_FC=orb_lc_n,
            orb_med_FC_FR=orb_cr_med, orb_p95_FC_FR=orb_cr_p95, orb_n_FC_FR=orb_cr_n,
            ssim_y_FL_FC=ssim_lc, ssim_y_FC_FR=ssim_cr,
            lpips_FL_FC=lpips_lc, lpips_FC_FR=lpips_cr,
            t_total_ms=timings["t_total_ms"],
        ))
        print(f"  frame {fi}: orb_lc={orb_lc_med:.2f} orb_cr={orb_cr_med:.2f} "
              f"ssim_lc={ssim_lc:.3f} ssim_cr={ssim_cr:.3f} "
              f"lpips_lc={lpips_lc:.3f} lpips_cr={lpips_cr:.3f} "
              f"hold_rot_lc={ho['rot_mean_FL_FC']:.2f} hold_tps_lc={ho['tps_mean_FL_FC']:.2f} "
              f"hold_rot_cr={ho['rot_mean_FC_FR']:.2f} hold_tps_cr={ho['tps_mean_FC_FR']:.2f}")

    return rows


# -- Pass 3: TPS λ sweep (§4.6) -----------------------------------------------

def pass3_lambda_sweep():
    print(f"\n--- Pass 3: TPS λ sweep ({N_FRAMES_SWEEP} frames x {len(LAMBDA_SWEEP)} λ) ---")
    all_rows = []
    for lam in LAMBDA_SWEEP:
        tag = f"lambda_{lam:g}"
        rows = run_metrics(tag, ["--tps-smooth", f"{lam:g}"], N_FRAMES_SWEEP)
        for r in rows:
            r["lambda"] = f"{lam:g}"
        print(f"  λ={lam:g}: {len(rows)} frames")
        all_rows.extend(rows)
    path = EVAL / "results_lambda.csv"
    write_csv(all_rows, path)
    return all_rows


# -- Pass 4: ctrl-pt density M sweep (§4.7) -----------------------------------

def pass4_m_sweep():
    print(f"\n--- Pass 4: ctrl-pt density M sweep ({N_FRAMES_SWEEP} frames x {len(M_SWEEP)} M) ---")
    all_rows = []
    for m in M_SWEEP:
        tag = f"M_{m}"
        rows = run_metrics(tag, ["--max-ctrl-per-overlap", str(m)], N_FRAMES_SWEEP)
        for r in rows:
            r["M"] = str(m)
        print(f"  M={m}: {len(rows)} frames")
        all_rows.extend(rows)
    path = EVAL / "results_m.csv"
    write_csv(all_rows, path)
    return all_rows


# -- Pass 5: video α sweep (§4.8, MP4 size as compressibility proxy) ----------

def pass5_alpha_video():
    print(f"\n--- Pass 5: video α sweep ({N_FRAMES_ALPHA} frames x {len(ALPHA_SWEEP)} α via CPU H.264) ---")
    rows = []
    out_dir = EVAL / "alpha_videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    for alpha in ALPHA_SWEEP:
        mp4 = out_dir / f"alpha_{alpha:g}.mp4"
        cmd = [
            str(BIN),
            "--video", str(mp4),
            "--num-frames", str(N_FRAMES_ALPHA),
            "--video-preset",       # locks λ=1
            "--disp-alpha", f"{alpha:g}",
        ]
        print(f"  α={alpha:g}: rendering -> {mp4.name} …")
        p = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True, check=False)
        if p.returncode != 0:
            print(p.stdout); print(p.stderr, file=sys.stderr)
            raise RuntimeError(f"lidartps --video failed (alpha={alpha})")
        size_bytes = mp4.stat().st_size
        rows.append({
            "alpha": f"{alpha:g}",
            "n_frames": N_FRAMES_ALPHA,
            "mp4_bytes": size_bytes,
            "mp4_MB":    f"{size_bytes / 1e6:.3f}",
        })
        print(f"  α={alpha:g}: {size_bytes / 1e6:.2f} MB")
    # Add relative-size column once all sizes are known
    if rows:
        baseline = next((r for r in rows if r["alpha"] == "1"), rows[0])
        b = int(baseline["mp4_bytes"]) or 1
        for r in rows:
            r["relative_size"] = f"{int(r['mp4_bytes']) / b:.4f}"
    path = EVAL / "results_alpha.csv"
    write_csv(rows, path,
        fieldnames=["alpha", "n_frames", "mp4_bytes", "mp4_MB", "relative_size"])
    return rows


# -- Pass 6: multi-log generalisation (§4.9) ----------------------------------

def pass6_multilog():
    print(f"\n--- Pass 6: multi-log generalisation ({N_FRAMES_MULTI} frames x {len(MULTILOG_DIRS)} logs) ---")
    all_rows = []
    for log_path in MULTILOG_DIRS:
        log_dir = REPO / log_path
        if not log_dir.exists():
            print(f"  [skip] log dir missing: {log_dir}")
            continue
        log_tag = log_dir.name.split("-")[0]   # short hash
        extra_flags = [
            "--calib",  str(log_dir / "calibration.json"),
            "--frames", str(log_dir / "frames.json"),
            "--lidar",  str(log_dir / "sensors" / "lidar"),
        ]
        rows = run_metrics(f"log_{log_tag}", extra_flags, N_FRAMES_MULTI)
        for r in rows:
            r["log"] = log_tag
        print(f"  log {log_tag}: {len(rows)} frames")
        all_rows.extend(rows)
    path = EVAL / "results_multilog.csv"
    write_csv(all_rows, path)
    return all_rows


# -- Pass 6b: rich metrics on the multi-log set --------------------------------
# Held-out reprojection + ORB + SSIM + LPIPS on the 3 extra AV2 logs, so the
# headline geometric numbers (Table 2) can be reported across logs and the §5
# generalisation claim is backed by direct measurement rather than asserted
# from the photometric pass alone.

def pass6b_rich_multilog():
    print(f"\n--- Pass 6b: rich multi-log ({N_FRAMES_RICH} frames x {len(MULTILOG_DIRS)} logs) ---")
    all_rows = []
    for log_path in MULTILOG_DIRS:
        log_dir = REPO / log_path
        if not log_dir.exists():
            print(f"  [skip] log dir missing: {log_dir}")
            continue
        log_tag = log_dir.name.split("-")[0]
        tag = f"multilog_{log_tag}"
        extra_flags = [
            "--calib",  str(log_dir / "calibration.json"),
            "--frames", str(log_dir / "frames.json"),
            "--lidar",  str(log_dir / "sensors" / "lidar"),
        ]
        rows = _rich_one_config(tag, extra_flags)
        for r in rows:
            r["log"] = log_tag
        all_rows.extend(rows)
    path = EVAL / "results_rich_multilog.csv"
    write_csv(all_rows, path)
    return all_rows


# -- Pass 7: codec walltime (§4.9 tab:results_video_encoder) ------------------

CODEC_FRAMES = 64  # short -- only need end-to-end wallclock per codec

def pass7_encoder():
    print(f"\n--- Pass 7: video encoder walltime ({CODEC_FRAMES} frames x 2 codecs) ---")
    out_dir = EVAL / "encoder_videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for codec, flag, ext in [
        ("cpu_h264",  "--video",       ".mp4"),
        ("nvenc_hevc","--video-nvenc", ".mp4"),
    ]:
        out = out_dir / f"{codec}{ext}"
        cmd = [str(BIN), flag, str(out),
               "--num-frames", str(CODEC_FRAMES),
               "--video-preset"]
        print(f"  {codec}: rendering -> {out.name} …")
        t0 = time.perf_counter()
        p = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True, check=False)
        wall_s = time.perf_counter() - t0
        if p.returncode != 0:
            print(p.stdout); print(p.stderr, file=sys.stderr)
            raise RuntimeError(f"encoder pass failed ({codec})")
        size = out.stat().st_size if out.exists() else 0
        rows.append({
            "codec":   codec,
            "n_frames": CODEC_FRAMES,
            "wall_s":  f"{wall_s:.3f}",
            "fps":     f"{CODEC_FRAMES / max(wall_s, 1e-9):.2f}",
            "mp4_MB":  f"{size / 1e6:.2f}",
        })
        print(f"  {codec}: {wall_s:.2f}s ({CODEC_FRAMES/max(wall_s,1e-9):.1f} fps), {size/1e6:.1f} MB")
    write_csv(rows, EVAL / "results_encoder.csv",
              fieldnames=["codec","n_frames","wall_s","fps","mp4_MB"])
    return rows


# -- Summary -------------------------------------------------------------------

def summarise(cheap_rows, rich_rows,
              lambda_rows=None, m_rows=None, alpha_rows=None, multilog_rows=None,
              encoder_rows=None):
    print("\n--- Aggregate summary ---")
    import pandas as pd
    dfc = pd.DataFrame(cheap_rows)
    numeric = ["n_shared_FL_FC","n_shared_FC_FR",
               "seam_l1_FL_FC","seam_l1_FC_FR",
               "seam_std_FL_FC","seam_std_FC_FR",
               "overlap_n_FL_FC","overlap_n_FC_FR",
               "overlap_psnr_y_FL_FC","overlap_psnr_y_FC_FR",
               "overlap_psnr_cb_FL_FC","overlap_psnr_cb_FC_FR",
               "overlap_psnr_cr_FL_FC","overlap_psnr_cr_FC_FR",
               "warp_mean_FL","warp_p95_FL","warp_max_FL",
               "warp_mean_FC","warp_p95_FC","warp_max_FC",
               "warp_mean_FR","warp_p95_FR","warp_max_FR",
               "tps_bend_FL","tps_bend_FC","tps_bend_FR",
               "t_project","t_tps","t_warp","t_seam","t_composite","t_total"]
    for c in numeric:
        if c in dfc.columns: dfc[c] = pd.to_numeric(dfc[c], errors="coerce")
    agg = dfc.groupby("config")[numeric].agg(["mean", "std"])
    cheap_summary_path = EVAL / "results_summary_cheap.csv"
    agg.to_csv(cheap_summary_path)
    print(f"  wrote {cheap_summary_path}")

    rich_agg = None
    if rich_rows:
        dfr = pd.DataFrame(rich_rows)
        rich_cols = ["holdout_rot_mean_FL_FC","holdout_rot_p95_FL_FC",
                     "holdout_tps_mean_FL_FC","holdout_tps_p95_FL_FC",
                     "holdout_rot_mean_FC_FR","holdout_rot_p95_FC_FR",
                     "holdout_tps_mean_FC_FR","holdout_tps_p95_FC_FR",
                     "orb_med_FL_FC","orb_p95_FL_FC",
                     "orb_med_FC_FR","orb_p95_FC_FR",
                     "ssim_y_FL_FC","ssim_y_FC_FR",
                     "lpips_FL_FC","lpips_FC_FR"]
        for c in rich_cols:
            if c in dfr.columns: dfr[c] = pd.to_numeric(dfr[c], errors="coerce")
        rich_agg = dfr.groupby("config")[rich_cols].agg(["mean", "std"])
        rich_summary_path = EVAL / "results_summary_rich.csv"
        rich_agg.to_csv(rich_summary_path)
        print(f"  wrote {rich_summary_path}")

    # Per-sweep aggregates (Y-PSNR + warp + bending + timings as a function of
    # the swept parameter). Each sweep writes its own summary CSV so the §4
    # tables can be regenerated by reading a single file per table.
    sweep_cols = ["overlap_psnr_y_FL_FC", "overlap_psnr_y_FC_FR",
                  "overlap_psnr_cb_FL_FC", "overlap_psnr_cb_FC_FR",
                  "overlap_psnr_cr_FL_FC", "overlap_psnr_cr_FC_FR",
                  "seam_l1_FL_FC", "seam_l1_FC_FR",
                  "warp_mean_FL", "warp_p95_FL", "warp_max_FL",
                  "warp_mean_FC", "warp_p95_FC", "warp_max_FC",
                  "warp_mean_FR", "warp_p95_FR", "warp_max_FR",
                  "tps_bend_FL", "tps_bend_FC", "tps_bend_FR",
                  "t_project", "t_tps", "t_warp", "t_seam", "t_composite", "t_total"]

    def _summarise_sweep(rows, key, out_name):
        if not rows: return None
        df = pd.DataFrame(rows)
        for c in sweep_cols:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
        sweep_agg = df.groupby(key)[sweep_cols].agg(["mean", "std"])
        path = EVAL / out_name
        sweep_agg.to_csv(path)
        print(f"  wrote {path}")
        return sweep_agg

    lambda_agg   = _summarise_sweep(lambda_rows,   "lambda", "results_summary_lambda.csv")
    m_agg        = _summarise_sweep(m_rows,        "M",      "results_summary_m.csv")
    multilog_agg = _summarise_sweep(multilog_rows, "log",    "results_summary_multilog.csv")

    # α sweep is a single-row-per-α MP4-size table; passthrough into summary.
    alpha_df = None
    if alpha_rows:
        alpha_df = pd.DataFrame(alpha_rows)
        alpha_path = EVAL / "results_summary_alpha.csv"
        alpha_df.to_csv(alpha_path, index=False)
        print(f"  wrote {alpha_path}")

    # Combined one-pager: per-config cheap + per-config rich + each sweep
    combined = EVAL / "results_summary.csv"
    with open(combined, "w") as f:
        f.write("# CHEAP per-config aggregate\n")
        agg.to_csv(f)
        f.write("\n# RICH per-config aggregate\n")
        if rich_agg is not None:
            rich_agg.to_csv(f)
        if lambda_agg is not None:
            f.write("\n# LAMBDA sweep (§4.6)\n")
            lambda_agg.to_csv(f)
        if m_agg is not None:
            f.write("\n# M sweep (§4.7)\n")
            m_agg.to_csv(f)
        if alpha_df is not None:
            f.write("\n# ALPHA video sweep (§4.8)\n")
            alpha_df.to_csv(f, index=False)
        if multilog_agg is not None:
            f.write("\n# MULTILOG generalisation (§4.9)\n")
            multilog_agg.to_csv(f)
        if encoder_rows:
            f.write("\n# ENCODER walltime (§4.9)\n")
            pd.DataFrame(encoder_rows).to_csv(f, index=False)
    print(f"  wrote {combined}")


# -- Main ----------------------------------------------------------------------

def main():
    if not BIN.exists():
        raise SystemExit(f"binary not found: {BIN} -- run `cd build && make` first")

    # Wipe stale dumps so old frame_NNNN/ dirs from earlier (longer) runs
    # don't shadow this run's outputs.
    dump_dir = EVAL / "dump"
    if dump_dir.exists():
        print(f"[init] wiping stale dump dir: {dump_dir}")
        shutil.rmtree(dump_dir, ignore_errors=True)

    pass0_smoke()
    cheap_rows    = pass1_cheap()
    rich_rows     = pass2_rich()
    lambda_rows   = pass3_lambda_sweep()
    m_rows        = pass4_m_sweep()
    alpha_rows    = pass5_alpha_video()
    multilog_rows = pass6_multilog()
    encoder_rows  = pass7_encoder()
    summarise(cheap_rows, rich_rows,
              lambda_rows=lambda_rows, m_rows=m_rows,
              alpha_rows=alpha_rows, multilog_rows=multilog_rows,
              encoder_rows=encoder_rows)
    print("\nDone.")


if __name__ == "__main__":
    main()
