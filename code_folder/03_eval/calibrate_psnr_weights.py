#!/usr/bin/env python3
"""
SUPERSEDED bitrate-based weighting prototype -- NOT the deployed metric.

The thesis (Eq. 3.6 / section 3.4) weights the channel PSNRs by the per-channel
signal-energy share (wY=0.918, wCb=0.047, wCr=0.035), derived by
calibrate_channel_weights.py. This file is the earlier bitrate-based attempt,
kept only for provenance. It is confounded by the NVENC rate cap: on high-detail
logs the encoder saturates its bitrate ceiling, so forcing chroma to 128 frees
no bytes (the bits move to luma instead) and the measured chroma cost collapses
to ~0 -- yielding wY ~ 1.0 (and even <0 from noise) rather than the 0.75-0.85
below. Use calibrate_channel_weights.py instead. The original description
follows.

---

Calibrate the bitrate-derived weights for the composite Y/Cb/Cr PSNR metric.

For each evaluated log this runs the deployed configuration twice via NVENC
HEVC at the recommended `--video-preset` settings:

    1. full       : normal BGR->NV12 -> HEVC encode. Total bytes B_full.
    2. y_only     : same Y plane, but Cb/Cr forced to 128. Total bytes B_y_only.

The chroma bit cost is `B_full − B_y_only`. Per-channel weights split chroma
evenly across Cb/Cr:

    w_Y  = B_y_only / B_full
    w_Cb = w_Cr = (1 − w_Y) / 2

Composite PSNR (computed in MSE space, not log space) is then

    MSE_w  = w_Y · MSE_Y + w_Cb · MSE_Cb + w_Cr · MSE_Cr
    PSNR_w = 10 log10(255² / MSE_w)

The weights are an empirical property of the deployed HEVC encoder + content
and are reported per-log alongside the metric. A typical natural-content
4:2:0 encode lands around w_Y ~ 0.75–0.85.

Usage:
    python3 calibrate_psnr_weights.py
"""

import csv
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BIN  = REPO / "code_folder/02_implementations/lidar_tps_pipeline/build/lidartps"
OUT  = REPO / "code_folder/02_implementations/lidar_tps_pipeline/eval/psnr_weights.csv"

# Same logs as multi_log_eval.py.
LOGS = [
    ("orig",
     "argo2_data/extracted/calibration.json",
     "argo2_data/extracted/frames.json",
     "argo2_data/extracted/sensors/lidar"),
    ("log_01bb",
     "argo2_data/extra_extracted/01bb304d-7bd8-35f8-bbef-7086b688e35e/calibration.json",
     "argo2_data/extra_extracted/01bb304d-7bd8-35f8-bbef-7086b688e35e/frames.json",
     "argo2_data/extra_extracted/01bb304d-7bd8-35f8-bbef-7086b688e35e/sensors/lidar"),
    ("log_022a",
     "argo2_data/extra_extracted/022af476-9937-3e70-be52-f65420d52703/calibration.json",
     "argo2_data/extra_extracted/022af476-9937-3e70-be52-f65420d52703/frames.json",
     "argo2_data/extra_extracted/022af476-9937-3e70-be52-f65420d52703/sensors/lidar"),
    ("log_0497",
     "argo2_data/extra_extracted/04973bcf-fc64-367c-9642-6d6c5f363b61/calibration.json",
     "argo2_data/extra_extracted/04973bcf-fc64-367c-9642-6d6c5f363b61/frames.json",
     "argo2_data/extra_extracted/04973bcf-fc64-367c-9642-6d6c5f363b61/sensors/lidar"),
]

NUM_FRAMES = 318  # match the 318-frame video render referenced in §4.6
CANVAS_W   = 5238
CANVAS_H   = 2303


def encode(tag, calib, frames, lidar, out_mp4, y_only):
    cmd = [
        str(BIN),
        "--calib",  calib,
        "--frames", frames,
        "--lidar",  lidar,
        "--video-nvenc", out_mp4,
        "--video-preset",
        "--num-frames", str(NUM_FRAMES),
    ]
    if y_only:
        cmd.append("--video-nvenc-y-only")
    print(f"  [{tag}] {'y_only' if y_only else 'full   '}: {out_mp4}")
    p = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True, check=False)
    if p.returncode != 0:
        print(p.stdout)
        print(p.stderr, file=sys.stderr)
        raise RuntimeError(f"lidartps failed (log={tag}, y_only={y_only})")


def main():
    if not BIN.exists():
        raise SystemExit(f"binary not found: {BIN}")
    rows = []
    for tag, calib, frames, lidar in LOGS:
        full_path = f"/tmp/calib_{tag}_full.mp4"
        yo_path   = f"/tmp/calib_{tag}_y_only.mp4"
        print(f"=== {tag} ===")
        encode(tag, calib, frames, lidar, full_path, y_only=False)
        encode(tag, calib, frames, lidar, yo_path,   y_only=True)
        B_full   = os.path.getsize(full_path)
        B_y_only = os.path.getsize(yo_path)
        if B_full <= 0 or B_y_only <= 0:
            raise RuntimeError(f"empty mp4 for {tag}")
        w_Y  = B_y_only / B_full
        w_Cb = w_Cr = (1.0 - w_Y) / 2.0
        bpp  = (B_full * 8) / (CANVAS_W * CANVAS_H * NUM_FRAMES)
        rows.append({
            "log":        tag,
            "B_full":     B_full,
            "B_y_only":   B_y_only,
            "B_chroma":   B_full - B_y_only,
            "w_Y":        round(w_Y,  4),
            "w_Cb":       round(w_Cb, 4),
            "w_Cr":       round(w_Cr, 4),
            "bits_per_pixel": round(bpp, 4),
        })
        print(f"  B_full={B_full/1e6:.1f}MB  B_y_only={B_y_only/1e6:.1f}MB  "
              f"w_Y={w_Y:.3f}  bpp={bpp:.3f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT}")
    # Echo
    with open(OUT) as f:
        print(f.read())


if __name__ == "__main__":
    main()
