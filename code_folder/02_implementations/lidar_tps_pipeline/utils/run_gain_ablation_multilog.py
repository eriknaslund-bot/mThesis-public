#!/usr/bin/env python3
"""Gain-compensation ablation on the 3 extra AV2 logs (cheap pass +
rich SSIM/LPIPS via the existing pass2_rich machinery).

The existing results_multilog.csv already has the deployed (gain-on)
config across all 3 extra logs. This script adds the no-gain variant
for cheap metrics + a rich pass at 32 frames per log for SSIM.

Outputs:
- eval/results_gain_multilog.csv (cheap pass, gain-off only, 318 fr/log)
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_pipeline as ep

EVAL = ep.EVAL

if not ep.BIN.exists():
    raise SystemExit(f"binary not found: {ep.BIN} -- run `cd build && make` first")

# Cheap pass: no-gain only (gain-on is already in results_multilog.csv).
all_rows = []
for log_path in ep.MULTILOG_DIRS:
    log_dir = ep.REPO / log_path
    if not log_dir.exists():
        print(f"  [skip] log dir missing: {log_dir}")
        continue
    log_tag = log_dir.name.split("-")[0]
    extra_log_flags = [
        "--calib",  str(log_dir / "calibration.json"),
        "--frames", str(log_dir / "frames.json"),
        "--lidar",  str(log_dir / "sensors" / "lidar"),
    ]
    tag = f"seamdp_nogain__{log_tag}"
    rows = ep.run_metrics(
        tag,
        ["--no-gain", *extra_log_flags],
        ep.N_FRAMES_CHEAP,
    )
    for r in rows:
        r["log"] = log_tag
        r["gain"] = "off"
    print(f"  log={log_tag} no-gain: {len(rows)} frames")
    all_rows.extend(rows)

path = EVAL / "results_gain_multilog.csv"
ep.write_csv(all_rows, path)
print(f"wrote {path} ({len(all_rows)} rows)")
