#!/usr/bin/env python3
"""r_min ablation panel (parallel to the λ and M sweeps in §4).

Sweeps the minimum LiDAR control-point range r_min ∈ {0, 6, 10, 15} m and
runs the rich pass (held-out reprojection, ORB residual, SSIM, LPIPS) on
each. r_min = 6 is the deployed default; the thesis currently flags this
parameter as picked qualitatively without quantitative backing.

Output: eval/results_rmin.csv + eval/results_summary_rmin.csv.
"""
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_pipeline as ep

EVAL = ep.EVAL
DUMP = EVAL / "dump"

RMIN_SWEEP = [0.0, 6.0, 10.0, 15.0]

# Wipe only rmin_* dump dirs from any prior run.
for d in DUMP.glob("rmin_*"):
    print(f"[init] wiping stale dump dir: {d}")
    shutil.rmtree(d, ignore_errors=True)

if not ep.BIN.exists():
    raise SystemExit(f"binary not found: {ep.BIN} -- run `cd build && make` first")

all_rows = []
for r in RMIN_SWEEP:
    tag = f"rmin_{r:g}"
    flags = ["--min-ctrl-range", str(r)]
    rows = ep._rich_one_config(tag, flags)
    for row in rows:
        row["r_min"] = r
    all_rows.extend(rows)

path = EVAL / "results_rmin.csv"
ep.write_csv(all_rows, path)

df = pd.DataFrame(all_rows)
for c in df.columns:
    if c not in ("config",):
        df[c] = pd.to_numeric(df[c], errors="coerce")

rich_cols = [c for c in df.columns
             if c.startswith(("holdout_", "orb_", "ssim_", "lpips_", "t_total"))]
agg = df.groupby("r_min")[rich_cols].agg(["mean", "std"])
out = EVAL / "results_summary_rmin.csv"
agg.to_csv(out)
print(f"\nwrote {out}")
print("\n=== Per-r_min rich summary ===")
print(agg.to_string())
