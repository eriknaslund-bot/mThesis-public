#!/usr/bin/env python3
"""One-shot driver for pass6b_rich_multilog.

Runs the rich-metrics pass (held-out reprojection, ORB residual, SSIM, LPIPS)
on the 3 extra AV2 logs and writes results_rich_multilog.csv and
results_summary_rich_multilog.csv to eval/.

Disk: ~108 MB per frame x N_FRAMES_RICH x len(MULTILOG_DIRS) of scratch under
eval/dump/multilog_*/. Stale multilog dump dirs are wiped at start; primary-log
pass2 dumps (eval/dump/seamdp*/) are left alone.
"""
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_pipeline as ep

EVAL = ep.EVAL
DUMP = EVAL / "dump"

# Wipe only multilog_* dump dirs from any prior run.
for d in DUMP.glob("multilog_*"):
    print(f"[init] wiping stale dump dir: {d}")
    shutil.rmtree(d, ignore_errors=True)

if not ep.BIN.exists():
    raise SystemExit(f"binary not found: {ep.BIN} -- run `cd build && make` first")

rows = ep.pass6b_rich_multilog()
if not rows:
    raise SystemExit("no rows produced -- check log dirs / binary output")

df = pd.DataFrame(rows)
for c in df.columns:
    if c not in ("config", "log"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Per-log aggregates over the same rich-metric columns _rich_one_config emits.
rich_cols = [c for c in df.columns
             if c.startswith(("holdout_", "orb_", "ssim_", "lpips_", "t_total"))]
agg = df.groupby("log")[rich_cols].agg(["mean", "std"])
out = EVAL / "results_summary_rich_multilog.csv"
agg.to_csv(out)
print(f"\nwrote {out}")
print("\n=== Per-log rich summary ===")
print(agg.to_string())
