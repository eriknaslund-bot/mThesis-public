#!/usr/bin/env python3
"""r_min sweep on the 3 extra AV2 logs (rich pass, mirrors
run_rmin_sweep.py on the primary log). Output combined with
results_rmin.csv gives 4-log coverage for the §4.7 ablation table.

Disk: rich pass dumps ~108 MB/frame to eval/dump/rmin_*__<log>/.
Sequential per log so peak disk stays at ~14 GB; multilog_* dump dirs
from earlier runs are untouched.

Output: eval/results_rmin_multilog.csv (rows tagged with r_min and log).
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

# Wipe only rmin_*__multilog_* dirs from prior runs (preserve other dumps).
for d in DUMP.glob("rmin_*__*"):
    print(f"[init] wiping stale dump dir: {d}")
    shutil.rmtree(d, ignore_errors=True)

if not ep.BIN.exists():
    raise SystemExit(f"binary not found: {ep.BIN} -- run `cd build && make` first")

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
    for r in RMIN_SWEEP:
        tag = f"rmin_{r:g}__{log_tag}"
        flags = ["--min-ctrl-range", str(r), *extra_log_flags]
        rows = ep._rich_one_config(tag, flags)
        for row in rows:
            row["r_min"] = r
            row["log"] = log_tag
        all_rows.extend(rows)
    # Free dump disk after each log finishes (we've extracted what we need).
    for d in DUMP.glob(f"rmin_*__{log_tag}"):
        print(f"[clean] {d}")
        shutil.rmtree(d, ignore_errors=True)

path = EVAL / "results_rmin_multilog.csv"
ep.write_csv(all_rows, path)
print(f"wrote {path} ({len(all_rows)} rows)")
