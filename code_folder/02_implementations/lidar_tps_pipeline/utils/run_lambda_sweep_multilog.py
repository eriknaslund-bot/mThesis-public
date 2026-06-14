#!/usr/bin/env python3
"""λ sweep on the 3 extra AV2 logs (cheap pass, mirrors pass3_lambda_sweep
on the primary log). Output combined with the existing results_lambda.csv
gives 4-log coverage for the §4.5 ablation table.

Output: eval/results_lambda_multilog.csv (rows tagged with both λ and log).
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_pipeline as ep

EVAL = ep.EVAL

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
    for lam in ep.LAMBDA_SWEEP:
        tag = f"lambda_{lam:g}__{log_tag}"
        rows = ep.run_metrics(
            tag,
            ["--tps-smooth", f"{lam:g}", *extra_log_flags],
            ep.N_FRAMES_SWEEP,
        )
        for r in rows:
            r["lambda"] = f"{lam:g}"
            r["log"] = log_tag
        print(f"  log={log_tag} λ={lam:g}: {len(rows)} frames")
        all_rows.extend(rows)

path = EVAL / "results_lambda_multilog.csv"
ep.write_csv(all_rows, path)
print(f"wrote {path} ({len(all_rows)} rows)")
