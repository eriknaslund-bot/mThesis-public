#!/usr/bin/env python3
"""
Multi-log evaluation harness for the §4 thesis claims.

Runs `lidartps --benchmark N --metrics` with the recommended config
(deployed default + gain on + ctrl-attract hard, --disp-blur 0.5) across the
original thesis evaluation log plus the three additional AV2 logs prepped
by `prep_av2_logs.py`. Emits a single combined CSV with a `log` column,
plus a per-log summary CSV suitable for the §4 multi-log table.

Output:
    code_folder/03_eval/results_multilog.csv
    code_folder/03_eval/results_multilog_summary.csv

Usage:
    python3 code_folder/03_eval/multi_log_eval.py
"""

import csv
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev

REPO = Path(__file__).resolve().parents[2]
BIN  = REPO / "code_folder/02_implementations/lidar_tps_pipeline/build/lidartps"
OUT  = REPO / "code_folder/03_eval/results_multilog.csv"
SUMM = REPO / "code_folder/03_eval/results_multilog_summary.csv"

N_FRAMES = 64

LOGS = [
    # (log_tag, calib, frames, lidar_dir)
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


def run_log(tag, calib, frames, lidar):
    cmd = [
        str(BIN),
        "--calib",  calib,
        "--frames", frames,
        "--lidar",  lidar,
        "--benchmark", str(N_FRAMES),
        "--metrics",
        "--disp-blur", "0.5",
        "--config-tag", tag,
    ]
    print(f"[{tag}] {' '.join(cmd[1:])}")
    p = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True, check=False)
    if p.returncode != 0:
        print(p.stdout)
        print(p.stderr, file=sys.stderr)
        raise RuntimeError(f"lidartps failed (log={tag})")
    header = None
    rows = []
    for line in p.stdout.splitlines():
        if not line.startswith("METRIC,"):
            continue
        fields = line[len("METRIC,"):].split(",")
        if header is None:
            header = fields
        else:
            row = dict(zip(header, fields))
            row["log"] = tag
            rows.append(row)
    print(f"  -> {len(rows)} frames")
    return header, rows


def main():
    if not BIN.exists():
        raise SystemExit(f"binary not found: {BIN}")
    all_rows = []
    header  = None
    for tag, calib, frames, lidar in LOGS:
        h, rows = run_log(tag, calib, frames, lidar)
        if header is None:
            header = ["log"] + h
        all_rows.extend(rows)

    # Combined per-frame CSV
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nwrote {OUT} ({len(all_rows)} rows)")

    # Per-log mean ± std for the §4 multi-log table
    summary_cols = [
        "n_shared_FL_FC", "n_shared_FC_FR",
        "seam_l1_FL_FC", "seam_l1_FC_FR",
        "seam_std_FL_FC", "seam_std_FC_FR",
        "overlap_psnr_y_FL_FC", "overlap_psnr_y_FC_FR",
        "overlap_psnr_cb_FL_FC", "overlap_psnr_cb_FC_FR",
        "overlap_psnr_cr_FL_FC", "overlap_psnr_cr_FC_FR",
        "warp_p95_FL", "warp_p95_FC", "warp_p95_FR",
        "tps_bend_FL", "tps_bend_FC", "tps_bend_FR",
        "t_total",
    ]

    by_log = {tag: [] for tag, *_ in LOGS}
    for r in all_rows:
        by_log[r["log"]].append(r)

    with open(SUMM, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric"] + [tag for tag, *_ in LOGS] + ["mean_across_logs", "std_across_logs"])
        for col in summary_cols:
            per_log_means = []
            row = [col]
            for tag, *_ in LOGS:
                vals = [float(r[col]) for r in by_log[tag] if r[col] != ""]
                m = mean(vals) if vals else float("nan")
                s = stdev(vals) if len(vals) > 1 else 0.0
                row.append(f"{m:.3f} ± {s:.3f}")
                per_log_means.append(m)
            row.append(f"{mean(per_log_means):.3f}")
            row.append(f"{stdev(per_log_means) if len(per_log_means) > 1 else 0.0:.3f}")
            w.writerow(row)
    print(f"wrote {SUMM}")
    print()
    # Echo to stdout for quick eyeballing
    with open(SUMM) as f:
        print(f.read())


if __name__ == "__main__":
    main()
