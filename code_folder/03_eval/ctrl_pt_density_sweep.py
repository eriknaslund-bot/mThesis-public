#!/usr/bin/env python3
"""
Sweep --max-ctrl-per-overlap to characterise TPS quality vs control-point
density on the deployed pipeline.

Output:
  code_folder/03_eval/ctrl_pt_density.csv  (per-frame rows, per N)
  Stdout summary table for the thesis §4 sensitivity ablation.

Run from repo root:
  python3 code_folder/03_eval/ctrl_pt_density_sweep.py
"""

import csv
import statistics as st
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
BIN = REPO / "code_folder" / "02_implementations" / "lidar_tps_pipeline" / "build" / "lidartps"
OUT_CSV = REPO / "code_folder" / "03_eval" / "ctrl_pt_density.csv"

N_FRAMES = 16
SWEEP = [50, 100, 200, 400, 600]


def run_one(n_pts):
    cmd = [
        str(BIN),
        "--benchmark", str(N_FRAMES),
        "--metrics",
        "--disp-blur", "0.5",
        "--max-ctrl-per-overlap", str(n_pts),
        "--config-tag", f"M{n_pts}",
    ]
    print(f"[sweep] M={n_pts}: {' '.join(cmd[1:])}")
    p = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True, check=False)
    if p.returncode != 0:
        print(p.stdout)
        print(p.stderr, file=sys.stderr)
        raise RuntimeError(f"lidartps failed (M={n_pts})")
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


def main():
    if not BIN.exists():
        sys.exit(f"binary not found: {BIN} -- run `cd build && make` first")
    all_rows = []
    for n in SWEEP:
        rows = run_one(n)
        for r in rows:
            r["max_ctrl_per_overlap"] = str(n)
        all_rows.extend(rows)
    if not all_rows:
        sys.exit("no rows captured")

    fieldnames = ["max_ctrl_per_overlap"] + [
        k for k in all_rows[0].keys() if k != "max_ctrl_per_overlap"
    ]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f"\nwrote {OUT_CSV} ({len(all_rows)} rows)")

    # Summary
    print("\n" + "=" * 92)
    print(f"Control-point density sweep ({N_FRAMES} frames per config)")
    print("=" * 92)
    cols = [
        ("seam_l1_FL_FC",         "seam-L1 FL-FC"),
        ("seam_l1_FC_FR",         "seam-L1 FC-FR"),
        ("overlap_psnr_y_FL_FC",  "Y-PSNR FL-FC"),
        ("overlap_psnr_y_FC_FR",  "Y-PSNR FC-FR"),
        ("t_tps",                 "t_tps   (ms)"),
        ("t_total",               "t_total (ms)"),
    ]
    hdr = f"{'M':>5}  " + "  ".join(f"{l:>16}" for _, l in cols)
    print(hdr)
    for n in SWEEP:
        sub = [r for r in all_rows if r["max_ctrl_per_overlap"] == str(n)]
        cells = []
        for c, _ in cols:
            v = [float(r[c]) for r in sub]
            cells.append(f"{st.mean(v):7.2f} ± {st.pstdev(v):4.2f}")
        print(f"{n:>5}  " + "  ".join(f"{c:>16}" for c in cells))


if __name__ == "__main__":
    main()
