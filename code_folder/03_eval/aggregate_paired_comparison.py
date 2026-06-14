#!/usr/bin/env python3
"""Aggregate the paired baseline-vs-LiDAR-TPS comparison (§4.3 Tables 4.5/4.6).

For each baseline (SIFT+RANSAC, LiDAR-homography) and each AV2 log, this pairs
the baseline's per-frame quality against the deployed LiDAR-TPS pipeline on
exactly the frames where the baseline produced a geometrically valid stitch,
then averages per log.

Quality is the channel-weighted PSNR of thesis Eq. 3.6,

    PSNR_w = wY*PSNR_Y + wCb*PSNR_Cb + wCr*PSNR_Cr,

with weights set to the dataset-mean per-channel signal-energy share. The
baseline side comes from the re-run per-log CSVs (which now emit Y/Cb/Cr PSNR
in the pipeline's BT.709 space); the LiDAR-TPS side comes from the existing
cheap-pass per-frame CSVs (results_multilog.csv for the nine extra logs,
results_cheap.csv config `seamdp` for the primary), restricted to the same
64-frame window and the baseline's valid frames.

Outputs (overwritten):
    sift_baseline_10log_paired.csv
    lidar_homography_baseline_10log_paired.csv

Columns: log, n_paired, baseline_psnr_FL_FC, baseline_psnr_FC_FR,
         tps_psnr_FL_FC, tps_psnr_FC_FR, n_attempted, dropout_rate
(all PSNR columns are weighted PSNR; dropout = 1 - n_paired/n_attempted with
n_attempted = the 64-frame window).

Usage:
    python3 aggregate_paired_comparison.py
"""

import csv
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EVAL = REPO / "code_folder/02_implementations/lidar_tps_pipeline/eval"
B    = REPO / "code_folder/03_eval"

# Channel-energy weights (dataset-mean BT.709 variance share, 10 logs).
WY, WCB, WCR = 0.918, 0.047, 0.035

NWIN = 64  # paired window size per log

LOGS = ["00a6ffc1", "01bb304d", "022af476", "04973bcf", "05853f69",
        "072c8e90", "087695bd", "0a524e66", "12071817", "12c3c14b"]


def wp(y, cb, cr):
    return WY * y + WCB * cb + WCR * cr


def load_tps_per_frame():
    """{log: {frame: (wpsnr_FL_FC, wpsnr_FC_FR)}} for the LiDAR-TPS pipeline."""
    d = defaultdict(dict)

    def row_wp(r, ov):
        return wp(float(r[f"overlap_psnr_y_{ov}"]),
                  float(r[f"overlap_psnr_cb_{ov}"]),
                  float(r[f"overlap_psnr_cr_{ov}"]))

    with open(EVAL / "results_multilog.csv") as f:
        for r in csv.DictReader(f):
            try:
                fr = int(r["frame"])
                d[r["log"]][fr] = (row_wp(r, "FL_FC"), row_wp(r, "FC_FR"))
            except (KeyError, ValueError):
                continue
    # primary 00a6ffc1: cheap-pass deployed config `seamdp`
    with open(EVAL / "results_cheap.csv") as f:
        for r in csv.DictReader(f):
            if r.get("config") != "seamdp":
                continue
            try:
                fr = int(r["frame"])
                d["00a6ffc1"][fr] = (row_wp(r, "FL_FC"), row_wp(r, "FC_FR"))
            except (KeyError, ValueError):
                continue
    return d


def load_baseline(kind, tag):
    """{frame: (wpsnr_FL_FC, wpsnr_FC_FR)} for valid baseline stitches."""
    stem = "sift_baseline" if kind == "sift" else "lidar_homography_baseline"
    path = B / f"{stem}_{tag}.csv"
    out = {}
    try:
        with open(path) as f:
            for r in csv.DictReader(f):
                try:
                    fr = int(r["frame_idx"])
                    vals = [float(r[c]) for c in (
                        "overlap_psnr_y_FL_FC", "overlap_psnr_y_FC_FR",
                        "overlap_psnr_cb_FL_FC", "overlap_psnr_cb_FC_FR",
                        "overlap_psnr_cr_FL_FC", "overlap_psnr_cr_FC_FR")]
                except (KeyError, ValueError):
                    continue
                if any(math.isnan(v) or math.isinf(v) for v in vals):
                    continue
                y1, y2, cb1, cb2, cr1, cr2 = vals
                out[fr] = (wp(y1, cb1, cr1), wp(y2, cb2, cr2))
    except FileNotFoundError:
        return None
    return out


def aggregate(kind, tps):
    rows = []
    for tag in LOGS:
        bl = load_baseline(kind, tag)
        if bl is None:
            print(f"  [warn] missing {kind} CSV for {tag}")
            continue
        paired = sorted(fr for fr in bl if fr < NWIN and fr in tps.get(tag, {}))
        if not paired:
            continue
        n = len(paired)
        rows.append({
            "log": tag,
            "n_paired": n,
            "baseline_psnr_FL_FC": st.mean(bl[fr][0] for fr in paired),
            "baseline_psnr_FC_FR": st.mean(bl[fr][1] for fr in paired),
            "tps_psnr_FL_FC": st.mean(tps[tag][fr][0] for fr in paired),
            "tps_psnr_FC_FR": st.mean(tps[tag][fr][1] for fr in paired),
            "n_attempted": NWIN,
            "dropout_rate": 1.0 - n / NWIN,
        })
    return rows


def write_csv(path, rows):
    fields = ["log", "n_paired", "baseline_psnr_FL_FC", "baseline_psnr_FC_FR",
              "tps_psnr_FL_FC", "tps_psnr_FC_FR", "n_attempted", "dropout_rate"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def summary(kind, rows):
    def ms(key):
        v = [r[key] for r in rows]
        return st.mean(v), st.pstdev(v)
    tot = sum(r["n_paired"] for r in rows)
    drop = st.mean(r["dropout_rate"] for r in rows)
    print(f"=== {kind} ({len(rows)} logs, {tot} paired frames, "
          f"mean dropout {drop*100:.1f}%) ===")
    print(f"  baseline  FL-FC {ms('baseline_psnr_FL_FC')[0]:.2f}+/-{ms('baseline_psnr_FL_FC')[1]:.2f}"
          f"   FC-FR {ms('baseline_psnr_FC_FR')[0]:.2f}+/-{ms('baseline_psnr_FC_FR')[1]:.2f}")
    print(f"  LiDAR-TPS FL-FC {ms('tps_psnr_FL_FC')[0]:.2f}+/-{ms('tps_psnr_FL_FC')[1]:.2f}"
          f"   FC-FR {ms('tps_psnr_FC_FR')[0]:.2f}+/-{ms('tps_psnr_FC_FR')[1]:.2f}")


def main():
    tps = load_tps_per_frame()
    for kind, out in (("sift", "sift_baseline_10log_paired.csv"),
                      ("homog", "lidar_homography_baseline_10log_paired.csv")):
        rows = aggregate(kind, tps)
        write_csv(B / out, rows)
        summary(kind, rows)
        print(f"  wrote {out}\n")


if __name__ == "__main__":
    main()
