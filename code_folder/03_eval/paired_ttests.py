#!/usr/bin/env python3
"""
Paired t-tests for the §4 ablation comparisons.

The thesis reports mean ± std per metric per configuration over 64
contiguous frames of the evaluated AV2 log. Several of the cross-
configuration deltas (DP vs feather temporal std, sym vs asym overlap
PSNR, gain vs no-gain SSIM) are smaller than the within-configuration
standard deviation, so a paired t-test on the per-frame paired samples
is the right way to ask whether the difference is meaningful.

Reads results_cheap.csv (64 frames x 5 configs); writes a CSV/text table
of paired-test p-values for the comparisons that drive the §4 claims.

Usage:
    python3 paired_ttests.py results_cheap.csv
"""

import argparse
import csv
import sys
from collections import defaultdict
from statistics import mean, stdev
from math import sqrt

# Use scipy if available; fall back to a manual paired t-test.
try:
    from scipy.stats import ttest_rel
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


def manual_paired_t(a, b):
    """Paired Student's t. Returns (t, p_two_sided)."""
    assert len(a) == len(b) and len(a) >= 2
    diffs = [ai - bi for ai, bi in zip(a, b)]
    n = len(diffs)
    md = mean(diffs)
    sd = stdev(diffs) if n > 1 else 0.0
    if sd == 0:
        return float("nan"), float("nan")
    t = md / (sd / sqrt(n))
    # Two-sided p-value using the survival function approximation
    # (Student's t with df=n-1). For df >= 30 the normal approximation
    # is within ~5% of the exact value; for the 64-frame thesis data
    # df=63, well in that regime.
    df = n - 1
    # Use scipy if it sneaks in via numpy; otherwise normal approximation.
    try:
        from math import erfc
        z = abs(t)
        # Normal approximation:  p = erfc(z / sqrt(2))
        p = erfc(z / sqrt(2.0))
    except Exception:
        p = float("nan")
    return t, p


def paired_t(a, b):
    if HAVE_SCIPY:
        res = ttest_rel(a, b)
        return float(res.statistic), float(res.pvalue)
    return manual_paired_t(a, b)


def load(path):
    """Returns dict[config][metric] = list of per-frame values, ordered by frame."""
    rows = list(csv.DictReader(open(path)))
    by_cfg = defaultdict(list)
    for r in rows:
        by_cfg[r["config"]].append(r)
    out = {}
    for cfg, frame_rows in by_cfg.items():
        frame_rows.sort(key=lambda r: int(r["frame"]))
        cols = {}
        for k in frame_rows[0]:
            if k in ("config", "frame"):
                continue
            try:
                cols[k] = [float(r[k]) for r in frame_rows]
            except ValueError:
                continue
        out[cfg] = cols
    return out


def report(label, a_label, b_label, data, metric, lower_is_better=True):
    """Run a paired t-test on data[a_label][metric] vs data[b_label][metric]."""
    a = data[a_label][metric]
    b = data[b_label][metric]
    if len(a) != len(b):
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
    t, p = paired_t(a, b)
    ma, mb = mean(a), mean(b)
    delta = ma - mb
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    direction = ("-> A better" if (delta < 0) == lower_is_better else "-> B better"
                 if (delta != 0) else "-> tie")
    print(f"  {label:35s}  {a_label:18s} {ma:9.3f}  vs  {b_label:18s} {mb:9.3f}  "
          f"Δ={delta:+8.3f}  t={t:+6.2f}  p={p:.4g} {sig}  {direction}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="Per-frame results CSV "
                                 "(e.g. results_cheap.csv)")
    args = ap.parse_args()

    data = load(args.csv)
    print(f"Loaded configs: {list(data.keys())}")
    print(f"  scipy available: {HAVE_SCIPY}\n")
    print("Significance thresholds: *** p<0.001, ** p<0.01, * p<0.05, ns otherwise.")
    print()

    print("--- DP vs Feather seam (RQ2) ------------------------------------")
    for m in ("seam_l1_FL_FC", "seam_l1_FC_FR",
              "seam_std_FL_FC", "seam_std_FC_FR"):
        report(m, "seamdp", "feather", data, m, lower_is_better=True)
    print()
    print("--- Gain compensation on/off (RQ2) ------------------------------")
    for m in ("seam_l1_FL_FC", "seam_l1_FC_FR",
              "overlap_psnr_y_FL_FC", "overlap_psnr_y_FC_FR"):
        report(m, "seamdp", "seamdp_nogain", data, m,
               lower_is_better=("psnr" not in m))
    print()
    print("--- Ctrl-pt seam attraction on/off ------------------------------")
    for m in ("seam_l1_FL_FC", "seam_l1_FC_FR",
              "seam_std_FL_FC", "seam_std_FC_FR"):
        report(m, "seamdp", "seamdp_ctrl", data, m, lower_is_better=True)


if __name__ == "__main__":
    main()
