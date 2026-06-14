#!/usr/bin/env python3
"""Aggregate the multilog ablation CSVs into 4-log summaries for thesis tables.

For each ablation (λ, M, r_min, gain, headline geom/photo), pool the primary
log's rows with the 3 extra logs' rows, group by the sweep parameter, and
report mean ± std with each log weighted equally (mean of per-log means,
std as across-log std). This is the statistic the thesis tables now lead with.

Per-log breakdowns remain in eval/results_*_multilog.csv files for the §4.8
supplementary tables.

Output: prints aggregated tables to stdout, also writes
eval/results_summary_4log_*.csv for each ablation.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

THIS = Path(__file__).resolve()
EVAL = THIS.parent.parent / "eval"


def four_log_agg(primary_csv, multilog_csv, sweep_key, value_cols, label):
    """Return DataFrame of (sweep_key, value_col_mean, value_col_std) over 4 logs.

    primary_csv: rows from the primary-log sweep (no `log` column; rows are
                 318 frames per sweep_value, all on primary)
    multilog_csv: rows from the multilog sweep (with `log` column; 318 frames
                  per (sweep_value, log) combination)
    """
    prim = pd.read_csv(primary_csv)
    prim["log"] = "00a6ffc1"
    if multilog_csv and Path(multilog_csv).exists():
        multi = pd.read_csv(multilog_csv)
        all_df = pd.concat([prim, multi], ignore_index=True)
    else:
        all_df = prim
        print(f"  [warn] multilog CSV missing: {multilog_csv}; primary only")

    for c in value_cols:
        if c in all_df.columns:
            all_df[c] = pd.to_numeric(all_df[c], errors="coerce")

    # Per-(sweep, log) mean over frames
    per_log = all_df.groupby([sweep_key, "log"])[value_cols].mean().reset_index()
    # Cross-log mean and std (equal log weight)
    agg = per_log.groupby(sweep_key)[value_cols].agg(["mean", "std"])
    print(f"\n=== {label}: 4-log aggregate ({all_df['log'].nunique()} logs) ===")
    print(agg.to_string())
    out = EVAL / f"results_summary_4log_{label}.csv"
    agg.to_csv(out)
    print(f"wrote {out}")
    return agg


def main():
    # -- λ sweep ----------------------------------------------------------------
    four_log_agg(
        EVAL / "results_lambda.csv",
        EVAL / "results_lambda_multilog.csv",
        sweep_key="lambda",
        value_cols=[
            "overlap_psnr_y_FL_FC", "overlap_psnr_y_FC_FR",
            "tps_bend_FL", "tps_bend_FC", "tps_bend_FR",
        ],
        label="lambda",
    )

    # -- M sweep ----------------------------------------------------------------
    four_log_agg(
        EVAL / "results_m.csv",
        EVAL / "results_m_multilog.csv",
        sweep_key="M",
        value_cols=[
            "overlap_psnr_y_FL_FC", "overlap_psnr_y_FC_FR",
            "t_total",
        ],
        label="M",
    )

    # -- Gain ablation (cheap pass) --------------------------------------------
    # Both on/off configs need 4-log coverage. The deployed (gain-on) config
    # is in results_multilog.csv (3 extras) + results_cheap.csv (primary).
    # The no-gain config is in results_gain_multilog.csv (3 extras) +
    # results_cheap.csv (primary, config=seamdp_nogain).
    on_prim = pd.read_csv(EVAL / "results_cheap.csv")
    on_prim = on_prim[on_prim.config == "seamdp"].copy()
    on_prim["log"] = "00a6ffc1"; on_prim["gain"] = "on"
    on_multi = pd.read_csv(EVAL / "results_multilog.csv").copy()
    on_multi["gain"] = "on"
    off_prim = pd.read_csv(EVAL / "results_cheap.csv")
    off_prim = off_prim[off_prim.config == "seamdp_nogain"].copy()
    off_prim["log"] = "00a6ffc1"; off_prim["gain"] = "off"
    off_path = EVAL / "results_gain_multilog.csv"
    if off_path.exists():
        off_multi = pd.read_csv(off_path)
        off_multi["gain"] = "off"
    else:
        off_multi = pd.DataFrame()
    gain_df = pd.concat([on_prim, on_multi, off_prim, off_multi], ignore_index=True)
    psnr_cols = [
        "overlap_psnr_y_FL_FC", "overlap_psnr_y_FC_FR",
        "overlap_psnr_cb_FL_FC", "overlap_psnr_cb_FC_FR",
        "overlap_psnr_cr_FL_FC", "overlap_psnr_cr_FC_FR",
    ]
    for c in psnr_cols:
        if c in gain_df.columns:
            gain_df[c] = pd.to_numeric(gain_df[c], errors="coerce")
    per_log_gain = gain_df.groupby(["gain", "log"])[psnr_cols].mean().reset_index()
    agg_gain = per_log_gain.groupby("gain")[psnr_cols].agg(["mean", "std"])
    print(f"\n=== gain: 4-log aggregate ({gain_df['log'].nunique()} logs) ===")
    print(agg_gain.to_string())
    agg_gain.to_csv(EVAL / "results_summary_4log_gain.csv")
    print(f"wrote {EVAL / 'results_summary_4log_gain.csv'}")

    # -- r_min sweep (rich pass) -----------------------------------------------
    four_log_agg(
        EVAL / "results_rmin.csv",
        EVAL / "results_rmin_multilog.csv",
        sweep_key="r_min",
        value_cols=[
            "holdout_tps_mean_FL_FC", "holdout_tps_mean_FC_FR",
            "orb_med_FL_FC", "orb_med_FC_FR",
            "ssim_y_FL_FC", "ssim_y_FC_FR",
            "t_total_ms",
        ],
        label="rmin",
    )

    # -- Headline geometric (held-out + ORB across 4 logs) ---------------------
    # results_rich.csv: primary log (config=seamdp), 32 frames
    # results_rich_multilog.csv: 3 extras (config=multilog_*), 32 frames each
    rich_prim = pd.read_csv(EVAL / "results_rich.csv")
    rich_prim = rich_prim[rich_prim.config == "seamdp"].copy()
    rich_prim["log"] = "00a6ffc1"
    rich_multi = pd.read_csv(EVAL / "results_rich_multilog.csv")
    rich_all = pd.concat([rich_prim, rich_multi], ignore_index=True)
    cols = ["holdout_rot_mean_FL_FC", "holdout_tps_mean_FL_FC",
            "holdout_rot_mean_FC_FR", "holdout_tps_mean_FC_FR",
            "orb_med_FL_FC", "orb_med_FC_FR",
            "ssim_y_FL_FC", "ssim_y_FC_FR"]
    for c in cols:
        rich_all[c] = pd.to_numeric(rich_all[c], errors="coerce")
    per_log_geom = rich_all.groupby("log")[cols].mean()
    agg_geom = per_log_geom.agg(["mean", "std"]).T
    print(f"\n=== headline geom (held-out + ORB + SSIM): 4-log aggregate ===")
    print(agg_geom.to_string())
    agg_geom.to_csv(EVAL / "results_summary_4log_headline.csv")
    print(f"wrote {EVAL / 'results_summary_4log_headline.csv'}")


if __name__ == "__main__":
    main()
