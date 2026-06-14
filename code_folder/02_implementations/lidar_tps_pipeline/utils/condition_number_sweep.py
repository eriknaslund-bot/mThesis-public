#!/usr/bin/env python3
"""κ(K) measurement across the M sweep from §4.7.

The thesis claims: at M=1000 the TPS solve at λ=0 becomes ill-conditioned
because the biharmonic kernel U(r) = r² log r² → 0 as r → 0; dense ctrl-pt
sampling drives some K_ij toward zero and Gaussian elimination returns large
weights that extrapolate to canvas-scale displacements.

This script measures the condition number κ(K) directly across the M sweep
to back the claim. Uses the same spatial-grid subsample as the pipeline
(Eq. 3.7), applied to a representative frame's shared LiDAR points loaded
from the primary-log rich dump.

Output: eval/results_condition_M.csv (M, n_ctrl, min_pair_sep, kappa_K).
"""
import csv
import sys
from pathlib import Path
import numpy as np
import pandas as pd

THIS = Path(__file__).resolve()
PIPE = THIS.parent.parent
EVAL = PIPE / "eval"
EVAL.mkdir(exist_ok=True)

# -- Same M sweep as pass4_m_sweep in eval_pipeline.py -----------------------
M_SWEEP = [10, 50, 100, 200, 400, 600, 1000, 1500, 2000]

# -- Biharmonic TPS kernel (per §2.6 Eq 2.27) --------------------------------
def U(r2):
    """U(r) = r² log r² with U(0)=0; takes squared distance directly."""
    out = np.zeros_like(r2)
    nz = r2 > 0
    out[nz] = r2[nz] * np.log(r2[nz])
    return out


def build_K(pts):
    """K_ij = U(||p_i - p_j||) for N×2 pts in [0,1]²."""
    diff = pts[:, None, :] - pts[None, :, :]
    r2 = (diff ** 2).sum(-1)
    return U(r2)


# -- Spatial-grid subsample (mirrors §3.4 Eq 3.7, max one pt per cell) --------
def subsample(pts, M):
    """Aspect-ratio-aware occupancy grid: keep ≤M points, one per cell
    (nearest cell-centre, with ties broken by lower index)."""
    if len(pts) <= M:
        return pts
    u_min, v_min = pts.min(0)
    u_max, v_max = pts.max(0)
    u_range = max(u_max - u_min, 1e-9)
    v_range = max(v_max - v_min, 1e-9)
    n_cols = int(np.ceil(np.sqrt(M * u_range / v_range)))
    n_rows = int(np.ceil(M / n_cols))
    cell_w = u_range / n_cols
    cell_h = v_range / n_rows

    # For each pt, compute (cell_row, cell_col) and distance to cell centre.
    col_idx = np.clip(((pts[:, 0] - u_min) / cell_w).astype(int), 0, n_cols - 1)
    row_idx = np.clip(((pts[:, 1] - v_min) / cell_h).astype(int), 0, n_rows - 1)
    cell_cu = u_min + (col_idx + 0.5) * cell_w
    cell_cv = v_min + (row_idx + 0.5) * cell_h
    d2 = (pts[:, 0] - cell_cu) ** 2 + (pts[:, 1] - cell_cv) ** 2

    # Keep nearest-to-centre per cell (tie-break: lower input index).
    cell_id = row_idx * n_cols + col_idx
    order = np.lexsort((np.arange(len(pts)), d2))    # primary: d2 asc; secondary: index asc
    seen = set()
    keep = []
    for i in order:
        cid = int(cell_id[i])
        if cid not in seen:
            seen.add(cid)
            keep.append(i)
    return pts[np.array(sorted(keep))]


# -- Load representative frame's shared canvas-coords ------------------------
# The rich dump's lidar_holdout.csv contains held-out points (20% of shared).
# For κ measurement, the spatial distribution is what matters, so the
# held-out subset is a faithful sample for grid subsampling.
def load_frame_canvas_pts(frame_dir):
    csv_path = frame_dir / "lidar_holdout.csv"
    if not csv_path.exists():
        raise SystemExit(f"missing {csv_path}; run pass2_rich first")
    df = pd.read_csv(csv_path)
    # Use canvas coords from the FC camera (cu_FC, cv_FC); same source for
    # both overlap 0 (FL/FC) and overlap 1 (FC/FR).
    lc = df[df.overlap == 0][["cu_FC", "cv_FC"]].values
    cr = df[df.overlap == 1][["cu_FC", "cv_FC"]].values
    return lc, cr


def normalise_to_unit_box(pts):
    """Normalise to [0,1]² by canvas bounding box (same as the pipeline does
    before the TPS solve per §3.5)."""
    if len(pts) == 0:
        return pts
    pmin = pts.min(0)
    pmax = pts.max(0)
    span = np.maximum(pmax - pmin, 1e-9)
    return (pts - pmin) / span


def sweep(pts_all, label):
    rows = []
    pts_norm = normalise_to_unit_box(pts_all)
    print(f"\n=== {label}: {len(pts_norm)} shared points (input) ===")
    print(f"{'M':>6} {'n_ctrl':>7} {'min_sep':>10} {'kappa(K)':>12}")
    for M in M_SWEEP:
        kept = subsample(pts_norm, M)
        n = len(kept)
        # Min pair separation
        diff = kept[:, None, :] - kept[None, :, :]
        r = np.sqrt((diff ** 2).sum(-1))
        np.fill_diagonal(r, np.inf)
        min_sep = float(r.min())
        # κ(K)
        K = build_K(kept)
        try:
            kappa = float(np.linalg.cond(K))
        except np.linalg.LinAlgError:
            kappa = float("inf")
        print(f"{M:>6} {n:>7} {min_sep:>10.4e} {kappa:>12.4e}")
        rows.append(dict(overlap=label, M=M, n_ctrl=n,
                         min_pair_sep=min_sep, kappa_K=kappa))
    return rows


def synthetic_uniform(n, aspect=2.5, rng=None):
    """N points uniformly on [0, aspect] × [0, 1] to match the typical
    FL/FC overlap bounding-box aspect on the primary canvas. Per-frame
    average shared count is ~2400-2600, so N=2500 is representative."""
    if rng is None:
        rng = np.random.default_rng(0xC0FFEE)
    return np.column_stack([
        rng.uniform(0, aspect, n),
        rng.uniform(0, 1, n),
    ])


def main():
    dump_root = EVAL / "dump" / "seamdp"
    # Real-frame pass: one representative frame (held-out 20% subset).
    # Saturates at the per-frame shared-pool size, useful as a sanity check.
    frame_dir = dump_root / "frame_0000"
    lc, cr = load_frame_canvas_pts(frame_dir)
    rows  = sweep(lc, "FL_FC_frame0")
    rows += sweep(cr, "FC_FR_frame0")

    # Synthetic pass: uniform pts at the typical per-frame shared density
    # (~2500). Engages every M in the sweep including M=1000, so the
    # ill-conditioning at the M=1000 thesis failure point can be measured
    # without the per-frame saturation observed above.
    rows += sweep(synthetic_uniform(2500), "synth_uniform_2500")

    out = EVAL / "results_condition_M.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
