"""
Build defense-quality debug visuals for the lidar_tps_pipeline.

Inputs per frame:
  - <repo>/output/lidar_ring_stitch/debug/frame_NNNN/ (from --debug)
      01_source_*, 01b_lidar_all_*, 01c_ctrl_src_*,
      02_warped_*, 02b_rotation_*, 03_ctrl_pts_*,
      04_seam_cost_{lc,cr}, 05_seam_paths,
      06_disp_{x,y}_*.{jpg,bin}, 06_disp_meta.txt
  - eval-dump dir (gain on):  <root>/frame_NNNN/
      warped_*.png, mask_*.png, seam_*.bin, lidar_holdout.csv
  - eval-dump dir (no gain):  <root>/frame_NNNN/  (only used for visual D)

Outputs (per frame, in <debug>/visuals/):
  A_disp_field.png         disp-field quiver + magnitude heatmap (3 cams)
  B_reproj_scatter.png     held-out LiDAR residuals on canvas overlap
  C_seam_cost_path.png     seam DP cost map + routed seam, both overlaps
  D_gain_compare.png       gain-on vs gain-off seam strip
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "figure.dpi": 200,
})


# ----------------------------- IO helpers ---------------------------------

def read_disp_meta(meta_path: Path) -> dict:
    out = {}
    for line in meta_path.read_text().splitlines():
        k, v = line.split()
        out[k] = int(v)
    return out


def read_disp_bin(bin_path: Path, W_half: int, H_half: int) -> np.ndarray:
    a = np.fromfile(bin_path, dtype=np.float32)
    return a.reshape((H_half, W_half))


def read_bgr(p: Path) -> np.ndarray:
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(p)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_seam(bin_path: Path) -> np.ndarray:
    return np.fromfile(bin_path, dtype=np.int32)


# ----------------------------- Visual A -----------------------------------

def visual_A_disp_field(debug_dir: Path, out_path: Path,
                        eval_frame_dir: Path | None = None,
                        only: str | None = None):
    meta_path = debug_dir / "06_disp_meta.txt"
    if not meta_path.exists():
        print("  skipping A (no 06_disp_meta.txt — run with --eval-dump to generate bins)")
        return
    meta = read_disp_meta(meta_path)
    Wh, Hh = meta["W_half"], meta["H_half"]
    W, H = meta["W"], meta["H"]
    half_scale_x = Wh / W
    half_scale_y = Hh / H

    # Per-camera LiDAR ctrl-pt canvas positions (held-out sample is a
    # representative slice of the fit-set distribution; their convex hull
    # bounds the region where the TPS is actually fitted).
    hulls = {"FL": None, "FR": None}
    if eval_frame_dir is not None and (eval_frame_dir / "lidar_holdout.csv").exists():
        csv = np.genfromtxt(eval_frame_dir / "lidar_holdout.csv",
                            delimiter=",", names=True)
        for cam, ovl in [("FL", 0), ("FR", 1)]:
            rows = csv[csv["overlap"] == ovl]
            cu = rows[f"cu_{cam}"]
            cv = rows[f"cv_{cam}"]
            ok = (cu > 0) & (cv > 0)
            pts_h = np.column_stack([cu[ok] * half_scale_x,
                                     cv[ok] * half_scale_y]).astype(np.float32)
            if len(pts_h) >= 3:
                hull = cv2.convexHull(pts_h)
                hulls[cam] = hull

    if only in ("FL", "FR"):
        cams = [only]
        fig, ax_single = plt.subplots(1, 1, figsize=(8, 6))
        axes = [ax_single]
    else:
        cams = ["FL", "FR"]
        fig, axes = plt.subplots(2, 1, figsize=(11, 9))
        fig.suptitle("TPS-deformed grid inside the LiDAR ctrl-pt hull "
                     "(displacement shown at 0.5×)",
                     y=0.995, fontsize=10)

    # Grid resolution (in canvas grid cells across the half-res width)
    NCOLS = 22
    NROWS = 10

    for ax, cam in zip(axes, cams):
        dx = read_disp_bin(debug_dir / f"06_disp_x_{cam}.bin", Wh, Hh)
        dy = read_disp_bin(debug_dir / f"06_disp_y_{cam}.bin", Wh, Hh)

        rot = cv2.imread(str(debug_dir / f"02b_rotation_{cam}.jpg"),
                         cv2.IMREAD_GRAYSCALE)
        if rot is None:
            mask_h = np.ones((Hh, Wh), dtype=bool)
        else:
            mask_full = (rot > 0)
            mask_h = cv2.resize(mask_full.astype(np.uint8), (Wh, Hh),
                                interpolation=cv2.INTER_NEAREST).astype(bool)
        # Restrict to the LiDAR ctrl-pt convex hull when available: outside
        # the hull the TPS is extrapolating and the pipeline doesn't rely on
        # those pixels (DP seam + cosine feather mask them off).
        if hulls.get(cam) is not None:
            hull_mask = np.zeros((Hh, Wh), dtype=np.uint8)
            cv2.fillConvexPoly(hull_mask,
                               hulls[cam].astype(np.int32).reshape(-1, 2), 255)
            mask_h = mask_h & (hull_mask > 0)

        # Restrict the grid to the bounding box of the camera FOV so the
        # plot is tight and the regular grid lines up with the warped grid.
        ys_in, xs_in = np.where(mask_h)
        if len(ys_in) == 0:
            ax.set_xticks([]); ax.set_yticks([])
            continue
        x0, x1 = int(xs_in.min()), int(xs_in.max())
        y0, y1 = int(ys_in.min()), int(ys_in.max())

        # Build a regular grid of (NROWS+1, NCOLS+1) intersections inside
        # this bounding box.
        gx = np.linspace(x0, x1, NCOLS + 1)
        gy = np.linspace(y0, y1, NROWS + 1)
        GX, GY = np.meshgrid(gx, gy)

        # Sample the disp field at each grid intersection (nearest neighbour
        # since the disp arrays are already smooth at sub-canvas resolution).
        ix = np.clip(np.round(GX).astype(int), 0, Wh - 1)
        iy = np.clip(np.round(GY).astype(int), 0, Hh - 1)
        DX = dx[iy, ix]
        DY = dy[iy, ix]
        in_fov_grid = mask_h[iy, ix]

        DISP_SCALE = 0.5
        WX = GX + DISP_SCALE * DX
        WY = GY + DISP_SCALE * DY

        # Suppress grid vertices whose disp magnitude is in the extrapolation
        # tail: cap at the p90 of the in-FOV magnitudes (after erosion).
        mag_grid = np.hypot(DX, DY)
        if in_fov_grid.any():
            cap = float(np.percentile(mag_grid[in_fov_grid], 90))
        else:
            cap = 0.0
        sane = mag_grid <= max(10.0, 1.1 * cap)
        ok = in_fov_grid & sane

        # Native grid spacing (canvas px between neighbouring intersections)
        step_x = (x1 - x0) / NCOLS
        step_y = (y1 - y0) / NROWS
        edge_cap = 2.2  # drop edges that stretch more than 2.2x the native spacing

        def draw_grid(X, Y, valid, color, lw, alpha, suppress_long=False):
            segs = []
            for r in range(X.shape[0]):
                for c in range(X.shape[1] - 1):
                    if not (valid[r, c] and valid[r, c + 1]):
                        continue
                    if suppress_long:
                        d = abs(X[r, c + 1] - X[r, c])
                        if d > edge_cap * step_x:
                            continue
                    segs.append([[X[r, c], Y[r, c]],
                                 [X[r, c + 1], Y[r, c + 1]]])
            for c in range(X.shape[1]):
                for r in range(X.shape[0] - 1):
                    if not (valid[r, c] and valid[r + 1, c]):
                        continue
                    if suppress_long:
                        d = abs(Y[r + 1, c] - Y[r, c])
                        if d > edge_cap * step_y:
                            continue
                    segs.append([[X[r, c], Y[r, c]],
                                 [X[r + 1, c], Y[r + 1, c]]])
            lc = LineCollection(segs, colors=color, linewidth=lw,
                                alpha=alpha)
            ax.add_collection(lc)

        ax.set_facecolor("white")
        ax.set_aspect("equal")
        pad = 0.04 * max(x1 - x0, y1 - y0)
        ax.set_xlim(x0 - pad, x1 + pad)
        ax.set_ylim(y1 + pad, y0 - pad)

        draw_grid(GX, GY, ok, color="0.72", lw=0.6, alpha=0.9,
                  suppress_long=False)
        draw_grid(WX, WY, ok, color="black", lw=0.9, alpha=0.95,
                  suppress_long=True)

        if only is None:
            ax.set_title(cam, loc="left", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(out_path.parents[2])}")


# ----------------------------- Visual B -----------------------------------

def visual_B_reproj_scatter(eval_frame_dir: Path, out_path: Path):
    csv = np.genfromtxt(eval_frame_dir / "lidar_holdout.csv",
                        delimiter=",", names=True)
    # Per overlap: 0 = FL-FC, 1 = FC-FR
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2),
                             gridspec_kw={"width_ratios": [1, 1]})
    fig.suptitle("Held-out LiDAR residuals after TPS warp", y=1.00)

    for ax, ovl, label, c1, c2, A, B in [
        (axes[0], 0, "FL$\\leftrightarrow$FC", "tab:blue", "tab:orange",
         ("cu_tps_FL", "cv_tps_FL"), ("cu_tps_FC", "cv_tps_FC")),
        (axes[1], 1, "FC$\\leftrightarrow$FR", "tab:orange", "tab:green",
         ("cu_tps_FC", "cv_tps_FC"), ("cu_tps_FR", "cv_tps_FR")),
    ]:
        rows = csv[csv["overlap"] == ovl]
        if len(rows) == 0:
            ax.set_title(f"{label}: no holdout points")
            continue
        ax = ax
        u1, v1 = rows[A[0]], rows[A[1]]
        u2, v2 = rows[B[0]], rows[B[1]]
        # filter rows that have valid projections in both cameras (-1 sentinel)
        ok = (u1 > 0) & (u2 > 0)
        u1, v1, u2, v2 = u1[ok], v1[ok], u2[ok], v2[ok]
        midu = 0.5 * (u1 + u2)
        midv = 0.5 * (v1 + v2)
        resid = np.hypot(u1 - u2, v1 - v2)

        # canvas-space, flipped y so up = up
        ax.scatter(midu, midv, c=resid, cmap="viridis", s=8,
                   vmin=0, vmax=np.percentile(resid, 95))
        # residual arrows (scaled for visibility)
        SC = 30.0
        segs = np.stack([
            np.column_stack([midu, midv]),
            np.column_stack([midu + SC * (u2 - u1),
                             midv + SC * (v2 - v1)]),
        ], axis=1)
        lc = LineCollection(segs, colors=plt.cm.viridis(
            np.clip(resid / np.percentile(resid, 95), 0, 1)),
            linewidth=0.4, alpha=0.6)
        ax.add_collection(lc)

        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.set_title(f"{label}   N={len(u1)}   "
                     f"mean={resid.mean():.2f} px   "
                     f"median={np.median(resid):.2f} px")
        ax.set_xlabel("canvas u (px)")
        ax.set_ylabel("canvas v (px)")
        # colorbar
        sm = matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(
                0, np.percentile(resid, 95)),
            cmap="viridis")
        plt.colorbar(sm, ax=ax, fraction=0.04, pad=0.02,
                     label="residual (px), arrow $\\times 30$")

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(out_path.parents[2])}")


# ----------------------------- Visual C -----------------------------------

def visual_C_seam_cost_path(debug_dir: Path, eval_frame_dir: Path,
                            out_path: Path):
    cost_lc = read_bgr(debug_dir / "04_seam_cost_lc.jpg")
    cost_cr = read_bgr(debug_dir / "04_seam_cost_cr.jpg")
    seam_lc = read_seam(eval_frame_dir / "seam_FL_FC.bin")
    seam_cr = read_seam(eval_frame_dir / "seam_FC_FR.bin")
    # composite with seam overlay (already drawn by --debug pipeline)
    composite = read_bgr(debug_dir / "05_seam_paths.jpg")
    H = cost_lc.shape[0]

    def seam_extent(seam, margin=160):
        valid = seam[seam >= 0]
        if len(valid) == 0:
            return 0, cost_lc.shape[1]
        x0 = max(0, int(valid.min()) - margin)
        x1 = min(cost_lc.shape[1], int(valid.max()) + margin)
        return x0, x1

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.0),
                             gridspec_kw={"hspace": 0.18, "wspace": 0.08})
    fig.suptitle("Seam DP: cost map (top) and routed seam in the composite "
                 "(bottom)", y=1.00)

    pairs = [
        (cost_lc, seam_lc, "cyan",  "FL$\\leftrightarrow$FC", axes[0, 0], axes[1, 0]),
        (cost_cr, seam_cr, "lime",  "FC$\\leftrightarrow$FR", axes[0, 1], axes[1, 1]),
    ]
    for cost, seam, color, label, ax_top, ax_bot in pairs:
        x0, x1 = seam_extent(seam)
        cc = cost[:, x0:x1]
        comp = composite[:, x0:x1]

        ys = np.arange(H)
        xs = seam.astype(np.float32) - x0
        ok = (seam >= 0) & (xs >= 0) & (xs < cc.shape[1])

        ax_top.imshow(cc, origin="upper")
        ax_top.plot(xs[ok], ys[ok], color=color, lw=1.0)
        ax_top.set_title(f"{label} cost (red = high) + DP path")
        ax_top.set_xticks([]); ax_top.set_yticks([])

        ax_bot.imshow(comp, origin="upper")
        ax_bot.set_title(f"{label} on composite")
        ax_bot.set_xticks([]); ax_bot.set_yticks([])

    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(out_path.parents[2])}")


# ----------------------------- Visual D -----------------------------------

def visual_D_gain_compare(eval_gain_dir: Path, eval_nogain_dir: Path,
                          out_path: Path):
    """Strip across the FL-FC seam, gain-on (top) vs gain-off (bottom)."""

    def load_three(d):
        fl = read_bgr(d / "warped_FL.png")
        fc = read_bgr(d / "warped_FC.png")
        return fl, fc

    fl_g, fc_g = load_three(eval_gain_dir)
    fl_n, fc_n = load_three(eval_nogain_dir)
    # Composite: where FL valid -> FL pixel, else where FC valid -> FC pixel.
    # We use the same nonzero rule for both variants for a clean visual diff.
    mask_lc = cv2.imread(str(eval_gain_dir / "mask_FL_FC.png"),
                         cv2.IMREAD_GRAYSCALE)
    H, W = mask_lc.shape

    # Crop a strip: vertical band ~ seam-column +/- 350 px, all rows where the
    # overlap exists. Find the FL-FC overlap horizontal extent.
    cols_with_overlap = mask_lc.any(axis=0)
    overlap_cols = np.where(cols_with_overlap)[0]
    if len(overlap_cols) == 0:
        print("  no FL-FC overlap; skipping D")
        return
    x0 = max(0, overlap_cols.min() - 100)
    x1 = min(W, overlap_cols.max() + 100)

    def composite(fl, fc):
        fl_alpha = fl.sum(axis=2, keepdims=True) > 0
        fc_alpha = fc.sum(axis=2, keepdims=True) > 0
        out = np.where(fl_alpha, fl, fc).astype(np.uint8)
        out = np.where(fl_alpha | fc_alpha, out, 0)
        return out

    strip_g = composite(fl_g, fc_g)[:, x0:x1]
    strip_n = composite(fl_n, fc_n)[:, x0:x1]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.0))
    fig.suptitle("Per-channel gain compensation: FL$\\cup$FC overlap strip",
                 y=1.00)
    axes[0].imshow(strip_n, origin="upper")
    axes[0].set_title("(a) without gain compensation")
    axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[1].imshow(strip_g, origin="upper")
    axes[1].set_title("(b) with per-channel gain compensation")
    axes[1].set_xticks([]); axes[1].set_yticks([])
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(out_path.parents[2])}")


# ----------------------------- Visual E -----------------------------------

def visual_E_rotation_vs_tps(debug_dir: Path, out_path: Path):
    """
    2×2 grid: rotation-only (top) vs TPS-corrected (bottom) for each overlap strip.
    Each cell is a hard-cut composite: left half = side cam, right half = FC.
    The cut line shows residual misalignment directly.
    """
    try:
        rot_fl = read_bgr(debug_dir / "02b_rotation_FL.jpg")
        rot_fr = read_bgr(debug_dir / "02b_rotation_FR.jpg")
        tps_fl = read_bgr(debug_dir / "02_warped_FL.jpg")
        tps_fc = read_bgr(debug_dir / "02_warped_FC.jpg")
        tps_fr = read_bgr(debug_dir / "02_warped_FR.jpg")
    except FileNotFoundError as e:
        print(f"  skipping E: {e}")
        return

    H, W = tps_fc.shape[:2]

    def valid_col_range(img):
        gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
        cols = np.where((gray > 4).any(axis=0))[0]
        return (int(cols[0]), int(cols[-1])) if len(cols) else (0, W - 1)

    fl_l, fl_r = valid_col_range(tps_fl)
    fc_l, fc_r = valid_col_range(tps_fc)
    fr_l, fr_r = valid_col_range(tps_fr)

    MARGIN = 260
    lc_l = max(0, max(fl_l, fc_l) - MARGIN)
    lc_r = min(W, min(fl_r, fc_r) + MARGIN)
    cr_l = max(0, max(fc_l, fr_l) - MARGIN)
    cr_r = min(W, min(fc_r, fr_r) + MARGIN)

    def hard_cut_strip(img_left, img_right, x0, x1):
        """Left half = img_left, right half = img_right, magenta cut line."""
        strip = np.zeros((H, x1 - x0, 3), dtype=np.uint8)
        mid = (x1 - x0) // 2
        sl = img_left[:, x0:x1]
        sr = img_right[:, x0:x1]
        strip[:, :mid] = sl[:, :mid]
        strip[:, mid:] = sr[:, mid:]
        strip[:, mid - 1:mid + 1] = [255, 0, 200]  # magenta cut line
        return strip

    fig, axes = plt.subplots(2, 2, figsize=(14, 6),
                             gridspec_kw={"hspace": 0.08, "wspace": 0.03})
    fig.suptitle(
        "Overlap strip — rotation-only (top) vs TPS-corrected (bottom).\n"
        "Left of magenta line = side camera; right = FC. Misalignment shows as edge step.",
        y=1.03, fontsize=10)

    pairs = [
        (rot_fl, tps_fc, tps_fl, tps_fc, lc_l, lc_r, "FL$\\leftrightarrow$FC"),
        (rot_fr, tps_fc, tps_fr, tps_fc, cr_l, cr_r, "FC$\\leftrightarrow$FR"),
    ]
    for ci, (rot_side, rot_ref, tps_side, tps_ref, x0, x1, label) in enumerate(pairs):
        if x0 >= x1:
            for row in range(2):
                axes[row, ci].set_visible(False)
            continue
        strip_rot = hard_cut_strip(rot_side, rot_ref, x0, x1)
        strip_tps = hard_cut_strip(tps_side, tps_ref, x0, x1)
        axes[0, ci].imshow(strip_rot, origin="upper")
        axes[0, ci].set_title(f"{label}  —  rotation only", fontsize=9)
        axes[0, ci].set_xticks([]); axes[0, ci].set_yticks([])
        axes[1, ci].imshow(strip_tps, origin="upper")
        axes[1, ci].set_title(f"{label}  —  TPS-corrected", fontsize=9)
        axes[1, ci].set_xticks([]); axes[1, ci].set_yticks([])

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(out_path.parents[2])}")


# ----------------------------- driver -------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug-dir", required=True,
                    help="output/lidar_ring_stitch/debug/frame_NNNN/")
    ap.add_argument("--eval-gain", required=True,
                    help="eval-dump root (gain on); frame_NNNN/ inside")
    ap.add_argument("--eval-nogain", default=None,
                    help="eval-dump root (no gain); needed for visual D")
    ap.add_argument("--frame", type=int, required=True,
                    help="frame index NNNN")
    ap.add_argument("--out-dir", default=None,
                    help="where to write visuals; default <debug-dir>/visuals/")
    ap.add_argument("--skip", nargs="*", default=[],
                    choices=["A", "B", "C", "D", "E"],
                    help="visuals to skip")
    ap.add_argument("--a-only", choices=["FL", "FR"], default=None,
                    help="render visual A for a single camera only")
    args = ap.parse_args()

    debug_dir = Path(args.debug_dir).resolve()
    frame_tag = f"frame_{args.frame:04d}"
    eval_gain = Path(args.eval_gain).resolve() / frame_tag
    eval_nogain = (Path(args.eval_nogain).resolve() / frame_tag
                   if args.eval_nogain else None)
    # Default layout: <pipeline>/output/visuals/<visual>/frame_NNNN.png
    if args.out_dir:
        root = Path(args.out_dir).resolve()
    else:
        # debug_dir = .../output/lidar_ring_stitch/debug/frame_NNNN
        # → .../output/visuals/
        root = debug_dir.parents[2] / "visuals"

    def out_for(sub: str) -> Path:
        d = root / sub
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{frame_tag}.png"

    if "A" not in args.skip:
        visual_A_disp_field(debug_dir, out_for("A_disp"),
                            eval_frame_dir=eval_gain,
                            only=args.a_only)
    if "B" not in args.skip:
        visual_B_reproj_scatter(eval_gain, out_for("B_reproj"))
    if "C" not in args.skip:
        visual_C_seam_cost_path(debug_dir, eval_gain, out_for("C_seam"))
    if "D" not in args.skip:
        if eval_nogain is None:
            print("  skipping D (no --eval-nogain provided)")
        else:
            visual_D_gain_compare(eval_gain, eval_nogain, out_for("D_gain"))
    if "E" not in args.skip:
        visual_E_rotation_vs_tps(debug_dir, out_for("E_rot_vs_tps"))


if __name__ == "__main__":
    main()
