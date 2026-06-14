#!/usr/bin/env python3
"""LiDAR-driven homography baseline.

Same control-point set as the LiDAR-TPS pipeline (3-D LiDAR returns visible
in two adjacent ring cameras), but fits a 3x3 perspective homography to
those correspondences instead of a Thin-Plate Spline. This isolates the
TPS contribution: any improvement that LiDAR-TPS shows over this baseline
is attributable to the deformable warp, not to the use of LiDAR points
per se.

Output: one CSV row per frame with seam-L1 and Y-PSNR over the same
camera-pair overlap regions used in §4 (BT.709 luma, intersection of
the per-camera vmasks). Stdout summary at the end.

Run from repo root:
    python3 code_folder/03_eval/lidar_homography_baseline.py \\
        --frames argo2_data/extracted/frames.json \\
        --calib  argo2_data/extracted/calibration.json \\
        --lidar  argo2_data/extracted/sensors/lidar \\
        --num-frames 64
"""

import argparse
import gc
import json
import statistics as st
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "code_folder" / "02_implementations" / "lidar_tps_pipeline" / "utils"))
from lidar_ring_stitch import quat_to_mat, project_with_ego  # noqa: E402

sys.path.insert(0, str(REPO / "code_folder" / "03_eval"))
from sift_baseline import stitch_pair, compute_overlap_psnr, compute_seam_l1  # noqa: E402

LIDAR_MIN_RANGE_M = 1.0
LIDAR_MAX_RANGE_M = 120.0
FRONT_CAMS = ["ring_front_left", "ring_front_center", "ring_front_right"]


def load_calib(path):
    with open(path) as f:
        raw = json.load(f)
    cams = {}
    for name, v in raw.items():
        if "ring_" not in name:
            continue
        cams[name] = {
            "R": quat_to_mat(v["qw"], v["qx"], v["qy"], v["qz"]),
            "t": np.array([v["tx_m"], v["ty_m"], v["tz_m"]], dtype=np.float32),
            "fx": v["fx"], "fy": v["fy"],
            "cx": v["cx"], "cy": v["cy"],
            "W":  v["width"], "H": v["height"],
        }
    return cams


def load_lidar_bin(p):
    return np.fromfile(p, dtype=np.float32).reshape(-1, 3)


def find_shared(px_l, idx_l, px_r, idx_r):
    """Intersect two camera projections on original LiDAR indices."""
    if len(idx_l) == 0 or len(idx_r) == 0:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
    map_l = {int(i): j for j, i in enumerate(idx_l)}
    map_r = {int(i): j for j, i in enumerate(idx_r)}
    shared = sorted(set(map_l.keys()) & set(map_r.keys()))
    if not shared:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
    jl = np.array([map_l[i] for i in shared], dtype=np.int64)
    jr = np.array([map_r[i] for i in shared], dtype=np.int64)
    return px_l[jl], px_r[jr]


def fit_h(src_pts, dst_pts):
    """Least-squares homography. LiDAR pts are clean correspondences;
    no RANSAC inlier filtering required."""
    if len(src_pts) < 4:
        return None
    H, _ = cv2.findHomography(src_pts, dst_pts, method=0)  # 0 = least-squares
    return H


def find_lidar_for_frame(frame, lidar_dir):
    """Pick the .bin sweep nearest to the FC image timestamp."""
    img_ts = int(Path(frame["ring_front_center"]).stem)
    bins = sorted(lidar_dir.glob("*.bin"))
    ts = np.array([int(p.stem) for p in bins], dtype=np.int64)
    j = int(np.argmin(np.abs(ts - img_ts)))
    return bins[j]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--calib",  required=True)
    ap.add_argument("--lidar",  required=True,
                    help="Directory of LiDAR .bin sweeps (xyz float32, stride 3)")
    ap.add_argument("--num-frames", type=int, default=64)
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--out", default="/tmp/lidar_h_out")
    ap.add_argument("--no-summary", action="store_true")
    args = ap.parse_args()

    cams = load_calib(args.calib)
    frames = json.load(open(args.frames))
    lidar_dir = Path(args.lidar)
    Path(args.out).mkdir(parents=True, exist_ok=True)

    start = args.start_frame
    end = min(start + args.num_frames, len(frames))
    rows = []
    if start == 0:
        print("frame_idx,t_total_ms,n_shared_FL_FC,n_shared_FC_FR,"
              "seam_l1_FL_FC,seam_l1_FC_FR,"
              "overlap_psnr_y_FL_FC,overlap_psnr_y_FC_FR,"
              "overlap_psnr_cb_FL_FC,overlap_psnr_cb_FC_FR,"
              "overlap_psnr_cr_FL_FC,overlap_psnr_cr_FC_FR,"
              "overlap_n_FL_FC,overlap_n_FC_FR")

    for i in range(start, end):
        fr = frames[i]
        try:
            FL = cv2.imread(fr["ring_front_left"])
            FC = cv2.imread(fr["ring_front_center"])
            FR = cv2.imread(fr["ring_front_right"])
            if FL is None or FC is None or FR is None:
                raise RuntimeError("image load failed")
        except Exception as e:
            print(f"# frame {i}: {e}", file=sys.stderr)
            continue

        try:
            lidar_path = find_lidar_for_frame(fr, lidar_dir)
            pts = load_lidar_bin(lidar_path)
        except Exception as e:
            print(f"# frame {i}: lidar load failed: {e}", file=sys.stderr)
            continue

        t0 = time.perf_counter()
        # Project LiDAR into each camera, with the original-index hook so
        # we can intersect two cameras' visible-point sets.
        px_l, _, idx_l = project_with_ego(pts, cams["ring_front_left"],   return_indices=True)
        px_c, _, idx_c = project_with_ego(pts, cams["ring_front_center"], return_indices=True)
        px_r, _, idx_r = project_with_ego(pts, cams["ring_front_right"],  return_indices=True)

        # Shared sets per overlap pair
        fl_sh, fc_sh_lc = find_shared(px_l, idx_l, px_c, idx_c)
        fr_sh, fc_sh_cr = find_shared(px_r, idx_r, px_c, idx_c)
        n_LC = len(fl_sh)
        n_CR = len(fr_sh)

        H_LC = fit_h(fl_sh, fc_sh_lc)
        H_CR = fit_h(fr_sh, fc_sh_cr)

        if H_LC is None or H_CR is None:
            print(f"# frame {i}: insufficient shared pts (LC={n_LC}, CR={n_CR})",
                  file=sys.stderr)
            del FL, FC, FR, pts; gc.collect()
            continue

        # FL -> FC
        try:
            wL, wC_LC, mL_LC, mC_LC = stitch_pair(FL, FC, H_LC)
        except RuntimeError as e:
            print(f"# frame {i}: FL-FC stitch_pair skipped ({e})",
                  file=sys.stderr)
            del FL, FC, FR, pts; gc.collect()
            continue
        yLC, cbLC, crLC, n_pix_LC = compute_overlap_psnr(wL, wC_LC, mL_LC, mC_LC)
        sl1_LC = compute_seam_l1(wL, wC_LC, mL_LC, mC_LC)

        if i < 4:
            comp = wC_LC.copy()
            m_only_L = mL_LC & ~mC_LC
            comp[m_only_L] = wL[m_only_L]
            cv2.imwrite(f"{args.out}/frame_{i:04d}.jpg", comp,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            del comp, m_only_L
        del wL, wC_LC, mL_LC, mC_LC; gc.collect()

        # FR -> FC
        try:
            wR, wC_CR, mR_CR, mC_CR = stitch_pair(FR, FC, H_CR)
        except RuntimeError as e:
            print(f"# frame {i}: FC-FR stitch_pair skipped ({e})",
                  file=sys.stderr)
            del FL, FC, FR, pts; gc.collect()
            continue
        yCR, cbCR, crCR, n_pix_CR = compute_overlap_psnr(wR, wC_CR, mR_CR, mC_CR)
        sl1_CR = compute_seam_l1(wR, wC_CR, mR_CR, mC_CR)
        del wR, wC_CR, mR_CR, mC_CR

        t1 = time.perf_counter()
        rows.append((i, (t1 - t0) * 1000.0, n_LC, n_CR,
                     sl1_LC, sl1_CR, yLC, yCR, n_pix_LC, n_pix_CR))
        print(f"{i},{(t1 - t0) * 1000:.1f},{n_LC},{n_CR},"
              f"{sl1_LC:.2f},{sl1_CR:.2f},"
              f"{yLC:.2f},{yCR:.2f},"
              f"{cbLC:.2f},{cbCR:.2f},"
              f"{crLC:.2f},{crCR:.2f},{n_pix_LC},{n_pix_CR}")

        del FL, FC, FR, pts; gc.collect()

    if not rows or args.no_summary:
        return
    print()
    print(f"# n = {len(rows)}")
    cols = [("t_total_ms", 1), ("seam_l1_FL_FC", 4), ("seam_l1_FC_FR", 5),
            ("overlap_psnr_FL_FC", 6), ("overlap_psnr_FC_FR", 7)]
    for name, k in cols:
        v = [r[k] for r in rows]
        print(f"#   {name}: {st.mean(v):.2f} +/- {st.pstdev(v):.2f}")


if __name__ == "__main__":
    main()
