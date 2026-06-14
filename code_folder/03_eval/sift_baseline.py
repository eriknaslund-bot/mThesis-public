#!/usr/bin/env python3
"""
SIFT + RANSAC homography baseline for the LiDAR-TPS pipeline.

This is the classical feature-based stitcher described in §1.2 of the thesis,
implemented as a head-to-head comparison against the LiDAR-TPS pipeline. The
input is the same three forward-facing cameras (FL, FC, FR) of the evaluated
Argoverse 2 log; the metrics computed (seam-L1, overlap PSNR, latency) match
those reported for the LiDAR-TPS pipeline in §4 so the comparison is
apples-to-apples on the same canvas geometry.

Pipeline (per frame):
  1. SIFT keypoints + descriptors in each input.
  2. Lowe's ratio test for cross-camera matching (FL<->FC, FC<->FR).
  3. RANSAC-fitted homography per pair.
  4. Warp + composite into a panorama using OpenCV's standard warpPerspective
     with a fixed-column feather seam.

This script intentionally stays within the OpenCV-native abstractions used by
most published feature-based stitchers (Brown & Lowe 2007; Szeliski 2006), so
the numbers reported here are representative of "what a textbook classical
stitcher delivers" on the same input.

Output: prints a CSV row per frame with seam-L1 and overlap PSNR; also writes
one stitched JPEG per frame to /tmp/sift_out/ for qualitative inspection.

Usage:
    python3 sift_baseline.py \\
        --frames argo2_data/extracted/frames.json \\
        --num-frames 64 \\
        --out /tmp/sift_out
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def load_frame_images(frame, cam_keys=("ring_front_left",
                                       "ring_front_center",
                                       "ring_front_right")):
    imgs = []
    for k in cam_keys:
        p = frame[k]
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read {p}")
        imgs.append(img)
    return imgs


def estimate_homography(left, right, sift, ratio=0.75, ransac_thresh=4.0):
    """Estimate the homography that maps `left` into `right`'s frame.
    Returns (H, n_inliers, n_matches, n_kp_left, n_kp_right) or
    (None, 0, 0, n_kp_left, n_kp_right) on failure."""
    kpL, desL = sift.detectAndCompute(cv2.cvtColor(left,  cv2.COLOR_BGR2GRAY), None)
    kpR, desR = sift.detectAndCompute(cv2.cvtColor(right, cv2.COLOR_BGR2GRAY), None)
    n_kpL = len(kpL) if kpL is not None else 0
    n_kpR = len(kpR) if kpR is not None else 0
    if desL is None or desR is None or n_kpL < 10 or n_kpR < 10:
        return None, 0, 0, n_kpL, n_kpR

    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw = bf.knnMatch(desL, desR, k=2)
    good = [m for m, n in raw if m.distance < ratio * n.distance]
    if len(good) < 8:
        return None, 0, len(good), n_kpL, n_kpR

    src = np.float32([kpL[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kpR[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh)
    # Free large descriptor arrays before returning so the next frame
    # doesn't accumulate them.
    del desL, desR, raw, kpL, kpR, src, dst, bf
    if H is None:
        return None, 0, len(good), n_kpL, n_kpR
    return H, int(mask.sum()), len(good), n_kpL, n_kpR


def stitch_pair(left, right, H):
    """Warp `left` into `right`'s frame using H, return composite canvas + the
    pixel-wise valid masks. Output canvas spans the bounding box of both
    images after warping."""
    hL, wL = left.shape[:2]
    hR, wR = right.shape[:2]
    # Canvas bounds: project corners of left through H, union with right's box.
    corners_L = np.float32([[0, 0], [wL, 0], [wL, hL], [0, hL]]).reshape(-1, 1, 2)
    corners_L_warped = cv2.perspectiveTransform(corners_L, H)
    pts = np.concatenate([corners_L_warped.reshape(-1, 2),
                          np.float32([[0, 0], [wR, 0], [wR, hR], [0, hR]])])
    x0, y0 = np.floor(pts.min(axis=0)).astype(int)
    x1, y1 = np.ceil(pts.max(axis=0)).astype(int)
    W, H_canvas = x1 - x0, y1 - y0
    # Sanity-cap the derived canvas. A near-degenerate RANSAC homography
    # projects corners far outside the image plane, which blows up
    # (W, H_canvas) into multi-GB warp allocations and OOM-kills the host.
    # Legitimate FL+FC or FC+FR canvases on 2048x1550 inputs land around
    # 4000-5000 px wide; anything past 10000 in either axis is degenerate.
    if W > 10000 or H_canvas > 10000 or W <= 0 or H_canvas <= 0:
        raise RuntimeError(f"degenerate canvas size ({W}x{H_canvas})")
    # Translate so the canvas origin is (x0, y0).
    T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], dtype=np.float32)
    H_t = T @ H

    warped_L = cv2.warpPerspective(left,  H_t, (W, H_canvas))
    warped_R = cv2.warpPerspective(right, T,   (W, H_canvas))
    mask_L = cv2.warpPerspective(np.ones((hL, wL), dtype=np.uint8), H_t, (W, H_canvas))
    mask_R = cv2.warpPerspective(np.ones((hR, wR), dtype=np.uint8), T,   (W, H_canvas))
    return warped_L, warped_R, mask_L > 0, mask_R > 0


def compute_overlap_psnr(left, right, mask_L, mask_R):
    """Y-PSNR (BT.709 luma) over the pixel-wise intersection of the two
    valid masks, with content-aware exclusion of black source samples.
    Matches the LiDAR-TPS path's headline photometric metric defined in
    §3.7 of the thesis: AV2 ring cameras carry small rectification-black
    corner regions whose source content is (0,0,0); a warp can sample
    these legitimately (the source coord is in bounds) and the warpPerspective
    'valid mask' still marks them valid, but the sampled pixel carries no
    information. Excluding both-black pixels keeps the metric an honest
    measure of stitch quality."""
    nonblack_L = left.sum(axis=2) > 0
    nonblack_R = right.sum(axis=2) > 0
    overlap = mask_L & mask_R & nonblack_L & nonblack_R
    n = int(overlap.sum())
    if n < 100:
        return float("nan"), float("nan"), float("nan"), 0
    # BT.709: Y = 0.2126 R + 0.7152 G + 0.0722 B; chroma Cb = (B-Y)/1.8556,
    # Cr = (R-Y)/1.5748. Matches the LiDAR-TPS pipeline's metric color space
    # (lidar_tps_pipeline.cpp computeOverlapPsnr) so the paired weighted-PSNR
    # comparison is apples-to-apples. OpenCV BGR layout: index 0=B, 1=G, 2=R.
    bL = left[..., 0].astype(np.float32);  gL = left[..., 1].astype(np.float32);  rL = left[..., 2].astype(np.float32)
    bR = right[..., 0].astype(np.float32); gR = right[..., 1].astype(np.float32); rR = right[..., 2].astype(np.float32)
    yL = 0.2126 * rL + 0.7152 * gL + 0.0722 * bL
    yR = 0.2126 * rR + 0.7152 * gR + 0.0722 * bR
    cbL = (bL - yL) / 1.8556; cbR = (bR - yR) / 1.8556
    crL = (rL - yL) / 1.5748; crR = (rR - yR) / 1.5748

    def _ch_psnr(a, b):
        diff = a[overlap] - b[overlap]
        mse = float((diff ** 2).mean())
        if mse <= 0:
            return float("inf")
        return 10.0 * np.log10(255.0 ** 2 / mse)

    return _ch_psnr(yL, yR), _ch_psnr(cbL, cbR), _ch_psnr(crL, crR), n


def compute_seam_l1(left, right, mask_L, mask_R):
    """Mean |ΔRGB| along a fixed-column seam at the centre of the overlap.
    Matches §3.6.2's seam-L1 definition modulo the seam-routing strategy:
    here the seam is a fixed column rather than DP-routed."""
    overlap = mask_L & mask_R
    if not overlap.any():
        return float("nan")
    cols_with_overlap = np.where(overlap.any(axis=0))[0]
    if cols_with_overlap.size == 0:
        return float("nan")
    seam_col = int(cols_with_overlap[len(cols_with_overlap) // 2])
    rows = np.where(overlap[:, seam_col])[0]
    if rows.size == 0:
        return float("nan")
    diff = np.abs(left[rows, seam_col].astype(np.int32) -
                  right[rows, seam_col].astype(np.int32))
    return float(diff.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True,
                    help="Path to AV2 frames.json")
    ap.add_argument("--num-frames", type=int, default=64,
                    help="Number of frames to evaluate (default: 64)")
    ap.add_argument("--start-frame", type=int, default=0,
                    help="Frame index to start from (default: 0). Combined "
                         "with --num-frames lets the run be chunked across "
                         "multiple Python invocations to bound memory growth "
                         "from cv2.SIFT internal allocators.")
    ap.add_argument("--out", default="/tmp/sift_out",
                    help="Output directory for stitched JPEGs (default: /tmp/sift_out)")
    ap.add_argument("--no-summary", action="store_true",
                    help="Skip the per-run summary block (use when chunking).")
    args = ap.parse_args()

    frames = json.load(open(args.frames))
    start = args.start_frame
    end   = min(start + args.num_frames, len(frames))
    Path(args.out).mkdir(parents=True, exist_ok=True)

    # Fix RANSAC seed so the per-frame homography fit is reproducible: with
    # the default seed, cv2.findHomography(RANSAC) samples are non-deterministic
    # and the failure rate on the canvas-size guard varies meaningfully run
    # to run on this log (sometimes 5/64, sometimes 30+/64).
    cv2.setRNGSeed(42)

    # Cap SIFT keypoints. Default (nfeatures=0) returns ~15-25k points on
    # 2048x1550 daytime road scenes. BFMatcher.knnMatch then materialises
    # a |kp_left| x |kp_right| float32 distance matrix, which on default
    # settings is ~1 GB per overlap and 26+ GB of cumulative cv2 internal
    # allocations across 8 frames before the kernel OOM-kills the process.
    # 2000 keypoints is far more than RANSAC homography needs (8 inliers),
    # caps the per-overlap distance matrix at ~16 MB, and keeps peak Python
    # heap under ~1 GB across an entire 64-frame run.
    sift = cv2.SIFT_create(nfeatures=2000)
    rows = []
    if start == 0:
        print("frame_idx,t_total_ms,n_kp_FL,n_kp_FC,n_kp_FR,n_inliers_FL_FC,"
              "n_inliers_FC_FR,seam_l1_FL_FC,seam_l1_FC_FR,"
              "overlap_psnr_y_FL_FC,overlap_psnr_y_FC_FR,"
              "overlap_psnr_cb_FL_FC,overlap_psnr_cb_FC_FR,"
              "overlap_psnr_cr_FL_FC,overlap_psnr_cr_FC_FR,"
              "overlap_n_FL_FC,overlap_n_FC_FR")
    import gc
    for i in range(start, end):
        fr = frames[i]
        try:
            FL, FC, FR = load_frame_images(fr)
        except RuntimeError as e:
            print(f"# frame {i}: {e}", file=sys.stderr)
            continue
        t0 = time.perf_counter()
        H_LC, ni_LC, nm_LC, kpL_count, kpC_count_lc = estimate_homography(FL, FC, sift)
        H_CR, ni_CR, nm_CR, kpR_count, kpC_count_cr = estimate_homography(FR, FC, sift)
        # Two FC keypoint counts come from independent SIFT runs (one against
        # FL, one against FR). They are typically within a few ‰ of each other
        # because the feature density on FC is content- not query-driven; we
        # report the FL-pass count as the canonical value.
        kpC_count = kpC_count_lc
        if H_LC is None or H_CR is None:
            print(f"# frame {i}: homography estimation failed "
                  f"(LC inliers={ni_LC}/{nm_LC}, CR inliers={ni_CR}/{nm_CR})",
                  file=sys.stderr)
            del FL, FC, FR
            gc.collect()
            continue
        # Stitch FL->FC pair
        try:
            wL, wR_LC, mL_LC, mR_LC = stitch_pair(FL, FC, H_LC)
        except RuntimeError as e:
            print(f"# frame {i}: FL-FC stitch_pair skipped ({e})",
                  file=sys.stderr)
            del FL, FC, FR, H_LC, H_CR
            gc.collect()
            continue
        yLC, cbLC, crLC, n_LC = compute_overlap_psnr(wL, wR_LC, mL_LC, mR_LC)
        sl1_LC = compute_seam_l1(wL, wR_LC, mL_LC, mR_LC)
        # Save composite for the first 4 frames only (qualitative inspection);
        # writing JPEGs for every frame is wasteful and the canvases are
        # already 36 MB each.
        if i < 4:
            comp = wR_LC.copy()
            m_only_L = mL_LC & ~mR_LC
            comp[m_only_L] = wL[m_only_L]
            cv2.imwrite(f"{args.out}/frame_{i:04d}.jpg", comp,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            del comp, m_only_L
        # Free FL<->FC canvases before allocating the FC<->FR pair so memory is
        # bounded at one pair at a time rather than two.
        del wL, wR_LC, mL_LC, mR_LC
        gc.collect()
        # Stitch FR->FC pair (note: FR is the "left" input here in stitch_pair
        # because the homography H_CR maps FR into FC's frame).
        try:
            wR_, wC_CR, mR_CR, mC_CR = stitch_pair(FR, FC, H_CR)
        except RuntimeError as e:
            print(f"# frame {i}: FC-FR stitch_pair skipped ({e})",
                  file=sys.stderr)
            del FL, FC, FR, H_LC, H_CR
            gc.collect()
            continue
        yCR, cbCR, crCR, n_CR = compute_overlap_psnr(wR_, wC_CR, mR_CR, mC_CR)
        sl1_CR = compute_seam_l1(wR_, wC_CR, mR_CR, mC_CR)
        del wR_, wC_CR, mR_CR, mC_CR
        t1 = time.perf_counter()
        rows.append((i, (t1 - t0) * 1000.0,
                     kpL_count, kpC_count, kpR_count,
                     ni_LC, ni_CR,
                     sl1_LC, sl1_CR,
                     yLC, yCR, n_LC, n_CR))
        print(f"{i},{(t1-t0)*1000:.1f},{kpL_count},{kpC_count},{kpR_count},"
              f"{ni_LC},{ni_CR},"
              f"{sl1_LC:.2f},{sl1_CR:.2f},"
              f"{yLC:.2f},{yCR:.2f},"
              f"{cbLC:.2f},{cbCR:.2f},"
              f"{crLC:.2f},{crCR:.2f},{n_LC},{n_CR}")
        # End-of-iteration cleanup so the next frame starts with a clean heap.
        del FL, FC, FR, H_LC, H_CR
        gc.collect()

    if not rows or args.no_summary:
        return
    arr = np.array([(r[1], r[7], r[8], r[9], r[10]) for r in rows])
    print()
    print(f"# {len(rows)} frames evaluated")
    print(f"# t_total_ms     mean={arr[:,0].mean():.1f}  std={arr[:,0].std():.1f}")
    print(f"# seam_l1 FL-FC  mean={arr[:,1].mean():.2f}  std={arr[:,1].std():.2f}")
    print(f"# seam_l1 FC-FR  mean={arr[:,2].mean():.2f}  std={arr[:,2].std():.2f}")
    print(f"# psnr   FL-FC  mean={arr[:,3].mean():.2f}  std={arr[:,3].std():.2f}")
    print(f"# psnr   FC-FR  mean={arr[:,4].mean():.2f}  std={arr[:,4].std():.2f}")


if __name__ == "__main__":
    main()
