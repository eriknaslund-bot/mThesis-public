#!/usr/bin/env python3
"""
Temporal-coherence metrics for stitched-panorama MP4s.

Measures how visually stable the panorama is across consecutive frames using
two metrics that complement the existing per-frame photometric numbers in
§4 of the thesis:

  1. Frame-to-frame absolute difference. For each adjacent pair of frames
     (t, t+1) compute mean |I_t − I_{t+1}| over pixels where both frames are
     valid (non-black). This includes scene motion, but for a fixed log the
     scene-motion component is identical across compared videos, so the
     *delta* between configurations isolates pipeline-induced jitter.

  2. File size on disk. H.264/HEVC encoders allocate fewer bits to video
     that compresses well; for a fixed encoder + bitrate target the size
     ratio is a clean proxy for temporal stability. We have rendered the
     same 318-frame log under several --disp-alpha settings; comparing
     file sizes is a one-line statement of the temporal-coherence effect.

Usage:
    python3 temporal_video_metrics.py video1.mp4 video2.mp4 ...

Outputs a CSV row per video plus a summary table at the end.
"""

import argparse
import os
import sys

import cv2
import numpy as np


def frame_to_frame_diff(path, max_frames=None):
    """Mean of mean |ΔRGB| between adjacent frames, over the whole video.
    Skips pixels where either frame is fully black (canvas border)."""
    v = cv2.VideoCapture(path)
    if not v.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    n_frames = int(v.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        n_frames = min(n_frames, max_frames)

    ok, prev = v.read()
    if not ok:
        v.release()
        return float("nan"), 0
    diffs = []
    for _ in range(n_frames - 1):
        ok, cur = v.read()
        if not ok:
            break
        # Mask to the union of the two valid regions (sum > 0 in BGR).
        valid = ((prev.sum(axis=2) > 0) & (cur.sum(axis=2) > 0))
        if valid.any():
            d = np.abs(prev[valid].astype(np.int16) - cur[valid].astype(np.int16))
            diffs.append(float(d.mean()))
        prev = cur
    v.release()
    return float(np.mean(diffs)) if diffs else float("nan"), len(diffs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+", help="MP4 paths")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="Cap each video at N frames")
    args = ap.parse_args()

    print("video,size_MB,n_frames,frame_diff_mean")
    rows = []
    for path in args.videos:
        size_mb = os.path.getsize(path) / (1024 ** 2)
        diff, n = frame_to_frame_diff(path, args.max_frames)
        v = cv2.VideoCapture(path)
        n_total = int(v.get(cv2.CAP_PROP_FRAME_COUNT))
        v.release()
        print(f"{os.path.basename(path)},{size_mb:.1f},{n_total},{diff:.4f}")
        rows.append((path, size_mb, n_total, diff))

    if len(rows) > 1:
        print()
        print("# Relative change vs first video:")
        ref = rows[0]
        for path, size_mb, n, diff in rows[1:]:
            d_size = (size_mb - ref[1]) / ref[1] * 100
            d_diff = (diff - ref[3]) / ref[3] * 100 if ref[3] > 0 else float("nan")
            print(f"# {os.path.basename(path):30s}  size: {d_size:+5.1f}%   "
                  f"frame-diff: {d_diff:+5.1f}%")


if __name__ == "__main__":
    main()
