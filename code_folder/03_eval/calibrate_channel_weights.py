#!/usr/bin/env python3
"""Derive the channel-weighted-PSNR weights used as the headline metric (Eq. 3.6).

The weighted PSNR of thesis Eq. 3.6 combines the per-channel BT.709 PSNRs as

    PSNR_w = wY*PSNR_Y + wCb*PSNR_Cb + wCr*PSNR_Cr,

with weights set to the per-channel signal-energy share of the imagery. This
script measures that share as the dataset-mean fraction of total BT.709
luma/chroma variance across the ten evaluation logs and prints the weights.

Method: for each log, sample every 8th frame of the 240-frame window, read all
three forward cameras at native resolution, convert BGR -> YCrCb, and take the
per-channel variance. The per-log energy share is the channel variance over the
sum of the three; the reported weights are the mean of the per-log shares.

This is the canonical derivation of the (0.918, 0.047, 0.035) weights baked into
aggregate_paired_comparison.py and quoted in section 3.4. It supersedes the
bitrate-based weighting prototyped in calibrate_psnr_weights.py, which is
confounded by the NVENC rate cap on high-detail logs (see that file's header).

Usage:
    python3 calibrate_channel_weights.py
"""

import glob
import json
import statistics as st
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]

# (short tag, data dir) for the ten evaluation logs.
LOGS = [("00a6ffc1", REPO / "argo2_data/extracted")] + [
    (Path(d).name[:8], Path(d))
    for d in sorted(glob.glob(str(REPO / "argo2_data/extra_extracted/*")))
]

CAMS = ["ring_front_left", "ring_front_center", "ring_front_right"]
FRAME_STRIDE = 8
MAX_FRAMES = 240


def log_energy_share(frames_json):
    frames = json.load(open(frames_json))
    vY, vCb, vCr = [], [], []
    for i in range(0, min(len(frames), MAX_FRAMES), FRAME_STRIDE):
        for cam in CAMS:
            img = cv2.imread(frames[i][cam])
            if img is None:
                continue
            ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb).astype(np.float32)
            vY.append(ycc[:, :, 0].var())   # Y
            vCr.append(ycc[:, :, 1].var())  # Cr (OpenCV YCrCb order)
            vCb.append(ycc[:, :, 2].var())  # Cb
    eY, eCb, eCr = np.mean(vY), np.mean(vCb), np.mean(vCr)
    tot = eY + eCb + eCr
    return eY / tot, eCb / tot, eCr / tot


def main():
    pY, pCb, pCr = [], [], []
    print(f"{'log':10} {'Y%':>6} {'Cb%':>6} {'Cr%':>6}")
    for tag, base in LOGS:
        fj = base / "frames.json"
        if not fj.exists():
            print(f"  [skip] {tag} (no frames.json)")
            continue
        y, cb, cr = log_energy_share(fj)
        pY.append(y); pCb.append(cb); pCr.append(cr)
        print(f"{tag:10} {y*100:6.1f} {cb*100:6.1f} {cr*100:6.1f}")
    wY, wCb, wCr = st.mean(pY), st.mean(pCb), st.mean(pCr)
    print("\nDataset-mean channel-weighted-PSNR weights:")
    print(f"  wY = {wY:.3f}   wCb = {wCb:.3f}   wCr = {wCr:.3f}   "
          f"(sum {wY+wCb+wCr:.3f})")


if __name__ == "__main__":
    main()
