#!/usr/bin/env python3
"""Phase 1 diversity scan over all locally-downloaded AV2 logs.

For each log under argo2_data/training/logs/ (cameras + disp only, no LiDAR
in this set) compute a feature vector that lets us pick a maximally-diverse
subset for the thesis evaluation:

- Lighting: mean luminance, luminance std, dark-pixel %, bright-pixel %
- Scene depth (from precomputed disp/ maps): median disparity (= 1/depth
  proxy), disparity std (depth diversity)
- Frame count, sensor list

Writes both a JSON dump and a printed ranked summary.
"""
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("argo2_data/training/logs")
OUT_CSV = Path("code_folder/03_eval/log_diversity_scan.csv")
OUT_JSON = Path("code_folder/03_eval/log_diversity_scan.json")


def lum_features(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return dict(
        mean_lum=float(gray.mean()),
        std_lum=float(gray.std()),
        dark_frac=float((gray < 40).mean()),
        bright_frac=float((gray > 200).mean()),
    )


def disp_features(disp_dir):
    """Read mid-sequence disp map, return depth-proxy stats."""
    frames = sorted(disp_dir.glob("*"))
    if not frames:
        return dict(disp_median=float("nan"), disp_std=float("nan"),
                    n_disp_frames=0)
    mid = frames[len(frames) // 2]
    # AV2 disp maps are typically 16-bit PNG or .npy / .feather; try a few.
    img = None
    if mid.suffix in (".png", ".jpg", ".jpeg"):
        img = cv2.imread(str(mid), cv2.IMREAD_UNCHANGED)
    if img is None:
        return dict(disp_median=float("nan"), disp_std=float("nan"),
                    n_disp_frames=len(frames))
    arr = img.astype(np.float32)
    valid = arr[arr > 0]
    if valid.size == 0:
        return dict(disp_median=float("nan"), disp_std=float("nan"),
                    n_disp_frames=len(frames))
    return dict(
        disp_median=float(np.median(valid)),
        disp_std=float(valid.std()),
        n_disp_frames=len(frames),
    )


def scan():
    rows = []
    for log_dir in sorted(ROOT.iterdir()):
        if not log_dir.is_dir():
            continue
        fc_dir = log_dir / "sensors" / "cameras" / "ring_front_center"
        if not fc_dir.is_dir():
            continue
        cam_frames = sorted(fc_dir.glob("*.jpg"))
        if not cam_frames:
            continue
        mid = cam_frames[len(cam_frames) // 2]
        img = cv2.imread(str(mid))
        if img is None:
            continue
        row = dict(log=log_dir.name[:8], log_full=log_dir.name,
                   n_cam_frames=len(cam_frames),
                   mid_frame=mid.name)
        row.update(lum_features(img))
        disp_dir = log_dir / "sensors" / "disp" / "ring_front_center"
        row.update(disp_features(disp_dir) if disp_dir.is_dir()
                   else dict(disp_median=float("nan"),
                             disp_std=float("nan"),
                             n_disp_frames=0))
        rows.append(row)
    return rows


def main():
    rows = scan()
    OUT_JSON.write_text(json.dumps(rows, indent=2))
    # Also CSV
    if rows:
        keys = list(rows[0].keys())
        with open(OUT_CSV, "w") as f:
            f.write(",".join(keys) + "\n")
            for r in rows:
                f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")

    print(f"Scanned {len(rows)} logs. Wrote {OUT_CSV} and {OUT_JSON}.\n")

    # Ranked summaries on each axis
    def show(key, label, n=8, ascending=True):
        sorted_rows = sorted(
            (r for r in rows if isinstance(r.get(key), float) and not np.isnan(r[key])),
            key=lambda r: r[key], reverse=not ascending,
        )
        print(f"=== {label} ({'lowest' if ascending else 'highest'} {n}) ===")
        for r in sorted_rows[:n]:
            print(f"  {r['log']:<10} {key}={r[key]:>8.2f}  mean_lum={r['mean_lum']:>6.1f}"
                  f"  std_lum={r['std_lum']:>5.1f}  dark%={r['dark_frac']*100:>4.1f}"
                  f"  bright%={r['bright_frac']*100:>4.1f}")
        print()

    show("mean_lum",    "Darkest scenes",         n=10, ascending=True)
    show("std_lum",     "Highest dynamic range",  n=8,  ascending=False)
    show("dark_frac",   "Most dark pixels",       n=8,  ascending=False)
    show("bright_frac", "Most sky-bright pixels", n=8,  ascending=False)
    show("disp_median", "Furthest avg scene",     n=8,  ascending=True)
    show("disp_median", "Closest avg scene",      n=8,  ascending=False)
    show("disp_std",    "Most varied scene depth",n=8,  ascending=False)


if __name__ == "__main__":
    main()
