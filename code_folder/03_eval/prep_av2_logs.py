#!/usr/bin/env python3
"""
Convert downloaded AV2 sensor logs (cameras + lidar + calibration in
their native feather/jpg format) into the layout the lidar_tps_pipeline
binary expects:

  <out_root>/<log_id>/
      calibration.json                       -- flat per-camera dict
      frames.json                            -- list of {cam: img_path}
      sensors/lidar/<timestamp>.bin          -- float32 Nx3, ego frame

Reads the AV2-native files in-place (no copies), so an extra log
that was downloaded with cameras/<ring_*>/<timestamp>.jpg, lidar/
<timestamp>.feather, and calibration/{intrinsics,egovehicle_SE3_sensor}
.feather can be plugged into the existing pipeline harness.

Usage:
    python3 prep_av2_logs.py \\
        --in-root  argo2_data/sensor/train_extra \\
        --out-root argo2_data/extra_extracted
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

RING_CAMERAS = ["ring_front_left", "ring_front_center", "ring_front_right"]
MAX_DELTA_NS = 50_000_000  # 50 ms


def parse_calib(intr_path, extr_path):
    intr = pd.read_feather(intr_path)
    extr = pd.read_feather(extr_path)
    intr = intr.set_index("sensor_name")
    extr = extr.set_index("sensor_name")
    calib = {}
    for cam in RING_CAMERAS:
        if cam not in intr.index or cam not in extr.index:
            continue
        i = intr.loc[cam]
        e = extr.loc[cam]
        calib[cam] = {
            "fx": float(i.fx_px),
            "fy": float(i.fy_px),
            "cx": float(i.cx_px),
            "cy": float(i.cy_px),
            "width":  int(i.width_px),
            "height": int(i.height_px),
            "k1": float(i.k1),
            "k2": float(i.k2),
            "k3": float(i.k3),
            "qw": float(e.qw),
            "qx": float(e.qx),
            "qy": float(e.qy),
            "qz": float(e.qz),
            "tx_m": float(e.tx_m),
            "ty_m": float(e.ty_m),
            "tz_m": float(e.tz_m),
        }
    return calib


def match_timestamps(cam_ts):
    """{cam: sorted [ts]} -> list of {cam: matched_ts} synced within MAX_DELTA_NS."""
    ref = sorted(cam_ts["ring_front_center"])
    frames = []
    for t_ref in ref:
        matched = {"ring_front_center": t_ref}
        ok = True
        for cam in RING_CAMERAS:
            if cam == "ring_front_center":
                continue
            ts_list = cam_ts[cam]
            if not ts_list:
                ok = False; break
            idx = np.searchsorted(ts_list, t_ref)
            cands = []
            if idx < len(ts_list): cands.append(ts_list[idx])
            if idx > 0:            cands.append(ts_list[idx - 1])
            best = min(cands, key=lambda t: abs(t - t_ref))
            if abs(best - t_ref) > MAX_DELTA_NS:
                ok = False; break
            matched[cam] = best
        if ok:
            frames.append(matched)
    return frames


def prep_log(in_log, out_log):
    cam_root = in_log / "sensors" / "cameras"
    lidar_root = in_log / "sensors" / "lidar"
    calib_root = in_log / "calibration"

    if not (calib_root / "intrinsics.feather").exists():
        print(f"  SKIP: {in_log} (no calibration)")
        return False

    calib = parse_calib(calib_root / "intrinsics.feather",
                         calib_root / "egovehicle_SE3_sensor.feather")
    out_log.mkdir(parents=True, exist_ok=True)
    with open(out_log / "calibration.json", "w") as f:
        json.dump(calib, f, indent=2)

    cam_ts = {}
    for cam in RING_CAMERAS:
        d = cam_root / cam
        cam_ts[cam] = sorted(int(p.stem) for p in d.glob("*.jpg"))
    print(f"  cameras: " + ", ".join(f"{c}={len(cam_ts[c])}" for c in RING_CAMERAS))
    matched = match_timestamps(cam_ts)
    print(f"  matched {len(matched)} synchronised frames "
          f"(±{MAX_DELTA_NS//1_000_000} ms)")

    frames_json = []
    for m in matched:
        frames_json.append({
            cam: str((cam_root / cam / f"{m[cam]}.jpg").resolve())
            for cam in RING_CAMERAS
        })
    with open(out_log / "frames.json", "w") as f:
        json.dump(frames_json, f)

    out_lidar = out_log / "sensors" / "lidar"
    out_lidar.mkdir(parents=True, exist_ok=True)
    feather_files = sorted(lidar_root.glob("*.feather"))
    written = 0
    for fp in feather_files:
        out_path = out_lidar / f"{fp.stem}.bin"
        if out_path.exists():
            continue
        df = pd.read_feather(fp)
        xyz = df[["x", "y", "z"]].values.astype(np.float32)
        xyz.tofile(out_path)
        written += 1
    print(f"  lidar: {len(feather_files)} sweeps, {written} new bins")
    print(f"  -> {out_log}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-root", required=True)
    ap.add_argument("--out-root", required=True)
    args = ap.parse_args()

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)
    log_dirs = sorted(d for d in in_root.iterdir() if d.is_dir())
    if not log_dirs:
        raise SystemExit(f"No log directories under {in_root}")

    for in_log in log_dirs:
        out_log = out_root / in_log.name
        print(f"=== {in_log.name} ===")
        prep_log(in_log, out_log)


if __name__ == "__main__":
    main()
