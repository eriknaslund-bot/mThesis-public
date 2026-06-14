#!/usr/bin/env python3
"""Pre-extract AV2 LiDAR feather files to raw float32 binary files.

Each output file is named {timestamp}.bin and contains N*3 float32
values (x, y, z in ego frame, row-major).  This lets the C++ pipeline
(lidartps) read LiDAR data without depending on Apache Arrow.

Usage:
    python3 utils/extract_lidar_bin.py
    python3 utils/extract_lidar_bin.py --sensor-root ~/mThesis/argo2_data/sensor
                                       --out ~/mThesis/argo2_data/extracted/sensors/lidar
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sensor-root', default=str(Path.home() / 'mThesis/argo2_data/sensor'))
    ap.add_argument('--out', default=str(Path.home() / 'mThesis/argo2_data/extracted/sensors/lidar'))
    args = ap.parse_args()

    sensor_root = Path(args.sensor_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    feather_files = sorted(sensor_root.glob('train/*/sensors/lidar/*.feather'))
    if not feather_files:
        raise RuntimeError(f'No feather files found under {sensor_root}/train/*/sensors/lidar/')

    print(f'Found {len(feather_files)} LiDAR feather files')

    n_written = 0
    for fp in feather_files:
        out_path = out_dir / f'{fp.stem}.bin'
        if out_path.exists():
            continue  # skip already-extracted
        try:
            df = pd.read_feather(fp)
            xyz = df[['x', 'y', 'z']].values.astype(np.float32)
            xyz.tofile(out_path)
            n_written += 1
            if n_written % 50 == 0:
                print(f'  {n_written}/{len(feather_files)}  {fp.stem}  {len(xyz)} pts')
        except Exception as e:
            print(f'  WARNING: failed {fp}: {e}')

    total = sum(1 for _ in out_dir.glob('*.bin'))
    print(f'Done.  {n_written} new files written, {total} total in {out_dir}')


if __name__ == '__main__':
    main()
