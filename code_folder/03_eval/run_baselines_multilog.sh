#!/usr/bin/env bash
# Run SIFT+RANSAC and LiDAR-homography baselines on the 9 extra AV2 logs
# so the §4.3 paired comparisons can be extended from primary-log-only to
# 10-log aggregates (matching the rest of §4).
#
# Output: code_folder/03_eval/sift_baseline_<short>.csv
#         code_folder/03_eval/lidar_homography_baseline_<short>.csv
#
# Wall time estimate: ~5 min per baseline per log * 2 baselines * 9 logs ≈ 90 min.

set -e
N_FRAMES=64

LOGS=(
  "01bb304d-7bd8-35f8-bbef-7086b688e35e"
  "022af476-9937-3e70-be52-f65420d52703"
  "04973bcf-fc64-367c-9642-6d6c5f363b61"
  "087695bd-c662-3e86-83b4-aedc3b8eec36"
  "05853f69-f948-3d04-8d64-d4e721c0e1a5"
  "12071817-ba53-35a4-bf6c-a8e8e7ad8969"
  "12c3c14b-9cf2-3434-9a5d-e0bfa332f6ce"
  "0a524e66-ee33-3b6c-89ef-eac1985316db"
  "072c8e90-a51c-3429-9cdf-4dababb4e9d8"
)

echo "=== START $(date +%H:%M:%S) ==="
for L in "${LOGS[@]}"; do
  TAG=${L%%-*}
  DD=argo2_data/extra_extracted/$L
  if [ ! -f "$DD/calibration.json" ]; then
    echo "[skip] $TAG (no extracted data at $DD)"
    continue
  fi

  echo "=== $(date +%H:%M:%S)  log $TAG ==="

  echo "  SIFT baseline..."
  python3 code_folder/03_eval/sift_baseline.py \
      --frames "$DD/frames.json" \
      --num-frames "$N_FRAMES" \
      --out "/tmp/sift_out_$TAG" \
    > "code_folder/03_eval/sift_baseline_$TAG.csv" 2>"/tmp/sift_$TAG.err"
  rm -rf "/tmp/sift_out_$TAG"

  echo "  LiDAR-homography baseline..."
  python3 code_folder/03_eval/lidar_homography_baseline.py \
      --frames "$DD/frames.json" \
      --calib  "$DD/calibration.json" \
      --lidar  "$DD/sensors/lidar" \
      --num-frames "$N_FRAMES" \
      --out "/tmp/homog_out_$TAG" \
    > "code_folder/03_eval/lidar_homography_baseline_$TAG.csv" 2>"/tmp/homog_$TAG.err"
  rm -rf "/tmp/homog_out_$TAG"
done
echo "=== DONE $(date +%H:%M:%S) ==="
