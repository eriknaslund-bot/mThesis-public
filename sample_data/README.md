# Smoke-test sample (one Argoverse 2 frame)

A single forward-facing frame (FL/FC/FR + the nearest LiDAR sweeps) so the
pipeline can be built and run on real data straight from a clean clone,
without downloading the full Argoverse 2 dataset.

```
sample_data/
  calibration.json
  frames.json                      # 1 frame, paths relative to the repo root
  sensors/cameras/ring_front_{left,center,right}/*.jpg
  sensors/lidar/*.bin              # 2 sweeps; nearest to the frame is auto-picked
```

## Run (from the repository root)

```bash
# build once
cd code_folder/02_implementations/lidar_tps_pipeline
mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)
cd ../../../../        # back to repo root

# stitch the sample frame
code_folder/02_implementations/lidar_tps_pipeline/build/lidartps \
    --calib  sample_data/calibration.json \
    --frames sample_data/frames.json \
    --lidar  sample_data/sensors/lidar \
    --frame  0 --out /tmp/sample_out
```

Expected: `/tmp/sample_out/frame_0.jpg`, a 5238 × 2303 cylindrical panorama,
in ~40 ms.

## Attribution / licence

This sample is excerpted from the **Argoverse 2 Sensor Dataset** (Wilson et
al., NeurIPS 2021 Datasets & Benchmarks), licensed **CC BY-NC-SA 4.0**
(non-commercial, share-alike, attribution). It is included here only as a
minimal reproducibility sample; the full dataset must be obtained from
<https://www.argoverse.org/av2.html> under that licence.
