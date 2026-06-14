#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include "mat3f.cuh"

namespace lidartps {

// Project N ego-frame LiDAR points into one camera's image plane.
// AV2 convention: points are already in ego frame; R_cam_ego is cam->ego.
//
// d_pts        [N*3]   float  ego x/y/z
// d_cam_uv     [N*2]   float  out -- (u,v) in camera image; (-1,-1) if invalid
// d_depth      [N]     float  out -- ego-frame range in metres; 0 if invalid
// d_valid      [N]     uint8  out -- 1 if point is visible in this camera
void projectLidarEgoGpu(
    const float* d_pts,
    int N,
    Mat3f  R_cam_ego,      // cam->ego rotation (row-major)
    float3 t_cam_ego,      // cam->ego translation (camera origin in ego frame)
    float  fx, float fy, float cx, float cy,
    int    cam_W, int cam_H,
    float  max_range_m,
    float*   d_cam_uv,
    float*   d_depth,
    uint8_t* d_valid,
    cudaStream_t stream
);

} // namespace lidartps
