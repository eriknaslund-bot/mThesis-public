#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include "mat3f.cuh"

namespace lidartps {

// Build true-cylindrical rotation-only remap (one-time per calibration).
// For each canvas pixel (u,v):
//   az = (cx_canvas - u) / fx_canvas
//   el = atan((cy_canvas - v) / fy_canvas)
//   ego_dir = (cos(el)*cos(az), cos(el)*sin(az), sin(el))
//   cam_dir = R_cam_ego.T() * ego_dir
//   -> pinhole project to (u_cam, v_cam)
// Output float2 per canvas pixel: (src_u, src_v); (-1,-1) if outside camera.
// d_valid_mask: 1 where remap is valid (cam_dir.z > 0 and inside image).
void buildRotRemap(
    int W_c, int H_c,
    Mat3f  R_cam_ego,
    float  fx, float fy, float cx, float cy,
    int    src_W, int src_H,
    float  fx_canvas, float fy_canvas, float cx_canvas, float cy_canvas,
    float2*  d_remap,
    uint8_t* d_valid_mask,
    cudaStream_t stream
);

// Apply a precomputed float2 remap to a U8 BGR source image (bilinear).
// d_src   [src_H * src_W * 3]  uint8  BGR
// d_remap [H_c * W_c]          float2 (src_u, src_v)
// d_dst   [H_c * W_c * 3]      uint8  BGR out
void applyRemapU8(
    const uint8_t* d_src,
    int src_W, int src_H,
    const float2* d_remap,
    int W_c, int H_c,
    uint8_t* d_dst,
    cudaStream_t stream
);

} // namespace lidartps
