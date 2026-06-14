#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace lidartps {

// -- Two-pass TPS: evaluate on a coarse grid, apply with bilinear upsample ----

// Pass 1: evaluate the TPS displacement on a coarse W_h x H_h grid.
void evalTpsDispHalf(
    int W_h, int H_h,
    const float* d_ctrl_x,
    const float* d_ctrl_y,
    const float* d_wx,
    const float* d_wy,
    int   N_ctrl,
    float* d_disp_x,    // [W_h * H_h]
    float* d_disp_y,
    cudaStream_t stream
);

// Temporal IIR on the displacement field: disp_t = α·new + (1−α)·prev,
// then prev <- disp_t (both in place). No-op when alpha >= 1.
void applyDispTemporalIIR(
    float* d_disp_x,
    float* d_disp_y,
    float* d_disp_x_prev,
    float* d_disp_y_prev,
    int W_h, int H_h, float alpha,
    cudaStream_t stream
);

// Pass 2: full-res rotation remap + bilinear-upsampled TPS displacement.
// Out-of-bounds source coords mark d_vmask = 0 so seam/blend route around
// edge-clamp smear (every pixel writes a fresh 0/1).
void applyRemapWithDispU8(
    const uint8_t* d_src,
    int src_W, int src_H,
    const float2* d_remap,
    int W_c, int H_c,
    int W_h, int H_h,
    const float* d_disp_x,   // [W_h * H_h]
    const float* d_disp_y,
    uint8_t* d_dst,
    uint8_t* d_vmask,
    cudaStream_t stream
);

} // namespace lidartps
