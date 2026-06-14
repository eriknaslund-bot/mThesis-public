#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace lidartps {

// Per-pixel seam cost: |ΔL| + grad_weight·(|∇L1_l|+|∇L1_r|)/4, where
// ∇L1 = |Sobel_x|+|Sobel_y|. Non-overlap pixels -> 1e9.
// d_left/right: BGR warped canvases; d_valid_*: 1 where valid; d_cost: out.
void computeSeamCostU8(
    const uint8_t* d_left,
    const uint8_t* d_right,
    const uint8_t* d_valid_left,
    const uint8_t* d_valid_right,
    int W_c, int H_c,
    float grad_weight,
    float* d_cost,
    cudaStream_t stream
);

// Seam DP via CUDA Graph: first call captures the H per-row launches as a
// graph; later calls replay it in one launch. io_* are caller-owned state
// (init nullptr/nullptr/false).
void findSeamDPGraph(
    const float* cost,
    int width, int height,
    int* seam_out,
    float* d_dp,
    int* d_backtrack,
    cudaGraph_t&     io_graph,
    cudaGraphExec_t& io_exec,
    bool&            io_ready,
    cudaStream_t stream
);

} // namespace lidartps
