// seam_cost.cu
// Stage 9a: per-pixel seam cost for DP.
// Cost = |ΔL| + grad_weight * (l1_l + l1_r)
//   where l1_x = (|gx| + |gy|) / 4  -- L1 Sobel, divide-by-4 matches Python.
// Non-overlap pixels -> 1e9 to steer the seam away.
// DP backtrack runs on CPU (serial per-row, < 0.5 ms).

#include "seam_cost.cuh"
#include "cuda_check.cuh"


namespace lidartps {

__global__ void seamCostU8Kernel(
    const uint8_t* __restrict__ d_left,
    const uint8_t* __restrict__ d_right,
    const uint8_t* __restrict__ d_valid_l,
    const uint8_t* __restrict__ d_valid_r,
    int W_c, int H_c,
    float grad_w,
    float* __restrict__ d_cost
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W_c || y >= H_c) return;

    int idx = y * W_c + x;

    if (!d_valid_l[idx] || !d_valid_r[idx]) {
        d_cost[idx] = 1e9f;
        return;
    }

    // d_left/d_right use 4-byte/pixel stride (R, G, B, padding).
    auto lum = [](const uint8_t* img, int i) -> float {
        return 0.299f*(float)img[i*4+2] + 0.587f*(float)img[i*4+1] + 0.114f*(float)img[i*4+0];
    };
    auto lumXY = [&](const uint8_t* img, int px, int py) -> float {
        px = max(0, min(W_c-1, px));
        py = max(0, min(H_c-1, py));
        return lum(img, py*W_c + px);
    };

    // Idea 4: full per-channel colour difference steers seam to where all channels agree,
    // not just luminance -- avoids chrominance-mismatch artifacts (magenta lines).
    float dB = fabsf((float)d_left[idx*4+0] - (float)d_right[idx*4+0]);
    float dG = fabsf((float)d_left[idx*4+1] - (float)d_right[idx*4+1]);
    float dR = fabsf((float)d_left[idx*4+2] - (float)d_right[idx*4+2]);
    float color_diff = (dB + dG + dR) * (1.f/3.f);

    // Sobel for left
    float gxl =  -lumXY(d_left,x-1,y-1) + lumXY(d_left,x+1,y-1)
               - 2.f*lumXY(d_left,x-1,y) + 2.f*lumXY(d_left,x+1,y)
               -  lumXY(d_left,x-1,y+1) + lumXY(d_left,x+1,y+1);
    float gyl =  -lumXY(d_left,x-1,y-1) - 2.f*lumXY(d_left,x,y-1) - lumXY(d_left,x+1,y-1)
               +  lumXY(d_left,x-1,y+1) + 2.f*lumXY(d_left,x,y+1) + lumXY(d_left,x+1,y+1);

    // Sobel for right
    float gxr =  -lumXY(d_right,x-1,y-1) + lumXY(d_right,x+1,y-1)
               - 2.f*lumXY(d_right,x-1,y) + 2.f*lumXY(d_right,x+1,y)
               -  lumXY(d_right,x-1,y+1) + lumXY(d_right,x+1,y+1);
    float gyr =  -lumXY(d_right,x-1,y-1) - 2.f*lumXY(d_right,x,y-1) - lumXY(d_right,x+1,y-1)
               +  lumXY(d_right,x-1,y+1) + 2.f*lumXY(d_right,x,y+1) + lumXY(d_right,x+1,y+1);

    // L1 Sobel per side, divide by 4 -- matches Python blend_with_seam formula
    float l1_l = (fabsf(gxl) + fabsf(gyl)) / 4.f;
    float l1_r = (fabsf(gxr) + fabsf(gyr)) / 4.f;

    d_cost[idx] = color_diff + grad_w * (l1_l + l1_r);
}

void computeSeamCostU8(
    const uint8_t* d_left, const uint8_t* d_right,
    const uint8_t* d_valid_left, const uint8_t* d_valid_right,
    int W_c, int H_c,
    float grad_weight,
    float* d_cost,
    cudaStream_t stream
) {
    dim3 block(32, 8);
    dim3 grid((W_c + 31) / 32, (H_c + 7) / 8);
    seamCostU8Kernel<<<grid, block, 0, stream>>>(
        d_left, d_right, d_valid_left, d_valid_right,
        W_c, H_c, grad_weight, d_cost
    );
    CUDA_CHECK(cudaGetLastError());
}

// -----------------------------------------------------------------------------
// GPU Seam DP -- forward pass + backtrack (port from lidar_pipeline)
// -----------------------------------------------------------------------------

__global__ void dpForwardKernel(
    const float* cost,
    float* dp,
    int* backtrack,
    int width, int height,
    int row
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= width) return;

    int idx = row * width + x;

    if (row == 0) {
        dp[idx] = cost[idx];
        backtrack[idx] = 0;
        return;
    }

    int prev_row = row - 1;
    float min_val = 3.402823466e+38f; // FLT_MAX
    int min_offset = 0;

    for (int dx = -1; dx <= 1; dx++) {
        int prev_x = x + dx;
        if (prev_x >= 0 && prev_x < width) {
            float val = dp[prev_row * width + prev_x];
            if (val < min_val) {
                min_val = val;
                min_offset = dx;
            }
        }
    }

    dp[idx] = cost[idx] + min_val;
    backtrack[idx] = min_offset;
}

// Single block: warp-shuffle argmin over the last DP row, then serial
// backtrack in thread 0. Ties prefer lower x (matches the serial scan).
__global__ void dpBacktrackKernel(
    const float* __restrict__ dp,
    const int*   __restrict__ backtrack,
    int width, int height,
    int* __restrict__ seam_out
) {
    constexpr int BLK = 256;
    constexpr int N_WARPS = BLK / 32;

    __shared__ float s_min_v[N_WARPS];
    __shared__ int   s_min_x[N_WARPS];

    const int tid       = threadIdx.x;
    const int warp_id   = tid >> 5;
    const int lane      = tid & 31;
    const int last_row  = height - 1;
    const int last_base = last_row * width;

    float local_min = 3.402823466e+38f;   // FLT_MAX
    int   local_x   = 0;
    for (int x = tid; x < width; x += BLK) {
        float v = dp[last_base + x];
        if (v < local_min) { local_min = v; local_x = x; }
    }

    #pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        float ov = __shfl_xor_sync(0xffffffff, local_min, delta);
        int   ox = __shfl_xor_sync(0xffffffff, local_x,   delta);
        if (ov < local_min || (ov == local_min && ox < local_x)) {
            local_min = ov;
            local_x   = ox;
        }
    }

    if (lane == 0) {
        s_min_v[warp_id] = local_min;
        s_min_x[warp_id] = local_x;
    }
    __syncthreads();

    if (warp_id == 0) {
        float warp_min = (lane < N_WARPS) ? s_min_v[lane] : 3.402823466e+38f;
        int   warp_x   = (lane < N_WARPS) ? s_min_x[lane] : 0;
        #pragma unroll
        for (int delta = N_WARPS / 2; delta > 0; delta >>= 1) {
            float ov = __shfl_xor_sync(0xffffffff, warp_min, delta);
            int   ox = __shfl_xor_sync(0xffffffff, warp_x,   delta);
            if (ov < warp_min || (ov == warp_min && ox < warp_x)) {
                warp_min = ov;
                warp_x   = ox;
            }
        }

        if (lane == 0) {
            seam_out[last_row] = warp_x;
            int next_x = warp_x;
            for (int row = last_row - 1; row >= 0; row--) {
                int offset = backtrack[(row + 1) * width + next_x];
                next_x     = max(0, min(width - 1, next_x + offset));
                seam_out[row] = next_x;
            }
        }
    }
}

// -----------------------------------------------------------------------------
// CUDA Graph seam DP
// -----------------------------------------------------------------------------

void findSeamDPGraph(
    const float* cost, int width, int height, int* seam_out,
    float* d_dp, int* d_backtrack,
    cudaGraph_t& io_graph, cudaGraphExec_t& io_exec, bool& io_ready,
    cudaStream_t stream
) {
    int block = 256;
    int grid  = (width + block - 1) / block;

    if (!io_ready) {
        // Capture the H forward-pass launches as a graph (records the fixed
        // device pointers; replay after refilling d_cost works).
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        for (int row = 0; row < height; row++) {
            dpForwardKernel<<<grid, block, 0, stream>>>(
                cost, d_dp, d_backtrack, width, height, row);
        }
        CUDA_CHECK(cudaStreamEndCapture(stream, &io_graph));
        CUDA_CHECK(cudaGraphInstantiate(&io_exec, io_graph, nullptr, nullptr, 0));
        io_ready = true;
    }

    CUDA_CHECK(cudaGraphLaunch(io_exec, stream));
    dpBacktrackKernel<<<1, 256, 0, stream>>>(d_dp, d_backtrack, width, height, seam_out);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace lidartps
