// gain.cu -- per-camera colour gain compensation (Idea 3).
// Overlap-zone mean computation via shared-memory reduction + atomic flush,
// followed by per-pixel gain application.

#include "gain.cuh"
#include "cuda_check.cuh"

namespace lidartps {

// -- Overlap mean computation --------------------------------------------------

__global__ void overlapSumKernel(
    const uint8_t* __restrict__ d_left,
    const uint8_t* __restrict__ d_right,
    const uint8_t* __restrict__ d_vl,
    const uint8_t* __restrict__ d_vr,
    int WH,
    unsigned int* out  // [7]: B_l,G_l,R_l, B_r,G_r,R_r, count
) {
    __shared__ unsigned int sh[7];
    if (threadIdx.x < 7) sh[threadIdx.x] = 0;
    __syncthreads();

    // d_left/d_right use 4-byte/pixel stride.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < WH && d_vl[i] && d_vr[i]) {
        atomicAdd(&sh[0], (unsigned int)d_left [i*4+0]);
        atomicAdd(&sh[1], (unsigned int)d_left [i*4+1]);
        atomicAdd(&sh[2], (unsigned int)d_left [i*4+2]);
        atomicAdd(&sh[3], (unsigned int)d_right[i*4+0]);
        atomicAdd(&sh[4], (unsigned int)d_right[i*4+1]);
        atomicAdd(&sh[5], (unsigned int)d_right[i*4+2]);
        atomicAdd(&sh[6], 1u);
    }
    __syncthreads();
    if (threadIdx.x < 7) atomicAdd(&out[threadIdx.x], sh[threadIdx.x]);
}

void computeOverlapMeans(
    const uint8_t* d_left,  const uint8_t* d_right,
    const uint8_t* d_vl,    const uint8_t* d_vr,
    int WH,
    unsigned int* d_accum,
    unsigned int* h_out,
    cudaStream_t stream
) {
    CUDA_CHECK(cudaMemsetAsync(d_accum, 0, 7 * sizeof(unsigned int), stream));
    int block = 256;
    int grid  = (WH + block - 1) / block;
    overlapSumKernel<<<grid, block, 0, stream>>>(
        d_left, d_right, d_vl, d_vr, WH, d_accum);
    // Optional D2H for debugging / metrics -- skipped when h_out is null so the
    // fused device-side applyGainFromAccum path incurs zero host-sync overhead.
    if (h_out) {
        CUDA_CHECK(cudaMemcpyAsync(h_out, d_accum, 7 * sizeof(unsigned int),
                                   cudaMemcpyDeviceToHost, stream));
    }
}

// -- Gain application ----------------------------------------------------------

__global__ void applyGainKernel(
    uint8_t* __restrict__ img, int WH,
    float gb, float gg, float gr
) {
    // img uses 4-byte/pixel stride; alpha lane (byte 3) untouched.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= WH) return;
    img[i*4+0] = (uint8_t)min(255, (int)(img[i*4+0] * gb + 0.5f));
    img[i*4+1] = (uint8_t)min(255, (int)(img[i*4+1] * gg + 0.5f));
    img[i*4+2] = (uint8_t)min(255, (int)(img[i*4+2] * gr + 0.5f));
}

void applyGain(
    uint8_t* d_img, int WH,
    float gain_b, float gain_g, float gain_r,
    cudaStream_t stream
) {
    int block = 256;
    int grid  = (WH + block - 1) / block;
    applyGainKernel<<<grid, block, 0, stream>>>(d_img, WH, gain_b, gain_g, gain_r);
    CUDA_CHECK(cudaGetLastError());
}

// -- Fused gain: read accum, compute clamped gain, apply (no host sync) -------
// Thread 0 resolves the gain into shared memory (pixel-count guarded), then
// the block applies it in parallel.
__global__ void applyGainFromAccumKernel(
    uint8_t* __restrict__ img, int WH,
    const unsigned int* __restrict__ accum,
    int src_base, int dst_base, int min_count
) {
    __shared__ float s_gain[3];
    __shared__ int   s_apply;   // 0 = skip, 1 = apply

    if (threadIdx.x == 0) {
        unsigned int cnt = accum[6];
        if (cnt < (unsigned int)min_count) {
            s_apply = 0;
        } else {
            auto sg = [](unsigned int tgt, unsigned int src) -> float {
                if (src == 0) return 1.f;
                float g = (float)tgt / (float)src;
                return fmaxf(0.5f, fminf(2.f, g));
            };
            s_gain[0] = sg(accum[dst_base+0], accum[src_base+0]);
            s_gain[1] = sg(accum[dst_base+1], accum[src_base+1]);
            s_gain[2] = sg(accum[dst_base+2], accum[src_base+2]);
            s_apply   = 1;
        }
    }
    __syncthreads();

    if (!s_apply) return;

    // img uses 4-byte/pixel stride; alpha lane (byte 3) untouched.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= WH) return;

    img[i*4+0] = (uint8_t)min(255, (int)(img[i*4+0] * s_gain[0] + 0.5f));
    img[i*4+1] = (uint8_t)min(255, (int)(img[i*4+1] * s_gain[1] + 0.5f));
    img[i*4+2] = (uint8_t)min(255, (int)(img[i*4+2] * s_gain[2] + 0.5f));
}

void applyGainFromAccum(
    uint8_t* d_img, int WH,
    const unsigned int* d_accum,
    int accum_src_base, int accum_dst_base,
    int min_count,
    cudaStream_t stream
) {
    int block = 256;
    int grid  = (WH + block - 1) / block;
    applyGainFromAccumKernel<<<grid, block, 0, stream>>>(
        d_img, WH, d_accum, accum_src_base, accum_dst_base, min_count);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace lidartps
