// tps_remap.cu -- apply TPS displacement on top of the rotation baseline.
// Per canvas pixel: final_src = d_remap[P] + TPS(x_n,y_n); dst = bilinear(src).
// TPS (Bookstein): disp = affine(wx[N..N+2]) + Σ wx[i]·U(‖p−ctrl_i‖²),
//   U(r²) = ½·r²·log(r²).

#include "tps_remap.cuh"
#include "cuda_check.cuh"
#include <cmath>

namespace lidartps {

// -- Temporal IIR on TPS disp field -------------------------------------------
// Mix new disp with the previous snapshot; write the result to both buffers
// so the snapshot matches what was rendered.
__global__ void dispIIRBlendKernel(
    float* __restrict__ d_disp_x,
    float* __restrict__ d_disp_y,
    float* __restrict__ d_disp_x_prev,
    float* __restrict__ d_disp_y_prev,
    int N, float alpha)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float omega = 1.f - alpha;
    float mx = alpha * d_disp_x[i] + omega * d_disp_x_prev[i];
    float my = alpha * d_disp_y[i] + omega * d_disp_y_prev[i];
    d_disp_x[i]      = mx;
    d_disp_y[i]      = my;
    d_disp_x_prev[i] = mx;
    d_disp_y_prev[i] = my;
}

void applyDispTemporalIIR(
    float* d_disp_x, float* d_disp_y,
    float* d_disp_x_prev, float* d_disp_y_prev,
    int W_h, int H_h, float alpha,
    cudaStream_t stream)
{
    if (alpha >= 1.f) return;  // no blend; caller snapshots via a separate D2D copy
    int N = W_h * H_h;
    int block = 256;
    int grid  = (N + block - 1) / block;
    dispIIRBlendKernel<<<grid, block, 0, stream>>>(
        d_disp_x, d_disp_y, d_disp_x_prev, d_disp_y_prev, N, alpha);
    CUDA_CHECK(cudaGetLastError());
}

__device__ __forceinline__ float tpsU(float r2) {
    return (r2 > 1e-6f) ? 0.5f * r2 * logf(r2) : 0.f;
}

// -- Pass 1: evaluate TPS displacement on a coarse W_h x H_h grid -------------
// Each pixel loops over all N ctrl pts. Ctrl pts + weights are staged in
// __shared__ memory (broadcast reads); this was ~69% of GPU time, the main
// optimisation target. Smem ~8.25 KB/block at N_MAX=512; occupancy unchanged.

static constexpr int EVAL_TPS_N_MAX = 512;   // matches N_MAX_CTRL in pipeline.cpp

__global__ void evalTpsDispKernelShared(
    int W_h, int H_h,
    const float* __restrict__ d_ctrl_x,
    const float* __restrict__ d_ctrl_y,
    const float* __restrict__ d_wx,
    const float* __restrict__ d_wy,
    int N,
    float* __restrict__ d_disp_x,
    float* __restrict__ d_disp_y
) {
    // Each thread emits two pixels (x0, x1=x0+blockDim.x) sharing the same row
    // and weights: the two logf-bound chains overlap (register ILP). FP order
    // matches the 1-pixel version, so output is bit-identical.
    __shared__ float2 s_ctrl[EVAL_TPS_N_MAX];
    __shared__ float2 s_w   [EVAL_TPS_N_MAX + 3];

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int bsz = blockDim.x * blockDim.y;

    // Cooperative load: ctrl has N entries, w has N+3 (affine triple last).
    for (int i = tid; i < N; i += bsz) {
        s_ctrl[i] = make_float2(d_ctrl_x[i], d_ctrl_y[i]);
    }
    for (int i = tid; i < N + 3; i += bsz) {
        s_w[i] = make_float2(d_wx[i], d_wy[i]);
    }
    __syncthreads();

    // Block covers 2*blockDim.x columns; x0/x1 layout keeps both writes coalesced.
    const int x0 = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    const int x1 = x0 + blockDim.x;
    const int y  = blockIdx.y * blockDim.y + threadIdx.y;
    if (y >= H_h) return;

    const bool x0_valid = (x0 < W_h);
    const bool x1_valid = (x1 < W_h);
    if (!x0_valid && !x1_valid) return;

    float u_n0 = (W_h > 1) ? (float)x0 / (float)(W_h - 1) : 0.f;
    float u_n1 = (W_h > 1) ? (float)x1 / (float)(W_h - 1) : 0.f;
    float v_n  = (H_h > 1) ? (float)y  / (float)(H_h - 1) : 0.f;

    float2 wN  = s_w[N];
    float2 wN1 = s_w[N+1];
    float2 wN2 = s_w[N+2];
    float disp_x0 = wN.x + wN1.x*u_n0 + wN2.x*v_n;
    float disp_y0 = wN.y + wN1.y*u_n0 + wN2.y*v_n;
    float disp_x1 = wN.x + wN1.x*u_n1 + wN2.x*v_n;
    float disp_y1 = wN.y + wN1.y*u_n1 + wN2.y*v_n;

    for (int i = 0; i < N; i++) {
        float2 c    = s_ctrl[i];
        float  dx0  = u_n0 - c.x;
        float  dx1  = u_n1 - c.x;
        float  dy   = v_n  - c.y;
        float  dy2  = dy * dy;
        float  r2_0 = dx0*dx0 + dy2;
        float  r2_1 = dx1*dx1 + dy2;
        float  phi0 = tpsU(r2_0);
        float  phi1 = tpsU(r2_1);
        float2 w    = s_w[i];
        disp_x0 += w.x * phi0;
        disp_y0 += w.y * phi0;
        disp_x1 += w.x * phi1;
        disp_y1 += w.y * phi1;
    }

    if (x0_valid) {
        d_disp_x[y * W_h + x0] = disp_x0;
        d_disp_y[y * W_h + x0] = disp_y0;
    }
    if (x1_valid) {
        d_disp_x[y * W_h + x1] = disp_x1;
        d_disp_y[y * W_h + x1] = disp_y1;
    }
}

void evalTpsDispHalf(
    int W_h, int H_h,
    const float* d_ctrl_x, const float* d_ctrl_y,
    const float* d_wx, const float* d_wy,
    int N_ctrl,
    float* d_disp_x, float* d_disp_y,
    cudaStream_t stream
) {
    // 256 threads, 2 pixels each along x -> x-grid halved (ceil(W_h/64)).
    dim3 block(32, 8);
    dim3 grid((W_h + 63) / 64, (H_h + 7) / 8);
    evalTpsDispKernelShared<<<grid, block, 0, stream>>>(
        W_h, H_h, d_ctrl_x, d_ctrl_y, d_wx, d_wy, N_ctrl, d_disp_x, d_disp_y
    );
    CUDA_CHECK(cudaGetLastError());
}

// -- Pass 2: apply rotation remap + bilinear-upsampled TPS displacement --------
//
// NOTE: shared-memory tiling of the disp fields was tried and regressed ~25%
// on SM 8.6 (L1 already serves the 4-tap bilinear; load+sync cost more). Kept
// the plain global-read version.

__global__ void applyRemapWithDispU8Kernel(
    const uint8_t* __restrict__ d_src,
    int src_W, int src_H,
    const float2* __restrict__ d_remap,
    int W_c, int H_c,
    int W_h, int H_h,
    const float* __restrict__ d_disp_x,
    const float* __restrict__ d_disp_y,
    uint8_t* __restrict__ d_dst,
    uint8_t* __restrict__ d_vmask
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W_c || y >= H_c) return;

    int out_idx = y * W_c + x;
    float2 rot = d_remap[out_idx];

    if (rot.x < 0.f) {
        // d_dst is W*H*4 (R,G,B,padding). Match the post-disp invalid-exit below.
        d_dst[out_idx * 4 + 0] = 0;
        d_dst[out_idx * 4 + 1] = 0;
        d_dst[out_idx * 4 + 2] = 0;
        d_vmask[out_idx] = 0;
        return;
    }

    // Bilinear lookup in the coarse displacement grid
    float hx = (W_h > 1) ? (float)x * (float)(W_h - 1) / (float)(W_c - 1) : 0.f;
    float hy = (H_h > 1) ? (float)y * (float)(H_h - 1) / (float)(H_c - 1) : 0.f;
    int hx0 = max(0, (int)floorf(hx));
    int hy0 = max(0, (int)floorf(hy));
    int hx1 = min(hx0 + 1, W_h - 1);
    int hy1 = min(hy0 + 1, H_h - 1);
    float fhx = hx - (float)hx0;
    float fhy = hy - (float)hy0;

    auto bilin = [&](const float* buf) -> float {
        return (1.f-fhx)*(1.f-fhy)*buf[hy0*W_h+hx0]
             +     fhx *(1.f-fhy)*buf[hy0*W_h+hx1]
             + (1.f-fhx)*    fhy *buf[hy1*W_h+hx0]
             +     fhx *    fhy *buf[hy1*W_h+hx1];
    };

    float su = rot.x + bilin(d_disp_x);
    float sv = rot.y + bilin(d_disp_y);

    // If the TPS disp pushes the sample outside the source, edge-clamping
    // would smear the border column -- invalidate instead so seam/feather
    // route around it. d_dst is W*H*4 (BGR + padding byte).
    if (su < 0.f || su > (float)(src_W - 1) - 0.001f ||
        sv < 0.f || sv > (float)(src_H - 1) - 0.001f) {
        d_dst[out_idx * 4 + 0] = 0;
        d_dst[out_idx * 4 + 1] = 0;
        d_dst[out_idx * 4 + 2] = 0;
        d_vmask[out_idx] = 0;
        return;
    }

    int u0 = (int)floorf(su); int v0 = (int)floorf(sv);
    int u1 = min(u0+1, src_W-1); int v1 = min(v0+1, src_H-1);
    u0 = max(0, u0); v0 = max(0, v0);
    float fu = su-(float)u0, fv = sv-(float)v0;
    float w00=(1.f-fu)*(1.f-fv), w10=fu*(1.f-fv), w01=(1.f-fu)*fv, w11=fu*fv;
    const int i00=(v0*src_W+u0)*3, i10=(v0*src_W+u1)*3;
    const int i01=(v1*src_W+u0)*3, i11=(v1*src_W+u1)*3;
    uint8_t b = (uint8_t)__float2uint_rn(__fmaf_rn(w00,(float)d_src[i00],   __fmaf_rn(w10,(float)d_src[i10],   __fmaf_rn(w01,(float)d_src[i01],   w11*(float)d_src[i11]))));
    uint8_t g = (uint8_t)__float2uint_rn(__fmaf_rn(w00,(float)d_src[i00+1], __fmaf_rn(w10,(float)d_src[i10+1], __fmaf_rn(w01,(float)d_src[i01+1], w11*(float)d_src[i11+1]))));
    uint8_t r = (uint8_t)__float2uint_rn(__fmaf_rn(w00,(float)d_src[i00+2], __fmaf_rn(w10,(float)d_src[i10+2], __fmaf_rn(w01,(float)d_src[i01+2], w11*(float)d_src[i11+2]))));
    d_dst[out_idx * 4 + 0] = b;
    d_dst[out_idx * 4 + 1] = g;
    d_dst[out_idx * 4 + 2] = r;
    // AV2 rectification-black corners: an in-bounds coord can still sample a
    // (0,0,0) pixel. Mark those invalid so they don't bias seam/gain/metrics.
    d_vmask[out_idx] = ((b | g | r) != 0) ? 1 : 0;
}

void applyRemapWithDispU8(
    const uint8_t* d_src, int src_W, int src_H,
    const float2* d_remap,
    int W_c, int H_c,
    int W_h, int H_h,
    const float* d_disp_x, const float* d_disp_y,
    uint8_t* d_dst,
    uint8_t* d_vmask,
    cudaStream_t stream
) {
    dim3 block(32, 8);
    dim3 grid((W_c + 31) / 32, (H_c + 7) / 8);
    applyRemapWithDispU8Kernel<<<grid, block, 0, stream>>>(
        d_src, src_W, src_H,
        d_remap,
        W_c, H_c, W_h, H_h,
        d_disp_x, d_disp_y,
        d_dst, d_vmask
    );
    CUDA_CHECK(cudaGetLastError());
}

} // namespace lidartps
