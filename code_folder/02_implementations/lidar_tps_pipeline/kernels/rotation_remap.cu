// rotation_remap.cu
// Stage 6: build the cylindrical rotation remap (once at init) and apply it
// per-frame (one thread per output canvas pixel, bilinear sampling).

#include "rotation_remap.cuh"
#include "cuda_check.cuh"
#include <cmath>

namespace lidartps {

// -- Build rotation remap ------------------------------------------------------

__global__ void buildRotRemapKernel(
    int W_c, int H_c,
    Mat3f  R_cam_ego,
    float  fx, float fy, float cx, float cy,
    int    src_W, int src_H,
    float  fx_canvas, float fy_canvas, float cx_canvas, float cy_canvas,
    float2*  __restrict__ d_remap,
    uint8_t* __restrict__ d_valid_mask
) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= W_c || v >= H_c) return;

    int idx = v * W_c + u;

    // Canvas pixel -> azimuth / elevation (true-cylindrical: el = atan(v/fy))
    float az = (cx_canvas - (float)u) / fx_canvas;
    float v_scaled = (cy_canvas - (float)v) / fy_canvas;
    float el = atanf(v_scaled);

    // Ego-frame direction (unit vector on the unit cylinder)
    float cos_el = cosf(el);
    float3 ego_dir = make_float3(
        cos_el * cosf(az),
        cos_el * sinf(az),
        sinf(el)
    );

    // Camera-frame direction: R^T * ego_dir  (R is cam->ego, R^T is ego->cam)
    float3 cam_dir = R_cam_ego.T() * ego_dir;

    if (cam_dir.z <= 1e-6f) {
        d_remap[idx]      = make_float2(-1.f, -1.f);
        d_valid_mask[idx] = 0;
        return;
    }

    float u_cam = fx * cam_dir.x / cam_dir.z + cx;
    float v_cam = fy * cam_dir.y / cam_dir.z + cy;

    // Also validate that the projected pixel is within the source image bounds
    if (u_cam < 0.f || v_cam < 0.f || u_cam >= (float)src_W || v_cam >= (float)src_H) {
        d_remap[idx]      = make_float2(-1.f, -1.f);
        d_valid_mask[idx] = 0;
        return;
    }

    d_remap[idx]      = make_float2(u_cam, v_cam);
    d_valid_mask[idx] = 1;
}

void buildRotRemap(
    int W_c, int H_c,
    Mat3f R_cam_ego,
    float fx, float fy, float cx, float cy,
    int src_W, int src_H,
    float fx_canvas, float fy_canvas, float cx_canvas, float cy_canvas,
    float2* d_remap, uint8_t* d_valid_mask,
    cudaStream_t stream
) {
    dim3 block(32, 8);
    dim3 grid((W_c + 31) / 32, (H_c + 7) / 8);
    buildRotRemapKernel<<<grid, block, 0, stream>>>(
        W_c, H_c, R_cam_ego, fx, fy, cx, cy,
        src_W, src_H,
        fx_canvas, fy_canvas, cx_canvas, cy_canvas,
        d_remap, d_valid_mask
    );
    CUDA_CHECK(cudaGetLastError());
}

// -- Apply remap ---------------------------------------------------------------

__global__ void applyRemapU8Kernel(
    const uint8_t* __restrict__ d_src,
    int src_W, int src_H,
    const float2* __restrict__ d_remap,
    int W_c, int H_c,
    uint8_t* __restrict__ d_dst
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W_c || y >= H_c) return;

    int out_idx = y * W_c + x;
    float2 src_uv = d_remap[out_idx];
    float su = src_uv.x, sv = src_uv.y;

    // d_dst is W*H*4 (R,G,B,padding). 4th byte stays 0 from the init memset.
    if (su < 0.f || sv < 0.f || su >= (float)src_W || sv >= (float)src_H) {
        d_dst[out_idx * 4 + 0] = 0;
        d_dst[out_idx * 4 + 1] = 0;
        d_dst[out_idx * 4 + 2] = 0;
        return;
    }

    int u0 = (int)floorf(su);
    int v0 = (int)floorf(sv);
    int u1 = min(u0 + 1, src_W - 1);
    int v1 = min(v0 + 1, src_H - 1);
    u0 = max(0, u0);
    v0 = max(0, v0);

    float fu = su - (float)u0;
    float fv = sv - (float)v0;
    float w00 = (1.f-fu)*(1.f-fv);
    float w10 =     fu *(1.f-fv);
    float w01 = (1.f-fu)*    fv;
    float w11 =     fu *    fv;

    const int i00 = (v0*src_W + u0)*3;
    const int i10 = (v0*src_W + u1)*3;
    const int i01 = (v1*src_W + u0)*3;
    const int i11 = (v1*src_W + u1)*3;

    d_dst[out_idx * 4 + 0] = (uint8_t)__float2uint_rn(__fmaf_rn(w00,(float)d_src[i00],   __fmaf_rn(w10,(float)d_src[i10],   __fmaf_rn(w01,(float)d_src[i01],   w11*(float)d_src[i11]))));
    d_dst[out_idx * 4 + 1] = (uint8_t)__float2uint_rn(__fmaf_rn(w00,(float)d_src[i00+1], __fmaf_rn(w10,(float)d_src[i10+1], __fmaf_rn(w01,(float)d_src[i01+1], w11*(float)d_src[i11+1]))));
    d_dst[out_idx * 4 + 2] = (uint8_t)__float2uint_rn(__fmaf_rn(w00,(float)d_src[i00+2], __fmaf_rn(w10,(float)d_src[i10+2], __fmaf_rn(w01,(float)d_src[i01+2], w11*(float)d_src[i11+2]))));
}

void applyRemapU8(
    const uint8_t* d_src, int src_W, int src_H,
    const float2* d_remap,
    int W_c, int H_c,
    uint8_t* d_dst,
    cudaStream_t stream
) {
    dim3 block(32, 8);
    dim3 grid((W_c + 31) / 32, (H_c + 7) / 8);
    applyRemapU8Kernel<<<grid, block, 0, stream>>>(
        d_src, src_W, src_H, d_remap, W_c, H_c, d_dst
    );
    CUDA_CHECK(cudaGetLastError());
}

} // namespace lidartps
