// composite.cu
// Stage 10: composite FL, FC, FR warped canvas images using two per-row DP seams.
// Feather blend of ±feather_half px around each seam.
// Where only one camera has valid content, use it directly (no black mixing).

#include "composite.cuh"
#include "cuda_check.cuh"

namespace lidartps {

__global__ void compositeThreeU8Kernel(
    const uint8_t* __restrict__ d_fl,
    const uint8_t* __restrict__ d_fc,
    const uint8_t* __restrict__ d_fr,
    const uint8_t* __restrict__ d_vfl,
    const uint8_t* __restrict__ d_vfc,
    const uint8_t* __restrict__ d_vfr,
    const int* __restrict__ d_seam_lc,
    const int* __restrict__ d_seam_cr,
    int feather_half,
    int W_c, int H_c,
    uint8_t* __restrict__ d_out
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W_c || y >= H_c) return;

    int idx = y * W_c + x;

    int slc = d_seam_lc[y];   // FL<->FC seam column
    int scr = d_seam_cr[y];   // FC<->FR seam column

    bool have_fl = d_vfl[idx] != 0;
    bool have_fc = d_vfc[idx] != 0;
    bool have_fr = d_vfr[idx] != 0;

    // Blend weight for FL at the FL<->FC seam:  1 = pure FL, 0 = pure FC
    // Smoothstep matches Python: t*t*(3-2*t) where t = clip((seam+fh-col)/(2*fh), 0,1)
    float alpha_lc;
    if (feather_half <= 0) {
        alpha_lc = (x <= slc) ? 1.f : 0.f;
    } else {
        float t = (float)(x - slc) / (float)feather_half;
        float s = fmaxf(0.f, fminf(1.f, 0.5f - 0.5f*t));
        alpha_lc = s*s*(3.f - 2.f*s);
    }

    // Blend weight for FC at the FC<->FR seam: 1 = pure FC, 0 = pure FR
    float alpha_cr;
    if (feather_half <= 0) {
        alpha_cr = (x <= scr) ? 1.f : 0.f;
    } else {
        float t = (float)(x - scr) / (float)feather_half;
        float s = fmaxf(0.f, fminf(1.f, 0.5f - 0.5f*t));
        alpha_cr = s*s*(3.f - 2.f*s);
    }

    // Determine pixel colour
    float3 col;

    if (alpha_lc >= 1.f) {
        // Pure FL zone
        if (have_fl) col = make_float3(d_fl[idx*4], d_fl[idx*4+1], d_fl[idx*4+2]);
        else if (have_fc) col = make_float3(d_fc[idx*4], d_fc[idx*4+1], d_fc[idx*4+2]);
        else col = make_float3(0, 0, 0);

    } else if (alpha_cr <= 0.f) {
        // Pure FR zone
        if (have_fr) col = make_float3(d_fr[idx*4], d_fr[idx*4+1], d_fr[idx*4+2]);
        else if (have_fc) col = make_float3(d_fc[idx*4], d_fc[idx*4+1], d_fc[idx*4+2]);
        else col = make_float3(0, 0, 0);

    } else if (alpha_lc > 0.f && alpha_lc < 1.f) {
        // FL<->FC blend zone
        float3 cfl = have_fl ? make_float3(d_fl[idx*4], d_fl[idx*4+1], d_fl[idx*4+2])
                             : make_float3(0, 0, 0);
        float3 cfc = have_fc ? make_float3(d_fc[idx*4], d_fc[idx*4+1], d_fc[idx*4+2])
                             : make_float3(0, 0, 0);
        if (!have_fl) col = cfc;
        else if (!have_fc) col = cfl;
        else col = make_float3(
            alpha_lc*cfl.x + (1.f-alpha_lc)*cfc.x,
            alpha_lc*cfl.y + (1.f-alpha_lc)*cfc.y,
            alpha_lc*cfl.z + (1.f-alpha_lc)*cfc.z);

    } else if (alpha_cr > 0.f && alpha_cr < 1.f) {
        // FC<->FR blend zone
        float3 cfc = have_fc ? make_float3(d_fc[idx*4], d_fc[idx*4+1], d_fc[idx*4+2])
                             : make_float3(0, 0, 0);
        float3 cfr = have_fr ? make_float3(d_fr[idx*4], d_fr[idx*4+1], d_fr[idx*4+2])
                             : make_float3(0, 0, 0);
        if (!have_fc) col = cfr;
        else if (!have_fr) col = cfc;
        else col = make_float3(
            alpha_cr*cfc.x + (1.f-alpha_cr)*cfr.x,
            alpha_cr*cfc.y + (1.f-alpha_cr)*cfr.y,
            alpha_cr*cfc.z + (1.f-alpha_cr)*cfr.z);

    } else {
        // Pure FC zone (between the two seams)
        if (have_fc) col = make_float3(d_fc[idx*4], d_fc[idx*4+1], d_fc[idx*4+2]);
        else if (have_fl && x < W_c/2) col = make_float3(d_fl[idx*4], d_fl[idx*4+1], d_fl[idx*4+2]);
        else if (have_fr) col = make_float3(d_fr[idx*4], d_fr[idx*4+1], d_fr[idx*4+2]);
        else col = make_float3(0, 0, 0);
    }

    d_out[idx*3]   = (uint8_t)__float2uint_rn(fminf(255.f, fmaxf(0.f, col.x)));
    d_out[idx*3+1] = (uint8_t)__float2uint_rn(fminf(255.f, fmaxf(0.f, col.y)));
    d_out[idx*3+2] = (uint8_t)__float2uint_rn(fminf(255.f, fmaxf(0.f, col.z)));
}

void compositeThreeU8(
    const uint8_t* d_fl, const uint8_t* d_fc, const uint8_t* d_fr,
    const uint8_t* d_vfl, const uint8_t* d_vfc, const uint8_t* d_vfr,
    const int* d_seam_lc, const int* d_seam_cr,
    int feather_half, int W_c, int H_c,
    uint8_t* d_canvas,
    cudaStream_t stream
) {
    dim3 block(32, 8);
    dim3 grid((W_c + 31) / 32, (H_c + 7) / 8);
    compositeThreeU8Kernel<<<grid, block, 0, stream>>>(
        d_fl, d_fc, d_fr, d_vfl, d_vfc, d_vfr,
        d_seam_lc, d_seam_cr,
        feather_half, W_c, H_c, d_canvas
    );
    CUDA_CHECK(cudaGetLastError());
}

} // namespace lidartps
