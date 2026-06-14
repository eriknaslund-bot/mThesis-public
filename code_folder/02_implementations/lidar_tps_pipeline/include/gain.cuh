#pragma once
// Per-camera colour gain: match FL/FR to FC's per-channel mean over the
// overlap, before seam DP and compositing.
#include <cuda_runtime.h>
#include <cstdint>

namespace lidartps {

// Per-channel BGR sums over the overlap (both valid masks = 1) -> h_out
// (pinned, 7 uint32): [0..2]=left BGR, [3..5]=right BGR, [6]=pixel count.
// d_accum is device scratch [7] (zeroed internally). Sync before reading h_out.
void computeOverlapMeans(
    const uint8_t* d_left,  const uint8_t* d_right,
    const uint8_t* d_vl,    const uint8_t* d_vr,
    int WH,
    unsigned int* d_accum,
    unsigned int* h_out,
    cudaStream_t stream
);

// Apply per-channel multiplicative gain to a BGR image in-place.
// Each channel saturates to [0, 255].
void applyGain(
    uint8_t* d_img, int WH,
    float gain_b, float gain_g, float gain_r,
    cudaStream_t stream
);

// Fused device-side variant: reads d_accum and applies gain in one launch
// (no host sync). *_base index the source/target BGR triplets (0 or 3).
// Skips if pixel count < min_count. Gain clamped to [0.5, 2.0].
void applyGainFromAccum(
    uint8_t* d_img, int WH,
    const unsigned int* d_accum,     // [7] overlap sums, same layout as computeOverlapMeans
    int accum_src_base,              // 0 for left-is-source, 3 for right-is-source
    int accum_dst_base,              // 3 or 0 (complement of src)
    int min_count,                   // skip gain if pixel count below this
    cudaStream_t stream
);

} // namespace lidartps
