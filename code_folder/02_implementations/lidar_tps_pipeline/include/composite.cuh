#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace lidartps {

// Composite three warped canvas images using two per-row seam arrays.
// seam_lc[y]: FL<->FC seam column for row y
// seam_cr[y]: FC<->FR seam column for row y
// feather_half: ±px narrow feather band around each seam (0 = hard cut)
// Where only one camera has valid content, uses it directly (no mixing).
void compositeThreeU8(
    const uint8_t* d_warped_fl,
    const uint8_t* d_warped_fc,
    const uint8_t* d_warped_fr,
    const uint8_t* d_valid_fl,
    const uint8_t* d_valid_fc,
    const uint8_t* d_valid_fr,
    const int* d_seam_lc,
    const int* d_seam_cr,
    int feather_half,
    int W_c, int H_c,
    uint8_t* d_canvas,
    cudaStream_t stream
);

} // namespace lidartps
