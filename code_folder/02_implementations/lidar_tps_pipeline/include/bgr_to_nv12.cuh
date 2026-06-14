#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace lidartps {

// Convert a WxH interleaved BGR canvas to NV12 layout (Y plane WxH + interleaved
// UV plane WxH/2) entirely on the device. Caller is responsible for allocating
// the Y / UV plane buffers with the strides reported by their consumer (e.g.
// libav's hwframes_ctx pool); both planes may have stride > W due to alignment
// padding. W and H must both be even (NVENC requirement). BT.601 integer math.
//
// When `y_only == true`, the UV plane is filled with 128 (neutral grey)
// instead of the computed Cb/Cr -- used by the bitrate-weight calibration
// path to encode the same Y plane with the chroma signal removed. The
// resulting MP4 size delta against the full-colour encode is the chroma
// bit cost, used to derive the per-channel composite-PSNR weights.
void bgrToNv12(
    const uint8_t* d_bgr,
    uint8_t*       d_y_plane,
    uint8_t*       d_uv_plane,
    int W, int H,
    int y_stride, int uv_stride,
    cudaStream_t stream,
    bool y_only = false);

} // namespace lidartps
