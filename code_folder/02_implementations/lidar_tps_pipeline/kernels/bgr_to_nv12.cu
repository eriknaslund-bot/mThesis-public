// bgr_to_nv12.cu
// Converts a WxH interleaved BGR canvas (the pipeline's d_canvas_buf, stride 3)
// into NV12 -- the YUV 4:2:0 layout NVENC consumes -- entirely on-device.

#include "bgr_to_nv12.cuh"
#include "cuda_check.cuh"
#include <cstdint>

namespace lidartps {

__global__ void bgrToNv12Kernel(
    const uint8_t* __restrict__ bgr,
    uint8_t*       __restrict__ y_plane,
    uint8_t*       __restrict__ uv_plane,
    int W, int H,
    int y_stride, int uv_stride,
    bool y_only)
{
    int bx = blockIdx.x * blockDim.x + threadIdx.x;  // 2x2 block column
    int by = blockIdx.y * blockDim.y + threadIdx.y;  // 2x2 block row
    int W2 = W >> 1, H2 = H >> 1;
    if (bx >= W2 || by >= H2) return;

    int x0 = bx << 1;
    int y0 = by << 1;

    auto load = [&](int x, int y, int& B, int& G, int& R) {
        const uint8_t* p = bgr + (y * W + x) * 3;
        B = p[0]; G = p[1]; R = p[2];
    };

    int B00, G00, R00, B10, G10, R10, B01, G01, R01, B11, G11, R11;
    load(x0,   y0,   B00, G00, R00);
    load(x0+1, y0,   B10, G10, R10);
    load(x0,   y0+1, B01, G01, R01);
    load(x0+1, y0+1, B11, G11, R11);

    // Y = (76·R + 150·G + 29·B + 128) >> 8  (BT.601, full range)
    auto Y = [](int B, int G, int R) -> uint8_t {
        int y = (76*R + 150*G + 29*B + 128) >> 8;
        return (uint8_t)max(0, min(255, y));
    };

    y_plane[(y0  ) * y_stride + x0    ] = Y(B00, G00, R00);
    y_plane[(y0  ) * y_stride + x0 + 1] = Y(B10, G10, R10);
    y_plane[(y0+1) * y_stride + x0    ] = Y(B01, G01, R01);
    y_plane[(y0+1) * y_stride + x0 + 1] = Y(B11, G11, R11);

    int uv_off = by * uv_stride + (bx << 1);
    if (y_only) {
        // Neutral chroma -- used by the bitrate-weight calibration encode.
        uv_plane[uv_off    ] = 128;
        uv_plane[uv_off + 1] = 128;
        return;
    }

    int Bavg = (B00 + B10 + B01 + B11) >> 2;
    int Gavg = (G00 + G10 + G01 + G11) >> 2;
    int Ravg = (R00 + R10 + R01 + R11) >> 2;

    // U = ((-43·R − 84·G + 128·B + 128) >> 8) + 128
    // V = (( 128·R − 107·G − 21·B + 128) >> 8) + 128
    int u = ((-43*Ravg - 84*Gavg + 128*Bavg + 128) >> 8) + 128;
    int v = ((128*Ravg - 107*Gavg -  21*Bavg + 128) >> 8) + 128;
    u = max(0, min(255, u));
    v = max(0, min(255, v));

    uv_plane[uv_off    ] = (uint8_t)u;
    uv_plane[uv_off + 1] = (uint8_t)v;
}

void bgrToNv12(
    const uint8_t* d_bgr,
    uint8_t*       d_y_plane,
    uint8_t*       d_uv_plane,
    int W, int H,
    int y_stride, int uv_stride,
    cudaStream_t stream,
    bool y_only)
{
    int W2 = W >> 1, H2 = H >> 1;
    dim3 block(16, 16);
    dim3 grid((W2 + 15) / 16, (H2 + 15) / 16);
    bgrToNv12Kernel<<<grid, block, 0, stream>>>(
        d_bgr, d_y_plane, d_uv_plane, W, H, y_stride, uv_stride, y_only);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace lidartps
