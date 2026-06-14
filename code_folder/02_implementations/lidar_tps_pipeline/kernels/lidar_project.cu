// lidar_project.cu
// Stage 1: project ego-frame LiDAR points into each camera and onto the
// cylindrical canvas.  One thread per LiDAR point.
//
// AV2 convention: points are already in ego frame.
// Output is per-camera image-plane projection only; canvas-space coords are
// computed CPU-side from the camera pixels via camPixToCanvas() when needed.

#include "lidar_project.cuh"
#include "cuda_check.cuh"
#include <cmath>

namespace lidartps {

__global__ void projectLidarEgoKernel(
    const float* __restrict__ d_pts,
    int N,
    Mat3f  R_cam_ego,
    float3 t_cam_ego,
    float  fx, float fy, float cx, float cy,
    int    cam_W, int cam_H,
    float  max_range_m,
    float*   __restrict__ d_cam_uv,
    float*   __restrict__ d_depth,
    uint8_t* __restrict__ d_valid
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float ex = d_pts[idx*3 + 0];
    float ey = d_pts[idx*3 + 1];
    float ez = d_pts[idx*3 + 2];

    float range = sqrtf(ex*ex + ey*ey + ez*ez);

    // ego -> camera: cam_pt = R^T * (ego_pt - t)
    float3 delta = make_float3(ex - t_cam_ego.x, ey - t_cam_ego.y, ez - t_cam_ego.z);
    float3 cam_pt = R_cam_ego.T() * delta;

    // default: invalid
    d_cam_uv[idx*2]       = -1.f;
    d_cam_uv[idx*2 + 1]   = -1.f;
    d_depth[idx]           = 0.f;
    d_valid[idx]           = 0;

    if (cam_pt.z < 0.5f || range > max_range_m) return;

    float u_cam = fx * cam_pt.x / cam_pt.z + cx;
    float v_cam = fy * cam_pt.y / cam_pt.z + cy;

    if (u_cam < 0.f || u_cam >= (float)cam_W ||
        v_cam < 0.f || v_cam >= (float)cam_H) return;

    d_cam_uv[idx*2]        = u_cam;
    d_cam_uv[idx*2 + 1]    = v_cam;
    d_depth[idx]            = range;
    d_valid[idx]            = 1;
}

void projectLidarEgoGpu(
    const float* d_pts, int N,
    Mat3f R_cam_ego, float3 t_cam_ego,
    float fx, float fy, float cx, float cy,
    int cam_W, int cam_H,
    float max_range_m,
    float* d_cam_uv,
    float* d_depth, uint8_t* d_valid,
    cudaStream_t stream
) {
    if (N <= 0) return;
    int block = 256;
    int grid  = (N + block - 1) / block;
    projectLidarEgoKernel<<<grid, block, 0, stream>>>(
        d_pts, N, R_cam_ego, t_cam_ego,
        fx, fy, cx, cy, cam_W, cam_H,
        max_range_m,
        d_cam_uv, d_depth, d_valid
    );
    CUDA_CHECK(cudaGetLastError());
}

} // namespace lidartps
