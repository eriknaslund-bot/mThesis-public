// lidar_tps_pipeline.cpp
// Host-side orchestration for the LiDAR-TPS stitching pipeline.
//
// Per-frame stages:
//   1  GPU  LiDAR projection (3 cameras in parallel)
//   2  CPU  Shared-point extraction, range filter, spatial-grid subsample
//   3  CPU  Canvas-pixel targets (FC at rotation baseline, FL/FR carry the warp)
//   4  CPU  Pack ctrl pts and H2D upload
//   5  CPU  TPS solve (N+3)x(N+3) per side camera, two RHS
//   6  GPU  Rotation remap + quarter-res TPS disp + bilinear-upsample apply
//   7  GPU  Per-channel gain compensation (FL/FR matched to FC)
//   8  GPU  Seam-cost + CUDA-graph DP forward + GPU backtrack
//   9  GPU  Cosine-feather composite
//  10  D2H  Canvas to cv::Mat

#include "lidar_tps_pipeline.hpp"
#include "lidar_project.cuh"
#include "rotation_remap.cuh"
#include "tps_remap.cuh"
#include "seam_cost.cuh"
#include "composite.cuh"
#include "gain.cuh"
#include "cuda_check.cuh"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
namespace lidartps {

static constexpr int N_MAX_PTS  = 200000;  // max LiDAR pts / sweep
static constexpr int N_MAX_CTRL = 1024;    // max TPS ctrl pts per camera

// Tuned-once internal constants. Were `LidarTpsConfig` fields with no CLI flag,
// hoisted here so the public config only carries knobs that actually get swept.
static constexpr float        K_LIDAR_MAX_RANGE_M   = 120.0f;
static constexpr float        K_CANVAS_MARGIN_FRAC  = 0.05f;
static constexpr int          K_FEATHER_HALF_PX     = 40;      // Cosine feather half-width (px)
static constexpr float        K_SEAM_GRAD_WEIGHT    = 2.0f;    // Sobel-gradient weight in seam cost (thesis App. A)
static constexpr unsigned int K_HOLDOUT_SEED        = 0xC0FFEEu;

// -----------------------------------------------------------------------------
// Canvas geometry helpers
// -----------------------------------------------------------------------------

// Canvas geometry from each camera's image-plane border projected to ego az/el.
// The full border is sampled (not just corners) since the cylindrical warp
// curves edges, so the az/el extrema can sit mid-edge. Preserves full FOV.
static CanvasGeometry computeCanvasFromCameras(
    const std::array<Camera, 3>& cams,
    const LidarTpsConfig& cfg)
{
    float az_min = 1e9f, az_max = -1e9f;
    float el_min = 1e9f, el_max = -1e9f;

    const Camera& fc = cams[1];
    float fx_canvas = fc.K.fx;
    float fy_canvas = fc.K.fy;

    constexpr int N = 64;  // samples per edge
    auto sample = [&](const Camera& cam, float u, float v) {
        float xc = (u - cam.K.cx) / cam.K.fx;
        float yc = (v - cam.K.cy) / cam.K.fy;
        const auto& R = cam.E.R;
        // cam -> ego direction: R * [xc, yc, 1]. Far-field limit, so t drops out.
        float xe = R[0]*xc + R[1]*yc + R[2];
        float ye = R[3]*xc + R[4]*yc + R[5];
        float ze = R[6]*xc + R[7]*yc + R[8];
        float az = std::atan2(ye, xe);
        float el = std::atan2(ze, std::sqrt(xe*xe + ye*ye));
        az_min = std::min(az_min, az); az_max = std::max(az_max, az);
        el_min = std::min(el_min, el); el_max = std::max(el_max, el);
    };

    for (const auto& cam : cams) {
        float Wm = (float)cam.width  - 1.f;
        float Hm = (float)cam.height - 1.f;
        for (int i = 0; i <= N; i++) {
            float t = (float)i / (float)N;
            sample(cam, t * Wm, 0.f);    // top
            sample(cam, t * Wm, Hm);     // bottom
            sample(cam, 0.f,    t * Hm); // left
            sample(cam, Wm,     t * Hm); // right
        }
    }

    float az_mg = K_CANVAS_MARGIN_FRAC * (az_max - az_min);
    float el_mg = K_CANVAS_MARGIN_FRAC * (el_max - el_min);
    az_min -= az_mg; az_max += az_mg;
    el_min -= el_mg; el_max += el_mg;

    // True-cylindrical projection: v = fy * tan(el). World-verticals stay straight.
    float v_top    = std::tan(el_max);
    float v_bottom = std::tan(el_min);

    CanvasGeometry cg;
    cg.fx_canvas  = fx_canvas;
    cg.fy_canvas  = fy_canvas;
    cg.az_min     = az_min;  cg.az_max = az_max;
    cg.el_min     = el_min;  cg.el_max = el_max;
    cg.W          = std::max(1, (int)std::ceil(fx_canvas * (az_max - az_min)));
    cg.H          = std::max(1, (int)std::ceil(fy_canvas * (v_top - v_bottom)));
    cg.cx_canvas  = az_max * fx_canvas;   // az_max -> u=0 (left edge)
    cg.cy_canvas  = v_top  * fy_canvas;   // top -> v=0
    return cg;
}

// Evaluate rotation-baseline camera coords at a canvas pixel (CPU, for TPS delta).
static bool rotBaselineAtCanvas(
    float cu, float cv,
    const Camera& cam,
    const CanvasGeometry& cg,
    float& out_u, float& out_v)
{
    float az = (cg.cx_canvas - cu) / cg.fx_canvas;
    float v_scaled = (cg.cy_canvas - cv) / cg.fy_canvas;
    float el = std::atan(v_scaled);
    float cos_el = std::cos(el);
    float ex = cos_el * std::cos(az);
    float ey = cos_el * std::sin(az);
    float ez = std::sin(el);
    // R is cam->ego; R^T is ego->cam (transpose: swap row/col)
    const auto& R = cam.E.R;
    float cx_d = R[0]*ex + R[3]*ey + R[6]*ez;
    float cy_d = R[1]*ex + R[4]*ey + R[7]*ez;
    float cz_d = R[2]*ex + R[5]*ey + R[8]*ez;
    if (cz_d <= 0.0f) { out_u = -1; out_v = -1; return false; }
    out_u = cam.K.fx * cx_d / cz_d + cam.K.cx;
    out_v = cam.K.fy * cy_d / cz_d + cam.K.cy;
    return true;
}

// Camera pixel -> canvas via rotation-only model (CPU).
static void camPixToCanvas(
    float uc, float vc,
    const Camera& cam,
    const CanvasGeometry& cg,
    float& out_cu, float& out_cv)
{
    float xc = (uc - cam.K.cx) / cam.K.fx;
    float yc = (vc - cam.K.cy) / cam.K.fy;
    const auto& R = cam.E.R;
    float xe = R[0]*xc + R[1]*yc + R[2];
    float ye = R[3]*xc + R[4]*yc + R[5];
    float ze = R[6]*xc + R[7]*yc + R[8];
    float az = std::atan2(ye, xe);
    float r_xy = std::sqrt(xe*xe + ye*ye);
    float el = std::atan2(ze, r_xy);
    out_cu = -cg.fx_canvas * az + cg.cx_canvas;
    out_cv = -cg.fy_canvas * std::tan(el) + cg.cy_canvas;
}

// -----------------------------------------------------------------------------
// Impl
// -----------------------------------------------------------------------------

struct LidarTpsPipeline::Impl {
    LidarTpsConfig        cfg;
    CanvasGeometry        canvas;
    std::array<Camera, 3> cameras;
    float                 z_ref = 0.f;  // mean camera height (ego z)

    int W = 0, H = 0;          // canvas dimensions (full resolution)
    int W_half = 0, H_half = 0;// half-res dims for TPS displacement field

    // -- GPU buffers -- permanent -----------------------------------------------
    float*   d_pts         = nullptr;  // [N_MAX_PTS * 3]
    float*   d_cam_uv[3]   = {};       // [N_MAX_PTS * 2] per camera
    float*   d_depth[3]    = {};       // [N_MAX_PTS]
    uint8_t* d_pvalid[3]   = {};       // [N_MAX_PTS]

    float2*  d_remap[3]    = {};       // [W*H] per camera (rotation remap)
    uint8_t* d_vmask[3]    = {};       // [W*H] per camera (remap valid)

    uint8_t* d_src[3]      = {};       // [src_H*src_W*3] camera source images
    uint8_t* d_warped[3]   = {};       // [W*H*3] per camera
    uint8_t* d_canvas_buf  = nullptr;  // [W*H*3]
    float*        d_cost        = nullptr;  // [W*H] seam cost (FL<->FC)
    float*        d_dp          = nullptr;  // [W*H] DP forward accumulator (FL<->FC)
    int*          d_backtrack   = nullptr;  // [W*H] DP backtrack offsets (FL<->FC)
    float*        d_cost2       = nullptr;  // [W*H] seam cost (FC<->FR; parallel on stream3)
    float*        d_dp2         = nullptr;  // [W*H] DP forward accumulator (FC<->FR)
    int*          d_backtrack2  = nullptr;  // [W*H] DP backtrack offsets (FC<->FR)
    unsigned int* d_gain_accum  = nullptr;  // [7]   gain computation scratch (device)

    // CUDA Graph seam DP (one graph per overlap; baked-in buffer pointers force
    // two separate captures so the second pair can run on stream3 in parallel).
    cudaGraph_t     seam_graph        = nullptr;
    cudaGraphExec_t seam_graph_exec   = nullptr;
    bool            seam_graph_ready  = false;
    cudaGraph_t     seam_graph2       = nullptr;
    cudaGraphExec_t seam_graph2_exec  = nullptr;
    bool            seam_graph2_ready = false;

    // TPS buffers: ctrl_x/ctrl_y/wx/wy packed contiguously per cam (stride
    // TPS_PACK_STRIDE = N_MAX_CTRL+3), one cudaMemcpyAsync per cam from the
    // pinned h_tps_pack. The pointers below alias offsets into the pack.
    float* d_tps_pack[3] = {};          // device backing for ctrl_x/ctrl_y/wx/wy
    float* h_tps_pack[3] = {};          // pinned host staging, same layout
    float* d_ctrl_x[3]   = {};          // offsets into d_tps_pack[i]
    float* d_ctrl_y[3]   = {};
    float* d_wx[3]       = {};
    float* d_wy[3]       = {};
    float* d_disp_x[3]   = {};          // [W_half * H_half] TPS displacement field
    float* d_disp_y[3]   = {};

    // Seam arrays
    int* d_seam[2]     = {};           // [H] per seam

    // -- CPU pinned memory for D2H ---------------------------------------------
    float*   h_cam_uv[3]    = {};
    uint8_t* h_pvalid[3]    = {};
    float*   h_depth[3]     = {};
    uint8_t* h_vmask[3]     = {};      // [W*H] per camera
    uint8_t* h_canvas       = nullptr; // [W*H*3]


    // Camera max source dimensions
    int src_W = 0, src_H = 0;

    // Seam column estimates: midpoint x of each two-camera overlap band.
    int seam_col_lc = 0, seam_col_cr = 0;

    cudaStream_t stream  = nullptr;  // main stream: upload, lidar, seam, composite
    cudaStream_t stream2 = nullptr;  // warp stream for FC (camera 1)
    cudaStream_t stream3 = nullptr;  // warp stream for FR (camera 2)

    // Device-side warp-done events (replace CPU syncs before gain/seam).
    cudaEvent_t  warp_done_2 = nullptr;
    cudaEvent_t  warp_done_3 = nullptr;
    // FC<->FR seam DP runs on stream3 in parallel with FL<->FC on stream;
    // this event joins stream3's work back into stream before composite.
    cudaEvent_t  seam2_done  = nullptr;

    // Async canvas D2H state (cfg.async_d2h). Recorded after the canvas D2H
    // memcpy on `stream`; synchronised at the start of the next process()
    // call (or by waitD2H()).
    cudaEvent_t  d2h_done    = nullptr;
    bool         d2h_pending = false;

    // -- Temporal state (consumed by the temporal-coherence levers) ----------
    // `temporal_initialized` is false until the first frame populates the prev
    // buffers; the disp-field IIR consumer falls back to its no-history
    // branch while it is false.
    bool   temporal_initialized = false;
    int    frame_counter        = 0;    // increments per process() call
    float* d_disp_x_prev[3]     = {};   // [W_half * H_half] per camera
    float* d_disp_y_prev[3]     = {};
};

// -----------------------------------------------------------------------------
// Constructor / Destructor
// -----------------------------------------------------------------------------

LidarTpsPipeline::LidarTpsPipeline(const LidarTpsConfig& cfg)
    : d_(new Impl)
{
    d_->cfg = cfg;
    CUDA_CHECK(cudaStreamCreate(&d_->stream));
    CUDA_CHECK(cudaStreamCreate(&d_->stream2));
    CUDA_CHECK(cudaStreamCreate(&d_->stream3));
    CUDA_CHECK(cudaEventCreateWithFlags(&d_->warp_done_2, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&d_->warp_done_3, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&d_->seam2_done, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&d_->d2h_done, cudaEventDisableTiming));
}

LidarTpsPipeline::~LidarTpsPipeline() {
    if (!d_) return;
    cudaFree(d_->d_pts);
    for (int i = 0; i < 3; i++) {
        cudaFree(d_->d_cam_uv[i]);
        cudaFree(d_->d_depth[i]);
        cudaFree(d_->d_pvalid[i]);
        cudaFree(d_->d_remap[i]);
        cudaFree(d_->d_vmask[i]);
        cudaFree(d_->d_src[i]);
        cudaFree(d_->d_warped[i]);
        cudaFree(d_->d_tps_pack[i]);
        cudaFreeHost(d_->h_tps_pack[i]);
        cudaFree(d_->d_disp_x[i]);
        cudaFree(d_->d_disp_y[i]);
        cudaFree(d_->d_disp_x_prev[i]);
        cudaFree(d_->d_disp_y_prev[i]);
        cudaFreeHost(d_->h_cam_uv[i]);
        cudaFreeHost(d_->h_pvalid[i]);
        cudaFreeHost(d_->h_depth[i]);
        cudaFreeHost(d_->h_vmask[i]);
    }
    cudaFree(d_->d_canvas_buf);
    cudaFree(d_->d_cost);
    cudaFree(d_->d_dp);
    cudaFree(d_->d_backtrack);
    cudaFree(d_->d_cost2);
    cudaFree(d_->d_dp2);
    cudaFree(d_->d_backtrack2);
    cudaFree(d_->d_gain_accum);
    if (d_->seam_graph_exec)  cudaGraphExecDestroy(d_->seam_graph_exec);
    if (d_->seam_graph)       cudaGraphDestroy(d_->seam_graph);
    if (d_->seam_graph2_exec) cudaGraphExecDestroy(d_->seam_graph2_exec);
    if (d_->seam_graph2)      cudaGraphDestroy(d_->seam_graph2);
    for (int i = 0; i < 2; i++) {
        cudaFree(d_->d_seam[i]);
    }
    cudaFreeHost(d_->h_canvas);
    if (d_->warp_done_2)    cudaEventDestroy(d_->warp_done_2);
    if (d_->warp_done_3)    cudaEventDestroy(d_->warp_done_3);
    if (d_->seam2_done)     cudaEventDestroy(d_->seam2_done);
    if (d_->d2h_done)       cudaEventDestroy(d_->d2h_done);
    if (d_->stream)  cudaStreamDestroy(d_->stream);
    if (d_->stream2) cudaStreamDestroy(d_->stream2);
    if (d_->stream3) cudaStreamDestroy(d_->stream3);
    delete d_;
}

// -----------------------------------------------------------------------------
// init
// -----------------------------------------------------------------------------

void LidarTpsPipeline::init(const std::array<Camera, 3>& cameras,
                           const std::vector<std::array<float,3>>& lidar_pts) {
    auto& d = *d_;
    d.cameras = cameras;

    // Mean camera height (ego z) -- elevation reference for cylindrical projection
    d.z_ref = 0.f;
    for (const auto& c : cameras) d.z_ref += c.E.t[2];
    d.z_ref /= 3.f;

    // Canvas geometry from camera image-plane borders (preserves full FOV).
    d.canvas = computeCanvasFromCameras(cameras, d.cfg);
    d.W = d.canvas.W;
    d.H = d.canvas.H;

    // Max source image size across all cameras
    for (const auto& c : cameras) {
        d.src_W = std::max(d.src_W, c.width);
        d.src_H = std::max(d.src_H, c.height);
    }

    // Half-res dims for TPS displacement field
    d.W_half = std::max(4, (int)std::round(d.W * d.cfg.remap_scale));
    d.H_half = std::max(4, (int)std::round(d.H * d.cfg.remap_scale));

    std::cout << "Canvas: " << d.W << "x" << d.H
              << "  fx=" << d.canvas.fx_canvas
              << "  fy=" << d.canvas.fy_canvas
              << "  proj=cylindrical"
              << "  half=" << d.W_half << "x" << d.H_half << "\n";

    size_t WH  = (size_t)d.W * d.H;
    size_t WH3 = WH * 3;
    size_t WH4 = WH * 4;  // d_warped uses 4-byte/pixel stride (R, G, B, padding=0)

    // -- Allocate permanent GPU buffers ----------------------------------------
    CUDA_CHECK(cudaMalloc(&d.d_pts, (size_t)N_MAX_PTS * 3 * sizeof(float)));
    for (int i = 0; i < 3; i++) {
        CUDA_CHECK(cudaMalloc(&d.d_cam_uv[i],    (size_t)N_MAX_PTS * 2 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d.d_depth[i],     (size_t)N_MAX_PTS * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d.d_pvalid[i],    (size_t)N_MAX_PTS));

        CUDA_CHECK(cudaMalloc(&d.d_remap[i],  WH * sizeof(float2)));
        CUDA_CHECK(cudaMalloc(&d.d_vmask[i],  WH));

        CUDA_CHECK(cudaMalloc(&d.d_src[i],    (size_t)d.src_H * d.src_W * 3));
        CUDA_CHECK(cudaMalloc(&d.d_warped[i], WH4));
        // 4-byte/pixel stride aligns each pixel to a 32-bit boundary; producers
        // write all 3 BGR bytes, so the alpha lane just needs to be zeroed once.
        CUDA_CHECK(cudaMemset(d.d_warped[i], 0, WH4));

        // One packed alloc backs ctrl_x, ctrl_y, wx, wy -- a single H2D per
        // frame replaces 4 separate cudaMemcpyAsync launches.
        constexpr int TPS_PACK_STRIDE = N_MAX_CTRL + 3;   // max per-array size
        constexpr int TPS_PACK_FLOATS = 4 * TPS_PACK_STRIDE;
        CUDA_CHECK(cudaMalloc    (&d.d_tps_pack[i], TPS_PACK_FLOATS * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&d.h_tps_pack[i], TPS_PACK_FLOATS * sizeof(float)));
        d.d_ctrl_x[i] = d.d_tps_pack[i] + 0 * TPS_PACK_STRIDE;
        d.d_ctrl_y[i] = d.d_tps_pack[i] + 1 * TPS_PACK_STRIDE;
        d.d_wx[i]     = d.d_tps_pack[i] + 2 * TPS_PACK_STRIDE;
        d.d_wy[i]     = d.d_tps_pack[i] + 3 * TPS_PACK_STRIDE;

        CUDA_CHECK(cudaMalloc(&d.d_disp_x[i],    (size_t)d.W_half * d.H_half * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d.d_disp_y[i],    (size_t)d.W_half * d.H_half * sizeof(float)));

        // Temporal-state mirrors (read by the IIR/banded levers; populated at
        // the end of process() once temporal_initialized flips true).
        CUDA_CHECK(cudaMalloc(&d.d_disp_x_prev[i], (size_t)d.W_half * d.H_half * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d.d_disp_y_prev[i], (size_t)d.W_half * d.H_half * sizeof(float)));
    }
    CUDA_CHECK(cudaMalloc(&d.d_canvas_buf,  WH3));
    CUDA_CHECK(cudaMalloc(&d.d_cost,        WH * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d.d_dp,          WH * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d.d_backtrack,   WH * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d.d_cost2,       WH * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d.d_dp2,         WH * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d.d_backtrack2,  WH * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d.d_gain_accum,  7  * sizeof(unsigned int)));
    for (int i = 0; i < 2; i++) {
        CUDA_CHECK(cudaMalloc(&d.d_seam[i], (size_t)d.H * sizeof(int)));
    }

    // -- Allocate CPU pinned buffers -------------------------------------------
    for (int i = 0; i < 3; i++) {
        CUDA_CHECK(cudaMallocHost(&d.h_cam_uv[i],    (size_t)N_MAX_PTS * 2 * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&d.h_pvalid[i],    (size_t)N_MAX_PTS));
        CUDA_CHECK(cudaMallocHost(&d.h_depth[i],     (size_t)N_MAX_PTS * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&d.h_vmask[i],     WH));
    }
    CUDA_CHECK(cudaMallocHost(&d.h_canvas, WH3));

    // -- Precompute rotation remaps + valid masks (GPU) ------------------------
    const auto& cg = d.canvas;
    for (int i = 0; i < 3; i++) {
        const Camera& c = cameras[i];
        Mat3f R;
        for (int r = 0; r < 3; r++)
            for (int cc = 0; cc < 3; cc++)
                R.m[r*3+cc] = c.E.R[r*3+cc];

        buildRotRemap(
            d.W, d.H,
            R, c.K.fx, c.K.fy, c.K.cx, c.K.cy,
            c.width, c.height,
            cg.fx_canvas, cg.fy_canvas, cg.cx_canvas, cg.cy_canvas,
            d.d_remap[i], d.d_vmask[i], d.stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(d.stream));

    // Download valid masks -> compute overlap zones -> build overlap masks (CPU)
    for (int i = 0; i < 3; i++)
        CUDA_CHECK(cudaMemcpy(d.h_vmask[i], d.d_vmask[i], WH,
                              cudaMemcpyDeviceToHost));

    // Find seam columns: mean column where both neighbours overlap
    {
        // FL<->FC overlap (cameras 0 and 1)
        std::vector<int> ov_cols_lc;
        for (int col = 0; col < d.W; col++) {
            bool any = false;
            for (int row = 0; row < d.H && !any; row++)
                if (d.h_vmask[0][row*d.W+col] && d.h_vmask[1][row*d.W+col]) any = true;
            if (any) ov_cols_lc.push_back(col);
        }
        d.seam_col_lc = ov_cols_lc.empty()
            ? d.W / 3
            : (int)std::round((double)(ov_cols_lc.front() + ov_cols_lc.back()) / 2.0);

        // FC<->FR overlap (cameras 1 and 2)
        std::vector<int> ov_cols_cr;
        for (int col = 0; col < d.W; col++) {
            bool any = false;
            for (int row = 0; row < d.H && !any; row++)
                if (d.h_vmask[1][row*d.W+col] && d.h_vmask[2][row*d.W+col]) any = true;
            if (any) ov_cols_cr.push_back(col);
        }
        d.seam_col_cr = ov_cols_cr.empty()
            ? 2 * d.W / 3
            : (int)std::round((double)(ov_cols_cr.front() + ov_cols_cr.back()) / 2.0);

        std::cout << "Seam columns: FL<->FC=" << d.seam_col_lc
                  << "  FC<->FR=" << d.seam_col_cr << "\n";
    }

    std::cout << "LidarTpsPipeline init complete\n";
}

// -----------------------------------------------------------------------------
// Per-frame helpers
// -----------------------------------------------------------------------------

// Solve global TPS system (N+3)x(N+3) on CPU, two RHS vectors simultaneously.
// ctrl_x_n / ctrl_y_n must be in [0,1] (normalised by canvas dims).
// On success, w_x / w_y have layout: [w_0..w_{N-1}, a0, a_x, a_y].
// Returns false if the system is (nearly) singular.
static bool solveTPS(
    const std::vector<float>& ctrl_x_n,
    const std::vector<float>& ctrl_y_n,
    const std::vector<float>& disp_x,
    const std::vector<float>& disp_y,
    float smoothing,
    std::vector<float>& w_x,
    std::vector<float>& w_y)
{
    int N = (int)ctrl_x_n.size();
    int M = N + 3;

    auto tpsU = [](float r2) -> float {
        return (r2 > 1e-12f) ? 0.5f * r2 * std::log(r2) : 0.f;
    };

    std::vector<float> A(M * M, 0.f);
    w_x.assign(M, 0.f);
    w_y.assign(M, 0.f);

    for (int i = 0; i < N; i++) {
        A[i*M+i] = smoothing;
        for (int j = i+1; j < N; j++) {
            float dx = ctrl_x_n[i] - ctrl_x_n[j];
            float dy = ctrl_y_n[i] - ctrl_y_n[j];
            float v  = tpsU(dx*dx + dy*dy);
            A[i*M+j] = v;  A[j*M+i] = v;
        }
        A[i*M+N]   = 1.f;  A[N*M+i]   = 1.f;
        A[i*M+N+1] = ctrl_x_n[i];  A[(N+1)*M+i] = ctrl_x_n[i];
        A[i*M+N+2] = ctrl_y_n[i];  A[(N+2)*M+i] = ctrl_y_n[i];
        w_x[i] = disp_x[i];
        w_y[i] = disp_y[i];
    }

    // Gaussian elimination with partial pivoting
    for (int col = 0; col < M; col++) {
        int   pivot    = col;
        float best_abs = std::abs(A[col*M+col]);
        for (int row = col+1; row < M; row++) {
            float a = std::abs(A[row*M+col]);
            if (a > best_abs) { best_abs = a; pivot = row; }
        }
        if (best_abs < 1e-12f) return false;
        if (pivot != col) {
            for (int j = 0; j < M; j++) std::swap(A[col*M+j], A[pivot*M+j]);
            std::swap(w_x[col], w_x[pivot]);
            std::swap(w_y[col], w_y[pivot]);
        }
        float inv = 1.f / A[col*M+col];
        for (int row = col+1; row < M; row++) {
            float f = A[row*M+col] * inv;
            for (int j = col; j < M; j++) A[row*M+j] -= f * A[col*M+j];
            w_x[row] -= f * w_x[col];
            w_y[row] -= f * w_y[col];
        }
    }
    for (int i = M-1; i >= 0; i--) {
        w_x[i] /= A[i*M+i];
        w_y[i] /= A[i*M+i];
        for (int j = i-1; j >= 0; j--) {
            w_x[j] -= A[j*M+i] * w_x[i];
            w_y[j] -= A[j*M+i] * w_y[i];
        }
    }
    return true;
}

// Evaluate a solved TPS at a single canvas point. Inputs are in the same
// normalised [0,1]² convention as solveTPS(); output disp_x/disp_y are in
// camera-pixel units (same units as the RHS used in the solve).
static void evalTpsAtCanvas(
    float cu_n, float cv_n,
    const std::vector<float>& ctrl_x_n,
    const std::vector<float>& ctrl_y_n,
    const std::vector<float>& w_x,
    const std::vector<float>& w_y,
    float& disp_x, float& disp_y)
{
    int N = (int)ctrl_x_n.size();
    if (N == 0 || (int)w_x.size() < N + 3 || (int)w_y.size() < N + 3) {
        disp_x = 0.f; disp_y = 0.f; return;
    }
    auto tpsU = [](float r2) -> float {
        return (r2 > 1e-12f) ? 0.5f * r2 * std::log(r2) : 0.f;
    };
    double dx = w_x[N] + w_x[N+1] * cu_n + w_x[N+2] * cv_n;
    double dy = w_y[N] + w_y[N+1] * cu_n + w_y[N+2] * cv_n;
    for (int i = 0; i < N; i++) {
        float ex = cu_n - ctrl_x_n[i];
        float ey = cv_n - ctrl_y_n[i];
        float u  = tpsU(ex*ex + ey*ey);
        dx += (double)w_x[i] * u;
        dy += (double)w_y[i] * u;
    }
    disp_x = (float)dx;
    disp_y = (float)dy;
}

// Fine occupancy grid subsampling: each cell holds at most 1 point -- the one
// nearest to the cell centre. Grid dimensions are aspect-ratio-aware so cells
// are approximately square and total cells ~ max_pts.
// Returns the selected indices into the caller's shared-point vectors.
static std::vector<int> spatialGridIndices(
    const std::vector<float>& cvs_u,
    const std::vector<float>& cvs_v,
    int max_pts)
{
    int N = (int)cvs_u.size();
    if (N == 0) return {};
    if (N <= max_pts) {
        std::vector<int> idx(N);
        std::iota(idx.begin(), idx.end(), 0);
        return idx;
    }

    float u_min = *std::min_element(cvs_u.begin(), cvs_u.end());
    float u_max = *std::max_element(cvs_u.begin(), cvs_u.end());
    float v_min = *std::min_element(cvs_v.begin(), cvs_v.end());
    float v_max = *std::max_element(cvs_v.begin(), cvs_v.end());
    float u_range = std::max(1.f, u_max - u_min);
    float v_range = std::max(1.f, v_max - v_min);

    float aspect  = u_range / v_range;
    int fine_cols = std::max(1, (int)std::ceil(std::sqrt((double)max_pts * aspect)));
    int fine_rows = std::max(1, (int)std::ceil((double)max_pts / fine_cols));
    int n_cells   = fine_rows * fine_cols;

    std::vector<int>   occ(n_cells, -1);
    std::vector<float> occ_d2(n_cells, 1e30f);

    for (int i = 0; i < N; i++) {
        int fc = std::min(fine_cols - 1, (int)((cvs_u[i] - u_min) / u_range * fine_cols));
        int fr = std::min(fine_rows - 1, (int)((cvs_v[i] - v_min) / v_range * fine_rows));
        int ci = fr * fine_cols + fc;
        float cu   = u_min + (fc + 0.5f) / fine_cols * u_range;
        float cv_c = v_min + (fr + 0.5f) / fine_rows * v_range;
        float du   = cvs_u[i] - cu, dv = cvs_v[i] - cv_c;
        float d2   = du * du + dv * dv;
        if (d2 < occ_d2[ci]) { occ[ci] = i; occ_d2[ci] = d2; }
    }

    std::vector<int> sel;
    sel.reserve(max_pts);
    for (int ci = 0; ci < n_cells; ci++)
        if (occ[ci] != -1) sel.push_back(occ[ci]);
    return sel;
}

// -----------------------------------------------------------------------------
// process
// -----------------------------------------------------------------------------

cv::Mat LidarTpsPipeline::process(
    const std::array<cv::Mat, 3>& images,
    const std::vector<std::array<float,3>>& lidar_pts,
    bool profile,
    const std::string& debug_dir,
    ProcessMetrics* metrics,
    const std::string& eval_dump_dir)
{
    auto& d = *d_;
    const auto& cfg  = d.cfg;
    const auto& cg   = d.canvas;
    int N_pts = std::min((int)lidar_pts.size(), N_MAX_PTS);

    // -- Hold-out bookkeeping (only when cfg.holdout_frac > 0) ----------------
    // For each shared-pt in each overlap, we may reserve a fraction from the
    // TPS solve to use as ground truth for geometric evaluation.
    struct HoldoutPt {
        float xyz[3];                    // ego-frame 3D coord
        float cam_u[3], cam_v[3];        // projection in FL, FC, FR (−1 if invalid)
        float cvs_u[3], cvs_v[3];        // rotation-baseline canvas coord (−1 if invalid)
        float cvs_u_tps[3], cvs_v_tps[3];// post-TPS canvas coord (−1 if invalid)
        int   overlap = -1;              // 0 = FL<->FC, 1 = FC<->FR
    };
    std::vector<HoldoutPt> holdout_pts;
    const bool do_holdout = (cfg.holdout_frac > 0.f && !eval_dump_dir.empty());

    using hrc = std::chrono::high_resolution_clock;
    using ms  = std::chrono::duration<double, std::milli>;
    auto T = [&]() { CUDA_CHECK(cudaStreamSynchronize(d.stream)); return hrc::now(); };
    auto Tall = [&]() {
        CUDA_CHECK(cudaStreamSynchronize(d.stream));
        CUDA_CHECK(cudaStreamSynchronize(d.stream2));
        CUDA_CHECK(cudaStreamSynchronize(d.stream3));
        return hrc::now();
    };
    auto t0 = hrc::now();

    // -- Wait on the previous frame's deferred canvas D2H (cfg.async_d2h) ------
    // We're about to overwrite h_canvas, which the caller's last cv::Mat still
    // points at, so the prior D2H must finish first. Usually already done.
    if (cfg.async_d2h && d.d2h_pending) {
        CUDA_CHECK(cudaEventSynchronize(d.d2h_done));
        d.d2h_pending = false;
    }

    // -- Stage 1: Upload LiDAR + GPU projection --------------------------------
    CUDA_CHECK(cudaMemcpyAsync(d.d_pts,
        reinterpret_cast<const float*>(lidar_pts.data()),
        (size_t)N_pts * 3 * sizeof(float), cudaMemcpyHostToDevice, d.stream));

    for (int i = 0; i < 3; i++) {
        const Camera& c = d.cameras[i];
        Mat3f R;
        for (int r = 0; r < 3; r++)
            for (int cc = 0; cc < 3; cc++)
                R.m[r*3+cc] = c.E.R[r*3+cc];
        float3 t = make_float3(c.E.t[0], c.E.t[1], c.E.t[2]);

        projectLidarEgoGpu(
            d.d_pts, N_pts, R, t,
            c.K.fx, c.K.fy, c.K.cx, c.K.cy,
            c.width, c.height,
            K_LIDAR_MAX_RANGE_M,
            d.d_cam_uv[i], d.d_depth[i], d.d_pvalid[i],
            d.stream);
    }

    // Upload source images (cv::Mat memory is pageable, so cudaMemcpyAsync
    // here is effectively synchronous w.r.t. the host -- that's fine because
    // the host work that follows is the t1 sync below anyway).
    for (int i = 0; i < 3; i++) {
        const cv::Mat& img = images[i];
        CUDA_CHECK(cudaMemcpyAsync(d.d_src[i], img.data,
            (size_t)img.rows * img.cols * 3, cudaMemcpyHostToDevice, d.stream));
    }

    // D2H: projection results (needed for CPU stages 2-5)
    for (int i = 0; i < 3; i++) {
        CUDA_CHECK(cudaMemcpyAsync(d.h_cam_uv[i],    d.d_cam_uv[i],
            (size_t)N_pts * 2 * sizeof(float), cudaMemcpyDeviceToHost, d.stream));
        CUDA_CHECK(cudaMemcpyAsync(d.h_pvalid[i],    d.d_pvalid[i],
            (size_t)N_pts,                    cudaMemcpyDeviceToHost, d.stream));
        CUDA_CHECK(cudaMemcpyAsync(d.h_depth[i],     d.d_depth[i],
            (size_t)N_pts * sizeof(float),   cudaMemcpyDeviceToHost, d.stream));
    }
    auto t1 = T();  // after upload + lidar projection + D2H

    // -- Stages 2–5: CPU shared-pt extraction, ctrl pt storage ----------------
    // Per-camera TPS ctrl pt buffers (canvas pixel coords + camera displacements).
    struct CamTPS {
        std::vector<float> ctrl_x, ctrl_y;  // canvas pixel coords (NOT normalised)
        std::vector<float> disp_x, disp_y;  // camera-pixel displacements at ctrl pts
        std::vector<float> cam_u,  cam_v;   // source image pixel coords of ctrl pts
        // Cached after solveTPS() for offline metrics (bending energy):
        std::vector<float> ctrl_x_n, ctrl_y_n;  // normalised [0,1] ctrl coords
        std::vector<float> w_x, w_y;            // TPS weights, layout [w_0..w_{N-1}, a0, ax, ay]
        int N = 0;
    };
    std::array<CamTPS, 3> cam_tps;

    // Store ctrl pts for one camera: canvas-px coords + rotation-baseline deltas.
    auto storeCtrlPts = [&](
        int cam_idx,
        const std::vector<float>& cam_u_pts,
        const std::vector<float>& cam_v_pts,
        const std::vector<float>& cvs_u_pts,
        const std::vector<float>& cvs_v_pts)
    {
        int M = (int)cam_u_pts.size();
        if (M < 3) return;
        int NC = std::min(M, N_MAX_CTRL);

        const Camera& cam = d.cameras[cam_idx];
        auto& ct = cam_tps[cam_idx];
        ct.ctrl_x.resize(NC); ct.ctrl_y.resize(NC);
        ct.disp_x.resize(NC); ct.disp_y.resize(NC);
        ct.cam_u.resize(NC);  ct.cam_v.resize(NC);

        for (int i = 0; i < NC; i++) {
            ct.ctrl_x[i] = cvs_u_pts[i];   // canvas pixel (not normalised)
            ct.ctrl_y[i] = cvs_v_pts[i];
            ct.cam_u[i]  = cam_u_pts[i];   // source image pixel coord
            ct.cam_v[i]  = cam_v_pts[i];
            float rot_u, rot_v;
            bool ok = rotBaselineAtCanvas(cvs_u_pts[i], cvs_v_pts[i], cam, cg, rot_u, rot_v);
            ct.disp_x[i] = ok ? cam_u_pts[i] - rot_u : 0.f;
            ct.disp_y[i] = ok ? cam_v_pts[i] - rot_v : 0.f;
        }
        ct.N = NC;
    };

    // Find shared points between adjacent cameras.
    // h_pvalid[i][j] is already a bool array over all N_pts -- no hash map needed.
    // Overlap 0: FL(0)<->FC(1)
    // Overlap 1: FC(1)<->FR(2)
    int n_shared_per_overlap[2] = {0, 0};
    for (int ov = 0; ov < 2; ov++) {
        int cam_l = ov;           // 0 or 1
        int cam_r = ov + 1;       // 1 or 2

        // Linear scan -- O(N_pts), avoids all hash map allocations
        std::vector<int> shared_all;
        shared_all.reserve(4096);
        for (int j = 0; j < N_pts; j++) {
            if (!d.h_pvalid[cam_l][j] || !d.h_pvalid[cam_r][j]) continue;
            if (d.h_depth[cam_l][j] >= cfg.lidar_min_ctrl_range_m)
                shared_all.push_back(j);
        }
        n_shared_per_overlap[ov] = (int)shared_all.size();

        // Deterministic train/holdout split. Seed combines K_HOLDOUT_SEED with
        // overlap index and shared-count so different frames / configs get
        // reproducible but distinct splits.
        std::vector<int> shared;
        shared.reserve(shared_all.size());
        if (do_holdout) {
            std::vector<int> idx(shared_all.size());
            std::iota(idx.begin(), idx.end(), 0);
            std::mt19937 rng(K_HOLDOUT_SEED
                ^ (unsigned)(ov * 2654435761u)
                ^ (unsigned)(shared_all.size() * 83492791u));
            std::shuffle(idx.begin(), idx.end(), rng);
            int n_hold = (int)std::round(shared_all.size() * cfg.holdout_frac);
            // Record held-out points
            for (int k = 0; k < n_hold; k++) {
                int j = shared_all[idx[k]];
                HoldoutPt hp;
                const auto& p = lidar_pts[j];
                hp.xyz[0] = p[0]; hp.xyz[1] = p[1]; hp.xyz[2] = p[2];
                hp.overlap = ov;
                for (int ci = 0; ci < 3; ci++) {
                    if (d.h_pvalid[ci][j]) {
                        hp.cam_u[ci] = d.h_cam_uv[ci][j*2];
                        hp.cam_v[ci] = d.h_cam_uv[ci][j*2+1];
                        float cu, cv2;
                        camPixToCanvas(hp.cam_u[ci], hp.cam_v[ci],
                                       d.cameras[ci], cg, cu, cv2);
                        hp.cvs_u[ci] = cu; hp.cvs_v[ci] = cv2;
                    } else {
                        hp.cam_u[ci] = hp.cam_v[ci] = -1.f;
                        hp.cvs_u[ci] = hp.cvs_v[ci] = -1.f;
                    }
                    hp.cvs_u_tps[ci] = -1.f;
                    hp.cvs_v_tps[ci] = -1.f;
                }
                holdout_pts.push_back(hp);
            }
            // Remaining indices go into training set
            for (int k = n_hold; k < (int)idx.size(); k++)
                shared.push_back(shared_all[idx[k]]);
        } else {
            shared = std::move(shared_all);
        }

        if (profile)
            printf("  overlap %d: %d raw shared pts (train=%d holdout=%d)\n",
                   ov, n_shared_per_overlap[ov],
                   (int)shared.size(),
                   n_shared_per_overlap[ov] - (int)shared.size());
        if (shared.empty()) continue;

        // Canvas target = FC's rotation projection of the shared LiDAR point.
        // FC stays at its rotation baseline; the side camera carries the warp.
        std::vector<float> cam_u_l, cam_v_l;  // camera pixels for cam_l
        std::vector<float> cam_u_r, cam_v_r;  // camera pixels for cam_r
        std::vector<float> cvs_u, cvs_v;       // shared canvas targets
        std::vector<float> range_m_vec;         // ego-frame range per shared pt

        for (int idx : shared) {
            float ul = d.h_cam_uv[cam_l][idx*2], vl = d.h_cam_uv[cam_l][idx*2+1];
            float ur = d.h_cam_uv[cam_r][idx*2], vr = d.h_cam_uv[cam_r][idx*2+1];

            // For overlap 0 (FL<->FC): FC pixels are cam_r, ref_side='right'.
            // For overlap 1 (FC<->FR): FC pixels are cam_l, ref_side='left'.
            float ref_u = (ov == 0) ? ur : ul;
            float ref_v = (ov == 0) ? vr : vl;
            float cu_tgt, cv_tgt;
            camPixToCanvas(ref_u, ref_v, d.cameras[1], cg, cu_tgt, cv_tgt);

            cam_u_l.push_back(ul); cam_v_l.push_back(vl);
            cam_u_r.push_back(ur); cam_v_r.push_back(vr);
            cvs_u.push_back(cu_tgt); cvs_v.push_back(cv_tgt);

            // Ego-frame range for stratified sampling
            const auto& p = lidar_pts[idx];
            range_m_vec.push_back(std::sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2]));
        }

        // Spatial-grid subsample over the overlap; both cameras get the same
        // indices (pairing preserved). Budget cfg.max_ctrl_per_overlap (default
        // 50, --max-ctrl-per-overlap; swept in sec.4).
        auto sel = spatialGridIndices(cvs_u, cvs_v, cfg.max_ctrl_per_overlap);

        std::vector<float> sub_cu_l, sub_cv_l, sub_cu_r, sub_cv_r, sub_uu, sub_uv;
        for (int i : sel) {
            sub_cu_l.push_back(cam_u_l[i]); sub_cv_l.push_back(cam_v_l[i]);
            sub_cu_r.push_back(cam_u_r[i]); sub_cv_r.push_back(cam_v_r[i]);
            sub_uu.push_back(cvs_u[i]);     sub_uv.push_back(cvs_v[i]);
        }

        // Store ctrl pts for side cameras only; FC stays at its rotation
        // baseline (no TPS warp on FC).
        if (cam_l != 1) storeCtrlPts(cam_l, sub_cu_l, sub_cv_l, sub_uu, sub_uv);
        if (cam_r != 1) storeCtrlPts(cam_r, sub_cu_r, sub_cv_r, sub_uu, sub_uv);
    }


    auto t2 = hrc::now();  // after CPU stages 2-5

    // -- Stage 6 / 7: Warp each camera (rotation remap + global TPS) ---------
    // FL on d.stream, FC on d.stream2, FR on d.stream3 -- GPU kernels overlap.
    if (profile) {
        static const char* cnames[3] = {"FL","FC","FR"};
        for (int i = 0; i < 3; i++)
            printf("  ctrl pts [%s]: %d\n", cnames[i], cam_tps[i].N);
        fflush(stdout);
    }
    static const char* cnames_warp[3] = {"FL", "FC", "FR"};
    cudaStream_t sc[3] = {d.stream, d.stream2, d.stream3};
    for (int i = 0; i < 3; i++) {
        const Camera& c = d.cameras[i];
        auto& ct = cam_tps[i];

        if (ct.N < 3 || ct.disp_x.empty()) {
            // Fallback: rotation-only warp (no LiDAR ctrl pts found)
            applyRemapU8(d.d_src[i], c.width, c.height,
                d.d_remap[i], d.W, d.H, d.d_warped[i], sc[i]);
            if (!debug_dir.empty()) {
                CUDA_CHECK(cudaStreamSynchronize(sc[i]));
                std::vector<uint8_t> h_rot((size_t)d.W * d.H * 4);
                CUDA_CHECK(cudaMemcpy(h_rot.data(), d.d_warped[i],
                    (size_t)d.W * d.H * 4, cudaMemcpyDeviceToHost));
                cv::Mat rmat4(d.H, d.W, CV_8UC4, h_rot.data());
                cv::Mat rmat;
                cv::cvtColor(rmat4, rmat, cv::COLOR_BGRA2BGR);
                cv::imwrite(debug_dir + "/02b_rotation_" + cnames_warp[i] + ".jpg",
                    rmat, {cv::IMWRITE_JPEG_QUALITY, 90});
            }
        } else {
            // Solve global TPS system once on CPU (N+3)x(N+3).
            // Ctrl coords normalised to [0,1] -- matches evalTpsDispKernel convention.
            int N = ct.N;
            float norm_x = (d.W > 1) ? (float)(d.W - 1) : 1.f;
            float norm_y = (d.H > 1) ? (float)(d.H - 1) : 1.f;
            std::vector<float> ctrl_x_n(N), ctrl_y_n(N);
            for (int j = 0; j < N; j++) {
                ctrl_x_n[j] = ct.ctrl_x[j] / norm_x;
                ctrl_y_n[j] = ct.ctrl_y[j] / norm_y;
            }
            std::vector<float> w_x, w_y;
            if (!solveTPS(ctrl_x_n, ctrl_y_n, ct.disp_x, ct.disp_y,
                          cfg.tps_smoothing, w_x, w_y)) {
                applyRemapU8(d.d_src[i], c.width, c.height,
                    d.d_remap[i], d.W, d.H, d.d_warped[i], sc[i]);
            } else {
                // Cache weights + normalised ctrl coords for offline metrics
                // (bending energy). Cheap: (N+3) floats x 2 arrays per camera.
                ct.ctrl_x_n = ctrl_x_n;
                ct.ctrl_y_n = ctrl_y_n;
                ct.w_x = w_x;
                ct.w_y = w_y;
                // Debug: save rotation-only warp before applying TPS correction
                if (!debug_dir.empty()) {
                    applyRemapU8(d.d_src[i], c.width, c.height,
                        d.d_remap[i], d.W, d.H, d.d_warped[i], sc[i]);
                    CUDA_CHECK(cudaStreamSynchronize(sc[i]));
                    std::vector<uint8_t> h_rot((size_t)d.W * d.H * 4);
                    CUDA_CHECK(cudaMemcpy(h_rot.data(), d.d_warped[i],
                        (size_t)d.W * d.H * 4, cudaMemcpyDeviceToHost));
                    cv::Mat rmat4(d.H, d.W, CV_8UC4, h_rot.data());
                    cv::Mat rmat;
                    cv::cvtColor(rmat4, rmat, cv::COLOR_BGRA2BGR);
                    cv::imwrite(debug_dir + "/02b_rotation_" + cnames_warp[i] + ".jpg",
                        rmat, {cv::IMWRITE_JPEG_QUALITY, 90});
                }

                // Upload normalised ctrl coords + TPS weights (N+3 each).
                // Pack the 4 arrays into one pinned host buffer, then a single
                // cudaMemcpyAsync replaces the 4 per-array transfers.
                constexpr int TPS_PACK_STRIDE = N_MAX_CTRL + 3;
                constexpr int TPS_PACK_FLOATS = 4 * TPS_PACK_STRIDE;
                float* H = d.h_tps_pack[i];
                std::memcpy(H + 0 * TPS_PACK_STRIDE, ctrl_x_n.data(), (size_t)N     * sizeof(float));
                std::memcpy(H + 1 * TPS_PACK_STRIDE, ctrl_y_n.data(), (size_t)N     * sizeof(float));
                std::memcpy(H + 2 * TPS_PACK_STRIDE, w_x.data(),      (size_t)(N+3) * sizeof(float));
                std::memcpy(H + 3 * TPS_PACK_STRIDE, w_y.data(),      (size_t)(N+3) * sizeof(float));
                CUDA_CHECK(cudaMemcpyAsync(d.d_tps_pack[i], H,
                    (size_t)TPS_PACK_FLOATS * sizeof(float),
                    cudaMemcpyHostToDevice, sc[i]));

                // Evaluate displacement field at remap_scale resolution,
                // then bilinear upsample + apply on top of rotation remap.
                evalTpsDispHalf(
                    d.W_half, d.H_half,
                    d.d_ctrl_x[i], d.d_ctrl_y[i],
                    d.d_wx[i], d.d_wy[i],
                    N,
                    d.d_disp_x[i], d.d_disp_y[i],
                    sc[i]);
                // Temporal IIR on the half-res disp field. First frame just
                // snapshots (no blend) so the next frame has a starting point.
                if (cfg.disp_temporal_alpha < 1.f) {
                    if (d.temporal_initialized) {
                        applyDispTemporalIIR(
                            d.d_disp_x[i], d.d_disp_y[i],
                            d.d_disp_x_prev[i], d.d_disp_y_prev[i],
                            d.W_half, d.H_half,
                            cfg.disp_temporal_alpha, sc[i]);
                    } else {
                        size_t bytes = (size_t)d.W_half * d.H_half * sizeof(float);
                        CUDA_CHECK(cudaMemcpyAsync(d.d_disp_x_prev[i], d.d_disp_x[i],
                            bytes, cudaMemcpyDeviceToDevice, sc[i]));
                        CUDA_CHECK(cudaMemcpyAsync(d.d_disp_y_prev[i], d.d_disp_y[i],
                            bytes, cudaMemcpyDeviceToDevice, sc[i]));
                    }
                }
                applyRemapWithDispU8(
                    d.d_src[i], c.width, c.height,
                    d.d_remap[i],
                    d.W, d.H, d.W_half, d.H_half,
                    d.d_disp_x[i], d.d_disp_y[i],
                    d.d_warped[i], d.d_vmask[i], sc[i]);
            }
        }
    }

    // -- Post-TPS canvas coord for held-out points ----------------------------
    // Invert the warp for each held-out pixel (u,v): find canvas CC with
    // rot(CC) + TPS_disp(CC) = (u,v), i.e. CC = camPixToCanvas((u,v) − disp(CC)).
    // Fixed-point iteration, 3-5 steps for smooth fields.
    if (do_holdout) {
        const float norm_x = (d.W > 1) ? (float)(d.W - 1) : 1.f;
        const float norm_y = (d.H > 1) ? (float)(d.H - 1) : 1.f;
        for (auto& hp : holdout_pts) {
            for (int ci = 0; ci < 3; ci++) {
                if (hp.cam_u[ci] < 0.f) continue;
                // <3 ctrl pts -> rotation remap only (post-TPS = baseline).
                // Includes FC under the deployed asymmetric warp (N == 0).
                if (cam_tps[ci].N < 3) {
                    hp.cvs_u_tps[ci] = hp.cvs_u[ci];
                    hp.cvs_v_tps[ci] = hp.cvs_v[ci];
                    continue;
                }
                const auto& ct = cam_tps[ci];
                float cu = hp.cvs_u[ci];
                float cv = hp.cvs_v[ci];
                for (int it = 0; it < 6; it++) {
                    float dx = 0.f, dy = 0.f;
                    evalTpsAtCanvas(cu / norm_x, cv / norm_y,
                                    ct.ctrl_x_n, ct.ctrl_y_n, ct.w_x, ct.w_y,
                                    dx, dy);
                    float new_cu, new_cv;
                    camPixToCanvas(hp.cam_u[ci] - dx, hp.cam_v[ci] - dy,
                                   d.cameras[ci], cg, new_cu, new_cv);
                    float du = new_cu - cu;
                    float dv = new_cv - cv;
                    cu = new_cu;
                    cv = new_cv;
                    if (du*du + dv*dv < 1e-4f) break;
                }
                hp.cvs_u_tps[ci] = cu;
                hp.cvs_v_tps[ci] = cv;
            }
        }
    }

    // Serialize warp -> gain/seam/composite. Device-side events avoid the
    // host-device round-trips that cudaStreamSynchronize would do. stream
    // already owns the downstream work (gain/seam/composite), so it only
    // needs to wait on stream2 (FC) and stream3 (FR). In metrics mode we
    // additionally CPU-sync so t_warp remains a wall-clock number.
    CUDA_CHECK(cudaEventRecord(d.warp_done_2, d.stream2));
    CUDA_CHECK(cudaEventRecord(d.warp_done_3, d.stream3));
    CUDA_CHECK(cudaStreamWaitEvent(d.stream, d.warp_done_2, 0));
    CUDA_CHECK(cudaStreamWaitEvent(d.stream, d.warp_done_3, 0));
    if (metrics != nullptr) {
        CUDA_CHECK(cudaStreamSynchronize(d.stream));
        CUDA_CHECK(cudaStreamSynchronize(d.stream2));
        CUDA_CHECK(cudaStreamSynchronize(d.stream3));
    }
    // Stage timestamps are stream-synced under profile/metrics so t_*_ms
    // measure real GPU durations, not host launch time.
    auto t3 = (profile || metrics != nullptr) ? Tall() : hrc::now();

    // Debug: snapshot FL and FR before in-place gain modifies them
    std::vector<uint8_t> h_pregain_fl, h_pregain_fr;
    if (!debug_dir.empty() && cfg.gain_compensation) {
        if (!(profile || metrics != nullptr))
            CUDA_CHECK(cudaStreamSynchronize(d.stream));
        const size_t WH4 = (size_t)d.W * d.H * 4;
        h_pregain_fl.resize(WH4);
        h_pregain_fr.resize(WH4);
        CUDA_CHECK(cudaMemcpy(h_pregain_fl.data(), d.d_warped[0], WH4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_pregain_fr.data(), d.d_warped[2], WH4, cudaMemcpyDeviceToHost));
    }

    // -- Gain compensation (device-side) --------------------------------------
    // Match FL/FR to FC's per-channel overlap mean (FC is the reference).
    // Sums stay on device; gain is resolved+clamped+applied in one fused kernel.
    if (cfg.gain_compensation) {
        int WH = d.W * d.H;
        constexpr int MIN_OVERLAP_COUNT = 500;

        // FL<->FC overlap: accum layout [FL.B,FL.G,FL.R, FC.B,FC.G,FC.R, count].
        // Source=FL (0), target=FC (3) -- FL is adjusted toward FC.
        computeOverlapMeans(d.d_warped[0], d.d_warped[1],
            d.d_vmask[0], d.d_vmask[1], WH,
            d.d_gain_accum, /*h_out=*/nullptr, d.stream);
        applyGainFromAccum(d.d_warped[0], WH,
            d.d_gain_accum, /*src=*/0, /*dst=*/3, MIN_OVERLAP_COUNT, d.stream);

        // FC<->FR overlap: accum layout [FC.B,FC.G,FC.R, FR.B,FR.G,FR.R, count].
        // Source=FR (3), target=FC (0) -- FR is adjusted toward FC.
        computeOverlapMeans(d.d_warped[1], d.d_warped[2],
            d.d_vmask[1], d.d_vmask[2], WH,
            d.d_gain_accum, /*h_out=*/nullptr, d.stream);
        applyGainFromAccum(d.d_warped[2], WH,
            d.d_gain_accum, /*src=*/3, /*dst=*/0, MIN_OVERLAP_COUNT, d.stream);
    }

    // -- Debug: source, warped, ctrl pts, displacement fields -----------------
    if (!debug_dir.empty()) {
        namespace fs = std::filesystem;
        fs::create_directories(debug_dir);
        static const char* cnames[3] = {"FL", "FC", "FR"};

        // 01a: source images
        for (int i = 0; i < 3; i++)
            cv::imwrite(debug_dir + "/01_source_" + cnames[i] + ".jpg",
                        images[i], {cv::IMWRITE_JPEG_QUALITY, 90});

        // 01b: All visible LiDAR points on source images (color = depth, blue=near red=far)
        for (int i = 0; i < 3; i++) {
            cv::Mat src_copy = images[i].clone();
            float dmin = 1e9f, dmax_d = 0.f;
            for (int j = 0; j < N_pts; j++) {
                if (!d.h_pvalid[i][j]) continue;
                dmin  = std::min(dmin,  d.h_depth[i][j]);
                dmax_d = std::max(dmax_d, d.h_depth[i][j]);
            }
            if (dmin >= dmax_d) dmax_d = dmin + 1.f;
            for (int j = 0; j < N_pts; j++) {
                if (!d.h_pvalid[i][j]) continue;
                float u = d.h_cam_uv[i][j*2], v = d.h_cam_uv[i][j*2+1];
                float t_d = (d.h_depth[i][j] - dmin) / (dmax_d - dmin);
                uint8_t lv = (uint8_t)(255.f * t_d);
                cv::Vec3b color(255 - lv, 0, lv);  // BGR: near=blue, far=red
                int iu = (int)std::round(u), iv = (int)std::round(v);
                if (iu >= 0 && iu < images[i].cols && iv >= 0 && iv < images[i].rows)
                    cv::circle(src_copy, {iu, iv}, 2, color, -1);
            }
            cv::imwrite(debug_dir + "/01b_lidar_all_" + cnames[i] + ".jpg",
                src_copy, {cv::IMWRITE_JPEG_QUALITY, 90});
        }

        // 02: warped canvas images. d_warped is W*H*4.
        std::vector<uint8_t> h_warp((size_t)d.W * d.H * 4);
        std::vector<uint8_t> h_postgain_fl, h_postgain_fr;  // for 07b gain compare
        for (int i = 0; i < 3; i++) {
            CUDA_CHECK(cudaMemcpy(h_warp.data(), d.d_warped[i],
                                  (size_t)d.W * d.H * 4, cudaMemcpyDeviceToHost));
            if (cfg.gain_compensation && (i == 0 || i == 2)) {
                if (i == 0) h_postgain_fl = h_warp;
                else        h_postgain_fr = h_warp;
            }
            cv::Mat wmat4(d.H, d.W, CV_8UC4, h_warp.data());
            cv::Mat wmat;
            cv::cvtColor(wmat4, wmat, cv::COLOR_BGRA2BGR);
            cv::imwrite(debug_dir + "/02_warped_" + cnames[i] + ".jpg",
                        wmat, {cv::IMWRITE_JPEG_QUALITY, 90});
        }

        // 03b: Correspondence strip — FL | FC | FR scaled to common height,
        //      lines connecting the same LiDAR point across adjacent cameras.
        {
            constexpr int STRIP_H  = 600;
            constexpr int MAX_CORR = 10;
            auto scale_to_h = [&](const cv::Mat& src) -> cv::Mat {
                cv::Mat out;
                cv::resize(src, out, cv::Size(
                    (int)std::round((float)src.cols * STRIP_H / src.rows), STRIP_H));
                return out;
            };
            cv::Mat sFL = scale_to_h(images[0]);
            cv::Mat sFC = scale_to_h(images[1]);
            cv::Mat sFR = scale_to_h(images[2]);
            float sc0 = (float)STRIP_H / images[0].rows;
            float sc1 = (float)STRIP_H / images[1].rows;
            float sc2 = (float)STRIP_H / images[2].rows;

            cv::Mat sep(STRIP_H, 6, CV_8UC3, cv::Scalar(40, 40, 40));
            cv::Mat strip;
            cv::hconcat(std::vector<cv::Mat>{sFL, sep, sFC, sep, sFR}, strip);

            int x_fc = sFL.cols + sep.cols;
            int x_fr = x_fc + sFC.cols + sep.cols;

            // Build JET LUT for magnitude colouring
            cv::Mat cmap_lut(1, 256, CV_8UC1);
            for (int k = 0; k < 256; k++) cmap_lut.data[k] = (uint8_t)k;
            cv::Mat cmap_rgb;
            cv::applyColorMap(cmap_lut, cmap_rgb, cv::COLORMAP_JET);
            const auto* cmap = cmap_rgb.ptr<cv::Vec3b>(0);

            // FL<->FC correspondences
            {
                int N = cam_tps[0].N;
                if (N > 0) {
                    float dmax = 1.f;
                    for (int j = 0; j < N; j++)
                        dmax = std::max(dmax, std::hypot(cam_tps[0].disp_x[j], cam_tps[0].disp_y[j]));
                    int stride = std::max(1, N / MAX_CORR);
                    for (int j = 0; j < N; j += stride) {
                        float fc_u, fc_v;
                        if (!rotBaselineAtCanvas(cam_tps[0].ctrl_x[j], cam_tps[0].ctrl_y[j],
                                                 d.cameras[1], cg, fc_u, fc_v)) continue;
                        cv::Point2f p0(cam_tps[0].cam_u[j] * sc0,
                                       cam_tps[0].cam_v[j] * sc0);
                        cv::Point2f p1(fc_u * sc1 + x_fc, fc_v * sc1);
                        int lut = std::max(0, std::min(255, (int)(255.f *
                            std::hypot(cam_tps[0].disp_x[j], cam_tps[0].disp_y[j]) / dmax)));
                        cv::Scalar col(cmap[lut][0], cmap[lut][1], cmap[lut][2]);
                        cv::line(strip, p0, p1, col, 1, cv::LINE_AA);
                        cv::circle(strip, p0, 4, col, -1, cv::LINE_AA);
                        cv::circle(strip, p1, 4, col, -1, cv::LINE_AA);
                    }
                }
            }

            // FC<->FR correspondences
            {
                int N = cam_tps[2].N;
                if (N > 0) {
                    float dmax = 1.f;
                    for (int j = 0; j < N; j++)
                        dmax = std::max(dmax, std::hypot(cam_tps[2].disp_x[j], cam_tps[2].disp_y[j]));
                    int stride = std::max(1, N / MAX_CORR);
                    for (int j = 0; j < N; j += stride) {
                        float fc_u, fc_v;
                        if (!rotBaselineAtCanvas(cam_tps[2].ctrl_x[j], cam_tps[2].ctrl_y[j],
                                                 d.cameras[1], cg, fc_u, fc_v)) continue;
                        cv::Point2f p1(fc_u * sc1 + x_fc, fc_v * sc1);
                        cv::Point2f p2(cam_tps[2].cam_u[j] * sc2 + x_fr,
                                       cam_tps[2].cam_v[j] * sc2);
                        int lut = std::max(0, std::min(255, (int)(255.f *
                            std::hypot(cam_tps[2].disp_x[j], cam_tps[2].disp_y[j]) / dmax)));
                        cv::Scalar col(cmap[lut][0], cmap[lut][1], cmap[lut][2]);
                        cv::line(strip, p1, p2, col, 1, cv::LINE_AA);
                        cv::circle(strip, p1, 4, col, -1, cv::LINE_AA);
                        cv::circle(strip, p2, 4, col, -1, cv::LINE_AA);
                    }
                }
            }

            cv::imwrite(debug_dir + "/03b_ctrl_arrows.jpg",
                        strip, {cv::IMWRITE_JPEG_QUALITY, 92});
        }

        // 07: Displacement magnitude heatmap (INFERNO, half-res, masked to camera FOV)
        {
            std::vector<float> h_dx2((size_t)d.W_half * d.H_half);
            std::vector<float> h_dy2((size_t)d.W_half * d.H_half);
            for (int i = 0; i < 3; i++) {
                CUDA_CHECK(cudaMemcpy(h_dx2.data(), d.d_disp_x[i],
                    (size_t)d.W_half * d.H_half * sizeof(float), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_dy2.data(), d.d_disp_y[i],
                    (size_t)d.W_half * d.H_half * sizeof(float), cudaMemcpyDeviceToHost));
                // Build half-res FOV mask by downsampling h_vmask (nearest neighbour)
                float sx = (float)d.W / d.W_half;
                float sy = (float)d.H / d.H_half;
                float mag_max = 1.f;
                for (int k = 0; k < d.W_half * d.H_half; k++) {
                    int hrow = k / d.W_half, hcol = k % d.W_half;
                    int frow = std::min(d.H-1, (int)(hrow * sy));
                    int fcol = std::min(d.W-1, (int)(hcol * sx));
                    if (!d.h_vmask[i][frow * d.W + fcol]) continue;
                    mag_max = std::max(mag_max, std::hypot(h_dx2[k], h_dy2[k]));
                }
                cv::Mat mag_u8(d.H_half, d.W_half, CV_8UC1, cv::Scalar(0));
                for (int k = 0; k < d.W_half * d.H_half; k++) {
                    int hrow = k / d.W_half, hcol = k % d.W_half;
                    int frow = std::min(d.H-1, (int)(hrow * sy));
                    int fcol = std::min(d.W-1, (int)(hcol * sx));
                    if (!d.h_vmask[i][frow * d.W + fcol]) continue;
                    mag_u8.data[k] = (uint8_t)(255.f * std::hypot(h_dx2[k], h_dy2[k]) / mag_max);
                }
                cv::Mat mag_color;
                cv::applyColorMap(mag_u8, mag_color, cv::COLORMAP_INFERNO);
                // Re-black pixels outside FOV (colormap maps 0 to a non-black colour)
                for (int k = 0; k < d.W_half * d.H_half; k++) {
                    int hrow = k / d.W_half, hcol = k % d.W_half;
                    int frow = std::min(d.H-1, (int)(hrow * sy));
                    int fcol = std::min(d.W-1, (int)(hcol * sx));
                    if (!d.h_vmask[i][frow * d.W + fcol])
                        mag_color.at<cv::Vec3b>(hrow, hcol) = {0, 0, 0};
                }
                cv::imwrite(debug_dir + "/07_disp_mag_" + cnames[i] + ".jpg",
                            mag_color, {cv::IMWRITE_JPEG_QUALITY, 92});
            }
        }

        // 07b: Gain compensation before/after comparison strip (FL-FC and FC-FR zones)
        if (cfg.gain_compensation && !h_pregain_fl.empty() &&
                !h_postgain_fl.empty() && !h_postgain_fr.empty()) {
            const int CROP_HALF = 350;
            auto to_bgr = [&](const std::vector<uint8_t>& src4) -> cv::Mat {
                cv::Mat m4(d.H, d.W, CV_8UC4, const_cast<uint8_t*>(src4.data()));
                cv::Mat m3;
                cv::cvtColor(m4, m3, cv::COLOR_BGRA2BGR);
                return m3;
            };
            cv::Mat pre_fl  = to_bgr(h_pregain_fl);
            cv::Mat post_fl = to_bgr(h_postgain_fl);
            cv::Mat pre_fr  = to_bgr(h_pregain_fr);
            cv::Mat post_fr = to_bgr(h_postgain_fr);

            auto make_col = [&](cv::Mat& pre, cv::Mat& post, int sc) -> cv::Mat {
                int x0 = std::max(0, sc - CROP_HALF);
                int x1 = std::min(d.W, sc + CROP_HALF);
                cv::Mat pc  = pre (cv::Rect(x0, 0, x1-x0, d.H)).clone();
                cv::Mat psc = post(cv::Rect(x0, 0, x1-x0, d.H)).clone();
                int sc_rel = sc - x0;
                cv::line(pc,  {sc_rel, 0}, {sc_rel, d.H-1}, {0,255,255}, 2);
                cv::line(psc, {sc_rel, 0}, {sc_rel, d.H-1}, {0,255,255}, 2);
                auto lbl = [](cv::Mat& m, const char* t) {
                    cv::putText(m, t, {6, 40}, cv::FONT_HERSHEY_DUPLEX, 1.2,
                                {0,0,0}, 4, cv::LINE_AA);
                    cv::putText(m, t, {6, 40}, cv::FONT_HERSHEY_DUPLEX, 1.2,
                                {255,255,255}, 2, cv::LINE_AA);
                };
                lbl(pc,  "pre-gain");
                lbl(psc, "post-gain");
                cv::Mat div(4, x1-x0, CV_8UC3, cv::Scalar(255,255,0));
                cv::Mat col;
                cv::vconcat(std::vector<cv::Mat>{pc, div, psc}, col);
                cv::Mat out;
                cv::resize(col, out, cv::Size(col.cols/2, col.rows/2));
                return out;
            };

            cv::Mat lc = make_col(pre_fl, post_fl, d.seam_col_lc);
            cv::Mat cr = make_col(pre_fr, post_fr, d.seam_col_cr);
            int H_out = std::max(lc.rows, cr.rows);
            if (lc.rows < H_out)
                cv::copyMakeBorder(lc, lc, 0, H_out-lc.rows, 0, 0,
                                   cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
            if (cr.rows < H_out)
                cv::copyMakeBorder(cr, cr, 0, H_out-cr.rows, 0, 0,
                                   cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
            cv::Mat sep(H_out, 8, CV_8UC3, cv::Scalar(50,50,50));
            cv::Mat strip;
            cv::hconcat(std::vector<cv::Mat>{lc, sep, cr}, strip);
            cv::imwrite(debug_dir + "/07b_gain_compare.jpg",
                        strip, {cv::IMWRITE_JPEG_QUALITY, 92});
        }

        // 08: Camera coverage zones: FL(blue) FC(green) FR(red) overlaps(mixed)
        {
            cv::Mat zones(d.H, d.W, CV_8UC3, cv::Scalar(20, 20, 20));
            for (int row = 0; row < d.H; row++) {
                for (int col = 0; col < d.W; col++) {
                    int k = row * d.W + col;
                    bool va = d.h_vmask[0][k] != 0;
                    bool vb = d.h_vmask[1][k] != 0;
                    bool vc = d.h_vmask[2][k] != 0;
                    cv::Vec3b color(0, 0, 0);
                    if      (va && vb && vc) color = {200, 200, 200};   // all-three: white
                    else if (va && vb)       color = {200, 200,  80};   // FL∩FC: cyan
                    else if (vb && vc)       color = { 80, 200, 200};   // FC∩FR: yellow
                    else if (va)             color = {180,  60,  60};   // FL: blue
                    else if (vb)             color = { 60, 180,  60};   // FC: green
                    else if (vc)             color = { 60,  60, 180};   // FR: red
                    zones.at<cv::Vec3b>(row, col) = color;
                }
            }
            cv::line(zones, {d.seam_col_lc, 0}, {d.seam_col_lc, d.H - 1},
                     {0, 255, 255}, 2);
            cv::line(zones, {d.seam_col_cr, 0}, {d.seam_col_cr, d.H - 1},
                     {0, 255, 255}, 2);
            auto put_label_z = [&](const char* txt, int x, int y) {
                cv::putText(zones, txt, {x, y}, cv::FONT_HERSHEY_DUPLEX, 1.4,
                            {0, 0, 0}, 4, cv::LINE_AA);
                cv::putText(zones, txt, {x, y}, cv::FONT_HERSHEY_DUPLEX, 1.4,
                            {255, 255, 255}, 2, cv::LINE_AA);
            };
            put_label_z("FL", std::max(8,      d.seam_col_lc / 2 - 24),              d.H / 2);
            put_label_z("FC", (d.seam_col_lc + d.seam_col_cr) / 2 - 24,             d.H / 2);
            put_label_z("FR", std::min(d.W-64, (d.seam_col_cr + d.W) / 2 - 24),     d.H / 2);
            cv::imwrite(debug_dir + "/08_coverage_zones.jpg",
                        zones, {cv::IMWRITE_JPEG_QUALITY, 92});
        }
    }

    // -- Stage 8: SeamDP (two overlap pairs in parallel) ----------------------
    // FL<->FC on d.stream, FC<->FR on d.stream3 (own scratch + CUDA Graph; the
    // graph bakes in buffer pointers, so each stream needs its own capture).
    // FC<->FR also reads FC (stream2), so stream3 waits on warp_done_2.
    CUDA_CHECK(cudaStreamWaitEvent(d.stream3, d.warp_done_2, 0));

    // FL<->FC -- GPU writes seam directly to d_seam[0]
    computeSeamCostU8(d.d_warped[0], d.d_warped[1],
        d.d_vmask[0], d.d_vmask[1],
        d.W, d.H, K_SEAM_GRAD_WEIGHT, d.d_cost, d.stream);
    if (!debug_dir.empty()) {
        CUDA_CHECK(cudaStreamSynchronize(d.stream));
        std::vector<float> h_cost((size_t)d.W * d.H);
        CUDA_CHECK(cudaMemcpy(h_cost.data(), d.d_cost,
            (size_t)d.W * d.H * sizeof(float), cudaMemcpyDeviceToHost));
        // Normalize only within valid overlap (exclude large penalty values).
        // Cost kernel sets ~1e6 outside valid mask; actual pixel costs are <=255.
        constexpr float PENALTY = 1000.f;
        float cmax = 1.f;
        for (float v : h_cost) if (v < PENALTY) cmax = std::max(cmax, v);
        cv::Mat cu8(d.H, d.W, CV_8UC1);
        for (int k = 0; k < d.W * d.H; k++)
            cu8.data[k] = (h_cost[k] >= PENALTY)
                ? 0 : (uint8_t)(255.f * h_cost[k] / cmax);
        cv::Mat ccol; cv::applyColorMap(cu8, ccol, cv::COLORMAP_HOT);
        // Crop to the bounding box of the valid overlap region (non-penalty pixels)
        {
            constexpr int SEAM_MARGIN = 60;
            int x0 = d.W, x1 = 0, y0 = d.H, y1 = 0;
            for (int r = 0; r < d.H; r++)
                for (int c = 0; c < d.W; c++)
                    if (h_cost[r * d.W + c] < PENALTY) {
                        x0 = std::min(x0, c); x1 = std::max(x1, c);
                        y0 = std::min(y0, r); y1 = std::max(y1, r);
                    }
            if (x1 > x0 && y1 > y0) {
                x0 = std::max(0, x0 - SEAM_MARGIN); x1 = std::min(d.W, x1 + SEAM_MARGIN);
                y0 = std::max(0, y0 - SEAM_MARGIN); y1 = std::min(d.H, y1 + SEAM_MARGIN);
                cv::imwrite(debug_dir + "/04_seam_cost_lc.jpg",
                            ccol(cv::Rect(x0, y0, x1 - x0, y1 - y0)),
                            {cv::IMWRITE_JPEG_QUALITY, 90});
            }
        }
    }
    findSeamDPGraph(d.d_cost, d.W, d.H, d.d_seam[0],
        d.d_dp, d.d_backtrack,
        d.seam_graph, d.seam_graph_exec, d.seam_graph_ready, d.stream);

    // FC<->FR -- GPU writes seam directly to d_seam[1] (on stream3, in parallel)
    computeSeamCostU8(d.d_warped[1], d.d_warped[2],
        d.d_vmask[1], d.d_vmask[2],
        d.W, d.H, K_SEAM_GRAD_WEIGHT, d.d_cost2, d.stream3);
    if (!debug_dir.empty()) {
        CUDA_CHECK(cudaStreamSynchronize(d.stream3));
        std::vector<float> h_cost((size_t)d.W * d.H);
        CUDA_CHECK(cudaMemcpy(h_cost.data(), d.d_cost2,
            (size_t)d.W * d.H * sizeof(float), cudaMemcpyDeviceToHost));
        constexpr float PENALTY = 1000.f;
        float cmax = 1.f;
        for (float v : h_cost) if (v < PENALTY) cmax = std::max(cmax, v);
        cv::Mat cu8(d.H, d.W, CV_8UC1);
        for (int k = 0; k < d.W * d.H; k++)
            cu8.data[k] = (h_cost[k] >= PENALTY)
                ? 0 : (uint8_t)(255.f * h_cost[k] / cmax);
        cv::Mat ccol; cv::applyColorMap(cu8, ccol, cv::COLORMAP_HOT);
        {
            constexpr int SEAM_MARGIN = 60;
            int x0 = d.W, x1 = 0, y0 = d.H, y1 = 0;
            for (int r = 0; r < d.H; r++)
                for (int c = 0; c < d.W; c++)
                    if (h_cost[r * d.W + c] < PENALTY) {
                        x0 = std::min(x0, c); x1 = std::max(x1, c);
                        y0 = std::min(y0, r); y1 = std::max(y1, r);
                    }
            if (x1 > x0 && y1 > y0) {
                x0 = std::max(0, x0 - SEAM_MARGIN); x1 = std::min(d.W, x1 + SEAM_MARGIN);
                y0 = std::max(0, y0 - SEAM_MARGIN); y1 = std::min(d.H, y1 + SEAM_MARGIN);
                cv::imwrite(debug_dir + "/04_seam_cost_cr.jpg",
                            ccol(cv::Rect(x0, y0, x1 - x0, y1 - y0)),
                            {cv::IMWRITE_JPEG_QUALITY, 90});
            }
        }
    }
    findSeamDPGraph(d.d_cost2, d.W, d.H, d.d_seam[1],
        d.d_dp2, d.d_backtrack2,
        d.seam_graph2, d.seam_graph2_exec, d.seam_graph2_ready, d.stream3);

    // Join stream3's seam DP back into stream before composite/D2H consume it.
    CUDA_CHECK(cudaEventRecord(d.seam2_done, d.stream3));
    CUDA_CHECK(cudaStreamWaitEvent(d.stream, d.seam2_done, 0));

    auto t4 = (profile || metrics != nullptr) ? T() : hrc::now();  // after seam DP

    // -- Stage 9: Composite ----------------------------------------------------
    const int feather_half = (cfg.feather_half_px >= 0)
                             ? cfg.feather_half_px : K_FEATHER_HALF_PX;
    compositeThreeU8(
        d.d_warped[0], d.d_warped[1], d.d_warped[2],
        d.d_vmask[0],  d.d_vmask[1],  d.d_vmask[2],
        d.d_seam[0], d.d_seam[1],
        feather_half,
        d.W, d.H, d.d_canvas_buf, d.stream);

    // -- D2H: canvas -> cv::Mat -------------------------------------------------
    // sync (default): block until the canvas lands. async_d2h: record an event
    // the next process() syncs on, returning while the D2H is in flight.
    // skip_canvas_d2h: no host consumer (NVENC reads the device pointer), so
    // skip the memcpy + sync entirely.
    if (cfg.skip_canvas_d2h) {
        // No-op: caller owns synchronisation via canvasDevicePtr() / mainStream().
    } else {
        CUDA_CHECK(cudaMemcpyAsync(d.h_canvas, d.d_canvas_buf,
            (size_t)d.W * d.H * 3, cudaMemcpyDeviceToHost, d.stream));
        if (cfg.async_d2h) {
            CUDA_CHECK(cudaEventRecord(d.d2h_done, d.stream));
            d.d2h_pending = true;
        } else {
            CUDA_CHECK(cudaStreamSynchronize(d.stream));
        }
    }
    auto t5 = hrc::now();

    // Debug, profile, metrics, eval-dump paths below all read h_canvas or
    // walk other buffers that race with the in-flight D2H. Force the sync
    // when any of them are active so the async path stays a pure benchmark
    // optimisation and never corrupts a debug/eval frame.
    const bool need_canvas_sync = cfg.async_d2h && d.d2h_pending &&
        (!debug_dir.empty() || metrics != nullptr || !eval_dump_dir.empty());
    if (need_canvas_sync) {
        CUDA_CHECK(cudaEventSynchronize(d.d2h_done));
        d.d2h_pending = false;
    }

    // -- Debug: seam path overlay ----------------------------------------------
    if (!debug_dir.empty()) {
        // D2H seam arrays (device -> host, synchronous -- already synced at t5)
        std::vector<int> seam0(d.H), seam1(d.H);
        CUDA_CHECK(cudaMemcpy(seam0.data(), d.d_seam[0], (size_t)d.H * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(seam1.data(), d.d_seam[1], (size_t)d.H * sizeof(int), cudaMemcpyDeviceToHost));

        cv::Mat seam_vis(d.H, d.W, CV_8UC3, d.h_canvas);
        seam_vis = seam_vis.clone();
        for (int row = 0; row < d.H; row++) {
            for (int dc = -1; dc <= 1; dc++) {
                int c0 = std::max(0, std::min(d.W - 1, seam0[row] + dc));
                int c1 = std::max(0, std::min(d.W - 1, seam1[row] + dc));
                seam_vis.at<cv::Vec3b>(row, c0) = cv::Vec3b(0, 0, 255);   // red  = FL<->FC
                seam_vis.at<cv::Vec3b>(row, c1) = cv::Vec3b(0, 255, 0);   // green = FC<->FR
            }
        }
        cv::imwrite(debug_dir + "/05_seam_paths.jpg", seam_vis, {cv::IMWRITE_JPEG_QUALITY, 90});

        // 09: Final composite with solid header bars per zone + labelled seam lines
        {
            cv::Mat annotated(d.H, d.W, CV_8UC3, d.h_canvas);
            annotated = annotated.clone();

            // Seam lines (3 px wide): red = FL↔FC, green = FC↔FR
            for (int row = 0; row < d.H; row++) {
                for (int dc = -1; dc <= 1; dc++) {
                    int c0 = std::max(0, std::min(d.W - 1, seam0[row] + dc));
                    int c1 = std::max(0, std::min(d.W - 1, seam1[row] + dc));
                    annotated.at<cv::Vec3b>(row, c0) = cv::Vec3b(0,   0, 255);
                    annotated.at<cv::Vec3b>(row, c1) = cv::Vec3b(0, 255,   0);
                }
            }

            // Solid coloured header bar: 40 px tall at the top of each zone.
            // Seam extents vary per row; use the median seam column for bar boundaries.
            std::vector<int> s0_sorted(seam0), s1_sorted(seam1);
            std::sort(s0_sorted.begin(), s0_sorted.end());
            std::sort(s1_sorted.begin(), s1_sorted.end());
            int med0 = s0_sorted[s0_sorted.size() / 2];
            int med1 = s1_sorted[s1_sorted.size() / 2];
            const int BAR_H = 40;
            // FL bar (blue): columns 0..med0
            cv::rectangle(annotated, {0, 0}, {med0, BAR_H - 1},
                          cv::Scalar(200, 80, 80), cv::FILLED);
            // FC bar (green): columns med0+1..med1
            cv::rectangle(annotated, {med0 + 1, 0}, {med1, BAR_H - 1},
                          cv::Scalar(80, 200, 80), cv::FILLED);
            // FR bar (red): columns med1+1..W-1
            cv::rectangle(annotated, {med1 + 1, 0}, {d.W - 1, BAR_H - 1},
                          cv::Scalar(80, 80, 200), cv::FILLED);

            // Labels centred in each bar
            auto put_bar_label = [&](const char* txt, int x_centre) {
                int baseline = 0;
                cv::Size sz = cv::getTextSize(txt, cv::FONT_HERSHEY_DUPLEX,
                                              1.1, 2, &baseline);
                int tx = std::max(4, x_centre - sz.width / 2);
                int ty = (BAR_H + sz.height) / 2;
                cv::putText(annotated, txt, {tx, ty},
                            cv::FONT_HERSHEY_DUPLEX, 1.1,
                            {0, 0, 0}, 4, cv::LINE_AA);
                cv::putText(annotated, txt, {tx, ty},
                            cv::FONT_HERSHEY_DUPLEX, 1.1,
                            {255, 255, 255}, 2, cv::LINE_AA);
            };
            put_bar_label("FL", med0 / 2);
            put_bar_label("FC", (med0 + med1) / 2);
            put_bar_label("FR", (med1 + d.W) / 2);

            cv::imwrite(debug_dir + "/09_final_annotated.jpg",
                        annotated, {cv::IMWRITE_JPEG_QUALITY, 92});
        }

        // 10: Feather weight map — per-pixel blend contribution (B=FL, G=FC, R=FR)
        {
            static constexpr float kPi = 3.14159265358979f;
            const float K = (float)feather_half;
            cv::Mat fmap(d.H, d.W, CV_8UC3, cv::Scalar(0, 0, 0));
            for (int r = 0; r < d.H; r++) {
                int s0 = seam0[r], s1 = seam1[r];
                for (int c = 0; c < d.W; c++) {
                    int k = r * d.W + c;
                    if (!d.h_vmask[0][k] && !d.h_vmask[1][k] && !d.h_vmask[2][k])
                        continue;
                    float d0f = (float)(c - s0);
                    float d1f = (float)(c - s1);
                    float fl_w = (d0f <= -K) ? 1.f : (d0f >= K) ? 0.f :
                        0.5f * (1.f + std::cos(kPi * d0f / K));
                    float fr_w = (d1f >= K) ? 1.f : (d1f <= -K) ? 0.f :
                        0.5f * (1.f - std::cos(kPi * d1f / K));
                    float fc_w = std::max(0.f, 1.f - fl_w - fr_w);
                    fmap.at<cv::Vec3b>(r, c) = cv::Vec3b(
                        (uint8_t)(255.f * fl_w),
                        (uint8_t)(255.f * fc_w),
                        (uint8_t)(255.f * fr_w));
                }
            }
            cv::imwrite(debug_dir + "/10_feather_weights.jpg",
                        fmap, {cv::IMWRITE_JPEG_QUALITY, 92});
        }
    }

    if (profile) {
        printf("[profile] upload+proj+D2H=%.1f  CPU_TPS=%.1f  warp=%.1f  seam=%.1f  composite+D2H=%.1f  total=%.1f ms\n",
            ms(t1-t0).count(), ms(t2-t1).count(), ms(t3-t2).count(),
            ms(t4-t3).count(), ms(t5-t4).count(), ms(t5-t0).count());
        fflush(stdout);
    }

    // -- Metrics + eval dump --------------------------------------------------
    const bool want_metrics = (metrics != nullptr);
    const bool want_dump    = !eval_dump_dir.empty();

    std::vector<int> h_seam0, h_seam1;
    std::vector<uint8_t> h_warped_fl, h_warped_fc, h_warped_fr;
    if (want_metrics || want_dump) {
        h_seam0.resize(d.H);
        h_seam1.resize(d.H);
        CUDA_CHECK(cudaMemcpy(h_seam0.data(), d.d_seam[0],
            (size_t)d.H * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_seam1.data(), d.d_seam[1],
            (size_t)d.H * sizeof(int), cudaMemcpyDeviceToHost));

        // Refresh h_vmask from the per-frame post-TPS d_vmask (the warp also
        // excludes out-of-bounds samples and AV2 black corners), so the
        // metric/dump paths don't count empty pixels.
        size_t WH = (size_t)d.W * d.H;
        for (int i = 0; i < 3; i++)
            CUDA_CHECK(cudaMemcpy(d.h_vmask[i], d.d_vmask[i], WH,
                                  cudaMemcpyDeviceToHost));

        // d_warped is W*H*4 (R,G,B,padding). Pull 4-channel then strip alpha to
        // give the eval routines the W*H*3 layout they assume.
        size_t WH3 = (size_t)d.W * d.H * 3;
        size_t WH4_local = (size_t)d.W * d.H * 4;
        std::vector<uint8_t> h_w4(WH4_local);
        h_warped_fl.resize(WH3);
        h_warped_fc.resize(WH3);
        h_warped_fr.resize(WH3);
        auto pull = [&](int cam, std::vector<uint8_t>& out) {
            CUDA_CHECK(cudaMemcpy(h_w4.data(), d.d_warped[cam], WH4_local, cudaMemcpyDeviceToHost));
            cv::Mat m4(d.H, d.W, CV_8UC4, h_w4.data());
            cv::Mat m3(d.H, d.W, CV_8UC3, out.data());
            cv::cvtColor(m4, m3, cv::COLOR_BGRA2BGR);
        };
        pull(0, h_warped_fl);
        pull(1, h_warped_fc);
        pull(2, h_warped_fr);
    }

    if (want_metrics) {
        metrics->n_shared_FL_FC = n_shared_per_overlap[0];
        metrics->n_shared_FC_FR = n_shared_per_overlap[1];
        metrics->t_project_ms   = ms(t1-t0).count();
        metrics->t_tps_ms       = ms(t2-t1).count();
        metrics->t_warp_ms      = ms(t3-t2).count();
        metrics->t_seam_ms      = ms(t4-t3).count();
        metrics->t_composite_ms = ms(t5-t4).count();
        metrics->t_total_ms     = ms(t5-t0).count();

        // -- Per-camera TPS warp magnitude (camera-pixel units) ----------------
        // Computed directly from the ctrl-pt displacements -- no GPU readback.
        auto warpStats = [](const std::vector<float>& dx,
                            const std::vector<float>& dy,
                            float& m, float& p95, float& mx)
        {
            if (dx.empty()) { m = p95 = mx = 0.f; return; }
            std::vector<float> mag(dx.size());
            for (size_t k = 0; k < dx.size(); k++)
                mag[k] = std::sqrt(dx[k]*dx[k] + dy[k]*dy[k]);
            double s = 0;
            for (float v : mag) s += v;
            m = (float)(s / mag.size());
            std::vector<float> sorted = mag;
            std::sort(sorted.begin(), sorted.end());
            size_t p95_idx = (size_t)std::floor(0.95 * sorted.size());
            if (p95_idx >= sorted.size()) p95_idx = sorted.size() - 1;
            p95 = sorted[p95_idx];
            mx  = sorted.back();
        };
        warpStats(cam_tps[0].disp_x, cam_tps[0].disp_y,
                  metrics->warp_mean_FL, metrics->warp_p95_FL, metrics->warp_max_FL);
        warpStats(cam_tps[1].disp_x, cam_tps[1].disp_y,
                  metrics->warp_mean_FC, metrics->warp_p95_FC, metrics->warp_max_FC);
        warpStats(cam_tps[2].disp_x, cam_tps[2].disp_y,
                  metrics->warp_mean_FR, metrics->warp_p95_FR, metrics->warp_max_FR);

        // -- Per-camera TPS bending energy (Bookstein 1989) -------------------
        // E_bend = w_xᵀK w_x + w_yᵀK w_y, K_ij = U(r²) (matches solveTPS).
        // Ctrl coords normalised to [0,1]²; O(N²), sub-ms at N ≤ 512.
        auto bendingEnergy = [](const std::vector<float>& ctrl_x_n,
                                const std::vector<float>& ctrl_y_n,
                                const std::vector<float>& w_x,
                                const std::vector<float>& w_y) -> float
        {
            int N = (int)ctrl_x_n.size();
            if (N < 3 || (int)w_x.size() < N || (int)w_y.size() < N) return 0.f;
            auto tpsU = [](float r2) -> float {
                return (r2 > 1e-12f) ? 0.5f * r2 * std::log(r2) : 0.f;
            };
            double Ex = 0.0, Ey = 0.0;
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    float dx = ctrl_x_n[i] - ctrl_x_n[j];
                    float dy = ctrl_y_n[i] - ctrl_y_n[j];
                    float u  = tpsU(dx*dx + dy*dy);
                    Ex += (double)w_x[i] * w_x[j] * u;
                    Ey += (double)w_y[i] * w_y[j] * u;
                }
            }
            return (float)(Ex + Ey);
        };
        metrics->tps_bend_FL = bendingEnergy(cam_tps[0].ctrl_x_n, cam_tps[0].ctrl_y_n,
                                             cam_tps[0].w_x,     cam_tps[0].w_y);
        metrics->tps_bend_FC = bendingEnergy(cam_tps[1].ctrl_x_n, cam_tps[1].ctrl_y_n,
                                             cam_tps[1].w_x,     cam_tps[1].w_y);
        metrics->tps_bend_FR = bendingEnergy(cam_tps[2].ctrl_x_n, cam_tps[2].ctrl_y_n,
                                             cam_tps[2].w_x,     cam_tps[2].w_y);

        // -- Full pairwise-overlap photometric agreement (Y / Cb / Cr) --------
        // PSNR over every pixel where both cameras have a valid sample
        // (vmask_a ∩ vmask_b), incl. the extrapolation region. BT.709 per channel.
        auto overlapStats = [&](const std::vector<uint8_t>& A,
                                const std::vector<uint8_t>& B,
                                int ca, int cb,
                                int& out_n,
                                float& out_mse_y, float& out_mse_cb, float& out_mse_cr,
                                float& out_psnr_y, float& out_psnr_cb, float& out_psnr_cr)
        {
            // BT.709 BGR->{Y, Cb, Cr}, full-range. Loop fuses convert + diff +
            // MSE so we never materialise a full Y/Cb/Cr buffer for both
            // images (saves 6 x W·H bytes per frame).
            double s_y = 0, s_cb = 0, s_cr = 0;
            long long n = 0;
            for (int i = 0; i < d.H * d.W; i++) {
                if (!d.h_vmask[ca][i] || !d.h_vmask[cb][i]) continue;
                size_t idx = (size_t)i * 3;
                // OpenCV BGR layout: A[idx+0]=B, A[idx+1]=G, A[idx+2]=R.
                float Ab = (float)A[idx+0], Ag = (float)A[idx+1], Ar = (float)A[idx+2];
                float Bb = (float)B[idx+0], Bg = (float)B[idx+1], Br = (float)B[idx+2];
                float Ya = 0.2126f * Ar + 0.7152f * Ag + 0.0722f * Ab;
                float Yb = 0.2126f * Br + 0.7152f * Bg + 0.0722f * Bb;
                float Cba = (Ab - Ya) / 1.8556f;
                float Cbb = (Bb - Yb) / 1.8556f;
                float Cra = (Ar - Ya) / 1.5748f;
                float Crb = (Br - Yb) / 1.5748f;
                float dy  = Ya  - Yb;
                float dcb = Cba - Cbb;
                float dcr = Cra - Crb;
                s_y  += (double)dy  * dy;
                s_cb += (double)dcb * dcb;
                s_cr += (double)dcr * dcr;
                n++;
            }
            out_n = (int)n;
            auto psnr = [](double mse) {
                return (mse > 0.0)
                    ? (float)(10.0 * std::log10(255.0 * 255.0 / mse))
                    : 99.f;
            };
            if (n > 0) {
                out_mse_y  = (float)(s_y  / (double)n);
                out_mse_cb = (float)(s_cb / (double)n);
                out_mse_cr = (float)(s_cr / (double)n);
            } else {
                out_mse_y = out_mse_cb = out_mse_cr = 0.f;
            }
            out_psnr_y  = psnr(out_mse_y);
            out_psnr_cb = psnr(out_mse_cb);
            out_psnr_cr = psnr(out_mse_cr);
        };
        overlapStats(h_warped_fl, h_warped_fc, 0, 1,
                     metrics->overlap_n_FL_FC,
                     metrics->overlap_mse_y_FL_FC,
                     metrics->overlap_mse_cb_FL_FC,
                     metrics->overlap_mse_cr_FL_FC,
                     metrics->overlap_psnr_y_FL_FC,
                     metrics->overlap_psnr_cb_FL_FC,
                     metrics->overlap_psnr_cr_FL_FC);
        overlapStats(h_warped_fc, h_warped_fr, 1, 2,
                     metrics->overlap_n_FC_FR,
                     metrics->overlap_mse_y_FC_FR,
                     metrics->overlap_mse_cb_FC_FR,
                     metrics->overlap_mse_cr_FC_FR,
                     metrics->overlap_psnr_y_FC_FR,
                     metrics->overlap_psnr_cb_FC_FR,
                     metrics->overlap_psnr_cr_FC_FR);

        // Seam L1: mean |ΔRGB| at the seam column between the two source cameras.
        // For row r and seam column c, measure |FL[r,c] - FC[r,c]| / 3 averaged
        // over B/G/R; then aggregate mean and std across rows.
        auto seamStats = [&](const std::vector<int>& seam,
                             const std::vector<uint8_t>& A,
                             const std::vector<uint8_t>& B,
                             float& out_mean, float& out_std)
        {
            std::vector<float> per_row;
            per_row.reserve(d.H);
            for (int r = 0; r < d.H; r++) {
                int c = seam[r];
                if (c < 0 || c >= d.W) continue;
                if (!d.h_vmask[0][r*d.W+c] && !d.h_vmask[1][r*d.W+c]) continue;
                size_t idx = ((size_t)r * d.W + c) * 3;
                float diff = 0.f;
                for (int ch = 0; ch < 3; ch++) {
                    int da = (int)A[idx+ch];
                    int db = (int)B[idx+ch];
                    diff += std::abs(da - db);
                }
                diff /= 3.f;
                per_row.push_back(diff);
            }
            if (per_row.empty()) { out_mean = 0.f; out_std = 0.f; return; }
            double s = 0, s2 = 0;
            for (float v : per_row) { s += v; s2 += (double)v*v; }
            double m = s / per_row.size();
            double var = s2 / per_row.size() - m*m;
            out_mean = (float)m;
            out_std  = (float)std::sqrt(std::max(0.0, var));
        };
        seamStats(h_seam0, h_warped_fl, h_warped_fc,
                  metrics->seam_l1_FL_FC, metrics->seam_std_FL_FC);
        seamStats(h_seam1, h_warped_fc, h_warped_fr,
                  metrics->seam_l1_FC_FR, metrics->seam_std_FC_FR);
    }

    if (want_dump) {
        namespace fs = std::filesystem;
        fs::create_directories(eval_dump_dir);

        // Warped PNGs
        cv::Mat mfl(d.H, d.W, CV_8UC3, h_warped_fl.data());
        cv::Mat mfc(d.H, d.W, CV_8UC3, h_warped_fc.data());
        cv::Mat mfr(d.H, d.W, CV_8UC3, h_warped_fr.data());
        cv::imwrite(eval_dump_dir + "/warped_FL.png", mfl);
        cv::imwrite(eval_dump_dir + "/warped_FC.png", mfc);
        cv::imwrite(eval_dump_dir + "/warped_FR.png", mfr);

        // Full pairwise-overlap masks: every canvas pixel where both
        // adjacent cameras have a valid post-warp sample. The Python
        // rich-pass SSIM/LPIPS read these masks, so they evaluate on the
        // same region as the C++ cheap-pass PSNR.
        cv::Mat mask_lc(d.H, d.W, CV_8UC1);
        cv::Mat mask_cr(d.H, d.W, CV_8UC1);
        for (int i = 0; i < d.H * d.W; i++) {
            mask_lc.data[i] = (d.h_vmask[0][i] && d.h_vmask[1][i]) ? 255 : 0;
            mask_cr.data[i] = (d.h_vmask[1][i] && d.h_vmask[2][i]) ? 255 : 0;
        }
        cv::imwrite(eval_dump_dir + "/mask_FL_FC.png", mask_lc);
        cv::imwrite(eval_dump_dir + "/mask_FC_FR.png", mask_cr);

        // Seam columns (int32 per row)
        {
            std::ofstream fs0(eval_dump_dir + "/seam_FL_FC.bin", std::ios::binary);
            fs0.write(reinterpret_cast<const char*>(h_seam0.data()),
                      (std::streamsize)h_seam0.size() * sizeof(int));
            std::ofstream fs1(eval_dump_dir + "/seam_FC_FR.bin", std::ios::binary);
            fs1.write(reinterpret_cast<const char*>(h_seam1.data()),
                      (std::streamsize)h_seam1.size() * sizeof(int));
        }

        // Hold-out LiDAR points CSV
        {
            std::ofstream fcsv(eval_dump_dir + "/lidar_holdout.csv");
            fcsv << "overlap,x,y,z,"
                    "u_FL,v_FL,u_FC,v_FC,u_FR,v_FR,"
                    "cu_FL,cv_FL,cu_FC,cv_FC,cu_FR,cv_FR,"
                    "cu_tps_FL,cv_tps_FL,cu_tps_FC,cv_tps_FC,cu_tps_FR,cv_tps_FR\n";
            fcsv << std::fixed;
            for (const auto& hp : holdout_pts) {
                fcsv << hp.overlap << ","
                     << hp.xyz[0] << "," << hp.xyz[1] << "," << hp.xyz[2];
                for (int ci = 0; ci < 3; ci++)
                    fcsv << "," << hp.cam_u[ci] << "," << hp.cam_v[ci];
                for (int ci = 0; ci < 3; ci++)
                    fcsv << "," << hp.cvs_u[ci] << "," << hp.cvs_v[ci];
                for (int ci = 0; ci < 3; ci++)
                    fcsv << "," << hp.cvs_u_tps[ci] << "," << hp.cvs_v_tps[ci];
                fcsv << "\n";
            }
        }

        // Timings JSON
        {
            std::ofstream fj(eval_dump_dir + "/timings.json");
            fj << "{\n"
               << "  \"t_project_ms\": " << ms(t1-t0).count() << ",\n"
               << "  \"t_tps_ms\": "     << ms(t2-t1).count() << ",\n"
               << "  \"t_warp_ms\": "    << ms(t3-t2).count() << ",\n"
               << "  \"t_seam_ms\": "    << ms(t4-t3).count() << ",\n"
               << "  \"t_composite_ms\": " << ms(t5-t4).count() << ",\n"
               << "  \"t_total_ms\": "   << ms(t5-t0).count() << ",\n"
               << "  \"n_shared_FL_FC\": " << n_shared_per_overlap[0] << ",\n"
               << "  \"n_shared_FC_FR\": " << n_shared_per_overlap[1] << ",\n"
               << "  \"n_holdout\": "    << holdout_pts.size() << "\n"
               << "}\n";
        }
    }

    // Mark temporal state available for the next frame, and bump the frame
    // counter so the periodic full-DP fallback can fire on schedule.
    d.temporal_initialized = true;
    d.frame_counter++;

    // skip_canvas_d2h: host buffer was never filled. Return an empty Mat
    // and let the caller read d.d_canvas_buf via canvasDevicePtr() instead.
    if (cfg.skip_canvas_d2h)
        return cv::Mat();

    // Non-owning view over the pinned host buffer. Caller must consume / copy
    // before invoking process() again, since the next call overwrites h_canvas.
    return cv::Mat(d.H, d.W, CV_8UC3, d.h_canvas);
}

// -----------------------------------------------------------------------------
// waitD2H: drain any deferred canvas D2H from cfg.async_d2h mode.
// -----------------------------------------------------------------------------

void LidarTpsPipeline::waitD2H() {
    auto& d = *d_;
    if (d.d2h_pending) {
        CUDA_CHECK(cudaEventSynchronize(d.d2h_done));
        d.d2h_pending = false;
    }
}

const uint8_t* LidarTpsPipeline::canvasDevicePtr() const {
    return d_->d_canvas_buf;
}

cudaStream_t LidarTpsPipeline::mainStream() const {
    return d_->stream;
}

int LidarTpsPipeline::canvasWidth()  const { return d_->W; }
int LidarTpsPipeline::canvasHeight() const { return d_->H; }

} // namespace lidartps
