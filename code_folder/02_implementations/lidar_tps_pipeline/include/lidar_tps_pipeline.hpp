#pragma once
#include "lidar_types.hpp"
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <array>
#include <string>

namespace lidartps {

// Per-frame evaluation metrics populated by process() when `metrics` is non-null.
struct ProcessMetrics {
    int   n_shared_FL_FC = 0;  // post-range-filter shared-pt count (FL<->FC overlap)
    int   n_shared_FC_FR = 0;
    float seam_l1_FL_FC  = 0.f; // mean |ΔRGB| across seam column (per-row, averaged)
    float seam_l1_FC_FR  = 0.f;
    float seam_std_FL_FC = 0.f; // std-dev of per-row seam |ΔRGB|
    float seam_std_FC_FR = 0.f;

    // Photometric agreement over the full pairwise overlap (vmask_a ∩ vmask_b).
    // BT.709 Y/Cb/Cr; MSE kept alongside PSNR (=10·log10(255²/MSE)) so a
    // weighted PSNR_w can be formed in MSE space.
    int   overlap_n_FL_FC = 0;     // # pixels used (vmask_a ∩ vmask_b)
    int   overlap_n_FC_FR = 0;
    float overlap_mse_y_FL_FC  = 0.f, overlap_mse_y_FC_FR  = 0.f;
    float overlap_mse_cb_FL_FC = 0.f, overlap_mse_cb_FC_FR = 0.f;
    float overlap_mse_cr_FL_FC = 0.f, overlap_mse_cr_FC_FR = 0.f;
    float overlap_psnr_y_FL_FC  = 0.f, overlap_psnr_y_FC_FR  = 0.f;
    float overlap_psnr_cb_FL_FC = 0.f, overlap_psnr_cb_FC_FR = 0.f;
    float overlap_psnr_cr_FL_FC = 0.f, overlap_psnr_cr_FC_FR = 0.f;

    // TPS displacement magnitude at ctrl pts (camera px): mean/p95/max.
    // FC is at its rotation baseline (zero); FL/FR carry the warp.
    float warp_mean_FL = 0.f, warp_p95_FL = 0.f, warp_max_FL = 0.f;
    float warp_mean_FC = 0.f, warp_p95_FC = 0.f, warp_max_FC = 0.f;
    float warp_mean_FR = 0.f, warp_p95_FR = 0.f, warp_max_FR = 0.f;

    // TPS bending energy per camera, E_bend = wᵀKw (Bookstein 1989); x- and
    // y-warp summed. The smoothness scalar the solve minimises (lower = smoother).
    float tps_bend_FL = 0.f;
    float tps_bend_FC = 0.f;
    float tps_bend_FR = 0.f;

    double t_project_ms   = 0.0;
    double t_tps_ms       = 0.0;
    double t_warp_ms      = 0.0;
    double t_seam_ms      = 0.0;
    double t_composite_ms = 0.0;
    double t_total_ms     = 0.0;
};

class LidarTpsPipeline {
public:
    explicit LidarTpsPipeline(const LidarTpsConfig& cfg);
    ~LidarTpsPipeline();

    // Call once per log: 3-camera calibration (FL, FC, FR) + one LiDAR sweep
    // (for canvas bounds). Precomputes rotation remaps, allocates GPU buffers.
    void init(const std::array<Camera, 3>& cameras,
              const std::vector<std::array<float,3>>& lidar_pts);

    // Process one frame. images = FL,FC,FR (BGR); lidar_pts = ego-frame Nx3.
    // metrics: filled if non-null. eval_dump_dir: if set, dump rich
    // intermediates (warped PNGs, masks, seam bins, holdout CSV) for offline
    // metrics. Returns the composited panorama (BGR).
    cv::Mat process(const std::array<cv::Mat, 3>& images,
                    const std::vector<std::array<float,3>>& lidar_pts,
                    bool profile = false,
                    const std::string& debug_dir = "",
                    ProcessMetrics* metrics = nullptr,
                    const std::string& eval_dump_dir = "");

    // Block until any in-flight async canvas D2H (cfg.async_d2h) finishes.
    // No-op if off/none pending. Call before reading process()'s Mat if not
    // calling process() again first.
    void waitD2H();

    // -- Device-side accessors (for NVENC, to encode without a host round-trip)
    // GPU BGR canvas (W*H*3); valid until the next process() call.
    const uint8_t* canvasDevicePtr() const;
    // Main CUDA stream (composite + canvas D2H run here).
    cudaStream_t mainStream() const;

    // Canvas dimensions in pixels. Available after init() returns.
    int canvasWidth()  const;
    int canvasHeight() const;

private:
    // --- defined in lidar_tps_pipeline.cpp ---
    struct Impl;
    Impl* d_;
};

} // namespace lidartps
