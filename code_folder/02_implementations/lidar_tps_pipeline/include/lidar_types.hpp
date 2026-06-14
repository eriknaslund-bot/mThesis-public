#pragma once
#include <string>
#include <vector>

// -- Camera calibration --------------------------------------------------------

struct Intrinsics {
    float fx, fy, cx, cy;
    float k1 = 0, k2 = 0, k3 = 0;
};

struct Extrinsics {
    float R[9];   // cam->ego rotation, row-major  (R[r*3+c])
    float t[3];   // cam->ego translation (camera origin in ego frame)
};

struct Camera {
    std::string name;
    Intrinsics  K;
    Extrinsics  E;
    int         width  = 0;
    int         height = 0;
};

// -- Canvas geometry -----------------------------------------------------------

struct CanvasGeometry {
    int   W = 0, H = 0;
    float fx_canvas = 0;   // px/rad azimuth (from FC fx)
    float fy_canvas = 0;   // px/tan(rad) elevation (from FC fy)
    float az_min = 0, az_max = 0;
    float el_min = 0, el_max = 0;   // radians
    float cx_canvas = 0;   // fx_canvas * az_max
    float cy_canvas = 0;   // fy_canvas * tan(el_max)
};

// -- Pipeline config -----------------------------------------------------------

struct LidarTpsConfig {
    float lidar_min_ctrl_range_m = 6.0f;  // drop ctrl pts closer than this (calms road-shear)
    int   max_ctrl_per_overlap   = 50;    // ctrl-pt budget/overlap; quality saturates ~50 (sec.4)
    float tps_smoothing          = 0.0f;  // TPS regularisation λ (0 = exact interpolation)
    float remap_scale            = 0.25f; // TPS disp eval resolution (quarter-canvas, upsampled)
    // FC held at its rotation baseline; FL/FR carry the full TPS displacement.
    bool  gain_compensation      = true;  // per-channel colour gain match before seam DP
    float holdout_frac           = 0.f;   // eval: fraction of shared pts held out of solve (0 = off)

    // Temporal levers (video; defaults are no-ops, matching single-frame output).
    float disp_temporal_alpha    = 1.f;   // disp IIR: disp_t = α·new + (1−α)·prev  (1 = off)
    int   feather_half_px        = -1;    // seam feather half-width px (-1 = default 40)
    bool  skip_canvas_d2h        = false; // skip canvas D2H (NVENC-only output; process() returns empty Mat)
    bool  async_d2h              = false; // overlap canvas D2H with next frame's upload (streaming only)
};
