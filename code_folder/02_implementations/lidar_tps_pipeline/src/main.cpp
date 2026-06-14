// main.cpp -- lidartps binary entry point
//
// Usage:
//   lidartps --calib  argo2_data/extracted/calibration.json
//            --frames argo2_data/extracted/frames.json
//            --lidar  argo2_data/extracted/sensors/lidar/   (pre-extracted .bin files)
//            --frame  0
//            --out    output/stitched/
//            [--benchmark N]
//
// Pre-extraction of LiDAR feather files:
//   python3 utils/extract_lidar_bin.py

#include "lidar_tps_pipeline.hpp"
#include "nvenc_writer.hpp"

#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;
using namespace lidartps;

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

// Returns row-major 3x3 rotation matrix as array[9]
static std::array<float,9> quatToMat(float qw, float qx, float qy, float qz) {
    float n = std::sqrt(qw*qw + qx*qx + qy*qy + qz*qz);
    qw /= n; qx /= n; qy /= n; qz /= n;
    return {
        1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw),
        2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw),
        2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)
    };
}

// Load AV2 calibration.json -> 3 front cameras [FL, FC, FR]
static std::array<Camera, 3> loadCalib(const std::string& path) {
    std::ifstream f(path);
    json j = json::parse(f);

    static const char* names[3] = {
        "ring_front_left", "ring_front_center", "ring_front_right"
    };

    std::array<Camera, 3> cams;
    for (int i = 0; i < 3; i++) {
        const char* name = names[i];
        if (!j.contains(name))
            throw std::runtime_error(std::string("calibration missing ") + name);
        const auto& v = j[name];

        Camera& c = cams[i];
        c.name   = name;
        c.width  = v.at("width");
        c.height = v.at("height");
        c.K.fx   = v.at("fx");  c.K.fy = v.at("fy");
        c.K.cx   = v.at("cx");  c.K.cy = v.at("cy");
        c.K.k1   = v.value("k1", 0.f);
        c.K.k2   = v.value("k2", 0.f);
        c.K.k3   = v.value("k3", 0.f);
        auto Rm  = quatToMat(v.at("qw"), v.at("qx"), v.at("qy"), v.at("qz"));
        std::copy(Rm.begin(), Rm.end(), c.E.R);
        c.E.t[0] = (float)v.at("tx_m");
        c.E.t[1] = (float)v.at("ty_m");
        c.E.t[2] = (float)v.at("tz_m");
    }
    return cams;
}

// Read a pre-extracted LiDAR binary file: raw float32, N*3 (x,y,z).
static std::vector<std::array<float,3>> readLidarBin(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("Cannot open lidar bin: " + path);
    size_t sz = f.tellg();
    size_t N  = sz / (3 * sizeof(float));
    f.seekg(0);
    std::vector<std::array<float,3>> pts(N);
    f.read(reinterpret_cast<char*>(pts.data()), sz);
    return pts;
}

// Index LiDAR binary files in a directory by timestamp (filename stem).
static std::unordered_map<long long, std::string> indexLidarDir(const std::string& dir) {
    std::unordered_map<long long, std::string> idx;
    for (const auto& e : fs::directory_iterator(dir)) {
        if (e.path().extension() != ".bin") continue;
        try {
            long long ts = std::stoll(e.path().stem().string());
            idx[ts] = e.path().string();
        } catch (...) {}
    }
    if (idx.empty())
        throw std::runtime_error("No .bin files found in: " + dir);
    return idx;
}

// Find nearest timestamp in index to the given reference timestamp.
static std::string nearestLidar(
    const std::unordered_map<long long, std::string>& idx,
    long long ref_ts)
{
    long long best_ts = -1;
    long long best_dt = std::numeric_limits<long long>::max();
    for (const auto& [ts, path] : idx) {
        long long dt = std::abs(ts - ref_ts);
        if (dt < best_dt) { best_dt = dt; best_ts = ts; }
    }
    return idx.at(best_ts);
}

// -----------------------------------------------------------------------------
// main
// -----------------------------------------------------------------------------

int main(int argc, char** argv) {
    // -- CLI parsing -----------------------------------------------------------
    std::string calib_path  = "argo2_data/extracted/calibration.json";
    std::string frames_path = "argo2_data/extracted/frames.json";
    std::string lidar_dir   = "argo2_data/extracted/sensors/lidar";
    std::string out_dir     = "code_folder/02_implementations/lidar_tps_pipeline/output/stitched";
    int  frame_idx   = 0;
    int  benchmark_N = 0;
    bool profile       = false;
    bool async_d2h     = false;  // skip trailing canvas-D2H sync; wait at next process()
    std::string video_path;       // --video PATH : write all rendered frames to MP4 (H.264, 20 fps)
    std::string nvenc_path;       // --video-nvenc PATH : NVENC HEVC, on-GPU canvas -> no D2H bottleneck
    bool        nvenc_y_only = false;  // --video-nvenc-y-only : flatten chroma to 128 (bitrate-weight calibration)
    int  num_frames    = 0;       // --num-frames N : limit video render to first N frames (0 = all)
    int  dump_stride   = 0;       // --dump-stride N : walk frames sequentially (like --video) and
                                  //                   write <out>/samples/frame_<fi>.jpg every N frames.
                                  //                   Honours temporal IIR. Compatible with or without --video.
    float disp_temporal_alpha = 1.f;// --disp-alpha A : TPS disp IIR (1 = off)
    bool  no_gain       = false;
    bool  do_debug      = false;
    bool  remap_scale_set = false;
    float remap_scale    = 0.25f;
    bool  tps_smooth_set = false;  float tps_smooth = 0.f;
    bool  min_ctrl_range_set = false;  float min_ctrl_range = 0.f;  // --min-ctrl-range M (overrides cfg default)
    int   feather_half_px       = -1;       // --feather-half PX : composite feather half-width;
                                            //   -1 keeps the deployed default (20, sec.3.6.4)
    int   max_ctrl_per_overlap  = -1;       // -1 = use header default (50)
    // Evaluation-harness flags
    bool  metrics_mode  = false;      // --metrics : emit one CSV row per frame
    std::string eval_dump_dir;        // --eval-dump <dir>
    float holdout_frac  = 0.f;        // --holdout-frac F (0..1)
    std::string config_tag;           // --config-tag <name> : extra CSV column for sweep

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i+1 < argc) ? argv[++i] : ""; };
        if      (a == "--calib")     calib_path  = next();
        else if (a == "--frames")    frames_path = next();
        else if (a == "--lidar")     lidar_dir   = next();
        else if (a == "--out")       out_dir     = next();
        else if (a == "--frame")     frame_idx   = std::stoi(next());
        else if (a == "--benchmark") benchmark_N = std::stoi(next());
        else if (a == "--profile")       profile       = true;
        else if (a == "--async-d2h")     async_d2h     = true;
        else if (a == "--video")         video_path    = next();
        else if (a == "--video-nvenc")   nvenc_path    = next();
        else if (a == "--video-nvenc-y-only") nvenc_y_only = true;
        else if (a == "--num-frames")    num_frames    = std::stoi(next());
        else if (a == "--dump-stride")   dump_stride   = std::stoi(next());
        else if (a == "--disp-alpha")    disp_temporal_alpha = std::stof(next());
        else if (a == "--no-gain")       no_gain       = true;
        else if (a == "--debug")         do_debug      = true;
        else if (a == "--remap-scale") { remap_scale_set = true; remap_scale = std::stof(next()); }
        else if (a == "--tps-smooth")  { tps_smooth_set = true; tps_smooth = std::stof(next()); }
        else if (a == "--min-ctrl-range") { min_ctrl_range_set = true; min_ctrl_range = std::stof(next()); }
        else if (a == "--feather-half")         feather_half_px = std::stoi(next());
        else if (a == "--video-preset") {
            // Video bundle: --tps-smooth 1.0 --disp-alpha 0.3 (later flags override).
            tps_smooth_set      = true;  tps_smooth          = 1.0f;
            disp_temporal_alpha = 0.3f;
        }
        else if (a == "--max-ctrl-per-overlap") max_ctrl_per_overlap = std::stoi(next());
        else if (a == "--metrics")       metrics_mode  = true;
        else if (a == "--eval-dump")     eval_dump_dir = next();
        else if (a == "--holdout-frac")  holdout_frac  = std::stof(next());
        else if (a == "--config-tag")    config_tag    = next();
        else { std::cerr << "Unknown flag: " << a << "\n"; return 1; }
    }

    // -- Load calibration ------------------------------------------------------
    std::cout << "Loading calibration: " << calib_path << "\n";
    auto cameras = loadCalib(calib_path);
    for (const auto& c : cameras)
        std::cout << "  " << c.name << ": " << c.width << "x" << c.height
                  << "  fx=" << c.K.fx << "\n";

    LidarTpsConfig cfg;
    cfg.async_d2h            = async_d2h;
    // Suppress the canvas D2H entirely when NVENC is the only output and no
    // host-side debug/eval reads h_canvas. Saves ~5 ms / frame on warm GPU.
    cfg.skip_canvas_d2h      = !nvenc_path.empty()
                                && video_path.empty()
                                && !do_debug
                                && !metrics_mode
                                && eval_dump_dir.empty()
                                && dump_stride <= 0;
    cfg.disp_temporal_alpha  = disp_temporal_alpha;
    cfg.gain_compensation    = !no_gain;
    if (remap_scale_set) cfg.remap_scale = remap_scale;
    cfg.holdout_frac         = holdout_frac;
    if (tps_smooth_set) cfg.tps_smoothing = tps_smooth;
    if (min_ctrl_range_set) cfg.lidar_min_ctrl_range_m = min_ctrl_range;
    if (max_ctrl_per_overlap > 0) cfg.max_ctrl_per_overlap = max_ctrl_per_overlap;
    cfg.feather_half_px      = feather_half_px;
    // In metrics mode, structured CSV rows are prefixed with "METRIC," so the
    // Python driver can grep them out of mixed stdout. Header emitted once.
    if (metrics_mode) {
        std::cout << "METRIC,config,frame,"
                     "n_shared_FL_FC,n_shared_FC_FR,"
                     "seam_l1_FL_FC,seam_l1_FC_FR,"
                     "seam_std_FL_FC,seam_std_FC_FR,"
                     "overlap_n_FL_FC,overlap_n_FC_FR,"
                     "overlap_mse_y_FL_FC,overlap_mse_y_FC_FR,"
                     "overlap_mse_cb_FL_FC,overlap_mse_cb_FC_FR,"
                     "overlap_mse_cr_FL_FC,overlap_mse_cr_FC_FR,"
                     "overlap_psnr_y_FL_FC,overlap_psnr_y_FC_FR,"
                     "overlap_psnr_cb_FL_FC,overlap_psnr_cb_FC_FR,"
                     "overlap_psnr_cr_FL_FC,overlap_psnr_cr_FC_FR,"
                     "warp_mean_FL,warp_p95_FL,warp_max_FL,"
                     "warp_mean_FC,warp_p95_FC,warp_max_FC,"
                     "warp_mean_FR,warp_p95_FR,warp_max_FR,"
                     "tps_bend_FL,tps_bend_FC,tps_bend_FR,"
                     "t_project,t_tps,t_warp,t_seam,t_composite,t_total\n";
    }

    // -- Load frames index + LiDAR index --------------------------------------
    std::ifstream ff(frames_path);
    json frames_j = json::parse(ff);

    if (frame_idx >= (int)frames_j.size())
        throw std::runtime_error("Frame index out of range");

    std::cout << "Indexing LiDAR binaries in: " << lidar_dir << "\n";
    auto lidar_idx = indexLidarDir(lidar_dir);
    std::cout << "  Found " << lidar_idx.size() << " sweeps\n";

    // Load the target frame's LiDAR sweep to derive canvas bounds from point visibility
    std::string fc_path0 = frames_j[frame_idx].at("ring_front_center");
    long long ts0 = std::stoll(fs::path(fc_path0).stem().string());
    auto init_lidar = readLidarBin(nearestLidar(lidar_idx, ts0));
    std::cout << "  Init LiDAR pts: " << init_lidar.size() << "\n";

    LidarTpsPipeline pipe(cfg);
    pipe.init(cameras, init_lidar);

    // Create output directory
    fs::create_directories(out_dir);

    // -- Video writer (--video / --video-nvenc) -------------------------------
    //   --video       cv::VideoWriter (CPU H.264)
    //   --video-nvenc NvencWriter (GPU HEVC, no canvas D2H)
    // Opened lazily on the first frame (canvas dims known then); 20 fps (AV2).
    cv::VideoWriter video_writer;
    NvencWriter     nvenc_writer;
    const bool do_video       = !video_path.empty();
    const bool do_video_nvenc = !nvenc_path.empty();
    const bool do_dump_stride = dump_stride > 0;
    if (do_video || do_video_nvenc) {
        const auto& vp = do_video_nvenc ? nvenc_path : video_path;
        fs::create_directories(fs::path(vp).parent_path());
    }
    if (do_video && do_video_nvenc)
        throw std::runtime_error("--video and --video-nvenc are mutually exclusive");

    // -- Frame loop ------------------------------------------------------------
    // --video mode iterates over all frames in frames.json (or the first
    // --num-frames N). --benchmark wins over --video if both are set.
    int n_frames;
    if (benchmark_N > 0)                                         n_frames = benchmark_N;
    else if (do_video || do_video_nvenc || do_dump_stride)       n_frames = (num_frames > 0)
                                    ? std::min(num_frames, (int)frames_j.size())
                                    : (int)frames_j.size();
    else                                                         n_frames = 1;
    std::vector<double> times_ms;
    times_ms.reserve(n_frames);
    // Per-stage timings collected when benchmark is on (m_ptr is non-null).
    // Used to compute mean / p50 / p95 / p99 in the trailing report.
    std::vector<double> t_project_ms, t_tps_ms, t_warp_ms, t_seam_ms,
                        t_composite_ms, t_total_ms;
    if (benchmark_N > 0) {
        t_project_ms.reserve(n_frames);   t_tps_ms.reserve(n_frames);
        t_warp_ms.reserve(n_frames);      t_seam_ms.reserve(n_frames);
        t_composite_ms.reserve(n_frames); t_total_ms.reserve(n_frames);
    }

    static const char* cam_keys[3] = {
        "ring_front_left", "ring_front_center", "ring_front_right"
    };

    // Map loop counter -> actual frames.json index. Behaves per the three
    // modes documented at n_frames assignment above.
    auto resolve_fi = [&](int fi) -> int {
        if (benchmark_N > 0)                                  return fi % (int)frames_j.size();
        if (do_video || do_video_nvenc || do_dump_stride)     return fi;
        return frame_idx;
    };

    // Per-frame input bundle produced by the prefetcher.
    struct LoadedFrame {
        std::array<cv::Mat, 3> images;
        std::vector<std::array<float, 3>> lidar_pts;
        std::string lidar_path;
        int fi_actual = 0;
    };

    // Synchronous loader: 3 imreads + 1 LiDAR bin read for frames.json[fi].
    // Fans the 4 CPU-bound disk reads out across 4 worker threads so its own
    // wall is ~30 ms (max of one cv::imread) rather than 100 ms serial.
    auto load_frame = [&](int fi_actual) -> LoadedFrame {
        LoadedFrame out;
        out.fi_actual = fi_actual;
        const auto& frame = frames_j[fi_actual];

        std::array<std::future<cv::Mat>, 3> img_futs;
        for (int i = 0; i < 3; i++) {
            std::string img_path = frame.at(cam_keys[i]);
            img_futs[i] = std::async(std::launch::async,
                [img_path]() { return cv::imread(img_path, cv::IMREAD_COLOR); });
        }

        std::string fc_path = frame.at("ring_front_center");
        long long img_ts = std::stoll(fs::path(fc_path).stem().string());
        out.lidar_path = nearestLidar(lidar_idx, img_ts);
        auto lidar_fut = std::async(std::launch::async,
            [path = out.lidar_path]() { return readLidarBin(path); });

        for (int i = 0; i < 3; i++) {
            out.images[i] = img_futs[i].get();
            if (out.images[i].empty())
                throw std::runtime_error("Cannot read: " + std::string(cam_keys[i]));
        }
        out.lidar_pts = lidar_fut.get();
        return out;
    };

    // Prefetch: load frame N+1 (imreads + bin) on a background thread while
    // process(N) is GPU-bound, so next_frame.get() is a no-op in the loop.
    std::future<LoadedFrame> next_frame = std::async(std::launch::async,
        [&]() { return load_frame(resolve_fi(0)); });

    for (int fi = 0; fi < n_frames; fi++) {
        LoadedFrame current = next_frame.get();
        int fi_actual = current.fi_actual;
        auto& images   = current.images;
        auto& lidar_pts = current.lidar_pts;

        // Launch prefetch for fi+1 BEFORE we touch the GPU -- this is the
        // overlap window. The lambda copies fi1 by value; everything else
        // is by reference (frames_j / lidar_idx / cam_keys are read-only).
        if (fi + 1 < n_frames) {
            int fi1 = fi + 1;
            next_frame = std::async(std::launch::async,
                [&, fi1]() { return load_frame(resolve_fi(fi1)); });
        }

        if (fi == 0) {
            for (int i = 0; i < 3; i++) {
                std::cout << "  " << cam_keys[i] << ": "
                          << images[i].cols << "x" << images[i].rows << "\n";
            }
            std::cout << "  LiDAR: " << current.lidar_path << "\n";
            std::cout << "  LiDAR pts: " << lidar_pts.size() << "\n";
        }

        // Process
        auto t0 = std::chrono::high_resolution_clock::now();
        bool do_profile = profile && (fi == 0);
        std::string frame_debug_dir;
        if (do_debug && benchmark_N == 0) {
            char buf[32]; std::snprintf(buf, sizeof(buf), "frame_%04d", fi_actual);
            frame_debug_dir = (fs::path(out_dir).parent_path()
                               / "lidar_ring_stitch" / "debug" / buf).string();
        }
        // Per-frame eval-dump dir is <eval_dump_dir>/frame_XXXX
        std::string frame_dump_dir;
        if (!eval_dump_dir.empty()) {
            char buf[32]; std::snprintf(buf, sizeof(buf), "frame_%04d", fi_actual);
            frame_dump_dir = (fs::path(eval_dump_dir) / buf).string();
        }
        ProcessMetrics m;
        // In --benchmark mode we always populate metrics internally so we can
        // report per-stage p95/p99 in the trailing summary; only --metrics
        // mode emits the CSV stream.
        ProcessMetrics* m_ptr = (metrics_mode || !frame_dump_dir.empty()
                                 || benchmark_N > 0) ? &m : nullptr;
        cv::Mat canvas = pipe.process(images, lidar_pts, do_profile,
                                      frame_debug_dir, m_ptr, frame_dump_dir);
        auto t1 = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        times_ms.push_back(ms);
        if (benchmark_N > 0) {
            t_project_ms.push_back(m.t_project_ms);
            t_tps_ms.push_back(m.t_tps_ms);
            t_warp_ms.push_back(m.t_warp_ms);
            t_seam_ms.push_back(m.t_seam_ms);
            t_composite_ms.push_back(m.t_composite_ms);
            t_total_ms.push_back(m.t_total_ms);
        }

        if (metrics_mode) {
            std::printf("METRIC,%s,%d,%d,%d,"
                "%.3f,%.3f,%.3f,%.3f,"
                "%d,%d,"
                "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,"
                "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,"
                "%.3f,%.3f,%.3f,"
                "%.3f,%.3f,%.3f,"
                "%.3f,%.3f,%.3f,"
                "%.6f,%.6f,%.6f,"
                "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
                config_tag.empty() ? "default" : config_tag.c_str(),
                fi_actual,
                m.n_shared_FL_FC, m.n_shared_FC_FR,
                m.seam_l1_FL_FC,  m.seam_l1_FC_FR,
                m.seam_std_FL_FC, m.seam_std_FC_FR,
                m.overlap_n_FL_FC,  m.overlap_n_FC_FR,
                m.overlap_mse_y_FL_FC,  m.overlap_mse_y_FC_FR,
                m.overlap_mse_cb_FL_FC, m.overlap_mse_cb_FC_FR,
                m.overlap_mse_cr_FL_FC, m.overlap_mse_cr_FC_FR,
                m.overlap_psnr_y_FL_FC,  m.overlap_psnr_y_FC_FR,
                m.overlap_psnr_cb_FL_FC, m.overlap_psnr_cb_FC_FR,
                m.overlap_psnr_cr_FL_FC, m.overlap_psnr_cr_FC_FR,
                m.warp_mean_FL, m.warp_p95_FL, m.warp_max_FL,
                m.warp_mean_FC, m.warp_p95_FC, m.warp_max_FC,
                m.warp_mean_FR, m.warp_p95_FR, m.warp_max_FR,
                m.tps_bend_FL, m.tps_bend_FC, m.tps_bend_FR,
                m.t_project_ms, m.t_tps_ms, m.t_warp_ms,
                m.t_seam_ms, m.t_composite_ms, m.t_total_ms);
            std::fflush(stdout);
        }

        if (do_video_nvenc) {
            // NVENC reads the canvas from device memory (writeFrame syncs
            // mainStream first). With skip_canvas_d2h the host canvas is empty,
            // so use the pipeline's reported dimensions.
            int W = pipe.canvasWidth();
            int H = pipe.canvasHeight();
            if (!nvenc_writer.isOpen()) {
                nvenc_writer.open(nvenc_path, W, H, 20, nvenc_y_only);
                std::cout << "Video output: " << nvenc_path << "  ("
                          << W << "x" << H
                          << ", HEVC NVENC, 20 fps)\n";
            }
            nvenc_writer.writeFrame(pipe.canvasDevicePtr(), pipe.mainStream());
            if ((fi & 31) == 0)
                std::cout << "  frame " << fi << "/" << n_frames
                          << "  (" << ms << " ms)\n";
        } else if (do_video) {
            // Async-D2H mode: drain the in-flight memcpy so the canvas bytes are
            // valid when VideoWriter reads them.
            if (async_d2h) pipe.waitD2H();
            if (!video_writer.isOpened()) {
                int fourcc = cv::VideoWriter::fourcc('a','v','c','1');  // H.264
                video_writer.open(video_path, fourcc, 20.0,
                                  cv::Size(canvas.cols, canvas.rows), /*isColor=*/true);
                if (!video_writer.isOpened())
                    throw std::runtime_error("cv::VideoWriter failed to open " + video_path);
                std::cout << "Video output: " << video_path << "  ("
                          << canvas.cols << "x" << canvas.rows << ", H.264, 20 fps)\n";
            }
            video_writer.write(canvas);
            if ((fi & 31) == 0)
                std::cout << "  frame " << fi << "/" << n_frames
                          << "  (" << ms << " ms)\n";
        } else if (!metrics_mode && !do_dump_stride && (benchmark_N == 0 || fi == 0)) {
            // Save output. In async-D2H mode the cv::Mat returned by process()
            // points at h_canvas while the device->host memcpy is still in
            // flight; drain it before cv::imwrite reads the bytes.
            if (async_d2h) pipe.waitD2H();
            std::string out_path = out_dir + "/frame_" +
                std::to_string(fi_actual) + ".jpg";
            cv::imwrite(out_path, canvas, {cv::IMWRITE_JPEG_QUALITY, 95});
            std::cout << "Saved: " << out_path << "  ("
                      << canvas.cols << "x" << canvas.rows << ")  "
                      << ms << " ms\n";
        }

        // --dump-stride: write a JPG every N frames into <out>/samples/
        // (every frame is still processed; only the dump cadence is decimated).
        if (do_dump_stride && (fi_actual % dump_stride) == 0) {
            if (async_d2h) pipe.waitD2H();
            std::string samples_dir = out_dir + "/samples";
            fs::create_directories(samples_dir);
            std::string out_path = samples_dir + "/frame_" +
                std::to_string(fi_actual) + ".jpg";
            cv::imwrite(out_path, canvas, {cv::IMWRITE_JPEG_QUALITY, 95});
            if ((fi & 31) == 0 || fi == n_frames - 1)
                std::cout << "  dumped frame " << fi_actual << " -> " << out_path << "\n";
        }
    }
    if (video_writer.isOpened()) {
        video_writer.release();
        std::cout << "Wrote " << n_frames << " frames to " << video_path << "\n";
    }
    if (nvenc_writer.isOpen()) {
        nvenc_writer.close();
        std::cout << "Wrote " << n_frames << " frames to " << nvenc_path << "\n";
    }

    // -- Benchmark report ------------------------------------------------------
    if (benchmark_N > 0 && !times_ms.empty()) {
        // Helper: copy + sort + compute mean / p50 / p95 / p99 of one vector.
        auto stats = [](std::vector<double> v) {
            std::sort(v.begin(), v.end());
            double mean = 0;
            for (double t : v) mean += t;
            mean /= v.size();
            double p50 = v[v.size() / 2];
            double p95 = v[std::min<size_t>(v.size() - 1, (size_t)(v.size() * 0.95))];
            double p99 = v[std::min<size_t>(v.size() - 1, (size_t)(v.size() * 0.99))];
            return std::array<double, 4>{mean, p50, p95, p99};
        };
        auto wall_s = stats(times_ms);
        std::cout << "\nBenchmark (" << benchmark_N << " frames):\n";
        std::cout << "  wall    mean=" << wall_s[0] << "  p50=" << wall_s[1]
                  << "  p95=" << wall_s[2] << "  p99=" << wall_s[3] << " ms\n";
        std::cout << "          ("
                  << 1000.0 / wall_s[0] << " /  " << 1000.0 / wall_s[1]
                  << " /  " << 1000.0 / wall_s[2]
                  << " /  " << 1000.0 / wall_s[3] << " FPS)\n";
        // Per-stage tails -- only meaningful when m_ptr was filled. The total
        // here is the pipeline-internal stream-synced total; it's slightly
        // higher than wall because the profile syncs cost a few hundred µs.
        if (!t_total_ms.empty() && t_total_ms.front() > 0.0) {
            auto pr = [&](const char* name, std::vector<double>& v) {
                auto s = stats(std::move(v));
                std::printf("  %-8s mean=%.2f  p50=%.2f  p95=%.2f  p99=%.2f ms\n",
                            name, s[0], s[1], s[2], s[3]);
            };
            std::cout << "Per-stage timings (synced):\n";
            pr("project", t_project_ms);
            pr("tps",     t_tps_ms);
            pr("warp",    t_warp_ms);
            pr("seam",    t_seam_ms);
            pr("compos",  t_composite_ms);
            pr("total",   t_total_ms);
        }
    }

    return 0;
}
