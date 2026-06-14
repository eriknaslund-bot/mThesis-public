#pragma once
//
// NvencWriter -- GPU BGR canvas -> NV12 (bgrToNv12) -> HEVC NVENC -> MP4 mux.
// Canvas never leaves the GPU (skips the ~36 MB D2H). HEVC not H.264 because
// consumer Ampere H.264 NVENC caps at 4096 wide; our canvas is 5238.
//
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <string>

namespace lidartps {

class NvencWriter {
public:
    NvencWriter();
    ~NvencWriter();

    // Open output. width/height rounded up to even (NVENC). fps = playback
    // rate (20 for AV2). y_only: fill UV with 128 (neutral chroma) — used by
    // the bitrate-weight calibration. Throws on failure.
    void open(const std::string& path, int width, int height, int fps,
              bool y_only = false);

    // Encode one frame: d_bgr (device W*H*3 BGR) -> bgrToNv12 on `stream`
    // (synced first) -> NVENC -> mux.
    void writeFrame(const uint8_t* d_bgr, cudaStream_t stream);

    // Flush encoder and finalise the file. Idempotent.
    void close();

    bool isOpen() const;

private:
    struct Impl;
    Impl* d_;
};

} // namespace lidartps
