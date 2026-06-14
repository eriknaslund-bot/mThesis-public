cmake_minimum_required(VERSION 3.18)

# Disable feature detection for unknown compilers (avoids ABI issues with new GCC)
set(CMAKE_CXX_COMPILE_FEATURES "" CACHE STRING "" FORCE)
set(CMAKE_CUDA_COMPILE_FEATURES "" CACHE STRING "" FORCE)

# Default to Release if the user did not pass -DCMAKE_BUILD_TYPE.
# The pipeline is dominated by a TPS warp kernel whose hot loop is logf-heavy;
# without -O3 + --use_fast_math the warp stage runs ~25× slower (212 ms vs 8 ms).
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING
        "Build type: Release | Debug | RelWithDebInfo" FORCE)
endif()

project(lidar_tps_pipeline LANGUAGES CXX CUDA)

# Set C++ standard via flags (avoids feature detection issues with new GCC)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17 --use_fast_math -lineinfo")

# CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)

find_package(OpenCV REQUIRED)

# FFmpeg libs for the NVENC video writer (libavcodec already supports h264_nvenc
# / hevc_nvenc on Ubuntu — no extra Video Codec SDK headers needed).
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBAV REQUIRED IMPORTED_TARGET
    libavformat libavcodec libavutil)

# nlohmann/json (header-only; fetch if not installed system-wide)
find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
    include(FetchContent)
    FetchContent_Declare(json
        URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
        DOWNLOAD_EXTRACT_TIMESTAMP ON)
    FetchContent_MakeAvailable(json)
endif()

include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
)

# ── Sources ────────────────────────────────────────────────────────────────────
set(CUDA_SOURCES
    kernels/lidar_project.cu
    kernels/rotation_remap.cu
    kernels/tps_remap.cu
    kernels/seam_cost.cu
    kernels/composite.cu
    kernels/gain.cu
    kernels/bgr_to_nv12.cu
)

set(CPP_SOURCES
    src/lidar_tps_pipeline.cpp
    src/nvenc_writer.cpp
    src/main.cpp
)

add_executable(lidartps ${CPP_SOURCES} ${CUDA_SOURCES})

target_link_libraries(lidartps
    ${OpenCV_LIBS}
    nlohmann_json::nlohmann_json
    cudart
    PkgConfig::LIBAV
)

# Force GCC 15 libstdc++ path before the CUDA-injected GCC 13 path
target_link_options(lidartps PRIVATE -L/usr/lib/gcc/x86_64-linux-gnu/15)

set_target_properties(lidartps PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)
