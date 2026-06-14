#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d -- %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(_e)); \
        std::exit(1); \
    } \
} while(0)
