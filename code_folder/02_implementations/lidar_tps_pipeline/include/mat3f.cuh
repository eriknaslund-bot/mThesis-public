#pragma once
#include <cuda_runtime.h>

// Row-major 3x3 float matrix usable on host and device.
struct Mat3f {
    float m[9];

    __host__ __device__ __forceinline__
    float3 operator*(float3 v) const {
        return make_float3(
            m[0]*v.x + m[1]*v.y + m[2]*v.z,
            m[3]*v.x + m[4]*v.y + m[5]*v.z,
            m[6]*v.x + m[7]*v.y + m[8]*v.z
        );
    }

    __host__ __device__ __forceinline__
    Mat3f T() const {
        Mat3f t;
        t.m[0]=m[0]; t.m[1]=m[3]; t.m[2]=m[6];
        t.m[3]=m[1]; t.m[4]=m[4]; t.m[5]=m[7];
        t.m[6]=m[2]; t.m[7]=m[5]; t.m[8]=m[8];
        return t;
    }
};
