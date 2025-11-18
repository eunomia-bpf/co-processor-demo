// Simple kernel 2 for fatbin demo
#include <cuda_runtime.h>

extern "C" __global__ void vectorMul(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

extern "C" __global__ void vectorDiv(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = (b[idx] != 0.0f) ? (a[idx] / b[idx]) : 0.0f;
    }
}
