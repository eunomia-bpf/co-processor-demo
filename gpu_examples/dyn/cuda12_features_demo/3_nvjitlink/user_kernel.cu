// User's original kernel - to be linked with wrapper at runtime
#include <cuda_runtime.h>

extern "C" __device__ float compute_element(float a, float b) {
    return a * a + b * b + 2.0f * a * b;
}

extern "C" __global__ void user_kernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = compute_element(a[idx], b[idx]);
    }
}
