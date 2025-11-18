// User's original kernel - to be linked with wrapper at runtime
#include <cuda_runtime.h>

extern "C" __device__ float compute_element(float a, float b) {
    return a * a + b * b + 2.0f * a * b;
}

// Device-callable implementation so policy kernels can reuse the logic
extern "C" __device__ void user_kernel_impl(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = compute_element(a[idx], b[idx]);
    }
}

// Public kernel entry point (still required for standalone launches)
extern "C" __global__ void user_kernel(const float* a, const float* b, float* c, int n) {
    user_kernel_impl(a, b, c, n);
}
