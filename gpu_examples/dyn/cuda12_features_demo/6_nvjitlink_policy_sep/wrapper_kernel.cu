// wrapper_kernel.cu
// Generic wrapper kernel that can be extracted and linked with any user kernel

#include <cuda_runtime.h>

// Function pointer type for policy - takes thread index
typedef void (*policy_func_t)(int);

// Forward declaration of user kernel implementation (will be linked at runtime)
extern "C" __device__ void gemm_kernel_impl(float *A, float *B, float *C,
                                             int M, int N, int K,
                                             float alpha, float beta);

// Generic wrapper kernel that applies policy
extern "C" __global__ void gemm_kernel_with_policy(float *A, float *B, float *C,
                                                     int M, int N, int K,
                                                     float alpha, float beta,
                                                     policy_func_t policy_func) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x +
              (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x;

    // Call original user kernel
    gemm_kernel_impl(A, B, C, M, N, K, alpha, beta);

    // Synchronize before applying policy
    __syncthreads();

    // Apply policy if provided
    if (policy_func != nullptr) {
        policy_func(idx);
    }
}
