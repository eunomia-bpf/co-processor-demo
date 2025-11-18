// wrapper_kernel.cu
// Generic wrapper kernel that calls user kernel + policy (device functions)

#include <cuda_runtime.h>

// External device functions (defined in other compilation units)
extern "C" __device__ void gemm_kernel(float*, float*, float*, int, int, int, float, float);
extern "C" __device__ void policy_upper_triangle_zero(float*, int, int);

// Wrapper kernel: calls user kernel + policy directly
extern "C"
__global__ void run_with_policy_kernel(float *A, float *B, float *C,
                                       int M, int N, int K,
                                       float alpha, float beta) {
    // Call user kernel (each thread executes its part)
    gemm_kernel(A, B, C, M, N, K, alpha, beta);

    // Sync threads (ensure all threads complete computation before policy)
    __syncthreads();

    // Apply policy (each thread applies policy to its element)
    policy_upper_triangle_zero(C, M, N);
}
