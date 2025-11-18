// policy.cu
// Policy wrapper that links with user kernel at runtime

#include <cuda_runtime.h>

// External reference to user's kernel implementation
extern "C" __device__ void gemm_kernel_impl(float *A, float *B, float *C,
                                             int M, int N, int K,
                                             float alpha, float beta);

// Policy-wrapped kernel
extern "C" __global__ void gemm_with_policy(float *A, float *B, float *C,
                                            int M, int N, int K,
                                            float alpha, float beta) {
    // Call original user kernel
    gemm_kernel_impl(A, B, C, M, N, K, alpha, beta);

    // Apply policy: zero upper triangle
    __syncthreads();

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N && col > row) {
        C[row * N + col] = 0.0f;
    }
}
