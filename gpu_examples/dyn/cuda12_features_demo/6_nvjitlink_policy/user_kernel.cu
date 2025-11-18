// user_kernel.cu
// User's GEMM kernel - will be linked with policy at runtime

#include <cuda_runtime.h>

// Device-callable implementation for linking
extern "C" __device__ void gemm_kernel_impl(float *A, float *B, float *C,
                                             int M, int N, int K,
                                             float alpha, float beta) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

// Standalone kernel for testing without policy
extern "C" __global__ void gemm_kernel(float *A, float *B, float *C,
                                       int M, int N, int K,
                                       float alpha, float beta) {
    gemm_kernel_impl(A, B, C, M, N, K, alpha, beta);
}
