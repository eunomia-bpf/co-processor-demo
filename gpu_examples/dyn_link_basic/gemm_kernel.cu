// gemm_kernel.cu
// Contains only the user GEMM kernel as device function

#include <cuda_runtime.h>

// GEMM kernel as DEVICE function (not global) for inlining into wrapper
extern "C"
__device__ void gemm_kernel(float *A, float *B, float *C,
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

