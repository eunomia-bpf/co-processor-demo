// policy.cu
// Policy to inject into extracted kernels

#include <cuda_runtime.h>

// Policy: Rate limiting
// Only allow execution if block index is within limit
__device__ int g_max_blocks = 1024;

// Wrapper for vector add with policy
extern "C" __global__ void vectorAdd_with_policy(const float *a, const float *b,
                                                  float *c, int n) {
    // Policy: Rate limit by block ID
    if (blockIdx.x >= g_max_blocks) {
        return;  // Skip execution for blocks beyond limit
    }

    // Original kernel logic
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }

    // Policy: Mark that we executed
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("[POLICY] vectorAdd executed with rate limit: %d blocks\n", g_max_blocks);
    }
}

// Wrapper for GEMM with policy
extern "C" __global__ void gemm_with_policy(float *A, float *B, float *C,
                                            int M, int N, int K,
                                            float alpha, float beta) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        // Original GEMM computation
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];

        // Policy: Zero upper triangle
        if (col > row) {
            C[row * N + col] = 0.0f;
        }
    }

    // Policy: Log execution
    if (threadIdx.x == 0 && threadIdx.y == 0 &&
        blockIdx.x == 0 && blockIdx.y == 0) {
        printf("[POLICY] GEMM executed with upper-triangle-zero policy\n");
    }
}
