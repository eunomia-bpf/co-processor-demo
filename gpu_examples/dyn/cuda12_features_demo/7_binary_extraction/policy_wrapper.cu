// policy_wrapper.cu
// Policy wrappers that call the extracted original kernels

#include <cuda_runtime.h>

// External references to extracted kernels
// These will be resolved by nvJitLink when linking with the extracted CUBIN
extern "C" __device__ void _Z9vectorAddPKfS0_Pfi(const float*, const float*, float*, int);
extern "C" __device__ void _Z11gemm_kernelPfS_S_iiiff(float*, float*, float*, int, int, int, float, float);

// Policy: Rate limiting
__device__ int g_max_blocks = 1024;

// Wrapper for vector add with policy
extern "C" __global__ void vectorAdd_with_policy(const float *a, const float *b,
                                                  float *c, int n) {
    // Policy: Rate limit by block ID
    if (blockIdx.x >= g_max_blocks) {
        return;  // Skip execution for blocks beyond limit
    }

    // Call the original extracted kernel
    _Z9vectorAddPKfS0_Pfi(a, b, c, n);

    // Policy: Mark that we executed
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("[POLICY] vectorAdd executed with rate limit: %d blocks\n", g_max_blocks);
    }
}

// Wrapper for GEMM with policy
extern "C" __global__ void gemm_with_policy(float *A, float *B, float *C,
                                            int M, int N, int K,
                                            float alpha, float beta) {
    // First call the original GEMM kernel
    _Z11gemm_kernelPfS_S_iiiff(A, B, C, M, N, K, alpha, beta);

    // Apply policy: Zero upper triangle
    __syncthreads();

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N && col > row) {
        C[row * N + col] = 0.0f;
    }

    // Policy: Log execution
    if (threadIdx.x == 0 && threadIdx.y == 0 &&
        blockIdx.x == 0 && blockIdx.y == 0) {
        printf("[POLICY] GEMM executed with upper-triangle-zero policy\n");
    }
}
