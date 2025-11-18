// policy.cu
// Policy functions only - no user kernel, no wrapper kernel

#include <cuda_runtime.h>

// Policy as DEVICE function (thread-level policy, not a separate kernel)
extern "C"
__device__ void policy_upper_triangle_zero(float* C, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    // Zero upper triangle
    if (col > row) {
        C[row * N + col] = 0.0f;
    }
}

