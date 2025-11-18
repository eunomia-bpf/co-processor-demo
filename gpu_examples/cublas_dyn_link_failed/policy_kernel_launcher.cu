// Standalone policy kernel that can be launched from CUPTI
// This applies the upper triangle zero policy to a matrix

#include <cuda_runtime.h>

// Policy kernel that zeros upper triangle
extern "C"
__global__ void apply_policy_upper_triangle_zero(float* C, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    // Zero upper triangle (col > row)
    if (col > row) {
        C[row * N + col] = 0.0f;
    }
}
