// policy.cu
// Policy device function - increment counter

#include <cuda_runtime.h>

// Policy function: increment counter
extern "C" __device__ void apply_policy(float *C, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        C[row * N + col] += 1.0f;
    }
}
