#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda_runtime.h>
#include <cmath>

// Kernel types for different workload patterns
enum KernelType { COMPUTE, MEMORY, MIXED, GEMM };

// Compute-bound kernel: matrix operations
__global__ void compute_kernel(float *data, int size, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float value = data[idx];
        for (int i = 0; i < iterations; i++) {
            value = sqrtf(value * value + 1.0f);
            value = sinf(value) * cosf(value);
        }
        data[idx] = value;
    }
}

// Memory-bound kernel: strided memory access
__global__ void memory_kernel(float *input, float *output, int size, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int read_idx = (idx * stride) % size;
        output[idx] = input[read_idx] * 2.0f;
    }
}

// Mixed workload kernel
__global__ void mixed_kernel(float *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Memory phase
        float value = data[idx];

        // Compute phase
        for (int i = 0; i < 50; i++) {
            value = sqrtf(value + 1.0f);
        }

        // Write back
        data[idx] = value;
    }
}

// GEMM kernel: C = A * B (tiled matrix multiplication)
// For simplicity, we use square matrices of size sqrt(size) x sqrt(size)
__global__ void gemm_kernel(float *A, float *B, float *C, int N) {
    // Tile size
    const int TILE_SIZE = 16;
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles into shared memory
        if (row < N && (t * TILE_SIZE + threadIdx.x) < N)
            As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && (t * TILE_SIZE + threadIdx.y) < N)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial product
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write result
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// Launch appropriate kernel based on type
inline void launch_kernel(KernelType type, float *d_data, float *d_temp, int size,
                   dim3 grid, dim3 block, cudaStream_t stream, float *d_matrix_c = nullptr) {
    switch (type) {
        case COMPUTE:
            compute_kernel<<<grid, block, 0, stream>>>(d_data, size, 100);
            break;
        case MEMORY:
            memory_kernel<<<grid, block, 0, stream>>>(d_data, d_temp, size, 16);
            break;
        case MIXED:
            mixed_kernel<<<grid, block, 0, stream>>>(d_data, size);
            break;
        case GEMM:
            {
                // For GEMM, interpret size as N*N elements, so N = sqrt(size)
                int N = (int)sqrtf((float)size);
                const int TILE_SIZE = 16;
                dim3 gemm_grid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
                dim3 gemm_block(TILE_SIZE, TILE_SIZE);
                gemm_kernel<<<gemm_grid, gemm_block, 0, stream>>>(d_data, d_temp, d_matrix_c, N);
            }
            break;
    }
}

#endif // KERNELS_CUH
