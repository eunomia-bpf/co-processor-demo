// Minimal CLC using CUDA 12.9 libcu++ API
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda/ptx>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;
namespace ptx = cuda::ptx;

__global__ void clc_kernel_12_9(float* data, int n, int* work_count) {
    // CLC response storage (16 bytes as uint4)
    __shared__ uint4 clc_response;
    __shared__ uint64_t barrier;

    int tid = threadIdx.x;
    int bx = blockIdx.x;

    // Initialize barrier
    if (tid == 0) {
        ptx::mbarrier_init(&barrier, 1);
    }
    __syncthreads();

    // Work-stealing loop
    while (true) {
        __syncthreads();

        // Submit CLC request (single thread)
        if (tid == 0) {
            // Use CUDA 12.9 API - pass pointer
            ptx::clusterlaunchcontrol_try_cancel(&clc_response, &barrier);
        }

        // Do work
        int i = bx * blockDim.x + tid;
        if (i < n) {
            data[i] *= 2.5f;
        }

        // Wait for response
        __syncthreads();

        // Query if canceled
        bool canceled = false;
        int new_bx = 0;

        if (tid == 0) {
            canceled = ptx::clusterlaunchcontrol_query_cancel_is_canceled(clc_response);

            if (canceled) {
                new_bx = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_x<int>(clc_response);
            }
        }

        // Broadcast to all threads
        __syncthreads();
        canceled = __shfl_sync(0xffffffff, canceled, 0);

        if (!canceled) {
            break;  // No more work
        }

        new_bx = __shfl_sync(0xffffffff, new_bx, 0);
        bx = new_bx;

        if (tid == 0) {
            atomicAdd(work_count, 1);
        }
    }
}

int main() {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);

    if (prop.major < 10) {
        printf("ERROR: CLC requires CC 10.0+\n");
        return 1;
    }

    int n = 1024;
    size_t size = n * sizeof(float);

    float *h_data = (float*)malloc(size);
    for (int i = 0; i < n; i++) {
        h_data[i] = i % 100;
    }

    float *d_data;
    int *d_work_count;
    cudaMalloc(&d_data, size);
    cudaMalloc(&d_work_count, sizeof(int));
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
    cudaMemset(d_work_count, 0, sizeof(int));

    int threadsPerBlock = 256;
    int blocksPerGrid = 4;  // Small number for testing

    printf("\nLaunching CLC kernel (CUDA 12.9 API):\n");
    printf("  Grid: %d blocks\n", blocksPerGrid);
    printf("  Block: %d threads\n\n", threadsPerBlock);

    clc_kernel_12_9<<<blocksPerGrid, threadsPerBlock>>>(d_data, n, d_work_count);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel error: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        cudaFree(d_work_count);
        free(h_data);
        return 1;
    }

    printf("Kernel completed!\n\n");

    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);

    int h_work_count = 0;
    cudaMemcpy(&h_work_count, d_work_count, sizeof(int), cudaMemcpyDeviceToHost);

    bool success = true;
    for (int i = 0; i < 10; i++) {
        float expected = (i % 100) * 2.5f;
        printf("[%d] %.1f (expected %.1f) %s\n",
               i, h_data[i], expected,
               (h_data[i] == expected) ? "✓" : "✗");
        if (h_data[i] != expected) success = false;
    }

    printf("\nWork stolen: %d times\n", h_work_count);
    printf("Result: %s\n", success ? "✅ PASSED" : "❌ FAILED");

    cudaFree(d_data);
    cudaFree(d_work_count);
    free(h_data);

    return 0;
}
