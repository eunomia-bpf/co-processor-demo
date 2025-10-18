// Minimal Cluster Launch Control using inline PTX for CUDA 12.8
// Requires: SM 120 (Blackwell), Driver R570+
// This demonstrates the raw PTX instructions without libcu++ wrappers

#include <stdio.h>
#include <cuda_runtime.h>
#include <stdint.h>

// Helper to get shared memory address as uint32_t
__device__ __forceinline__ uint32_t smem_addr(void* ptr) {
    uint32_t addr;
    asm volatile("{\n\t"
                 ".reg .u64 u64addr;\n\t"
                 "cvta.to.shared.u64 u64addr, %1;\n\t"
                 "cvt.u32.u64 %0, u64addr;\n\t"
                 "}"
                 : "=r"(addr)
                 : "l"(ptr));
    return addr;
}

__global__ void clc_minimal_kernel(float* data, int n, int* work_count) {
    __shared__ uint32_t clc_result[4];  // 16-byte CLC response
    __shared__ uint64_t barrier;

    int tid = threadIdx.x;
    int bx = blockIdx.x;

    // Step 1: Initialize barrier (single arrival count)
    if (tid == 0) {
        uint32_t bar_addr = smem_addr(&barrier);
        asm volatile(
            "mbarrier.init.shared.b64 [%0], 1;"
            :: "r"(bar_addr)
        );
    }
    __syncthreads();

    // Work-stealing loop
    int phase = 0;

    while (true) {
        __syncthreads();

        // Step 2: Submit CLC request (single thread)
        if (tid == 0) {
            uint32_t result_addr = smem_addr(&clc_result[0]);
            uint32_t bar_addr = smem_addr(&barrier);

            // Fence for async proxy (PTX 8.7 syntax)
            asm volatile("fence.proxy.async;" ::: "memory");

            // Issue clusterlaunchcontrol.try_cancel
            // PTX syntax: clusterlaunchcontrol.try_cancel.b128 [result_addr];
            asm volatile(
                "clusterlaunchcontrol.try_cancel.b128 [%0];"
                :: "r"(result_addr)
            );

            // Set mbarrier expected transaction bytes (16 bytes)
            asm volatile(
                "mbarrier.arrive.expect_tx.shared.b64 _, [%0], 16;"
                :: "r"(bar_addr)
            );
        }

        // Step 3: Do actual work
        int i = bx * blockDim.x + tid;
        if (i < n) {
            data[i] *= 2.5f;
        }

        // Step 4: Wait for CLC response
        if (tid == 0) {
            uint32_t bar_addr = smem_addr(&barrier);
            uint32_t complete = 0;

            // mbarrier.try_wait.parity with acquire semantics
            while (!complete) {
                asm volatile(
                    "{\n\t"
                    ".reg .pred p;\n\t"
                    "mbarrier.try_wait.parity.acquire.cta.shared.b64 p, [%1], %2;\n\t"
                    "selp.u32 %0, 1, 0, p;\n\t"
                    "}"
                    : "=r"(complete)
                    : "r"(bar_addr), "r"(phase)
                );
            }
        }
        __syncthreads();

        // Step 5: Query result (broadcast to all threads)
        uint32_t success = 0;
        int new_bx = 0;

        if (tid == 0) {
            uint32_t result_addr = smem_addr(&clc_result[0]);
            uint32_t is_canceled = 0;
            uint32_t ctaid_x = 0;

            // Query if canceled
            asm volatile(
                "clusterlaunchcontrol.query_cancel.is_canceled.b32 %0, [%1];"
                : "=r"(is_canceled)
                : "r"(result_addr)
            );

            if (is_canceled) {
                // Query the canceled CTA ID (x-dimension)
                asm volatile(
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid.x.b32 %0, [%1];"
                    : "=r"(ctaid_x)
                    : "r"(result_addr)
                );
                new_bx = ctaid_x;
                success = 1;
            }

            // Release fence (PTX 8.7 syntax)
            asm volatile("fence.proxy.async;" ::: "memory");
        }

        // Broadcast success and new_bx to all threads
        __syncthreads();
        success = __shfl_sync(0xffffffff, success, 0);

        if (!success) {
            // No more work to steal
            break;
        }

        // Get new block index
        new_bx = __shfl_sync(0xffffffff, new_bx, 0);
        bx = new_bx;
        phase ^= 1;  // Flip phase

        // Count work items (for verification)
        if (tid == 0) {
            atomicAdd(work_count, 1);
        }
    }
}

int main() {
    // Check device
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);

    if (prop.major < 10) {
        printf("\nERROR: CLC requires Compute Capability 10.0+ (Blackwell)\n");
        printf("Current device: %d.%d\n", prop.major, prop.minor);
        return 1;
    }

    // Setup
    int n = 1024 * 1024;
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

    // Launch
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    printf("\nLaunching CLC kernel (inline PTX):\n");
    printf("  Grid: %d blocks\n", blocksPerGrid);
    printf("  Block: %d threads\n", threadsPerBlock);

    clc_minimal_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, n, d_work_count);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel error: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        cudaFree(d_work_count);
        free(h_data);
        return 1;
    }

    cudaDeviceSynchronize();

    // Verify
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);

    int h_work_count = 0;
    cudaMemcpy(&h_work_count, d_work_count, sizeof(int), cudaMemcpyDeviceToHost);

    bool success = true;
    for (int i = 0; i < 10; i++) {
        float expected = (i % 100) * 2.5f;
        if (h_data[i] != expected) {
            printf("Error at %d: got %f, expected %f\n",
                   i, h_data[i], expected);
            success = false;
        }
    }

    printf("\nResult: %s\n", success ? "PASSED" : "FAILED");
    printf("Work items stolen: %d\n", h_work_count);

    cudaFree(d_data);
    cudaFree(d_work_count);
    free(h_data);
    return 0;
}
