// Minimal CLC using CUDA 13.0 PTX syntax
// Based on /usr/local/cuda-13.0/include/cccl/cuda/__ptx/instructions/generated/clusterlaunchcontrol.h

#include <stdio.h>
#include <cuda_runtime.h>
#include <stdint.h>

// Helper to get shared memory address
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

__global__ void clc_cuda13_kernel(float* data, int n, int* work_count) {
    __shared__ uint64_t clc_result[2];  // 16-byte (b128) response as 2x uint64_t
    __shared__ uint64_t barrier;

    int tid = threadIdx.x;
    int bx = blockIdx.x;

    // Initialize barrier
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

        // Submit CLC request (single thread)
        if (tid == 0) {
            uint32_t result_addr = smem_addr(&clc_result[0]);
            uint32_t bar_addr = smem_addr(&barrier);

            // Fence
            asm volatile("fence.proxy.async;" ::: "memory");

            // Full PTX syntax from CUDA 13.0 header:
            // clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128 [addr], [smem_bar];
            asm volatile(
                "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128 [%0], [%1];"
                :: "r"(result_addr), "r"(bar_addr)
                : "memory"
            );
        }

        // Do work
        int i = bx * blockDim.x + tid;
        if (i < n) {
            data[i] *= 2.5f;
        }

        // Wait for CLC response
        if (tid == 0) {
            uint32_t bar_addr = smem_addr(&barrier);
            uint32_t complete = 0;

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

        // Query result
        uint32_t success = 0;
        int new_bx = 0;

        if (tid == 0) {
            uint64_t result_lo = clc_result[0];
            uint64_t result_hi = clc_result[1];
            uint32_t is_canceled = 0;
            uint32_t ctaid_x = 0;

            // Full PTX syntax from CUDA 13.0:
            // clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 pred, b128_response;
            asm volatile(
                "{\n\t"
                ".reg .b128 B128_try_cancel_response;\n\t"
                "mov.b128 B128_try_cancel_response, {%1, %2};\n\t"
                "{\n\t"
                ".reg .pred P_OUT;\n\t"
                "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 P_OUT, B128_try_cancel_response;\n\t"
                "selp.b32 %0, 1, 0, P_OUT;\n\t"
                "}\n\t"
                "}"
                : "=r"(is_canceled)
                : "l"(result_lo), "l"(result_hi)
            );

            if (is_canceled) {
                // clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 ret, b128_response;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 B128_try_cancel_response;\n\t"
                    "mov.b128 B128_try_cancel_response, {%1, %2};\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, B128_try_cancel_response;\n\t"
                    "}"
                    : "=r"(ctaid_x)
                    : "l"(result_lo), "l"(result_hi)
                );
                new_bx = ctaid_x;
                success = 1;
            }

            asm volatile("fence.proxy.async;" ::: "memory");
        }

        // Broadcast
        __syncthreads();
        success = __shfl_sync(0xffffffff, success, 0);

        if (!success) {
            break;
        }

        new_bx = __shfl_sync(0xffffffff, new_bx, 0);
        bx = new_bx;
        phase ^= 1;

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
        printf("\nERROR: CLC requires CC 10.0+\n");
        return 1;
    }

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

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    printf("\nLaunching CLC kernel (CUDA 13.0 PTX syntax):\n");
    printf("  Grid: %d blocks\n", blocksPerGrid);
    printf("  Block: %d threads\n", threadsPerBlock);

    clc_cuda13_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, n, d_work_count);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel error: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        cudaFree(d_work_count);
        free(h_data);
        return 1;
    }

    cudaDeviceSynchronize();

    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);

    int h_work_count = 0;
    cudaMemcpy(&h_work_count, d_work_count, sizeof(int), cudaMemcpyDeviceToHost);

    bool success = true;
    for (int i = 0; i < 10; i++) {
        float expected = (i % 100) * 2.5f;
        if (h_data[i] != expected) {
            printf("Error at %d: got %f, expected %f\n", i, h_data[i], expected);
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
