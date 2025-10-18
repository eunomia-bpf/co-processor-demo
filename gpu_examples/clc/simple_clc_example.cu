// Cluster Launch Control Example for SM 120 (Blackwell)
// Requires: CUDA Toolkit with Blackwell support, Compute Capability 10.0+

#include <stdio.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda/ptx>

namespace cg = cooperative_groups;
namespace ptx = cuda::ptx;

__device__ float compute_scalar() {
    return 2.5f;
}

// Cluster Launch Control kernel demonstrating work-stealing
__global__
void clc_vector_mul(float* data, int n)
{
    // Step 1: Declare CLC variables
    __shared__ uint4 result;      // Cancellation result
    __shared__ uint64_t bar;      // Memory barrier
    int phase = 0;                // Barrier phase

    // Step 2: Initialize barrier (single arrival count)
    if (cg::thread_block::thread_rank() == 0)
        ptx::mbarrier_init(&bar, 1);

    // Compute scalar once (prologue optimization)
    float alpha = compute_scalar();

    // Work-stealing loop
    int bx = blockIdx.x;

    while (true) {
        // Protect result from overwrite
        __syncthreads();

        // Step 3: Submit cancellation request (single thread)
        if (cg::thread_block::thread_rank() == 0) {
            // Fence for async proxy synchronization
            ptx::fence_proxy_async_generic_sync_restrict(
                ptx::sem_acquire, ptx::space_cluster, ptx::scope_cluster);

            // Request cancellation
            cg::invoke_one(cg::coalesced_threads(), [&]() {
                ptx::clusterlaunchcontrol_try_cancel(&result, &bar);
            });

            // Set transaction count
            ptx::mbarrier_arrive_expect_tx(
                ptx::sem_relaxed, ptx::scope_cta, ptx::space_shared,
                &bar, sizeof(uint4));
        }

        // Do actual work
        int i = bx * blockDim.x + threadIdx.x;
        if (i < n)
            data[i] *= alpha;

        // Step 4: Synchronize (wait for cancellation result)
        while (!ptx::mbarrier_try_wait_parity(
            ptx::sem_acquire, ptx::scope_cta, &bar, phase))
        {}
        phase ^= 1;

        // Step 5: Decode cancellation result
        bool success = ptx::clusterlaunchcontrol_query_cancel_is_canceled(result);

        if (!success) {
            // No more blocks to steal, exit
            break;
        }

        // Steal work from cancelled block
        bx = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_x<int>(result);

        // Release fence
        ptx::fence_proxy_async_generic_sync_restrict(
            ptx::sem_release, ptx::space_shared, ptx::scope_cluster);
    }
}

int main()
{
    // Check device capability
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);

    if (prop.major < 10) {
        printf("\nERROR: This example requires Compute Capability 10.0+\n");
        printf("Current device: %d.%d (Blackwell is 10.0)\n",
               prop.major, prop.minor);
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
    cudaMalloc(&d_data, size);
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    // Launch with CLC
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    printf("\nLaunching CLC kernel:\n");
    printf("  Grid: %d blocks\n", blocksPerGrid);
    printf("  Block: %d threads\n", threadsPerBlock);

    clc_vector_mul<<<blocksPerGrid, threadsPerBlock>>>(d_data, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel error: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        free(h_data);
        return 1;
    }

    cudaDeviceSynchronize();

    // Verify
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);

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

    cudaFree(d_data);
    free(h_data);
    return 0;
}
