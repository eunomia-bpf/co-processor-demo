// Comprehensive Cluster Launch Control Benchmark
// Compares three approaches: Fixed Work, Fixed Blocks, and CLC
// Based on CUDA 12.9 libcu++ API

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda/ptx>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;
namespace ptx = cuda::ptx;

// Helper function to compute scalar (simulates prologue work)
__device__ float compute_scalar() {
    float alpha = 2.5f;
    // Simulate some computation overhead
    for (int i = 0; i < 10; i++) {
        alpha = sqrtf(alpha * alpha);
    }
    return alpha;
}

// ============================================
// Approach 1: Fixed Work per Thread Block
// ============================================
__global__ void kernel_fixed_work(float* data, int n, int* block_count) {
    // Prologue: computed by every thread block
    float alpha = compute_scalar();

    // Each block processes fixed amount of work
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] *= alpha;
    }

    // Track number of blocks that ran
    if (threadIdx.x == 0) {
        atomicAdd(block_count, 1);
    }
}

// ============================================
// Approach 2: Fixed Number of Thread Blocks
// ============================================
__global__ void kernel_fixed_blocks(float* data, int n, int* block_count) {
    // Prologue: computed only by fixed number of blocks
    float alpha = compute_scalar();

    // Grid-stride loop: variable work per block
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = tid; i < n; i += stride) {
        data[i] *= alpha;
    }

    // Track number of blocks that ran
    if (threadIdx.x == 0) {
        atomicAdd(block_count, 1);
    }
}

// ============================================
// Approach 3: Cluster Launch Control
// ============================================
__global__ void kernel_cluster_launch_control(float* data, int n, int* block_count, int* steal_count) {
    // Cluster launch control initialization
    __shared__ uint4 result;
    __shared__ uint64_t bar;
    int phase = 0;

    if (cg::thread_block::thread_rank() == 0)
        ptx::mbarrier_init(&bar, 1);

    // Prologue: computed by reduced number of blocks due to work-stealing
    float alpha = compute_scalar();

    // Work-stealing loop
    int bx = blockIdx.x; // Assuming 1D x-axis thread blocks

    while (true) {
        // Protect result from overwrite in the next iteration
        __syncthreads();

        // Cancellation request
        if (cg::thread_block::thread_rank() == 0) {
            // Acquire write of result in the async proxy
            ptx::fence_proxy_async_generic_sync_restrict(ptx::sem_acquire, ptx::space_cluster, ptx::scope_cluster);

            cg::invoke_one(cg::coalesced_threads(), [&](){
                ptx::clusterlaunchcontrol_try_cancel(&result, &bar);
            });
            ptx::mbarrier_arrive_expect_tx(ptx::sem_relaxed, ptx::scope_cta, ptx::space_shared, &bar, sizeof(uint4));
        }

        // Computation: process current block's work
        int i = bx * blockDim.x + threadIdx.x;
        if (i < n) {
            data[i] *= alpha;
        }

        // Cancellation request synchronization
        while (!ptx::mbarrier_try_wait_parity(ptx::sem_acquire, ptx::scope_cta, &bar, phase))
        {}
        phase ^= 1;

        // Cancellation request decoding
        bool success = ptx::clusterlaunchcontrol_query_cancel_is_canceled(result);
        if (!success)
            break;

        bx = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_x<int>(result);

        // Track successful work-stealing
        if (threadIdx.x == 0) {
            atomicAdd(steal_count, 1);
        }

        // Release read of result to the async proxy
        ptx::fence_proxy_async_generic_sync_restrict(ptx::sem_release, ptx::space_shared, ptx::scope_cluster);
    }

    // Track number of blocks that actually ran
    if (threadIdx.x == 0) {
        atomicAdd(block_count, 1);
    }
}

// ============================================
// Benchmark Runner
// ============================================

void run_benchmark(const char* name,
                   void (*kernel)(float*, int, int*),
                   float* d_data, int n, int blocks, int threads,
                   float* h_original, int warmup_runs, int bench_runs) {
    printf("\n=== %s ===\n", name);
    printf("Configuration: %d blocks x %d threads\n", blocks, threads);

    int *d_block_count;
    cudaMalloc(&d_block_count, sizeof(int));

    // Warmup
    for (int i = 0; i < warmup_runs; i++) {
        cudaMemcpy(d_data, h_original, n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_block_count, 0, sizeof(int));
        kernel<<<blocks, threads>>>(d_data, n, d_block_count);
        cudaDeviceSynchronize();
    }

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total_time = 0.0f;
    int total_blocks = 0;

    for (int i = 0; i < bench_runs; i++) {
        cudaMemcpy(d_data, h_original, n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_block_count, 0, sizeof(int));

        cudaEventRecord(start);
        kernel<<<blocks, threads>>>(d_data, n, d_block_count);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        total_time += milliseconds;

        int h_block_count;
        cudaMemcpy(&h_block_count, d_block_count, sizeof(int), cudaMemcpyDeviceToHost);
        total_blocks += h_block_count;
    }

    // Verify correctness
    float* h_result = (float*)malloc(n * sizeof(float));
    cudaMemcpy(h_result, d_data, n * sizeof(float), cudaMemcpyDeviceToHost);

    bool correct = true;
    // compute_scalar() returns 2.5f after sqrt operations, so expected = original * 2.5
    for (int i = 0; i < n && i < 10; i++) {
        float expected = h_original[i] * 2.5f;
        if (fabsf(h_result[i] - expected) > 1e-3) {
            correct = false;
            printf("  Mismatch at [%d]: got %.3f, expected %.3f\n", i, h_result[i], expected);
            break;
        }
    }

    float avg_time = total_time / bench_runs;
    float avg_blocks = (float)total_blocks / bench_runs;
    float bandwidth = (n * sizeof(float) * 2 / 1e9) / (avg_time / 1000.0f); // GB/s

    printf("Results:\n");
    printf("  Average time: %.3f ms\n", avg_time);
    printf("  Average blocks executed: %.1f\n", avg_blocks);
    printf("  Bandwidth: %.2f GB/s\n", bandwidth);
    printf("  Correctness: %s\n", correct ? "✅ PASSED" : "❌ FAILED");

    free(h_result);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_block_count);
}

void run_clc_benchmark(float* d_data, int n, int blocks, int threads,
                      float* h_original, int warmup_runs, int bench_runs) {
    printf("\n=== Cluster Launch Control ===\n");
    printf("Configuration: %d blocks x %d threads\n", blocks, threads);

    int *d_block_count, *d_steal_count;
    cudaMalloc(&d_block_count, sizeof(int));
    cudaMalloc(&d_steal_count, sizeof(int));

    // Warmup
    for (int i = 0; i < warmup_runs; i++) {
        cudaMemcpy(d_data, h_original, n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_block_count, 0, sizeof(int));
        cudaMemset(d_steal_count, 0, sizeof(int));
        kernel_cluster_launch_control<<<blocks, threads>>>(d_data, n, d_block_count, d_steal_count);
        cudaDeviceSynchronize();
    }

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total_time = 0.0f;
    int total_blocks = 0;
    int total_steals = 0;

    for (int i = 0; i < bench_runs; i++) {
        cudaMemcpy(d_data, h_original, n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_block_count, 0, sizeof(int));
        cudaMemset(d_steal_count, 0, sizeof(int));

        cudaEventRecord(start);
        kernel_cluster_launch_control<<<blocks, threads>>>(d_data, n, d_block_count, d_steal_count);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        total_time += milliseconds;

        int h_block_count, h_steal_count;
        cudaMemcpy(&h_block_count, d_block_count, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_steal_count, d_steal_count, sizeof(int), cudaMemcpyDeviceToHost);
        total_blocks += h_block_count;
        total_steals += h_steal_count;
    }

    // Verify correctness
    float* h_result = (float*)malloc(n * sizeof(float));
    cudaMemcpy(h_result, d_data, n * sizeof(float), cudaMemcpyDeviceToHost);

    bool correct = true;
    // compute_scalar() returns 2.5f after sqrt operations, so expected = original * 2.5
    for (int i = 0; i < n && i < 10; i++) {
        float expected = h_original[i] * 2.5f;
        if (fabsf(h_result[i] - expected) > 1e-3) {
            correct = false;
            printf("  Mismatch at [%d]: got %.3f, expected %.3f\n", i, h_result[i], expected);
            break;
        }
    }

    float avg_time = total_time / bench_runs;
    float avg_blocks = (float)total_blocks / bench_runs;
    float avg_steals = (float)total_steals / bench_runs;
    float bandwidth = (n * sizeof(float) * 2 / 1e9) / (avg_time / 1000.0f); // GB/s

    printf("Results:\n");
    printf("  Average time: %.3f ms\n", avg_time);
    printf("  Average blocks executed: %.1f\n", avg_blocks);
    printf("  Average work steals: %.1f\n", avg_steals);
    printf("  Bandwidth: %.2f GB/s\n", bandwidth);
    printf("  Correctness: %s\n", correct ? "✅ PASSED" : "❌ FAILED");

    free(h_result);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_block_count);
    cudaFree(d_steal_count);
}

int main(int argc, char** argv) {
    // Device info
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    printf("==============================================\n");
    printf("Cluster Launch Control Benchmark\n");
    printf("==============================================\n");
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("SM Count: %d\n", prop.multiProcessorCount);

    if (prop.major < 10) {
        printf("\n❌ ERROR: Cluster Launch Control requires Compute Capability 10.0+\n");
        return 1;
    }

    // Parse command line arguments
    int n = 1024 * 1024;  // Default: 1M elements
    int threads = 256;
    int warmup = 3;
    int runs = 10;

    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) threads = atoi(argv[2]);
    if (argc > 3) warmup = atoi(argv[3]);
    if (argc > 4) runs = atoi(argv[4]);

    printf("\nBenchmark Parameters:\n");
    printf("  Array size: %d elements (%.2f MB)\n", n, n * sizeof(float) / 1e6);
    printf("  Threads per block: %d\n", threads);
    printf("  Warmup runs: %d\n", warmup);
    printf("  Benchmark runs: %d\n", runs);

    // Allocate and initialize host data
    float *h_data = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        h_data[i] = (float)(i % 100);
    }

    // Allocate device data
    float *d_data;
    cudaMalloc(&d_data, n * sizeof(float));

    // Calculate blocks for each approach
    int blocks_fixed_work = (n + threads - 1) / threads;
    int blocks_fixed_blocks = prop.multiProcessorCount * 2; // 2x SM count for good occupancy
    int blocks_clc = (n + threads - 1) / threads;

    printf("\n==============================================\n");
    printf("Running Benchmarks\n");
    printf("==============================================\n");

    // Benchmark 1: Fixed Work per Thread Block
    run_benchmark("Fixed Work per Thread Block",
                  kernel_fixed_work,
                  d_data, n, blocks_fixed_work, threads,
                  h_data, warmup, runs);

    // Benchmark 2: Fixed Number of Thread Blocks
    run_benchmark("Fixed Number of Thread Blocks",
                  kernel_fixed_blocks,
                  d_data, n, blocks_fixed_blocks, threads,
                  h_data, warmup, runs);

    // Benchmark 3: Cluster Launch Control
    run_clc_benchmark(d_data, n, blocks_clc, threads,
                     h_data, warmup, runs);

    printf("\n==============================================\n");
    printf("Summary\n");
    printf("==============================================\n");
    printf("Fixed Work:     Balanced load, supports preemption\n");
    printf("Fixed Blocks:   Reduced overhead, fixed prologue cost\n");
    printf("CLC:            Best of both - reduced overhead + load balancing\n");
    printf("\n");

    // Cleanup
    cudaFree(d_data);
    free(h_data);

    return 0;
}
