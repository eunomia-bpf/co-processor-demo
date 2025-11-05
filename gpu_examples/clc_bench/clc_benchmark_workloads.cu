// Comprehensive Cluster Launch Control Benchmark with Configurable Workloads
// Tests different scenarios to find when CLC outperforms other approaches
// Based on CUDA 12.9 libcu++ API

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda/ptx>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;
namespace ptx = cuda::ptx;

// ============================================
// Workload Types
// ============================================

enum WorkloadType {
    LIGHT_COMPUTE,      // Minimal computation
    MEDIUM_COMPUTE,     // Moderate computation
    HEAVY_COMPUTE,      // Heavy computation
    VARIABLE_COMPUTE,   // Variable work per element (based on index)
    MEMORY_BOUND,       // Memory intensive
    DIVERGENT_COMPUTE   // Branch divergence
};

// Helper function to compute scalar (simulates prologue work)
__device__ float compute_scalar(int prologue_iters) {
    float alpha = 2.5f;
    for (int i = 0; i < prologue_iters; i++) {
        alpha = sqrtf(alpha * alpha);
    }
    return alpha;
}

// Configurable workload function
__device__ void do_workload(float* data, int idx, float alpha, WorkloadType type, int work_intensity) {
    if (idx >= 0) {  // Always true, prevents optimization
        switch (type) {
            case LIGHT_COMPUTE: {
                // Simple multiplication
                data[idx] *= alpha;
                break;
            }

            case MEDIUM_COMPUTE: {
                // Multiple operations
                for (int i = 0; i < work_intensity; i++) {
                    data[idx] = data[idx] * alpha + 1.0f;
                    data[idx] = sqrtf(data[idx] * data[idx]);
                }
                break;
            }

            case HEAVY_COMPUTE: {
                // Expensive computation
                for (int i = 0; i < work_intensity; i++) {
                    data[idx] = sinf(data[idx] * alpha);
                    data[idx] = cosf(data[idx]) + expf(data[idx] * 0.01f);
                }
                break;
            }

            case VARIABLE_COMPUTE: {
                // Variable work based on index (some threads do more work)
                int iters = (idx % 8 == 0) ? work_intensity * 4 : work_intensity;
                for (int i = 0; i < iters; i++) {
                    data[idx] = data[idx] * alpha + sinf((float)i);
                }
                break;
            }

            case MEMORY_BOUND: {
                // Memory intensive with indirect access
                float temp = data[idx];
                for (int i = 0; i < work_intensity; i++) {
                    int offset = (int)(temp) % 1000;
                    temp = temp * alpha + (float)offset * 0.001f;
                }
                data[idx] = temp;
                break;
            }

            case DIVERGENT_COMPUTE: {
                // Branch divergence
                if (idx % 2 == 0) {
                    for (int i = 0; i < work_intensity; i++) {
                        data[idx] = sinf(data[idx] * alpha);
                    }
                } else {
                    for (int i = 0; i < work_intensity / 2; i++) {
                        data[idx] = cosf(data[idx] * alpha);
                    }
                }
                break;
            }
        }
    }
}

// ============================================
// Approach 1: Fixed Work per Thread Block
// ============================================
__global__ void kernel_fixed_work(float* data, int n, int* block_count,
                                   WorkloadType type, int work_intensity, int prologue_iters) {
    // Prologue: computed by every thread block
    float alpha = compute_scalar(prologue_iters);

    // Each block processes fixed amount of work
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        do_workload(data, i, alpha, type, work_intensity);
    }

    // Track number of blocks that ran
    if (threadIdx.x == 0) {
        atomicAdd(block_count, 1);
    }
}

// ============================================
// Approach 2: Fixed Number of Thread Blocks
// ============================================
__global__ void kernel_fixed_blocks(float* data, int n, int* block_count,
                                     WorkloadType type, int work_intensity, int prologue_iters) {
    // Prologue: computed only by fixed number of blocks
    float alpha = compute_scalar(prologue_iters);

    // Grid-stride loop: variable work per block
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = tid; i < n; i += stride) {
        do_workload(data, i, alpha, type, work_intensity);
    }

    // Track number of blocks that ran
    if (threadIdx.x == 0) {
        atomicAdd(block_count, 1);
    }
}

// ============================================
// Approach 3: Cluster Launch Control
// ============================================
__global__ void kernel_cluster_launch_control(float* data, int n, int* block_count, int* steal_count,
                                                WorkloadType type, int work_intensity, int prologue_iters) {
    // Cluster launch control initialization
    __shared__ uint4 result;
    __shared__ uint64_t bar;
    int phase = 0;

    if (cg::thread_block::thread_rank() == 0)
        ptx::mbarrier_init(&bar, 1);

    // Prologue: computed by reduced number of blocks due to work-stealing
    float alpha = compute_scalar(prologue_iters);

    // Work-stealing loop
    int bx = blockIdx.x;

    while (true) {
        __syncthreads();

        // Cancellation request
        if (cg::thread_block::thread_rank() == 0) {
            ptx::fence_proxy_async_generic_sync_restrict(ptx::sem_acquire, ptx::space_cluster, ptx::scope_cluster);

            cg::invoke_one(cg::coalesced_threads(), [&](){
                ptx::clusterlaunchcontrol_try_cancel(&result, &bar);
            });
            ptx::mbarrier_arrive_expect_tx(ptx::sem_relaxed, ptx::scope_cta, ptx::space_shared, &bar, sizeof(uint4));
        }

        // Computation: process current block's work
        int i = bx * blockDim.x + threadIdx.x;
        if (i < n) {
            do_workload(data, i, alpha, type, work_intensity);
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

struct BenchmarkResult {
    float avg_time;
    float avg_blocks;
    float bandwidth;
    bool correct;
};

const char* workload_names[] = {
    "Light Compute",
    "Medium Compute",
    "Heavy Compute",
    "Variable Compute",
    "Memory Bound",
    "Divergent Compute"
};

BenchmarkResult run_benchmark_generic(const char* name,
                                       void (*kernel)(float*, int, int*, WorkloadType, int, int),
                                       float* d_data, int n, int blocks, int threads,
                                       float* h_original, WorkloadType type, int work_intensity,
                                       int prologue_iters, int warmup_runs, int bench_runs) {
    int *d_block_count;
    cudaMalloc(&d_block_count, sizeof(int));

    // Warmup
    for (int i = 0; i < warmup_runs; i++) {
        cudaMemcpy(d_data, h_original, n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_block_count, 0, sizeof(int));
        kernel<<<blocks, threads>>>(d_data, n, d_block_count, type, work_intensity, prologue_iters);
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
        kernel<<<blocks, threads>>>(d_data, n, d_block_count, type, work_intensity, prologue_iters);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        total_time += milliseconds;

        int h_block_count;
        cudaMemcpy(&h_block_count, d_block_count, sizeof(int), cudaMemcpyDeviceToHost);
        total_blocks += h_block_count;
    }

    // Verify correctness (simplified - just check no NaN/Inf)
    float* h_result = (float*)malloc(n * sizeof(float));
    cudaMemcpy(h_result, d_data, n * sizeof(float), cudaMemcpyDeviceToHost);

    bool correct = true;
    for (int i = 0; i < n && i < 100; i++) {
        if (isnan(h_result[i]) || isinf(h_result[i])) {
            correct = false;
            break;
        }
    }

    BenchmarkResult result;
    result.avg_time = total_time / bench_runs;
    result.avg_blocks = (float)total_blocks / bench_runs;
    result.bandwidth = (n * sizeof(float) * 2 / 1e9) / (result.avg_time / 1000.0f);
    result.correct = correct;

    free(h_result);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_block_count);

    return result;
}

BenchmarkResult run_clc_benchmark(float* d_data, int n, int blocks, int threads,
                                   float* h_original, WorkloadType type, int work_intensity,
                                   int prologue_iters, int warmup_runs, int bench_runs) {
    int *d_block_count, *d_steal_count;
    cudaMalloc(&d_block_count, sizeof(int));
    cudaMalloc(&d_steal_count, sizeof(int));

    // Warmup
    for (int i = 0; i < warmup_runs; i++) {
        cudaMemcpy(d_data, h_original, n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_block_count, 0, sizeof(int));
        cudaMemset(d_steal_count, 0, sizeof(int));
        kernel_cluster_launch_control<<<blocks, threads>>>(d_data, n, d_block_count, d_steal_count,
                                                             type, work_intensity, prologue_iters);
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
        kernel_cluster_launch_control<<<blocks, threads>>>(d_data, n, d_block_count, d_steal_count,
                                                             type, work_intensity, prologue_iters);
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
    for (int i = 0; i < n && i < 100; i++) {
        if (isnan(h_result[i]) || isinf(h_result[i])) {
            correct = false;
            break;
        }
    }

    BenchmarkResult result;
    result.avg_time = total_time / bench_runs;
    result.avg_blocks = (float)total_blocks / bench_runs;
    result.bandwidth = (n * sizeof(float) * 2 / 1e9) / (result.avg_time / 1000.0f);
    result.correct = correct;

    printf("  CLC Stats: blocks=%.0f, steals=%.0f (%.1f%% reduction)\n",
           result.avg_blocks, (float)total_steals / bench_runs,
           100.0f * (blocks - result.avg_blocks) / blocks);

    free(h_result);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_block_count);
    cudaFree(d_steal_count);

    return result;
}

void run_workload_test(float* d_data, int n, int threads, cudaDeviceProp& prop,
                       float* h_data, WorkloadType type, int work_intensity, int prologue_iters) {
    printf("\n======================================\n");
    printf("Workload: %s\n", workload_names[type]);
    printf("Work Intensity: %d\n", work_intensity);
    printf("Prologue Iterations: %d\n", prologue_iters);
    printf("======================================\n");

    int blocks_fixed_work = (n + threads - 1) / threads;
    int blocks_fixed_blocks = prop.multiProcessorCount * 2;
    int blocks_clc = (n + threads - 1) / threads;

    int warmup = 2;
    int runs = 5;

    BenchmarkResult r1 = run_benchmark_generic("Fixed Work", kernel_fixed_work,
                                                d_data, n, blocks_fixed_work, threads,
                                                h_data, type, work_intensity, prologue_iters, warmup, runs);
    printf("Fixed Work:    %.3f ms, %.0f blocks, %.2f GB/s %s\n",
           r1.avg_time, r1.avg_blocks, r1.bandwidth, r1.correct ? "✅" : "❌");

    BenchmarkResult r2 = run_benchmark_generic("Fixed Blocks", kernel_fixed_blocks,
                                                d_data, n, blocks_fixed_blocks, threads,
                                                h_data, type, work_intensity, prologue_iters, warmup, runs);
    printf("Fixed Blocks:  %.3f ms, %.0f blocks, %.2f GB/s %s\n",
           r2.avg_time, r2.avg_blocks, r2.bandwidth, r2.correct ? "✅" : "❌");

    BenchmarkResult r3 = run_clc_benchmark(d_data, n, blocks_clc, threads,
                                           h_data, type, work_intensity, prologue_iters, warmup, runs);
    printf("CLC:           %.3f ms, %.0f blocks, %.2f GB/s %s\n",
           r3.avg_time, r3.avg_blocks, r3.bandwidth, r3.correct ? "✅" : "❌");

    // Analysis
    printf("\nPerformance vs Fixed Blocks: ");
    if (r3.avg_time < r2.avg_time) {
        printf("CLC WINS by %.1f%%! 🎉\n", 100.0f * (r2.avg_time - r3.avg_time) / r2.avg_time);
    } else {
        printf("Fixed Blocks wins by %.1f%%\n", 100.0f * (r3.avg_time - r2.avg_time) / r2.avg_time);
    }
}

int main(int argc, char** argv) {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    printf("==============================================\n");
    printf("CLC Workload Analysis Benchmark\n");
    printf("==============================================\n");
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("SM Count: %d\n", prop.multiProcessorCount);

    if (prop.major < 10) {
        printf("\n❌ ERROR: CLC requires Compute Capability 10.0+\n");
        return 1;
    }

    int n = 1024 * 1024;  // 1M elements
    int threads = 256;

    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) threads = atoi(argv[2]);

    printf("\nArray size: %d elements (%.2f MB)\n", n, n * sizeof(float) / 1e6);
    printf("Threads per block: %d\n", threads);

    // Allocate memory
    float *h_data = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        h_data[i] = (float)(i % 100) + 1.0f;
    }

    float *d_data;
    cudaMalloc(&d_data, n * sizeof(float));

    printf("\n==============================================\n");
    printf("Testing Different Workload Scenarios\n");
    printf("==============================================\n");

    // Test 1: Light compute, low prologue overhead
    run_workload_test(d_data, n, threads, prop, h_data, LIGHT_COMPUTE, 1, 10);

    // Test 2: Light compute, high prologue overhead
    run_workload_test(d_data, n, threads, prop, h_data, LIGHT_COMPUTE, 1, 100);

    // Test 3: Medium compute, medium prologue
    run_workload_test(d_data, n, threads, prop, h_data, MEDIUM_COMPUTE, 10, 50);

    // Test 4: Heavy compute, low prologue (compute dominates)
    run_workload_test(d_data, n, threads, prop, h_data, HEAVY_COMPUTE, 20, 10);

    // Test 5: Heavy compute, high prologue (CLC should win)
    run_workload_test(d_data, n, threads, prop, h_data, HEAVY_COMPUTE, 20, 100);

    // Test 6: Variable compute (load imbalance)
    run_workload_test(d_data, n, threads, prop, h_data, VARIABLE_COMPUTE, 15, 50);

    // Test 7: Memory bound workload
    run_workload_test(d_data, n, threads, prop, h_data, MEMORY_BOUND, 10, 30);

    // Test 8: Divergent workload (branch divergence)
    run_workload_test(d_data, n, threads, prop, h_data, DIVERGENT_COMPUTE, 20, 50);

    printf("\n==============================================\n");
    printf("Summary\n");
    printf("==============================================\n");
    printf("CLC performs best when:\n");
    printf("1. High prologue overhead + moderate/heavy compute\n");
    printf("2. Variable work per thread (load imbalance)\n");
    printf("3. Ratio: (prologue_cost * num_blocks) / total_work is high\n");

    cudaFree(d_data);
    free(h_data);

    return 0;
}
