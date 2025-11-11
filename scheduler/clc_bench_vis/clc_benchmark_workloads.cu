// CLC Workload Analysis: Real-World AI Inference Scenarios
// Template-based approach with zero switch overhead
// Based on CUDA 12.9 libcu++ API

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda/ptx>
#include <cooperative_groups.h>
#include "ai_workloads.cuh"

namespace cg = cooperative_groups;
namespace ptx = cuda::ptx;

// ============================================
// Template-based Kernel Implementations
// Zero switch overhead - direct function calls
// ============================================

template<typename WorkloadType>
__global__ void kernel_fixed_work(float* data, int n, int* block_count, int prologue_complexity) {
    float weight = compute_prologue(prologue_complexity);

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        process_workload(WorkloadType{}, data, i, n, weight);
    }

    if (threadIdx.x == 0) {
        atomicAdd(block_count, 1);
    }
}

template<typename WorkloadType>
__global__ void kernel_fixed_blocks(float* data, int n, int* block_count, int prologue_complexity) {
    float weight = compute_prologue(prologue_complexity);

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = tid; i < n; i += stride) {
        process_workload(WorkloadType{}, data, i, n, weight);
    }

    if (threadIdx.x == 0) {
        atomicAdd(block_count, 1);
    }
}

template<typename WorkloadType>
__global__ void kernel_cluster_launch_control(float* data, int n, int* block_count, int* steal_count,
                                                int prologue_complexity) {
    __shared__ uint4 result;
    __shared__ uint64_t bar;
    int phase = 0;

    if (cg::thread_block::thread_rank() == 0)
        ptx::mbarrier_init(&bar, 1);

    float weight = compute_prologue(prologue_complexity);

    int bx = blockIdx.x;

    while (true) {
        __syncthreads();

        if (cg::thread_block::thread_rank() == 0) {
            ptx::fence_proxy_async_generic_sync_restrict(ptx::sem_acquire, ptx::space_cluster, ptx::scope_cluster);

            cg::invoke_one(cg::coalesced_threads(), [&](){
                ptx::clusterlaunchcontrol_try_cancel(&result, &bar);
            });
            ptx::mbarrier_arrive_expect_tx(ptx::sem_relaxed, ptx::scope_cta, ptx::space_shared, &bar, sizeof(uint4));
        }

        int i = bx * blockDim.x + threadIdx.x;
        if (i < n) {
            process_workload(WorkloadType{}, data, i, n, weight);
        }

        while (!ptx::mbarrier_try_wait_parity(ptx::sem_acquire, ptx::scope_cta, &bar, phase))
        {}
        phase ^= 1;

        bool success = ptx::clusterlaunchcontrol_query_cancel_is_canceled(result);
        if (!success)
            break;

        bx = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_x<int>(result);

        if (threadIdx.x == 0) {
            atomicAdd(steal_count, 1);
        }

        ptx::fence_proxy_async_generic_sync_restrict(ptx::sem_release, ptx::space_shared, ptx::scope_cluster);
    }

    if (threadIdx.x == 0) {
        atomicAdd(block_count, 1);
    }
}

// ============================================
// Benchmark Runner
// ============================================

struct BenchmarkResult {
    float avg_time_ms;
    float avg_blocks;
    float avg_steals;
};

template<typename WorkloadType>
BenchmarkResult run_fixed_work(float* d_data, int n, int blocks, int threads,
                                float* h_original, int prologue, int warmup, int runs) {
    int *d_block_count;
    cudaMalloc(&d_block_count, sizeof(int));

    // Warmup
    for (int i = 0; i < warmup; i++) {
        cudaMemcpy(d_data, h_original, n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_block_count, 0, sizeof(int));
        kernel_fixed_work<WorkloadType><<<blocks, threads>>>(d_data, n, d_block_count, prologue);
        cudaDeviceSynchronize();
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total_time = 0.0f;
    float total_blocks = 0.0f;

    for (int i = 0; i < runs; i++) {
        cudaMemcpy(d_data, h_original, n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_block_count, 0, sizeof(int));

        cudaEventRecord(start);
        kernel_fixed_work<WorkloadType><<<blocks, threads>>>(d_data, n, d_block_count, prologue);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        total_time += ms;

        int h_blocks;
        cudaMemcpy(&h_blocks, d_block_count, sizeof(int), cudaMemcpyDeviceToHost);
        total_blocks += h_blocks;
    }

    BenchmarkResult result;
    result.avg_time_ms = total_time / runs;
    result.avg_blocks = total_blocks / runs;
    result.avg_steals = 0;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_block_count);

    return result;
}

template<typename WorkloadType>
BenchmarkResult run_fixed_blocks(float* d_data, int n, int blocks, int threads,
                                  float* h_original, int prologue, int warmup, int runs) {
    int *d_block_count;
    cudaMalloc(&d_block_count, sizeof(int));

    // Warmup
    for (int i = 0; i < warmup; i++) {
        cudaMemcpy(d_data, h_original, n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_block_count, 0, sizeof(int));
        kernel_fixed_blocks<WorkloadType><<<blocks, threads>>>(d_data, n, d_block_count, prologue);
        cudaDeviceSynchronize();
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total_time = 0.0f;
    float total_blocks = 0.0f;

    for (int i = 0; i < runs; i++) {
        cudaMemcpy(d_data, h_original, n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_block_count, 0, sizeof(int));

        cudaEventRecord(start);
        kernel_fixed_blocks<WorkloadType><<<blocks, threads>>>(d_data, n, d_block_count, prologue);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        total_time += ms;

        int h_blocks;
        cudaMemcpy(&h_blocks, d_block_count, sizeof(int), cudaMemcpyDeviceToHost);
        total_blocks += h_blocks;
    }

    BenchmarkResult result;
    result.avg_time_ms = total_time / runs;
    result.avg_blocks = total_blocks / runs;
    result.avg_steals = 0;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_block_count);

    return result;
}

template<typename WorkloadType>
BenchmarkResult run_clc(float* d_data, int n, int blocks, int threads,
                        float* h_original, int prologue, int warmup, int runs) {
    int *d_block_count, *d_steal_count;
    cudaMalloc(&d_block_count, sizeof(int));
    cudaMalloc(&d_steal_count, sizeof(int));

    // Warmup
    for (int i = 0; i < warmup; i++) {
        cudaMemcpy(d_data, h_original, n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_block_count, 0, sizeof(int));
        cudaMemset(d_steal_count, 0, sizeof(int));
        kernel_cluster_launch_control<WorkloadType><<<blocks, threads>>>(d_data, n, d_block_count, d_steal_count, prologue);
        cudaDeviceSynchronize();
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total_time = 0.0f;
    float total_blocks = 0.0f;
    float total_steals = 0.0f;

    for (int i = 0; i < runs; i++) {
        cudaMemcpy(d_data, h_original, n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_block_count, 0, sizeof(int));
        cudaMemset(d_steal_count, 0, sizeof(int));

        cudaEventRecord(start);
        kernel_cluster_launch_control<WorkloadType><<<blocks, threads>>>(d_data, n, d_block_count, d_steal_count, prologue);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        total_time += ms;

        int h_blocks, h_steals;
        cudaMemcpy(&h_blocks, d_block_count, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_steals, d_steal_count, sizeof(int), cudaMemcpyDeviceToHost);
        total_blocks += h_blocks;
        total_steals += h_steals;
    }

    BenchmarkResult result;
    result.avg_time_ms = total_time / runs;
    result.avg_blocks = total_blocks / runs;
    result.avg_steals = total_steals / runs;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_block_count);
    cudaFree(d_steal_count);

    return result;
}

template<typename WorkloadType>
void run_scenario(const char* scenario, int prologue,
                  float* d_data, int n, int threads, cudaDeviceProp& prop, float* h_data) {
    int blocks_fixed_work = (n + threads - 1) / threads;
    int blocks_fixed_blocks = prop.multiProcessorCount * 2;
    int blocks_clc = (n + threads - 1) / threads;

    int warmup = 3;
    int runs = 10;

    BenchmarkResult r1 = run_fixed_work<WorkloadType>(d_data, n, blocks_fixed_work, threads, h_data, prologue, warmup, runs);
    BenchmarkResult r2 = run_fixed_blocks<WorkloadType>(d_data, n, blocks_fixed_blocks, threads, h_data, prologue, warmup, runs);
    BenchmarkResult r3 = run_clc<WorkloadType>(d_data, n, blocks_clc, threads, h_data, prologue, warmup, runs);

    float block_reduction = 100.0f * (blocks_clc - r3.avg_blocks) / blocks_clc;
    float speedup_vs_fixed_blocks = ((r2.avg_time_ms - r3.avg_time_ms) / r2.avg_time_ms) * 100.0f;
    float speedup_vs_fixed_work = ((r1.avg_time_ms - r3.avg_time_ms) / r1.avg_time_ms) * 100.0f;

    printf("%s,%d,%.3f,%.0f,%.3f,%.0f,%.3f,%.0f,%.0f,%.1f,%.1f,%.1f\n",
           scenario, prologue,
           r1.avg_time_ms, r1.avg_blocks,
           r2.avg_time_ms, r2.avg_blocks,
           r3.avg_time_ms, r3.avg_blocks, r3.avg_steals,
           block_reduction, speedup_vs_fixed_work, speedup_vs_fixed_blocks);
}

int main(int argc, char** argv) {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    if (prop.major < 10) {
        fprintf(stderr, "ERROR: CLC requires CC 10.0+\n");
        return 1;
    }

    int n = 1024 * 1024;
    int threads = 256;

    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) threads = atoi(argv[2]);

    float *h_data = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        h_data[i] = (float)(i % 100) + 1.0f;
    }

    float *d_data;
    cudaMalloc(&d_data, n * sizeof(float));

    printf("Scenario,Prologue,FixedWork_ms,FixedWork_blocks,FixedBlocks_ms,FixedBlocks_blocks,CLC_ms,CLC_blocks,CLC_steals,BlockReduction_pct,SpeedupVsFixedWork_pct,SpeedupVsFixedBlocks_pct\n");

    run_scenario<NLPVariableSequence>(get_workload_name<NLPVariableSequence>(), 80, d_data, n, threads, prop, h_data);
    run_scenario<DynamicBatching>(get_workload_name<DynamicBatching>(), 60, d_data, n, threads, prop, h_data);
    run_scenario<SparseAttention>(get_workload_name<SparseAttention>(), 70, d_data, n, threads, prop, h_data);
    run_scenario<GraphNeuralNetwork>(get_workload_name<GraphNeuralNetwork>(), 50, d_data, n, threads, prop, h_data);
    run_scenario<MixtureOfExperts>(get_workload_name<MixtureOfExperts>(), 75, d_data, n, threads, prop, h_data);
    run_scenario<VideoProcessing>(get_workload_name<VideoProcessing>(), 65, d_data, n, threads, prop, h_data);

    cudaFree(d_data);
    free(h_data);

    return 0;
}
