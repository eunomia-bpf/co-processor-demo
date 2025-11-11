// CLC Policy Benchmark: Demonstrating and Comparing Scheduler Policies
// Based on the interface defined in clc_scheduler_policy.cuh

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda/ptx>
#include <cooperative_groups.h>
#include "ai_workloads.cuh"
#include "clc_scheduler_policy.cuh"
#include "benchmark_kernels.cuh"

namespace cg = cooperative_groups;
namespace ptx = cuda::ptx;
using namespace clc_policy;

// ============================================
// Policy-Aware CLC Kernel
// ============================================

template<typename WorkloadType, typename Policy>
__global__ void kernel_cluster_launch_control_policy(float* data, int n, int* block_count, int* steal_count,
                                                     int prologue_complexity) {
    __shared__ uint4 result;
    __shared__ uint64_t bar;
    int phase = 0;

    // Framework holds policy state in __shared__ memory
    __shared__ typename Policy::State policy_state;
    __shared__ int go;  // Broadcast flag for uniform control flow

    // Initialize the scheduler policy (thread 0 only, then sync)
    if (threadIdx.x == 0) {
        Policy::init(policy_state);
    }
    __syncthreads();

    if (cg::thread_block::thread_rank() == 0)
        ptx::mbarrier_init(&bar, 1);

    float weight = compute_prologue(prologue_complexity);
    int bx = blockIdx.x;

    while (true) {
        __syncthreads();

        // ELECT-AND-BROADCAST PATTERN: Thread 0 evaluates policy
        if (threadIdx.x == 0) {
            go = Policy::should_try_steal(policy_state) ? 1 : 0;
        }
        __syncthreads();

        // All threads read the same decision and take the same path
        if (!go) {
            break;  // Uniform exit - policy decided to stop
        }

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

        while (!ptx::mbarrier_try_wait_parity(ptx::sem_acquire, ptx::scope_cta, &bar, phase)) {}
        phase ^= 1;

        bool success = ptx::clusterlaunchcontrol_query_cancel_is_canceled(result);
        if (!success) {
            break;  // CLC failure - must exit immediately (no policy check)
        }

        bx = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_x<int>(result);

        if (threadIdx.x == 0) {
            atomicAdd(steal_count, 1);
        }

        ptx::fence_proxy_async_generic_sync_restrict(ptx::sem_release, ptx::space_shared, ptx::scope_cluster);

        // ELECT-AND-BROADCAST PATTERN: Thread 0 evaluates policy after success
        if (threadIdx.x == 0) {
            go = Policy::keep_going_after_success(bx, policy_state) ? 1 : 0;
        }
        __syncthreads();

        // All threads read the same decision and take the same path
        if (!go) {
            break;  // Uniform exit - policy decided to stop
        }
    }

    if (threadIdx.x == 0) {
        atomicAdd(block_count, 1);
    }
}

// ============================================
// Benchmark Runner
// ============================================

template<typename WorkloadType, typename Policy>
BenchmarkResult run_clc_policy(float* d_data, int n, int blocks, int threads,
                               float* h_original, int prologue, int warmup, int runs) {
    int *d_block_count, *d_steal_count;
    cudaMalloc(&d_block_count, sizeof(int));
    cudaMalloc(&d_steal_count, sizeof(int));

    // Warmup
    for (int i = 0; i < warmup; i++) {
        cudaMemcpy(d_data, h_original, n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_block_count, 0, sizeof(int));
        cudaMemset(d_steal_count, 0, sizeof(int));
        kernel_cluster_launch_control_policy<WorkloadType, Policy><<<blocks, threads>>>(d_data, n, d_block_count, d_steal_count, prologue);
        cudaDeviceSynchronize();
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total_time = 0.0f, total_blocks = 0.0f, total_steals = 0.0f;

    for (int i = 0; i < runs; i++) {
        cudaMemcpy(d_data, h_original, n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_block_count, 0, sizeof(int));
        cudaMemset(d_steal_count, 0, sizeof(int));

        cudaEventRecord(start);
        kernel_cluster_launch_control_policy<WorkloadType, Policy><<<blocks, threads>>>(d_data, n, d_block_count, d_steal_count, prologue);
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

    BenchmarkResult result = {total_time / runs, total_blocks / runs, total_steals / runs};
    
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

    printf("\n--- Scenario: %s (Prologue: %d) ---\n", scenario, prologue);
    printf("%-25s | %10s | %10s | %10s\n", "Scheduler", "Time (ms)", "Blocks", "Steals");
    printf("------------------------------------------------------------------\n");

    // Run Fixed Work
    BenchmarkResult r_fw = run_fixed_work<WorkloadType>(d_data, n, blocks_fixed_work, threads, h_data, prologue, warmup, runs);
    printf("%-25s | %10.3f | %10.0f | %10.0f\n", "FixedWork", r_fw.avg_time_ms, r_fw.avg_blocks, r_fw.avg_steals);

    // Run Fixed Blocks
    BenchmarkResult r_fb = run_fixed_blocks<WorkloadType>(d_data, n, blocks_fixed_blocks, threads, h_data, prologue, warmup, runs);
    printf("%-25s | %10.3f | %10.0f | %10.0f\n", "FixedBlocks", r_fb.avg_time_ms, r_fb.avg_blocks, r_fb.avg_steals);

    // Run CLC Baseline (no policy framework)
    BenchmarkResult r_clc_base = run_clc_baseline<WorkloadType>(d_data, n, blocks_clc, threads, h_data, prologue, warmup, runs);
    printf("%-25s | %10.3f | %10.0f | %10.0f\n", "CLC (Baseline)", r_clc_base.avg_time_ms, r_clc_base.avg_blocks, r_clc_base.avg_steals);

    // Run CLC with Greedy Policy
    BenchmarkResult r_clc_greedy = run_clc_policy<WorkloadType, GreedyPolicy>(d_data, n, blocks_clc, threads, h_data, prologue, warmup, runs);
    printf("%-25s | %10.3f | %10.0f | %10.0f\n", "CLC (Greedy Policy)", r_clc_greedy.avg_time_ms, r_clc_greedy.avg_blocks, r_clc_greedy.avg_steals);

    // Run CLC with MaxSteals Policy
    BenchmarkResult r_clc_maxsteals = run_clc_policy<WorkloadType, MaxStealsPolicy>(d_data, n, blocks_clc, threads, h_data, prologue, warmup, runs);
    printf("%-25s | %10.3f | %10.0f | %10.0f\n", "CLC (MaxSteals Policy)", r_clc_maxsteals.avg_time_ms, r_clc_maxsteals.avg_blocks, r_clc_maxsteals.avg_steals);

    // Run CLC with Voting Policy
    BenchmarkResult r_clc_voting = run_clc_policy<WorkloadType, VotingPolicy>(d_data, n, blocks_clc, threads, h_data, prologue, warmup, runs);
    printf("%-25s | %10.3f | %10.0f | %10.0f\n", "CLC (Voting Policy)", r_clc_voting.avg_time_ms, r_clc_voting.avg_blocks, r_clc_voting.avg_steals);

    printf("------------------------------------------------------------------\n");
    float framework_overhead = ((r_clc_greedy.avg_time_ms - r_clc_base.avg_time_ms) / r_clc_base.avg_time_ms) * 100.0f;
    float maxsteals_benefit = ((r_clc_base.avg_time_ms - r_clc_maxsteals.avg_time_ms) / r_clc_base.avg_time_ms) * 100.0f;
    printf("Policy Framework Overhead: %.2f%% (Greedy Policy vs. Baseline)\n", framework_overhead);
    printf("MaxSteals Policy Benefit: %.2f%% speedup vs. Baseline CLC.\n", maxsteals_benefit);
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
    for (int i = 0; i < n; i++) h_data[i] = (float)(i % 100) + 1.0f;

    float *d_data;
    cudaMalloc(&d_data, n * sizeof(float));

    printf("========================================================\n");
    printf("Comprehensive CLC Scheduler Benchmark\n");
    printf("Device: %s (CC %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("Elements: %d, Threads/Block: %d\n", n, threads);
    printf("========================================================\n");

    run_scenario<MixtureOfExperts>(get_workload_name<MixtureOfExperts>(), 75, d_data, n, threads, prop, h_data);
    run_scenario<NLPVariableSequence>(get_workload_name<NLPVariableSequence>(), 80, d_data, n, threads, prop, h_data);
    run_scenario<GraphNeuralNetwork>(get_workload_name<GraphNeuralNetwork>(), 50, d_data, n, threads, prop, h_data);

    cudaFree(d_data);
    free(h_data);

    return 0;
}
