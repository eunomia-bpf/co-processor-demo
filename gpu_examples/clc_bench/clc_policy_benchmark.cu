// CLC Policy Benchmark: Demonstrating Custom Scheduler Policies
// Based on the interface defined in clc_scheduler_policy.cuh

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda/ptx>
#include <cooperative_groups.h>
#include "ai_workloads.cuh"
#include "clc_scheduler_policy.cuh"

namespace cg = cooperative_groups;
namespace ptx = cuda::ptx;
using namespace clc_policy;

// ============================================
// Policy-Aware CLC Kernel
// ============================================

template<typename WorkloadType, typename Policy>
__global__ void kernel_cluster_launch_control_policy(float* data, int n, int* block_count, int* steal_count,
                                                     int prologue_complexity) {
    // Policy-specific shared memory
    __shared__ typename Policy::SharedMemory policy_smem;

    // CLC framework shared memory
    __shared__ uint4 result;
    __shared__ uint64_t bar;
    int phase = 0;

    // Initialize the scheduler policy
    Policy::init(policy_smem);
    __syncthreads();

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

        while (!ptx::mbarrier_try_wait_parity(ptx::sem_acquire, ptx::scope_cta, &bar, phase)) {}
        phase ^= 1;

        bool success = ptx::clusterlaunchcontrol_query_cancel_is_canceled(result);

        // Policy hook: check for preemption
        if (Policy::should_preempt(policy_smem)) {
            break;
        }

        if (!success) {
            break;
        }

        int hardware_cta_id = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_x<int>(result);
        
        // Policy hook: select the next work item
        bx = Policy::select_work(policy_smem, hardware_cta_id);

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
void run_policy_comparison(const char* scenario, int prologue,
                           float* d_data, int n, int threads, float* h_data) {
    int blocks_clc = (n + threads - 1) / threads;
    int warmup = 3;
    int runs = 10;

    printf("\n--- Scenario: %s (Prologue: %d) ---\n", scenario, prologue);
    printf("%-20s | %10s | %10s | %10s\n", "Policy", "Time (ms)", "Blocks", "Steals");
    printf("--------------------------------------------------------\n");

    // Baseline: DefaultGreedy Policy
    BenchmarkResult r_greedy = run_clc_policy<WorkloadType, DefaultGreedyPolicy>(d_data, n, blocks_clc, threads, h_data, prologue, warmup, runs);
    printf("%-20s | %10.3f | %10.0f | %10.0f\n", "DefaultGreedy", r_greedy.avg_time_ms, r_greedy.avg_blocks, r_greedy.avg_steals);

    // Test: PriorityBased Policy
    BenchmarkResult r_priority = run_clc_policy<WorkloadType, PriorityBasedPolicy>(d_data, n, blocks_clc, threads, h_data, prologue, warmup, runs);
    printf("%-20s | %10.3f | %10.0f | %10.0f\n", "PriorityBased", r_priority.avg_time_ms, r_priority.avg_blocks, r_priority.avg_steals);

    // Analysis
    float speedup = ((r_greedy.avg_time_ms - r_priority.avg_time_ms) / r_greedy.avg_time_ms) * 100.0f;
    printf("Benefit of Priority: %.2f%% speedup vs Greedy.\n", speedup);
    printf("Note: A slowdown is expected for this simple priority demo due to overhead.\n");
    printf("A real-world scenario with dependent work would show a larger benefit.\n");
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
    printf("CLC Scheduler Policy Benchmark\n");
    printf("Device: %s (CC %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("Elements: %d, Threads/Block: %d\n", n, threads);
    printf("========================================================\n");

    // For this benchmark, we will use a workload where priority could matter,
    // like MixtureOfExperts where some experts might be on a critical path.
    run_policy_comparison<MixtureOfExperts>(get_workload_name<MixtureOfExperts>(), 75, d_data, n, threads, h_data);
    run_policy_comparison<NLPVariableSequence>(get_workload_name<NLPVariableSequence>(), 80, d_data, n, threads, h_data);

    cudaFree(d_data);
    free(h_data);

    return 0;
}
