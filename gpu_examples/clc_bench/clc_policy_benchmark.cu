// CLC Policy Benchmark: Demonstrating and Comparing Scheduler Policies
// Compares basic and specialized policies against fixed-work baselines

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "ai_workloads.cuh"
#include "clc_policy_framework.cuh"
#include "clc_policies.cuh"
#include "benchmark_kernels.cuh"

using namespace clc_policy;

// Kernel and runner now provided by clc_policy_framework.cuh

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
