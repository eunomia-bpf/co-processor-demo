// CLC Policy Comparison Benchmark
// Compares specialized policies against baseline greedy CLC
// Demonstrates real-world use cases: latency, throughput, fairness

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
void run_policy_comparison(const char* workload_name, int prologue,
                           float* d_data, int n, int threads, float* h_data) {
    int blocks = (n + threads - 1) / threads;
    int warmup = 3;
    int runs = 10;

    printf("\n========================================\n");
    printf("Workload: %s (Prologue: %d)\n", workload_name, prologue);
    printf("Problem size: %d elements, %d blocks\n", n, blocks);
    printf("========================================\n");
    printf("%-30s | %10s | %10s | %10s\n", "Policy", "Time (ms)", "Blocks", "Steals");
    printf("-----------------------------------------------------------------------------------\n");

    // Baseline: Greedy
    BenchmarkResult r_greedy = run_clc_policy<WorkloadType, GreedyPolicy>(
        d_data, n, blocks, threads, h_data, prologue, warmup, runs);
    printf("%-30s | %10.3f | %10.0f | %10.0f\n",
           "Greedy (baseline)", r_greedy.avg_time_ms, r_greedy.avg_blocks, r_greedy.avg_steals);

    // ProbeEveryN_ExitOnFailure
    BenchmarkResult r_probe = run_clc_policy<WorkloadType, ProbeEveryN_ExitOnFailure>(
        d_data, n, blocks, threads, h_data, prologue, warmup, runs);
    printf("%-30s | %10.3f | %10.0f | %10.0f\n",
           "ProbeEveryN (N=2)", r_probe.avg_time_ms, r_probe.avg_blocks, r_probe.avg_steals);

    // LatencyBudgetPolicy
    BenchmarkResult r_latency = run_clc_policy<WorkloadType, LatencyBudgetPolicy>(
        d_data, n, blocks, threads, h_data, prologue, warmup, runs);
    printf("%-30s | %10.3f | %10.0f | %10.0f\n",
           "LatencyBudget (150us)", r_latency.avg_time_ms, r_latency.avg_blocks, r_latency.avg_steals);

    // TokenBucketPolicy
    BenchmarkResult r_token = run_clc_policy<WorkloadType, TokenBucketPolicy>(
        d_data, n, blocks, threads, h_data, prologue, warmup, runs);
    printf("%-30s | %10.3f | %10.0f | %10.0f\n",
           "TokenBucket (rate-limited)", r_token.avg_time_ms, r_token.avg_blocks, r_token.avg_steals);

    // MaxSteals (for comparison)
    BenchmarkResult r_maxsteals = run_clc_policy<WorkloadType, MaxStealsPolicy>(
        d_data, n, blocks, threads, h_data, prologue, warmup, runs);
    printf("%-30s | %10.3f | %10.0f | %10.0f\n",
           "MaxSteals (limit=8)", r_maxsteals.avg_time_ms, r_maxsteals.avg_blocks, r_maxsteals.avg_steals);

    printf("-----------------------------------------------------------------------------------\n");

    // Analysis
    printf("\nAnalysis:\n");

    float probe_overhead = ((r_probe.avg_time_ms - r_greedy.avg_time_ms) / r_greedy.avg_time_ms) * 100.0f;
    printf("  ProbeEveryN overhead: %.2f%% (reduces probe frequency for responsiveness)\n", probe_overhead);

    float latency_impact = ((r_latency.avg_time_ms - r_greedy.avg_time_ms) / r_greedy.avg_time_ms) * 100.0f;
    printf("  LatencyBudget impact: %.2f%% (bounds CTA stealing time for stable latency)\n", latency_impact);

    float token_impact = ((r_token.avg_time_ms - r_greedy.avg_time_ms) / r_greedy.avg_time_ms) * 100.0f;
    printf("  TokenBucket impact: %.2f%% (rate-limits steals for bandwidth fairness)\n", token_impact);

    printf("  Steals reduction:\n");
    printf("    ProbeEveryN: %.1f%% fewer steals\n",
           100.0f * (r_greedy.avg_steals - r_probe.avg_steals) / r_greedy.avg_steals);
    printf("    LatencyBudget: %.1f%% fewer steals\n",
           100.0f * (r_greedy.avg_steals - r_latency.avg_steals) / r_greedy.avg_steals);
    printf("    TokenBucket: %.1f%% fewer steals\n",
           100.0f * (r_greedy.avg_steals - r_token.avg_steals) / r_greedy.avg_steals);
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

    int n = 4 * 1024 * 1024;  // 4M elements for better work-stealing
    int threads = 256;

    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) threads = atoi(argv[2]);

    float *h_data = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) h_data[i] = (float)(i % 100) + 1.0f;

    float *d_data;
    cudaMalloc(&d_data, n * sizeof(float));

    printf("========================================================\n");
    printf("CLC Policy Comparison Benchmark\n");
    printf("Device: %s (CC %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("Elements: %d, Threads/Block: %d\n", n, threads);
    printf("========================================================\n");

    // Test different workloads
    printf("\n=== Scenario 1: Streaming Inference (Latency-Critical) ===\n");
    printf("Use case: Online inference with SLO requirements\n");
    printf("Best policy: LatencyBudgetPolicy - bounds tail latency\n");
    run_policy_comparison<MixtureOfExperts>("MoE Inference", 75, d_data, n, threads, h_data);

    printf("\n=== Scenario 2: Memory-Bound ETL/Compression ===\n");
    printf("Use case: High bandwidth utilization, prevent thrashing\n");
    printf("Best policy: TokenBucketPolicy - rate-limits for fairness\n");
    run_policy_comparison<NLPVariableSequence>("NLP Processing", 80, d_data, n, threads, h_data);

    printf("\n=== Scenario 3: Interactive/Cooperative Workloads ===\n");
    printf("Use case: Low-priority background job, drain for high-priority\n");
    printf("Best policy: ProbeEveryN - reduces overhead, fast drain\n");
    run_policy_comparison<GraphNeuralNetwork>("GNN Training", 50, d_data, n, threads, h_data);

    cudaFree(d_data);
    free(h_data);

    printf("\n========================================================\n");
    printf("Summary:\n");
    printf("  ProbeEveryN: Low overhead cooperative drain\n");
    printf("  LatencyBudget: Stable p95/p99 latency for SLOs\n");
    printf("  TokenBucket: Bandwidth fairness, prevents saturation\n");
    printf("  MaxSteals: Simple load balancing\n");
    printf("========================================================\n");

    return 0;
}
