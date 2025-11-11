// CLC Policy Benchmark - Unified benchmark for all scheduling policies
// Supports multiple modes: comprehensive comparison, specialized policies, quick test

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <vector>
#include "ai_workloads.cuh"
#include "clc_policy_framework.cuh"
#include "clc_policies.cuh"
#include "benchmark_kernels.cuh"

using namespace clc_policy;

// ============================================================================
// Comprehensive Benchmark Result Storage
// ============================================================================

struct ComprehensiveResult {
    const char* workload_name;
    int prologue;
    BenchmarkResult fixed_work;
    BenchmarkResult fixed_blocks;
    BenchmarkResult clc_baseline;
    BenchmarkResult greedy;
    BenchmarkResult maxsteals;
    BenchmarkResult throttled;
    BenchmarkResult selective;
    BenchmarkResult probe_everyn;
    BenchmarkResult latency_budget;
    BenchmarkResult token_bucket;
    BenchmarkResult voting;
};

// ============================================================================
// Mode 1: Comprehensive Comparison (baselines + all policies)
// ============================================================================

template<typename WorkloadType>
ComprehensiveResult run_comprehensive(const char* scenario, int prologue,
                       float* d_data, int n, int threads, cudaDeviceProp& prop, float* h_data) {
    int blocks_fixed_work = (n + threads - 1) / threads;
    int blocks_fixed_blocks = prop.multiProcessorCount * 2;
    int blocks_clc = (n + threads - 1) / threads;
    int warmup = 3;
    int runs = 10;

    ComprehensiveResult result;
    result.workload_name = scenario;
    result.prologue = prologue;

    // Run all benchmarks - baselines
    result.fixed_work = run_fixed_work<WorkloadType>(d_data, n, blocks_fixed_work, threads, h_data, prologue, warmup, runs);
    result.fixed_blocks = run_fixed_blocks<WorkloadType>(d_data, n, blocks_fixed_blocks, threads, h_data, prologue, warmup, runs);
    result.clc_baseline = run_clc_baseline<WorkloadType>(d_data, n, blocks_clc, threads, h_data, prologue, warmup, runs);

    // Run all policies
    result.greedy = run_clc_policy<WorkloadType, GreedyPolicy>(d_data, n, blocks_clc, threads, h_data, prologue, warmup, runs);
    result.maxsteals = run_clc_policy<WorkloadType, MaxStealsPolicy>(d_data, n, blocks_clc, threads, h_data, prologue, warmup, runs);
    result.throttled = run_clc_policy<WorkloadType, ThrottledPolicy>(d_data, n, blocks_clc, threads, h_data, prologue, warmup, runs);
    result.selective = run_clc_policy<WorkloadType, SelectiveBlocksPolicy>(d_data, n, blocks_clc, threads, h_data, prologue, warmup, runs);
    result.probe_everyn = run_clc_policy<WorkloadType, ProbeEveryN_ExitOnFailure>(d_data, n, blocks_clc, threads, h_data, prologue, warmup, runs);
    result.latency_budget = run_clc_policy<WorkloadType, LatencyBudgetPolicy>(d_data, n, blocks_clc, threads, h_data, prologue, warmup, runs);
    result.token_bucket = run_clc_policy<WorkloadType, TokenBucketPolicy>(d_data, n, blocks_clc, threads, h_data, prologue, warmup, runs);
    result.voting = run_clc_policy<WorkloadType, VotingPolicy>(d_data, n, blocks_clc, threads, h_data, prologue, warmup, runs);

    return result;
}

// ============================================================================
// Mode 2: Specialized Policy Comparison (production policies)
// ============================================================================

template<typename WorkloadType>
void run_specialized(const char* workload_name, const char* use_case, const char* best_policy,
                     int prologue, float* d_data, int n, int threads, float* h_data) {
    int blocks = (n + threads - 1) / threads;
    int warmup = 3;
    int runs = 10;

    printf("\n========================================\n");
    printf("Workload: %s (Prologue: %d)\n", workload_name, prologue);
    printf("Use case: %s\n", use_case);
    printf("Best policy: %s\n", best_policy);
    printf("Problem size: %d elements, %d blocks\n", n, blocks);
    printf("========================================\n");
    printf("%-30s | %10s | %10s | %10s\n", "Policy", "Time (ms)", "Blocks", "Steals");
    printf("-----------------------------------------------------------------------------------\n");

    BenchmarkResult r_greedy = run_clc_policy<WorkloadType, GreedyPolicy>(
        d_data, n, blocks, threads, h_data, prologue, warmup, runs);
    printf("%-30s | %10.3f | %10.0f | %10.0f\n",
           "Greedy (baseline)", r_greedy.avg_time_ms, r_greedy.avg_blocks, r_greedy.avg_steals);

    BenchmarkResult r_probe = run_clc_policy<WorkloadType, ProbeEveryN_ExitOnFailure>(
        d_data, n, blocks, threads, h_data, prologue, warmup, runs);
    printf("%-30s | %10.3f | %10.0f | %10.0f\n",
           "ProbeEveryN (N=2)", r_probe.avg_time_ms, r_probe.avg_blocks, r_probe.avg_steals);

    BenchmarkResult r_latency = run_clc_policy<WorkloadType, LatencyBudgetPolicy>(
        d_data, n, blocks, threads, h_data, prologue, warmup, runs);
    printf("%-30s | %10.3f | %10.0f | %10.0f\n",
           "LatencyBudget (150us)", r_latency.avg_time_ms, r_latency.avg_blocks, r_latency.avg_steals);

    BenchmarkResult r_token = run_clc_policy<WorkloadType, TokenBucketPolicy>(
        d_data, n, blocks, threads, h_data, prologue, warmup, runs);
    printf("%-30s | %10.3f | %10.0f | %10.0f\n",
           "TokenBucket (rate-limited)", r_token.avg_time_ms, r_token.avg_blocks, r_token.avg_steals);

    BenchmarkResult r_maxsteals = run_clc_policy<WorkloadType, MaxStealsPolicy>(
        d_data, n, blocks, threads, h_data, prologue, warmup, runs);
    printf("%-30s | %10.3f | %10.0f | %10.0f\n",
           "MaxSteals (limit=8)", r_maxsteals.avg_time_ms, r_maxsteals.avg_blocks, r_maxsteals.avg_steals);

    printf("-----------------------------------------------------------------------------------\n");

    printf("\nAnalysis:\n");
    float probe_overhead = ((r_probe.avg_time_ms - r_greedy.avg_time_ms) / r_greedy.avg_time_ms) * 100.0f;
    printf("  ProbeEveryN: %.2f%% (cooperative drain, fast response)\n", probe_overhead);

    float latency_impact = ((r_latency.avg_time_ms - r_greedy.avg_time_ms) / r_greedy.avg_time_ms) * 100.0f;
    printf("  LatencyBudget: %.2f%% (stable p95/p99 latency)\n", latency_impact);

    float token_impact = ((r_token.avg_time_ms - r_greedy.avg_time_ms) / r_greedy.avg_time_ms) * 100.0f;
    printf("  TokenBucket: %.2f%% (bandwidth fairness)\n", token_impact);

    printf("  Steals reduction:\n");
    printf("    ProbeEveryN: %.1f%%\n", 100.0f * (r_greedy.avg_steals - r_probe.avg_steals) / r_greedy.avg_steals);
    printf("    LatencyBudget: %.1f%%\n", 100.0f * (r_greedy.avg_steals - r_latency.avg_steals) / r_greedy.avg_steals);
    printf("    TokenBucket: %.1f%%\n", 100.0f * (r_greedy.avg_steals - r_token.avg_steals) / r_greedy.avg_steals);
}

// ============================================================================
// Table Printing Functions
// ============================================================================

void print_table_header() {
    printf("\n");
    printf("==============================================================================================================================================================================\n");
    printf("%-35s | %10s | %10s | %10s | %10s | %10s | %10s | %10s | %10s | %10s | %10s | %10s\n",
           "Workload (Prologue)",
           "FixedWork", "FixedBlks", "CLC-Base",
           "Greedy", "MaxSteals", "Throttled", "Selective",
           "ProbeEvryN", "LatencyBdg", "TokenBkt", "Voting");
    printf("==============================================================================================================================================================================\n");
}

void print_table_row_time(const ComprehensiveResult& r) {
    printf("%-35s | %10.3f | %10.3f | %10.3f | %10.3f | %10.3f | %10.3f | %10.3f | %10.3f | %10.3f | %10.3f | %10.3f | (ms)\n",
           r.workload_name,
           r.fixed_work.avg_time_ms,
           r.fixed_blocks.avg_time_ms,
           r.clc_baseline.avg_time_ms,
           r.greedy.avg_time_ms,
           r.maxsteals.avg_time_ms,
           r.throttled.avg_time_ms,
           r.selective.avg_time_ms,
           r.probe_everyn.avg_time_ms,
           r.latency_budget.avg_time_ms,
           r.token_bucket.avg_time_ms,
           r.voting.avg_time_ms);
}

void print_table_row_steals(const ComprehensiveResult& r) {
    printf("  └─ Steals                        | %10s | %10s | %10.0f | %10.0f | %10.0f | %10.0f | %10.0f | %10.0f | %10.0f | %10.0f | %10.0f |\n",
           "-", "-",
           r.clc_baseline.avg_steals,
           r.greedy.avg_steals,
           r.maxsteals.avg_steals,
           r.throttled.avg_steals,
           r.selective.avg_steals,
           r.probe_everyn.avg_steals,
           r.latency_budget.avg_steals,
           r.token_bucket.avg_steals,
           r.voting.avg_steals);
}

void print_speedup_analysis(const std::vector<ComprehensiveResult>& results) {
    printf("\n");
    printf("==============================================================================================================================================================================\n");
    printf("SPEEDUP ANALYSIS (vs CLC Baseline, negative = slower)\n");
    printf("==============================================================================================================================================================================\n");
    printf("%-35s | %10s | %10s | %10s | %10s | %10s | %10s | %10s | %10s | %10s | %10s\n",
           "Workload", "Greedy", "MaxSteals", "Throttled", "Selective",
           "ProbeEvryN", "LatencyBdg", "TokenBkt", "Voting",
           "FixedWork", "FixedBlks");
    printf("==============================================================================================================================================================================\n");

    for (const auto& r : results) {
        float base = r.clc_baseline.avg_time_ms;
        printf("%-35s | %9.1f%% | %9.1f%% | %9.1f%% | %9.1f%% | %9.1f%% | %9.1f%% | %9.1f%% | %9.1f%% | %9.1f%% | %9.1f%%\n",
               r.workload_name,
               ((base - r.greedy.avg_time_ms) / base) * 100.0f,
               ((base - r.maxsteals.avg_time_ms) / base) * 100.0f,
               ((base - r.throttled.avg_time_ms) / base) * 100.0f,
               ((base - r.selective.avg_time_ms) / base) * 100.0f,
               ((base - r.probe_everyn.avg_time_ms) / base) * 100.0f,
               ((base - r.latency_budget.avg_time_ms) / base) * 100.0f,
               ((base - r.token_bucket.avg_time_ms) / base) * 100.0f,
               ((base - r.voting.avg_time_ms) / base) * 100.0f,
               ((base - r.fixed_work.avg_time_ms) / base) * 100.0f,
               ((base - r.fixed_blocks.avg_time_ms) / base) * 100.0f);
    }
    printf("==============================================================================================================================================================================\n");
}

// ============================================================================
// Main - Run All Benchmarks
// ============================================================================

int main(int argc, char** argv) {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    if (prop.major < 10) {
        fprintf(stderr, "ERROR: CLC requires CC 10.0+\n");
        return 1;
    }

    // Parse arguments
    int n = 1024 * 1024;  // Default: 1M elements
    int threads = 256;

    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) threads = atoi(argv[2]);

    // Allocate data
    float *h_data = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) h_data[i] = (float)(i % 100) + 1.0f;

    float *d_data;
    cudaMalloc(&d_data, n * sizeof(float));

    // Print header
    printf("========================================================\n");
    printf("CLC Policy Benchmark - Comprehensive Evaluation\n");
    printf("Device: %s (CC %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("Elements: %d, Threads/Block: %d\n", n, threads);
    printf("========================================================\n");

    // Run all comprehensive benchmarks
    std::vector<ComprehensiveResult> results;

    printf("\nRunning comprehensive benchmarks...\n");
    results.push_back(run_comprehensive<MixtureOfExperts>(get_workload_name<MixtureOfExperts>(), 75, d_data, n, threads, prop, h_data));
    results.push_back(run_comprehensive<NLPVariableSequence>(get_workload_name<NLPVariableSequence>(), 80, d_data, n, threads, prop, h_data));
    results.push_back(run_comprehensive<GraphNeuralNetwork>(get_workload_name<GraphNeuralNetwork>(), 50, d_data, n, threads, prop, h_data));
    results.push_back(run_comprehensive<DynamicBatching>(get_workload_name<DynamicBatching>(), 60, d_data, n, threads, prop, h_data));
    results.push_back(run_comprehensive<SparseAttention>(get_workload_name<SparseAttention>(), 70, d_data, n, threads, prop, h_data));
    results.push_back(run_comprehensive<VideoProcessing>(get_workload_name<VideoProcessing>(), 65, d_data, n, threads, prop, h_data));

    // Print comprehensive table
    print_table_header();
    for (const auto& r : results) {
        print_table_row_time(r);
        print_table_row_steals(r);
    }
    printf("==============================================================================================================================================================================\n");

    // Print speedup analysis
    print_speedup_analysis(results);

    // Part 2: Specialized policies for production use cases
    printf("\n");
    printf("========================================================\n");
    printf("SPECIALIZED POLICY EVALUATION\n");
    printf("(Production Policies for Real-World Use Cases)\n");
    printf("========================================================\n");

    printf("\n=== Scenario 1: Streaming Inference (Latency-Critical) ===\n");
    run_specialized<MixtureOfExperts>("MoE Inference",
        "Online inference with SLO requirements",
        "LatencyBudgetPolicy - bounds tail latency",
        75, d_data, n, threads, h_data);

    printf("\n=== Scenario 2: Memory-Bound ETL/Compression ===\n");
    run_specialized<NLPVariableSequence>("NLP Processing",
        "High bandwidth utilization, prevent thrashing",
        "TokenBucketPolicy - rate-limits for fairness",
        80, d_data, n, threads, h_data);

    printf("\n=== Scenario 3: Interactive/Cooperative Workloads ===\n");
    run_specialized<GraphNeuralNetwork>("GNN Training",
        "Low-priority background job, drain for high-priority",
        "ProbeEveryN - reduces overhead, fast drain",
        50, d_data, n, threads, h_data);

    cudaFree(d_data);
    free(h_data);

    return 0;
}
