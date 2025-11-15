#ifndef METRICS_H
#define METRICS_H

#include <vector>
#include <algorithm>
#include <cmath>
#include <map>
#include <cuda_runtime.h>
#include "common.h"

/**
 * Compute comprehensive benchmark metrics from kernel timing data
 *
 * Calculates performance, latency, concurrency, fairness, and memory metrics
 * and outputs them in both human-readable and CSV formats.
 */
void compute_metrics(const std::vector<KernelTiming> &timings,
                     const BenchmarkConfig &config, int grid_x, int block_x) {
    if (timings.empty()) return;

    // Find global start and end times
    float global_start = timings[0].start_time_ms;
    float global_end = timings[0].end_time_ms;
    float global_enqueue = timings[0].enqueue_time_ms;
    float total_kernel_time = 0.0f;

    std::vector<float> svc_times;      // Service time (execution time only)
    std::vector<float> e2e_times;      // End-to-end latency (including queue wait)
    std::vector<float> queue_waits;    // Queue waiting time

    for (const auto &t : timings) {
        global_start = fminf(global_start, t.start_time_ms);
        global_end = fmaxf(global_end, t.end_time_ms);
        global_enqueue = fminf(global_enqueue, t.enqueue_time_ms);
        total_kernel_time += t.duration_ms;

        svc_times.push_back(t.duration_ms);
        e2e_times.push_back(t.e2e_latency_ms);
        queue_waits.push_back(t.launch_latency_ms);
    }

    // Wall time: service window makespan (first kernel start -> last kernel end)
    // This is appropriate for throughput and GPU utilization calculations
    float total_wall_time = global_end - global_start;

    // E2E wall time: includes queue time of first kernel (first enqueue -> last end)
    // This represents true end-to-end system response time
    float total_e2e_wall_time = global_end - global_enqueue;

    int total_kernels = timings.size();

    // Average service time (execution time only)
    float avg_service_time = total_kernel_time / total_kernels;

    // Average e2e latency
    float total_e2e = 0.0f;
    for (float e : e2e_times) total_e2e += e;
    float avg_e2e_latency = total_e2e / total_kernels;

    // Throughput
    float throughput = total_kernels / (total_wall_time / 1000.0f); // kernels/sec

    // Load imbalance (standard deviation of service time)
    float variance = 0.0f;
    for (float d : svc_times) {
        variance += (d - avg_service_time) * (d - avg_service_time);
    }
    float stddev = sqrtf(variance / svc_times.size());

    // Service time percentiles (P50, P95, P99)
    std::vector<float> sorted_svc = svc_times;
    std::sort(sorted_svc.begin(), sorted_svc.end());
    float svc_p50 = sorted_svc[sorted_svc.size() * 50 / 100];
    float svc_p95 = sorted_svc[sorted_svc.size() * 95 / 100];
    float svc_p99 = sorted_svc[sorted_svc.size() * 99 / 100];

    // E2E latency percentiles (P50, P95, P99)
    std::vector<float> sorted_e2e = e2e_times;
    std::sort(sorted_e2e.begin(), sorted_e2e.end());
    float e2e_p50 = sorted_e2e[sorted_e2e.size() * 50 / 100];
    float e2e_p95 = sorted_e2e[sorted_e2e.size() * 95 / 100];
    float e2e_p99 = sorted_e2e[sorted_e2e.size() * 99 / 100];

    // Jain's Fairness Index: (sum xi)^2 / (n * sum xi^2)
    // Perfect fairness = 1.0, lower values indicate unfairness
    std::vector<float> per_stream_time(config.num_streams, 0.0f);
    for (const auto &t : timings) {
        per_stream_time[t.stream_id] += t.duration_ms;
    }
    float sum_time = 0.0f, sum_time_squared = 0.0f;
    for (float t : per_stream_time) {
        sum_time += t;
        sum_time_squared += t * t;
    }
    float jains_index = (sum_time * sum_time) / (config.num_streams * sum_time_squared);

    // Launch latency statistics
    float avg_launch_latency = 0.0f;
    float max_launch_latency = 0.0f;
    for (const auto &t : timings) {
        avg_launch_latency += t.launch_latency_ms;
        max_launch_latency = fmaxf(max_launch_latency, t.launch_latency_ms);
    }
    avg_launch_latency /= total_kernels;

    // Priority inversion detection and per-priority latency
    int priority_inversions = 0;
    int total_cross_priority_pairs = 0;
    std::map<int, std::vector<float>> priority_latencies; // priority -> list of e2e latencies

    // Detect inversions and count cross-priority pairs
    for (size_t i = 0; i < timings.size(); i++) {
        for (size_t j = i + 1; j < timings.size(); j++) {
            // Count pairs with different priorities
            if (timings[i].priority != timings[j].priority) {
                total_cross_priority_pairs++;
            }
            // If higher priority (lower number) started later than lower priority
            if (timings[i].priority < timings[j].priority &&
                timings[i].start_time_ms > timings[j].start_time_ms &&
                timings[j].end_time_ms > timings[i].start_time_ms) {
                priority_inversions++;
            }
        }
    }

    // Normalized inversion rate
    float inversion_rate = (total_cross_priority_pairs > 0)
                         ? (float)priority_inversions / total_cross_priority_pairs
                         : 0.0f;

    // Collect per-priority latencies
    for (const auto &t : timings) {
        priority_latencies[t.priority].push_back(t.e2e_latency_ms);
    }

    // Working set size calculation
    float working_set_mb = (config.num_streams * config.workload_size * sizeof(float)) / (1024.0f * 1024.0f);

    // Get GPU L2 cache size
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    float l2_cache_mb = prop.l2CacheSize / (1024.0f * 1024.0f);
    bool fits_in_l2 = working_set_mb <= l2_cache_mb;

    // Queue depth analysis - find peak concurrent kernels and compute concurrency metrics
    std::vector<std::pair<float, int>> events; // time, delta (+1 start, -1 end)
    for (const auto &t : timings) {
        events.push_back({t.start_time_ms, 1});
        events.push_back({t.end_time_ms, -1});
    }
    std::sort(events.begin(), events.end());

    int current_concurrent = 0;
    int max_concurrent = 0;
    float avg_concurrent = 0.0f;
    float time_ge1 = 0.0f;  // Time with >=1 kernels active
    float time_ge2 = 0.0f;  // Time with >=2 kernels active
    float last_time = events[0].first;

    for (const auto &e : events) {
        float dt = e.first - last_time;
        if (dt > 0) {
            if (current_concurrent >= 1) time_ge1 += dt;
            if (current_concurrent >= 2) time_ge2 += dt;
            avg_concurrent += current_concurrent * dt;
        }
        current_concurrent += e.second;
        max_concurrent = std::max(max_concurrent, current_concurrent);
        last_time = e.first;
    }
    avg_concurrent /= total_wall_time;

    // Concurrent execution rate: percentage of active time with >=2 kernels running
    float concurrent_rate = (time_ge1 > 0) ? (time_ge2 / time_ge1 * 100.0f) : 0.0f;

    // GPU utilization: percentage of time with at least 1 kernel active
    float gpu_utilization = (time_ge1 / total_wall_time) * 100.0f;

    // Print results
    printf("\n====================================\n");
    printf("Multi-Stream Scheduler Benchmark\n");
    printf("====================================\n");
    printf("Configuration:\n");
    printf("  Streams: %d\n", config.num_streams);

    // Print kernels per stream - handle load imbalance case
    if (!config.kernels_per_stream_custom.empty()) {
        printf("  Kernels per stream (load imbalance): [");
        for (size_t i = 0; i < config.kernels_per_stream_custom.size(); i++) {
            if (i > 0) printf(", ");
            printf("%d", config.kernels_per_stream_custom[i]);
        }
        printf("]\n");
    } else {
        printf("  Kernels per stream: %d (uniform)\n", config.num_kernels_per_stream);
    }

    printf("  Total kernels launched: %d\n", total_kernels);
    printf("  Workload size: %d elements\n", config.workload_size);

    // Print kernel type - handle heterogeneous case
    if (config.use_heterogeneous && !config.kernel_types_per_stream.empty()) {
        printf("  Kernel types (heterogeneous): [");
        for (size_t i = 0; i < config.kernel_types_per_stream.size(); i++) {
            if (i > 0) printf(", ");
            KernelType kt = config.kernel_types_per_stream[i];
            printf("%s", kt == COMPUTE ? "compute" :
                        kt == MEMORY ? "memory" :
                        kt == GEMM ? "gemm" : "mixed");
        }
        printf("]\n");
    } else {
        printf("  Kernel type: %s (uniform)\n",
               config.kernel_type == COMPUTE ? "compute" :
               config.kernel_type == MEMORY ? "memory" :
               config.kernel_type == GEMM ? "gemm" : "mixed");
    }
    printf("\nResults:\n");
    printf("  Service window time: %.2f ms (first start -> last end)\n", total_wall_time);
    printf("  E2E wall time: %.2f ms (first enqueue -> last end)\n", total_e2e_wall_time);
    printf("  Aggregate throughput: %.2f kernels/sec\n", throughput);
    printf("\nService Time Metrics (execution only):\n");
    printf("  Mean: %.2f ms\n", avg_service_time);
    printf("  P50: %.2f ms\n", svc_p50);
    printf("  P95: %.2f ms\n", svc_p95);
    printf("  P99: %.2f ms\n", svc_p99);
    printf("\nEnd-to-End Latency Metrics (including queue wait):\n");
    printf("  Mean: %.2f ms\n", avg_e2e_latency);
    printf("  P50: %.2f ms\n", e2e_p50);
    printf("  P95: %.2f ms\n", e2e_p95);
    printf("  P99: %.2f ms\n", e2e_p99);
    printf("  Avg queue wait: %.4f ms\n", avg_launch_latency);
    printf("  Max queue wait: %.4f ms\n", max_launch_latency);
    printf("\nScheduler Metrics:\n");
    printf("  Concurrent execution rate: %.1f%% (time with >=2 kernels active)\n", concurrent_rate);
    printf("  GPU utilization: %.1f%% (time with >=1 kernel active)\n", gpu_utilization);
    printf("\nConcurrency Metrics:\n");
    printf("  Max concurrent kernels: %d\n", max_concurrent);
    printf("  Avg concurrent kernels: %.2f\n", avg_concurrent);
    printf("  Peak concurrency: %d / %d streams (%.1f%%)\n",
           max_concurrent, config.num_streams,
           (float)max_concurrent / config.num_streams * 100.0f);
    printf("\nFairness Metrics:\n");
    printf("  Jain's Fairness Index: %.4f (1.0 = perfect)\n", jains_index);
    printf("  Load imbalance (stddev): %.2f ms\n", stddev);

    printf("\nMemory Metrics:\n");
    printf("  Working set size: %.2f MB\n", working_set_mb);
    printf("  L2 cache size: %.2f MB\n", l2_cache_mb);
    printf("  Fits in L2: %s\n", fits_in_l2 ? "YES" : "NO");

    if (!priority_latencies.empty() && priority_latencies.size() > 1) {
        printf("\nPriority Metrics:\n");
        printf("  Priority inversions: %d (%.1f%% of cross-priority pairs)\n",
               priority_inversions, inversion_rate * 100.0f);

        printf("  Per-Priority E2E Latency:\n");
        for (const auto &pair : priority_latencies) {
            std::vector<float> sorted_lats = pair.second;
            std::sort(sorted_lats.begin(), sorted_lats.end());

            float sum = 0.0f;
            for (float lat : sorted_lats) sum += lat;
            float avg = sum / sorted_lats.size();
            float p50 = sorted_lats[sorted_lats.size() / 2];
            float p99 = sorted_lats[(sorted_lats.size() * 99) / 100];

            printf("    Priority %d: avg=%.3fms, P50=%.3fms, P99=%.3fms (n=%zu)\n",
                   pair.first, avg, p50, p99, sorted_lats.size());
        }
    }
    printf("====================================\n\n");

    // CSV output for easy parsing - output to stderr for separation
    // Header row (skip if --no-header specified)
    if (!config.csv_no_header) {
        fprintf(stderr, "streams,kernels_per_stream,kernels_per_stream_detail,total_kernels,type,type_detail,wall_time_ms,e2e_wall_time_ms,throughput,");
        fprintf(stderr, "svc_mean,svc_p50,svc_p95,svc_p99,");
        fprintf(stderr, "e2e_mean,e2e_p50,e2e_p95,e2e_p99,");
        fprintf(stderr, "avg_queue_wait,max_queue_wait,");
        fprintf(stderr, "concurrent_rate,util,jains_index,max_concurrent,avg_concurrent,");
        fprintf(stderr, "inversions,inversion_rate,working_set_mb,fits_in_l2,svc_stddev,grid_size,block_size,");
        fprintf(stderr, "per_priority_avg,per_priority_p50,per_priority_p99,");
        fprintf(stderr, "launch_freq,seed\n");
    }

    // Data row
    fprintf(stderr, "%d,%d,",
           config.num_streams, config.num_kernels_per_stream);

    // Output kernels_per_stream_detail (colon-separated for load imbalance)
    if (!config.kernels_per_stream_custom.empty()) {
        for (size_t i = 0; i < config.kernels_per_stream_custom.size(); i++) {
            if (i > 0) fprintf(stderr, ":");
            fprintf(stderr, "%d", config.kernels_per_stream_custom[i]);
        }
    } else {
        fprintf(stderr, "uniform");
    }
    fprintf(stderr, ",");

    fprintf(stderr, "%d,", total_kernels);

    // Output kernel type (nominal)
    const char *type_str = config.kernel_type == COMPUTE ? "compute" :
                          config.kernel_type == MEMORY ? "memory" :
                          config.kernel_type == GEMM ? "gemm" : "mixed";
    fprintf(stderr, "%s,", type_str);

    // Output type_detail (colon-separated for heterogeneous)
    if (config.use_heterogeneous && !config.kernel_types_per_stream.empty()) {
        for (size_t i = 0; i < config.kernel_types_per_stream.size(); i++) {
            if (i > 0) fprintf(stderr, ":");
            KernelType kt = config.kernel_types_per_stream[i];
            fprintf(stderr, "%s", kt == COMPUTE ? "compute" :
                        kt == MEMORY ? "memory" :
                        kt == GEMM ? "gemm" : "mixed");
        }
    } else {
        fprintf(stderr, "uniform");
    }
    fprintf(stderr, ",");

    fprintf(stderr, "%.2f,%.2f,%.2f,", total_wall_time, total_e2e_wall_time, throughput);
    fprintf(stderr, "%.2f,%.2f,%.2f,%.2f,", avg_service_time, svc_p50, svc_p95, svc_p99);
    fprintf(stderr, "%.2f,%.2f,%.2f,%.2f,", avg_e2e_latency, e2e_p50, e2e_p95, e2e_p99);
    fprintf(stderr, "%.4f,%.4f,", avg_launch_latency, max_launch_latency);
    fprintf(stderr, "%.1f,%.1f,%.4f,%d,%.2f,", concurrent_rate, gpu_utilization,
           jains_index, max_concurrent, avg_concurrent);
    fprintf(stderr, "%d,%.6f,%.2f,%d,%.2f,%d,%d,",
           priority_inversions, inversion_rate,
           working_set_mb, fits_in_l2 ? 1 : 0, stddev, grid_x, block_x);

    // Output per-priority metrics in three separate fields
    // Each field contains colon-separated values for each priority
    // Format: per_priority_avg = val1:val2:val3, per_priority_p50 = val1:val2:val3, etc.

    // Compute metrics for each priority
    std::map<int, float> priority_avg, priority_p50_map, priority_p99_map;
    for (const auto &pair : priority_latencies) {
        std::vector<float> sorted_lats = pair.second;
        std::sort(sorted_lats.begin(), sorted_lats.end());

        float sum = 0.0f;
        for (float lat : sorted_lats) sum += lat;
        float avg = sum / sorted_lats.size();
        float p50_val = sorted_lats[sorted_lats.size() / 2];
        float p99_val = sorted_lats[(sorted_lats.size() * 99) / 100];

        priority_avg[pair.first] = avg;
        priority_p50_map[pair.first] = p50_val;
        priority_p99_map[pair.first] = p99_val;
    }

    // Output avg values
    if (!priority_avg.empty()) {
        bool first = true;
        for (const auto &pair : priority_avg) {
            if (!first) fprintf(stderr, ":");
            fprintf(stderr, "%.3f", pair.second);
            first = false;
        }
    }
    fprintf(stderr, ",");

    // Output p50 values
    if (!priority_p50_map.empty()) {
        bool first = true;
        for (const auto &pair : priority_p50_map) {
            if (!first) fprintf(stderr, ":");
            fprintf(stderr, "%.3f", pair.second);
            first = false;
        }
    }
    fprintf(stderr, ",");

    // Output p99 values
    if (!priority_p99_map.empty()) {
        bool first = true;
        for (const auto &pair : priority_p99_map) {
            if (!first) fprintf(stderr, ":");
            fprintf(stderr, "%.3f", pair.second);
            first = false;
        }
    }
    fprintf(stderr, ",");

    // Output launch_freq (average across streams, or first stream if uniform)
    if (!config.launch_frequency_per_stream.empty()) {
        fprintf(stderr, "%.1f", config.launch_frequency_per_stream[0]);
    } else {
        fprintf(stderr, "0");
    }
    fprintf(stderr, ",");

    // Output random seed
    fprintf(stderr, "%u", config.random_seed);

    fprintf(stderr, "\n");
}

// ============================================================================
// RQ3: CUDA Priority Mechanism Analysis
// ============================================================================

/**
 * Detect priority inversions
 *
 * A priority inversion occurs when a higher-priority kernel (lower number)
 * starts executing after a lower-priority kernel, even though both are queued.
 *
 * @param timings Vector of kernel timing data with priority information
 * @return Number of priority inversions detected
 */
inline int detect_priority_inversions(const std::vector<KernelTiming> &timings) {
    int priority_inversions = 0;

    for (size_t i = 0; i < timings.size(); i++) {
        for (size_t j = i + 1; j < timings.size(); j++) {
            // If higher priority (lower number) started later than lower priority
            // and there's overlap in their execution windows
            if (timings[i].priority < timings[j].priority &&
                timings[i].start_time_ms > timings[j].start_time_ms &&
                timings[j].end_time_ms > timings[i].start_time_ms) {
                priority_inversions++;
            }
        }
    }

    return priority_inversions;
}

/**
 * Compute per-priority-class E2E latency statistics
 *
 * Groups kernels by priority class and computes average E2E latency for each.
 * This helps determine if priority affects end-to-end performance.
 *
 * @param timings Vector of kernel timing data
 * @param config Benchmark configuration
 */
inline void compute_priority_class_latency(const std::vector<KernelTiming> &timings,
                                          const BenchmarkConfig &config) {
    if (config.priorities_per_stream.empty() || timings.empty()) {
        return;
    }

    // Group by priority - use e2e_latency_ms for consistency
    std::map<int, std::vector<float>> priority_e2e_latencies;
    for (const auto &t : timings) {
        priority_e2e_latencies[t.priority].push_back(t.e2e_latency_ms);
    }

    printf("\nPer-Priority-Class E2E Latency:\n");
    for (const auto &pair : priority_e2e_latencies) {
        int priority = pair.first;
        const std::vector<float> &latencies = pair.second;

        float sum = 0.0f;
        for (float lat : latencies) {
            sum += lat;
        }
        float avg = sum / latencies.size();

        // Compute standard deviation
        float variance = 0.0f;
        for (float lat : latencies) {
            variance += (lat - avg) * (lat - avg);
        }
        float stddev = sqrtf(variance / latencies.size());

        printf("  Priority %d: %.3f ms (±%.3f ms, n=%zu)\n",
               priority, avg, stddev, latencies.size());
    }
}

/**
 * Output detailed per-kernel timing data for RQ3 analysis
 *
 * Outputs CSV format with per-kernel details including priority,
 * start/end times, duration, launch latency, and end-to-end latency.
 *
 * NOTE: For RQ3 priority analysis, use e2e_latency_ms (not duration_ms)
 * to measure the impact of priority on end-to-end performance including
 * queue wait time.
 *
 * @param timings Vector of kernel timing data
 */
inline void output_rq3_detailed_csv(const std::vector<KernelTiming> &timings) {
    printf("DETAIL_CSV_HEADER: stream_id,kernel_id,priority,start_ms,end_ms,duration_ms,launch_latency_ms,e2e_latency_ms\n");
    for (const auto& t : timings) {
        printf("DETAIL_CSV: %d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f\n",
               t.stream_id, t.kernel_id, t.priority,
               t.start_time_ms, t.end_time_ms,
               t.duration_ms, t.launch_latency_ms, t.e2e_latency_ms);
    }
}

/**
 * Compute all RQ3-specific metrics
 *
 * Runs complete RQ3 analysis including inversion detection,
 * per-priority latency analysis, and detailed output.
 *
 * @param timings Vector of kernel timing data
 * @param config Benchmark configuration
 * @return Number of priority inversions detected
 */
inline int compute_rq3_metrics(const std::vector<KernelTiming> &timings,
                               const BenchmarkConfig &config) {
    if (config.priorities_per_stream.empty()) {
        return 0;
    }

    // Detect priority inversions
    int inversions = detect_priority_inversions(timings);

    // Compute per-priority-class latency
    compute_priority_class_latency(timings, config);

    // Output detailed CSV for analysis
    output_rq3_detailed_csv(timings);

    return inversions;
}

#endif // METRICS_H
