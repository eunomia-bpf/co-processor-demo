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
    float total_kernel_time = 0.0f;

    std::vector<float> durations;
    for (const auto &t : timings) {
        global_start = fminf(global_start, t.start_time_ms);
        global_end = fmaxf(global_end, t.end_time_ms);
        total_kernel_time += t.duration_ms;
        durations.push_back(t.duration_ms);
    }

    float total_wall_time = global_end - global_start;
    int total_kernels = timings.size();

    // Concurrent execution rate
    float ideal_parallel_time = total_kernel_time / config.num_streams;
    float concurrent_rate = (ideal_parallel_time / total_wall_time) * 100.0f;
    if (concurrent_rate > 100.0f) concurrent_rate = 100.0f;

    // Average latency
    float avg_latency = total_kernel_time / total_kernels;

    // Throughput
    float throughput = total_kernels / (total_wall_time / 1000.0f); // kernels/sec

    // Load imbalance (standard deviation)
    float mean_duration = avg_latency;
    float variance = 0.0f;
    for (float d : durations) {
        variance += (d - mean_duration) * (d - mean_duration);
    }
    float stddev = sqrtf(variance / durations.size());

    // Percentile latencies (P50, P95, P99)
    std::vector<float> sorted_durations = durations;
    std::sort(sorted_durations.begin(), sorted_durations.end());
    float p50 = sorted_durations[sorted_durations.size() * 50 / 100];
    float p95 = sorted_durations[sorted_durations.size() * 95 / 100];
    float p99 = sorted_durations[sorted_durations.size() * 99 / 100];

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

    // Priority inversion detection (if priorities enabled)
    int priority_inversions = 0;
    if (config.enable_priorities) {
        for (size_t i = 0; i < timings.size(); i++) {
            for (size_t j = i + 1; j < timings.size(); j++) {
                // If higher priority (lower number) started later than lower priority
                if (timings[i].priority < timings[j].priority &&
                    timings[i].start_time_ms > timings[j].start_time_ms &&
                    timings[j].end_time_ms > timings[i].start_time_ms) {
                    priority_inversions++;
                }
            }
        }
    }

    // Scheduler overhead (approximate)
    float scheduler_overhead = ((total_wall_time - ideal_parallel_time) / total_wall_time) * 100.0f;
    if (scheduler_overhead < 0.0f) scheduler_overhead = 0.0f;

    // Estimate GPU utilization (simplified)
    float gpu_utilization = concurrent_rate * 0.95f; // Approximate

    // Working set size calculation
    float working_set_mb = (config.num_streams * config.workload_size * sizeof(float)) / (1024.0f * 1024.0f);

    // Get GPU L2 cache size
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    float l2_cache_mb = prop.l2CacheSize / (1024.0f * 1024.0f);
    bool fits_in_l2 = working_set_mb <= l2_cache_mb;

    // Queue depth analysis - find peak concurrent kernels
    std::vector<std::pair<float, int>> events; // time, delta (+1 start, -1 end)
    for (const auto &t : timings) {
        events.push_back({t.start_time_ms, 1});
        events.push_back({t.end_time_ms, -1});
    }
    std::sort(events.begin(), events.end());

    int current_concurrent = 0;
    int max_concurrent = 0;
    float avg_concurrent = 0.0f;
    float last_time = events[0].first;

    for (const auto &e : events) {
        if (e.first > last_time) {
            avg_concurrent += current_concurrent * (e.first - last_time);
        }
        current_concurrent += e.second;
        max_concurrent = std::max(max_concurrent, current_concurrent);
        last_time = e.first;
    }
    avg_concurrent /= total_wall_time;

    // Print results
    printf("\n====================================\n");
    printf("Multi-Stream Scheduler Benchmark\n");
    printf("====================================\n");
    printf("Configuration:\n");
    printf("  Streams: %d\n", config.num_streams);
    printf("  Kernels per stream: %d\n", config.num_kernels_per_stream);
    printf("  Total kernels launched: %d\n", total_kernels);
    printf("  Workload size: %d elements\n", config.workload_size);
    printf("  Kernel type: %s\n",
           config.kernel_type == COMPUTE ? "compute" :
           config.kernel_type == MEMORY ? "memory" :
           config.kernel_type == GEMM ? "gemm" : "mixed");
    printf("\nResults:\n");
    printf("  Total execution time: %.2f ms\n", total_wall_time);
    printf("  Aggregate throughput: %.2f kernels/sec\n", throughput);
    printf("\nLatency Metrics:\n");
    printf("  Mean latency: %.2f ms\n", avg_latency);
    printf("  P50 latency: %.2f ms\n", p50);
    printf("  P95 latency: %.2f ms\n", p95);
    printf("  P99 latency: %.2f ms\n", p99);
    printf("  Avg launch latency: %.4f ms\n", avg_launch_latency);
    printf("  Max launch latency: %.4f ms\n", max_launch_latency);
    printf("\nScheduler Metrics:\n");
    printf("  Concurrent execution rate: %.1f%%\n", concurrent_rate);
    printf("  Scheduler overhead: %.1f%%\n", scheduler_overhead);
    printf("  GPU utilization (est): %.1f%%\n", gpu_utilization);
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

    if (config.enable_priorities) {
        printf("\nPriority Metrics:\n");
        printf("  Priority inversions detected: %d\n", priority_inversions);
    }
    printf("====================================\n\n");

    // CSV output for easy parsing
    printf("CSV: streams,kernels_per_stream,total_kernels,type,wall_time_ms,throughput,mean_lat,p50,p95,p99,");
    printf("concurrent_rate,overhead,util,jains_index,max_concurrent,avg_concurrent,inversions,working_set_mb,fits_in_l2,stddev,grid_size,block_size,priority_enabled\n");
    printf("CSV: %d,%d,%d,%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.1f,%.1f,%.1f,%.4f,%d,%.2f,%d,%.2f,%d,%.2f,%d,%d,%d\n",
           config.num_streams, config.num_kernels_per_stream, total_kernels,
           config.kernel_type == COMPUTE ? "compute" :
           config.kernel_type == MEMORY ? "memory" :
           config.kernel_type == GEMM ? "gemm" : "mixed",
           total_wall_time, throughput, avg_latency, p50, p95, p99,
           concurrent_rate, scheduler_overhead, gpu_utilization,
           jains_index, max_concurrent, avg_concurrent, priority_inversions,
           working_set_mb, fits_in_l2 ? 1 : 0, stddev, grid_x, block_x,
           config.enable_priorities ? 1 : 0);

    // Output detailed per-kernel data if priorities are enabled
    if (config.enable_priorities) {
        printf("DETAIL_CSV_HEADER: stream_id,kernel_id,priority,start_ms,end_ms,duration_ms,latency_ms\n");
        for (const auto& t : timings) {
            printf("DETAIL_CSV: %d,%d,%d,%.3f,%.3f,%.3f,%.3f\n",
                   t.stream_id, t.kernel_id, t.priority,
                   t.start_time_ms, t.end_time_ms,
                   t.duration_ms, t.launch_latency_ms);
        }
    }
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
 * Compute per-priority-class latency statistics
 *
 * Groups kernels by priority class and computes average latency for each.
 * This helps determine if priority affects execution performance.
 *
 * @param timings Vector of kernel timing data
 * @param config Benchmark configuration
 */
inline void compute_priority_class_latency(const std::vector<KernelTiming> &timings,
                                          const BenchmarkConfig &config) {
    if (!config.enable_priorities || timings.empty()) {
        return;
    }

    // Group by priority
    std::map<int, std::vector<float>> priority_durations;
    for (const auto &t : timings) {
        priority_durations[t.priority].push_back(t.duration_ms);
    }

    printf("\nPer-Priority-Class Latency:\n");
    for (const auto &pair : priority_durations) {
        int priority = pair.first;
        const std::vector<float> &durations = pair.second;

        float sum = 0.0f;
        for (float d : durations) {
            sum += d;
        }
        float avg = sum / durations.size();

        // Compute standard deviation
        float variance = 0.0f;
        for (float d : durations) {
            variance += (d - avg) * (d - avg);
        }
        float stddev = sqrtf(variance / durations.size());

        printf("  Priority %d: %.3f ms (±%.3f ms, n=%zu)\n",
               priority, avg, stddev, durations.size());
    }
}

/**
 * Output detailed per-kernel timing data for RQ3 analysis
 *
 * Outputs CSV format with per-kernel details including priority,
 * start/end times, duration, and launch latency.
 *
 * @param timings Vector of kernel timing data
 */
inline void output_rq3_detailed_csv(const std::vector<KernelTiming> &timings) {
    printf("DETAIL_CSV_HEADER: stream_id,kernel_id,priority,start_ms,end_ms,duration_ms,latency_ms\n");
    for (const auto& t : timings) {
        printf("DETAIL_CSV: %d,%d,%d,%.3f,%.3f,%.3f,%.3f\n",
               t.stream_id, t.kernel_id, t.priority,
               t.start_time_ms, t.end_time_ms,
               t.duration_ms, t.launch_latency_ms);
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
    if (!config.enable_priorities) {
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
