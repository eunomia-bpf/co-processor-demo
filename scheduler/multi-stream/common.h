#ifndef COMMON_H
#define COMMON_H

#include <vector>
#include "kernels.cuh"

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Configuration structure
struct BenchmarkConfig {
    int num_streams;
    int num_kernels_per_stream;
    int workload_size;
    KernelType kernel_type;
    std::vector<int> priorities_per_stream; // Priority per stream (e.g., -5,-4,-2,0)
    std::vector<int> kernels_per_stream_custom; // For load imbalance experiments
    std::vector<KernelType> kernel_types_per_stream; // For heterogeneous workloads
    std::vector<float> launch_frequency_per_stream; // Launch frequency in Hz per stream (0 = max)
    bool use_heterogeneous; // Enable different kernel types per stream
    std::string csv_output_file; // CSV output file path for raw timing data
    unsigned int random_seed; // Random seed for jitter control

    BenchmarkConfig() : num_streams(4), num_kernels_per_stream(10),
                        workload_size(1048576), kernel_type(MIXED),
                        use_heterogeneous(false), csv_output_file(""),
                        random_seed(0) {}
};

// Timing information per kernel
struct KernelTiming {
    int stream_id;
    int kernel_id;
    int priority;
    KernelType kernel_type;
    float enqueue_time_ms;
    float start_time_ms;
    float end_time_ms;
    float duration_ms;          // Execution time: end_time - start_time
    float launch_latency_ms;    // Queue wait time: start_time - enqueue_time
    float e2e_latency_ms;       // End-to-end latency: end_time - enqueue_time
    // Host-side timestamps (microseconds since epoch)
    long long host_launch_us;   // Host timestamp when kernel launch was initiated
    long long host_sync_us;     // Host timestamp when synchronization completed
};

#endif // COMMON_H
