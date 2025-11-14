#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <getopt.h>
#include <vector>
#include <algorithm>
#include <memory>
#include <functional>
#include <thread>
#include <mutex>
#include <map>
#include "common.h"
#include "metrics.h"

// CUDA_CHECK macro now defined in common.h

// RAII wrappers for CUDA resources
struct CudaDeleter {
    void operator()(void* ptr) const {
        if (ptr) cudaFree(ptr);
    }
};

struct CudaEventDeleter {
    void operator()(cudaEvent_t* event) const {
        if (event) {
            cudaEventDestroy(*event);
            delete event;
        }
    }
};

struct CudaStreamDeleter {
    void operator()(cudaStream_t* stream) const {
        if (stream) {
            cudaStreamDestroy(*stream);
            delete stream;
        }
    }
};

template<typename T>
using cuda_unique_ptr = std::unique_ptr<T, CudaDeleter>;

using cuda_event_ptr = std::unique_ptr<cudaEvent_t, CudaEventDeleter>;
using cuda_stream_ptr = std::unique_ptr<cudaStream_t, CudaStreamDeleter>;

// Helper to create CUDA device memory
template<typename T>
cuda_unique_ptr<T> make_cuda_memory(size_t count) {
    T* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
    return cuda_unique_ptr<T>(ptr);
}

// Helper to create CUDA event
cuda_event_ptr make_cuda_event() {
    cudaEvent_t* event = new cudaEvent_t;
    CUDA_CHECK(cudaEventCreate(event));
    return cuda_event_ptr(event);
}

// Helper to create CUDA stream
cuda_stream_ptr make_cuda_stream(int priority = 0, bool use_priority = false) {
    cudaStream_t* stream = new cudaStream_t;
    if (use_priority) {
        CUDA_CHECK(cudaStreamCreateWithPriority(stream, cudaStreamDefault, priority));
    } else {
        CUDA_CHECK(cudaStreamCreate(stream));
    }
    return cuda_stream_ptr(stream);
}

// BenchmarkConfig and KernelTiming now defined in common.h

// Queue depth snapshot
struct QueueSnapshot {
    float time_ms;
    int queued_count;
    int executing_count;
};

// Compute metrics from timing data
// compute_metrics function moved to metrics.h

void print_usage(const char *prog_name) {
    printf("Usage: %s [options]\n", prog_name);
    printf("Options:\n");
    printf("  -s, --streams NUM       Number of CUDA streams (default: 4)\n");
    printf("  -k, --kernels NUM       Kernels per stream (default: 10)\n");
    printf("  -w, --size NUM          Workload size in elements (default: 1048576)\n");
    printf("  -t, --type TYPE         Kernel type: compute|memory|mixed|gemm (default: mixed)\n");
    printf("  -p, --priority          Enable stream priorities\n");
    printf("  -l, --load-imbalance SPEC  Custom kernels per stream (e.g., \"5,10,20,40\")\n");
    printf("  -H, --heterogeneous SPEC   Heterogeneous kernel types (e.g., \"memory,memory,compute,compute\")\n");
    printf("  -h, --help              Show this help message\n");
}

int main(int argc, char **argv) {
    BenchmarkConfig config;

    // Parse command line arguments
    static struct option long_options[] = {
        {"streams", required_argument, 0, 's'},
        {"kernels", required_argument, 0, 'k'},
        {"size", required_argument, 0, 'w'},
        {"type", required_argument, 0, 't'},
        {"priority", no_argument, 0, 'p'},
        {"load-imbalance", required_argument, 0, 'l'},
        {"heterogeneous", required_argument, 0, 'H'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "s:k:w:t:pl:H:h", long_options, NULL)) != -1) {
        switch (opt) {
            case 's':
                config.num_streams = atoi(optarg);
                break;
            case 'k':
                config.num_kernels_per_stream = atoi(optarg);
                break;
            case 'w':
                config.workload_size = atoi(optarg);
                break;
            case 't':
                if (strcmp(optarg, "compute") == 0) config.kernel_type = COMPUTE;
                else if (strcmp(optarg, "memory") == 0) config.kernel_type = MEMORY;
                else if (strcmp(optarg, "mixed") == 0) config.kernel_type = MIXED;
                else if (strcmp(optarg, "gemm") == 0) config.kernel_type = GEMM;
                else {
                    fprintf(stderr, "Invalid kernel type: %s\n", optarg);
                    return 1;
                }
                break;
            case 'p':
                config.enable_priorities = true;
                break;
            case 'l':
                {
                    // Parse comma-separated list of kernels per stream
                    char *token = strtok(optarg, ",");
                    while (token != NULL) {
                        config.kernels_per_stream_custom.push_back(atoi(token));
                        token = strtok(NULL, ",");
                    }
                    // Override num_streams if custom specified
                    if (!config.kernels_per_stream_custom.empty()) {
                        config.num_streams = config.kernels_per_stream_custom.size();
                    }
                }
                break;
            case 'H':
                {
                    // Parse comma-separated list of kernel types per stream
                    config.use_heterogeneous = true;
                    char *token = strtok(optarg, ",");
                    while (token != NULL) {
                        KernelType ktype;
                        if (strcmp(token, "compute") == 0) ktype = COMPUTE;
                        else if (strcmp(token, "memory") == 0) ktype = MEMORY;
                        else if (strcmp(token, "mixed") == 0) ktype = MIXED;
                        else if (strcmp(token, "gemm") == 0) ktype = GEMM;
                        else {
                            fprintf(stderr, "Invalid kernel type in heterogeneous spec: %s\n", token);
                            return 1;
                        }
                        config.kernel_types_per_stream.push_back(ktype);
                        token = strtok(NULL, ",");
                    }
                    // Override num_streams
                    if (!config.kernel_types_per_stream.empty()) {
                        config.num_streams = config.kernel_types_per_stream.size();
                    }
                }
                break;
            case 'h':
                print_usage(argv[0]);
                return 0;
            default:
                print_usage(argv[0]);
                return 1;
        }
    }

    // Initialize CUDA
    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Running on: %s\n", prop.name);

    // Allocate device memory using RAII
    std::vector<cuda_unique_ptr<float>> d_data;
    std::vector<cuda_unique_ptr<float>> d_temp;
    std::vector<cuda_unique_ptr<float>> d_matrix_c;

    d_data.reserve(config.num_streams);
    d_temp.reserve(config.num_streams);
    d_matrix_c.reserve(config.num_streams);

    for (int i = 0; i < config.num_streams; i++) {
        d_data.emplace_back(make_cuda_memory<float>(config.workload_size));
        d_temp.emplace_back(make_cuda_memory<float>(config.workload_size));
        CUDA_CHECK(cudaMemset(d_data[i].get(), 0, config.workload_size * sizeof(float)));

        // Allocate result matrix for GEMM
        if (config.kernel_type == GEMM) {
            d_matrix_c.emplace_back(make_cuda_memory<float>(config.workload_size));
        } else {
            d_matrix_c.emplace_back(nullptr);
        }
    }

    // Create streams with optional priorities using RAII
    std::vector<cuda_stream_ptr> streams;
    std::vector<int> stream_priorities;

    streams.reserve(config.num_streams);
    stream_priorities.reserve(config.num_streams);

    if (config.enable_priorities) {
        int least_priority, greatest_priority;
        CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));
        printf("Stream priority range: %d (high) to %d (low)\n", greatest_priority, least_priority);

        for (int i = 0; i < config.num_streams; i++) {
            // Assign priorities: distribute evenly across range
            int priority = greatest_priority +
                          (i * (least_priority - greatest_priority)) / (config.num_streams - 1);
            stream_priorities.push_back(priority);
            streams.emplace_back(make_cuda_stream(priority, true));
            printf("Stream %d created with priority %d\n", i, priority);
        }
    } else {
        for (int i = 0; i < config.num_streams; i++) {
            stream_priorities.push_back(0);
            streams.emplace_back(make_cuda_stream());
        }
    }

    // Create events for timing using RAII
    std::vector<cuda_event_ptr> start_events;
    std::vector<cuda_event_ptr> end_events;
    std::vector<KernelTiming> timings;

    // Calculate total kernels (accounting for custom load imbalance)
    int total_kernels;
    if (!config.kernels_per_stream_custom.empty()) {
        total_kernels = 0;
        for (int k : config.kernels_per_stream_custom) {
            total_kernels += k;
        }
    } else {
        total_kernels = config.num_streams * config.num_kernels_per_stream;
    }
    start_events.reserve(total_kernels);
    end_events.reserve(total_kernels);

    for (int i = 0; i < total_kernels; i++) {
        start_events.emplace_back(make_cuda_event());
        end_events.emplace_back(make_cuda_event());
    }

    // Launch configuration
    dim3 block(256);
    dim3 grid((config.workload_size + block.x - 1) / block.x);

    // Warmup
    for (int s = 0; s < config.num_streams; s++) {
        launch_kernel(config.kernel_type, d_data[s].get(), d_temp[s].get(),
                     config.workload_size, grid, block, *streams[s], d_matrix_c[s].get());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Create additional events for enqueue time tracking
    auto global_start = make_cuda_event();
    CUDA_CHECK(cudaEventRecord(*global_start, 0));
    CUDA_CHECK(cudaEventSynchronize(*global_start));

    // Main benchmark: launch kernels across all streams using multi-threading
    // Each stream gets its own host thread for truly concurrent launching

    // Determine kernels per stream (either uniform or custom load imbalance)
    std::vector<int> kernels_to_launch(config.num_streams);
    if (!config.kernels_per_stream_custom.empty()) {
        // Custom load imbalance
        for (int s = 0; s < config.num_streams; s++) {
            kernels_to_launch[s] = config.kernels_per_stream_custom[s];
        }
    } else {
        // Uniform load
        for (int s = 0; s < config.num_streams; s++) {
            kernels_to_launch[s] = config.num_kernels_per_stream;
        }
    }

    // Pre-allocate per-stream data structures to avoid thread contention
    std::vector<std::vector<KernelTiming>> per_stream_timings(config.num_streams);
    std::vector<std::vector<cuda_event_ptr>> per_stream_enqueue_events(config.num_streams);
    std::vector<std::vector<int>> per_stream_event_indices(config.num_streams);

    // Pre-calculate event indices for each stream
    int cumulative_idx = 0;
    for (int s = 0; s < config.num_streams; s++) {
        per_stream_timings[s].reserve(kernels_to_launch[s]);
        per_stream_enqueue_events[s].reserve(kernels_to_launch[s]);
        per_stream_event_indices[s].reserve(kernels_to_launch[s]);

        for (int k = 0; k < kernels_to_launch[s]; k++) {
            per_stream_event_indices[s].push_back(cumulative_idx++);
        }
    }

    // Launch threads - one thread per stream for truly concurrent launching
    std::vector<std::thread> launch_threads;

    for (int s = 0; s < config.num_streams; s++) {
        launch_threads.emplace_back([&, s]() {
            // Determine kernel type for this stream
            KernelType current_kernel_type = config.kernel_type;
            if (config.use_heterogeneous && s < (int)config.kernel_types_per_stream.size()) {
                current_kernel_type = config.kernel_types_per_stream[s];
            }

            // Launch all kernels for this stream
            for (int k = 0; k < kernels_to_launch[s]; k++) {
                int event_idx = per_stream_event_indices[s][k];

                // Create and record enqueue event
                auto enqueue_event = make_cuda_event();
                CUDA_CHECK(cudaEventRecord(*enqueue_event, 0));
                per_stream_enqueue_events[s].push_back(std::move(enqueue_event));

                // Record kernel start event
                CUDA_CHECK(cudaEventRecord(*start_events[event_idx], *streams[s]));

                // Launch kernel to this stream
                launch_kernel(current_kernel_type, d_data[s].get(), d_temp[s].get(),
                             config.workload_size, grid, block, *streams[s], d_matrix_c[s].get());

                // Record kernel end event
                CUDA_CHECK(cudaEventRecord(*end_events[event_idx], *streams[s]));

                // Create timing record
                KernelTiming timing;
                timing.stream_id = s;
                timing.kernel_id = k;
                timing.priority = stream_priorities[s];
                timing.kernel_type = current_kernel_type;

                per_stream_timings[s].push_back(timing);
            }

            // Synchronize this stream when all its kernels are launched
            CUDA_CHECK(cudaStreamSynchronize(*streams[s]));
        });
    }

    // Wait for all launch threads to complete
    for (auto& thread : launch_threads) {
        thread.join();
    }

    // Merge per-stream timings into global timings vector
    for (int s = 0; s < config.num_streams; s++) {
        for (size_t k = 0; k < per_stream_timings[s].size(); k++) {
            timings.push_back(per_stream_timings[s][k]);
        }
    }

    // Compute enqueue times for all events
    for (int s = 0; s < config.num_streams; s++) {
        for (size_t k = 0; k < per_stream_enqueue_events[s].size(); k++) {
            int event_idx = per_stream_event_indices[s][k];
            CUDA_CHECK(cudaEventElapsedTime(&timings[event_idx].enqueue_time_ms,
                                            *global_start, *per_stream_enqueue_events[s][k]));
        }
    }

    // Calculate timings
    cudaEvent_t first_event = *start_events[0];
    for (size_t i = 0; i < start_events.size(); i++) {
        CUDA_CHECK(cudaEventElapsedTime(&timings[i].start_time_ms,
                                        first_event, *start_events[i]));
        CUDA_CHECK(cudaEventElapsedTime(&timings[i].end_time_ms,
                                        first_event, *end_events[i]));
        timings[i].duration_ms = timings[i].end_time_ms - timings[i].start_time_ms;
        timings[i].launch_latency_ms = timings[i].start_time_ms - timings[i].enqueue_time_ms;
    }

    // Compute and print metrics
    compute_metrics(timings, config, grid.x, block.x);

    // All cleanup is automatic via RAII smart pointers

    return 0;
}
