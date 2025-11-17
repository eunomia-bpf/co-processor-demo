#ifndef SYNTHETIC_CUH
#define SYNTHETIC_CUH

#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <limits>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
        } \
    } while (0)

// Result structure to return performance metrics
struct KernelResult {
    size_t bytes_accessed;  // Total bytes logically accessed
    float median_ms;
    float min_ms;
    float max_ms;
};

// ============================================================================
// Kernel 1: Sequential Stream (read with light computation)
// ============================================================================

__global__ void seq_stream_kernel(const float* input, float* output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < n; i += stride) {
        // Sequential read with simple computation
        float val = input[i];
        // Light computation to avoid pure memory copy
        val = val * 1.5f + 2.0f;
        output[i] = val;
    }
}

inline void run_seq_stream(size_t total_working_set, const std::string& mode, int iterations,
                    std::vector<float>& runtimes, KernelResult& result) {
    // Split working set: input (50%) + output (50%)
    size_t array_bytes = total_working_set / 2;
    size_t n = array_bytes / sizeof(float);

    // Sanity check for device mode
    if (mode == "device") {
        size_t free_bytes, total_bytes;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        if (total_working_set > free_bytes * 0.8) {
            throw std::runtime_error("Working set too large for device memory! Use size_factor < 0.8 for device mode");
        }
    }

    float *input, *output;

    // Allocate memory based on mode
    if (mode == "device") {
        CUDA_CHECK(cudaMalloc(&input, array_bytes));
        CUDA_CHECK(cudaMalloc(&output, array_bytes));
    } else { // uvm or uvm_prefetch
        CUDA_CHECK(cudaMallocManaged(&input, array_bytes));
        CUDA_CHECK(cudaMallocManaged(&output, array_bytes));
    }

    // Initialize input data
    if (mode == "device") {
        std::vector<float> host_data(n, 1.0f);
        CUDA_CHECK(cudaMemcpy(input, host_data.data(), array_bytes, cudaMemcpyHostToDevice));
    } else {
        // For managed memory, initialize on host
        for (size_t i = 0; i < n; ++i) {
            input[i] = 1.0f;
        }
    }

    // Touch pages if requested (for uvm_prefetch mode)
    // Note: CUDA 12.9 changed cudaMemPrefetchAsync API significantly
    // For now, we use simple page touch to trigger initial faults
    if (mode == "uvm_prefetch") {
        // Touch all pages to trigger faults before timing
        volatile float touch = 0.0f;
        for (size_t i = 0; i < n; i += 4096/sizeof(float)) {
            touch += input[i];
        }
        for (size_t i = 0; i < n; i += 4096/sizeof(float)) {
            output[i] = 0.0f;
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Launch configuration
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    // Cap at reasonable number of blocks
    numBlocks = std::min(numBlocks, 1024);

    // Warmup
    for (int i = 0; i < 3; ++i) {
        seq_stream_kernel<<<numBlocks, blockSize>>>(input, output, n);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Timed iterations
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < iterations; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        seq_stream_kernel<<<numBlocks, blockSize>>>(input, output, n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        runtimes.push_back(ms);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Calculate bytes accessed: read input + write output
    result.bytes_accessed = array_bytes * 2;  // Read all input, write all output

    // Compute statistics
    std::sort(runtimes.begin(), runtimes.end());
    result.median_ms = runtimes[runtimes.size() / 2];
    result.min_ms = runtimes.front();
    result.max_ms = runtimes.back();

    // Cleanup
    CUDA_CHECK(cudaFree(input));
    CUDA_CHECK(cudaFree(output));
}

// ============================================================================
// Kernel 2: Random Stream (random access pattern)
// ============================================================================

__global__ void rand_stream_kernel(const float* input, float* output,
                                   const unsigned int* indices, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < n; i += stride) {
        // Random access via index array
        unsigned int access_idx = indices[i] % n;
        float val = input[access_idx];
        val = val * 1.5f + 2.0f;
        output[i] = val;
    }
}

inline void run_rand_stream(size_t total_working_set, const std::string& mode, int iterations,
                     std::vector<float>& runtimes, KernelResult& result) {
    // Split working set: input (40%) + output (40%) + indices (20%)
    size_t input_bytes = static_cast<size_t>(total_working_set * 0.4);
    size_t output_bytes = static_cast<size_t>(total_working_set * 0.4);
    size_t indices_bytes = static_cast<size_t>(total_working_set * 0.2);

    size_t n = input_bytes / sizeof(float);
    size_t n_indices = std::min(n, indices_bytes / sizeof(unsigned int));

    // Limit n to avoid overflow
    if (n > std::numeric_limits<unsigned int>::max() / 2) {
        n = std::numeric_limits<unsigned int>::max() / 2;
    }

    // Ensure n_indices matches n (kernel expects this)
    n_indices = n;
    indices_bytes = n * sizeof(unsigned int);

    // Sanity check for device mode
    if (mode == "device") {
        size_t free_bytes, total_bytes;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        if (total_working_set > free_bytes * 0.8) {
            throw std::runtime_error("Working set too large for device memory! Use size_factor < 0.8 for device mode");
        }
    }

    float *input, *output;
    unsigned int *indices;

    // Allocate memory based on mode
    if (mode == "device") {
        CUDA_CHECK(cudaMalloc(&input, input_bytes));
        CUDA_CHECK(cudaMalloc(&output, output_bytes));
        CUDA_CHECK(cudaMalloc(&indices, indices_bytes));
    } else {
        CUDA_CHECK(cudaMallocManaged(&input, input_bytes));
        CUDA_CHECK(cudaMallocManaged(&output, output_bytes));
        CUDA_CHECK(cudaMallocManaged(&indices, indices_bytes));
    }

    // Initialize data
    if (mode == "device") {
        std::vector<float> host_input(n, 1.0f);
        std::vector<unsigned int> host_indices(n_indices);
        for (size_t i = 0; i < n_indices; ++i) {
            host_indices[i] = static_cast<unsigned int>(rand()) % n;
        }
        CUDA_CHECK(cudaMemcpy(input, host_input.data(), input_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(indices, host_indices.data(), indices_bytes, cudaMemcpyHostToDevice));
    } else {
        for (size_t i = 0; i < n; ++i) {
            input[i] = 1.0f;
        }
        for (size_t i = 0; i < n_indices; ++i) {
            indices[i] = static_cast<unsigned int>(rand()) % n;
        }
    }

    // Touch pages if requested (for uvm_prefetch mode)
    if (mode == "uvm_prefetch") {
        // Touch all pages to trigger faults before timing
        volatile float touch = 0.0f;
        for (size_t i = 0; i < n; i += 4096/sizeof(float)) {
            touch += input[i];
        }
        for (size_t i = 0; i < n; i += 4096/sizeof(float)) {
            output[i] = 0.0f;
        }
        for (size_t i = 0; i < n_indices; i += 4096/sizeof(unsigned int)) {
            indices[i] = indices[i];
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Launch configuration
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    numBlocks = std::min(numBlocks, 1024);

    // Warmup
    for (int i = 0; i < 3; ++i) {
        rand_stream_kernel<<<numBlocks, blockSize>>>(input, output, indices, n);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Timed iterations
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < iterations; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        rand_stream_kernel<<<numBlocks, blockSize>>>(input, output, indices, n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        runtimes.push_back(ms);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Calculate bytes accessed: read input (random) + read indices + write output
    result.bytes_accessed = input_bytes + indices_bytes + output_bytes;

    // Compute statistics
    std::sort(runtimes.begin(), runtimes.end());
    result.median_ms = runtimes[runtimes.size() / 2];
    result.min_ms = runtimes.front();
    result.max_ms = runtimes.back();

    // Cleanup
    CUDA_CHECK(cudaFree(input));
    CUDA_CHECK(cudaFree(output));
    CUDA_CHECK(cudaFree(indices));
}

// ============================================================================
// Kernel 3: Pointer Chase (worst case for TLB/cache)
// ============================================================================

struct Node {
    unsigned int next;  // Index of next node
    float data;         // Payload
    float padding[1];   // Align to 16 bytes
};

__global__ void pointer_chase_kernel(const Node* nodes, float* result, size_t n, int steps) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) {
        unsigned int current = tid;
        float sum = 0.0f;

        // Chase pointers for 'steps' iterations
        for (int i = 0; i < steps; ++i) {
            sum += nodes[current].data;
            current = nodes[current].next;
        }

        result[tid] = sum;
    }
}

inline void run_pointer_chase(size_t total_working_set, const std::string& mode, int iterations,
                       std::vector<float>& runtimes, KernelResult& result) {
    // CRITICAL: Cap pointer_chase size to prevent CPU initialization slowdown
    // Max 128M nodes (~2GB for nodes + result)
    const size_t MAX_NODES = 128 * 1024 * 1024;
    const size_t MAX_WORKING_SET = MAX_NODES * sizeof(Node) + MAX_NODES * sizeof(float);

    if (total_working_set > MAX_WORKING_SET) {
        total_working_set = MAX_WORKING_SET;
    }

    // Split: nodes (90%) + result (10%)
    size_t nodes_bytes = static_cast<size_t>(total_working_set * 0.9);
    size_t result_bytes = static_cast<size_t>(total_working_set * 0.1);

    size_t n = nodes_bytes / sizeof(Node);
    size_t n_result = result_bytes / sizeof(float);
    n = std::min(n, n_result);  // Ensure consistency

    // Ensure n doesn't overflow unsigned int
    if (n > std::numeric_limits<unsigned int>::max() / 2) {
        n = std::numeric_limits<unsigned int>::max() / 2;
    }

    nodes_bytes = n * sizeof(Node);
    result_bytes = n * sizeof(float);

    // Sanity check for device mode
    if (mode == "device") {
        size_t free_bytes, total_bytes;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        if (nodes_bytes + result_bytes > free_bytes * 0.8) {
            throw std::runtime_error("Working set too large for device memory! Use smaller size_factor for pointer_chase");
        }
    }

    Node *nodes;
    float *result_array;

    // Allocate memory based on mode
    if (mode == "device") {
        CUDA_CHECK(cudaMalloc(&nodes, nodes_bytes));
        CUDA_CHECK(cudaMalloc(&result_array, result_bytes));
    } else {
        CUDA_CHECK(cudaMallocManaged(&nodes, nodes_bytes));
        CUDA_CHECK(cudaMallocManaged(&result_array, result_bytes));
    }

    // Initialize linked structure (random chain)
    // Use fast initialization to avoid CPU bottleneck
    if (mode == "device") {
        std::vector<Node> host_nodes(n);
        // Fast parallel-friendly initialization
        #pragma omp parallel for if(n > 1000000)
        for (size_t i = 0; i < n; ++i) {
            host_nodes[i].next = (static_cast<unsigned int>(rand()) ^ i) % n;
            host_nodes[i].data = 1.0f;
            host_nodes[i].padding[0] = 0.0f;
        }
        CUDA_CHECK(cudaMemcpy(nodes, host_nodes.data(), nodes_bytes, cudaMemcpyHostToDevice));
    } else {
        // For UVM, initialize in chunks to avoid long stall
        for (size_t i = 0; i < n; ++i) {
            nodes[i].next = (static_cast<unsigned int>(rand()) ^ i) % n;
            nodes[i].data = 1.0f;
            nodes[i].padding[0] = 0.0f;
        }
    }

    // Touch pages if requested (for uvm_prefetch mode)
    if (mode == "uvm_prefetch") {
        // Touch all pages to trigger faults before timing
        volatile float touch = 0.0f;
        for (size_t i = 0; i < n; i += 4096/sizeof(Node)) {
            touch += nodes[i].data;
        }
        for (size_t i = 0; i < n; i += 4096/sizeof(float)) {
            result_array[i] = 0.0f;
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Launch configuration
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    numBlocks = std::min(numBlocks, 1024);
    int chase_steps = 100;  // Number of pointer chases per thread

    // Warmup
    for (int i = 0; i < 3; ++i) {
        pointer_chase_kernel<<<numBlocks, blockSize>>>(nodes, result_array, n, chase_steps);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Timed iterations
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < iterations; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        pointer_chase_kernel<<<numBlocks, blockSize>>>(nodes, result_array, n, chase_steps);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        runtimes.push_back(ms);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Calculate bytes accessed: each thread does chase_steps reads of Node
    // Total active threads = min(n, numBlocks * blockSize)
    size_t active_threads = std::min(n, static_cast<size_t>(numBlocks * blockSize));
    result.bytes_accessed = active_threads * chase_steps * sizeof(Node);

    // Compute statistics
    std::sort(runtimes.begin(), runtimes.end());
    result.median_ms = runtimes[runtimes.size() / 2];
    result.min_ms = runtimes.front();
    result.max_ms = runtimes.back();

    // Cleanup
    CUDA_CHECK(cudaFree(nodes));
    CUDA_CHECK(cudaFree(result_array));
}

#endif // SYNTHETIC_CUH
