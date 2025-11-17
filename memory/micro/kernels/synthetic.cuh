#ifndef SYNTHETIC_CUH
#define SYNTHETIC_CUH

#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <limits>
#include <functional>

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
// Generic timing and statistics template
// ============================================================================

template <typename LaunchFunc>
inline void time_kernel(LaunchFunc launch_kernel, int warmup_iterations, int timed_iterations,
                        std::vector<float>& runtimes, KernelResult& result) {
    // Defensive check for invalid iterations
    if (timed_iterations <= 0) {
        result.median_ms = result.min_ms = result.max_ms = 0.0f;
        return;
    }

    // Warmup
    for (int i = 0; i < warmup_iterations; ++i) {
        launch_kernel();
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Timed iterations
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < timed_iterations; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        launch_kernel();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        runtimes.push_back(ms);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Compute statistics
    std::sort(runtimes.begin(), runtimes.end());
    result.median_ms = runtimes[runtimes.size() / 2];
    result.min_ms = runtimes.front();
    result.max_ms = runtimes.back();
}

// ============================================================================
// Kernel 1: Sequential Stream (read with light computation)
// ============================================================================

__global__ void seq_stream_kernel(const float* input, float* output, size_t n, size_t stride_elems) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t grid_stride = gridDim.x * blockDim.x;

    // stride_elems determines granularity:
    // - stride_elems = 1: access every element (element-level)
    // - stride_elems = page_size/sizeof(float): access one element per page (page-level)
    for (size_t page = tid; page * stride_elems < n; page += grid_stride) {
        size_t i = page * stride_elems;
        // Sequential read with simple computation
        float val = input[i];
        // Light computation to avoid pure memory copy
        val = val * 1.5f + 2.0f;
        output[i] = val;
    }
}

inline void run_seq_stream(size_t total_working_set, const std::string& mode, size_t stride_bytes,
                    int iterations, std::vector<float>& runtimes, KernelResult& result) {
    // Split working set: input (50%) + output (50%)
    size_t array_bytes = total_working_set / 2;
    size_t n = array_bytes / sizeof(float);
    size_t stride_elems = std::max(1UL, stride_bytes / sizeof(float));

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

    // Prefetch to GPU for uvm_prefetch mode
    if (mode == "uvm_prefetch") {
        int dev;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaMemPrefetchAsync(input, array_bytes, dev, 0));
        CUDA_CHECK(cudaMemPrefetchAsync(output, array_bytes, dev, 0));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Launch configuration
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    // Cap at reasonable number of blocks
    numBlocks = std::min(numBlocks, 1024);

    // Use generic timing template
    auto launch = [&]() {
        seq_stream_kernel<<<numBlocks, blockSize>>>(input, output, n, stride_elems);
    };

    time_kernel(launch, /*warmup=*/3, iterations, runtimes, result);

    // Calculate bytes accessed based on stride
    // Number of actual accesses = ceil(n / stride_elems)
    size_t num_accesses = (n + stride_elems - 1) / stride_elems;
    size_t logical_bytes = num_accesses * sizeof(float) * 2;  // read + write

    // For page-level stride, also calculate effective migration bytes
    if (stride_bytes >= 4096) {
        // UVM migrates entire pages, so count page-level traffic
        size_t num_pages = num_accesses;
        result.bytes_accessed = num_pages * 4096 * 2;  // input pages + output pages
    } else {
        // Element-level access: use logical bytes
        result.bytes_accessed = logical_bytes;
    }

    // Cleanup
    CUDA_CHECK(cudaFree(input));
    CUDA_CHECK(cudaFree(output));
}

// ============================================================================
// Kernel 2: Random Stream (random access pattern)
// ============================================================================

__global__ void rand_stream_kernel(const float* input, float* output,
                                   const unsigned int* indices, size_t n, size_t stride_elems) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t grid_stride = gridDim.x * blockDim.x;

    for (size_t page = tid; page * stride_elems < n; page += grid_stride) {
        size_t i = page * stride_elems;
        // Random access via index array
        unsigned int access_idx = indices[i] % n;
        float val = input[access_idx];
        val = val * 1.5f + 2.0f;
        output[i] = val;
    }
}

inline void run_rand_stream(size_t total_working_set, const std::string& mode, size_t stride_bytes,
                     int iterations, std::vector<float>& runtimes, KernelResult& result) {
    // Split working set: input (40%) + output (40%) + indices (20%)
    size_t input_bytes = static_cast<size_t>(total_working_set * 0.4);
    size_t output_bytes = static_cast<size_t>(total_working_set * 0.4);
    size_t indices_bytes = static_cast<size_t>(total_working_set * 0.2);

    size_t n = input_bytes / sizeof(float);
    size_t n_indices = std::min(n, indices_bytes / sizeof(unsigned int));
    size_t stride_elems = std::max(1UL, stride_bytes / sizeof(float));

    // Limit n to avoid overflow
    if (n > std::numeric_limits<unsigned int>::max() / 2) {
        n = std::numeric_limits<unsigned int>::max() / 2;
    }

    // Ensure n_indices matches n (kernel expects this)
    n_indices = n;
    indices_bytes = n * sizeof(unsigned int);

    // Update input/output bytes to match actual n
    input_bytes = n * sizeof(float);
    output_bytes = n * sizeof(float);

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

    // Prefetch to GPU for uvm_prefetch mode
    if (mode == "uvm_prefetch") {
        int dev;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaMemPrefetchAsync(input, input_bytes, dev, 0));
        CUDA_CHECK(cudaMemPrefetchAsync(output, output_bytes, dev, 0));
        CUDA_CHECK(cudaMemPrefetchAsync(indices, indices_bytes, dev, 0));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Launch configuration
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    numBlocks = std::min(numBlocks, 1024);

    // Use generic timing template
    auto launch = [&]() {
        rand_stream_kernel<<<numBlocks, blockSize>>>(input, output, indices, n, stride_elems);
    };

    time_kernel(launch, /*warmup=*/3, iterations, runtimes, result);

    // Calculate bytes accessed based on stride
    size_t num_accesses = (n + stride_elems - 1) / stride_elems;

    if (stride_bytes >= 4096) {
        // Page-level: count UVM migration bytes
        size_t num_pages = num_accesses;
        // Random input access + sequential output + sequential indices
        result.bytes_accessed = num_pages * 4096 * 3;  // input + output + indices pages
    } else {
        // Element-level: count logical bytes
        size_t logical_bytes = num_accesses * (sizeof(float) * 2 + sizeof(unsigned int));
        result.bytes_accessed = logical_bytes;
    }

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

// GPU-based node initialization (faster than CPU for large arrays)
__device__ __forceinline__
unsigned int lcg_hash(unsigned int x) {
    return 1664525u * x + 1013904223u;
}

__global__ void init_nodes_kernel(Node* nodes, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < n; i += stride) {
        unsigned int r = lcg_hash(static_cast<unsigned int>(i));
        nodes[i].next = r % n;
        nodes[i].data = 1.0f;
        nodes[i].padding[0] = 0.0f;
    }
}

// Multi-segment pointer chase kernel for oversubscription testing
// stride_elems: sample every Nth node (for page-level testing)
__global__ void pointer_chase_kernel_segmented(const Node* nodes, float* result,
                                                size_t n_per_seg, int steps, int segments,
                                                size_t stride_elems) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t grid_stride = gridDim.x * blockDim.x;

    // Process each segment
    for (int seg = 0; seg < segments; ++seg) {
        size_t base = seg * n_per_seg;

        // Grid-stride loop with sampling
        for (size_t i = tid * stride_elems; i < n_per_seg; i += grid_stride * stride_elems) {
            unsigned int current = static_cast<unsigned int>(base + i);
            float sum = 0.0f;

            // Chase pointers for 'steps' iterations
            for (int s = 0; s < steps; ++s) {
                sum += nodes[current].data;
                current = nodes[current].next;
            }

            result[base + i] = sum;
        }
    }
}

inline void run_pointer_chase(size_t total_working_set, const std::string& mode, size_t stride_bytes,
                       int iterations, std::vector<float>& runtimes, KernelResult& result) {
    // Multi-segment design for oversubscription:
    // - Each segment has at most MAX_NODES_PER_SEG nodes
    // - Total working set is divided into multiple segments
    // - This allows oversub without huge CPU init cost
    const size_t MAX_NODES_PER_SEG = 32 * 1024 * 1024;  // 32M nodes per segment

    // Split: nodes (90%) + result (10%)
    size_t nodes_bytes = static_cast<size_t>(total_working_set * 0.9);
    size_t result_bytes = static_cast<size_t>(total_working_set * 0.1);

    // Calculate total nodes needed
    size_t total_nodes = nodes_bytes / sizeof(Node);

    // Determine segments - limit to max 16 segments for reasonable runtime
    size_t n_per_seg = std::min(total_nodes, MAX_NODES_PER_SEG);
    int segments = static_cast<int>((total_nodes + n_per_seg - 1) / n_per_seg);
    segments = std::min(segments, 16);  // Cap at 16 segments to keep runtime manageable

    // Actual allocation (may be slightly larger due to rounding up)
    size_t n_alloc = static_cast<size_t>(segments) * n_per_seg;
    nodes_bytes = n_alloc * sizeof(Node);
    result_bytes = n_alloc * sizeof(float);

    // Calculate stride in elements (how many nodes per page to sample)
    size_t stride_elems = std::max(1UL, stride_bytes / sizeof(Node));

    // Sanity check for device mode
    if (mode == "device") {
        size_t free_bytes, total_bytes;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        if (nodes_bytes + result_bytes > free_bytes * 0.8) {
            throw std::runtime_error("pointer_chase working set too large for device mode");
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

    // Use GPU-based initialization (much faster than CPU for large arrays)
    int init_blockSize = 256;
    int init_numBlocks = (n_alloc + init_blockSize - 1) / init_blockSize;
    init_numBlocks = std::min(init_numBlocks, 2048);

    init_nodes_kernel<<<init_numBlocks, init_blockSize>>>(nodes, n_alloc);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Prefetch to GPU for uvm_prefetch mode
    if (mode == "uvm_prefetch") {
        int dev;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaMemPrefetchAsync(nodes, nodes_bytes, dev, 0));
        CUDA_CHECK(cudaMemPrefetchAsync(result_array, result_bytes, dev, 0));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Launch configuration
    int blockSize = 256;
    int numBlocks = (n_per_seg + blockSize - 1) / blockSize;
    numBlocks = std::min(numBlocks, 1024);
    int chase_steps = 4;  // Minimal steps for oversub testing to keep runtime reasonable

    // Use generic timing template with multi-segment kernel
    auto launch = [&]() {
        pointer_chase_kernel_segmented<<<numBlocks, blockSize>>>(
            nodes, result_array, n_per_seg, chase_steps, segments, stride_elems);
    };

    time_kernel(launch, /*warmup=*/2, iterations, runtimes, result);

    // Calculate bytes accessed accounting for stride sampling
    size_t num_samples_per_seg = (n_per_seg + stride_elems - 1) / stride_elems;
    size_t total_accesses = static_cast<size_t>(segments) * num_samples_per_seg * chase_steps;

    if (stride_bytes >= 4096) {
        // Page-level: count UVM migration bytes (each access touches a 4KB page)
        result.bytes_accessed = total_accesses * 4096;
    } else {
        // Element-level: count logical bytes
        result.bytes_accessed = total_accesses * sizeof(Node);
    }

    // Cleanup
    CUDA_CHECK(cudaFree(nodes));
    CUDA_CHECK(cudaFree(result_array));
}

#endif // SYNTHETIC_CUH
