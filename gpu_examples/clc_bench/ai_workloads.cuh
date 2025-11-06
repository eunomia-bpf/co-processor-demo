// AI Workload Functions Header
// Real-world AI inference workload simulations
// Each function represents a specific AI scenario
//
// TUNABLE PARAMETERS (passed at runtime via device constant memory):
// - imbalance_scale: Scale imbalance factor (default 1.0, higher = more imbalance)
// - workload_scale: Scale overall workload (default 1.0, higher = more work)

#ifndef AI_WORKLOADS_CUH
#define AI_WORKLOADS_CUH

#include <cuda_runtime.h>

// Device constant memory for runtime-configurable parameters
__constant__ float imbalance_scale = 1.0f;
__constant__ float workload_scale = 1.0f;

// ============================================
// Workload Type Tags (for template dispatch)
// ============================================

struct NLPVariableSequence {};
struct DynamicBatching {};
struct SparseAttention {};
struct GraphNeuralNetwork {};
struct MixtureOfExperts {};
struct VideoProcessing {};

// GEMM workload tags
struct GEMMBalanced {};
struct GEMMImbalanced {};
struct GEMMVariableSize {};

// ============================================
// Prologue: Simulates model weight loading
// ============================================

__device__ inline float compute_prologue(int prologue_complexity) {
    float result = 1.0f;
    for (int i = 0; i < prologue_complexity; i++) {
        result = sqrtf(result * result + 0.001f);
    }
    return result;
}

// ============================================
// Workload 1: Variable Sequence Lengths (NLP)
// Simulates BERT/GPT with variable token lengths
// ============================================

__device__ inline void process_workload(NLPVariableSequence, float* data, int idx, int n, float weight) {
    // Sequence length varies: some sequences are short, some are long
    int base_seq_length;
    if (idx % 16 == 0) base_seq_length = 512;      // 6.25% long sequences
    else if (idx % 8 == 0) base_seq_length = 256;  // 12.5% medium sequences
    else if (idx % 4 == 0) base_seq_length = 128;  // 25% medium-short
    else base_seq_length = 64;                      // 56.25% short sequences

    // Apply scaling factors
    // imbalance_scale: scales the difference between long and short sequences
    // workload_scale: scales overall work
    float imb_factor = (base_seq_length - 64.0f) * imbalance_scale + 64.0f;
    int seq_length = (int)(imb_factor * workload_scale);

    float value = data[idx];
    // Simulate attention computation over sequence
    for (int i = 0; i < seq_length; i++) {
        value = tanhf(value * weight + 0.01f * sinf((float)i));
    }
    data[idx] = value;
}

// ============================================
// Workload 2: Dynamic Batching (AI Inference)
// Simulates serving with variable request complexity
// ============================================

__device__ inline void process_workload(DynamicBatching, float* data, int idx, int n, float weight) {
    // Simulate variable model complexity per request
    int model_ops;
    int request_id = idx % 32;
    if (request_id < 4) model_ops = 200;      // 12.5% complex requests
    else if (request_id < 12) model_ops = 100; // 25% medium requests
    else model_ops = 50;                       // 62.5% simple requests

    float value = data[idx];
    for (int i = 0; i < model_ops; i++) {
        value = value * weight + 0.1f;
        if (i % 10 == 0) value = sqrtf(fabsf(value));
    }
    data[idx] = value;
}

// ============================================
// Workload 3: Sparse Attention (Transformer)
// Simulates sparse attention patterns
// ============================================

__device__ inline void process_workload(SparseAttention, float* data, int idx, int n, float weight) {
    float value = data[idx];

    // Sparse attention pattern: attend to subset of tokens
    int attention_mask = idx % 16;
    int attend_count;

    if (attention_mask < 2) {
        // 12.5%: Full attention (attend to many tokens)
        attend_count = 128;
    } else if (attention_mask < 6) {
        // 25%: Medium attention
        attend_count = 64;
    } else {
        // 62.5%: Local attention only
        attend_count = 32;
    }

    // Compute attention scores
    for (int i = 0; i < attend_count; i++) {
        float score = tanhf(value * weight + (float)i * 0.001f);
        value = value * 0.99f + score * 0.01f;
    }
    data[idx] = value;
}

// ============================================
// Workload 4: Graph Neural Networks (GNN)
// Simulates GNN with variable node degrees
// ============================================

__device__ inline void process_workload(GraphNeuralNetwork, float* data, int idx, int n, float weight) {
    float value = data[idx];

    // Node degree varies significantly in real graphs (power law distribution)
    int degree;
    int node_type = idx % 100;

    if (node_type < 5) {
        // 5%: Hub nodes with many neighbors
        degree = 200;
    } else if (node_type < 20) {
        // 15%: Well-connected nodes
        degree = 100;
    } else if (node_type < 50) {
        // 30%: Moderately connected
        degree = 50;
    } else {
        // 50%: Sparse connections
        degree = 20;
    }

    // Message passing: aggregate from neighbors
    for (int i = 0; i < degree; i++) {
        float neighbor_msg = sinf(value * weight + (float)i);
        value = value * 0.9f + neighbor_msg * 0.1f;
    }
    data[idx] = value;
}

// ============================================
// Workload 5: Mixture of Experts (MoE)
// Simulates MoE routing to different experts
// ============================================

__device__ inline void process_workload(MixtureOfExperts, float* data, int idx, int n, float weight) {
    float value = data[idx];

    // Router selects which expert(s) to use
    int expert_route = (int)(value * 1000.0f) % 8;

    // Different experts have different complexity
    int expert_ops;
    if (expert_route == 0 || expert_route == 7) {
        // 25%: Complex experts
        expert_ops = 150;
    } else if (expert_route < 4) {
        // 50%: Medium experts
        expert_ops = 80;
    } else {
        // 25%: Simple experts
        expert_ops = 40;
    }

    // Expert computation
    for (int i = 0; i < expert_ops; i++) {
        value = tanhf(value * weight + 0.01f * cosf((float)i));
    }
    data[idx] = value;
}

// ============================================
// Workload 6: Video Frame Processing (CV)
// Simulates video processing with variable complexity
// ============================================

__device__ inline void process_workload(VideoProcessing, float* data, int idx, int n, float weight) {
    float value = data[idx];

    // Frame complexity varies: static backgrounds vs motion
    int frame_id = idx % 30; // 30 fps
    int ops;

    if (frame_id < 3) {
        // 10%: Scene changes (high complexity)
        ops = 180;
    } else if (frame_id % 5 == 0) {
        // 20%: Moderate motion
        ops = 100;
    } else {
        // 70%: Static/low motion
        ops = 50;
    }

    // Image processing pipeline
    for (int i = 0; i < ops; i++) {
        value = value * weight + 0.1f * expf(-fabsf(value) * 0.01f);
    }
    data[idx] = value;
}

// ============================================
// Workload Names (for reporting)
// ============================================

template<typename WorkloadType>
__host__ const char* get_workload_name();

template<> __host__ const char* get_workload_name<NLPVariableSequence>() {
    return "NLP: Variable Sequence Lengths (BERT/GPT)";
}

template<> __host__ const char* get_workload_name<DynamicBatching>() {
    return "AI Inference: Dynamic Batching";
}

template<> __host__ const char* get_workload_name<SparseAttention>() {
    return "Transformer: Sparse Attention";
}

template<> __host__ const char* get_workload_name<GraphNeuralNetwork>() {
    return "GNN: Variable Graph Node Degrees";
}

template<> __host__ const char* get_workload_name<MixtureOfExperts>() {
    return "MoE: Mixture of Experts Routing";
}

template<> __host__ const char* get_workload_name<VideoProcessing>() {
    return "CV: Video Frame Processing";
}

template<> __host__ const char* get_workload_name<GEMMBalanced>() {
    return "GEMM: Balanced (16x16x16)";
}

template<> __host__ const char* get_workload_name<GEMMImbalanced>() {
    return "GEMM: Imbalanced (Variable MxNxK)";
}

template<> __host__ const char* get_workload_name<GEMMVariableSize>() {
    return "GEMM: Variable Matrix Sizes";
}

// ============================================
// GEMM Data Structures and Helpers
// ============================================

// Tile sizes for GEMM
#define TILE_SIZE 16

// ============================================
// Workload 7: Balanced GEMM
// Real tiled matrix multiplication (MxNxK with M=N=K=64)
// ============================================

__device__ inline void process_workload(GEMMBalanced, float* data, int idx, int n, float weight) {
    // Real GEMM: C = A * B where matrices are stored in data array
    // Each thread computes multiple elements using tiling
    const int M = 64, N = 64, K = 64;
    const int TILE = 16;

    // Determine which batch/matrix this thread works on
    int num_matrices = n / (M * N + N * K + M * K);  // Rough estimate of how many matrix sets fit
    if (num_matrices < 1) num_matrices = 1;

    int matrix_id = idx / (M * N);
    int elem_in_matrix = idx % (M * N);

    // Calculate row and column for output matrix C
    int c_row = elem_in_matrix / N;
    int c_col = elem_in_matrix % N;

    if (c_row >= M || c_col >= N || idx >= n) return;

    // Base pointers for this matrix triple (A, B, C are laid out sequentially)
    // A is M x K, B is K x N, C is M x N
    int base = (matrix_id * (M * K + K * N + M * N)) % n;
    int a_base = base;
    int b_base = (base + M * K) % n;
    int c_base = (base + M * K + K * N) % n;

    // Compute C[c_row][c_col] = sum over k of A[c_row][k] * B[k][c_col]
    float sum = 0.0f;

    // Tiled computation for better memory access
    for (int tile_k = 0; tile_k < K; tile_k += TILE) {
        // Load tiles from A and B
        for (int k = tile_k; k < tile_k + TILE && k < K; k++) {
            int a_idx = (a_base + c_row * K + k) % n;
            int b_idx = (b_base + k * N + c_col) % n;

            float a_val = data[a_idx] * weight;
            float b_val = data[b_idx];

            sum += a_val * b_val;
        }
    }

    // Write result to C
    int c_idx = (c_base + c_row * N + c_col) % n;
    data[c_idx] = tanhf(sum);
}

// ============================================
// Workload 8: Imbalanced GEMM
// Variable matrix sizes creating load imbalance (real GEMM)
// ============================================

__device__ inline void process_workload(GEMMImbalanced, float* data, int idx, int n, float weight) {
    // Real GEMM with variable sizes creating imbalance
    // Matrix sizes follow power-law distribution
    int size_class = idx % 100;
    int M, N, K;

    if (size_class < 5) {
        // 5%: Large matrices (64x64x64)
        M = N = K = 64;
    } else if (size_class < 20) {
        // 15%: Medium matrices (48x48x48)
        M = N = K = 48;
    } else if (size_class < 50) {
        // 30%: Small-medium matrices (32x32x32)
        M = N = K = 32;
    } else {
        // 50%: Small matrices (16x16x16)
        M = N = K = 16;
    }

    int matrix_id = idx / (M * N);
    int elem_in_matrix = idx % (M * N);

    int c_row = elem_in_matrix / N;
    int c_col = elem_in_matrix % N;

    if (c_row >= M || c_col >= N || idx >= n) return;

    // Base offset for this matrix set
    int base = (matrix_id * (M * K + K * N + M * N)) % n;
    int a_base = base;
    int b_base = (base + M * K) % n;
    int c_base = (base + M * K + K * N) % n;

    // Compute real GEMM: C[row, col] = sum_k A[row, k] * B[k, col]
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        int a_idx = (a_base + c_row * K + k) % n;
        int b_idx = (b_base + k * N + c_col) % n;

        float a_val = data[a_idx] * weight;
        float b_val = data[b_idx];
        sum += a_val * b_val;
    }

    int c_idx = (c_base + c_row * N + c_col) % n;
    data[c_idx] = tanhf(sum);
}

// ============================================
// Workload 9: Variable-Size GEMM
// Batched GEMM with heterogeneous rectangular matrices (real GEMM)
// ============================================

__device__ inline void process_workload(GEMMVariableSize, float* data, int idx, int n, float weight) {
    // Real batched GEMM with different matrix dimensions per batch
    // Simulates attention heads with different sequence lengths
    int batch_id = idx % 16;
    int M, N, K;

    // Different batches have different sizes (rectangular matrices)
    switch (batch_id % 4) {
        case 0:  M = 64; N = 32; K = 48; break;  // Wide matrices
        case 1:  M = 32; N = 64; K = 48; break;  // Tall matrices
        case 2:  M = 48; N = 48; K = 32; break;  // Square-ish
        case 3:  M = 32; N = 32; K = 64; break;  // Deep
    }

    int matrix_id = idx / (M * N);
    int elem_in_matrix = idx % (M * N);

    int c_row = elem_in_matrix / N;
    int c_col = elem_in_matrix % N;

    if (c_row >= M || c_col >= N || idx >= n) return;

    // Base offset for this matrix set
    int base = (matrix_id * (M * K + K * N + M * N)) % n;
    int a_base = base;
    int b_base = (base + M * K) % n;
    int c_base = (base + M * K + K * N) % n;

    // Compute real GEMM: C[row, col] = sum_k A[row, k] * B[k, col]
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        int a_idx = (a_base + c_row * K + k) % n;
        int b_idx = (b_base + k * N + c_col) % n;

        float a_val = data[a_idx] * weight;
        float b_val = data[b_idx];
        sum += a_val * b_val;
    }

    int c_idx = (c_base + c_row * N + c_col) % n;
    data[c_idx] = tanhf(sum);
}

#endif // AI_WORKLOADS_CUH
