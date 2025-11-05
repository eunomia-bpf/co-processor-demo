// AI Workload Functions Header
// Real-world AI inference workload simulations
// Each function represents a specific AI scenario

#ifndef AI_WORKLOADS_CUH
#define AI_WORKLOADS_CUH

#include <cuda_runtime.h>

// ============================================
// Workload Type Tags (for template dispatch)
// ============================================

struct NLPVariableSequence {};
struct DynamicBatching {};
struct SparseAttention {};
struct GraphNeuralNetwork {};
struct MixtureOfExperts {};
struct VideoProcessing {};

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
    int seq_length;
    if (idx % 16 == 0) seq_length = 512;      // 6.25% long sequences
    else if (idx % 8 == 0) seq_length = 256;  // 12.5% medium sequences
    else if (idx % 4 == 0) seq_length = 128;  // 25% medium-short
    else seq_length = 64;                      // 56.25% short sequences

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

#endif // AI_WORKLOADS_CUH
