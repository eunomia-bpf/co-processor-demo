// policy.cu
// CLC Policy implementations for nvJitLink framework

#include <cuda_runtime.h>

// ============================================================================
// CLC Policy: MaxStealsPolicy
// Stop after N executions (similar to max steals in CLC framework)
// ============================================================================

struct MaxStealsPolicy_State {
    int executions_done;
    static constexpr int max_executions = 8;
};

extern "C" __device__ void MaxStealsPolicy_init(void* state_ptr) {
    MaxStealsPolicy_State* s = (MaxStealsPolicy_State*)state_ptr;
    s->executions_done = 0;
}

extern "C" __device__ bool MaxStealsPolicy_should_try(void* state_ptr, int current_block) {
    MaxStealsPolicy_State* s = (MaxStealsPolicy_State*)state_ptr;
    bool can_execute = s->executions_done < MaxStealsPolicy_State::max_executions;
    if (can_execute) {
        s->executions_done++;
    }
    return can_execute;
}

// ============================================================================
// CLC Policy: GreedyPolicy
// Always execute (baseline - no throttling)
// ============================================================================

struct GreedyPolicy_State {
    // Empty state - stateless policy
};

extern "C" __device__ void GreedyPolicy_init(void* state_ptr) {
    // No state to initialize
}

extern "C" __device__ bool GreedyPolicy_should_try(void* state_ptr, int current_block) {
    return true;  // Always execute
}

// ============================================================================
// CLC Policy: SelectiveBlocksPolicy
// Only certain blocks execute (e.g., first half)
// ============================================================================

struct SelectiveBlocksPolicy_State {
    int block_id;
    int half_blocks;
};

extern "C" __device__ void SelectiveBlocksPolicy_init(void* state_ptr) {
    SelectiveBlocksPolicy_State* s = (SelectiveBlocksPolicy_State*)state_ptr;
    s->block_id = blockIdx.x;
    s->half_blocks = gridDim.x / 2;
}

extern "C" __device__ bool SelectiveBlocksPolicy_should_try(void* state_ptr, int current_block) {
    SelectiveBlocksPolicy_State* s = (SelectiveBlocksPolicy_State*)state_ptr;
    // Only first half of blocks execute
    return s->block_id < s->half_blocks;
}

// ============================================================================
// Function pointers for policy selection
// ============================================================================

// Function pointer types
typedef void (*policy_init_func_t)(void*);
typedef bool (*policy_decision_func_t)(void*, int);

// Policy function pointers for MaxStealsPolicy
extern "C" __device__ policy_init_func_t d_MaxStealsPolicy_init = MaxStealsPolicy_init;
extern "C" __device__ policy_decision_func_t d_MaxStealsPolicy_should_try = MaxStealsPolicy_should_try;

// Policy function pointers for GreedyPolicy
extern "C" __device__ policy_init_func_t d_GreedyPolicy_init = GreedyPolicy_init;
extern "C" __device__ policy_decision_func_t d_GreedyPolicy_should_try = GreedyPolicy_should_try;

// Policy function pointers for SelectiveBlocksPolicy
extern "C" __device__ policy_init_func_t d_SelectiveBlocksPolicy_init = SelectiveBlocksPolicy_init;
extern "C" __device__ policy_decision_func_t d_SelectiveBlocksPolicy_should_try = SelectiveBlocksPolicy_should_try;
