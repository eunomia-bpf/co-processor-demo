// policy_nosteal.cu
// CLC NoStealPolicy - Execute assigned work once, never steal

#include <cuda_runtime.h>

// ============================================================================
// CLC Policy: NoStealPolicy
// Execute assigned work once, then exit (no work-stealing)
// This is equivalent to traditional CUDA kernel behavior
// ============================================================================

struct NoStealPolicy_State {
    bool executed;  // Track if we've executed our initial work
};

extern "C" __device__ void Policy_init(void* state_ptr) {
    NoStealPolicy_State* s = (NoStealPolicy_State*)state_ptr;
    s->executed = false;
}

extern "C" __device__ bool Policy_should_try_steal(void* state_ptr, int current_block) {
    NoStealPolicy_State* s = (NoStealPolicy_State*)state_ptr;

    if (!s->executed) {
        // First iteration: execute our assigned work
        s->executed = true;
        return false;  // Don't try to steal, just execute once
    }

    // After first execution: never try to steal
    return false;
}

// Function pointer types
typedef void (*policy_init_func_t)(void*);
typedef bool (*policy_decision_func_t)(void*, int);

// Policy function pointers
extern "C" __device__ policy_init_func_t d_Policy_init = Policy_init;
extern "C" __device__ policy_decision_func_t d_Policy_should_try_steal = Policy_should_try_steal;
