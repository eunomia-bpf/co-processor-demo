// policy_greedy.cu
// CLC GreedyPolicy - Always execute (baseline, no throttling)

#include <cuda_runtime.h>

// ============================================================================
// CLC Policy: GreedyPolicy
// Always execute (baseline - mimics default behavior)
// ============================================================================

struct GreedyPolicy_State {
    // Empty state - stateless policy
};

extern "C" __device__ void Policy_init(void* state_ptr) {
    // No state to initialize
}

extern "C" __device__ bool Policy_should_try_steal(void* state_ptr, int current_block) {
    return true;  // Always try to steal - greedy policy
}

// Function pointer types
typedef void (*policy_init_func_t)(void*);
typedef bool (*policy_decision_func_t)(void*, int);

// Policy function pointers
extern "C" __device__ policy_init_func_t d_Policy_init = Policy_init;
extern "C" __device__ policy_decision_func_t d_Policy_should_try_steal = Policy_should_try_steal;
