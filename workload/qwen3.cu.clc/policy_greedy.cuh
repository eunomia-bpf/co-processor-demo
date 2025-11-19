// policy_greedy.cuh
// INLINED Greedy Policy - Always try to steal more work

#ifndef POLICY_GREEDY_CUH
#define POLICY_GREEDY_CUH

struct GreedyPolicy_State {
    // No state needed - always steal
};

__device__ __forceinline__ void Policy_init(void* state_ptr) {
    // No initialization needed
}

__device__ __forceinline__ bool Policy_should_try_steal(void* state_ptr, int current_block) {
    return true;  // Always try to steal more work
}

#endif // POLICY_GREEDY_CUH
