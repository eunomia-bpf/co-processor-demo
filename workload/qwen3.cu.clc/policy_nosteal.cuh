// policy_nosteal.cuh
// INLINED NoSteal Policy - Execute assigned work once, never steal

#ifndef POLICY_NOSTEAL_CUH
#define POLICY_NOSTEAL_CUH

struct NoStealPolicy_State {
    bool executed;
};

__device__ __forceinline__ void Policy_init(void* state_ptr) {
    NoStealPolicy_State* s = (NoStealPolicy_State*)state_ptr;
    s->executed = false;
}

__device__ __forceinline__ bool Policy_should_try_steal(void* state_ptr, int current_block) {
    NoStealPolicy_State* s = (NoStealPolicy_State*)state_ptr;

    if (!s->executed) {
        s->executed = true;
        return false;  // Don't try to steal, just execute once
    }

    return false;
}

#endif // POLICY_NOSTEAL_CUH
