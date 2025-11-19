// policy_maxsteals.cu
// CLC MaxStealsPolicy - Stop after N executions

#include <cuda_runtime.h>

// ============================================================================
// CLC Policy: MaxStealsPolicy
// Stop after N executions (similar to max steals in CLC framework)
// ============================================================================

struct MaxStealsPolicy_State {
    int executions_done;
    static constexpr int max_executions = 8;
};

extern "C" __device__ void Policy_init(void* state_ptr) {
    MaxStealsPolicy_State* s = (MaxStealsPolicy_State*)state_ptr;
    s->executions_done = 0;
}

extern "C" __device__ bool Policy_should_try_steal(void* state_ptr, int current_block) {
    MaxStealsPolicy_State* s = (MaxStealsPolicy_State*)state_ptr;
    bool can_steal = s->executions_done < MaxStealsPolicy_State::max_executions;
    if (can_steal) {
        s->executions_done++;
    }
    return can_steal;
}

// Function pointer types
typedef void (*policy_init_func_t)(void*);
typedef bool (*policy_decision_func_t)(void*, int);

// Policy function pointers
extern "C" __device__ policy_init_func_t d_Policy_init = Policy_init;
extern "C" __device__ policy_decision_func_t d_Policy_should_try_steal = Policy_should_try_steal;
