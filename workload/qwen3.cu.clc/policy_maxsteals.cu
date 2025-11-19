// policy_maxsteals.cu
// CLC MaxStealsPolicy - Stop after N work-stealing attempts

#include <cuda_runtime.h>

// ============================================================================
// CLC Policy: MaxStealsPolicy
// Stop after N work-stealing attempts (prevents excessive stealing)
// ============================================================================

struct MaxStealsPolicy_State {
    int steals_done;
    static constexpr int max_steals = 8;
};

extern "C" __device__ void Policy_init(void* state_ptr) {
    MaxStealsPolicy_State* s = (MaxStealsPolicy_State*)state_ptr;
    s->steals_done = 0;
}

extern "C" __device__ bool Policy_should_try_steal(void* state_ptr, int current_block) {
    MaxStealsPolicy_State* s = (MaxStealsPolicy_State*)state_ptr;
    bool can_steal = s->steals_done < MaxStealsPolicy_State::max_steals;
    if (can_steal) {
        s->steals_done++;
    }
    return can_steal;
}

// Function pointer types
typedef void (*policy_init_func_t)(void*);
typedef bool (*policy_decision_func_t)(void*, int);

// Policy function pointers
extern "C" __device__ policy_init_func_t d_Policy_init = Policy_init;
extern "C" __device__ policy_decision_func_t d_Policy_should_try_steal = Policy_should_try_steal;
