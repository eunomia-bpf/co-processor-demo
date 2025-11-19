// policy_maxsteals.cuh
// INLINED MaxSteals Policy - Limit stealing to max_steals attempts

#ifndef POLICY_MAXSTEALS_CUH
#define POLICY_MAXSTEALS_CUH

struct MaxStealsPolicy_State {
    int steals_done;
    static constexpr int max_steals = 8;
};

__device__ __forceinline__ void Policy_init(void* state_ptr) {
    MaxStealsPolicy_State* s = (MaxStealsPolicy_State*)state_ptr;
    s->steals_done = 0;
}

__device__ __forceinline__ bool Policy_should_try_steal(void* state_ptr, int current_block) {
    MaxStealsPolicy_State* s = (MaxStealsPolicy_State*)state_ptr;
    bool can_steal = s->steals_done < MaxStealsPolicy_State::max_steals;
    if (can_steal) {
        s->steals_done++;
    }
    return can_steal;
}

#endif // POLICY_MAXSTEALS_CUH
