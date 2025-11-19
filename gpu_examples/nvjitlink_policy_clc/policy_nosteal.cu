// policy_nosteal.cu
// No-Steal Policy: Never attempts to steal work
// Executes the initial assigned block only, then exits
// Useful as a baseline to measure CLC overhead

#include <cuda_runtime.h>

// Policy state structure (not needed for this policy, but kept for consistency)
struct NoStealPolicy_State {
    // No state needed - never steal
};

// Initialize policy state
extern "C" __device__ void Policy_init(void* state_ptr) {
    // No initialization needed
}

// Policy decision function
// Returns false always - never attempt to steal
extern "C" __device__ bool Policy_should_try_steal(void* state_ptr, int current_block) {
    return false;  // Never steal - execute only initial block
}
