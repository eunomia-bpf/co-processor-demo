// policy.cu
// Policy device function

#include <cuda_runtime.h>

// Device counter to track how many times policy was called
__device__ int policy_call_count = 0;

// Policy function: increment counter
extern "C" __device__ void apply_policy_impl(int idx) {
    // Atomically increment the global counter
    atomicAdd(&policy_call_count, 1);
}

// Function pointer type
typedef void (*policy_func_t)(int);

// Static device pointer to the policy function
extern "C" __device__ policy_func_t d_apply_policy = apply_policy_impl;
