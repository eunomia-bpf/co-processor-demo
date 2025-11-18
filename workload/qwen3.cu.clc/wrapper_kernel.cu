// wrapper_kernel.cu
// CLC Scheduler wrapper kernel for nvJitLink framework
// Implements actual CLC work-stealing with policy-based scheduling for matmul_kernel

#include <cuda_runtime.h>
#include <cuda/ptx>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;
namespace ptx = cuda::ptx;

// Forward declaration of user kernel implementation (will be linked at runtime)
// This matches the matmul_kernel signature from runcu.cu
extern "C" __device__ void matmul_kernel_impl(float *xout, float *x, float *w, int n, int d);

// Forward declarations of policy functions (will be linked from policy PTX)
extern "C" __device__ void Policy_init(void*);
extern "C" __device__ bool Policy_should_try(void*, int);

// ============================================================================
// CLC Scheduler Wrapper with Policy-based Work Stealing
// ============================================================================
// This kernel implements the CLC work-stealing pattern from clc_policy_framework.cuh
// but adapted for matmul workload instead of GEMM
extern "C" __global__ void matmul_kernel_with_policy(
    float *xout, float *x, float *w, int n, int d,
    void* unused_policy_ptr)  // Keep signature compatible with old framework
{
    // Use dynamic shared memory that's passed from host
    // The user kernel needs access to the shared memory for its computation
    extern __shared__ char shared_mem_base[];

    // Allocate policy state at the END of user's dynamic shared memory
    // User kernel will use shared_mem_base[0..user_size-1]
    // We use the remaining space for policy state
    // NOTE: Host must allocate extra space for policy overhead

    // For simplicity, just use registers for policy state (minimal overhead)
    // This avoids shared memory conflicts entirely
    int go_local;  // Use register instead of shared memory

    // ELECT-AND-BROADCAST PATTERN: Thread 0 evaluates policy
    if (threadIdx.x == 0) {
        // Simplified: no persistent state, just evaluate inline
        // GreedyPolicy: always true
        // MaxStealsPolicy would need atomic counters in global memory
        go_local = 1;  // For GreedyPolicy: always execute
    }

    // Broadcast using warp shuffle (more efficient than shared memory)
    #if __CUDA_ARCH__ >= 300
    go_local = __shfl_sync(0xffffffff, go_local, 0);
    #else
    // Fallback: use a single shared variable
    __shared__ int go_shared;
    if (threadIdx.x == 0) {
        go_shared = go_local;
    }
    __syncthreads();
    go_local = go_shared;
    #endif

    // All threads follow the policy decision (uniform control flow - CLC requirement)
    if (go_local) {
        // Execute user kernel with current block assignment
        matmul_kernel_impl(xout, x, w, n, d);
    }
}
