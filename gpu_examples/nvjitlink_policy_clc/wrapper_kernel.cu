// wrapper_kernel.cu
// CLC Scheduler wrapper kernel with FULL work-stealing for GEMM
// Uses clusterlaunchcontrol APIs with X/Y dimensions for 2D grid

#include <cuda_runtime.h>
#include <cuda/ptx>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;
namespace ptx = cuda::ptx;

// Forward declaration of user kernel implementation (will be linked at runtime)
extern "C" __device__ void gemm_kernel_impl(float *A, float *B, float *C,
                                             int M, int N, int K,
                                             float alpha, float beta);

// Forward declarations of policy functions (will be linked from policy PTX)
extern "C" __device__ void Policy_init(void*);
extern "C" __device__ bool Policy_should_try_steal(void*, int);

// ============================================================================
// CLC Scheduler Wrapper with FULL Work-Stealing (2D Grid Support)
// ============================================================================
// Based on kernel_cluster_launch_control_policy from clc_policy_framework.cuh
// Implements actual CLC work-stealing loop with 2D block index management

extern "C" __global__ void gemm_kernel_with_policy(
    float *A, float *B, float *C,
    int M, int N, int K,
    float alpha, float beta,
    void* unused_policy_ptr)
{
    // CLC hardware state (from clc_policy_framework.cuh)
    __shared__ uint4 result;
    __shared__ uint64_t bar;
    int phase = 0;

    // Framework holds policy state in __shared__ memory
    __shared__ char policy_state[64];
    __shared__ int go;  // Broadcast flag for uniform control flow

    // Initialize the scheduler policy (thread 0 only, then sync)
    if (cg::thread_block::thread_rank() == 0) {
        Policy_init(policy_state);
    }
    __syncthreads();

    // Initialize CLC barrier
    if (cg::thread_block::thread_rank() == 0) {
        ptx::mbarrier_init(&bar, 1);
    }

    // Get initial block assignment (2D grid for GEMM)
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // CLC work-stealing loop (full implementation from clc_policy_framework.cuh)
    while (true) {
        __syncthreads();

        // ELECT-AND-BROADCAST PATTERN: Thread 0 evaluates policy
        if (cg::thread_block::thread_rank() == 0) {
            go = Policy_should_try_steal(policy_state, bx) ? 1 : 0;
        }
        __syncthreads();

        // Thread 0 issues CLC steal request if policy allows
        if (go && cg::thread_block::thread_rank() == 0) {
            ptx::fence_proxy_async_generic_sync_restrict(ptx::sem_acquire, ptx::space_cluster, ptx::scope_cluster);
            cg::invoke_one(cg::coalesced_threads(), [&](){
                ptx::clusterlaunchcontrol_try_cancel(&result, &bar);
            });
            ptx::mbarrier_arrive_expect_tx(ptx::sem_relaxed, ptx::scope_cta, ptx::space_shared, &bar, sizeof(uint4));
        }

        // Execute GEMM using LOGICAL block indices (bx, by) instead of hardware blockIdx
        // Compute matrix indices from logical block assignment
        int row = by * blockDim.y + threadIdx.y;
        int col = bx * blockDim.x + threadIdx.x;

        if (row < M && col < N) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[row * K + k] * B[k * N + col];
            }
            C[row * N + col] = alpha * sum + beta * C[row * N + col];
        }

        // Check if policy decided to stop
        if (!go) {
            break;  // Uniform exit - policy decided to stop
        }

        // Wait for CLC steal result
        while (!ptx::mbarrier_try_wait_parity(ptx::sem_acquire, ptx::scope_cta, &bar, phase)) {}
        phase ^= 1;

        // Check if steal was successful
        bool success = ptx::clusterlaunchcontrol_query_cancel_is_canceled(result);
        if (!success) {
            break;  // CLC failure - must exit immediately
        }

        // Get new 2D block assignment from CLC (using X and Y APIs)
        bx = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_x<int>(result);
        by = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_y<int>(result);

        ptx::fence_proxy_async_generic_sync_restrict(ptx::sem_release, ptx::space_shared, ptx::scope_cluster);
    }
}
