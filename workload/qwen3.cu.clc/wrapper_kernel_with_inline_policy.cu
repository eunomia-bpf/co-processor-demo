// wrapper_kernel.cu
// CLC Scheduler wrapper with INLINED policy support
// Policy is selected via compile-time flag: -DPOLICY_NOSTEAL, -DPOLICY_MAXSTEALS, or -DPOLICY_GREEDY
// Default: POLICY_MAXSTEALS

#include <cuda_runtime.h>
#include <cuda/ptx>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;
namespace ptx = cuda::ptx;

// ============================================================================
// Include policy based on compile-time flag
// ============================================================================
#if defined(POLICY_NOSTEAL)
    #include "policy_nosteal.cuh"
#elif defined(POLICY_GREEDY)
    #include "policy_greedy.cuh"
#else
    // Default to MaxSteals if no policy specified
    #include "policy_maxsteals.cuh"
#endif

// ============================================================================
// CLC Scheduler Wrapper with INLINED Policy
// ============================================================================

extern "C" __global__ void matmul_kernel_with_policy(
    float *xout, float *x, float *w, int n, int d,
    void* unused_policy_ptr)
{
    // CLC hardware state
    __shared__ uint4 result;
    __shared__ uint64_t bar;
    int phase = 0;

    // Policy state (will be inlined)
    __shared__ char policy_state[64];
    __shared__ int go;

    // Dynamic shared memory for user kernel computation
    extern __shared__ float shared_x[];

    // Initialize the scheduler policy (INLINED - no function call overhead!)
    if (cg::thread_block::thread_rank() == 0) {
        Policy_init(policy_state);
    }
    __syncthreads();

    // Initialize CLC barrier
    if (cg::thread_block::thread_rank() == 0) {
        ptx::mbarrier_init(&bar, 1);
    }
    __syncthreads();

    // Get initial block assignment (1D grid for matmul)
    int bx = blockIdx.x;

    // CLC work-stealing loop
    while (true) {
        __syncthreads();

        // ELECT-AND-BROADCAST PATTERN: Thread 0 evaluates policy (INLINED!)
        if (cg::thread_block::thread_rank() == 0) {
            go = Policy_should_try_steal(policy_state, bx) ? 1 : 0;
        }
        __syncthreads();

        // ========================================================================
        // INLINE MATMUL KERNEL COMPUTATION
        // ========================================================================
        int i = bx * blockDim.x + threadIdx.x;
        int tid = threadIdx.x;

        // Load x into shared memory in chunks
        for (int offset = 0; offset < n; offset += blockDim.x) {
            if (offset + tid < n) {
                shared_x[tid] = x[offset + tid];
            }
            __syncthreads();

            if (i < d) {
                float sum = 0.0f;
                int chunk_size = min(blockDim.x, n - offset);

                // Vectorized loads and computation
                float4 *w_vec = (float4*)(w + i * n + offset);
                float4 *x_vec = (float4*)shared_x;

                int vec_ops = chunk_size / 4;
                for (int v = 0; v < vec_ops; v++) {
                    float4 w4 = w_vec[v];
                    float4 x4 = x_vec[v];
                    sum += w4.x * x4.x + w4.y * x4.y + w4.z * x4.z + w4.w * x4.w;
                }

                // Handle remaining elements
                for (int j = vec_ops * 4; j < chunk_size; j++) {
                    sum += w[i * n + offset + j] * shared_x[j];
                }

                if (offset == 0) xout[i] = sum;
                else xout[i] += sum;
            }
            __syncthreads();
        }
        // ========================================================================
        // END INLINE MATMUL KERNEL
        // ========================================================================

        // Check if policy decided to stop
        if (!go) {
            break;  // Policy decided to stop
        }

        // CLC steal operations
        if (cg::thread_block::thread_rank() == 0) {
            ptx::fence_proxy_async_generic_sync_restrict(ptx::sem_acquire, ptx::space_cluster, ptx::scope_cluster);
            cg::invoke_one(cg::coalesced_threads(), [&](){
                ptx::clusterlaunchcontrol_try_cancel(&result, &bar);
            });
            ptx::mbarrier_arrive_expect_tx(ptx::sem_relaxed, ptx::scope_cta, ptx::space_shared, &bar, sizeof(uint4));
        }

        // Wait for CLC steal result
        while (!ptx::mbarrier_try_wait_parity(ptx::sem_acquire, ptx::scope_cta, &bar, phase)) {}
        phase ^= 1;

        // Check if steal was successful
        bool success = ptx::clusterlaunchcontrol_query_cancel_is_canceled(result);
        if (!success) {
            break;  // CLC failure - must exit immediately
        }

        // Get new 1D block assignment from CLC
        bx = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_x<int>(result);

        ptx::fence_proxy_async_generic_sync_restrict(ptx::sem_release, ptx::space_shared, ptx::scope_cluster);
    }
}
