// wrapper_kernel.cu
// CLC Scheduler wrapper kernel with FULL work-stealing for matmul
// Based on GEMM implementation with 1D grid support

#include <cuda_runtime.h>
#include <cuda/ptx>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;
namespace ptx = cuda::ptx;

// Forward declaration of user kernel implementation (will be linked at runtime)
// NOTE: The user kernel is now a __device__ function that takes logical block index
extern "C" __device__ void matmul_kernel_impl(float *xout, float *x, float *w, int n, int d);

// Forward declarations of policy functions (will be linked from policy PTX)
extern "C" __device__ void Policy_init(void*);
extern "C" __device__ bool Policy_should_try_steal(void*, int);

// ============================================================================
// CLC Scheduler Wrapper with FULL Work-Stealing (1D Grid Support)
// ============================================================================
// Based on kernel_cluster_launch_control_policy from clc_policy_framework.cuh
// Implements actual CLC work-stealing loop with 1D block index management
//
// KEY CHALLENGE: The user kernel (matmul_kernel_impl) uses blockIdx.x internally!
// We CANNOT change blockIdx from within a running kernel, so we need to either:
// 1. Manually compute all indices and pass them to kernel (requires kernel rewrite)
// 2. Use a simpler work-stealing pattern that doesn't reassign blocks
//
// SOLUTION: We'll compute the indices here and inline the kernel logic
// This allows us to use logical block IDs instead of hardware blockIdx

extern "C" __global__ void matmul_kernel_with_policy(
    float *xout, float *x, float *w, int n, int d,
    void* unused_policy_ptr)
{
    // CLC hardware state (from clc_policy_framework.cuh)
    __shared__ uint4 result;
    __shared__ uint64_t bar;
    int phase = 0;

    // Framework holds policy state in __shared__ memory
    __shared__ char policy_state[64];
    __shared__ int go;  // Broadcast flag for uniform control flow

    // Dynamic shared memory for user kernel computation
    extern __shared__ float shared_x[];

    // Initialize CLC barrier
    if (cg::thread_block::thread_rank() == 0) {
        ptx::mbarrier_init(&bar, 1);
    }
    __syncthreads();

    // Get initial block assignment (1D grid for matmul)
    int bx = blockIdx.x;  // Logical block index (can be updated by CLC)

    // CLC work-stealing loop (full implementation from clc_policy_framework.cuh)
    while (true) {
        __syncthreads();

        // ELECT-AND-BROADCAST PATTERN: NoSteal hardcoded (no function call)
        if (cg::thread_block::thread_rank() == 0) {
            go = 0;  // NoSteal: hardcoded, no function call overhead
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

        // ========================================================================
        // INLINE MATMUL KERNEL COMPUTATION (using logical block index 'bx')
        // ========================================================================
        // We inline the kernel instead of calling matmul_kernel_impl because
        // the user kernel uses blockIdx.x internally, which we cannot modify.
        // By inlining, we can substitute blockIdx.x with our logical 'bx'.

        int i = bx * blockDim.x + threadIdx.x;  // Use logical bx instead of blockIdx.x
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

        // Get new 1D block assignment from CLC (using X API only)
        bx = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_x<int>(result);

        ptx::fence_proxy_async_generic_sync_restrict(ptx::sem_release, ptx::space_shared, ptx::scope_cluster);
    }
}
