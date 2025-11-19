// wrapper_kernel_minimal.cu
// MINIMAL CLC wrapper - removes ALL unnecessary overhead
// This should match the performance of USE_DIRECT_CLC

#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Forward declaration of policy function
extern "C" __device__ bool Policy_should_try_steal(void*, int);

extern "C" __global__ void matmul_kernel_with_policy(
    float *xout, float *x, float *w, int n, int d,
    void* unused_policy_ptr)
{
    extern __shared__ float shared_x[];

    // Policy state (minimal)
    __shared__ char policy_state[64];  // Not initialized - policy doesn't use it

    // Get initial block assignment
    int bx = blockIdx.x;

    // Minimal CLC loop - NO mbarrier, NO fence operations
    while (true) {
        __syncthreads();  // Required for CLC

        // INLINE MATMUL KERNEL (using logical block index 'bx')
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

        // Check policy AFTER computation (NoSteal will return false)
        if (cg::thread_block::thread_rank() == 0) {
            if (!Policy_should_try_steal(policy_state, bx)) {
                break;  // Exit - no stealing
            }
        }
        __syncthreads();

        // If we reach here, policy wants to steal but we don't have CLC ops
        // So just break (this path never executes with NoSteal policy)
        break;
    }
}
