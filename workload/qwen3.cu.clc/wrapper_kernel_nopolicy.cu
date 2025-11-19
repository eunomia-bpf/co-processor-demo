// wrapper_kernel_nopolicy.cu
// ABSOLUTE MINIMAL wrapper - NO policy calls at all
// Should match USE_DIRECT_CLC exactly

#include <cuda_runtime.h>

extern "C" __global__ void matmul_kernel_with_policy(
    float *xout, float *x, float *w, int n, int d,
    void* unused_policy_ptr)
{
    extern __shared__ float shared_x[];

    // Get initial block assignment
    int bx = blockIdx.x;

    // Exact same loop as USE_DIRECT_CLC
    while (true) {
        __syncthreads();

        // INLINE MATMUL KERNEL
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

        // NoSteal: execute once then exit
        break;
    }
}
