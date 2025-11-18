// wrapper_kernel.cu
// Generic wrapper kernel that calls user kernel + policy via function pointers

#include <cuda_runtime.h>

// Function pointer types
typedef void (*KernelFunc)(float*, float*, float*, int, int, int, float, float);
typedef void (*PolicyFunc)(float*, int, int);

// External device function (policy is dynamically loaded)
extern "C" __device__ void policy_upper_triangle_zero(float*, int, int);

// Wrapper kernel: calls user kernel + policy via function pointer
// This version takes a kernel function pointer as first argument
extern "C"
__global__ void run_with_policy_kernel(KernelFunc user_kernel,  // Function pointer to user's kernel
                                       float *A, float *B, float *C,
                                       int M, int N, int K,
                                       float alpha, float beta) {
    // Call user kernel via function pointer (works with any kernel!)
    user_kernel(A, B, C, M, N, K, alpha, beta);

    // Sync threads (ensure all threads complete computation before policy)
    __syncthreads();

    // Apply policy (each thread applies policy to its element)
    policy_upper_triangle_zero(C, M, N);
}

