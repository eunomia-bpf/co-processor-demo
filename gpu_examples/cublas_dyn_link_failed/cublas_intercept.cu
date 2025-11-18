#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Policy kernel to apply after cuBLAS
__global__ void policy_upper_triangle_zero_kernel(float* C, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    // Zero upper triangle
    if (col > row) {
        C[row * N + col] = 0.0f;
    }
}

void check_cuda_error(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error at %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void check_cublas_error(cublasStatus_t status, const char* msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS Error at %s: %d\n", msg, status);
        exit(EXIT_FAILURE);
    }
}

// Wrapper function that calls cuBLAS + applies policy
void cublas_sgemm_with_policy(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float *alpha,
    const float *A, int lda,
    const float *B, int ldb,
    const float *beta,
    float *C, int ldc)
{
    printf("\n=== cuBLAS SGEMM with Policy Interception ===\n");

    // 1. Call real cuBLAS GEMM
    printf("Step 1: Calling real cuBLAS cublasSgemm...\n");
    check_cublas_error(
        cublasSgemm(handle, transa, transb, m, n, k,
                   alpha, A, lda, B, ldb, beta, C, ldc),
        "cublasSgemm");

    // 2. Apply policy kernel
    printf("Step 2: Applying policy (zero upper triangle)...\n");
    dim3 blockDim(16, 16);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x,
                 (m + blockDim.y - 1) / blockDim.y);

    // For row-major C matrix of size m x n
    policy_upper_triangle_zero_kernel<<<gridDim, blockDim>>>(C, m, n);
    check_cuda_error(cudaGetLastError(), "policy kernel launch");
    check_cuda_error(cudaDeviceSynchronize(), "policy kernel execution");

    printf("Step 3: Policy applied successfully!\n");
}

int main(int argc, char **argv) {
    // Matrix dimensions
    int M = 512;  // Rows of A and C
    int N = 512;  // Columns of B and C
    int K = 512;  // Columns of A, Rows of B

    float alpha = 1.0f;
    float beta = 0.0f;

    printf("=================================================================\n");
    printf("Real cuBLAS GEMM with Runtime Policy Enforcement\n");
    printf("=================================================================\n");
    printf("This uses REAL cuBLAS library and applies policy post-execution\n\n");

    printf("GEMM: C(%dx%d) = alpha * A(%dx%d) * B(%dx%d) + beta * C\n",
           M, N, M, K, K, N);
    printf("alpha = %.1f, beta = %.1f\n", alpha, beta);
    printf("Policy: Zero upper triangle (col > row)\n\n");

    // Allocate host memory
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);
    float *h_C_ref = (float*)malloc(size_C);

    // Initialize matrices
    srand(42);
    for (int i = 0; i < M * K; i++) {
        h_A[i] = (float)(rand() % 100) / 100.0f;
    }
    for (int i = 0; i < K * N; i++) {
        h_B[i] = (float)(rand() % 100) / 100.0f;
    }
    for (int i = 0; i < M * N; i++) {
        h_C[i] = 0.0f;
        h_C_ref[i] = 0.0f;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    check_cuda_error(cudaMalloc(&d_A, size_A), "cudaMalloc A");
    check_cuda_error(cudaMalloc(&d_B, size_B), "cudaMalloc B");
    check_cuda_error(cudaMalloc(&d_C, size_C), "cudaMalloc C");

    // Copy data to device
    check_cuda_error(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice), "cudaMemcpy A");
    check_cuda_error(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice), "cudaMemcpy B");
    check_cuda_error(cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice), "cudaMemcpy C");

    // Create cuBLAS handle
    cublasHandle_t handle;
    check_cublas_error(cublasCreate(&handle), "cublasCreate");

    // Create events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Call cuBLAS with policy enforcement
    // cuBLAS uses column-major, so we transpose: C^T = B^T * A^T for row-major
    cublas_sgemm_with_policy(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        d_B, N,
        d_A, K,
        &beta,
        d_C, N
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy result back
    check_cuda_error(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost), "cudaMemcpy result");

    // Verify result
    printf("\n=== Verifying Results ===\n");
    bool correct = true;
    int errors = 0;
    const int max_errors = 10;

    for (int i = 0; i < M && errors < max_errors; i++) {
        for (int j = 0; j < N && errors < max_errors; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += h_A[i * K + k] * h_B[k * N + j];
            }
            h_C_ref[i * N + j] = alpha * sum + beta * h_C_ref[i * N + j];

            // Apply policy to reference: zero upper triangle
            if (j > i) {
                h_C_ref[i * N + j] = 0.0f;
            }

            float diff = fabs(h_C[i * N + j] - h_C_ref[i * N + j]);
            if (diff > 1e-2) {  // Slightly higher tolerance for cuBLAS
                if (errors < max_errors) {
                    printf("Mismatch at C[%d][%d]: GPU=%.6f, CPU=%.6f, diff=%.6f\n",
                           i, j, h_C[i * N + j], h_C_ref[i * N + j], diff);
                    errors++;
                }
                correct = false;
            }
        }
    }

    if (correct) {
        printf("✓ Results verified successfully!\n");
        printf("  Policy correctly enforced on REAL cuBLAS kernel!\n");
    } else {
        printf("✗ Found %d+ errors\n", errors);
    }

    // Performance metrics
    double gflops = (2.0 * M * N * K) / (milliseconds * 1e6);
    printf("\nPerformance:\n");
    printf("  Time: %.3f ms\n", milliseconds);
    printf("  GFLOPS: %.2f\n", gflops);

    printf("\n=================================================================\n");
    printf("Key Points:\n");
    printf("  ✓ Uses REAL cuBLAS library (not simulation)\n");
    printf("  ✓ Policy applied immediately after cuBLAS execution\n");
    printf("  ✓ Demonstrates practical policy enforcement approach\n");
    printf("  ✓ Works without extracting/modifying cuBLAS internals\n");
    printf("=================================================================\n");

    // Cleanup
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
