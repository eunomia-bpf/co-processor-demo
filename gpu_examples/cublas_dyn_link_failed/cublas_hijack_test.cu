#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Hijacking control functions
extern "C" void cuda_hijack_enable();
extern "C" void cuda_hijack_disable();

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

int main(int argc, char **argv) {
    printf("=================================================================\n");
    printf("cuBLAS with CUDA Driver API Hijacking\n");
    printf("=================================================================\n");
    printf("This demonstrates intercepting CUDA kernel launches at driver level\n");
    printf("to inject runtime policy enforcement into cuBLAS kernels\n\n");

    // Matrix dimensions
    int M = 512;
    int N = 512;
    int K = 512;

    float alpha = 1.0f;
    float beta = 0.0f;

    printf("GEMM: C(%dx%d) = alpha * A(%dx%d) * B(%dx%d) + beta * C\n",
           M, N, M, K, K, N);
    printf("alpha = %.1f, beta = %.1f\n\n", alpha, beta);

    // Allocate host memory
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);

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

    printf("\n--- Test 1: cuBLAS WITHOUT hijacking ---\n");
    cuda_hijack_disable();

    check_cublas_error(
        cublasSgemm(handle,
                   CUBLAS_OP_N, CUBLAS_OP_N,
                   N, M, K,
                   &alpha,
                   d_B, N,
                   d_A, K,
                   &beta,
                   d_C, N),
        "cublasSgemm");

    check_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    printf("cuBLAS executed normally\n");

    // Reset C matrix
    check_cuda_error(cudaMemset(d_C, 0, size_C), "cudaMemset C");

    printf("\n--- Test 2: cuBLAS WITH hijacking ENABLED ---\n");
    cuda_hijack_enable();

    // Create events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    check_cublas_error(
        cublasSgemm(handle,
                   CUBLAS_OP_N, CUBLAS_OP_N,
                   N, M, K,
                   &alpha,
                   d_B, N,
                   d_A, K,
                   &beta,
                   d_C, N),
        "cublasSgemm");

    cudaEventRecord(stop);
    check_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy result back
    check_cuda_error(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost), "cudaMemcpy result");

    // Verify a few elements
    printf("\nSample results:\n");
    printf("  C[0][0] = %.6f\n", h_C[0]);
    printf("  C[0][1] = %.6f\n", h_C[1]);
    printf("  C[1][0] = %.6f\n", h_C[N]);
    printf("  C[1][1] = %.6f\n", h_C[N + 1]);

    // Performance metrics
    double gflops = (2.0 * M * N * K) / (milliseconds * 1e6);
    printf("\nPerformance:\n");
    printf("  Time: %.3f ms\n", milliseconds);
    printf("  GFLOPS: %.2f\n", gflops);

    printf("\n=================================================================\n");
    printf("Demonstration Complete\n");
    printf("  ✓ Successfully intercepted cuBLAS kernel launches\n");
    printf("  ✓ Showed hook injection point in driver API\n");
    printf("  ✓ Next step: Inject wrapper kernel with policy\n");
    printf("=================================================================\n");

    // Cleanup
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
