#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "policy_framework.h"

void check_cuda_error(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error at %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char **argv) {
    printf("========================================\n");
    printf("CUDA 12 nvJitLink Policy Framework Demo\n");
    printf("========================================\n\n");

    // Get device info
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n\n", prop.major, prop.minor);

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

    // Copy to device
    check_cuda_error(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice), "cudaMemcpy A");
    check_cuda_error(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice), "cudaMemcpy B");
    check_cuda_error(cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice), "cudaMemcpy C");

    // Setup policy framework
    printf("=== Setting up Policy Framework ===\n");
    PolicyFramework framework;

    if (!framework.loadUserKernel("user_kernel.ptx")) {
        fprintf(stderr, "Failed to load user kernel\n");
        return EXIT_FAILURE;
    }

    if (!framework.loadPolicy("policy.ptx")) {
        fprintf(stderr, "Failed to load policy\n");
        return EXIT_FAILURE;
    }

    if (!framework.link(prop.major, prop.minor)) {
        fprintf(stderr, "Failed to link\n");
        return EXIT_FAILURE;
    }

    // Launch kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);

    printf("\n=== Launching Kernel ===\n");
    printf("Grid: (%d, %d), Block: (%d, %d)\n",
           gridDim.x, gridDim.y, blockDim.x, blockDim.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    if (!framework.launch(gridDim, blockDim, 0, d_A, d_B, d_C,
                         M, N, K, alpha, beta)) {
        fprintf(stderr, "Failed to launch kernel\n");
        return EXIT_FAILURE;
    }

    cudaEventRecord(stop);
    check_cuda_error(cudaDeviceSynchronize(), "kernel execution");

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("✓ Kernel executed successfully\n");

    // Copy result back
    check_cuda_error(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost), "cudaMemcpy result");

    // Verify
    printf("\n=== Verification ===\n");
    bool correct = true;
    int errors = 0;
    const int max_errors = 5;

    for (int i = 0; i < M && errors < max_errors; i++) {
        for (int j = 0; j < N && errors < max_errors; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += h_A[i * K + k] * h_B[k * N + j];
            }
            h_C_ref[i * N + j] = alpha * sum + beta * h_C_ref[i * N + j];

            // Apply policy: zero upper triangle
            if (j > i) {
                h_C_ref[i * N + j] = 0.0f;
            }

            float diff = fabs(h_C[i * N + j] - h_C_ref[i * N + j]);
            if (diff > 1e-3) {
                if (errors < max_errors) {
                    printf("Mismatch at [%d][%d]: GPU=%.6f, CPU=%.6f\n",
                           i, j, h_C[i * N + j], h_C_ref[i * N + j]);
                    errors++;
                }
                correct = false;
            }
        }
    }

    if (correct) {
        printf("✓ Results verified! Policy correctly applied.\n");
    } else {
        printf("✗ Verification failed (%d+ errors)\n", errors);
    }

    // Performance
    double gflops = (2.0 * M * N * K) / (milliseconds * 1e6);
    printf("\n=== Performance ===\n");
    printf("Time: %.3f ms\n", milliseconds);
    printf("GFLOPS: %.2f\n", gflops);

    printf("\n=== Sample Results ===\n");
    printf("First 5x5 block (lower triangle should be non-zero, upper zero):\n");
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            printf("%8.4f ", h_C[i * N + j]);
        }
        printf("\n");
    }

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("\n========================================\n");
    printf("Demo completed successfully!\n");
    printf("========================================\n");

    return 0;
}
