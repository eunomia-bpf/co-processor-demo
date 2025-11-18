#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// ADD: Include policy framework header
#include "policy_framework.h"

// CHANGE: Kernel is now a device function instead of __global__
// The framework will wrap it automatically
extern "C" __device__ void gemm_kernel_impl(float *A, float *B, float *C,
                                             int M, int N, int K,
                                             float alpha, float beta) {
    // Kernel body is COMPLETELY UNCHANGED from original!
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

void check_cuda_error(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error at %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char **argv) {
    // Matrix dimensions
    int M = 512;  // Rows of A and C
    int N = 512;  // Columns of B and C
    int K = 512;  // Columns of A, Rows of B

    float alpha = 1.0f;
    float beta = 0.0f;

    printf("GEMM Test: C(%dx%d) = alpha * A(%dx%d) * B(%dx%d) + beta * C\n",
           M, N, M, K, K, N);
    printf("alpha = %.1f, beta = %.1f\n\n", alpha, beta);

    // ADD: Get device properties (using macro from framework header)
    GET_DEVICE_PROPS(prop, device);
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n\n", prop.major, prop.minor);

    // Allocate host memory
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);
    float *h_C_ref = (float*)malloc(size_C);

    // Initialize matrices
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

    // Launch kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);

    printf("Launching kernel with grid(%d, %d) and block(%d, %d)\n",
           gridDim.x, gridDim.y, blockDim.x, blockDim.y);

    // Create events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ADD: Setup policy framework (single macro replaces all boilerplate!)
    printf("\n=== Setting up Policy Framework ===\n");
    POLICY_FRAMEWORK_SETUP(framework, "user_kernel.ptx", "policy.ptx", prop.major, prop.minor);

    // CHANGE: Use framework.launch() instead of <<<...>>>
    // Original: gemm_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    printf("\n=== Launching Kernel with Policy ===\n");
    cudaEventRecord(start);
    framework.launch(gridDim, blockDim, 0, d_A, d_B, d_C, M, N, K, alpha, beta);
    cudaEventRecord(stop);

    check_cuda_error(cudaGetLastError(), "kernel launch");
    check_cuda_error(cudaDeviceSynchronize(), "kernel execution");

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy result back
    check_cuda_error(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost), "cudaMemcpy result");

    // Verify result (simple check on a few elements)
    printf("\nVerifying results (CPU reference)...\n");
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

            float diff = fabs(h_C[i * N + j] - h_C_ref[i * N + j]);
            if (diff > 1e-3) {
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
    } else {
        printf("✗ Found %d+ errors\n", errors);
    }

    // Performance metrics
    double gflops = (2.0 * M * N * K) / (milliseconds * 1e6);
    printf("\nPerformance:\n");
    printf("  Time: %.3f ms\n", milliseconds);
    printf("  GFLOPS: %.2f\n", gflops);

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

    printf("\nGEMM test with policy completed successfully!\n");
    return 0;
}
