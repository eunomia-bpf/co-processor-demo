// sample_app.cu
// Original application with embedded kernel
// This will be the "target" binary we extract from

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Simple vector add kernel
__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// GEMM kernel for extraction
__global__ void gemm_kernel(float *A, float *B, float *C,
                           int M, int N, int K,
                           float alpha, float beta) {
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

int main() {
    printf("========================================\n");
    printf("Sample Application (Target for Extraction)\n");
    printf("========================================\n\n");

    const int N = 1024;
    const int M = 256, K = 256;

    // Test vector add
    printf("=== Testing Vector Add ===\n");
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));

    float *h_a = (float*)malloc(N * sizeof(float));
    float *h_b = (float*)malloc(N * sizeof(float));
    float *h_c = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    vectorAdd<<<(N+255)/256, 256>>>(d_a, d_b, d_c, N);
    cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Result[0] = %.1f (expected %.1f)\n", h_c[0], h_a[0] + h_b[0]);
    printf("Result[10] = %.1f (expected %.1f)\n", h_c[10], h_a[10] + h_b[10]);
    printf("✓ Vector add completed\n\n");

    // Test GEMM
    printf("=== Testing GEMM ===\n");
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);

    gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f);
    cudaDeviceSynchronize();

    printf("✓ GEMM completed\n");

    // Cleanup
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_a); free(h_b); free(h_c);

    printf("\n========================================\n");
    printf("This binary contains 2 kernels:\n");
    printf("  1. vectorAdd\n");
    printf("  2. gemm_kernel\n");
    printf("\nUse cuobjdump to extract them!\n");
    printf("  cuobjdump -xelf all sample_app\n");
    printf("========================================\n");

    return 0;
}
