#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CHECK_CUBLAS(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error in %s:%d: %d\n", __FILE__, __LINE__, status); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

void printMatrix(const char* name, float* matrix, int rows, int cols) {
    printf("%s (%dx%d):\n", name, rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%8.2f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    // Matrix dimensions: C(m x n) = A(m x k) * B(k x n)
    const int m = 3;  // rows of A and C
    const int k = 4;  // cols of A, rows of B
    const int n = 2;  // cols of B and C

    printf("cuBLAS Matrix Multiplication Demo\n");
    printf("==================================\n");
    printf("Computing C(%dx%d) = A(%dx%d) * B(%dx%d)\n\n", m, n, m, k, k, n);

    // Allocate host memory
    float *h_A = (float*)malloc(m * k * sizeof(float));
    float *h_B = (float*)malloc(k * n * sizeof(float));
    float *h_C = (float*)malloc(m * n * sizeof(float));

    // Initialize matrices A and B
    printf("Initializing matrices...\n");
    for (int i = 0; i < m * k; i++) {
        h_A[i] = (float)(i + 1);
    }
    for (int i = 0; i < k * n; i++) {
        h_B[i] = (float)(i + 1);
    }

    printMatrix("Matrix A", h_A, m, k);
    printMatrix("Matrix B", h_B, k, n);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void**)&d_A, m * k * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_B, k * n * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_C, m * n * sizeof(float)));

    // Copy matrices from host to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyHostToDevice));

    // Create cuBLAS handle
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // Define scalars for GEMM operation: C = alpha * A * B + beta * C
    float alpha = 1.0f;
    float beta = 0.0f;

    // Perform matrix multiplication using cublasSgemm
    // Note: cuBLAS uses column-major order, so we compute B^T * A^T = (A * B)^T
    printf("Performing matrix multiplication using cuBLAS...\n");
    CHECK_CUBLAS(cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,  // No transpose for both matrices
        n, m, k,                     // n, m, k dimensions
        &alpha,                      // alpha scalar
        d_B, n,                      // B matrix and leading dimension
        d_A, k,                      // A matrix and leading dimension
        &beta,                       // beta scalar
        d_C, n                       // C matrix and leading dimension
    ));

    // Wait for GPU to finish
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy result from device to host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost));

    // Print result
    printMatrix("Result Matrix C = A * B", h_C, m, n);

    // Verify result with CPU computation
    printf("Verifying result...\n");
    bool correct = true;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                sum += h_A[i * k + p] * h_B[p * n + j];
            }
            float diff = fabs(h_C[i * n + j] - sum);
            if (diff > 1e-3) {
                printf("Mismatch at C[%d][%d]: GPU=%f, CPU=%f\n",
                       i, j, h_C[i * n + j], sum);
                correct = false;
            }
        }
    }

    if (correct) {
        printf("✓ Results verified successfully!\n");
    } else {
        printf("✗ Verification failed!\n");
    }

    // Cleanup
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    printf("\nDemo completed successfully!\n");

    return 0;
}
