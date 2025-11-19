#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef USE_POLICY_FRAMEWORK
// Include policy framework header for nvJitLink + CLC scheduler
#include "policy_framework.h"
#endif

// GEMM kernel implementation
// Original version: __global__ kernel for direct launch
// Modified version: __device__ function for policy framework extraction
#ifdef USE_POLICY_FRAMEWORK
// User kernel as device function - will be extracted from binary using cuobjdump
extern "C" __device__ void gemm_kernel_impl(float *A, float *B, float *C,
                                             int M, int N, int K,
                                             float alpha, float beta)
#else
// Original: Standard CUDA global kernel
__global__ void gemm_kernel(float *A, float *B, float *C,
                            int M, int N, int K,
                            float alpha, float beta)
#endif
{
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

bool read_matrix_binary(const char* filename, float** data, int* rows, int* cols) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open %s\n", filename);
        return false;
    }

    // Read dimensions
    if (fread(rows, sizeof(int), 1, f) != 1 ||
        fread(cols, sizeof(int), 1, f) != 1) {
        fprintf(stderr, "Failed to read dimensions from %s\n", filename);
        fclose(f);
        return false;
    }

    // Allocate and read data
    size_t size = (*rows) * (*cols);
    *data = (float*)malloc(size * sizeof(float));
    if (fread(*data, sizeof(float), size, f) != size) {
        fprintf(stderr, "Failed to read matrix data from %s\n", filename);
        free(*data);
        fclose(f);
        return false;
    }

    fclose(f);
    return true;
}

int main(int argc, char **argv) {
    // Parse command line arguments
    const char* matrix_A_file = nullptr;
    const char* matrix_B_file = nullptr;
    const char* matrix_C_file = nullptr;

    // Default matrix dimensions
    int M = 512;  // Rows of A and C
    int N = 512;  // Columns of B and C
    int K = 512;  // Columns of A, Rows of B

    float alpha = 1.0f;
    float beta = 0.0f;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--matrix-a") == 0 && i + 1 < argc) {
            matrix_A_file = argv[++i];
        } else if (strcmp(argv[i], "--matrix-b") == 0 && i + 1 < argc) {
            matrix_B_file = argv[++i];
        } else if (strcmp(argv[i], "--matrix-c") == 0 && i + 1 < argc) {
            matrix_C_file = argv[++i];
        } else if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
            // Format: MxNxK
            if (sscanf(argv[++i], "%dx%dx%d", &M, &N, &K) != 3) {
                fprintf(stderr, "Invalid size format. Use MxNxK (e.g., 512x512x512)\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--alpha") == 0 && i + 1 < argc) {
            alpha = atof(argv[++i]);
        } else if (strcmp(argv[i], "--beta") == 0 && i + 1 < argc) {
            beta = atof(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("Options:\n");
            printf("  --matrix-a <file>  Load matrix A from binary file\n");
            printf("  --matrix-b <file>  Load matrix B from binary file\n");
            printf("  --matrix-c <file>  Load matrix C from binary file\n");
            printf("  --size MxNxK       Set matrix dimensions (default: 512x512x512)\n");
            printf("  --alpha <value>    Set alpha coefficient (default: 1.0)\n");
            printf("  --beta <value>     Set beta coefficient (default: 0.0)\n");
            printf("  --help             Show this help message\n");
            return 0;
        }
    }

    printf("GEMM Test: C(%dx%d) = alpha * A(%dx%d) * B(%dx%d) + beta * C\n",
           M, N, M, K, K, N);
    printf("alpha = %.1f, beta = %.1f\n\n", alpha, beta);

    // Allocate host memory
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *h_A = nullptr;
    float *h_B = nullptr;
    float *h_C = nullptr;
    float *h_C_ref = (float*)malloc(size_C);

    // Load or generate matrices
    if (matrix_A_file) {
        printf("Loading matrix A from %s\n", matrix_A_file);
        int A_rows, A_cols;
        if (!read_matrix_binary(matrix_A_file, &h_A, &A_rows, &A_cols)) {
            return 1;
        }
        if (A_rows != M || A_cols != K) {
            fprintf(stderr, "Matrix A dimension mismatch: expected %dx%d, got %dx%d\n",
                    M, K, A_rows, A_cols);
            free(h_A);
            return 1;
        }
    } else {
        h_A = (float*)malloc(size_A);
        for (int i = 0; i < M * K; i++) {
            h_A[i] = (float)(rand() % 100) / 100.0f;
        }
    }

    if (matrix_B_file) {
        printf("Loading matrix B from %s\n", matrix_B_file);
        int B_rows, B_cols;
        if (!read_matrix_binary(matrix_B_file, &h_B, &B_rows, &B_cols)) {
            free(h_A);
            return 1;
        }
        if (B_rows != K || B_cols != N) {
            fprintf(stderr, "Matrix B dimension mismatch: expected %dx%d, got %dx%d\n",
                    K, N, B_rows, B_cols);
            free(h_A);
            free(h_B);
            return 1;
        }
    } else {
        h_B = (float*)malloc(size_B);
        for (int i = 0; i < K * N; i++) {
            h_B[i] = (float)(rand() % 100) / 100.0f;
        }
    }

    if (matrix_C_file) {
        printf("Loading matrix C from %s\n", matrix_C_file);
        int C_rows, C_cols;
        if (!read_matrix_binary(matrix_C_file, &h_C, &C_rows, &C_cols)) {
            free(h_A);
            free(h_B);
            return 1;
        }
        if (C_rows != M || C_cols != N) {
            fprintf(stderr, "Matrix C dimension mismatch: expected %dx%d, got %dx%d\n",
                    M, N, C_rows, C_cols);
            free(h_A);
            free(h_B);
            free(h_C);
            return 1;
        }
    } else {
        h_C = (float*)malloc(size_C);
        for (int i = 0; i < M * N; i++) {
            h_C[i] = 0.0f;
        }
    }

    // Initialize reference matrix
    for (int i = 0; i < M * N; i++) {
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

#ifdef USE_POLICY_FRAMEWORK
    // Modified version: Setup policy framework with FULL AUTO loading
    // - Extracts user kernel PTX from current binary using cuobjdump
    // - Loads wrapper kernel PTX from WRAPPER_KERNEL_PATH env var
    // - Loads policy PTX from POLICY_PTX_PATH env var
    printf("\n=== Setting up Policy Framework ===\n");
    POLICY_FRAMEWORK_SETUP_FULL_AUTO(framework);

    // Launch using policy framework with CLC scheduler
    printf("\n=== Launching Kernel with Policy ===\n");
    cudaEventRecord(start);
    framework.launch("gemm_kernel", gridDim, blockDim, 0, d_A, d_B, d_C, M, N, K, alpha, beta);
    cudaEventRecord(stop);
#else
    // Original version: Direct kernel launch
    cudaEventRecord(start);
    gemm_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    cudaEventRecord(stop);
#endif

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

#ifdef USE_POLICY_FRAMEWORK
    printf("\nGEMM test with policy completed successfully!\n");
#else
    printf("\nGEMM test completed successfully!\n");
#endif
    return 0;
}
