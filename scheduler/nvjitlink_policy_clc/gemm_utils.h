// gemm_utils.h
// Utility functions for GEMM testing: argument parsing, matrix I/O, verification

#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ============================================================================
// Error Checking
// ============================================================================

inline void check_cuda_error(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error at %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// ============================================================================
// Matrix I/O
// ============================================================================

inline bool read_matrix_binary(const char* filename, float** data, int* rows, int* cols) {
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

// ============================================================================
// Argument Parsing
// ============================================================================

struct GemmArgs {
    const char* matrix_A_file;
    const char* matrix_B_file;
    const char* matrix_C_file;
    int M, N, K;
    float alpha, beta;
    bool help;

    GemmArgs() :
        matrix_A_file(nullptr),
        matrix_B_file(nullptr),
        matrix_C_file(nullptr),
        M(512), N(512), K(512),
        alpha(1.0f), beta(0.0f),
        help(false) {}
};

inline bool parse_args(int argc, char** argv, GemmArgs& args) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--matrix-a") == 0 && i + 1 < argc) {
            args.matrix_A_file = argv[++i];
        } else if (strcmp(argv[i], "--matrix-b") == 0 && i + 1 < argc) {
            args.matrix_B_file = argv[++i];
        } else if (strcmp(argv[i], "--matrix-c") == 0 && i + 1 < argc) {
            args.matrix_C_file = argv[++i];
        } else if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
            // Format: MxNxK
            if (sscanf(argv[++i], "%dx%dx%d", &args.M, &args.N, &args.K) != 3) {
                fprintf(stderr, "Invalid size format. Use MxNxK (e.g., 512x512x512)\n");
                return false;
            }
        } else if (strcmp(argv[i], "--alpha") == 0 && i + 1 < argc) {
            args.alpha = atof(argv[++i]);
        } else if (strcmp(argv[i], "--beta") == 0 && i + 1 < argc) {
            args.beta = atof(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            args.help = true;
            return true;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            return false;
        }
    }
    return true;
}

inline void print_usage(const char* program_name) {
    printf("Usage: %s [options]\n", program_name);
    printf("\nOptions:\n");
    printf("  --matrix-a <file>  Load matrix A from binary file\n");
    printf("  --matrix-b <file>  Load matrix B from binary file\n");
    printf("  --matrix-c <file>  Load matrix C from binary file\n");
    printf("  --size MxNxK       Set matrix dimensions (default: 512x512x512)\n");
    printf("  --alpha <value>    Set alpha coefficient (default: 1.0)\n");
    printf("  --beta <value>     Set beta coefficient (default: 0.0)\n");
    printf("  --help, -h         Show this help message\n");
    printf("\nExample:\n");
    printf("  %s --matrix-a A.bin --matrix-b B.bin --matrix-c C.bin\n", program_name);
    printf("  %s --size 1024x1024x1024 --alpha 2.0\n", program_name);
}

// ============================================================================
// Matrix Verification
// ============================================================================

struct VerificationResult {
    bool correct;
    int errors;
    float max_diff;
    int max_diff_row;
    int max_diff_col;
};

inline VerificationResult verify_gemm_result(
    const float* h_A, const float* h_B,
    const float* h_C_gpu, int M, int N, int K,
    float alpha, float beta,
    int max_errors_to_print = 10)
{
    VerificationResult result;
    result.correct = true;
    result.errors = 0;
    result.max_diff = 0.0f;
    result.max_diff_row = -1;
    result.max_diff_col = -1;

    printf("Verifying results (CPU reference)...\n");

    for (int i = 0; i < M && result.errors < max_errors_to_print; i++) {
        for (int j = 0; j < N && result.errors < max_errors_to_print; j++) {
            // Compute expected value
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += h_A[i * K + k] * h_B[k * N + j];
            }
            float expected = alpha * sum + beta * 0.0f;  // h_C initialized to 0

            // Compare with GPU result
            float diff = fabs(h_C_gpu[i * N + j] - expected);

            if (diff > result.max_diff) {
                result.max_diff = diff;
                result.max_diff_row = i;
                result.max_diff_col = j;
            }

            if (diff > 1e-3) {
                if (result.errors < max_errors_to_print) {
                    printf("  Mismatch at C[%d][%d]: GPU=%.6f, CPU=%.6f, diff=%.6f\n",
                           i, j, h_C_gpu[i * N + j], expected, diff);
                    result.errors++;
                }
                result.correct = false;
            }
        }
    }

    if (result.correct) {
        printf("✓ Results verified successfully!\n");
        printf("  Max difference: %.6e\n", result.max_diff);
    } else {
        printf("✗ Found %d+ errors\n", result.errors);
        printf("  Max difference: %.6e at C[%d][%d]\n",
               result.max_diff, result.max_diff_row, result.max_diff_col);
    }

    return result;
}

// ============================================================================
// Matrix Initialization
// ============================================================================

inline bool load_or_generate_matrices(
    const GemmArgs& args,
    float** h_A, float** h_B, float** h_C)
{
    size_t size_A = args.M * args.K * sizeof(float);
    size_t size_B = args.K * args.N * sizeof(float);
    size_t size_C = args.M * args.N * sizeof(float);

    // Load or generate matrix A
    if (args.matrix_A_file) {
        printf("Loading matrix A from %s\n", args.matrix_A_file);
        int A_rows, A_cols;
        if (!read_matrix_binary(args.matrix_A_file, h_A, &A_rows, &A_cols)) {
            return false;
        }
        if (A_rows != args.M || A_cols != args.K) {
            fprintf(stderr, "Matrix A dimension mismatch: expected %dx%d, got %dx%d\n",
                    args.M, args.K, A_rows, A_cols);
            free(*h_A);
            return false;
        }
    } else {
        *h_A = (float*)malloc(size_A);
        for (int i = 0; i < args.M * args.K; i++) {
            (*h_A)[i] = (float)(rand() % 100) / 100.0f;
        }
    }

    // Load or generate matrix B
    if (args.matrix_B_file) {
        printf("Loading matrix B from %s\n", args.matrix_B_file);
        int B_rows, B_cols;
        if (!read_matrix_binary(args.matrix_B_file, h_B, &B_rows, &B_cols)) {
            free(*h_A);
            return false;
        }
        if (B_rows != args.K || B_cols != args.N) {
            fprintf(stderr, "Matrix B dimension mismatch: expected %dx%d, got %dx%d\n",
                    args.K, args.N, B_rows, B_cols);
            free(*h_A);
            free(*h_B);
            return false;
        }
    } else {
        *h_B = (float*)malloc(size_B);
        for (int i = 0; i < args.K * args.N; i++) {
            (*h_B)[i] = (float)(rand() % 100) / 100.0f;
        }
    }

    // Load or generate matrix C
    if (args.matrix_C_file) {
        printf("Loading matrix C from %s\n", args.matrix_C_file);
        int C_rows, C_cols;
        if (!read_matrix_binary(args.matrix_C_file, h_C, &C_rows, &C_cols)) {
            free(*h_A);
            free(*h_B);
            return false;
        }
        if (C_rows != args.M || C_cols != args.N) {
            fprintf(stderr, "Matrix C dimension mismatch: expected %dx%d, got %dx%d\n",
                    args.M, args.N, C_rows, C_cols);
            free(*h_A);
            free(*h_B);
            free(*h_C);
            return false;
        }
    } else {
        *h_C = (float*)malloc(size_C);
        for (int i = 0; i < args.M * args.N; i++) {
            (*h_C)[i] = 0.0f;
        }
    }

    return true;
}
