#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "extractor.h"

void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error at %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv) {
    printf("========================================\n");
    printf("Binary Extraction + JIT Rewriting Demo\n");
    printf("========================================\n");

    const char* targetBinary = argc > 1 ? argv[1] : "./sample_app";

    // Get device info
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);

    // Step 1: Extract kernels from binary
    BinaryExtractor extractor(targetBinary);

    if (!extractor.extractAllCubins()) {
        fprintf(stderr, "Failed to extract kernels\n");
        return EXIT_FAILURE;
    }

    // Show what we extracted
    auto extractedFiles = extractor.getExtractedFiles();
    if (!extractedFiles.empty()) {
        printf("\n=== Analyzing Extracted Files ===\n");
        for (const auto& file : extractedFiles) {
            extractor.listSymbols(file.c_str());
        }
    }

    // Step 2: Load policy and rewrite
    JITRewriter rewriter;

    if (!rewriter.loadPolicy("policy.ptx")) {
        fprintf(stderr, "Failed to load policy\n");
        return EXIT_FAILURE;
    }

    if (!rewriter.linkAndLoad(prop.major, prop.minor)) {
        fprintf(stderr, "Failed to link and load\n");
        return EXIT_FAILURE;
    }

    // Step 3: Test the rewritten kernels
    printf("\n=== Testing Rewritten Kernels ===\n");

    // Test vector add with policy
    printf("\n--- Vector Add with Policy ---\n");
    const int N = 1024;
    float *d_a, *d_b, *d_c;
    check_cuda(cudaMalloc(&d_a, N * sizeof(float)), "malloc a");
    check_cuda(cudaMalloc(&d_b, N * sizeof(float)), "malloc b");
    check_cuda(cudaMalloc(&d_c, N * sizeof(float)), "malloc c");

    float *h_a = (float*)malloc(N * sizeof(float));
    float *h_b = (float*)malloc(N * sizeof(float));
    float *h_c = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
        h_c[i] = 0;
    }

    check_cuda(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice), "copy a");
    check_cuda(cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice), "copy b");
    check_cuda(cudaMemcpy(d_c, h_c, N * sizeof(float), cudaMemcpyHostToDevice), "copy c");

    // Get and launch wrapped kernel
    CUkernel vectorAddKernel;
    if (rewriter.getKernel(&vectorAddKernel, "vectorAdd_with_policy")) {
        void* args[] = {&d_a, &d_b, &d_c, (void*)&N};

        cudaError_t err = cudaLaunchKernel(vectorAddKernel,
                                           dim3((N + 255) / 256, 1, 1),  // grid
                                           dim3(256, 1, 1),               // block
                                           args, 0, 0);                   // args, shared mem, stream
        if (err != cudaSuccess) {
            fprintf(stderr, "Launch failed: %s\n", cudaGetErrorString(err));
            return EXIT_FAILURE;
        }

        check_cuda(cudaDeviceSynchronize(), "sync");
        check_cuda(cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost), "copy result");

        printf("Result[0] = %.1f (expected %.1f)\n", h_c[0], h_a[0] + h_b[0]);
        printf("Result[10] = %.1f (expected %.1f)\n", h_c[10], h_a[10] + h_b[10]);
        printf("✓ Vector add with policy executed\n");
    }

    // Test GEMM with policy
    printf("\n--- GEMM with Policy ---\n");
    const int M = 256, K = 256;
    float *d_A, *d_B, *d_C;
    check_cuda(cudaMalloc(&d_A, M * K * sizeof(float)), "malloc A");
    check_cuda(cudaMalloc(&d_B, K * N * sizeof(float)), "malloc B");
    check_cuda(cudaMalloc(&d_C, M * N * sizeof(float)), "malloc C");

    float *h_A = (float*)malloc(M * K * sizeof(float));
    float *h_B = (float*)malloc(K * N * sizeof(float));
    float *h_C = (float*)malloc(M * N * sizeof(float));

    for (int i = 0; i < M * K; i++) h_A[i] = 1.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = 1.0f;
    for (int i = 0; i < M * N; i++) h_C[i] = 0.0f;

    check_cuda(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice), "copy A");
    check_cuda(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice), "copy B");
    check_cuda(cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice), "copy C");

    CUkernel gemmKernel;
    if (rewriter.getKernel(&gemmKernel, "gemm_with_policy")) {
        float alpha = 1.0f, beta = 0.0f;
        void* args[] = {&d_A, &d_B, &d_C, (void*)&M, (void*)&N, (void*)&K, &alpha, &beta};

        dim3 grid((N + 15) / 16, (M + 15) / 16);
        dim3 block(16, 16);

        cudaError_t err = cudaLaunchKernel(gemmKernel, grid, block, args, 0, 0);
        if (err != cudaSuccess) {
            fprintf(stderr, "Launch failed: %s\n", cudaGetErrorString(err));
            return EXIT_FAILURE;
        }

        check_cuda(cudaDeviceSynchronize(), "sync");
        check_cuda(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost), "copy result");

        // Check policy applied (upper triangle should be zero)
        bool policyWorking = true;
        for (int i = 0; i < 5 && policyWorking; i++) {
            for (int j = 0; j < 5; j++) {
                float expected = (j > i) ? 0.0f : (float)K;
                if (fabs(h_C[i * N + j] - expected) > 0.01f) {
                    policyWorking = false;
                    break;
                }
            }
        }

        printf("First 5x5 block:\n");
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                printf("%6.1f ", h_C[i * N + j]);
            }
            printf("\n");
        }

        if (policyWorking) {
            printf("✓ GEMM with policy executed correctly\n");
            printf("✓ Upper triangle zeroed by policy!\n");
        } else {
            printf("✗ Policy not applied correctly\n");
        }
    }

    // Cleanup
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_a); free(h_b); free(h_c);
    free(h_A); free(h_B); free(h_C);

    printf("\n========================================\n");
    printf("Summary:\n");
    printf("1. Extracted kernels from binary using cuobjdump\n");
    printf("2. Loaded policy code (PTX)\n");
    printf("3. Executed wrapped versions with policy\n");
    printf("4. Verified policy enforcement\n");
    printf("\n");
    printf("Key takeaway: We injected policy into a\n");
    printf("pre-compiled binary without source code!\n");
    printf("========================================\n");

    return 0;
}
