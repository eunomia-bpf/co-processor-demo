// policy_framework.h
// Modern CUDA 12 policy framework using nvJitLink

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvJitLink.h>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <cstdio>

// Error checking macros
#define CHECK_CU(call) do { \
    CUresult res = call; \
    if (res != CUDA_SUCCESS) { \
        const char* errName = nullptr; \
        const char* errStr = nullptr; \
        cuGetErrorName(res, &errName); \
        cuGetErrorString(res, &errStr); \
        fprintf(stderr, "CUDA Driver Error at %s:%d: %s (%s)\n", \
                __FILE__, __LINE__, errName ? errName : "unknown", \
                errStr ? errStr : "no description"); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CHECK_NVJITLINK(call) do { \
    nvJitLinkResult res = call; \
    if (res != NVJITLINK_SUCCESS) { \
        fprintf(stderr, "nvJitLink Error at %s:%d\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

class PolicyFramework {
private:
    std::vector<char> userPTX;
    std::vector<char> policyPTX;
    std::vector<char> linkedCubin;
    CUmodule module;
    CUfunction kernelFunc;
    bool linked;

    std::vector<char> readFile(const char* filename) {
        std::ifstream file(filename, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            fprintf(stderr, "Failed to open %s\n", filename);
            return {};
        }

        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<char> buffer(size);
        if (!file.read(buffer.data(), size)) {
            fprintf(stderr, "Failed to read %s\n", filename);
            return {};
        }

        return buffer;
    }

public:
    PolicyFramework() : module(nullptr), kernelFunc(nullptr), linked(false) {
        CHECK_CU(cuInit(0));
    }

    ~PolicyFramework() {
        if (module) {
            cuModuleUnload(module);
        }
    }

    bool loadUserKernel(const char* ptxFile) {
        std::cout << "Loading user kernel: " << ptxFile << std::endl;
        userPTX = readFile(ptxFile);
        if (userPTX.empty()) {
            return false;
        }
        std::cout << "✓ Loaded user kernel PTX (" << userPTX.size() << " bytes)" << std::endl;
        return true;
    }

    bool loadPolicy(const char* ptxFile) {
        std::cout << "Loading policy: " << ptxFile << std::endl;
        policyPTX = readFile(ptxFile);
        if (policyPTX.empty()) {
            return false;
        }
        std::cout << "✓ Loaded policy PTX (" << policyPTX.size() << " bytes)" << std::endl;
        return true;
    }

    bool link(int computeCapabilityMajor, int computeCapabilityMinor) {
        if (userPTX.empty() || policyPTX.empty()) {
            fprintf(stderr, "User kernel or policy not loaded!\n");
            return false;
        }

        std::cout << "\n=== Linking with nvJitLink ===" << std::endl;

        // Create nvJitLink handle
        nvJitLinkHandle handle;

        char archOpt[32];
        snprintf(archOpt, sizeof(archOpt), "-arch=sm_%d%d",
                 computeCapabilityMajor, computeCapabilityMinor);

        const char* options[] = {
            archOpt,
            "-O3"
        };

        CHECK_NVJITLINK(nvJitLinkCreate(&handle, 2, options));
        std::cout << "✓ Created nvJitLink handle" << std::endl;
        std::cout << "  Options: " << options[0] << " " << options[1] << std::endl;

        // Add user kernel PTX (includes wrapper)
        CHECK_NVJITLINK(nvJitLinkAddData(handle, NVJITLINK_INPUT_PTX,
                                         userPTX.data(), userPTX.size(),
                                         "user_kernel"));
        std::cout << "✓ Added user kernel PTX (includes wrapper)" << std::endl;

        // Add policy PTX
        CHECK_NVJITLINK(nvJitLinkAddData(handle, NVJITLINK_INPUT_PTX,
                                         policyPTX.data(), policyPTX.size(),
                                         "policy"));
        std::cout << "✓ Added policy PTX" << std::endl;

        // Complete the link
        std::cout << "Linking..." << std::endl;
        nvJitLinkResult linkResult = nvJitLinkComplete(handle);

        if (linkResult != NVJITLINK_SUCCESS) {
            size_t logSize;
            nvJitLinkGetErrorLogSize(handle, &logSize);
            if (logSize > 0) {
                std::vector<char> log(logSize);
                nvJitLinkGetErrorLog(handle, log.data());
                fprintf(stderr, "Link error:\n%s\n", log.data());
            }
            nvJitLinkDestroy(&handle);
            return false;
        }

        std::cout << "✓ Linking completed successfully!" << std::endl;

        // Get linked CUBIN
        size_t cubinSize;
        CHECK_NVJITLINK(nvJitLinkGetLinkedCubinSize(handle, &cubinSize));
        linkedCubin.resize(cubinSize);
        CHECK_NVJITLINK(nvJitLinkGetLinkedCubin(handle, linkedCubin.data()));

        std::cout << "✓ Generated linked CUBIN (" << cubinSize << " bytes)" << std::endl;

        nvJitLinkDestroy(&handle);

        // Load the linked module
        CHECK_CU(cuModuleLoadData(&module, linkedCubin.data()));
        std::cout << "✓ Loaded linked module" << std::endl;

        // Get the wrapped kernel function
        CHECK_CU(cuModuleGetFunction(&kernelFunc, module, "gemm_with_policy"));
        std::cout << "✓ Got wrapped kernel function" << std::endl;

        linked = true;
        return true;
    }

    bool launch(dim3 gridDim, dim3 blockDim, cudaStream_t stream,
                float *d_A, float *d_B, float *d_C,
                int M, int N, int K, float alpha, float beta) {
        if (!linked) {
            fprintf(stderr, "Framework not linked! Call link() first.\n");
            return false;
        }

        void* params[] = {&d_A, &d_B, &d_C, &M, &N, &K, &alpha, &beta};

        CHECK_CU(cuLaunchKernel(kernelFunc,
                               gridDim.x, gridDim.y, gridDim.z,
                               blockDim.x, blockDim.y, blockDim.z,
                               0, (CUstream)stream,
                               params, nullptr));

        return true;
    }
};

// ========================================
// Generic Wrapper Kernel (device code)
// ========================================
// Only compile when generating PTX, not for main executable

#if defined(__CUDACC__) && !defined(HOST_COMPILE)

// External reference to user's kernel implementation
extern "C" __device__ void gemm_kernel_impl(float *A, float *B, float *C,
                                             int M, int N, int K,
                                             float alpha, float beta);

// External reference to policy function
extern "C" __device__ void apply_policy(float *C, int M, int N);

// Wrapper kernel that combines user kernel with policy
extern "C" __global__ void gemm_with_policy(float *A, float *B, float *C,
                                            int M, int N, int K,
                                            float alpha, float beta) {
    // Call original user kernel
    gemm_kernel_impl(A, B, C, M, N, K, alpha, beta);

    // Synchronize before applying policy
    __syncthreads();

    // Apply policy
    apply_policy(C, M, N);
}

#endif // defined(__CUDACC__) && !defined(HOST_COMPILE)
