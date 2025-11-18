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
    void* h_policy_func_ptr;  // Host-side copy of device function pointer

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
    PolicyFramework() : module(nullptr), kernelFunc(nullptr), linked(false), h_policy_func_ptr(nullptr) {
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
        // Ensure null terminator for PTX
        if (userPTX.back() != '\0') {
            userPTX.push_back('\0');
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
        // Ensure null terminator for PTX
        if (policyPTX.back() != '\0') {
            policyPTX.push_back('\0');
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
        std::cout << "Adding user kernel PTX (" << userPTX.size() << " bytes)..." << std::endl;
        nvJitLinkResult result1 = nvJitLinkAddData(handle, NVJITLINK_INPUT_PTX,
                                         (void*)userPTX.data(), userPTX.size(),
                                         "user_kernel");
        if (result1 != NVJITLINK_SUCCESS) {
            fprintf(stderr, "nvJitLink error code: %d\n", result1);
            size_t logSize;
            if (nvJitLinkGetErrorLogSize(handle, &logSize) == NVJITLINK_SUCCESS && logSize > 0) {
                std::vector<char> log(logSize);
                nvJitLinkGetErrorLog(handle, log.data());
                fprintf(stderr, "Error log:\n%s\n", log.data());
            }
            nvJitLinkDestroy(&handle);
            return false;
        }
        std::cout << "✓ Added user kernel PTX (includes wrapper)" << std::endl;

        // Add policy PTX
        CHECK_NVJITLINK(nvJitLinkAddData(handle, NVJITLINK_INPUT_PTX,
                                         (void*)policyPTX.data(), policyPTX.size(),
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

        // Get the device policy function pointer and copy to host
        CUdeviceptr d_policy_func_ptr_addr;
        size_t size;
        CUresult res = cuModuleGetGlobal(&d_policy_func_ptr_addr, &size, module, "d_apply_policy");
        if (res == CUDA_SUCCESS) {
            CHECK_CU(cuMemcpyDtoH(&h_policy_func_ptr, d_policy_func_ptr_addr, sizeof(void*)));
            std::cout << "✓ Got policy function pointer" << std::endl;
        } else {
            std::cout << "Note: No policy function pointer found, will use nullptr" << std::endl;
            h_policy_func_ptr = nullptr;
        }

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

        void* params[] = {&d_A, &d_B, &d_C, &M, &N, &K, &alpha, &beta, &h_policy_func_ptr};

        CHECK_CU(cuLaunchKernel(kernelFunc,
                               gridDim.x, gridDim.y, gridDim.z,
                               blockDim.x, blockDim.y, blockDim.z,
                               0, (CUstream)stream,
                               params, nullptr));

        return true;
    }

    void getModule(CUmodule* mod) {
        *mod = module;
    }
};

// ========================================
// Generic Wrapper Kernel (device code)
// ========================================

#ifdef __CUDACC__

// External reference to user's kernel implementation
extern "C" __device__ void gemm_kernel_impl(float *A, float *B, float *C,
                                             int M, int N, int K,
                                             float alpha, float beta);

// Function pointer type for policy
typedef void (*policy_func_t)(int);

// Wrapper kernel that combines user kernel with policy
extern "C" __global__ void gemm_with_policy(float *A, float *B, float *C,
                                            int M, int N, int K,
                                            float alpha, float beta,
                                            policy_func_t policy_func) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row * N + col;

    // Call original user kernel
    gemm_kernel_impl(A, B, C, M, N, K, alpha, beta);

    // Synchronize before applying policy
    __syncthreads();

    // Apply policy with matrix info
    if (row < M && col < N && policy_func != nullptr) {
        policy_func(idx);
    }
}

#endif // __CUDACC__
