// gemm_policy_wrapper.h
// Generic wrapper kernel - dynamically links user kernel + policy
#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Error handling
inline void check_cuda(cudaError_t err, const char* where) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[CUDA] %s failed: %s\n",
                     where, cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}

inline void check_cu(CUresult res, const char* where) {
    if (res != CUDA_SUCCESS) {
        const char* msg = nullptr;
        cuGetErrorName(res, &msg);
        std::fprintf(stderr, "[CU] %s failed: %s\n",
                     where, msg ? msg : "<unknown>");
        std::exit(EXIT_FAILURE);
    }
}

// Helper function to read file into memory
inline void* read_file(const char* filename, size_t* size) {
    FILE* f = std::fopen(filename, "rb");
    if (!f) {
        std::fprintf(stderr, "Failed to open %s\n", filename);
        return nullptr;
    }
    std::fseek(f, 0, SEEK_END);
    *size = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    void* data = std::malloc(*size);
    std::fread(data, 1, *size, f);
    std::fclose(f);
    return data;
}

// Helper to launch wrapper kernel via RUNTIME dynamic linking
// Takes kernel cubin path (for use with any pre-compiled kernel)
// kernel_cubin_path: Path to the cubin containing the user's device kernel
// kernel_func_name: Name of the device function pointer symbol to load
inline void run_with_policy(dim3 gridDim, dim3 blockDim, cudaStream_t stream,
                           const char* kernel_cubin_path,  // Path to kernel cubin
                           const char* kernel_func_name,   // Name of kernel function symbol
                           const char* policy_path,        // Path to policy cubin
                           float* d_A, float* d_B, float* d_C,
                           int M, int N, int K, float alpha, float beta) {
    // Initialize CUDA driver API
    check_cu(cuInit(0), "cuInit");

    printf("Runtime dynamic linking: kernel cubin + policy\n");

    // Create a linker state
    CUlinkState linkState;
    check_cu(cuLinkCreate(0, nullptr, nullptr, &linkState), "cuLinkCreate");

    // Read cubin files
    size_t wrapper_size, kernel_size, policy_size;
    void* wrapper_data = read_file("./wrapper_kernel.cubin", &wrapper_size);
    void* kernel_data = read_file(kernel_cubin_path, &kernel_size);
    void* policy_data = read_file(policy_path, &policy_size);

    if (!wrapper_data || !kernel_data || !policy_data) {
        std::fprintf(stderr, "Failed to read cubin files\n");
        std::exit(EXIT_FAILURE);
    }

    printf("Linking wrapper_kernel.cubin (%zu bytes)\n", wrapper_size);
    printf("Linking kernel from %s (%zu bytes)\n", kernel_cubin_path, kernel_size);
    printf("Linking policy from %s (%zu bytes)\n", policy_path, policy_size);

    // Add cubins to the linker (order matters for resolving symbols)
    check_cu(cuLinkAddData(linkState, CU_JIT_INPUT_CUBIN, wrapper_data, wrapper_size,
                          "wrapper_kernel.cubin", 0, nullptr, nullptr),
             "cuLinkAddData(wrapper)");

    check_cu(cuLinkAddData(linkState, CU_JIT_INPUT_CUBIN, kernel_data, kernel_size,
                          kernel_cubin_path, 0, nullptr, nullptr),
             "cuLinkAddData(kernel)");

    check_cu(cuLinkAddData(linkState, CU_JIT_INPUT_CUBIN, policy_data, policy_size,
                          policy_path, 0, nullptr, nullptr),
             "cuLinkAddData(policy)");

    // Complete the linking
    void* linked_cubin;
    size_t linked_size;
    check_cu(cuLinkComplete(linkState, &linked_cubin, &linked_size), "cuLinkComplete");

    printf("Runtime linking complete, linked cubin size: %zu bytes\n", linked_size);

    // Load the linked module
    CUmodule module;
    check_cu(cuModuleLoadData(&module, linked_cubin), "cuModuleLoadData");

    // Get address of the device function pointer variable
    CUdeviceptr kernel_var_addr = 0;
    size_t kernel_var_size = 0;
    check_cu(cuModuleGetGlobal(&kernel_var_addr, &kernel_var_size, module, kernel_func_name),
             "cuModuleGetGlobal(kernel_func_name)");

    printf("Device function pointer variable at 0x%llx, size %zu bytes\n",
           (unsigned long long)kernel_var_addr, kernel_var_size);

    // Read the VALUE of the device function pointer variable
    // This gives us the actual function pointer that device code can call
    void* kernel_func_ptr = nullptr;
    check_cu(cuMemcpyDtoH(&kernel_func_ptr, kernel_var_addr, sizeof(void*)),
             "cuMemcpyDtoH(kernel function pointer)");

    printf("Kernel function pointer value: %p\n", kernel_func_ptr);

    // Get the wrapper kernel function
    CUfunction wrapper_func;
    check_cu(cuModuleGetFunction(&wrapper_func, module, "run_with_policy_kernel"),
             "cuModuleGetFunction(run_with_policy_kernel)");

    // Prepare kernel arguments: (kernel_func_ptr, A, B, C, M, N, K, alpha, beta)
    // Pass the VALUE of the function pointer, not the address of the variable
    void* params[] = {&kernel_func_ptr, &d_A, &d_B, &d_C, &M, &N, &K, &alpha, &beta};

    // Launch the dynamically linked kernel
    check_cu(cuLaunchKernel(wrapper_func,
                           gridDim.x, gridDim.y, gridDim.z,
                           blockDim.x, blockDim.y, blockDim.z,
                           0, (CUstream)stream,
                           params, nullptr),
             "cuLaunchKernel");

    check_cu(cuCtxSynchronize(), "cuCtxSynchronize");

    // Cleanup
    cuLinkDestroy(linkState);
    std::free(wrapper_data);
    std::free(kernel_data);
    std::free(policy_data);
}

