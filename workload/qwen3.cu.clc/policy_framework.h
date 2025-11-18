// policy_framework.h
// Modern CUDA 12 policy framework using nvJitLink

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvJitLink.h>
#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <fcntl.h>

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
    std::vector<char> wrapperPTX;
    std::vector<char> policyPTX;
    std::vector<char> linkedCubin;
    CUmodule module;
    std::map<std::string, CUfunction> kernelFuncs;  // Support multiple kernels
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

    // Extract PTX from a binary file using cuobjdump
    // Extracts all PTX sections from the fatbinary
    std::vector<char> extractPTXFromBinary(const char* binary_path) {
        // Create temporary file for PTX output
        char ptx_temp[] = "/tmp/kernel_XXXXXX.ptx";
        int fd = mkstemps(ptx_temp, 4);
        if (fd == -1) {
            fprintf(stderr, "Failed to create temp file\n");
            return {};
        }
        close(fd);

        // Use cuobjdump to extract all PTX
        // The --dump-ptx-text option gives cleaner output
        char cmd[2048];
        snprintf(cmd, sizeof(cmd),
                 "cuobjdump --dump-ptx '%s' > '%s' 2>&1",
                 binary_path, ptx_temp);

        int ret = system(cmd);
        if (ret != 0) {
            fprintf(stderr, "Failed to extract PTX from %s (cuobjdump returned %d)\n", binary_path, ret);
            unlink(ptx_temp);
            return {};
        }

        // Read the full cuobjdump output
        std::vector<char> full_output = readFile(ptx_temp);

        // Clean up temp file
        unlink(ptx_temp);

        if (full_output.empty()) {
            fprintf(stderr, "cuobjdump output is empty from %s\n", binary_path);
            return {};
        }

        // Parse the output to extract actual PTX code
        // Look for lines between "Fatbin ptx code:" and the next "Fatbin" or end
        std::string output_str(full_output.begin(), full_output.end());
        size_t ptx_start = output_str.find(".version");

        if (ptx_start == std::string::npos) {
            fprintf(stderr, "No PTX .version directive found in cuobjdump output\n");
            return {};
        }

        // Extract from .version to the end (or until next Fatbin section)
        std::string ptx_code = output_str.substr(ptx_start);

        // Convert back to vector
        std::vector<char> ptx(ptx_code.begin(), ptx_code.end());

        return ptx;
    }

    // Find wrapper PTX using environment variable
    std::string findWrapperPTX() {
        const char* wrapper_path = getenv("WRAPPER_KERNEL_PATH");
        if (!wrapper_path) {
            fprintf(stderr, "ERROR: WRAPPER_KERNEL_PATH environment variable not set\n");
            return "";
        }
        std::cout << "Using WRAPPER_KERNEL_PATH: " << wrapper_path << std::endl;
        return std::string(wrapper_path);
    }

    // Find policy PTX using environment variable
    std::string findPolicyPTX() {
        const char* policy_path = getenv("POLICY_PTX_PATH");
        if (!policy_path) {
            fprintf(stderr, "ERROR: POLICY_PTX_PATH environment variable not set\n");
            return "";
        }
        std::cout << "Using POLICY_PTX_PATH: " << policy_path << std::endl;
        return std::string(policy_path);
    }

public:
    PolicyFramework() : module(nullptr), linked(false), h_policy_func_ptr(nullptr) {
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

    // Auto-load user kernel by extracting from current binary using cuobjdump
    bool loadUserKernelAuto() {
        std::cout << "Auto-extracting user kernel PTX from binary..." << std::endl;

        // Get current executable path
        char exe_path[1024];
        ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
        if (len == -1) {
            fprintf(stderr, "Failed to get executable path\n");
            return false;
        }
        exe_path[len] = '\0';

        userPTX = extractPTXFromBinary(exe_path);
        if (userPTX.empty()) {
            return false;
        }
        // Ensure null terminator for PTX
        if (userPTX.back() != '\0') {
            userPTX.push_back('\0');
        }
        std::cout << "✓ Auto-extracted user kernel PTX (" << userPTX.size() << " bytes)" << std::endl;
        return true;
    }

    // Auto-load wrapper kernel from ENV variable
    bool loadWrapperAuto() {
        std::cout << "Auto-loading wrapper kernel PTX..." << std::endl;
        std::string wrapper_path = findWrapperPTX();
        if (wrapper_path.empty()) {
            return false;
        }

        wrapperPTX = readFile(wrapper_path.c_str());
        if (wrapperPTX.empty()) {
            return false;
        }
        // Ensure null terminator for PTX
        if (wrapperPTX.back() != '\0') {
            wrapperPTX.push_back('\0');
        }
        std::cout << "✓ Loaded wrapper kernel PTX (" << wrapperPTX.size() << " bytes)" << std::endl;
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

    // Auto-load policy from ENV variable
    bool loadPolicyAuto() {
        std::cout << "Auto-loading policy PTX..." << std::endl;
        std::string policy_path = findPolicyPTX();
        if (policy_path.empty()) {
            return false;
        }
        return loadPolicy(policy_path.c_str());
    }

    bool link(int computeCapabilityMajor, int computeCapabilityMinor) {
        if (userPTX.empty() || wrapperPTX.empty() || policyPTX.empty()) {
            fprintf(stderr, "User kernel, wrapper, or policy not loaded!\n");
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

        // Add user kernel PTX
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
        std::cout << "✓ Added user kernel PTX" << std::endl;

        // Add wrapper kernel PTX
        std::cout << "Adding wrapper kernel PTX (" << wrapperPTX.size() << " bytes)..." << std::endl;
        CHECK_NVJITLINK(nvJitLinkAddData(handle, NVJITLINK_INPUT_PTX,
                                         (void*)wrapperPTX.data(), wrapperPTX.size(),
                                         "wrapper_kernel"));
        std::cout << "✓ Added wrapper kernel PTX" << std::endl;

        // Add policy PTX
        std::cout << "Adding policy PTX (" << policyPTX.size() << " bytes)..." << std::endl;
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

    // Get kernel function by name (lazy loading)
    CUfunction getKernel(const char* kernel_name) {
        std::string kname(kernel_name);

        // Check if already loaded
        auto it = kernelFuncs.find(kname);
        if (it != kernelFuncs.end()) {
            return it->second;
        }

        // Load the kernel function
        char wrapped_kernel_name[256];
        snprintf(wrapped_kernel_name, sizeof(wrapped_kernel_name), "%s_with_policy", kernel_name);

        CUfunction func;
        CUresult res = cuModuleGetFunction(&func, module, wrapped_kernel_name);
        if (res != CUDA_SUCCESS) {
            const char* errName = nullptr;
            cuGetErrorName(res, &errName);
            fprintf(stderr, "Failed to get kernel '%s': %s\n", wrapped_kernel_name, errName ? errName : "unknown");
            return nullptr;
        }

        std::cout << "✓ Got wrapped kernel function: " << wrapped_kernel_name << std::endl;
        kernelFuncs[kname] = func;
        return func;
    }

    // Generic launch method - works with ANY kernel signature!
    // Uses variadic templates to accept any number/type of parameters
    // Now accepts kernel name as first parameter for multi-kernel support
    template<typename... Args>
    bool launch(const char* kernel_name, dim3 gridDim, dim3 blockDim, cudaStream_t stream, Args... args) {
        return launch_with_shared(kernel_name, gridDim, blockDim, 0, stream, args...);
    }

    // Launch with shared memory support
    template<typename... Args>
    bool launch_with_shared(const char* kernel_name, dim3 gridDim, dim3 blockDim,
                           size_t sharedMem, cudaStream_t stream, Args... args) {
        if (!linked) {
            fprintf(stderr, "Framework not linked! Call link() first.\n");
            return false;
        }

        // Get the kernel function for this specific kernel
        CUfunction kernelFunc = getKernel(kernel_name);
        if (!kernelFunc) {
            fprintf(stderr, "Failed to get kernel: %s\n", kernel_name);
            return false;
        }

        // Create a tuple of pointers to all arguments plus the policy function pointer
        void* params[] = {(void*)&args..., &h_policy_func_ptr};

        CHECK_CU(cuLaunchKernel(kernelFunc,
                               gridDim.x, gridDim.y, gridDim.z,
                               blockDim.x, blockDim.y, blockDim.z,
                               sharedMem, (CUstream)stream,
                               params, nullptr));

        return true;
    }


    void getModule(CUmodule* mod) {
        *mod = module;
    }
};

// ========================================
// CONVENIENCE MACROS - Hide all boilerplate!
// ========================================

// MACRO: Setup policy framework with error handling
// Usage: POLICY_FRAMEWORK_SETUP(framework_var, kernel_ptx, policy_ptx, compute_major, compute_minor)
#define POLICY_FRAMEWORK_SETUP(fw, kernel_ptx, policy_ptx, major, minor) \
    PolicyFramework fw; \
    do { \
        if (!fw.loadUserKernel(kernel_ptx)) { \
            fprintf(stderr, "Failed to load user kernel: %s\n", kernel_ptx); \
            exit(EXIT_FAILURE); \
        } \
        if (!fw.loadPolicy(policy_ptx)) { \
            fprintf(stderr, "Failed to load policy: %s\n", policy_ptx); \
            exit(EXIT_FAILURE); \
        } \
        if (!fw.link(major, minor)) { \
            fprintf(stderr, "Failed to link framework\n"); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// MACRO: Setup policy framework with automatic device detection
// Usage: POLICY_FRAMEWORK_SETUP_AUTO(framework_var, kernel_ptx, policy_ptx)
#define POLICY_FRAMEWORK_SETUP_AUTO(fw, kernel_ptx, policy_ptx) \
    int device_##fw = 0; \
    cudaDeviceProp prop_##fw; \
    cudaGetDeviceProperties(&prop_##fw, device_##fw); \
    POLICY_FRAMEWORK_SETUP(fw, kernel_ptx, policy_ptx, prop_##fw.major, prop_##fw.minor)

// MACRO: Setup policy framework with PTX auto-extraction from binary
// Usage: POLICY_FRAMEWORK_SETUP_FULL_AUTO(framework_var)
// Uses ENV variables: WRAPPER_KERNEL_PATH (default: ./wrapper_kernel), POLICY_PTX_PATH (default: ./policy.ptx)
#define POLICY_FRAMEWORK_SETUP_FULL_AUTO(fw) \
    PolicyFramework fw; \
    do { \
        int device_##fw = 0; \
        cudaDeviceProp prop_##fw; \
        cudaGetDeviceProperties(&prop_##fw, device_##fw); \
        if (!fw.loadUserKernelAuto()) { \
            fprintf(stderr, "Failed to auto-extract user kernel from binary\n"); \
            exit(EXIT_FAILURE); \
        } \
        if (!fw.loadWrapperAuto()) { \
            fprintf(stderr, "Failed to auto-load wrapper kernel\n"); \
            exit(EXIT_FAILURE); \
        } \
        if (!fw.loadPolicyAuto()) { \
            fprintf(stderr, "Failed to auto-load policy\n"); \
            exit(EXIT_FAILURE); \
        } \
        if (!fw.link(prop_##fw.major, prop_##fw.minor)) { \
            fprintf(stderr, "Failed to link framework\n"); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// MACRO: Setup policy framework with manual policy path (for backward compatibility)
// Usage: POLICY_FRAMEWORK_SETUP_FULL_AUTO_WITH_POLICY(framework_var, policy_ptx)
#define POLICY_FRAMEWORK_SETUP_FULL_AUTO_WITH_POLICY(fw, policy_ptx) \
    PolicyFramework fw; \
    do { \
        int device_##fw = 0; \
        cudaDeviceProp prop_##fw; \
        cudaGetDeviceProperties(&prop_##fw, device_##fw); \
        if (!fw.loadUserKernelAuto()) { \
            fprintf(stderr, "Failed to auto-extract user kernel from binary\n"); \
            exit(EXIT_FAILURE); \
        } \
        if (!fw.loadWrapperAuto()) { \
            fprintf(stderr, "Failed to auto-load wrapper kernel\n"); \
            exit(EXIT_FAILURE); \
        } \
        if (!fw.loadPolicy(policy_ptx)) { \
            fprintf(stderr, "Failed to load policy: %s\n", policy_ptx); \
            exit(EXIT_FAILURE); \
        } \
        if (!fw.link(prop_##fw.major, prop_##fw.minor)) { \
            fprintf(stderr, "Failed to link framework\n"); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// MACRO: Get device properties (needed for manual setup)
#define GET_DEVICE_PROPS(prop_var, device_var) \
    int device_var = 0; \
    cudaDeviceProp prop_var; \
    cudaGetDeviceProperties(&prop_var, device_var)

// ========================================
// USAGE EXAMPLES:
// ========================================
//
// Example 1: FULL AUTO - Extract PTX from binary (EASIEST!)
// ----------------------------------------------------------
//   POLICY_FRAMEWORK_SETUP_FULL_AUTO(fw, "policy.ptx");
//   fw.launch(grid, block, 0, args...);
//
//   Benefits:
//   - NO separate PTX file needed for kernel
//   - NO extra compilation step
//   - Extracts PTX from binary at runtime using cuobjdump
//
// Example 2: Auto setup with external PTX
// ----------------------------------------
//   POLICY_FRAMEWORK_SETUP_AUTO(framework, "kernel.ptx", "policy.ptx");
//   framework.launch(grid, block, 0, args...);
//
// Example 3: Manual setup (most control)
// ---------------------------------------
//   GET_DEVICE_PROPS(prop, device);
//   printf("Device: %s\n", prop.name);
//   POLICY_FRAMEWORK_SETUP(framework, "kernel.ptx", "policy.ptx", prop.major, prop.minor);
//   framework.launch(grid, block, 0, args...);
//

// ========================================
// MACRO-BASED GENERIC POLICY FRAMEWORK
// ========================================
//
// This provides MACROS for wrapping ANY CUDA kernel with minimal code changes.
// Works with GEMM, vec_add, or ANY custom kernel!
//
// USAGE - Only 2 changes to your original code:
//
// 1. Change kernel declaration:
//    FROM: __global__ void my_kernel(args...)
//    TO:   POLICY_KERNEL_DECL(my_kernel, args...)
//
// 2. Change kernel launch:
//    FROM: my_kernel<<<grid, block>>>(args...);
//    TO:   POLICY_KERNEL_LAUNCH(my_kernel, grid, block, args...);
//
// That's it! The macros handle everything else automatically.
//
// ========================================

#ifdef __CUDACC__

// Function pointer type for policy - takes thread index
typedef void (*policy_func_t)(int);

// Generic policy wrapper template - works with ANY kernel signature!
// This is a __device__ function that can be called from any __global__ wrapper
template<typename KernelPtr, typename... Args>
__device__ void apply_policy_wrapper(KernelPtr kernel_impl,
                                     policy_func_t policy_func,
                                     Args... args) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x +
              (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x;

    // Call original user kernel with forwarded arguments
    kernel_impl(args...);

    // Synchronize before applying policy
    __syncthreads();

    // Apply policy
    if (policy_func != nullptr) {
        policy_func(idx);
    }
}

// ========================================
// MACRO SYSTEM - Minimal code changes needed!
// ========================================

// MACRO 1: Declare a policy-enabled kernel
// Usage: POLICY_KERNEL_DECL(kernel_name, param1_type param1_name, param2_type param2_name, ...)
//
// Example:
//   Original: __global__ void gemm_kernel(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta)
//   With policy: POLICY_KERNEL_DECL(gemm_kernel, float *A, float *B, float *C, int M, int N, int K, float alpha, float beta)
//
#define POLICY_KERNEL_DECL(kernel_name, ...) \
    __device__ void kernel_name##_impl(__VA_ARGS__); \
    extern "C" __global__ void kernel_name##_with_policy(__VA_ARGS__, policy_func_t policy_func) { \
        apply_policy_wrapper(kernel_name##_impl, policy_func, ##__VA_ARGS__); \
    } \
    __device__ void kernel_name##_impl(__VA_ARGS__)

// MACRO 2: Original kernel declaration (when not using policy)
// This allows you to keep the original __global__ kernel for direct launch
#define POLICY_KERNEL_DECL_WITH_ORIGINAL(kernel_name, ...) \
    __device__ void kernel_name##_impl(__VA_ARGS__); \
    __global__ void kernel_name(__VA_ARGS__) { \
        kernel_name##_impl(__VA_ARGS__); \
    } \
    extern "C" __global__ void kernel_name##_with_policy(__VA_ARGS__, policy_func_t policy_func) { \
        apply_policy_wrapper(kernel_name##_impl, policy_func, ##__VA_ARGS__); \
    } \
    __device__ void kernel_name##_impl(__VA_ARGS__)

// ========================================
// MACRO TO GENERATE POLICY WRAPPER
// ========================================
//
// Use this macro in your .cu file to generate the policy wrapper for your kernel.
// Place it AFTER your kernel implementation.
//
// Syntax: GENERATE_POLICY_WRAPPER(kernel_name, param_types...)
//
// Example:
//   // 1. Define your kernel as __device__
//   extern "C" __device__ void gemm_kernel_impl(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta) {
//       // your kernel code
//   }
//
//   // 2. Generate the policy wrapper (one line!)
//   GENERATE_POLICY_WRAPPER(gemm_kernel, float*, float*, float*, int, int, int, float, float)
//
// This creates:
//   - gemm_kernel_with_policy (the global kernel that PolicyFramework calls)
//
#define GENERATE_POLICY_WRAPPER(kernel_name, ...) \
    extern "C" __global__ void kernel_name##_with_policy(__VA_ARGS__, policy_func_t policy_func) { \
        apply_policy_wrapper(kernel_name##_impl, policy_func); \
    }

// Alternative: Generate wrapper with explicit parameter list
// This version allows you to specify parameter names for clarity
//
// Syntax: GENERATE_POLICY_WRAPPER_WITH_PARAMS(kernel_name, (param_declarations), (param_names))
//
// Example:
//   GENERATE_POLICY_WRAPPER_WITH_PARAMS(gemm_kernel,
//       (float *A, float *B, float *C, int M, int N, int K, float alpha, float beta),
//       (A, B, C, M, N, K, alpha, beta)
//   )
//
#define REMOVE_PARENS(...) __VA_ARGS__
#define GENERATE_POLICY_WRAPPER_WITH_PARAMS(kernel_name, params, args) \
    extern "C" __global__ void kernel_name##_with_policy(REMOVE_PARENS params, policy_func_t policy_func) { \
        apply_policy_wrapper(kernel_name##_impl, policy_func, REMOVE_PARENS args); \
    }

// ========================================
// Example Usage with Different Kernels
// ========================================
//
// EXAMPLE 1: GEMM Kernel
// ----------------------
//   extern "C" __device__ void gemm_kernel_impl(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta) {
//       // kernel body
//   }
//   GENERATE_POLICY_WRAPPER_WITH_PARAMS(gemm_kernel, (float *A, float *B, float *C, int M, int N, int K, float alpha, float beta))
//
// EXAMPLE 2: Vector Add Kernel
// -----------------------------
//   extern "C" __device__ void vec_add_impl(float *a, float *b, float *c, int n) {
//       // kernel body
//   }
//   GENERATE_POLICY_WRAPPER_WITH_PARAMS(vec_add, (float *a, float *b, float *c, int n))
//
// EXAMPLE 3: Matrix Transpose Kernel
// -----------------------------------
//   extern "C" __device__ void transpose_impl(float *in, float *out, int width, int height) {
//       // kernel body
//   }
//   GENERATE_POLICY_WRAPPER_WITH_PARAMS(transpose, (float *in, float *out, int width, int height))
//
// ========================================

#endif // __CUDACC__
