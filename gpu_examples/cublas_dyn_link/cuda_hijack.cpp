// CUDA Runtime API Hijacking for Runtime Policy Enforcement
// This intercepts cudaLaunchKernel calls to inject policy wrapper

#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Original CUDA runtime functions
static cudaError_t (*real_cudaLaunchKernel)(const void *, dim3, dim3, void **, size_t, cudaStream_t) = nullptr;
static cudaError_t (*real_cudaLaunchCooperativeKernel)(const void *, dim3, dim3, void **, size_t, cudaStream_t) = nullptr;
static cudaError_t (*real_cudaLaunchKernelExC)(const cudaLaunchConfig_t *, const void *, void **) = nullptr;

// Original CUDA driver functions
static CUresult (*real_cuLaunchKernel)(CUfunction, unsigned int, unsigned int, unsigned int,
                                       unsigned int, unsigned int, unsigned int,
                                       unsigned int, CUstream, void **, void **) = nullptr;
static CUresult (*real_cuModuleLoad)(CUmodule *, const char *) = nullptr;
static CUresult (*real_cuModuleLoadData)(CUmodule *, const void *) = nullptr;
static CUresult (*real_cuInit)(unsigned int) = nullptr;
static CUresult (*real_cuLinkCreate)(unsigned int, CUjit_option *, void **, CUlinkState *) = nullptr;
static CUresult (*real_cuLinkAddData)(CUlinkState, CUjitInputType, void *, size_t, const char *,
                                      unsigned int, CUjit_option *, void **) = nullptr;
static CUresult (*real_cuLinkComplete)(CUlinkState, void **, size_t *) = nullptr;
static CUresult (*real_cuModuleGetFunction)(CUfunction *, CUmodule, const char *) = nullptr;
static CUresult (*real_cuModuleGetGlobal)(CUdeviceptr *, size_t *, CUmodule, const char *) = nullptr;
static CUresult (*real_cuMemcpyDtoH)(void *, CUdeviceptr, size_t) = nullptr;
static CUresult (*real_cuCtxSynchronize)() = nullptr;
static CUresult (*real_cuLinkDestroy)(CUlinkState) = nullptr;

// Global state
static bool hijack_enabled = false;
static bool hijack_initialized = false;
static CUmodule wrapper_module = nullptr;
static CUfunction wrapper_function = nullptr;

// Load original CUDA functions
static void load_real_cuda_functions() {
    // Load runtime API
    void *rt_handle = dlopen("libcudart.so.12", RTLD_LAZY | RTLD_NOLOAD);
    if (!rt_handle) {
        rt_handle = dlopen("libcudart.so.12", RTLD_LAZY);
    }
    if (rt_handle) {
        real_cudaLaunchKernel = (decltype(real_cudaLaunchKernel))dlsym(rt_handle, "cudaLaunchKernel");
        real_cudaLaunchCooperativeKernel = (decltype(real_cudaLaunchCooperativeKernel))dlsym(rt_handle, "cudaLaunchCooperativeKernel");
        real_cudaLaunchKernelExC = (decltype(real_cudaLaunchKernelExC))dlsym(rt_handle, "cudaLaunchKernelExC");

        printf("[CUDA Hijack] Loaded runtime API functions:\n");
        printf("  cudaLaunchKernel: %p\n", (void*)real_cudaLaunchKernel);
        printf("  cudaLaunchCooperativeKernel: %p\n", (void*)real_cudaLaunchCooperativeKernel);
        printf("  cudaLaunchKernelExC: %p\n", (void*)real_cudaLaunchKernelExC);
    }

    // Load driver API
    void *handle = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_NOLOAD);
    if (!handle) {
        handle = dlopen("libcuda.so.1", RTLD_LAZY);
    }
    if (!handle) {
        fprintf(stderr, "Failed to load libcuda.so.1\n");
        exit(1);
    }

    real_cuLaunchKernel = (decltype(real_cuLaunchKernel))dlsym(handle, "cuLaunchKernel");
    real_cuModuleLoad = (decltype(real_cuModuleLoad))dlsym(handle, "cuModuleLoad");
    real_cuModuleLoadData = (decltype(real_cuModuleLoadData))dlsym(handle, "cuModuleLoadData");
    real_cuInit = (decltype(real_cuInit))dlsym(handle, "cuInit");
    real_cuLinkCreate = (decltype(real_cuLinkCreate))dlsym(handle, "cuLinkCreate");
    real_cuLinkAddData = (decltype(real_cuLinkAddData))dlsym(handle, "cuLinkAddData");
    real_cuLinkComplete = (decltype(real_cuLinkComplete))dlsym(handle, "cuLinkComplete");
    real_cuModuleGetFunction = (decltype(real_cuModuleGetFunction))dlsym(handle, "cuModuleGetFunction");
    real_cuModuleGetGlobal = (decltype(real_cuModuleGetGlobal))dlsym(handle, "cuModuleGetGlobal");
    real_cuMemcpyDtoH = (decltype(real_cuMemcpyDtoH))dlsym(handle, "cuMemcpyDtoH");
    real_cuCtxSynchronize = (decltype(real_cuCtxSynchronize))dlsym(handle, "cuCtxSynchronize");
    real_cuLinkDestroy = (decltype(real_cuLinkDestroy))dlsym(handle, "cuLinkDestroy");
}

// Read cubin file
static void* read_cubin_file(const char* filename, size_t* size) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open %s\n", filename);
        return nullptr;
    }
    fseek(f, 0, SEEK_END);
    *size = ftell(f);
    fseek(f, 0, SEEK_SET);
    void* data = malloc(*size);
    fread(data, 1, *size, f);
    fclose(f);
    return data;
}

// Initialize wrapper kernel for interception
static bool initialize_wrapper() {
    if (hijack_initialized) return true;

    printf("[CUDA Hijack] Initializing policy wrapper...\n");

    // Create linker
    CUlinkState linkState;
    CUresult res = real_cuLinkCreate(0, nullptr, nullptr, &linkState);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "[CUDA Hijack] Failed to create linker: %d\n", res);
        return false;
    }

    // Load wrapper and policy cubins
    size_t wrapper_size, policy_size;
    void* wrapper_data = read_cubin_file("./wrapper_kernel.cubin", &wrapper_size);
    void* policy_data = read_cubin_file("./policy.cubin", &policy_size);

    if (!wrapper_data || !policy_data) {
        fprintf(stderr, "[CUDA Hijack] Failed to load cubin files\n");
        return false;
    }

    printf("[CUDA Hijack] Linking wrapper (%zu bytes) and policy (%zu bytes)\n",
           wrapper_size, policy_size);

    // Add to linker
    res = real_cuLinkAddData(linkState, CU_JIT_INPUT_CUBIN, wrapper_data, wrapper_size,
                            "wrapper_kernel.cubin", 0, nullptr, nullptr);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "[CUDA Hijack] Failed to link wrapper: %d\n", res);
        return false;
    }

    res = real_cuLinkAddData(linkState, CU_JIT_INPUT_CUBIN, policy_data, policy_size,
                            "policy.cubin", 0, nullptr, nullptr);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "[CUDA Hijack] Failed to link policy: %d\n", res);
        return false;
    }

    // Complete linking
    void* linked_cubin;
    size_t linked_size;
    res = real_cuLinkComplete(linkState, &linked_cubin, &linked_size);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "[CUDA Hijack] Failed to complete linking: %d\n", res);
        return false;
    }

    printf("[CUDA Hijack] Linked module size: %zu bytes\n", linked_size);

    // Load module
    res = real_cuModuleLoadData(&wrapper_module, linked_cubin);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "[CUDA Hijack] Failed to load wrapper module: %d\n", res);
        return false;
    }

    // Get wrapper function
    res = real_cuModuleGetFunction(&wrapper_function, wrapper_module, "run_with_policy_kernel");
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "[CUDA Hijack] Failed to get wrapper function: %d\n", res);
        return false;
    }

    // Cleanup
    real_cuLinkDestroy(linkState);
    free(wrapper_data);
    free(policy_data);

    hijack_initialized = true;
    printf("[CUDA Hijack] Initialization complete!\n");
    return true;
}

// Enable/disable hijacking
extern "C" void cuda_hijack_enable() {
    if (!real_cuLaunchKernel) {
        load_real_cuda_functions();
    }
    hijack_enabled = true;
    printf("[CUDA Hijack] Interception ENABLED\n");
}

extern "C" void cuda_hijack_disable() {
    hijack_enabled = false;
    printf("[CUDA Hijack] Interception DISABLED\n");
}

// Intercept cudaLaunchKernel (Runtime API) - THIS IS WHAT cuBLAS USES
extern "C" cudaError_t cudaLaunchKernel(
    const void *func,
    dim3 gridDim,
    dim3 blockDim,
    void **args,
    size_t sharedMem,
    cudaStream_t stream)
{
    if (!real_cudaLaunchKernel) {
        load_real_cuda_functions();
    }

    // Check if this looks like a GEMM kernel (heuristic based on grid/block dimensions)
    bool is_gemm = (gridDim.x > 1 && gridDim.y > 1 && blockDim.x > 1 && blockDim.y > 1);

    if (hijack_enabled && is_gemm) {
        printf("[CUDA Hijack] Intercepting cudasLaunchKernel: grid(%u,%u,%u) block(%u,%u,%u)\n",
               gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z);
        printf("[CUDA Hijack] Kernel function: %p\n", func);
        printf("[CUDA Hijack] This is a cuBLAS kernel launch!\n");

        // TODO: To inject our wrapper:
        // 1. Would need to convert runtime API call to driver API
        // 2. Extract kernel from cuBLAS module
        // 3. Link with our wrapper + policy
        // 4. Launch combined kernel

        printf("[CUDA Hijack] Policy injection point identified\n");
    }

    return real_cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
}

// Intercept cuInit
extern "C" CUresult cuInit(unsigned int Flags) {
    if (!real_cuInit) {
        load_real_cuda_functions();
    }
    printf("[CUDA Hijack] cuInit called\n");
    return real_cuInit(Flags);
}

// Intercept cuLaunchKernel - THIS IS THE KEY FUNCTION
extern "C" CUresult cuLaunchKernel(
    CUfunction f,
    unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes,
    CUstream hStream,
    void **kernelParams,
    void **extra)
{
    if (!real_cuLaunchKernel) {
        load_real_cuda_functions();
    }

    // Check if this is a cuBLAS GEMM kernel (heuristic: check grid/block dimensions)
    bool is_gemm = (gridDimX > 1 && gridDimY > 1 && blockDimX > 1 && blockDimY > 1);

    if (hijack_enabled && is_gemm && kernelParams) {
        printf("[CUDA Hijack] Intercepting kernel launch: grid(%u,%u,%u) block(%u,%u,%u)\n",
               gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ);

        // Initialize wrapper if needed
        if (!initialize_wrapper()) {
            fprintf(stderr, "[CUDA Hijack] Wrapper initialization failed, passing through\n");
            return real_cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ,
                                      blockDimX, blockDimY, blockDimZ,
                                      sharedMemBytes, hStream, kernelParams, extra);
        }

        // For cuBLAS, we need to:
        // 1. Let cuBLAS execute normally (we can't easily wrap it)
        // 2. But we could inject our wrapper if we knew the kernel signature

        // For now, demonstrate that we CAN intercept
        printf("[CUDA Hijack] Would inject policy wrapper here\n");
        printf("[CUDA Hijack] Kernel params: %p\n", kernelParams[0]);

        // TODO: To actually wrap cuBLAS kernel, we would need to:
        // - Extract the original kernel function
        // - Create parameters that match wrapper signature
        // - Launch wrapper with original kernel as parameter

        // For demonstration, fall through to original launch
        printf("[CUDA Hijack] Passing through to original cuBLAS kernel\n");
    }

    return real_cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ,
                              blockDimX, blockDimY, blockDimZ,
                              sharedMemBytes, hStream, kernelParams, extra);
}

// Constructor to initialize on library load
__attribute__((constructor))
static void init_hijack() {
    printf("[CUDA Hijack] Library loaded\n");
    load_real_cuda_functions();

    // Check for environment variable to auto-enable
    const char* enable = getenv("CUDA_HIJACK_ENABLE");
    if (enable && strcmp(enable, "1") == 0) {
        cuda_hijack_enable();
    }
}
