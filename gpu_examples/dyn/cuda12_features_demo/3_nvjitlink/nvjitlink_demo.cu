/**
 * Demo 3: nvJitLink + JIT LTO
 *
 * This demo shows how to use CUDA 12+ nvJitLink library:
 * - nvJitLinkCreate: Create a JIT linker instance
 * - nvJitLinkAddData: Add PTX/CUBIN/LTO-IR to link
 * - nvJitLinkComplete: Perform JIT linking with LTO
 * - Runtime linking with optimization
 *
 * This is PERFECT for your CLC framework:
 * 1. Compile user_kernel.cu -> PTX/LTO-IR
 * 2. Compile policy.cu -> PTX/LTO-IR
 * 3. Link them at runtime with nvJitLink
 * 4. Get optimized binary (LTO can inline across modules!)
 *
 * Benefits vs old cuLink*:
 * - Independent of driver version
 * - Supports LTO (whole-program optimization at runtime!)
 * - Better performance than separate compilation
 * - Cleaner API
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvJitLink.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>

#define CHECK_NVJITLINK(call) { \
    nvJitLinkResult err = call; \
    if (err != NVJITLINK_SUCCESS) { \
        std::cerr << "nvJitLink Error at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CU(call) { \
    CUresult err = call; \
    if (err != CUDA_SUCCESS) { \
        const char* errStr; \
        cuGetErrorString(err, &errStr); \
        std::cerr << "CUDA Driver Error: " << errStr \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

std::vector<char> readFile(const char* filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Failed to open " << filename << std::endl;
        return {};
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (file.read(buffer.data(), size)) {
        return buffer;
    }
    return {};
}

void printArray(const char* name, const float* arr, int n, int limit = 10) {
    std::cout << name << ": [";
    for (int i = 0; i < std::min(n, limit); i++) {
        std::cout << arr[i];
        if (i < std::min(n, limit) - 1) std::cout << ", ";
    }
    if (n > limit) std::cout << " ...";
    std::cout << "]" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "CUDA 12+ nvJitLink + LTO Demo" << std::endl;
    std::cout << "========================================" << std::endl;

    // Initialize CUDA
    CHECK_CU(cuInit(0));

    CUdevice device;
    CHECK_CU(cuDeviceGet(&device, 0));

    char deviceName[256];
    CHECK_CU(cuDeviceGetName(deviceName, sizeof(deviceName), device));
    std::cout << "Device: " << deviceName << std::endl;

    int computeCapabilityMajor, computeCapabilityMinor;
    CHECK_CU(cuDeviceGetAttribute(&computeCapabilityMajor,
                                  CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
    CHECK_CU(cuDeviceGetAttribute(&computeCapabilityMinor,
                                  CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));

    std::cout << "Compute Capability: " << computeCapabilityMajor << "."
              << computeCapabilityMinor << std::endl;

    CUcontext context;
    CHECK_CU(cuCtxCreate(&context, 0, device));

    std::cout << "\n=== Part 1: Load PTX files to link ===" << std::endl;

    // Read PTX files
    auto userPTX = readFile("user_kernel.ptx");
    auto policyPTX = readFile("policy.ptx");

    if (userPTX.empty()) {
        std::cerr << "⚠ user_kernel.ptx not found. Run 'make ptx' first." << std::endl;
        return EXIT_FAILURE;
    }

    if (policyPTX.empty()) {
        std::cerr << "⚠ policy.ptx not found. Run 'make ptx' first." << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "✓ Loaded user_kernel.ptx (" << userPTX.size() << " bytes)" << std::endl;
    std::cout << "✓ Loaded policy.ptx (" << policyPTX.size() << " bytes)" << std::endl;

    std::cout << "\n=== Part 2: Create nvJitLink linker ===" << std::endl;

    nvJitLinkHandle handle;

    // Prepare linker options
    char archOpt[32];
    snprintf(archOpt, sizeof(archOpt), "-arch=sm_%d%d",
             computeCapabilityMajor, computeCapabilityMinor);

    const char* options[] = {
        archOpt,
        "-lto",  // Enable Link-Time Optimization!
        "-O3"    // Optimize
    };

    CHECK_NVJITLINK(nvJitLinkCreate(&handle, 3, options));
    std::cout << "✓ Created nvJitLink handle with options:" << std::endl;
    for (int i = 0; i < 3; i++) {
        std::cout << "    " << options[i] << std::endl;
    }

    std::cout << "\n=== Part 3: Add PTX inputs and link ===" << std::endl;

    // Add user kernel PTX
    CHECK_NVJITLINK(nvJitLinkAddData(handle, NVJITLINK_INPUT_PTX,
                                     userPTX.data(), userPTX.size(),
                                     "user_kernel"));
    std::cout << "✓ Added user_kernel.ptx" << std::endl;

    // Add policy PTX
    CHECK_NVJITLINK(nvJitLinkAddData(handle, NVJITLINK_INPUT_PTX,
                                     policyPTX.data(), policyPTX.size(),
                                     "policy"));
    std::cout << "✓ Added policy.ptx" << std::endl;

    // Complete the link
    std::cout << "\nLinking with LTO..." << std::endl;
    nvJitLinkResult linkResult = nvJitLinkComplete(handle);

    if (linkResult != NVJITLINK_SUCCESS) {
        // Get error log
        size_t logSize;
        nvJitLinkGetErrorLogSize(handle, &logSize);
        if (logSize > 0) {
            std::vector<char> log(logSize);
            nvJitLinkGetErrorLog(handle, log.data());
            std::cerr << "Link error log:\n" << log.data() << std::endl;
        }
        std::cerr << "✗ Linking failed" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "✓ Linking completed successfully!" << std::endl;

    // Get info log
    size_t infoLogSize;
    if (nvJitLinkGetInfoLogSize(handle, &infoLogSize) == NVJITLINK_SUCCESS && infoLogSize > 1) {
        std::vector<char> infoLog(infoLogSize);
        nvJitLinkGetInfoLog(handle, infoLog.data());
        std::cout << "\nInfo log:\n" << infoLog.data() << std::endl;
    }

    // Get the linked cubin
    size_t cubinSize;
    CHECK_NVJITLINK(nvJitLinkGetLinkedCubinSize(handle, &cubinSize));
    std::vector<char> cubin(cubinSize);
    CHECK_NVJITLINK(nvJitLinkGetLinkedCubin(handle, cubin.data()));

    std::cout << "✓ Generated linked CUBIN (" << cubinSize << " bytes)" << std::endl;

    // Destroy linker
    CHECK_NVJITLINK(nvJitLinkDestroy(&handle));

    std::cout << "\n=== Part 4: Load and execute linked module ===" << std::endl;

    CUmodule module;
    CHECK_CU(cuModuleLoadData(&module, cubin.data()));
    std::cout << "✓ Loaded linked module" << std::endl;

    CUfunction kernel;
    CHECK_CU(cuModuleGetFunction(&kernel, module, "user_kernel_with_policy"));
    std::cout << "✓ Got kernel function: user_kernel_with_policy" << std::endl;

    // Prepare test data
    const int N = 1024;
    const int bytes = N * sizeof(float);

    std::vector<float> h_a(N), h_b(N), h_c(N);
    for (int i = 0; i < N; i++) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 2);
    }

    float *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc(&d_a, bytes));
    CHECK_CUDA(cudaMalloc(&d_b, bytes));
    CHECK_CUDA(cudaMalloc(&d_c, bytes));

    CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    int n = N;
    void* args[] = {&d_a, &d_b, &d_c, &n};

    CHECK_CU(cuLaunchKernel(kernel, (N + 255) / 256, 1, 1, 256, 1, 1, 0, nullptr, args, nullptr));
    CHECK_CU(cuCtxSynchronize());

    std::cout << "✓ Kernel launched and completed" << std::endl;

    CHECK_CUDA(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
    printArray("Result", h_c.data(), N);

    // Verify result (compute_element: a*a + b*b + 2*a*b = (a+b)^2)
    bool correct = true;
    for (int i = 0; i < N && i < 10; i++) {
        float expected = (h_a[i] + h_b[i]) * (h_a[i] + h_b[i]);
        if (std::abs(h_c[i] - expected) > 1e-5) {
            correct = false;
            std::cout << "Mismatch at " << i << ": " << h_c[i] << " != " << expected << std::endl;
        }
    }

    if (correct) {
        std::cout << "✓ Results verified!" << std::endl;
    }

    // Cleanup
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CU(cuModuleUnload(module));
    CHECK_CU(cuCtxDestroy(context));

    std::cout << "\n========================================" << std::endl;
    std::cout << "Key Takeaways:" << std::endl;
    std::cout << "1. nvJitLink - Modern JIT linker API" << std::endl;
    std::cout << "2. LTO support - Whole-program optimization!" << std::endl;
    std::cout << "3. Can inline across modules at runtime" << std::endl;
    std::cout << "4. Perfect for runtime policy injection" << std::endl;
    std::cout << "5. Independent of driver version" << std::endl;
    std::cout << "\nFor your CLC framework:" << std::endl;
    std::cout << "  user.cu -> PTX -> \\        " << std::endl;
    std::cout << "                     nvJitLink + LTO -> optimized binary" << std::endl;
    std::cout << "  policy.cu -> PTX -> /      " << std::endl;
    std::cout << "========================================" << std::endl;

    return EXIT_SUCCESS;
}
