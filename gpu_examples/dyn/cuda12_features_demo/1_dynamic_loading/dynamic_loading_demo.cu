/**
 * Demo 1: cudaLibrary* Dynamic Loading API
 *
 * This demo shows how to use CUDA 12+ runtime dynamic loading features:
 * - cudaLibraryLoadFromFile: Load a fatbin/cubin from file
 * - cudaLibraryGetKernel: Get kernel handle from library
 * - cudaLaunchKernel: Launch kernel using handle
 * - cudaGetKernel: Get kernel handle from statically linked kernel
 *
 * Benefits:
 * 1. Pure runtime API - no need to mix runtime/driver
 * 2. Plugin-style kernel loading
 * 3. Runtime kernel selection
 * 4. Easier to integrate with your CLC framework
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstring>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// A statically compiled kernel for comparison
__global__ void staticVectorAdd(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
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
    std::cout << "CUDA 12+ Dynamic Loading Demo" << std::endl;
    std::cout << "========================================" << std::endl;

    // Check CUDA version
    int runtimeVersion;
    CHECK_CUDA(cudaRuntimeGetVersion(&runtimeVersion));
    std::cout << "CUDA Runtime Version: " << runtimeVersion / 1000 << "."
              << (runtimeVersion % 100) / 10 << std::endl;

    if (runtimeVersion < 12000) {
        std::cerr << "This demo requires CUDA 12.0 or higher" << std::endl;
        return EXIT_FAILURE;
    }

    // Setup test data
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

    dim3 blockDim(256);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x);

    std::cout << "\n=== Part 1: Static Kernel Launch ===" << std::endl;
    staticVectorAdd<<<gridDim, blockDim>>>(d_a, d_b, d_c, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
    printArray("Result (static)", h_c.data(), N);

    std::cout << "\n=== Part 2: cudaGetKernel (Get handle from static kernel) ===" << std::endl;
    // New in CUDA 12: Get kernel handle from statically compiled kernel
    cudaKernel_t staticKernelHandle;
    CHECK_CUDA(cudaGetKernel(&staticKernelHandle, (const void*)staticVectorAdd));
    std::cout << "✓ Got kernel handle from static kernel: " << staticKernelHandle << std::endl;

    // Launch using handle
    memset(h_c.data(), 0, bytes);
    CHECK_CUDA(cudaMemset(d_c, 0, bytes));

    int n = N;
    void* args[] = {&d_a, &d_b, &d_c, &n};
    CHECK_CUDA(cudaLaunchKernel(staticKernelHandle, gridDim, blockDim, args, 0, nullptr));
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
    printArray("Result (via handle)", h_c.data(), N);

    std::cout << "\n=== Part 3: Dynamic Library Loading ===" << std::endl;

    const char* libPath = "vector_kernels.fatbin";
    if (argc > 1) {
        libPath = argv[1];
    }

    std::cout << "Attempting to load library: " << libPath << std::endl;

    cudaLibrary_t library;
    cudaError_t loadErr = cudaLibraryLoadFromFile(
        &library,
        libPath,
        nullptr,  // jitOptions
        nullptr,  // jitOptionsValues
        0,        // numJitOptions
        nullptr,  // libraryOptions
        nullptr,  // libraryOptionValues
        0         // numLibraryOptions
    );

    if (loadErr == cudaSuccess) {
        std::cout << "✓ Library loaded successfully: " << library << std::endl;

        // Get kernel count
        unsigned int kernelCount = 0;
        CHECK_CUDA(cudaLibraryGetKernelCount(&kernelCount, library));
        std::cout << "✓ Kernel count in library: " << kernelCount << std::endl;

        // Enumerate kernels
        if (kernelCount > 0) {
            std::vector<cudaKernel_t> kernels(kernelCount);
            CHECK_CUDA(cudaLibraryEnumerateKernels(kernels.data(), kernelCount, library));
            std::cout << "✓ Enumerated kernels:" << std::endl;
            for (unsigned int i = 0; i < kernelCount; i++) {
                std::cout << "  Kernel " << i << ": " << kernels[i] << std::endl;
            }
        }

        // Get specific kernel by name
        std::cout << "\nGetting 'vectorAdd' kernel..." << std::endl;
        cudaKernel_t vectorAddKernel;
        cudaError_t getKernelErr = cudaLibraryGetKernel(&vectorAddKernel, library, "vectorAdd");

        if (getKernelErr == cudaSuccess) {
            std::cout << "✓ Got vectorAdd kernel: " << vectorAddKernel << std::endl;

            // Launch the dynamically loaded kernel
            memset(h_c.data(), 0, bytes);
            CHECK_CUDA(cudaMemset(d_c, 0, bytes));

            CHECK_CUDA(cudaLaunchKernel(vectorAddKernel, gridDim, blockDim, args, 0, nullptr));
            CHECK_CUDA(cudaDeviceSynchronize());

            CHECK_CUDA(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
            printArray("Result (dynamic)", h_c.data(), N);

            std::cout << "✓ Dynamic kernel executed successfully!" << std::endl;
        } else {
            std::cout << "⚠ Failed to get 'vectorAdd' kernel: "
                      << cudaGetErrorString(getKernelErr) << std::endl;
        }

        // Try to get vectorMul kernel
        std::cout << "\nGetting 'vectorMul' kernel..." << std::endl;
        cudaKernel_t vectorMulKernel;
        getKernelErr = cudaLibraryGetKernel(&vectorMulKernel, library, "vectorMul");

        if (getKernelErr == cudaSuccess) {
            std::cout << "✓ Got vectorMul kernel: " << vectorMulKernel << std::endl;

            memset(h_c.data(), 0, bytes);
            CHECK_CUDA(cudaMemset(d_c, 0, bytes));

            CHECK_CUDA(cudaLaunchKernel(vectorMulKernel, gridDim, blockDim, args, 0, nullptr));
            CHECK_CUDA(cudaDeviceSynchronize());

            CHECK_CUDA(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
            printArray("Result (multiply)", h_c.data(), N);
        } else {
            std::cout << "⚠ Failed to get 'vectorMul' kernel: "
                      << cudaGetErrorString(getKernelErr) << std::endl;
        }

        // Cleanup library
        CHECK_CUDA(cudaLibraryUnload(library));
        std::cout << "\n✓ Library unloaded" << std::endl;

    } else {
        std::cout << "⚠ Library loading failed: " << cudaGetErrorString(loadErr) << std::endl;
        std::cout << "  (This is expected if .fatbin not created yet)" << std::endl;
        std::cout << "  Run 'make' to build the fatbin first" << std::endl;
    }

    // Cleanup
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));

    std::cout << "\n========================================" << std::endl;
    std::cout << "Key Takeaways:" << std::endl;
    std::cout << "1. cudaGetKernel() - Get handle from static kernel" << std::endl;
    std::cout << "2. cudaLibraryLoadFromFile() - Load fatbin at runtime" << std::endl;
    std::cout << "3. cudaLibraryGetKernel() - Get kernel by name" << std::endl;
    std::cout << "4. cudaLaunchKernel() - Launch with kernel handle" << std::endl;
    std::cout << "5. Pure runtime API - no driver mixing!" << std::endl;
    std::cout << "========================================" << std::endl;

    return EXIT_SUCCESS;
}
