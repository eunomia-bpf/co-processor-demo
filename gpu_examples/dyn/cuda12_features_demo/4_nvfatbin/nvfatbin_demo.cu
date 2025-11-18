/**
 * Demo 4: nvFatbin - Runtime Fatbin Construction
 *
 * This demo shows how to use CUDA 12.4+ nvFatbin library:
 * - nvFatbinCreate: Create a fatbin builder
 * - nvFatbinAddPTX/AddCubin/AddLTOIR: Add multiple architectures/formats
 * - nvFatbinGet: Get the constructed fatbin
 * - Load it with cudaLibraryLoadData
 *
 * This is PERFECT for your binary rewriting use case:
 * 1. Extract original kernel from binary
 * 2. Compile your wrapper/policy to PTX/cubin
 * 3. Use nvFatbin to package them together
 * 4. Load the new fatbin at runtime
 *
 * Benefits:
 * - No need to reverse-engineer fatbin format
 * - Official API for multi-arch deployment
 * - Can mix PTX/cubin/LTO-IR
 * - Works with cudaLibraryLoadData
 */

#include <cuda_runtime.h>
#include <nvFatbin.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>

#define CHECK_NVFATBIN(call) { \
    nvFatbinResult err = call; \
    if (err != NVFATBIN_SUCCESS) { \
        std::cerr << "nvFatbin Error: " << nvFatbinGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
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

void writeFile(const char* filename, const void* data, size_t size) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to write " << filename << std::endl;
        return;
    }
    file.write(static_cast<const char*>(data), size);
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

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "CUDA 12.4+ nvFatbin Demo" << std::endl;
    std::cout << "========================================" << std::endl;

    // Check nvFatbin version
    unsigned int major, minor;
    CHECK_NVFATBIN(nvFatbinVersion(&major, &minor));
    std::cout << "nvFatbin Version: " << major << "." << minor << std::endl;

    int runtimeVersion;
    CHECK_CUDA(cudaRuntimeGetVersion(&runtimeVersion));
    std::cout << "CUDA Runtime Version: " << runtimeVersion / 1000 << "."
              << (runtimeVersion % 100) / 10 << std::endl;

    // Get device info
    int device;
    CHECK_CUDA(cudaGetDevice(&device));

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;

    char archStr[16];
    snprintf(archStr, sizeof(archStr), "%d%d", prop.major, prop.minor);

    std::cout << "\n=== Part 1: Load PTX/Cubin files ===" << std::endl;

    // For this demo, we'll use two kernels compiled separately
    auto kernel1PTX = readFile("kernel1.ptx");
    auto kernel2PTX = readFile("kernel2.ptx");

    // Alternative: Load cubins
    auto kernel1Cubin = readFile("kernel1.cubin");
    auto kernel2Cubin = readFile("kernel2.cubin");

    if (kernel1PTX.empty() && kernel1Cubin.empty()) {
        std::cerr << "⚠ No kernel files found. Run 'make kernels' first." << std::endl;
        std::cout << "\nDemo will show the API usage anyway..." << std::endl;
    }

    std::cout << "\n=== Part 2: Create nvFatbin handle ===" << std::endl;

    nvFatbinHandle handle;
    const char* options[] = {
        "-64",              // 64-bit
        "-compress=true",   // Enable compression
        "-host=linux"       // Host OS
    };

    CHECK_NVFATBIN(nvFatbinCreate(&handle, options, 3));
    std::cout << "✓ Created nvFatbin handle with options:" << std::endl;
    for (int i = 0; i < 3; i++) {
        std::cout << "    " << options[i] << std::endl;
    }

    std::cout << "\n=== Part 3: Add entries to fatbin ===" << std::endl;

    int entriesAdded = 0;

    // Add PTX entries (portable, can JIT compile for any architecture)
    if (!kernel1PTX.empty()) {
        std::cout << "Adding kernel1.ptx..." << std::endl;
        nvFatbinResult res = nvFatbinAddPTX(handle,
                                            kernel1PTX.data(),
                                            kernel1PTX.size(),
                                            archStr,        // Target arch
                                            "kernel1",      // Identifier
                                            nullptr);       // PTXAS options
        if (res == NVFATBIN_SUCCESS) {
            std::cout << "✓ Added kernel1.ptx (arch=" << archStr << ")" << std::endl;
            entriesAdded++;
        } else {
            std::cout << "⚠ Failed to add kernel1.ptx: " << nvFatbinGetErrorString(res) << std::endl;
        }
    }

    if (!kernel2PTX.empty()) {
        std::cout << "Adding kernel2.ptx..." << std::endl;
        nvFatbinResult res = nvFatbinAddPTX(handle,
                                            kernel2PTX.data(),
                                            kernel2PTX.size(),
                                            archStr,
                                            "kernel2",
                                            nullptr);
        if (res == NVFATBIN_SUCCESS) {
            std::cout << "✓ Added kernel2.ptx (arch=" << archStr << ")" << std::endl;
            entriesAdded++;
        } else {
            std::cout << "⚠ Failed to add kernel2.ptx: " << nvFatbinGetErrorString(res) << std::endl;
        }
    }

    // Add cubin entries (pre-compiled, architecture-specific)
    if (!kernel1Cubin.empty()) {
        std::cout << "Adding kernel1.cubin..." << std::endl;
        nvFatbinResult res = nvFatbinAddCubin(handle,
                                              kernel1Cubin.data(),
                                              kernel1Cubin.size(),
                                              archStr,
                                              "kernel1_cubin");
        if (res == NVFATBIN_SUCCESS) {
            std::cout << "✓ Added kernel1.cubin (arch=" << archStr << ")" << std::endl;
            entriesAdded++;
        } else {
            std::cout << "⚠ Failed to add kernel1.cubin: " << nvFatbinGetErrorString(res) << std::endl;
        }
    }

    if (!kernel2Cubin.empty()) {
        std::cout << "Adding kernel2.cubin..." << std::endl;
        nvFatbinResult res = nvFatbinAddCubin(handle,
                                              kernel2Cubin.data(),
                                              kernel2Cubin.size(),
                                              archStr,
                                              "kernel2_cubin");
        if (res == NVFATBIN_SUCCESS) {
            std::cout << "✓ Added kernel2.cubin (arch=" << archStr << ")" << std::endl;
            entriesAdded++;
        } else {
            std::cout << "⚠ Failed to add kernel2.cubin: " << nvFatbinGetErrorString(res) << std::endl;
        }
    }

    // You can also add multiple architectures:
    // nvFatbinAddPTX(handle, ptx_data, size, "80", "kernel_sm80", nullptr);
    // nvFatbinAddPTX(handle, ptx_data, size, "86", "kernel_sm86", nullptr);
    // nvFatbinAddPTX(handle, ptx_data, size, "90", "kernel_sm90", nullptr);

    if (entriesAdded == 0) {
        std::cout << "\n⚠ No entries added (kernels not compiled)" << std::endl;
        std::cout << "Run 'make kernels' to build kernel files" << std::endl;
        std::cout << "\nShowing API workflow anyway...\n" << std::endl;
    }

    std::cout << "\n=== Part 4: Finalize and get fatbin ===" << std::endl;

    size_t fatbinSize;
    CHECK_NVFATBIN(nvFatbinSize(handle, &fatbinSize));
    std::cout << "✓ Fatbin size: " << fatbinSize << " bytes" << std::endl;

    std::vector<char> fatbinData(fatbinSize);
    CHECK_NVFATBIN(nvFatbinGet(handle, fatbinData.data()));
    std::cout << "✓ Got fatbin data" << std::endl;

    // Save to file
    writeFile("runtime_generated.fatbin", fatbinData.data(), fatbinSize);
    std::cout << "✓ Saved to runtime_generated.fatbin" << std::endl;

    // Cleanup handle
    CHECK_NVFATBIN(nvFatbinDestroy(&handle));
    std::cout << "✓ Destroyed nvFatbin handle" << std::endl;

    if (entriesAdded > 0) {
        std::cout << "\n=== Part 5: Load and use the fatbin ===" << std::endl;

        cudaLibrary_t library;
        cudaError_t loadErr = cudaLibraryLoadData(&library,
                                                   fatbinData.data(),
                                                   nullptr, nullptr, 0,
                                                   nullptr, nullptr, 0);

        if (loadErr == cudaSuccess) {
            std::cout << "✓ Loaded fatbin as CUDA library" << std::endl;

            unsigned int kernelCount;
            CHECK_CUDA(cudaLibraryGetKernelCount(&kernelCount, library));
            std::cout << "✓ Kernel count: " << kernelCount << std::endl;

            // Try to get a kernel and launch it
            if (kernelCount > 0) {
                cudaKernel_t kernel;
                cudaError_t getErr = cudaLibraryGetKernel(&kernel, library, "vectorAdd");

                if (getErr == cudaSuccess) {
                    std::cout << "✓ Got 'vectorAdd' kernel" << std::endl;

                    // Test data
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
                    int n = N;
                    void* args[] = {&d_a, &d_b, &d_c, &n};

                    CHECK_CUDA(cudaLaunchKernel(kernel, gridDim, blockDim, args, 0, nullptr));
                    CHECK_CUDA(cudaDeviceSynchronize());

                    CHECK_CUDA(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
                    printArray("Result", h_c.data(), N);

                    CHECK_CUDA(cudaFree(d_a));
                    CHECK_CUDA(cudaFree(d_b));
                    CHECK_CUDA(cudaFree(d_c));

                    std::cout << "✓ Successfully executed kernel from runtime-built fatbin!" << std::endl;
                } else {
                    std::cout << "⚠ 'vectorAdd' kernel not found (expected)" << std::endl;
                }
            }

            CHECK_CUDA(cudaLibraryUnload(library));
            std::cout << "✓ Unloaded library" << std::endl;
        } else {
            std::cout << "⚠ Failed to load fatbin: " << cudaGetErrorString(loadErr) << std::endl;
        }
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "Key Takeaways:" << std::endl;
    std::cout << "1. nvFatbinCreate() - Build fatbin at runtime" << std::endl;
    std::cout << "2. nvFatbinAddPTX/AddCubin/AddLTOIR() - Add entries" << std::endl;
    std::cout << "3. Can mix multiple formats and architectures" << std::endl;
    std::cout << "4. nvFatbinGet() - Get final fatbin binary" << std::endl;
    std::cout << "5. Use with cudaLibraryLoadData()" << std::endl;
    std::cout << "\nFor your CLC framework:" << std::endl;
    std::cout << "  1. Extract user kernel from binary" << std::endl;
    std::cout << "  2. Compile wrapper/policy to cubin" << std::endl;
    std::cout << "  3. nvFatbin to package together" << std::endl;
    std::cout << "  4. Load and run the new fatbin" << std::endl;
    std::cout << "  → No need to reverse-engineer fatbin format!" << std::endl;
    std::cout << "========================================" << std::endl;

    return EXIT_SUCCESS;
}
