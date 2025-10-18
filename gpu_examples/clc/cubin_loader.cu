// Load and run the CLC cubin using CUDA Driver API
// This bypasses the runtime version check

#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>

#define CHECK_CU(call) \
do { \
    CUresult err = call; \
    if (err != CUDA_SUCCESS) { \
        const char* errStr; \
        cuGetErrorString(err, &errStr); \
        fprintf(stderr, "CUDA Driver API error at %s:%d: %s\n", \
                __FILE__, __LINE__, errStr); \
        exit(1); \
    } \
} while(0)

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("Usage: %s <cubin_file>\n", argv[0]);
        printf("Example: %s /tmp/clc_12.8.cubin\n", argv[0]);
        return 1;
    }

    const char* cubin_path = argv[1];

    // Initialize CUDA Driver API
    CHECK_CU(cuInit(0));

    // Get device
    CUdevice device;
    CHECK_CU(cuDeviceGet(&device, 0));

    // Get device properties
    char name[256];
    int major, minor;
    CHECK_CU(cuDeviceGetName(name, sizeof(name), device));
    CHECK_CU(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
    CHECK_CU(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));

    printf("Device: %s\n", name);
    printf("Compute Capability: %d.%d\n", major, minor);

    if (major < 10) {
        printf("ERROR: CLC requires CC 10.0+\n");
        return 1;
    }

    // Create context
    CUcontext context;
    CHECK_CU(cuCtxCreate(&context, 0, device));

    // Load cubin file
    FILE* f = fopen(cubin_path, "rb");
    if (!f) {
        fprintf(stderr, "Cannot open %s\n", cubin_path);
        return 1;
    }

    fseek(f, 0, SEEK_END);
    size_t cubin_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    void* cubin_data = malloc(cubin_size);
    fread(cubin_data, 1, cubin_size, f);
    fclose(f);

    printf("Loaded cubin: %zu bytes\n", cubin_size);

    // Load module from cubin
    CUmodule module;
    CHECK_CU(cuModuleLoadData(&module, cubin_data));
    printf("Module loaded successfully!\n");

    // Get kernel function
    CUfunction kernel;
    CHECK_CU(cuModuleGetFunction(&kernel, module, "_Z17clc_cuda13_kernelPfiPi"));
    printf("Kernel function found!\n");

    // Allocate memory
    int n = 1024 * 1024;
    size_t size = n * sizeof(float);

    float* h_data = (float*)malloc(size);
    for (int i = 0; i < n; i++) {
        h_data[i] = i % 100;
    }

    CUdeviceptr d_data, d_work_count;
    CHECK_CU(cuMemAlloc(&d_data, size));
    CHECK_CU(cuMemAlloc(&d_work_count, sizeof(int)));
    CHECK_CU(cuMemcpyHtoD(d_data, h_data, size));

    int zero = 0;
    CHECK_CU(cuMemcpyHtoD(d_work_count, &zero, sizeof(int)));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    printf("\nLaunching kernel:\n");
    printf("  Grid: %d blocks\n", blocksPerGrid);
    printf("  Block: %d threads\n", threadsPerBlock);

    void* args[] = {&d_data, &n, &d_work_count};

    CHECK_CU(cuLaunchKernel(
        kernel,
        blocksPerGrid, 1, 1,    // grid dim
        threadsPerBlock, 1, 1,  // block dim
        0,                      // shared mem
        NULL,                   // stream
        args,                   // kernel args
        NULL                    // extra
    ));

    // Wait for completion
    CHECK_CU(cuCtxSynchronize());
    printf("Kernel completed!\n");

    // Copy results back
    CHECK_CU(cuMemcpyDtoH(h_data, d_data, size));

    int h_work_count = 0;
    CHECK_CU(cuMemcpyDtoH(&h_work_count, d_work_count, sizeof(int)));

    // Verify
    bool success = true;
    for (int i = 0; i < 10; i++) {
        float expected = (i % 100) * 2.5f;
        printf("data[%d] = %f (expected %f)\n", i, h_data[i], expected);
        if (h_data[i] != expected) {
            success = false;
        }
    }

    printf("\nResult: %s\n", success ? "✅ PASSED" : "❌ FAILED");
    printf("Work items stolen: %d\n", h_work_count);

    // Cleanup
    cuMemFree(d_data);
    cuMemFree(d_work_count);
    cuModuleUnload(module);
    cuCtxDestroy(context);
    free(h_data);
    free(cubin_data);

    return success ? 0 : 1;
}
