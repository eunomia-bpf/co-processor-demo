// Simple cubin loader using pure C and CUDA Driver API
#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>

#define CHECK_CU(call) \
do { \
    CUresult err = call; \
    if (err != CUDA_SUCCESS) { \
        const char* errStr; \
        cuGetErrorString(err, &errStr); \
        fprintf(stderr, "CUDA error at line %d: %s (code %d)\n", \
                __LINE__, errStr, err); \
        return 1; \
    } \
} while(0)

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("Usage: %s <cubin_file>\n", argv[0]);
        return 1;
    }

    printf("Initializing CUDA Driver API...\n");
    CHECK_CU(cuInit(0));

    CUdevice device;
    CHECK_CU(cuDeviceGet(&device, 0));

    char name[256];
    int major, minor;
    CHECK_CU(cuDeviceGetName(name, sizeof(name), device));
    CHECK_CU(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
    CHECK_CU(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));

    printf("Device: %s (CC %d.%d)\n", name, major, minor);

    if (major < 10) {
        printf("ERROR: Need CC 10.0+ for CLC\n");
        return 1;
    }

    CUcontext context;
    CHECK_CU(cuCtxCreate(&context, 0, device));
    printf("Context created\n");

    // Load cubin
    FILE* f = fopen(argv[1], "rb");
    if (!f) {
        fprintf(stderr, "Cannot open %s\n", argv[1]);
        return 1;
    }

    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fseek(f, 0, SEEK_SET);

    void* data = malloc(size);
    fread(data, 1, size, f);
    fclose(f);

    printf("Loaded cubin: %zu bytes\n", size);

    CUmodule module;
    CUresult res = cuModuleLoadData(&module, data);
    if (res != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(res, &errStr);
        fprintf(stderr, "Failed to load module: %s\n", errStr);

        // Try to get more info
        CUjit_option options[1];
        void* optionVals[1];
        char error_log[8192];

        options[0] = CU_JIT_ERROR_LOG_BUFFER;
        optionVals[0] = (void*)error_log;

        CUlinkState linkState;
        if (cuLinkCreate(1, options, optionVals, &linkState) == CUDA_SUCCESS) {
            cuLinkAddData(linkState, CU_JIT_INPUT_CUBIN, data, size, "kernel", 0, NULL, NULL);
            printf("Link error log:\n%s\n", error_log);
            cuLinkDestroy(linkState);
        }

        free(data);
        cuCtxDestroy(context);
        return 1;
    }

    printf("✅ Module loaded successfully!\n");

    CUfunction kernel;
    res = cuModuleGetFunction(&kernel, module, "_Z17clc_cuda13_kernelPfiPi");
    if (res != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(res, &errStr);
        fprintf(stderr, "Failed to get kernel function: %s\n", errStr);

        // List all functions in the module
        printf("\nAttempting to list functions...\n");

        free(data);
        cuModuleUnload(module);
        cuCtxDestroy(context);
        return 1;
    }

    printf("✅ Kernel function found!\n");

    // Allocate test data
    int n = 1024;
    size_t data_size = n * sizeof(float);

    float* h_data = (float*)malloc(data_size);
    for (int i = 0; i < n; i++) {
        h_data[i] = (float)(i % 100);
    }

    CUdeviceptr d_data, d_work_count;
    CHECK_CU(cuMemAlloc(&d_data, data_size));
    CHECK_CU(cuMemAlloc(&d_work_count, sizeof(int)));
    CHECK_CU(cuMemcpyHtoD(d_data, h_data, data_size));

    int zero = 0;
    CHECK_CU(cuMemcpyHtoD(d_work_count, &zero, sizeof(int)));

    printf("\nLaunching kernel (4 blocks x 256 threads)...\n");

    void* args[] = {&d_data, &n, &d_work_count};

    res = cuLaunchKernel(
        kernel,
        4, 1, 1,        // grid
        256, 1, 1,      // block
        0, NULL,        // shared mem, stream
        args, NULL
    );

    if (res != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(res, &errStr);
        fprintf(stderr, "Kernel launch failed: %s\n", errStr);
    } else {
        printf("Kernel launched...\n");

        res = cuCtxSynchronize();
        if (res != CUDA_SUCCESS) {
            const char* errStr;
            cuGetErrorString(res, &errStr);
            fprintf(stderr, "Kernel execution failed: %s\n", errStr);
        } else {
            printf("✅ Kernel completed!\n");

            CHECK_CU(cuMemcpyDtoH(h_data, d_data, data_size));

            int work_count;
            CHECK_CU(cuMemcpyDtoH(&work_count, d_work_count, sizeof(int)));

            printf("\nResults (first 10):\n");
            int errors = 0;
            for (int i = 0; i < 10; i++) {
                float expected = (float)(i % 100) * 2.5f;
                printf("  [%d] = %.1f (expected %.1f) %s\n",
                       i, h_data[i], expected,
                       (h_data[i] == expected) ? "✓" : "✗");
                if (h_data[i] != expected) errors++;
            }

            printf("\nWork stolen: %d times\n", work_count);
            printf("Verification: %s\n", errors == 0 ? "✅ PASSED" : "❌ FAILED");
        }
    }

    // Cleanup
    cuMemFree(d_data);
    cuMemFree(d_work_count);
    cuModuleUnload(module);
    cuCtxDestroy(context);
    free(h_data);
    free(data);

    return 0;
}
