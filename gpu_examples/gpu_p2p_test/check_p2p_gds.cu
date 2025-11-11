#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error [%s]: %s\n", msg, cudaGetErrorString(err));
    }
}

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    checkCudaError(err, "cudaGetDeviceCount");

    printf("=== GPU P2P and Storage DMA Support Check ===\n\n");
    printf("Number of CUDA devices: %d\n\n", deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, i);
        checkCudaError(err, "cudaGetDeviceProperties");

        printf("Device %d: %s\n", i, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  PCIe Bus ID: %d:%d.%d\n", prop.pciBusID, prop.pciDeviceID, prop.pciDomainID);
        printf("  Unified Addressing: %s\n", prop.unifiedAddressing ? "YES" : "NO");
        printf("  Managed Memory: %s\n", prop.managedMemory ? "YES" : "NO");
        printf("  Concurrent Managed Access: %s\n", prop.concurrentManagedAccess ? "YES" : "NO");

        // Check if device can access host memory directly
        printf("  Can Map Host Memory: %s\n", prop.canMapHostMemory ? "YES" : "NO");

        // Check GPUDirect RDMA support
        int gpuDirectRDMASupported = 0;
        err = cudaDeviceGetAttribute(&gpuDirectRDMASupported, cudaDevAttrGPUDirectRDMASupported, i);
        if (err == cudaSuccess) {
            printf("  GPUDirect RDMA: %s\n", gpuDirectRDMASupported ? "SUPPORTED" : "NOT SUPPORTED");
        }

        // Check GPUDirect RDMA flush writes support
        int gpuDirectRDMAFlush = 0;
        err = cudaDeviceGetAttribute(&gpuDirectRDMAFlush, cudaDevAttrGPUDirectRDMAFlushWritesOptions, i);
        if (err == cudaSuccess) {
            printf("  GPUDirect RDMA Flush Writes: %d\n", gpuDirectRDMAFlush);
        }

        // Check GPUDirect RDMA writes ordering
        int gpuDirectRDMAOrder = 0;
        err = cudaDeviceGetAttribute(&gpuDirectRDMAOrder, cudaDevAttrGPUDirectRDMAWritesOrdering, i);
        if (err == cudaSuccess) {
            printf("  GPUDirect RDMA Writes Ordering: %d\n", gpuDirectRDMAOrder);
        }

        printf("\n");
    }

    // Check P2P access between multiple GPUs
    if (deviceCount > 1) {
        printf("=== GPU P2P Access Matrix ===\n");
        for (int i = 0; i < deviceCount; i++) {
            for (int j = 0; j < deviceCount; j++) {
                if (i != j) {
                    int canAccessPeer = 0;
                    err = cudaDeviceCanAccessPeer(&canAccessPeer, i, j);
                    checkCudaError(err, "cudaDeviceCanAccessPeer");
                    printf("GPU %d -> GPU %d: %s\n", i, j, canAccessPeer ? "YES" : "NO");
                }
            }
        }
        printf("\n");
    }

    // Check for GPUDirect Storage (cuFile) support via driver
    printf("=== GPUDirect Storage (GDS) Indicators ===\n");
    printf("cuFile library check: Use 'ldconfig -p | grep cufile' to verify\n");
    printf("For full GDS support, you need:\n");
    printf("  1. Compatible NVMe/storage device with GDS driver\n");
    printf("  2. nvidia-fs kernel module loaded\n");
    printf("  3. cuFile library installed\n");
    printf("  4. Proper /etc/cufile.json configuration\n\n");

    // Check if nvidia-fs module info is accessible
    printf("Check nvidia-fs module: Run 'lsmod | grep nvidia_fs'\n");
    printf("Check GDS status: Run 'cat /proc/driver/nvidia-fs/stats' if available\n");

    return 0;
}
