/*
 * GPU File I/O Benchmark using cuFile
 *
 * Benchmarks different approaches to GPU-File I/O:
 * 1. Standard copy (GPU→RAM→File) - baseline
 * 2. cuFile DMA (GPU→File direct) - P2P DMA
 * 3. GPU kernel processing - to show difference
 *
 * Use this to benchmark file access from GPU perspective.
 */

#include <cuda_runtime.h>
#include <cufile.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// Simple GPU kernel to fill data
__global__ void fillData(unsigned int* data, size_t n, unsigned int seed) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = seed + idx;
    }
}

// GPU kernel to verify data
__global__ void verifyData(unsigned int* data, size_t n, unsigned int seed, int* errors) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (data[idx] != seed + idx) {
            atomicAdd(errors, 1);
        }
    }
}

// GPU kernel example: XOR transform
__global__ void xorTransform(unsigned int* data, size_t n, unsigned int key) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] ^= key;
    }
}

double getTime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

typedef struct {
    double time_ms;
    double bandwidth_gbps;
    int success;
} BenchResult;

// Benchmark 1: Standard copy approach
BenchResult bench_standard_write(void* d_data, size_t size, const char* path) {
    BenchResult result = {0};
    void* h_data = malloc(size);
    if (!h_data) {
        result.success = 0;
        return result;
    }

    double start = getTime();
    CHECK_CUDA(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));
    int fd = open(path, O_CREAT | O_WRONLY | O_TRUNC, 0644);
    if (fd < 0) {
        free(h_data);
        result.success = 0;
        return result;
    }
    write(fd, h_data, size);
    close(fd);
    double elapsed = getTime() - start;

    free(h_data);
    result.time_ms = elapsed * 1000;
    result.bandwidth_gbps = (size / (1024.0*1024.0*1024.0)) / elapsed;
    result.success = 1;
    return result;
}

// Benchmark 2: cuFile DMA write
BenchResult bench_cufile_write(void* d_data, size_t size, const char* path) {
    BenchResult result = {0};

    CUfileError_t status = cuFileDriverOpen();
    if (status.err != CU_FILE_SUCCESS) {
        result.success = 0;
        return result;
    }

    int fd = open(path, O_CREAT | O_RDWR | O_DIRECT, 0644);
    if (fd < 0) {
        fd = open(path, O_CREAT | O_RDWR, 0644);
    }
    if (fd < 0) {
        cuFileDriverClose();
        result.success = 0;
        return result;
    }
    ftruncate(fd, size);

    CUfileDescr_t cf_descr;
    memset(&cf_descr, 0, sizeof(CUfileDescr_t));
    cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    cf_descr.handle.fd = fd;

    CUfileHandle_t cf_handle;
    status = cuFileHandleRegister(&cf_handle, &cf_descr);
    if (status.err != CU_FILE_SUCCESS) {
        close(fd);
        cuFileDriverClose();
        result.success = 0;
        return result;
    }

    double start = getTime();
    ssize_t written = cuFileWrite(cf_handle, d_data, size, 0, 0);
    cudaDeviceSynchronize();
    double elapsed = getTime() - start;

    cuFileHandleDeregister(cf_handle);
    close(fd);
    cuFileDriverClose();

    if (written < 0) {
        result.success = 0;
        return result;
    }

    result.time_ms = elapsed * 1000;
    result.bandwidth_gbps = (size / (1024.0*1024.0*1024.0)) / elapsed;
    result.success = 1;
    return result;
}

// Benchmark 3: cuFile DMA read
BenchResult bench_cufile_read(void* d_data, size_t size, const char* path) {
    BenchResult result = {0};

    CUfileError_t status = cuFileDriverOpen();
    if (status.err != CU_FILE_SUCCESS) {
        result.success = 0;
        return result;
    }

    int fd = open(path, O_RDONLY | O_DIRECT);
    if (fd < 0) {
        fd = open(path, O_RDONLY);
    }
    if (fd < 0) {
        cuFileDriverClose();
        result.success = 0;
        return result;
    }

    CUfileDescr_t cf_descr;
    memset(&cf_descr, 0, sizeof(CUfileDescr_t));
    cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    cf_descr.handle.fd = fd;

    CUfileHandle_t cf_handle;
    status = cuFileHandleRegister(&cf_handle, &cf_descr);
    if (status.err != CU_FILE_SUCCESS) {
        close(fd);
        cuFileDriverClose();
        result.success = 0;
        return result;
    }

    CHECK_CUDA(cudaMemset(d_data, 0, size));

    double start = getTime();
    ssize_t bytes_read = cuFileRead(cf_handle, d_data, size, 0, 0);
    cudaDeviceSynchronize();
    double elapsed = getTime() - start;

    cuFileHandleDeregister(cf_handle);
    close(fd);
    cuFileDriverClose();

    if (bytes_read < 0) {
        result.success = 0;
        return result;
    }

    result.time_ms = elapsed * 1000;
    result.bandwidth_gbps = (size / (1024.0*1024.0*1024.0)) / elapsed;
    result.success = 1;
    return result;
}

// Benchmark 4: Standard read
BenchResult bench_standard_read(void* d_data, size_t size, const char* path) {
    BenchResult result = {0};
    void* h_data = malloc(size);
    if (!h_data) {
        result.success = 0;
        return result;
    }

    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        free(h_data);
        result.success = 0;
        return result;
    }

    double start = getTime();
    read(fd, h_data, size);
    CHECK_CUDA(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
    double elapsed = getTime() - start;

    close(fd);
    free(h_data);

    result.time_ms = elapsed * 1000;
    result.bandwidth_gbps = (size / (1024.0*1024.0*1024.0)) / elapsed;
    result.success = 1;
    return result;
}

void print_usage(const char* prog) {
    printf("GPU File I/O Benchmark\n");
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  -s SIZE    Data size in MB (default: 512)\n");
    printf("  -f FILE    Test file path (default: /tmp/gpu_bench.dat)\n");
    printf("  -r RUNS    Number of runs (default: 3)\n");
    printf("  -m MODE    Mode: all, dma, compare (default: all)\n");
    printf("  -h         Show this help\n");
    printf("\nModes:\n");
    printf("  all      - Run all benchmarks (standard + cuFile + GPU kernel)\n");
    printf("  dma      - Only cuFile DMA benchmarks\n");
    printf("  compare  - Compare standard vs cuFile\n");
}

int main(int argc, char** argv) {
    size_t sizeMB = 512;
    const char* filepath = "/tmp/gpu_bench.dat";
    int runs = 3;
    const char* mode = "all";

    // Parse arguments
    int opt;
    while ((opt = getopt(argc, argv, "s:f:r:m:h")) != -1) {
        switch (opt) {
            case 's': sizeMB = atoi(optarg); break;
            case 'f': filepath = optarg; break;
            case 'r': runs = atoi(optarg); break;
            case 'm': mode = optarg; break;
            case 'h': print_usage(argv[0]); return 0;
            default: print_usage(argv[0]); return 1;
        }
    }

    size_t dataSize = sizeMB * 1024 * 1024;
    size_t numElements = dataSize / sizeof(unsigned int);

    printf("=============================================================\n");
    printf("GPU File I/O Benchmark\n");
    printf("=============================================================\n");
    printf("Size: %zu MB (%zu bytes)\n", sizeMB, dataSize);
    printf("File: %s\n", filepath);
    printf("Runs: %d\n", runs);
    printf("Mode: %s\n", mode);

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("GPU:  %s\n", prop.name);

    // Check configuration
    printf("\nConfiguration:\n");
    system("grep 'use_pci_p2pdma' /etc/cufile.json 2>/dev/null | head -1 || echo '  (cufile.json not checked)'");

    printf("\n");

    // Allocate GPU memory
    unsigned int* d_data;
    CHECK_CUDA(cudaMalloc(&d_data, dataSize));

    // Fill with pattern
    unsigned int seed = 0x12345678;
    int blockSize = 256;
    int numBlocks = (numElements + blockSize - 1) / blockSize;
    fillData<<<numBlocks, blockSize>>>(d_data, numElements, seed);
    CHECK_CUDA(cudaDeviceSynchronize());

    bool run_all = (strcmp(mode, "all") == 0);
    bool run_dma = (strcmp(mode, "dma") == 0);
    bool run_compare = (strcmp(mode, "compare") == 0);

    // Run benchmarks
    if (run_compare || run_all) {
        printf("=============================================================\n");
        printf("BENCHMARK: Standard Copy vs cuFile DMA\n");
        printf("=============================================================\n");

        double std_write_avg = 0, cufile_write_avg = 0;
        double std_read_avg = 0, cufile_read_avg = 0;

        for (int i = 0; i < runs; i++) {
            printf("\n--- Run %d/%d ---\n", i+1, runs);

            printf("Standard Write: ");
            fflush(stdout);
            BenchResult r1 = bench_standard_write(d_data, dataSize, "/tmp/std_write.dat");
            if (r1.success) {
                printf("%.2f GB/s (%.1f ms)\n", r1.bandwidth_gbps, r1.time_ms);
                std_write_avg += r1.bandwidth_gbps;
            } else {
                printf("FAILED\n");
            }

            printf("cuFile Write:   ");
            fflush(stdout);
            BenchResult r2 = bench_cufile_write(d_data, dataSize, filepath);
            if (r2.success) {
                printf("%.2f GB/s (%.1f ms)\n", r2.bandwidth_gbps, r2.time_ms);
                cufile_write_avg += r2.bandwidth_gbps;
            } else {
                printf("FAILED\n");
            }

            printf("cuFile Read:    ");
            fflush(stdout);
            BenchResult r3 = bench_cufile_read(d_data, dataSize, filepath);
            if (r3.success) {
                printf("%.2f GB/s (%.1f ms)\n", r3.bandwidth_gbps, r3.time_ms);
                cufile_read_avg += r3.bandwidth_gbps;
            } else {
                printf("FAILED\n");
            }

            printf("Standard Read:  ");
            fflush(stdout);
            BenchResult r4 = bench_standard_read(d_data, dataSize, filepath);
            if (r4.success) {
                printf("%.2f GB/s (%.1f ms)\n", r4.bandwidth_gbps, r4.time_ms);
                std_read_avg += r4.bandwidth_gbps;
            } else {
                printf("FAILED\n");
            }
        }

        printf("\n=============================================================\n");
        printf("AVERAGE RESULTS (%d runs)\n", runs);
        printf("=============================================================\n");
        printf("WRITE:\n");
        printf("  Standard (GPU→RAM→File):  %.2f GB/s\n", std_write_avg / runs);
        printf("  cuFile   (GPU→File DMA):  %.2f GB/s", cufile_write_avg / runs);
        if (cufile_write_avg > std_write_avg) {
            printf("  [+%.0f%% faster]\n", (cufile_write_avg/std_write_avg - 1) * 100);
        } else {
            printf("\n");
        }

        printf("\nREAD:\n");
        printf("  Standard (File→RAM→GPU):  %.2f GB/s\n", std_read_avg / runs);
        printf("  cuFile   (File→GPU DMA):  %.2f GB/s", cufile_read_avg / runs);
        if (cufile_read_avg > std_read_avg) {
            printf("  [+%.0f%% faster]\n", (cufile_read_avg/std_read_avg - 1) * 100);
        } else {
            printf("\n");
        }
    }

    if (run_dma || run_all) {
        printf("\n=============================================================\n");
        printf("BENCHMARK: cuFile DMA Only\n");
        printf("=============================================================\n");

        double write_avg = 0, read_avg = 0;

        for (int i = 0; i < runs; i++) {
            printf("\nRun %d/%d:\n", i+1, runs);

            BenchResult rw = bench_cufile_write(d_data, dataSize, filepath);
            if (rw.success) {
                printf("  Write: %.2f GB/s (%.1f ms)\n", rw.bandwidth_gbps, rw.time_ms);
                write_avg += rw.bandwidth_gbps;
            }

            BenchResult rr = bench_cufile_read(d_data, dataSize, filepath);
            if (rr.success) {
                printf("  Read:  %.2f GB/s (%.1f ms)\n", rr.bandwidth_gbps, rr.time_ms);
                read_avg += rr.bandwidth_gbps;
            }
        }

        printf("\nAverage:\n");
        printf("  Write: %.2f GB/s\n", write_avg / runs);
        printf("  Read:  %.2f GB/s\n", read_avg / runs);
    }

    if (run_all) {
        printf("\n=============================================================\n");
        printf("BENCHMARK: GPU Kernel Processing (for comparison)\n");
        printf("=============================================================\n");
        printf("This shows GPU processing speed vs I/O speed\n\n");

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        double write_total = 0, read_total = 0;
        for (int i = 0; i < runs; i++) {
            BenchResult rw = bench_cufile_write(d_data, dataSize, filepath);
            BenchResult rr = bench_cufile_read(d_data, dataSize, filepath);
            if (rw.success) write_total += rw.bandwidth_gbps;
            if (rr.success) read_total += rr.bandwidth_gbps;
        }

        CHECK_CUDA(cudaEventRecord(start));
        xorTransform<<<numBlocks, blockSize>>>(d_data, numElements, 0xDEADBEEF);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float kernel_time;
        CHECK_CUDA(cudaEventElapsedTime(&kernel_time, start, stop));
        double kernel_bw = (dataSize / (1024.0*1024.0*1024.0)) / (kernel_time / 1000.0);

        printf("XOR Transform: %.2f GB/s (%.3f ms)\n", kernel_bw, kernel_time);
        double avg_io = (write_total / runs + read_total / runs) / 2;
        if (avg_io > 0) {
            printf("\nGPU kernel is ~%.0fx faster than I/O\n", kernel_bw / avg_io);
        }

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }

    // Verify data integrity
    int* d_errors;
    CHECK_CUDA(cudaMalloc(&d_errors, sizeof(int)));
    CHECK_CUDA(cudaMemset(d_errors, 0, sizeof(int)));
    verifyData<<<numBlocks, blockSize>>>(d_data, numElements, seed, d_errors);
    CHECK_CUDA(cudaDeviceSynchronize());

    int h_errors = 0;
    CHECK_CUDA(cudaMemcpy(&h_errors, d_errors, sizeof(int), cudaMemcpyDeviceToHost));

    printf("\n=============================================================\n");
    printf("Data Verification: %s\n", h_errors == 0 ? "✓ PASSED" : "✗ FAILED");
    printf("=============================================================\n");

    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_errors));

    return 0;
}
