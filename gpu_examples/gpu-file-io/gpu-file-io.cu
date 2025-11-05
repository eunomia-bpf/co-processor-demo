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
    printf("  -m MODE    Mode: all, dma, compare, pipeline (default: all)\n");
    printf("  -h         Show this help\n");
    printf("\nDefault behavior (no -m flag):\n");
    printf("  Runs ALL benchmarks:\n");
    printf("    1. Standard Copy vs cuFile DMA comparison\n");
    printf("    2. cuFile DMA only benchmarks\n");
    printf("    3. Pipeline comparison (Traditional vs Zero-Copy)\n");
    printf("\nModes (use -m to run specific benchmark):\n");
    printf("  all      - Run all benchmarks (same as default)\n");
    printf("  dma      - Only cuFile DMA benchmarks\n");
    printf("  compare  - Only standard vs cuFile comparison\n");
    printf("  pipeline - Only pipeline comparison\n");
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
    bool run_pipeline = (strcmp(mode, "pipeline") == 0);

    // If mode is "all", run everything including pipeline
    if (run_all) {
        run_compare = true;
        run_dma = true;
        run_pipeline = true;
    }

    // Run benchmarks
    if (run_compare) {
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

    if (run_dma) {
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

    // Note: Removed standalone GPU kernel benchmark from "all" mode
    // It's now included in the pipeline comparison which is more useful
    if (false) {  // Keep code but disable - can be re-enabled if needed
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

    if (run_pipeline) {
        printf("\n=============================================================\n");
        printf("BENCHMARK: Pipeline Comparison\n");
        printf("=============================================================\n");
        printf("Comparing two complete data processing pipelines:\n");
        printf("  A) Traditional: File→RAM→GPU → Process → GPU→RAM→File\n");
        printf("  B) Zero-Copy:   File→GPU(DMA) → Process → GPU→File(DMA)\n\n");

        const char* input_file = "/tmp/gpu_pipeline_input.dat";
        const char* output_file_zerocopy = "/tmp/gpu_pipeline_zerocopy.dat";
        const char* output_file_traditional = "/tmp/gpu_pipeline_traditional.dat";

        // First, write initial data to input file
        printf("Preparing input file...\n");
        BenchResult prep = bench_cufile_write(d_data, dataSize, input_file);
        if (!prep.success) {
            printf("Failed to prepare input file\n");
        } else {
            printf("Input file ready: %s\n\n", input_file);

            // ==== Traditional Pipeline ====
            printf("-------------------------------------------------------------\n");
            printf("A) TRADITIONAL PIPELINE (CPU Memory Copy)\n");
            printf("-------------------------------------------------------------\n");

            double trad_total = 0, trad_read = 0, trad_write = 0, trad_kernel = 0;

            for (int i = 0; i < runs; i++) {
                printf("Run %d/%d:\n", i+1, runs);

                double pipeline_start = getTime();

                // Step 1: Standard read (File → RAM → GPU)
                double read_start = getTime();
                BenchResult read_result = bench_standard_read(d_data, dataSize, input_file);
                double read_time = getTime() - read_start;

                // Step 2: GPU kernel processes data
                double kernel_start = getTime();
                xorTransform<<<numBlocks, blockSize>>>(d_data, numElements, 0xABCD1234);
                CHECK_CUDA(cudaDeviceSynchronize());
                double kernel_time = getTime() - kernel_start;

                // Step 3: Standard write (GPU → RAM → File)
                double write_start = getTime();
                BenchResult write_result = bench_standard_write(d_data, dataSize, output_file_traditional);
                double write_time = getTime() - write_start;

                double pipeline_total = getTime() - pipeline_start;
                double bandwidth = (dataSize / (1024.0*1024.0*1024.0)) / pipeline_total;

                printf("  Read:       %.2f GB/s (%.1f ms) [File→RAM→GPU]\n",
                       (dataSize / (1024.0*1024.0*1024.0)) / read_time, read_time * 1000);
                printf("  GPU Kernel: %.2f GB/s (%.3f ms)\n",
                       (dataSize / (1024.0*1024.0*1024.0)) / kernel_time, kernel_time * 1000);
                printf("  Write:      %.2f GB/s (%.1f ms) [GPU→RAM→File]\n",
                       (dataSize / (1024.0*1024.0*1024.0)) / write_time, write_time * 1000);
                printf("  Pipeline:   %.2f GB/s (%.1f ms)\n\n", bandwidth, pipeline_total * 1000);

                trad_total += pipeline_total;
                trad_read += read_time;
                trad_write += write_time;
                trad_kernel += kernel_time;
            }

            double trad_avg_bw = (dataSize / (1024.0*1024.0*1024.0)) / (trad_total / runs);
            printf("Traditional Pipeline Average: %.2f GB/s (%.1f ms)\n\n",
                   trad_avg_bw, (trad_total / runs) * 1000);

            // ==== Zero-Copy Pipeline ====
            printf("-------------------------------------------------------------\n");
            printf("B) ZERO-COPY PIPELINE (GPU Direct Storage)\n");
            printf("-------------------------------------------------------------\n");

            cudaEvent_t start, stop;
            CHECK_CUDA(cudaEventCreate(&start));
            CHECK_CUDA(cudaEventCreate(&stop));

            double total_time = 0;
            double read_total = 0, write_total = 0, kernel_total = 0;

            // Open cuFile handles once for all runs
            CUfileError_t status = cuFileDriverOpen();
            if (status.err != CU_FILE_SUCCESS) {
                printf("Failed to open cuFile driver\n");
            } else {
                for (int i = 0; i < runs; i++) {
                    printf("Pipeline Run %d/%d:\n", i+1, runs);

                    double pipeline_start = getTime();

                    // Step 1: cuFile DMA Read from file to GPU
                    double read_start = getTime();
                    BenchResult read_result = bench_cufile_read(d_data, dataSize, input_file);
                    double read_time = getTime() - read_start;
                    if (!read_result.success) {
                        printf("  Read failed\n");
                        continue;
                    }

                    // Step 2: GPU kernel processes data (XOR transform)
                    double kernel_start = getTime();
                    xorTransform<<<numBlocks, blockSize>>>(d_data, numElements, 0xABCD1234);
                    CHECK_CUDA(cudaDeviceSynchronize());
                    double kernel_time = getTime() - kernel_start;

                    // Step 3: cuFile DMA Write from GPU to file
                    double write_start = getTime();
                    BenchResult write_result = bench_cufile_write(d_data, dataSize, output_file_zerocopy);
                    double write_time = getTime() - write_start;
                    if (!write_result.success) {
                        printf("  Write failed\n");
                        continue;
                    }

                    double pipeline_total = getTime() - pipeline_start;
                    double bandwidth = (dataSize / (1024.0*1024.0*1024.0)) / pipeline_total;

                    printf("  Read:       %.2f GB/s (%.1f ms) [File→GPU DMA]\n",
                           (dataSize / (1024.0*1024.0*1024.0)) / read_time, read_time * 1000);
                    printf("  GPU Kernel: %.2f GB/s (%.3f ms)\n",
                           (dataSize / (1024.0*1024.0*1024.0)) / kernel_time, kernel_time * 1000);
                    printf("  Write:      %.2f GB/s (%.1f ms) [GPU→File DMA]\n",
                           (dataSize / (1024.0*1024.0*1024.0)) / write_time, write_time * 1000);
                    printf("  Pipeline:   %.2f GB/s (%.1f ms)\n\n", bandwidth, pipeline_total * 1000);

                    total_time += pipeline_total;
                    read_total += read_time;
                    write_total += write_time;
                    kernel_total += kernel_time;
                }
                cuFileDriverClose();
            }

            double zerocopy_avg_bw = (dataSize / (1024.0*1024.0*1024.0)) / (total_time / runs);
            printf("Zero-Copy Pipeline Average: %.2f GB/s (%.1f ms)\n\n",
                   zerocopy_avg_bw, (total_time / runs) * 1000.0);

            // ==== Comparison Summary ====
            printf("=============================================================\n");
            printf("PIPELINE COMPARISON SUMMARY\n");
            printf("=============================================================\n");
            printf("Traditional (CPU Copy):  %.2f GB/s (%.1f ms)\n",
                   trad_avg_bw, (trad_total / runs) * 1000);
            printf("Zero-Copy (GPU Direct):  %.2f GB/s (%.1f ms)\n\n",
                   zerocopy_avg_bw, (total_time / runs) * 1000);

            if (zerocopy_avg_bw > trad_avg_bw) {
                double speedup = (zerocopy_avg_bw / trad_avg_bw - 1) * 100;
                printf("⚡ Zero-Copy is %.0f%% FASTER\n\n", speedup);
            } else {
                double slowdown = (trad_avg_bw / zerocopy_avg_bw - 1) * 100;
                printf("Traditional is %.0f%% faster\n\n", slowdown);
            }

            printf("Breakdown:\n");
            printf("                      Traditional    Zero-Copy      Improvement\n");
            printf("  Read:               %.2f GB/s      %.2f GB/s     %+.0f%%\n",
                   (dataSize / (1024.0*1024.0*1024.0)) / (trad_read / runs),
                   (dataSize / (1024.0*1024.0*1024.0)) / (read_total / runs),
                   ((read_total / trad_read) - 1) * -100);
            printf("  GPU Kernel:         %.2f GB/s      %.2f GB/s     (same)\n",
                   (dataSize / (1024.0*1024.0*1024.0)) / (trad_kernel / runs),
                   (dataSize / (1024.0*1024.0*1024.0)) / (kernel_total / runs));
            printf("  Write:              %.2f GB/s      %.2f GB/s     %+.0f%%\n",
                   (dataSize / (1024.0*1024.0*1024.0)) / (trad_write / runs),
                   (dataSize / (1024.0*1024.0*1024.0)) / (write_total / runs),
                   ((write_total / trad_write) - 1) * -100);
            printf("\nKey Benefit: Zero-copy eliminates CPU memory bottleneck!\n");

            CHECK_CUDA(cudaEventDestroy(start));
            CHECK_CUDA(cudaEventDestroy(stop));

            // Clean up pipeline files
            unlink(input_file);
            unlink(output_file_zerocopy);
            unlink(output_file_traditional);
        }
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
