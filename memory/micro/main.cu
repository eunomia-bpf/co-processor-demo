#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>
#include "kernels/synthetic.cuh"

struct Config {
    std::string kernel = "seq_stream";
    std::string mode = "uvm";
    float size_factor = 1.0;
    int iterations = 10;
    std::string output = "results.csv";
};

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --kernel=<name>        Kernel: seq_stream, rand_stream, pointer_chase (default: seq_stream)\n"
              << "  --mode=<mode>          Mode: device, uvm, uvm_prefetch (default: uvm)\n"
              << "  --size_factor=<float>  Size factor relative to GPU memory (default: 1.0)\n"
              << "  --iterations=<int>     Number of iterations (default: 10)\n"
              << "  --output=<path>        Output CSV file (default: results.csv)\n"
              << "  --help                 Show this help\n";
}

Config parse_args(int argc, char** argv) {
    Config config;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            exit(0);
        }

        if (arg.substr(0, 9) == "--kernel=") {
            config.kernel = arg.substr(9);
        } else if (arg.substr(0, 7) == "--mode=") {
            config.mode = arg.substr(7);
        } else if (arg.substr(0, 14) == "--size_factor=") {
            config.size_factor = std::stof(arg.substr(14));
        } else if (arg.substr(0, 13) == "--iterations=") {
            config.iterations = std::stoi(arg.substr(13));
        } else if (arg.substr(0, 9) == "--output=") {
            config.output = arg.substr(9);
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_usage(argv[0]);
            exit(1);
        }
    }

    return config;
}

size_t get_gpu_memory() {
    size_t free_bytes, total_bytes;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    return total_bytes;
}

float compute_median(std::vector<float>& times) {
    if (times.empty()) return 0.0f;
    std::sort(times.begin(), times.end());
    size_t n = times.size();
    if (n % 2 == 0) {
        return (times[n/2-1] + times[n/2]) / 2.0f;
    } else {
        return times[n/2];
    }
}

void write_results(const Config& config, const KernelResult& result,
                   size_t total_working_set) {
    FILE* f = fopen(config.output.c_str(), "w");
    if (!f) {
        std::cerr << "Failed to open output file: " << config.output << std::endl;
        return;
    }

    // Calculate bandwidth
    double bw_GBps = (result.bytes_accessed / (result.median_ms / 1000.0)) / 1e9;

    // CSV header
    fprintf(f, "kernel,mode,size_factor,working_set_bytes,bytes_accessed,iterations,"
               "median_ms,min_ms,max_ms,bw_GBps\n");

    // Write data
    fprintf(f, "%s,%s,%.2f,%zu,%zu,%d,%.3f,%.3f,%.3f,%.3f\n",
            config.kernel.c_str(),
            config.mode.c_str(),
            config.size_factor,
            total_working_set,
            result.bytes_accessed,
            config.iterations,
            result.median_ms,
            result.min_ms,
            result.max_ms,
            bw_GBps);

    fclose(f);

    std::cout << "\nResults:\n"
              << "  Kernel: " << config.kernel << "\n"
              << "  Mode: " << config.mode << "\n"
              << "  Working Set: " << total_working_set / (1024*1024) << " MB\n"
              << "  Bytes Accessed: " << result.bytes_accessed / (1024*1024) << " MB\n"
              << "  Median time: " << result.median_ms << " ms\n"
              << "  Min time: " << result.min_ms << " ms\n"
              << "  Max time: " << result.max_ms << " ms\n"
              << "  Bandwidth: " << bw_GBps << " GB/s\n"
              << "  Results written to: " << config.output << "\n";
}

int main(int argc, char** argv) {
    Config config = parse_args(argc, argv);

    // Initialize CUDA with error checking
    try {
        CUDA_CHECK(cudaSetDevice(0));
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize CUDA: " << e.what() << std::endl;
        return 1;
    }

    // Get GPU memory and compute size
    size_t gpu_mem = get_gpu_memory();
    size_t total_working_set = static_cast<size_t>(gpu_mem * config.size_factor);

    std::cout << "UVM Microbenchmark - Tier 0 Synthetic Kernels\n"
              << "==============================================\n"
              << "GPU Memory: " << gpu_mem / (1024*1024) << " MB\n"
              << "Size Factor: " << config.size_factor << "\n"
              << "Total Working Set: " << total_working_set / (1024*1024) << " MB\n"
              << "Kernel: " << config.kernel << "\n"
              << "Mode: " << config.mode << "\n"
              << "Iterations: " << config.iterations << "\n";

    // Warn for device mode oversubscription
    if (config.mode == "device" && config.size_factor > 0.8) {
        std::cout << "\n*** WARNING: size_factor > 0.8 in device mode may cause OOM! ***\n";
        std::cout << "*** Consider using size_factor <= 0.8 for device mode        ***\n";
    }

    std::cout << "\n";

    // Run the selected kernel
    std::vector<float> runtimes;
    KernelResult result;

    try {
        if (config.kernel == "seq_stream") {
            run_seq_stream(total_working_set, config.mode, config.iterations, runtimes, result);
        } else if (config.kernel == "rand_stream") {
            run_rand_stream(total_working_set, config.mode, config.iterations, runtimes, result);
        } else if (config.kernel == "pointer_chase") {
            run_pointer_chase(total_working_set, config.mode, config.iterations, runtimes, result);
        } else {
            std::cerr << "Unknown kernel: " << config.kernel << std::endl;
            return 1;
        }

        // Write results
        write_results(config, result, total_working_set);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
