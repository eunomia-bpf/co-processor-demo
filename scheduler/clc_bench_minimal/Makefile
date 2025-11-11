# Makefile for Cluster Launch Control Benchmark
# Tested on RTX 5090 (CC 12.0) with CUDA 12.9

# CUDA paths
CUDA_12_9 = /usr/local/cuda-12.9

# Compiler settings
NVCC = $(CUDA_12_9)/bin/nvcc

# Architecture
ARCH = sm_120

# Compiler flags
NVCC_FLAGS = -arch=$(ARCH) --no-device-link -O3

# ============================================
# BUILD TARGETS
# ============================================

.PHONY: all clean test help

# Default target
all: minimal_clc_12.9_api clc_benchmark clc_benchmark_workloads

# Minimal CLC example - single approach
minimal_clc_12.9_api: minimal_clc_12.9.cu
	$(NVCC) $(NVCC_FLAGS) -o minimal_clc_12.9_api minimal_clc_12.9.cu
	@echo "✅ Built: minimal_clc_12.9_api"

# Comprehensive benchmark - all three approaches
clc_benchmark: clc_benchmark.cu
	$(NVCC) $(NVCC_FLAGS) -o clc_benchmark clc_benchmark.cu
	@echo "✅ Built: clc_benchmark"

# Workload analysis benchmark - tests different scenarios
clc_benchmark_workloads: clc_benchmark_workloads.cu
	$(NVCC) $(NVCC_FLAGS) -o clc_benchmark_workloads clc_benchmark_workloads.cu
	@echo "✅ Built: clc_benchmark_workloads"

# ============================================
# TEST TARGETS
# ============================================

test: minimal_clc_12.9_api clc_benchmark
	@echo "=== Testing Minimal CLC Example ==="
	timeout 10 ./minimal_clc_12.9_api || echo "TIMEOUT or ERROR"
	@echo ""
	@echo "=== Testing Comprehensive Benchmark ==="
	timeout 30 ./clc_benchmark 1048576 256 3 10 || echo "TIMEOUT or ERROR"
	@echo ""

# Run minimal example
run-minimal: minimal_clc_12.9_api
	@echo "=== Running Minimal CLC Example ==="
	./minimal_clc_12.9_api

# Run benchmark with default parameters (1M elements)
run-benchmark: clc_benchmark
	@echo "=== Running CLC Benchmark (Default: 1M elements) ==="
	./clc_benchmark

# Run benchmark with small dataset (quick test)
run-small: clc_benchmark
	@echo "=== Running CLC Benchmark (Small: 64K elements) ==="
	./clc_benchmark 65536 256 2 5

# Run benchmark with large dataset
run-large: clc_benchmark
	@echo "=== Running CLC Benchmark (Large: 16M elements) ==="
	./clc_benchmark 16777216 256 3 10

# Run workload analysis benchmark
run-workloads: clc_benchmark_workloads
	@echo "=== Running CLC Workload Analysis ==="
	./clc_benchmark_workloads

# ============================================
# INFORMATION
# ============================================

info:
	@echo "=== Environment Information ==="
	@echo "GPU:"
	@nvidia-smi --query-gpu=name,compute_cap --format=csv
	@echo ""
	@echo "Driver:"
	@cat /proc/driver/nvidia/version | head -1
	@echo ""
	@echo "CUDA Version:"
	@$(NVCC) --version | grep "release"

# ============================================
# CLEANUP
# ============================================

clean:
	rm -f minimal_clc_12.9_api clc_benchmark clc_benchmark_workloads
	rm -f /tmp/clc*.ptx /tmp/clc*.cubin
	@echo "Cleaned all binaries and temp files"

# ============================================
# HELP
# ============================================

help:
	@echo "Cluster Launch Control Benchmark - Makefile"
	@echo ""
	@echo "BUILD TARGETS:"
	@echo "  make                         - Build all examples (default)"
	@echo "  make all                     - Build all benchmarks"
	@echo "  make minimal_clc_12.9_api    - Build minimal CLC example"
	@echo "  make clc_benchmark           - Build comprehensive benchmark"
	@echo "  make clc_benchmark_workloads - Build workload analysis benchmark"
	@echo ""
	@echo "TEST TARGETS:"
	@echo "  make test               - Build and run all tests"
	@echo "  make run-minimal        - Run minimal CLC example"
	@echo "  make run-benchmark      - Run benchmark (1M elements, default)"
	@echo "  make run-small          - Run benchmark (64K elements, quick)"
	@echo "  make run-large          - Run benchmark (16M elements)"
	@echo "  make run-workloads      - Run workload analysis (tests 8 scenarios)"
	@echo ""
	@echo "BENCHMARK USAGE:"
	@echo "  ./clc_benchmark [n] [threads] [warmup] [runs]"
	@echo "    n       - Number of elements (default: 1048576)"
	@echo "    threads - Threads per block (default: 256)"
	@echo "    warmup  - Warmup runs (default: 3)"
	@echo "    runs    - Benchmark runs (default: 10)"
	@echo ""
	@echo "EXAMPLES:"
	@echo "  ./clc_benchmark                    # Use all defaults"
	@echo "  ./clc_benchmark 1000000            # 1M elements"
	@echo "  ./clc_benchmark 1000000 512        # 1M elements, 512 threads/block"
	@echo "  ./clc_benchmark 1000000 512 5 20   # 1M elements, 512 threads, 5 warmup, 20 runs"
	@echo ""
	@echo "INFORMATION:"
	@echo "  make info               - Show GPU, driver, and CUDA info"
	@echo ""
	@echo "CLEANUP:"
	@echo "  make clean              - Remove all binaries and temp files"
	@echo ""
	@echo "REQUIREMENTS:"
	@echo "  - CUDA 12.9+"
	@echo "  - Compute Capability 10.0+ (Blackwell or newer)"
	@echo "  - Driver 575+"

.DEFAULT_GOAL := all
