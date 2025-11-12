# Makefile for Multi-Stream GPU Scheduler Benchmark

# CUDA compiler
NVCC := nvcc

# Target executable
TARGET := multi_stream_bench

# Source files
SOURCES := multi_stream_bench.cu

# CUDA architecture (adjust for your GPU)
# Volta (V100): 7.0, Turing (RTX 20xx): 7.5, Ampere (A100/RTX 30xx): 8.0, Ada (RTX 40xx): 8.9
CUDA_ARCH := 75

# Compiler flags
NVCC_FLAGS := -arch=sm_$(CUDA_ARCH) \
              -O3 \
              -std=c++14 \
              -Xcompiler -Wall,-Wextra \
              --use_fast_math \
              -Xlinker -z,noexecstack

# Debug flags (use with 'make debug')
DEBUG_FLAGS := -g -G -O0 -DDEBUG

# Default target
all: $(TARGET)

# Build release version
$(TARGET): $(SOURCES)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^
	@echo "Build complete: $(TARGET)"
	@echo "Run with: ./$(TARGET)"

# Build debug version
debug: NVCC_FLAGS := -arch=sm_$(CUDA_ARCH) $(DEBUG_FLAGS)
debug: clean $(TARGET)
	@echo "Debug build complete"

# Build for multiple architectures
multi-arch: NVCC_FLAGS := -gencode arch=compute_70,code=sm_70 \
                          -gencode arch=compute_75,code=sm_75 \
                          -gencode arch=compute_80,code=sm_80 \
                          -gencode arch=compute_86,code=sm_86 \
                          -gencode arch=compute_89,code=sm_89 \
                          -O3 -std=c++14 -Xlinker -z,noexecstack --no-device-link 
multi-arch: $(TARGET)
	@echo "Multi-architecture build complete"

# Run with default parameters
run: $(TARGET)
	./$(TARGET)

# Run with various configurations for testing
test: $(TARGET)
	@echo "Running benchmark with different configurations..."
	@echo "\n=== Test 1: Default (4 streams, mixed workload) ==="
	./$(TARGET)
	@echo "\n=== Test 2: High concurrency (8 streams) ==="
	./$(TARGET) --streams 8
	@echo "\n=== Test 3: Compute-bound workload ==="
	./$(TARGET) --type compute
	@echo "\n=== Test 4: Memory-bound workload ==="
	./$(TARGET) --type memory
	@echo "\n=== Test 5: Many small kernels ==="
	./$(TARGET) --streams 16 --kernels 5 --size 262144

# Clean build artifacts
clean:
	rm -f $(TARGET) *.o

# Show GPU information
gpu-info:
	nvidia-smi
	@echo "\nCUDA Version:"
	$(NVCC) --version

# Python experiment targets
PYTHON := python3
EXPERIMENT_DRIVER := experiment_driver.py
ANALYZE_SCRIPT := analyze_results.py
RESULTS_DIR := results
TRIALS := 10

# Check Python dependencies
check-deps:
	@echo "Checking Python dependencies..."
	@$(PYTHON) -c "import pandas, numpy, matplotlib, seaborn" 2>/dev/null || \
		(echo "Missing dependencies. Install with: pip3 install -r requirements.txt" && exit 1)
	@echo "✓ All Python dependencies found"

# Run quick pilot experiments (3 trials)
experiment-pilot: $(TARGET) check-deps
	@echo "Running pilot experiments (3 trials)..."
	$(PYTHON) $(EXPERIMENT_DRIVER) --experiments all --trials 3
	@echo "✓ Pilot experiments complete"

# Run full publication-quality experiments (10+ trials)
experiment-full: $(TARGET) check-deps
	@echo "Running full experiments ($(TRIALS) trials)..."
	@echo "This may take 30-60 minutes..."
	$(PYTHON) $(EXPERIMENT_DRIVER) --experiments all --trials $(TRIALS)
	@echo "✓ Full experiments complete"

# Run individual research questions
experiment-rq1: $(TARGET) check-deps
	@echo "Running RQ1: Stream Scalability ($(TRIALS) trials)..."
	$(PYTHON) $(EXPERIMENT_DRIVER) --experiments RQ1 --trials $(TRIALS)

experiment-rq2: $(TARGET) check-deps
	@echo "Running RQ2: Workload Characterization ($(TRIALS) trials)..."
	$(PYTHON) $(EXPERIMENT_DRIVER) --experiments RQ2 --trials $(TRIALS)

experiment-rq3: $(TARGET) check-deps
	@echo "Running RQ3: Priority Effectiveness ($(TRIALS) trials)..."
	$(PYTHON) $(EXPERIMENT_DRIVER) --experiments RQ3 --trials $(TRIALS)

experiment-rq4: $(TARGET) check-deps
	@echo "Running RQ4: Memory Pressure ($(TRIALS) trials)..."
	$(PYTHON) $(EXPERIMENT_DRIVER) --experiments RQ4 --trials $(TRIALS)

experiment-rq5: $(TARGET) check-deps
	@echo "Running RQ5: Multi-Process Interference ($(TRIALS) trials)..."
	$(PYTHON) $(EXPERIMENT_DRIVER) --experiments RQ5 --trials $(TRIALS)

experiment-rq6: $(TARGET) check-deps
	@echo "Running RQ6: Load Imbalance and Fairness ($(TRIALS) trials)..."
	$(PYTHON) $(EXPERIMENT_DRIVER) --experiments RQ6 --trials $(TRIALS)

experiment-rq7: $(TARGET) check-deps
	@echo "Running RQ7: Tail Latency ($(TRIALS) trials)..."
	$(PYTHON) $(EXPERIMENT_DRIVER) --experiments RQ7 --trials $(TRIALS)

experiment-rq9: $(TARGET) check-deps
	@echo "Running RQ9: Priority-Based Tail Latency ($(TRIALS) trials)..."
	$(PYTHON) $(EXPERIMENT_DRIVER) --experiments RQ9 --trials $(TRIALS)

experiment-rq10: $(TARGET) check-deps
	@echo "Running RQ10: Preemption Latency ($(TRIALS) trials)..."
	$(PYTHON) $(EXPERIMENT_DRIVER) --experiments RQ10 --trials $(TRIALS)

experiment-rq11: $(TARGET) check-deps
	@echo "Running RQ11: Bandwidth Partitioning ($(TRIALS) trials)..."
	$(PYTHON) $(EXPERIMENT_DRIVER) --experiments RQ11 --trials $(TRIALS)

# Analyze all available results
analyze: check-deps
	@echo "Analyzing experimental results..."
	$(PYTHON) $(ANALYZE_SCRIPT) --experiments all
	@echo "✓ Analysis complete"
	@echo "Results in: $(RESULTS_DIR)/"
	@echo "Figures in: $(RESULTS_DIR)/figures/"
	@echo "Report: $(RESULTS_DIR)/ANALYSIS_REPORT.md"

# Analyze specific research question
analyze-rq1: check-deps
	$(PYTHON) $(ANALYZE_SCRIPT) --experiments RQ1

analyze-rq2: check-deps
	$(PYTHON) $(ANALYZE_SCRIPT) --experiments RQ2

analyze-rq3: check-deps
	$(PYTHON) $(ANALYZE_SCRIPT) --experiments RQ3

analyze-rq4: check-deps
	$(PYTHON) $(ANALYZE_SCRIPT) --experiments RQ4

analyze-rq5: check-deps
	$(PYTHON) $(ANALYZE_SCRIPT) --experiments RQ5

analyze-rq6: check-deps
	$(PYTHON) $(ANALYZE_SCRIPT) --experiments RQ6

analyze-rq7: check-deps
	$(PYTHON) $(ANALYZE_SCRIPT) --experiments RQ7

analyze-rq9: check-deps
	$(PYTHON) $(ANALYZE_SCRIPT) --experiments RQ9

analyze-rq10: check-deps
	$(PYTHON) $(ANALYZE_SCRIPT) --experiments RQ10

analyze-rq11: check-deps
	$(PYTHON) $(ANALYZE_SCRIPT) --experiments RQ11

# Complete workflow: build, run experiments, analyze
workflow-quick: clean $(TARGET)
	@echo "=== Quick Workflow: Build → Experiment → Analyze ==="
	@$(MAKE) experiment-pilot
	@$(MAKE) analyze
	@echo "=== Quick workflow complete ==="
	@cat $(RESULTS_DIR)/ANALYSIS_REPORT.md

workflow-full: clean $(TARGET)
	@echo "=== Full Workflow: Build → Experiment → Analyze ==="
	@echo "Starting full experimental workflow with $(TRIALS) trials..."
	@$(MAKE) experiment-full
	@$(MAKE) analyze
	@echo "=== Full workflow complete ==="
	@cat $(RESULTS_DIR)/ANALYSIS_REPORT.md

# Archive results with timestamp
archive-results:
	@TIMESTAMP=$$(date +%Y%m%d_%H%M%S); \
	ARCHIVE="results_$$TIMESTAMP.tar.gz"; \
	echo "Archiving results to $$ARCHIVE..."; \
	tar -czf $$ARCHIVE $(RESULTS_DIR)/; \
	echo "✓ Results archived: $$ARCHIVE"

# View results
view-results:
	@echo "=== Experimental Results ==="
	@ls -lh $(RESULTS_DIR)/*.csv 2>/dev/null || echo "No CSV files found"
	@echo ""
	@echo "=== Figures ==="
	@ls -lh $(RESULTS_DIR)/figures/*.png 2>/dev/null || echo "No figures found"
	@echo ""
	@echo "=== Analysis Report ==="
	@cat $(RESULTS_DIR)/ANALYSIS_REPORT.md 2>/dev/null || echo "No analysis report found"

# Clean all experiment results
clean-results:
	@echo "Cleaning experiment results..."
	rm -rf $(RESULTS_DIR)
	@echo "✓ Results cleaned"

# Clean everything (build + results)
clean-all: clean clean-results
	@echo "✓ All cleaned"

# Install Python dependencies
install-deps:
	@echo "Installing Python dependencies..."
	pip3 install -r requirements.txt
	@echo "✓ Dependencies installed"

# Help target
help:
	@echo "Multi-Stream GPU Scheduler Benchmark - Makefile"
	@echo ""
	@echo "=== Build Targets ==="
	@echo "  make              - Build release version (default)"
	@echo "  make debug        - Build debug version with symbols"
	@echo "  make multi-arch   - Build for multiple GPU architectures"
	@echo "  make clean        - Remove build artifacts"
	@echo ""
	@echo "=== Quick Test Targets ==="
	@echo "  make run          - Run single benchmark with default parameters"
	@echo "  make test         - Run benchmark test suite"
	@echo "  make gpu-info     - Display GPU and CUDA information"
	@echo ""
	@echo "=== Experiment Targets ==="
	@echo "  make experiment-pilot    - Quick pilot (3 trials, ~5 min)"
	@echo "  make experiment-full     - Full experiments ($(TRIALS) trials, ~30-60 min)"
	@echo "  make experiment-rqN      - Run specific RQ (N=1,2,3,4,5,6,7,9,10,11)"
	@echo ""
	@echo "=== Analysis Targets ==="
	@echo "  make analyze             - Analyze all available results"
	@echo "  make analyze-rqN         - Analyze specific RQ results"
	@echo "  make view-results        - Display results summary"
	@echo ""
	@echo "=== Workflow Targets (Recommended) ==="
	@echo "  make workflow-quick      - Build → Pilot → Analyze (~10 min)"
	@echo "  make workflow-full       - Build → Full Exp → Analyze (~1 hour)"
	@echo ""
	@echo "=== Utility Targets ==="
	@echo "  make check-deps          - Check Python dependencies"
	@echo "  make install-deps        - Install Python dependencies"
	@echo "  make archive-results     - Archive results with timestamp"
	@echo "  make clean-results       - Remove experiment results"
	@echo "  make clean-all           - Remove build artifacts and results"
	@echo "  make help                - Show this help message"
	@echo ""
	@echo "=== Configuration ==="
	@echo "  CUDA_ARCH=$(CUDA_ARCH)  - GPU architecture (change for your GPU)"
	@echo "  TRIALS=$(TRIALS)        - Number of trials per config"
	@echo ""
	@echo "=== Examples ==="
	@echo "  make workflow-quick                  # Quick test workflow"
	@echo "  make TRIALS=20 experiment-full       # Full experiments with 20 trials"
	@echo "  make experiment-rq1 analyze-rq1      # Run and analyze RQ1 only"
	@echo "  make CUDA_ARCH=80 workflow-full      # For Ampere GPU (A100/RTX 30xx)"
	@echo ""
	@echo "=== Quick Start ==="
	@echo "  1. make workflow-quick               # Fast test (recommended first time)"
	@echo "  2. make view-results                 # View results"
	@echo "  3. make TRIALS=20 workflow-full      # Full experiments for publication"

.PHONY: all debug multi-arch run test clean gpu-info help \
        check-deps install-deps \
        experiment-pilot experiment-full \
        experiment-rq1 experiment-rq2 experiment-rq3 experiment-rq4 experiment-rq5 experiment-rq6 experiment-rq7 \
        experiment-rq9 experiment-rq10 experiment-rq11 \
        analyze analyze-rq1 analyze-rq2 analyze-rq3 analyze-rq4 analyze-rq5 analyze-rq6 analyze-rq7 \
        analyze-rq9 analyze-rq10 analyze-rq11 \
        workflow-quick workflow-full \
        archive-results view-results clean-results clean-all
