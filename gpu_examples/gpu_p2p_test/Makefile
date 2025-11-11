# Makefile for GPU P2P DMA Test Programs
# Uses CUDA 12.9 by default

# CUDA compiler and paths - Using CUDA 12.9
CUDA_PATH := /opt/spack/opt/spack/linux-sapphirerapids/cuda-12.9.0-3eylvnf4bglzu4xuvf4iqvqv5fq7bjpt
NVCC := $(CUDA_PATH)/bin/nvcc
CUDA_INC := $(CUDA_PATH)/include
CUDA_LIB := $(CUDA_PATH)/lib64

# Compiler flags
NVCC_FLAGS := -O2 --std=c++14
NVCC_FLAGS += -Wno-deprecated-gpu-targets
NVCC_FLAGS += -I$(CUDA_INC)

# Linker flags
LDFLAGS := -L$(CUDA_LIB) -lcudart -lcufile

# Source files
SOURCES := check_p2p_gds.cu gpu_file_benchmark.cu
PROGRAMS := check_p2p_gds gpu_file_benchmark

# Python scripts (not compiled, but listed for reference)
PYTHON_SCRIPTS := check_gds_support.py

# Documentation
DOCS := *.md

# Default target
.PHONY: all
all: $(PROGRAMS)

# Build check_p2p_gds
check_p2p_gds: check_p2p_gds.cu
	@echo "Compiling $< with CUDA 12.9..."
	$(NVCC) $(NVCC_FLAGS) $< -o $@ -L$(CUDA_LIB) -lcudart
	@echo "Build complete: $@"

# Build gpu_file_benchmark (merged DMA benchmark)
gpu_file_benchmark: gpu_file_benchmark.cu
	@echo "Compiling $< with CUDA 12.9 + cuFile..."
	$(NVCC) $(NVCC_FLAGS) $< -o $@ $(LDFLAGS)
	@echo "Build complete: $@"

# Run the GPU P2P check program
.PHONY: run
run: check_p2p_gds
	@echo "Running GPU P2P detection..."
	@./check_p2p_gds

# Run the GPU file benchmark
.PHONY: bench
bench: gpu_file_benchmark
	@echo "Running GPU file I/O benchmark..."
	@./gpu_file_benchmark

# Run Python GDS support checker
.PHONY: check
check:
	@echo "Running system GDS support check..."
	@python3 check_gds_support.py

# Run both tests
.PHONY: test
test: run check run-dma

# Clean build artifacts
.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	rm -f $(PROGRAMS) *.o *.log
	@echo "Clean complete"

# Clean everything including test data
.PHONY: clean-all
clean-all: clean
	@echo "Cleaning test data files..."
	rm -f /tmp/gpu_dma_test.dat *.dat
	@echo "Clean all complete"

# Show CUDA version
.PHONY: cuda-version
cuda-version:
	@$(NVCC) --version

# Show system info
.PHONY: info
info:
	@echo "=== Build Configuration ==="
	@echo "NVCC:      $(NVCC)"
	@echo "CUDA_PATH: $(CUDA_PATH)"
	@echo "Flags:     $(NVCC_FLAGS)"
	@echo ""
	@echo "=== CUDA Version ==="
	@$(NVCC) --version | grep "release"
	@echo ""
	@echo "=== GPU Info ==="
	@nvidia-smi -L
	@echo ""
	@echo "=== Programs to Build ==="
	@echo "$(PROGRAMS)"

# Install nvidia-fs (requires sudo)
.PHONY: install-gds
install-gds:
	@echo "Installing NVIDIA GPUDirect Storage..."
	@echo "Note: This requires sudo privileges"
	sudo apt-get update
	sudo apt-get install -y nvidia-gds-12-9
	@echo "Attempting to load nvidia-fs module..."
	-sudo modprobe nvidia-fs
	@echo "Checking nvidia-fs status..."
	@lsmod | grep nvidia_fs || echo "nvidia-fs not loaded (check dmesg for errors)"

# Help target
.PHONY: help
help:
	@echo "GPU P2P DMA Test Programs - Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  all              - Build all programs (default)"
	@echo "  check_p2p_gds    - Build GPU P2P detection program"
	@echo "  gpu_to_file_dma  - Build GPU-to-file DMA example"
	@echo "  run              - Build and run GPU P2P detection"
	@echo "  run-dma          - Build and run GPU-to-file DMA example"
	@echo "  check            - Run Python system compatibility checker"
	@echo "  test             - Run all tests"
	@echo "  clean            - Remove build artifacts and test files"
	@echo "  cuda-version     - Show CUDA compiler version"
	@echo "  info             - Show build configuration and system info"
	@echo "  install-gds      - Install nvidia-gds package (requires sudo)"
	@echo "  help             - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make             # Build all programs"
	@echo "  make run-dma     # Run GPU-to-file DMA example"
	@echo "  make test        # Run all tests"
	@echo "  make clean       # Clean build files"
	@echo ""
	@echo "GPU-to-file DMA options:"
	@echo "  ./gpu_to_file_dma -s 1024        # Test with 1GB file"
	@echo "  ./gpu_to_file_dma -f /mnt/test   # Custom file path"
	@echo "  ./gpu_to_file_dma -h             # Show help"
