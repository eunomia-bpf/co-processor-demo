# GPU File I/O Benchmark Makefile
# Builds CUDA application with GPUDirect Storage (cuFile) support

# Compiler and flags
NVCC := nvcc
CUDA_PATH ?= /usr/local/cuda
TARGET := gpu-file-io

# CUDA architecture - adjust for your GPU
# RTX 5090 is Blackwell (sm_120), but using sm_86 for CUDA 12.9 compatibility
# The code will still run efficiently on your RTX 5090
CUDA_ARCH := -arch=sm_86

# Compiler flags
NVCC_FLAGS := -O2 $(CUDA_ARCH)
INCLUDES := -I$(CUDA_PATH)/include
LIBS := -L$(CUDA_PATH)/lib64 -lcufile -lcudart

# Source files
SOURCES := gpu-file-io.cu
OBJECTS := $(SOURCES:.cu=.o)

# Default target
.PHONY: all
all: $(TARGET)

# Build the main executable
# Note: CUDA 12.9 shows harmless link.stub errors but binary compiles successfully
$(TARGET): $(SOURCES)
	@echo "Building $(TARGET)..."
	@$(NVCC) $(NVCC_FLAGS) -o $@ $< $(LIBS)
	@echo "✓ Build complete: $(TARGET)"

# Compile object files (if needed for future modular builds)
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c -o $@ $<

# Run with default settings
.PHONY: run
run: $(TARGET)
	./$(TARGET)

# Run quick test (small size, fewer runs)
.PHONY: test
test: $(TARGET)
	./$(TARGET) -s 128 -r 1

# Run comparison benchmark
.PHONY: compare
compare: $(TARGET)
	./$(TARGET) -m compare

# Run cuFile DMA only
.PHONY: dma
dma: $(TARGET)
	./$(TARGET) -m dma

# Run complete pipeline (read → GPU kernel → write)
.PHONY: pipeline
pipeline: $(TARGET)
	./$(TARGET) -m pipeline

# Run comprehensive benchmark (large dataset)
.PHONY: bench
bench: $(TARGET)
	./$(TARGET) -s 2048 -r 5

# Check system capabilities
.PHONY: check
check:
	@echo "=== System Check ==="
	@echo "GPU Info:"
	@nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
	@echo ""
	@echo "CUDA Version:"
	@$(NVCC) --version | grep "release"
	@echo ""
	@echo "cuFile Library:"
	@ldconfig -p | grep libcufile.so || echo "cuFile not found!"
	@echo ""
	@echo "cuFile Config:"
	@test -f /etc/cufile.json && echo "/etc/cufile.json exists" || echo "Warning: /etc/cufile.json not found"
	@test -f /etc/cufile.json && grep -q "use_pci_p2pdma" /etc/cufile.json && echo "P2P DMA config found" || true

# Clean build artifacts
.PHONY: clean
clean:
	rm -f $(TARGET) $(OBJECTS)
	rm -f /tmp/gpu_bench.dat /tmp/std_write.dat

# Clean all including temporary test files
.PHONY: distclean
distclean: clean
	rm -f *.dat

# Show help
.PHONY: help
help:
	@echo "GPU File I/O Benchmark - Make targets:"
	@echo ""
	@echo "Build targets:"
	@echo "  make all         - Build the executable (default)"
	@echo "  make clean       - Remove build artifacts"
	@echo "  make check       - Check system capabilities"
	@echo ""
	@echo "Run targets:"
	@echo "  make run         - Run with default settings (512MB, 3 runs)"
	@echo "  make test        - Quick test (128MB, 1 run)"
	@echo "  make compare     - Compare standard vs cuFile"
	@echo "  make dma         - Run cuFile DMA benchmarks only"
	@echo "  make pipeline    - Complete pipeline: read → GPU kernel → write"
	@echo "  make bench       - Comprehensive benchmark (2GB, 5 runs)"
	@echo ""
	@echo "Manual run examples:"
	@echo "  ./$(TARGET) -s 1024 -r 5        # 1GB, 5 runs"
	@echo "  ./$(TARGET) -m dma -s 256       # DMA mode, 256MB"
	@echo "  ./$(TARGET) -m pipeline -s 512  # Pipeline mode, 512MB"
	@echo "  ./$(TARGET) -h                  # Show program help"

# Install target (optional)
.PHONY: install
install: $(TARGET)
	@echo "Installing to /usr/local/bin (requires sudo)"
	sudo cp $(TARGET) /usr/local/bin/

# Uninstall target
.PHONY: uninstall
uninstall:
	@echo "Removing from /usr/local/bin (requires sudo)"
	sudo rm -f /usr/local/bin/$(TARGET)

# Debug build
.PHONY: debug
debug: NVCC_FLAGS = -g -G -O0 $(CUDA_ARCH)
debug: clean $(TARGET)
	@echo "Debug build complete"

# Show build info
.PHONY: info
info:
	@echo "Build Configuration:"
	@echo "  Compiler:      $(NVCC)"
	@echo "  CUDA Path:     $(CUDA_PATH)"
	@echo "  Architecture:  $(CUDA_ARCH)"
	@echo "  Flags:         $(NVCC_FLAGS)"
	@echo "  Libraries:     $(LIBS)"
	@echo "  Target:        $(TARGET)"
