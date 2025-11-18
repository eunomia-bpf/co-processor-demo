# Makefile for GEMM kernel

# CUDA compiler
NVCC = nvcc

# Target executable
TARGET = gemm_kernel

# Source files
MAIN_SRC = gemm_kernel.cu

# CUDA architecture
CUDA_ARCH = sm_120

# Compiler flags
NVCCFLAGS = -arch=$(CUDA_ARCH) -O3 -std=c++11

# Default target: build
all: $(TARGET)

# Build main app
$(TARGET): $(MAIN_SRC)
	$(NVCC) $(NVCCFLAGS) $(MAIN_SRC) -o $(TARGET)

# Run the test
run: all
	./$(TARGET)

# Clean build artifacts
clean:
	rm -f $(TARGET) *.o *.cubin

# Phony targets
.PHONY: all clean run
