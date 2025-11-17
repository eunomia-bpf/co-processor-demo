# Makefile for GEMM CUDA test application

# CUDA compiler
NVCC = nvcc

# Target executable
TARGET = gemm_test

# Source files
SRCS = gemm_test.cu

# CUDA architecture (adjust based on your GPU)
# Common values: 60 (Pascal), 70 (Volta), 75 (Turing), 80 (Ampere), 86 (Ampere), 89 (Ada)
CUDA_ARCH = sm_75

# Compiler flags
NVCCFLAGS = -arch=$(CUDA_ARCH) -O3 -std=c++11 --no-device-link

# Linker flags
LDFLAGS =

# Default target
all: $(TARGET)

# Build target
$(TARGET): $(SRCS)
	$(NVCC) $(NVCCFLAGS) $(SRCS) -o $(TARGET) $(LDFLAGS)

# Run the test
run: $(TARGET)
	./$(TARGET)

# Clean build artifacts
clean:
	rm -f $(TARGET) *.o

# Phony targets
.PHONY: all clean run
