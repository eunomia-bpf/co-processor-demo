# Makefile for GEMM CUDA test with runtime dynamic linking

# CUDA compiler
NVCC = nvcc

# Target executable
TARGET = gemm_test

# Source files
MAIN_SRC = gemm_test.cu
POLICY_SRC = policy.cu
WRAPPER_SRC = wrapper_kernel.cu
USER_KERNEL_SRC = gemm_kernel.cu

# Output cubin (combined wrapper + user kernel + policy)
WRAPPER_CUBIN = wrapper_kernel.cubin

# CUDA architecture
CUDA_ARCH = sm_120

# Compiler flags for main app
NVCCFLAGS = -arch=$(CUDA_ARCH) -O3 -std=c++11

# Cubin compilation flags
CUBIN_FLAGS = -arch=$(CUDA_ARCH) -O3 -std=c++11

# Default target: build all
all: $(TARGET) $(WRAPPER_CUBIN)

# Build main app (host code only)
$(TARGET): $(MAIN_SRC) gemm_policy_wrapper.h
	$(NVCC) $(NVCCFLAGS) $(MAIN_SRC) -o $(TARGET) -lcuda

# Build separate cubins for runtime dynamic linking
# Each module is compiled independently with relocatable device code
wrapper_kernel.cubin: $(WRAPPER_SRC)
	$(NVCC) -arch=$(CUDA_ARCH) -O3 -std=c++11 -rdc=true -cubin $(WRAPPER_SRC) -o $@

gemm_kernel.cubin: $(USER_KERNEL_SRC)
	$(NVCC) -arch=$(CUDA_ARCH) -O3 -std=c++11 -rdc=true -cubin $(USER_KERNEL_SRC) -o $@

policy.cubin: $(POLICY_SRC)
	$(NVCC) -arch=$(CUDA_ARCH) -O3 -std=c++11 -rdc=true -cubin $(POLICY_SRC) -o $@

# Run the test (requires all three cubins for runtime linking)
run: $(TARGET) wrapper_kernel.cubin gemm_kernel.cubin policy.cubin
	./$(TARGET)

# Clean build artifacts
clean:
	rm -f $(TARGET) *.o *.cubin

# Phony targets
.PHONY: all clean run
