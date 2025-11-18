# Makefile for GEMM CUDA test with runtime dynamic linking

# CUDA compiler
NVCC = nvcc

# Target executable
TARGET = gemm_test

# Source files
MAIN_SRC = gemm_test.cu
POLICY_SRC = policy.cu
WRAPPER_SRC = wrapper_kernel.cu

# Output cubins
WRAPPER_CUBIN = wrapper_kernel.cubin
GEMM_TEST_CUBIN = gemm_test_kernel.cubin

# CUDA architecture
CUDA_ARCH = sm_120

# Compiler flags for main app (no device linking to avoid CUDA 12.9 link.stub errors)
NVCCFLAGS = -arch=$(CUDA_ARCH) -O3 -std=c++11 --no-device-link

# Cubin compilation flags
CUBIN_FLAGS = -arch=$(CUDA_ARCH) -O3 -std=c++11

# Default target: build all
all: $(TARGET) $(WRAPPER_CUBIN) $(GEMM_TEST_CUBIN)

# Build main app with device code included
$(TARGET): $(MAIN_SRC) gemm_policy_wrapper.h
	$(NVCC) $(NVCCFLAGS) $(MAIN_SRC) -o $(TARGET) -lcuda

# Extract kernel cubin from gemm_test.cu (contains the user's gemm_kernel)
$(GEMM_TEST_CUBIN): $(MAIN_SRC)
	$(NVCC) -arch=$(CUDA_ARCH) -O3 -std=c++11 -rdc=true -cubin $(MAIN_SRC) -o $@

# Build wrapper cubin
wrapper_kernel.cubin: $(WRAPPER_SRC)
	$(NVCC) -arch=$(CUDA_ARCH) -O3 -std=c++11 -rdc=true -cubin $(WRAPPER_SRC) -o $@

# Build policy cubin
policy.cubin: $(POLICY_SRC)
	$(NVCC) -arch=$(CUDA_ARCH) -O3 -std=c++11 -rdc=true -cubin $(POLICY_SRC) -o $@

# Run the test (requires all three cubins for runtime linking)
run: $(TARGET) wrapper_kernel.cubin $(GEMM_TEST_CUBIN) policy.cubin
	./$(TARGET)

# Clean build artifacts
clean:
	rm -f $(TARGET) *.o *.cubin

# Phony targets
.PHONY: all clean run
