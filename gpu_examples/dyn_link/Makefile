# Makefile for GEMM CUDA test with runtime dynamic linking

# CUDA compiler
NVCC = nvcc

# Target executable
TARGET = gemm_test

# Source files
MAIN_SRC = gemm_test.cu
POLICY_SRC = policy.cu

# Output cubins (no longer need GEMM_TEST_CUBIN or WRAPPER_CUBIN)
# The kernel is now compiled into the main executable via template

# CUDA architecture
CUDA_ARCH = sm_120

# Compiler flags for main app
NVCCFLAGS = -arch=$(CUDA_ARCH) -O3 -std=c++11

# Default target: build all
all: $(TARGET) policy.cubin

# Build policy cubin for runtime dynamic loading
policy.cubin: $(POLICY_SRC)
	$(NVCC) -arch=$(CUDA_ARCH) -O3 -std=c++11 -rdc=true -cubin $(POLICY_SRC) -o $@

# Build main app with device code using -rdc=true (relocatable device code)
# This allows the template wrapper kernel to reference undefined external symbols
# which will be resolved when we load the policy cubin at runtime
$(TARGET): $(MAIN_SRC) gemm_policy_wrapper.h
	$(NVCC) $(NVCCFLAGS) -rdc=true $(MAIN_SRC) -o $(TARGET) -lcuda

# Run the test (requires all cubins for runtime linking)
run: all
	./$(TARGET)

# Clean build artifacts
clean:
	rm -f $(TARGET) *.o *.cubin

# Phony targets
.PHONY: all clean run
