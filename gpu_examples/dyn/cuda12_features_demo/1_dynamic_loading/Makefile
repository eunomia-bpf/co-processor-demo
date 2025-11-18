NVCC = nvcc
CUDA_ARCH = sm_80

# For CUDA 12.9, we have all features
NVCC_FLAGS = -std=c++17 -arch=$(CUDA_ARCH) -O2

all: dynamic_loading_demo vector_kernels.fatbin

# Main demo executable
dynamic_loading_demo: dynamic_loading_demo.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Create fatbin with the kernels to be dynamically loaded
# Method 1: Direct fatbin generation
vector_kernels.fatbin: vector_add_kernel.cu
	$(NVCC) $(NVCC_FLAGS) --fatbin $< -o $@

# Alternative: Create cubin first, then fatbin
vector_kernels.cubin: vector_add_kernel.cu
	$(NVCC) $(NVCC_FLAGS) --cubin $< -o $@

clean:
	rm -f dynamic_loading_demo *.fatbin *.cubin *.o

run: all
	./dynamic_loading_demo

help:
	@echo "CUDA 12+ Dynamic Loading Demo"
	@echo "=============================="
	@echo "Targets:"
	@echo "  make          - Build demo and fatbin"
	@echo "  make run      - Build and run demo"
	@echo "  make clean    - Clean build artifacts"
	@echo ""
	@echo "This demo shows:"
	@echo "  - cudaLibraryLoadFromFile() to load fatbin"
	@echo "  - cudaLibraryGetKernel() to get kernel handle"
	@echo "  - cudaLaunchKernel() to launch with handle"
	@echo "  - cudaGetKernel() to get handle from static kernel"

.PHONY: all clean run help
