NVCC = nvcc
CUDA_ARCH = sm_120

NVCC_FLAGS = -std=c++17 -arch=$(CUDA_ARCH)

all: nvfatbin_demo kernels

# Main demo executable
nvfatbin_demo: nvfatbin_demo.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@ -lnvfatbin

# Build kernel files for fatbin construction
kernels: kernel1.ptx kernel2.ptx kernel1.cubin kernel2.cubin

kernel1.ptx: kernel1.cu
	$(NVCC) $(NVCC_FLAGS) -ptx $< -o $@

kernel2.ptx: kernel2.cu
	$(NVCC) $(NVCC_FLAGS) -ptx $< -o $@

kernel1.cubin: kernel1.cu
	$(NVCC) $(NVCC_FLAGS) -cubin $< -o $@

kernel2.cubin: kernel2.cu
	$(NVCC) $(NVCC_FLAGS) -cubin $< -o $@

clean:
	rm -f nvfatbin_demo *.ptx *.cubin *.fatbin

run: all
	./nvfatbin_demo

help:
	@echo "CUDA 12.4+ nvFatbin Demo"
	@echo "========================="
	@echo "Targets:"
	@echo "  make          - Build demo and kernel files"
	@echo "  make kernels  - Build PTX and cubin files"
	@echo "  make run      - Build and run demo"
	@echo "  make clean    - Clean build artifacts"
	@echo ""
	@echo "This demo shows:"
	@echo "  - nvFatbinCreate() to build fatbin at runtime"
	@echo "  - nvFatbinAddPTX/AddCubin() to add entries"
	@echo "  - nvFatbinGet() to get the final fatbin"
	@echo "  - Multi-arch and multi-format packaging"
	@echo "  - Perfect for binary rewriting pipelines"

.PHONY: all kernels clean run help
