NVCC = nvcc
CUDA_ARCH = sm_120

NVCC_FLAGS = -std=c++17 -arch=$(CUDA_ARCH)
PTX_FLAGS = $(NVCC_FLAGS) --relocatable-device-code=true

all: nvjitlink_demo ptx

# Main demo executable
nvjitlink_demo: nvjitlink_demo.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@ -lcuda -lnvJitLink

# Generate PTX for runtime linking
ptx: user_kernel.ptx policy.ptx

user_kernel.ptx: user_kernel.cu
	$(NVCC) $(PTX_FLAGS) -ptx $< -o $@

policy.ptx: policy.cu
	$(NVCC) $(PTX_FLAGS) -ptx $< -o $@

# Alternative: Generate LTO-IR for even better optimization
ltoir: user_kernel.ltoir policy.ltoir

user_kernel.ltoir: user_kernel.cu
	$(NVCC) $(NVCC_FLAGS) -dlto -fatbin $< -o $@

policy.ltoir: policy.cu
	$(NVCC) $(NVCC_FLAGS) -dlto -fatbin $< -o $@

clean:
	rm -f nvjitlink_demo *.ptx *.ltoir *.cubin

run: all
	./nvjitlink_demo

help:
	@echo "CUDA 12+ nvJitLink + LTO Demo"
	@echo "=============================="
	@echo "Targets:"
	@echo "  make          - Build demo and generate PTX"
	@echo "  make ptx      - Generate PTX files for linking"
	@echo "  make ltoir    - Generate LTO-IR files (best optimization)"
	@echo "  make run      - Build and run demo"
	@echo "  make clean    - Clean build artifacts"
	@echo ""
	@echo "This demo shows:"
	@echo "  - nvJitLink API for runtime linking"
	@echo "  - Link-Time Optimization (LTO) at runtime"
	@echo "  - Combining user kernel + policy at runtime"
	@echo "  - Better performance than separate compilation"

.PHONY: all ptx ltoir clean run help
