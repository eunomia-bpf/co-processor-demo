NVCC = nvcc
CUDA_ARCH = sm_120

NVCC_FLAGS = -std=c++17 -arch=$(CUDA_ARCH)
PTX_FLAGS = $(NVCC_FLAGS) --relocatable-device-code=true

all: gemm_test ptx

# Main executable
gemm_test: gemm_test.cu policy_framework.h
	$(NVCC) $(NVCC_FLAGS) $< -o $@ -lcuda -lnvJitLink

# Generate PTX files for runtime linking
ptx: user_kernel.ptx policy.ptx

user_kernel.ptx: gemm_test.cu policy_framework.h
	$(NVCC) $(PTX_FLAGS) -ptx $< -o $@

policy.ptx: policy.cu
	$(NVCC) $(PTX_FLAGS) -ptx $< -o $@

clean:
	rm -f gemm_test *.ptx *.cubin *.o

run: all
	./gemm_test

help:
	@echo "CUDA 12 nvJitLink Policy Framework"
	@echo "==================================="
	@echo "Targets:"
	@echo "  make          - Build demo and PTX files"
	@echo "  make ptx      - Generate PTX files"
	@echo "  make run      - Build and run"
	@echo "  make clean    - Clean artifacts"
	@echo ""
	@echo "This demo shows:"
	@echo "  - nvJitLink for runtime policy injection"
	@echo "  - Clean API compared to old cuLinkCreate"
	@echo "  - Pure PTX-based linking"
	@echo "  - Context-independent ready"

.PHONY: all ptx clean run help
