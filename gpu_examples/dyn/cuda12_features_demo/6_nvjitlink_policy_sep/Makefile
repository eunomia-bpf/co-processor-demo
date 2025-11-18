NVCC = nvcc
CUDA_ARCH = sm_120

NVCC_FLAGS = -std=c++17 -arch=$(CUDA_ARCH)
PTX_FLAGS = $(NVCC_FLAGS) --relocatable-device-code=true

all: gemm_test_modify wrapper_kernel.ptx policy.ptx

# Main executable - simplified version
# Compile with --keep-device-functions to preserve device functions in PTX
# Use -Xptxas -v to see register usage
gemm_test_modify: gemm_test_modify.cu policy_framework.h
	$(NVCC) $(NVCC_FLAGS) --keep-device-functions $< -o $@ -lcuda -lnvJitLink

# Wrapper kernel PTX (allows unresolved external symbols with relocatable device code)
wrapper_kernel.ptx: wrapper_kernel.cu
	$(NVCC) $(PTX_FLAGS) -ptx $< -o $@

# Policy PTX
policy.ptx: policy.cu
	$(NVCC) $(PTX_FLAGS) -ptx $< -o $@

# Old targets for backward compatibility (if needed)
# gemm_test: gemm_test.cu policy_framework.h
# 	$(NVCC) $(NVCC_FLAGS) $< -o $@ -lcuda -lnvJitLink

clean:
	rm -f gemm_test gemm_test_modify wrapper_kernel *.ptx *.cubin *.o

run: all
	./gemm_test_modify

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
