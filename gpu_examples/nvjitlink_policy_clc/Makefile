NVCC = nvcc
CUDA_ARCH = sm_120

NVCC_FLAGS = -std=c++17 -arch=$(CUDA_ARCH)
PTX_FLAGS = $(NVCC_FLAGS) --relocatable-device-code=true

all: gemm_test_modify wrapper_kernel.ptx policy_greedy.ptx policy_maxsteals.ptx

# Main executable - CLC scheduler version
# Compile with --keep-device-functions to preserve device functions in PTX
gemm_test_modify: gemm_test_modify.cu policy_framework.h
	$(NVCC) $(NVCC_FLAGS) --keep-device-functions $< -o $@ -lcuda -lnvJitLink

# Wrapper kernel PTX (implements CLC scheduler with work-stealing)
wrapper_kernel.ptx: wrapper_kernel.cu
	$(NVCC) $(PTX_FLAGS) -ptx $< -o $@

# Policy PTX files - different scheduling policies
policy_greedy.ptx: policy_greedy.cu
	$(NVCC) $(PTX_FLAGS) -ptx $< -o $@

policy_maxsteals.ptx: policy_maxsteals.cu
	$(NVCC) $(PTX_FLAGS) -ptx $< -o $@

# Legacy policy (for backward compatibility)
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
	@echo "CUDA 12 nvJitLink + CLC Scheduler Policy Framework"
	@echo "===================================================="
	@echo "Targets:"
	@echo "  make          - Build demo and all policy PTX files"
	@echo "  make run      - Build and run with default policy"
	@echo "  make clean    - Clean artifacts"
	@echo ""
	@echo "Available Policies:"
	@echo "  policy_greedy.ptx      - Always execute (baseline CLC behavior)"
	@echo "  policy_maxsteals.ptx   - Stop after 8 executions per block"
	@echo ""
	@echo "Usage:"
	@echo "  WRAPPER_KERNEL_PATH=./wrapper_kernel.ptx POLICY_PTX_PATH=./policy_greedy.ptx ./gemm_test_modify"
	@echo "  WRAPPER_KERNEL_PATH=./wrapper_kernel.ptx POLICY_PTX_PATH=./policy_maxsteals.ptx ./gemm_test_modify"
	@echo ""
	@echo "This demo shows:"
	@echo "  - CLC (Cluster Launch Control) work-stealing scheduler"
	@echo "  - Policy-based scheduling decisions"
	@echo "  - nvJitLink for runtime policy injection"
	@echo "  - Pure PTX-based linking with cuobjdump extraction"

.PHONY: all ptx clean run help
