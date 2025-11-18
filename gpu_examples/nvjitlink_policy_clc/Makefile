NVCC = nvcc
CUDA_ARCH = sm_120

NVCC_FLAGS = -std=c++17 -arch=$(CUDA_ARCH)
PTX_FLAGS = $(NVCC_FLAGS) --relocatable-device-code=true

all: gemm_test_modify gemm_test_original wrapper_kernel.ptx policy_greedy.ptx policy_maxsteals.ptx

# Main executable - CLC scheduler version with policy framework
# Compile with --keep-device-functions to preserve device functions in PTX
# Define USE_POLICY_FRAMEWORK to enable nvJitLink + CLC scheduler
gemm_test_modify: gemm_test_modify.cu policy_framework.h
	$(NVCC) $(NVCC_FLAGS) -DUSE_POLICY_FRAMEWORK --keep-device-functions $< -o $@ -lcuda -lnvJitLink

# Original executable - Direct kernel launch (no policy framework)
# Compile without USE_POLICY_FRAMEWORK flag for baseline comparison
gemm_test_original: gemm_test_modify.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

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
	rm -f gemm_test gemm_test_modify gemm_test_original wrapper_kernel *.ptx *.cubin *.o

# Run original version (no policy framework)
run-original: gemm_test_original
	@echo "=== Running Original Version (Direct Kernel Launch) ==="
	./gemm_test_original

# Convenience targets for running with different policies
run-greedy: all
	@echo "=== Running with GreedyPolicy ==="
	WRAPPER_KERNEL_PATH=./wrapper_kernel.ptx POLICY_PTX_PATH=./policy_greedy.ptx ./gemm_test_modify

run-maxsteals: all
	@echo "=== Running with MaxStealsPolicy ==="
	WRAPPER_KERNEL_PATH=./wrapper_kernel.ptx POLICY_PTX_PATH=./policy_maxsteals.ptx ./gemm_test_modify

run-all: all
	@echo ""
	@echo "========================================="
	@echo "Comparison: Original vs Policy Framework"
	@echo "========================================="
	@echo ""
	@$(MAKE) run-original
	@echo ""
	@echo "========================================="
	@echo ""
	@$(MAKE) run-greedy
	@echo ""
	@echo "========================================="
	@echo ""
	@$(MAKE) run-maxsteals
	@echo ""
	@echo "========================================="
	@echo "Comparison Complete"
	@echo "========================================="

run: run-greedy

help:
	@echo "========================================="
	@echo "CUDA 12 nvJitLink + CLC Scheduler Framework"
	@echo "========================================="
	@echo ""
	@echo "BUILD TARGETS:"
	@echo "  make              - Build demo and all policy PTX files"
	@echo "  make clean        - Clean all build artifacts"
	@echo "  make help         - Show this help message"
	@echo ""
	@echo "RUN TARGETS:"
	@echo "  make run           - Run with GreedyPolicy (default)"
	@echo "  make run-original  - Run original version (no policy)"
	@echo "  make run-greedy    - Run with GreedyPolicy"
	@echo "  make run-maxsteals - Run with MaxStealsPolicy"
	@echo "  make run-all       - Compare original vs all policies"
	@echo ""
	@echo "========================================="
	@echo "AVAILABLE POLICIES:"
	@echo "========================================="
	@echo ""
	@echo "1. GreedyPolicy (policy_greedy.ptx)"
	@echo "   - Always executes (baseline CLC behavior)"
	@echo "   - No throttling or limiting"
	@echo "   - Best for uniform workloads"
	@echo ""
	@echo "2. MaxStealsPolicy (policy_maxsteals.ptx)"
	@echo "   - Stops after 8 executions per block"
	@echo "   - Useful for load balancing"
	@echo "   - Prevents excessive work stealing"
	@echo ""
	@echo "========================================="
	@echo "MANUAL USAGE:"
	@echo "========================================="
	@echo ""
	@echo "Run with GreedyPolicy:"
	@echo "  WRAPPER_KERNEL_PATH=./wrapper_kernel.ptx \\"
	@echo "  POLICY_PTX_PATH=./policy_greedy.ptx \\"
	@echo "  ./gemm_test_modify"
	@echo ""
	@echo "Run with MaxStealsPolicy:"
	@echo "  WRAPPER_KERNEL_PATH=./wrapper_kernel.ptx \\"
	@echo "  POLICY_PTX_PATH=./policy_maxsteals.ptx \\"
	@echo "  ./gemm_test_modify"
	@echo ""
	@echo "========================================="
	@echo "HOW IT WORKS:"
	@echo "========================================="
	@echo ""
	@echo "1. User kernel (gemm_kernel_impl) compiled into main binary"
	@echo "2. cuobjdump extracts PTX from binary at runtime"
	@echo "3. Wrapper kernel (CLC scheduler) loaded from ENV"
	@echo "4. Policy logic loaded from ENV"
	@echo "5. nvJitLink combines all three at runtime"
	@echo "6. Policy controls execution flow using CLC pattern:"
	@echo "   - Thread 0 evaluates policy decision"
	@echo "   - Decision broadcast to all threads"
	@echo "   - Uniform control flow maintained"
	@echo ""
	@echo "Based on CLC policy framework from:"
	@echo "  scheduler/clc_bench/clc_policy_framework.cuh"
	@echo "========================================="

.PHONY: all ptx clean run help
