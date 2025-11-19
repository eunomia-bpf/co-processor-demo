# choose your compiler, e.g. gcc/clang
# example override to clang: make run CC=clang
CC = gcc
# For CUDA compile
CUDA_INSTALL_PATH ?= /usr/local/cuda-12.9
NVCC := "$(CUDA_INSTALL_PATH)/bin/nvcc"
INCLUDES := -I"$(CUDA_INSTALL_PATH)/include"
LIB_PATH ?= $(CUDA_INSTALL_PATH)/lib64

CUDA_ARCH = sm_120
NVCC_FLAGS = -std=c++17 -arch=$(CUDA_ARCH) $(INCLUDES) -L $(LIB_PATH)
PTX_FLAGS = $(NVCC_FLAGS) --relocatable-device-code=true

# compile the Cuda version (with dynamic libcudart for eBPF uprobe profiling)
# Use -Xcompiler to pass frame pointer flag to host compiler for eBPF stack unwinding
.PHONY: runcu
runcu: runcu.cu
	$(NVCC) $(INCLUDES) -O2 -Xcompiler -fno-omit-frame-pointer -Wno-deprecated-gpu-targets -o runcu runcu.cu -L $(LIB_PATH) -lcudart -lm

# compile cublas included
.PHONY: runcublas
runcublas: runcu.cu
	$(NVCC) $(INCLUDES) -O3 -Wno-deprecated-gpu-targets -DUSE_CUBLAS -o runcublas runcu.cu -L $(LIB_PATH) -lcudart -lm -lcublas

# Policy framework version - CLC scheduler version with policy framework
# Compile with --keep-device-functions to preserve device functions in PTX
# Define USE_POLICY_FRAMEWORK to enable nvJitLink + CLC scheduler
.PHONY: runcu_policy
runcu_policy: runcu.cu policy_framework.h
	$(NVCC) $(NVCC_FLAGS) -O2 -Xcompiler -fno-omit-frame-pointer -DUSE_POLICY_FRAMEWORK --keep-device-functions runcu.cu -o runcu_policy -lcudart -lm -lcuda -lnvJitLink

# Wrapper kernel PTX (implements CLC scheduler with work-stealing for matmul)
wrapper_kernel.ptx: wrapper_kernel.cu
	$(NVCC) $(PTX_FLAGS) -ptx wrapper_kernel.cu -o wrapper_kernel.ptx

# Policy PTX files - different scheduling policies
policy_greedy.ptx: policy_greedy.cu
	$(NVCC) $(PTX_FLAGS) -ptx policy_greedy.cu -o policy_greedy.ptx

policy_maxsteals.ptx: policy_maxsteals.cu
	$(NVCC) $(PTX_FLAGS) -ptx policy_maxsteals.cu -o policy_maxsteals.ptx

policy_nosteal.ptx: policy_nosteal.cu
	$(NVCC) $(PTX_FLAGS) -ptx policy_nosteal.cu -o policy_nosteal.ptx

# Build all policy components
.PHONY: policy-all
policy-all: runcu_policy wrapper_kernel.ptx policy_greedy.ptx policy_maxsteals.ptx policy_nosteal.ptx

# download the model
.PHONY: download-model
download-model:
	@if [ -f Qwen3-0.6B-FP32.gguf ] && [ $$(stat -c%s Qwen3-0.6B-FP32.gguf) -gt 1000000 ]; then \
		echo "Model already exists (size: $$(du -h Qwen3-0.6B-FP32.gguf | cut -f1))"; \
	else \
		echo "Downloading Qwen3-0.6B model (3GB - this will take a while)..."; \
		wget -c https://huggingface.co/huggit0000/Qwen3-0.6B-GGUF-FP32/resolve/main/Qwen3-0.6B-FP32.gguf -O Qwen3-0.6B-FP32.gguf || \
		curl -L -C - https://huggingface.co/huggit0000/Qwen3-0.6B-GGUF-FP32/resolve/main/Qwen3-0.6B-FP32.gguf -o Qwen3-0.6B-FP32.gguf; \
		echo "Model downloaded successfully (size: $$(du -h Qwen3-0.6B-FP32.gguf | cut -f1))"; \
	fi

# =========================
# The below is not used hree.


# the most basic way of building that is most likely to work on most systems
.PHONY: run
run: run.c
	$(CC) -O3 -o run run.c -lm

# useful for a debug build, can then e.g. analyze with valgrind, example:
# $ valgrind --leak-check=full ./run out/model.bin -n 3
rundebug: run.c
	$(CC) -g -o run run.c -lm

# https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html
# https://simonbyrne.github.io/notes/fastmath/
# -Ofast enables all -O3 optimizations.
# Disregards strict standards compliance.
# It also enables optimizations that are not valid for all standard-compliant programs.
# It turns on -ffast-math, -fallow-store-data-races and the Fortran-specific
# -fstack-arrays, unless -fmax-stack-var-size is specified, and -fno-protect-parens.
# It turns off -fsemantic-interposition.
# In our specific application this is *probably* okay to use
#.PHONY: run
#runfast: run.c
#	$(CC) -O3 -o run -fopenmp -march=native run.c -lm

# additionally compiles with OpenMP, allowing multithreaded runs
# make sure to also enable multiple threads when running, e.g.:
# OMP_NUM_THREADS=4 ./run out/model.bin
.PHONY: runomp
runomp: run.c
	$(CC) -O3 -fopenmp -march=native run.c  -lm  -o run


# Run original version (no policy framework)
.PHONY: run-original
run-original: runcu
	@echo "=== Running Original Version (Direct Kernel Launch) ==="
	./runcu Qwen3-0.6B-FP32.gguf -q "What is CUDA?" -r 0

# Convenience targets for running with different policies
.PHONY: run-greedy
run-greedy: policy-all
	@echo "=== Running with GreedyPolicy ==="
	WRAPPER_KERNEL_PATH=./wrapper_kernel.ptx POLICY_PTX_PATH=./policy_greedy.ptx ./runcu_policy Qwen3-0.6B-FP32.gguf -q "What is CUDA?" -r 0

.PHONY: run-maxsteals
run-maxsteals: policy-all
	@echo "=== Running with MaxStealsPolicy ==="
	WRAPPER_KERNEL_PATH=./wrapper_kernel.ptx POLICY_PTX_PATH=./policy_maxsteals.ptx ./runcu_policy Qwen3-0.6B-FP32.gguf -q "What is CUDA?" -r 0

.PHONY: run-nosteal
run-nosteal: policy-all
	@echo "=== Running with NoStealPolicy ==="
	WRAPPER_KERNEL_PATH=./wrapper_kernel.ptx POLICY_PTX_PATH=./policy_nosteal.ptx ./runcu_policy Qwen3-0.6B-FP32.gguf -q "What is CUDA?" -r 0

.PHONY: run-all
run-all: runcu policy-all
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
	@echo ""
	@$(MAKE) run-nosteal
	@echo ""
	@echo "========================================="
	@echo "Comparison Complete"
	@echo "========================================="

.PHONY: help-policy
help-policy:
	@echo "========================================="
	@echo "Qwen3 CUDA Inference with CLC Framework"
	@echo "========================================="
	@echo ""
	@echo "BUILD TARGETS:"
	@echo "  make policy-all    - Build policy framework version + PTX files"
	@echo "  make runcu         - Build original version (no policy)"
	@echo "  make clean         - Clean all build artifacts"
	@echo "  make clean-all     - Clean build artifacts + PTX files"
	@echo ""
	@echo "RUN TARGETS:"
	@echo "  make run-greedy    - Run with GreedyPolicy"
	@echo "  make run-original  - Run original version (no policy)"
	@echo "  make run-maxsteals - Run with MaxStealsPolicy"
	@echo "  make run-nosteal   - Run with NoStealPolicy"
	@echo "  make run-all       - Compare original vs all policies"
	@echo ""
	@echo "========================================="
	@echo "AVAILABLE POLICIES:"
	@echo "========================================="
	@echo ""
	@echo "1. GreedyPolicy (policy_greedy.ptx)"
	@echo "   - Always tries to steal more work"
	@echo "   - Maximum throughput via work-stealing"
	@echo "   - Best for imbalanced workloads"
	@echo ""
	@echo "2. MaxStealsPolicy (policy_maxsteals.ptx)"
	@echo "   - Stops after 8 steal attempts"
	@echo "   - Useful for load balancing"
	@echo "   - Prevents excessive work stealing"
	@echo ""
	@echo "3. NoStealPolicy (policy_nosteal.ptx)"
	@echo "   - Execute assigned work once, never steal"
	@echo "   - Mimics traditional CUDA behavior"
	@echo "   - Lowest overhead, no stealing benefits"
	@echo ""
	@echo "========================================="
	@echo "MANUAL USAGE:"
	@echo "========================================="
	@echo ""
	@echo "Run with GreedyPolicy:"
	@echo "  WRAPPER_KERNEL_PATH=./wrapper_kernel.ptx \\"
	@echo "  POLICY_PTX_PATH=./policy_greedy.ptx \\"
	@echo "  ./runcu_policy Qwen3-0.6B-FP32.gguf -q \"What is CUDA?\""
	@echo ""
	@echo "Run with MaxStealsPolicy:"
	@echo "  WRAPPER_KERNEL_PATH=./wrapper_kernel.ptx \\"
	@echo "  POLICY_PTX_PATH=./policy_maxsteals.ptx \\"
	@echo "  ./runcu_policy Qwen3-0.6B-FP32.gguf -q \"What is CUDA?\""
	@echo ""
	@echo "========================================="

.PHONY: clean
clean:
	rm -f run runcu runcublas runcu_policy

.PHONY: clean-all
clean-all: clean
	rm -f *.ptx *.cubin *.o
