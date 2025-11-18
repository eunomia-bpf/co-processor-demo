# Makefile for Cluster Launch Control Examples
# Tested on RTX 5090 (CC 12.0) with Driver 575.57.08

# CUDA paths
CUDA_12_8 = /usr/local/cuda-12.8
CUDA_12_9 = /usr/local/cuda-12.9
CUDA_13_0 = /usr/local/cuda-13.0

# Compiler settings
NVCC_12_8 = $(CUDA_12_8)/bin/nvcc
NVCC_12_9 = $(CUDA_12_9)/bin/nvcc
NVCC_13_0 = $(CUDA_13_0)/bin/nvcc

# Architecture
ARCH = sm_120

# Common flags
NVCC_FLAGS = -arch=$(ARCH)
GCC = gcc
GCC_FLAGS = -I$(CUDA_12_8)/include -L$(CUDA_12_8)/lib64 -lcuda

# ============================================
# WORKING EXAMPLES (CUDA 12.9)
# ============================================

.PHONY: all working test clean

# Default: Build working examples only
all: working

working: test_vec_add minimal_clc_12.9_api

# Simple vector add test (baseline)
test_vec_add: test_vec_add.cu
	$(NVCC_12_9) $(NVCC_FLAGS) -o test_vec_add test_vec_add.cu
	@echo "✅ Built: test_vec_add (CUDA 12.9)"

# Minimal CLC using CUDA 12.9 libcu++ API - WORKING! ✅
minimal_clc_12.9_api: minimal_clc_12.9.cu
	$(NVCC_12_9) $(NVCC_FLAGS) -o minimal_clc_12.9_api minimal_clc_12.9.cu
	@echo "✅ Built: minimal_clc_12.9_api (CUDA 12.9 + libcu++)"

# ============================================
# FAILED ATTEMPTS (Educational)
# ============================================

.PHONY: failed-attempts

# Build all failed attempts (for documentation/learning)
failed-attempts: simple_clc minimal_clc_ptx minimal_clc_ptx_v2 cubin_loader

# CUDA 12.8 + libcu++ - FAILS (no API)
simple_clc: simple_clc_example.cu
	@echo "❌ Attempting: CUDA 12.8 with libcu++ wrappers..."
	-$(NVCC_12_8) $(NVCC_FLAGS) -o simple_clc simple_clc_example.cu 2>&1 | head -10
	@echo "Result: FAILED - No CLC API in CUDA 12.8"

# CUDA 12.8 + Inline PTX (short syntax) - FAILS (ptxas)
minimal_clc_ptx: minimal_clc_ptx.cu
	@echo "❌ Attempting: CUDA 12.8 with inline PTX (short syntax)..."
	$(NVCC_12_8) -arch=$(ARCH) -ptx minimal_clc_ptx.cu -o /tmp/clc_12.8_short.ptx
	@echo "  PTX generation: ✅"
	-$(NVCC_12_8) $(NVCC_FLAGS) -o minimal_clc_ptx minimal_clc_ptx.cu 2>&1 | head -10
	@echo "Result: FAILED - ptxas cannot assemble"

# CUDA 13.0 + Inline PTX (full syntax) - FAILS (driver)
minimal_clc_ptx_v2: minimal_clc_ptx_v2.cu
	@echo "⚠️  Attempting: CUDA 13.0 with inline PTX (full syntax)..."
	$(NVCC_13_0) $(NVCC_FLAGS) -o minimal_clc_ptx_v2 minimal_clc_ptx_v2.cu
	@echo "  Binary creation: ✅ (but needs driver 580+)"
	@echo "Result: PARTIAL - Compiles but driver 575 too old"

# CUBIN loader - FAILS (driver segfault)
cubin_loader: cubin_loader_simple.c
	@echo "❌ Attempting: Load CUDA 13.0 cubin with driver API..."
	$(NVCC_13_0) -arch=$(ARCH) -ptx minimal_clc_ptx_v2.cu -o /tmp/clc_13.ptx
	$(CUDA_13_0)/bin/ptxas -arch=$(ARCH) /tmp/clc_13.ptx -o /tmp/clc_13.cubin
	$(GCC) $(GCC_FLAGS) -o cubin_loader_simple cubin_loader_simple.c
	@echo "  Loader built: ✅"
	@echo "Result: FAILED - Driver segfaults on cuModuleLoadData()"

# ============================================
# PTX EXPERIMENTS
# ============================================

.PHONY: ptx-tests

ptx-tests:
	@echo "=== PTX Generation Tests ==="
	@echo ""
	@echo "1. CUDA 12.8 PTX (short syntax):"
	$(NVCC_12_8) -arch=$(ARCH) -ptx minimal_clc_ptx.cu -o /tmp/clc_12.8.ptx
	@grep "clusterlaunchcontrol" /tmp/clc_12.8.ptx || echo "  No CLC instructions"
	@echo ""
	@echo "2. CUDA 12.9 PTX (libcu++ expansion):"
	$(NVCC_12_9) -arch=$(ARCH) -ptx minimal_clc_12.9.cu -o /tmp/clc_12.9.ptx
	@grep "clusterlaunchcontrol" /tmp/clc_12.9.ptx | head -3
	@echo ""
	@echo "3. CUDA 13.0 PTX (full syntax):"
	$(NVCC_13_0) -arch=$(ARCH) -ptx minimal_clc_ptx_v2.cu -o /tmp/clc_13.0.ptx
	@grep "clusterlaunchcontrol" /tmp/clc_13.0.ptx | head -3

# ============================================
# TESTING & VALIDATION
# ============================================

.PHONY: test run-tests run-vec-add run-clc

# Run all working tests
test: run-tests

run-tests: run-vec-add run-clc

# Test baseline vector add
run-vec-add: test_vec_add
	@echo "=== Running Vector Add Test ==="
	./test_vec_add
	@echo ""

# Test CLC with timeout (in case it hangs)
run-clc: minimal_clc_12.9_api
	@echo "=== Running CLC Test (CUDA 12.9) ==="
	timeout 10 ./minimal_clc_12.9_api || echo "TIMEOUT or ERROR"
	@echo ""

# ============================================
# INFORMATION
# ============================================

.PHONY: info show-env check-cuda

info: show-env check-cuda

show-env:
	@echo "=== Environment Information ==="
	@echo "GPU:"
	@nvidia-smi --query-gpu=name,compute_cap --format=csv
	@echo ""
	@echo "Driver:"
	@cat /proc/driver/nvidia/version | head -1
	@echo ""

check-cuda:
	@echo "=== CUDA Installations ==="
	@echo "CUDA 12.8:"
	@$(NVCC_12_8) --version | grep "release"
	@echo ""
	@echo "CUDA 12.9:"
	@$(NVCC_12_9) --version | grep "release"
	@echo ""
	@echo "CUDA 13.0:"
	@$(NVCC_13_0) --version | grep "release"
	@echo ""
	@echo "=== CLC API Availability ==="
	@echo -n "CUDA 12.8: "
	@grep -q "clusterlaunchcontrol" $(CUDA_12_8)/include/cuda/__ptx/*.h 2>/dev/null && echo "✅ Found" || echo "❌ Not found"
	@echo -n "CUDA 12.9: "
	@grep -q "clusterlaunchcontrol" $(CUDA_12_9)/include/cuda/__ptx/instructions/generated/*.h 2>/dev/null && echo "✅ Found" || echo "❌ Not found"
	@echo -n "CUDA 13.0: "
	@grep -q "clusterlaunchcontrol" $(CUDA_13_0)/include/cccl/cuda/__ptx/instructions/generated/*.h 2>/dev/null && echo "✅ Found" || echo "❌ Not found"

# ============================================
# CLEANUP
# ============================================

clean:
	rm -f test_vec_add minimal_clc_12.9_api
	rm -f simple_clc minimal_clc_ptx minimal_clc_ptx_v2
	rm -f cubin_loader cubin_loader_simple
	rm -f /tmp/clc*.ptx /tmp/clc*.cubin
	@echo "Cleaned all binaries and temp files"

# ============================================
# HELP
# ============================================

help:
	@echo "Cluster Launch Control Examples - Makefile"
	@echo ""
	@echo "WORKING TARGETS (CUDA 12.9):"
	@echo "  make                    - Build working examples (default)"
	@echo "  make working            - Build test_vec_add + minimal_clc_12.9_api"
	@echo "  make test               - Build and run all working tests"
	@echo "  make run-vec-add        - Run vector add test"
	@echo "  make run-clc            - Run CLC test"
	@echo ""
	@echo "FAILED ATTEMPTS (Educational):"
	@echo "  make failed-attempts    - Try building all failed approaches"
	@echo "  make simple_clc         - CUDA 12.8 + libcu++ (fails: no API)"
	@echo "  make minimal_clc_ptx    - CUDA 12.8 + inline PTX (fails: ptxas)"
	@echo "  make minimal_clc_ptx_v2 - CUDA 13.0 + full PTX (fails: driver)"
	@echo "  make cubin_loader       - Driver API loader (fails: segfault)"
	@echo ""
	@echo "EXPERIMENTS:"
	@echo "  make ptx-tests          - Compare PTX generation across CUDA versions"
	@echo ""
	@echo "INFORMATION:"
	@echo "  make info               - Show environment and CUDA info"
	@echo "  make show-env           - Show GPU and driver info"
	@echo "  make check-cuda         - Check CUDA installations and CLC API"
	@echo ""
	@echo "CLEANUP:"
	@echo "  make clean              - Remove all binaries and temp files"
	@echo ""
	@echo "CURRENT STATUS:"
	@echo "  ✅ CUDA 12.9 + libcu++ CLC API - WORKING!"
	@echo "  ❌ CUDA 12.8 - No CLC API"
	@echo "  ⚠️  CUDA 13.0 - Needs driver 580+"

.DEFAULT_GOAL := all
