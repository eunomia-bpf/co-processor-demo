#!/bin/bash

# Script to test build and run status of all benchmarks
# Output: Summary of which benchmarks build and which run successfully

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

BASE_DIR="/home/yunwei37/workspace/playground/co-processor-demo/memory/uvm_bench/UVM_benchmark"
LOG_FILE="$BASE_DIR/benchmark_test_results.log"

# Clear previous log
> "$LOG_FILE"

echo "===============================================" | tee -a "$LOG_FILE"
echo "UVM Benchmark Test Report" | tee -a "$LOG_FILE"
echo "Date: $(date)" | tee -a "$LOG_FILE"
echo "===============================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Track statistics
TOTAL_BENCHMARKS=0
BUILD_SUCCESS=0
BUILD_FAIL=0
RUN_SUCCESS=0
RUN_FAIL=0

# Array to store results
declare -A BUILD_STATUS
declare -A RUN_STATUS

# Test a single benchmark
test_benchmark() {
    local bench_dir=$1
    local bench_name=$2

    TOTAL_BENCHMARKS=$((TOTAL_BENCHMARKS + 1))

    echo "Testing: $bench_name" | tee -a "$LOG_FILE"

    # Try to build
    cd "$bench_dir" || return

    # Clean first
    make clean &>/dev/null

    # Build
    if make &>> "$LOG_FILE"; then
        echo -e "  ${GREEN}[BUILD OK]${NC}" | tee -a "$LOG_FILE"
        BUILD_STATUS[$bench_name]="OK"
        BUILD_SUCCESS=$((BUILD_SUCCESS + 1))

        # Find executable - prefer run script if it exists
        local exec_name=""
        if [ -f "run" ] && [ -x "run" ]; then
            exec_name="./run"
        elif [ -f "main" ]; then
            exec_name="./main"
        elif [ -f "ordergraph" ]; then
            exec_name="./ordergraph"
        elif [ -f "CNN" ]; then
            exec_name="./CNN"
        elif [ -f "kmeans_cuda" ]; then
            exec_name="./kmeans_cuda"
        elif [ -f "knn" ]; then
            exec_name="./knn"
        elif [ -f "gpu_exec" ]; then
            exec_name="./gpu_exec"
        elif [ -f "svm_uvm" ]; then
            exec_name="./svm_uvm"
        else
            # Try to find any executable
            exec_name=$(find . -maxdepth 1 -type f -executable ! -name "*.sh" | head -n 1)
        fi

        # Try to run (with timeout)
        if [ -n "$exec_name" ]; then
            echo "  Attempting to run: $exec_name" >> "$LOG_FILE"
            if timeout 5s $exec_name &>> "$LOG_FILE"; then
                echo -e "  ${GREEN}[RUN OK]${NC}" | tee -a "$LOG_FILE"
                RUN_STATUS[$bench_name]="OK"
                RUN_SUCCESS=$((RUN_SUCCESS + 1))
            else
                local exit_code=$?
                if [ $exit_code -eq 124 ]; then
                    echo -e "  ${YELLOW}[RUN TIMEOUT]${NC}" | tee -a "$LOG_FILE"
                    RUN_STATUS[$bench_name]="TIMEOUT"
                else
                    echo -e "  ${RED}[RUN FAIL]${NC}" | tee -a "$LOG_FILE"
                    RUN_STATUS[$bench_name]="FAIL"
                    RUN_FAIL=$((RUN_FAIL + 1))
                fi
            fi
        else
            echo -e "  ${YELLOW}[NO EXECUTABLE FOUND]${NC}" | tee -a "$LOG_FILE"
            RUN_STATUS[$bench_name]="NO_EXEC"
        fi
    else
        echo -e "  ${RED}[BUILD FAIL]${NC}" | tee -a "$LOG_FILE"
        BUILD_STATUS[$bench_name]="FAIL"
        BUILD_FAIL=$((BUILD_FAIL + 1))
        RUN_STATUS[$bench_name]="N/A"
    fi

    echo "" | tee -a "$LOG_FILE"
}

echo "=== Testing UVM_benchmarks ===" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Test main benchmarks
test_benchmark "$BASE_DIR/UVM_benchmarks/bfs" "bfs"
test_benchmark "$BASE_DIR/UVM_benchmarks/BN" "BN"
test_benchmark "$BASE_DIR/UVM_benchmarks/CNN" "CNN"
test_benchmark "$BASE_DIR/UVM_benchmarks/kmeans" "kmeans"
test_benchmark "$BASE_DIR/UVM_benchmarks/knn" "knn"
test_benchmark "$BASE_DIR/UVM_benchmarks/logistic-regression" "logistic-regression"
test_benchmark "$BASE_DIR/UVM_benchmarks/SVM" "SVM"

# Test rodinia benchmarks
echo "=== Testing Rodinia benchmarks ===" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

for bench in backprop dwt2d gaussian hotspot hotspot3D nn nw particlefilter pathfinder srad streamcluster; do
    test_benchmark "$BASE_DIR/UVM_benchmarks/rodinia/$bench" "rodinia/$bench"
done

# Test polybench benchmarks
echo "=== Testing Polybench benchmarks ===" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

for bench in 2DCONV 2MM 3DCONV 3MM ATAX BICG CORR COVAR FDTD-2D GEMM GESUMMV GRAMSCHM MVT SYR2K SYRK; do
    test_benchmark "$BASE_DIR/UVM_benchmarks/polybench/$bench" "polybench/$bench"
done

# Print summary
echo "===============================================" | tee -a "$LOG_FILE"
echo "SUMMARY" | tee -a "$LOG_FILE"
echo "===============================================" | tee -a "$LOG_FILE"
echo "Total benchmarks tested: $TOTAL_BENCHMARKS" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Build Statistics:" | tee -a "$LOG_FILE"
echo -e "  ${GREEN}Success: $BUILD_SUCCESS${NC}" | tee -a "$LOG_FILE"
echo -e "  ${RED}Failed: $BUILD_FAIL${NC}" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Run Statistics:" | tee -a "$LOG_FILE"
echo -e "  ${GREEN}Success: $RUN_SUCCESS${NC}" | tee -a "$LOG_FILE"
echo -e "  ${RED}Failed: $RUN_FAIL${NC}" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# List failing benchmarks
echo "Build Failures:" | tee -a "$LOG_FILE"
for bench in "${!BUILD_STATUS[@]}"; do
    if [ "${BUILD_STATUS[$bench]}" == "FAIL" ]; then
        echo "  - $bench" | tee -a "$LOG_FILE"
    fi
done
echo "" | tee -a "$LOG_FILE"

echo "Run Failures:" | tee -a "$LOG_FILE"
for bench in "${!RUN_STATUS[@]}"; do
    if [ "${RUN_STATUS[$bench]}" == "FAIL" ]; then
        echo "  - $bench" | tee -a "$LOG_FILE"
    fi
done
echo "" | tee -a "$LOG_FILE"

echo "Run Timeouts (may be working correctly):" | tee -a "$LOG_FILE"
for bench in "${!RUN_STATUS[@]}"; do
    if [ "${RUN_STATUS[$bench]}" == "TIMEOUT" ]; then
        echo "  - $bench" | tee -a "$LOG_FILE"
    fi
done

echo "" | tee -a "$LOG_FILE"
echo "Full log saved to: $LOG_FILE" | tee -a "$LOG_FILE"
