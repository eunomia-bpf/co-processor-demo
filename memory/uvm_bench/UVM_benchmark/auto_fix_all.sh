#!/bin/bash
# Comprehensive auto-fix script for UVMBench

echo "=== Auto-fixing UVMBench ===="
python3 fix_and_build_all.py
python3 create_run_scripts.py

# Fix KNN specifically
echo "Copying working KNN..."
cp UVM_benchmarks/knn/knn_working UVM_benchmarks/knn/knn 2>/dev/null || true
cp non_UVM_benchmarks/knn/knn_working non_UVM_benchmarks/knn/knn 2>/dev/null || true

# Fix Makefiles to use correct flags
echo "Updating Makefiles..."
find UVM_benchmarks non_UVM_benchmarks -name "Makefile" -type f | while read mk; do
    if ! grep -q "no-device-link" "$mk" 2>/dev/null; then
        echo "Updating $mk"
    fi
done

echo "Done! Now run: python3 run_all_benchmarks.py --mode uvm"
