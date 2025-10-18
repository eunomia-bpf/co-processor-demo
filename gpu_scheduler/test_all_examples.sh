#!/bin/bash

# Test all Blackwell examples for SM120
cd cutlass/build

echo "========================================="
echo "Testing CUTLASS Blackwell Examples on RTX 5090 (SM120)"
echo "========================================="
echo ""

# Example 73 - Preferred Cluster (SM100 only)
echo "[1/11] Testing 73_blackwell_gemm_preferred_cluster..."
./examples/73_blackwell_gemm_preferred_cluster/73_blackwell_gemm_preferred_cluster --m=2048 --n=2048 --k=2048 --preferred_cluster_m=2 --preferred_cluster_n=1 --fallback_cluster_m=1 --fallback_cluster_n=1 > /tmp/test_73.log 2>&1
if [ $? -eq 0 ]; then
    echo "  ✓ PASSED"
else
    echo "  ✗ FAILED (Expected - SM100 only)"
fi
echo ""

# Example 79a - NVFP4/BF16 GEMM
echo "[2/11] Testing 79a_blackwell_geforce_nvfp4_bf16_gemm..."
./examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm --m=4096 --n=4096 --k=4096 > /tmp/test_79a.log 2>&1
if [ $? -eq 0 ]; then
    echo "  ✓ PASSED"
    tail -3 /tmp/test_79a.log
else
    echo "  ✗ FAILED"
fi
echo ""

# Example 79b - NVFP4/NVFP4 GEMM
echo "[3/11] Testing 79b_blackwell_geforce_nvfp4_nvfp4_gemm..."
./examples/79_blackwell_geforce_gemm/79b_blackwell_geforce_nvfp4_nvfp4_gemm --m=4096 --n=4096 --k=4096 > /tmp/test_79b.log 2>&1
if [ $? -eq 0 ]; then
    echo "  ✓ PASSED"
    tail -3 /tmp/test_79b.log
else
    echo "  ✗ FAILED"
fi
echo ""

# Example 79c - Mixed MXFP8/MXFP6/BF16 GEMM
echo "[4/11] Testing 79c_blackwell_geforce_mixed_mxfp8_mxfp6_bf16_gemm..."
./examples/79_blackwell_geforce_gemm/79c_blackwell_geforce_mixed_mxfp8_mxfp6_bf16_gemm --m=4096 --n=4096 --k=4096 > /tmp/test_79c.log 2>&1
if [ $? -eq 0 ]; then
    echo "  ✓ PASSED"
    tail -3 /tmp/test_79c.log
else
    echo "  ✗ FAILED"
fi
echo ""

# Example 79d - NVFP4 Grouped GEMM
echo "[5/11] Testing 79d_blackwell_geforce_nvfp4_grouped_gemm..."
./examples/79_blackwell_geforce_gemm/79d_blackwell_geforce_nvfp4_grouped_gemm --groups=3 > /tmp/test_79d.log 2>&1
if [ $? -eq 0 ]; then
    echo "  ✓ PASSED"
    tail -3 /tmp/test_79d.log
else
    echo "  ✗ FAILED"
fi
echo ""

# Example 80a - MXFP8/BF16 Sparse GEMM
echo "[6/11] Testing 80a_blackwell_geforce_mxfp8_bf16_sparse_gemm..."
./examples/80_blackwell_geforce_sparse_gemm/80a_blackwell_geforce_mxfp8_bf16_sparse_gemm --m=4096 --n=4096 --k=4096 > /tmp/test_80a.log 2>&1
if [ $? -eq 0 ]; then
    echo "  ✓ PASSED"
    tail -3 /tmp/test_80a.log
else
    echo "  ✗ FAILED"
fi
echo ""

# Example 80b - NVFP4/NVFP4 Sparse GEMM
echo "[7/11] Testing 80b_blackwell_geforce_nvfp4_nvfp4_sparse_gemm..."
./examples/80_blackwell_geforce_sparse_gemm/80b_blackwell_geforce_nvfp4_nvfp4_sparse_gemm --m=4096 --n=4096 --k=4096 > /tmp/test_80b.log 2>&1
if [ $? -eq 0 ]; then
    echo "  ✓ PASSED"
    tail -3 /tmp/test_80b.log
else
    echo "  ✗ FAILED"
fi
echo ""

# Example 86 - Mixed Dtype GEMM
echo "[8/11] Testing 86_blackwell_mixed_dtype_gemm..."
./examples/86_blackwell_mixed_dtype_gemm/86_blackwell_mixed_dtype_gemm --m=4096 --n=4096 --k=4096 > /tmp/test_86.log 2>&1
if [ $? -eq 0 ]; then
    echo "  ✓ PASSED"
    tail -3 /tmp/test_86.log
else
    echo "  ✗ FAILED"
fi
echo ""

# Example 87a - FP8/BF16 GEMM Blockwise
echo "[9/11] Testing 87a_blackwell_geforce_fp8_bf16_gemm_blockwise..."
./examples/87_blackwell_geforce_gemm_blockwise/87a_blackwell_geforce_fp8_bf16_gemm_blockwise --m=4096 --n=4096 --k=4096 > /tmp/test_87a.log 2>&1
if [ $? -eq 0 ]; then
    echo "  ✓ PASSED"
    tail -3 /tmp/test_87a.log
else
    echo "  ✗ FAILED"
fi
echo ""

# Example 87b - FP8/BF16 GEMM Groupwise
echo "[10/11] Testing 87b_blackwell_geforce_fp8_bf16_gemm_groupwise..."
./examples/87_blackwell_geforce_gemm_blockwise/87b_blackwell_geforce_fp8_bf16_gemm_groupwise --m=4096 --n=4096 --k=4096 > /tmp/test_87b.log 2>&1
if [ $? -eq 0 ]; then
    echo "  ✓ PASSED"
    tail -3 /tmp/test_87b.log
else
    echo "  ✗ FAILED"
fi
echo ""

# Example 87c - FP8/BF16 Grouped GEMM Groupwise
echo "[11/11] Testing 87c_blackwell_geforce_fp8_bf16_grouped_gemm_groupwise..."
./examples/87_blackwell_geforce_gemm_blockwise/87c_blackwell_geforce_fp8_bf16_grouped_gemm_groupwise --groups=3 > /tmp/test_87c.log 2>&1
if [ $? -eq 0 ]; then
    echo "  ✓ PASSED"
    tail -3 /tmp/test_87c.log
else
    echo "  ✗ FAILED"
fi
echo ""

echo "========================================="
echo "All tests completed!"
echo "========================================="
