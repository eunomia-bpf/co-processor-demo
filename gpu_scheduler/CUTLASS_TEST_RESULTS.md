# CUTLASS Blackwell Examples Test Results

**Hardware:** NVIDIA GeForce RTX 5090
**Compute Capability:** 12.0 (SM120)
**CUDA Version:** 12.8.93
**CUTLASS Version:** 4.2.1
**Date:** 2025-10-17

---

## Summary

**Total Examples Tested:** 11
**Passed:** 10
**Failed:** 1 (Expected - SM100-only feature)

---

## Detailed Results

### ✅ Example 79a: NVFP4/BF16 GEMM
- **Status:** PASSED
- **Problem Size:** 4096x4096x4096
- **Runtime:** 0.112 ms
- **Performance:** 1.22 TFLOPS

### ✅ Example 79b: NVFP4/NVFP4 GEMM
- **Status:** PASSED
- **Problem Size:** 4096x4096x4096
- **Runtime:** 0.113 ms
- **Performance:** 1.22 TFLOPS

### ✅ Example 79c: Mixed MXFP8/MXFP6/BF16 GEMM
- **Status:** PASSED
- **Problem Size:** 4096x4096x4096
- **Runtime:** 0.204 ms
- **Performance:** 675 GFLOPS

### ✅ Example 79d: NVFP4 Grouped GEMM
- **Status:** PASSED
- **Groups:** 3
- **Runtime:** 0.019 ms
- **Performance:** 320.9 TFLOPS

### ✅ Example 80a: MXFP8/BF16 Sparse GEMM
- **Status:** PASSED
- **Problem Size:** 4096x4096x4096
- **Runtime:** 0.163 ms
- **Performance:** 840.7 GFLOPS

### ✅ Example 80b: NVFP4/NVFP4 Sparse GEMM
- **Status:** PASSED
- **Problem Size:** 4096x4096x4096
- **Runtime:** 0.104 ms
- **Performance:** 1.33 TFLOPS

### ✅ Example 86: Mixed Dtype GEMM
- **Status:** PASSED
- **Problem Size:** 4096x4096x4096

### ✅ Example 87a: FP8/BF16 GEMM Blockwise
- **Status:** PASSED
- **Problem Size:** 4096x4096x4096
- **Runtime:** 0.346 ms
- **Performance:** 397.8 GFLOPS

### ✅ Example 87b: FP8/BF16 GEMM Groupwise
- **Status:** PASSED
- **Problem Size:** 4096x4096x4096
- **Runtime:** 0.321 ms
- **Performance:** 428.7 GFLOPS

### ✅ Example 87c: FP8/BF16 Grouped GEMM Groupwise
- **Status:** PASSED
- **Problem Size:** 1024x512x1024 (3 groups)
- **Runtime:** 0.019 ms
- **Performance:** 170.2 GFLOPS

### ❌ Example 73: Preferred Cluster (Cluster Launch Control)
- **Status:** FAILED (Expected)
- **Reason:** SM100-specific feature not compatible with SM120
- **Error:** CUTLASS Internal Error / Invalid Status
- **Notes:** This example uses `KernelTmaWarpSpecialized2SmSm100` which is specific to datacenter Blackwell GPUs (B100/B200). GeForce Blackwell (RTX 50-series) uses SM120 architecture which has different kernel implementations.

---

## Key Findings

### Performance Highlights
- **Best Performance:** 1.33 TFLOPS (NVFP4/NVFP4 Sparse GEMM)
- **Standard GEMM:** ~1.22 TFLOPS (FP4 precision)
- **Mixed Precision:** 675 GFLOPS (MXFP8/MXFP6)
- **FP8 Operations:** 398-429 GFLOPS

### Architecture Notes

1. **SM100 vs SM120:**
   - SM100 = Datacenter Blackwell (B100, B200)
   - SM120 = GeForce Blackwell (RTX 5090, 5080, etc.)
   - Some advanced features like Cluster Launch Control are SM100-only

2. **Supported Features on SM120:**
   - ✅ NVFP4 (4-bit floating point) operations
   - ✅ MXFP8/MXFP6 mixed precision
   - ✅ FP8 operations
   - ✅ Sparse GEMM (2:4 structured sparsity)
   - ✅ Grouped GEMM
   - ✅ Blockwise/Groupwise scaling
   - ❌ Cluster Launch Control (SM100 only)

3. **Build Configuration:**
   - CMake flag: `-DCUTLASS_NVCC_ARCHS=120a`
   - Required CUDA: 12.8+
   - Modified CMakeLists to include SM120 support where applicable

---

## Modifications Made

1. **Example 73 (blackwell_gemm_preferred_cluster.cu):**
   - Updated compute capability check to accept SM120
   - Updated preprocessor guards to include `CUTLASS_ARCH_MMA_SM120_SUPPORTED`
   - Updated CMakeLists.txt to build for SM120
   - **Result:** Builds successfully but fails at runtime (SM100-specific kernels)

2. **Build System:**
   - Configured CUTLASS for SM120 architecture
   - All GeForce-specific examples (79, 80, 86, 87) work perfectly

---

## Recommendations

1. **For SM120 Development:** Use examples 79, 80, 86, 87 as reference
2. **For Cluster Launch Control:** Requires SM100 datacenter GPU
3. **Best Practices:** Always check example documentation for compute capability requirements

---

## References

- CUTLASS Documentation: https://docs.nvidia.com/cutlass/
- Blackwell Cluster Launch Control: https://docs.nvidia.com/cutlass/media/docs/cpp/blackwell_cluster_launch_control.html
- Example Code: `gpu_scheduler/cutlass/examples/`
