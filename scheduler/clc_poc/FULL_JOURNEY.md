# Complete Journey: Cluster Launch Control Testing

## Hardware & Software Environment

```
GPU: NVIDIA GeForce RTX 5090
Compute Capability: 12.0 (Blackwell)
Driver: 575.57.08
CUDA Versions Available: 12.8, 12.9, 13.0
```

## Timeline of Attempts

---

### Attempt 1: CUDA 12.8 with libcu++ Wrappers ❌

**Goal:** Try using `<cuda/ptx>` wrappers for CLC

**Command:**
```bash
nvcc -arch=sm_120 -o simple_clc simple_clc_example.cu
```

**Result:** FAILED
```
Error: namespace "cuda::ptx" has no member "clusterlaunchcontrol_try_cancel"
Error: namespace "cuda::ptx" has no member "fence_proxy_async_generic_sync_restrict"
```

**Conclusion:** CUDA 12.8 does NOT have libcu++ CLC wrappers

---

### Attempt 2: CUDA 12.8 with Inline PTX (Short Syntax) ❌

**Goal:** Use inline PTX with basic instruction syntax

**File:** `minimal_clc_ptx.cu`

**PTX Used:**
```ptx
clusterlaunchcontrol.try_cancel.b128 [%r21];
```

**Command:**
```bash
nvcc -arch=sm_120 -ptx minimal_clc_ptx.cu -o test.ptx  # SUCCESS
nvcc -arch=sm_120 -o minimal_clc_ptx minimal_clc_ptx.cu  # FAILED
```

**Result:** PTX Generation ✅, Assembly ❌
```
ptxas error: Unknown modifier '.try_cancel'
ptxas error: Not a name of any known instruction: 'clusterlaunchcontrol'
```

**Conclusion:**
- CUDA 12.8 compiler generates CLC PTX
- CUDA 12.8 ptxas **cannot assemble** CLC instructions

---

### Attempt 3: CUDA 13.0 ptxas Only ⚠️

**Goal:** Use CUDA 13.0's ptxas to assemble PTX from CUDA 12.8

**Commands:**
```bash
# Generate PTX with 12.8
nvcc -arch=sm_120 -ptx minimal_clc_ptx.cu -o /tmp/clc.ptx

# Assemble with CUDA 13.0 ptxas
/usr/local/cuda-13.0/bin/ptxas -arch=sm_120 /tmp/clc.ptx -o /tmp/clc.cubin
```

**Result:** Same error - CUDA 13.0 ptxas also doesn't recognize short syntax

---

### Attempt 4: CUDA 13.0 with Full PTX Syntax ✅ (Partial)

**Goal:** Use complete PTX syntax from CUDA 13.0 headers

**File:** `minimal_clc_ptx_v2.cu`

**Full PTX Syntax:**
```ptx
clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128 [addr], [smem_bar];
clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 P_OUT, B128_response;
clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 ret, B128_response;
```

**Commands:**
```bash
/usr/local/cuda-13.0/bin/nvcc -arch=sm_120 -o clc_v2 minimal_clc_ptx_v2.cu
```

**Result:** Binary created ✅ (974 KB)

**Problem:** Runtime error
```
Error: CUDA driver version is insufficient for CUDA runtime version
```

**Reason:** Driver 575.57.08 supports CUDA 12.9, but binary uses CUDA 13.0 runtime

---

### Attempt 5: Load CUBIN with Driver API ❌

**Goal:** Manually load cubin to bypass runtime version check

**File:** `cubin_loader.cu`, `cubin_loader_simple.c`

**Commands:**
```bash
# Generate cubin with CUDA 13.0
nvcc -arch=sm_120 -ptx minimal_clc_ptx_v2.cu -o /tmp/clc.ptx
/usr/local/cuda-13.0/bin/ptxas -arch=sm_120 /tmp/clc.ptx -o /tmp/clc.cubin

# Compile loader
gcc -o cubin_loader_simple cubin_loader_simple.c -lcuda

# Try to load
./cubin_loader_simple /tmp/clc.cubin
```

**Result:** Segmentation fault in `cuModuleLoadData()`

**Reason:** Driver 575.57.08 doesn't support loading CUDA 13.0 cubins with CLC instructions

---

### Attempt 6: CUDA 12.9 - Vector Add Test ✅

**Goal:** Verify CUDA 12.9 works on RTX 5090

**File:** `test_vec_add.cu`

**Commands:**
```bash
/usr/local/cuda-12.9/bin/nvcc -arch=sm_120 -o test_vec_add test_vec_add.cu
./test_vec_add
```

**Result:** SUCCESS ✅
```
[0] 0.000000 + 0.000000 = 0.000000 ✓
...
Result: ✅ PASSED
```

**Conclusion:** CUDA 12.9 works perfectly on RTX 5090 with driver 575.57.08

---

### Attempt 7: CUDA 12.9 with CLC API ✅ **FINAL SUCCESS**

**Goal:** Use CUDA 12.9's libcu++ CLC API

**File:** `minimal_clc_12.9.cu`

**API Used:**
```cpp
#include <cuda/ptx>

__shared__ uint4 clc_response;
__shared__ uint64_t barrier;

ptx::mbarrier_init(&barrier, 1);
ptx::clusterlaunchcontrol_try_cancel(&clc_response, &barrier);
bool canceled = ptx::clusterlaunchcontrol_query_cancel_is_canceled(clc_response);
int bx = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_x<int>(clc_response);
```

**Commands:**
```bash
/usr/local/cuda-12.9/bin/nvcc -arch=sm_120 -o minimal_clc_12.9_api minimal_clc_12.9.cu
./minimal_clc_12.9_api
```

**Result:** SUCCESS ✅
```
Device: NVIDIA GeForce RTX 5090
Compute Capability: 12.0

Launching CLC kernel (CUDA 12.9 API):
  Grid: 4 blocks
  Block: 256 threads

Kernel completed!

[0] 0.0 (expected 0.0) ✓
[1] 2.5 (expected 2.5) ✓
...
Result: ✅ PASSED
```

---

## Summary of Results

| CUDA Version | Method | PTX Gen | PTX Asm | Runtime | Result |
|--------------|--------|---------|---------|---------|--------|
| **12.8** | libcu++ API | N/A | N/A | N/A | ❌ No API |
| **12.8** | Inline PTX (short) | ✅ | ❌ | N/A | ❌ ptxas fail |
| **13.0** | Inline PTX (full) | ✅ | ✅ | ❌ | ⚠️ Driver mismatch |
| **13.0** | CUBIN loader | ✅ | ✅ | ❌ | ❌ Driver segfault |
| **12.9** | libcu++ API | ✅ | ✅ | ✅ | ✅ **SUCCESS** |

## Key Findings

### 1. PTX Syntax Evolution

**CUDA 12.8 generates:**
```ptx
clusterlaunchcontrol.try_cancel.b128 [addr];
```

**CUDA 13.0 requires:**
```ptx
clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128 [addr], [smem_bar];
```

### 2. Assembler Support

- **CUDA 12.8 ptxas:** Cannot assemble any CLC instructions
- **CUDA 12.9 ptxas:** Can assemble CLC instructions ✅
- **CUDA 13.0 ptxas:** Can assemble CLC instructions ✅

### 3. Runtime Compatibility

- **Driver 575.57.08** supports:
  - ✅ CUDA 12.9 runtime
  - ❌ CUDA 13.0 runtime
  - ❌ Loading CUDA 13.0 cubins

### 4. API Availability

- **CUDA 12.8:** No libcu++ CLC wrappers
- **CUDA 12.9:** Full libcu++ CLC wrappers ✅
- **CUDA 13.0:** Full libcu++ CLC wrappers ✅

## Working Solution

**Environment:**
- CUDA: 12.9
- Driver: 575.57.08
- GPU: RTX 5090 (CC 12.0)

**Code Pattern:**
```cpp
#include <cuda/ptx>

__shared__ uint4 clc_response;
__shared__ uint64_t barrier;

// Initialize
ptx::mbarrier_init(&barrier, 1);

// Request cancellation
ptx::clusterlaunchcontrol_try_cancel(&clc_response, &barrier);

// Query result
bool canceled = ptx::clusterlaunchcontrol_query_cancel_is_canceled(clc_response);
int new_block = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_x<int>(clc_response);
```

## Files Created

### Working Examples
- ✅ `minimal_clc_12.9.cu` - CUDA 12.9 CLC implementation
- ✅ `test_vec_add.cu` - Simple vector add for testing

### Failed Attempts (Educational)
- ❌ `simple_clc_example.cu` - CUDA 12.8 libcu++ attempt
- ❌ `minimal_clc_ptx.cu` - Inline PTX short syntax
- ⚠️ `minimal_clc_ptx_v2.cu` - Inline PTX full syntax (CUDA 13.0)
- ❌ `cubin_loader.cu` - Driver API loader
- ❌ `cubin_loader_simple.c` - Simple C loader

### Documentation
- `FULL_JOURNEY.md` - This file
- `STATUS.md` - Compilation status
- `CUDA_12.8_PTX_TEST.md` - PTX generation test
- `CUDA_13_SUCCESS.md` - CUDA 13.0 findings
- `NOTES.md` - API patterns and usage
- `README.md` - Overview

## Lessons Learned

1. **PTX ISA specification ≠ ptxas implementation**
   - PTX 8.7 defines CLC, but CUDA 12.8 ptxas can't assemble it

2. **Driver version matters for CLC**
   - Need driver 575+ for CUDA 12.9 CLC support
   - Need driver 580+ for CUDA 13.0 support

3. **libcu++ is the easiest path**
   - Much simpler than inline PTX
   - Handles all the complex syntax

4. **CUDA 12.9 is the sweet spot**
   - Has CLC API support
   - Works with current drivers (575+)
   - Runs on RTX 5090

## Recommendations

### For Development
- Use **CUDA 12.9** with libcu++ API
- Target sm_120 for Blackwell
- Use `--no-device-link` flag to avoid link.stub issues

### For Testing
- Small grid sizes (4-8 blocks) for initial testing
- Check compute capability at runtime
- Use timeouts for debugging hanging kernels

### For Production
- Update to driver 580+ when available
- Consider CUDA 13.0 for latest features
- Profile work-stealing effectiveness

## Next Steps

1. ✅ Test CLC with larger workloads
2. ⏳ Measure work-stealing performance
3. ⏳ Compare vs fixed-blocks approach
4. ⏳ Test with thread block clusters
5. ⏳ Update driver to 580+ for CUDA 13.0

---

**Date:** 2025-10-18
**Status:** ✅ CLC Working on RTX 5090 with CUDA 12.9
