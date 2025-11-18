# CUDA 12 Features Demos Overview

Complete collection of CUDA 12+ feature demonstrations showing modern GPU programming techniques.

## Demo Summary

| Demo | Feature | Purpose | Status |
|------|---------|---------|--------|
| **1_dynamic_loading** | cudaLibrary* APIs | Dynamic kernel loading from fatbin | ✓ Working |
| **2_proc_address** | cuGetProcAddress | Function pointer lookup for hooking | ✓ Working |
| **3_nvjitlink** | nvJitLink | Runtime JIT linking with LTO | ✓ Working |
| **4_nvfatbin** | nvFatbin API | Build fatbin at runtime | ✓ Working |
| **5_context_independent** | cuLibrary* | Context-independent loading + caching | ✓ Working |
| **6_nvjitlink_policy** | Policy Framework | Modern policy injection framework | ✓ Working |
| **7_binary_extraction** | Binary Rewriting | Extract + inject into existing binaries | ✓ Working |

---

## Quick Start

```bash
# Test all demos
cd cuda12_features_demo
for demo in */; do
    cd "$demo"
    echo "Testing $demo..."
    make run
    cd ..
done
```

---

## Demos 1-5: Core CUDA 12 Features

### Demo 1: Dynamic Loading
**Location**: `1_dynamic_loading/`

Demonstrates CUDA 12's new runtime API for dynamic kernel loading.

**Key APIs**:
- `cudaLibraryLoadFromFile()` - Load fatbin at runtime
- `cudaLibraryGetKernel()` - Get kernel by name
- `cudaLaunchKernel()` - Launch with handle

**Use Cases**: Plugin systems, JIT compilation, dynamic code loading

---

### Demo 2: Proc Address
**Location**: `2_proc_address/`

Shows `cuGetProcAddress()` for building proxy libraries and intercepting CUDA calls.

**Key APIs**:
- `cuGetProcAddress()` - Get driver function pointers
- `cudaGetDriverEntryPoint()` - Runtime API version

**Use Cases**: Hooking, monitoring, proxy libraries, policy enforcement

---

### Demo 3: nvJitLink
**Location**: `3_nvjitlink/`

Runtime JIT linking of PTX modules with Link-Time Optimization.

**Key APIs**:
- `nvJitLinkCreate()` - Create linker
- `nvJitLinkAddData()` - Add PTX/CUBIN
- `nvJitLinkComplete()` - Link with LTO

**Use Cases**: Runtime policy injection, cross-module optimization

**Note**: LTO disabled for sm_120 compatibility in this demo.

---

### Demo 4: nvFatbin
**Location**: `4_nvfatbin/`

Build fat binary files at runtime by combining multiple architectures.

**Key APIs**:
- `nvFatbinCreate()` - Create builder
- `nvFatbinAddPTX/AddCubin()` - Add entries
- `nvFatbinGet()` - Get final fatbin

**Use Cases**: Binary rewriting, multi-arch packaging, code injection

---

### Demo 5: Context-Independent Loading
**Location**: `5_context_independent/`

Load modules once and share across multiple CUDA contexts.

**Key APIs**:
- `cuLibraryLoadData()` - Context-independent load
- Module caching with reference counting

**Use Cases**: Multi-tenant GPU servers, memory optimization

**Benefits**: 100× memory savings in multi-context scenarios!

---

## Demos 6-7: Advanced Applications

### Demo 6: nvJitLink Policy Framework ⭐
**Location**: `6_nvjitlink_policy/`

**Modern rewrite of dyn_link using CUDA 12 features.**

Complete policy injection framework using nvJitLink for runtime kernel wrapping.

**What's Different from Old dyn_link**:
- ✅ Clean nvJitLink API (vs old cuLinkCreate)
- ✅ PTX-based linking (vs CUBIN complexity)
- ✅ OOP design (PolicyFramework class)
- ✅ Better error handling
- ✅ Simpler to use

**Example**:
```cpp
PolicyFramework framework;
framework.loadUserKernel("user_kernel.ptx");
framework.loadPolicy("upper_triangle_zero.ptx");
framework.link(12, 0);  // sm_120
framework.launch(gridDim, blockDim, d_A, d_B, d_C, M, N, K, alpha, beta);
```

**Output**:
```
✓ Linking completed successfully!
✓ Generated linked CUBIN (18000 bytes)
✓ Results verified! Policy correctly applied.
Time: 0.075 ms, GFLOPS: 3575.71
```

---

### Demo 7: Binary Extraction + JIT ⭐
**Location**: `7_binary_extraction/`

**Extract kernels from existing binaries and inject policy code at runtime.**

Perfect for closed-source applications!

**Workflow**:
1. Extract kernels using `cuobjdump`
2. Load policy PTX
3. Execute wrapped versions
4. Verify policy enforcement

**Example**:
```bash
# Build target app
make sample_app

# Extract and rewrite
make run

# Output:
# ✓ Extracted 2 kernel file(s)
# ✓ Loaded policy PTX (8784 bytes)
# ✓ Upper triangle zeroed by policy!
```

**Key Insight**: Policy injection into pre-compiled binaries without source code!

---

## Architecture Compatibility

All demos support **sm_120** (Blackwell) by default. Modify `CUDA_ARCH` in Makefiles for other architectures:

```makefile
CUDA_ARCH = sm_80   # Ampere
CUDA_ARCH = sm_89   # Ada Lovelace
CUDA_ARCH = sm_90   # Hopper
CUDA_ARCH = sm_120  # Blackwell
```

---

## Technology Stack

| Layer | Technology | Demos Using |
|-------|-----------|-------------|
| **Dynamic Loading** | cudaLibrary* | 1, 7 |
| **Hooking** | cuGetProcAddress | 2 |
| **JIT Linking** | nvJitLink | 3, 6 |
| **Binary Building** | nvFatbin | 4 |
| **Caching** | cuLibrary* | 5, 6 |
| **Extraction** | cuobjdump | 7 |

---

## Common Patterns

### Pattern 1: Simple Dynamic Loading
```cpp
CUlibrary lib;
cudaLibraryLoadFromFile(&lib, "kernels.fatbin");
CUkernel kernel;
cudaLibraryGetKernel(&kernel, lib, "myKernel");
cudaLaunchKernel(kernel, grid, block, args, 0, 0);
```

### Pattern 2: Runtime Policy Injection (nvJitLink)
```cpp
nvJitLinkHandle handle;
nvJitLinkCreate(&handle, numOptions, options);
nvJitLinkAddData(handle, NVJITLINK_INPUT_PTX, userPTX, ...);
nvJitLinkAddData(handle, NVJITLINK_INPUT_PTX, policyPTX, ...);
nvJitLinkComplete(handle);
// Load and execute
```

### Pattern 3: Binary Extraction + Rewriting
```cpp
// Extract
system("cuobjdump -xelf all binary");

// Load policy
cudaLibraryLoadData(&lib, policyPTX, ...);

// Get wrapped kernel
cudaLibraryGetKernel(&kernel, lib, "wrapped_kernel");

// Execute
cudaLaunchKernel(kernel, ...);
```

### Pattern 4: Context-Independent Caching
```cpp
// Load once
cuLibraryLoadData(&lib, binary, ...);

// Use in multiple contexts
for (auto ctx : contexts) {
    cuCtxSetCurrent(ctx);
    cuLibraryGetKernel(&kernel, lib, "myKernel");
    // Execute in this context
}
// 100× memory savings!
```

---

## Performance Comparison

### nvJitLink vs Old cuLink
| Metric | Old (cuLink) | New (nvJitLink) |
|--------|-------------|-----------------|
| API complexity | High | Low |
| Input format | CUBIN | PTX (portable) |
| Optimization | Limited | LTO possible |
| Error handling | Poor | Good |
| Code size | More | Less |

### Context-Independent Loading
| Scenario | Per-Context | Shared |
|----------|-------------|--------|
| 1 context | 10 MB | 10 MB |
| 10 contexts | 100 MB | 10 MB |
| 100 contexts | 1000 MB | 10 MB |

**Savings**: 10-100× less memory!

---

## Integration Example: Complete Policy Framework

Combining all features:

```cpp
// 1. Hook kernel launches (Demo 2)
auto real_cuLaunchKernel = getCudaFunction("cuLaunchKernel");

// 2. Intercept
CUresult hooked_cuLaunchKernel(CUfunction f, ...) {
    // 3. Extract kernel if needed (Demo 7)
    auto kernelPTX = extractKernelPTX(f);

    // 4. Check cache (Demo 5)
    std::string cacheKey = getKernelHash(kernelPTX);
    if (!cache.has(cacheKey)) {
        // 5. JIT link with policy (Demo 6)
        PolicyFramework fw;
        fw.loadUserKernel(kernelPTX);
        fw.loadPolicy("my_policy.ptx");
        fw.link(major, minor);

        // 6. Cache result (Demo 5)
        cache.set(cacheKey, fw.getLibrary());
    }

    // 7. Launch wrapped version
    auto wrappedKernel = cache.get(cacheKey);
    return real_cuLaunchKernel(wrappedKernel, ...);
}
```

---

## Requirements

- CUDA 12.0 or later (12.4+ for nvFatbin)
- Compute capability 5.0 or later
- Libraries: `-lcuda -lnvJitLink -lnvfatbin`
- Tools: `cuobjdump` (included with CUDA)

---

## Building All Demos

```bash
#!/bin/bash
# Build all demos

demos=(
    "1_dynamic_loading"
    "2_proc_address"
    "3_nvjitlink"
    "4_nvfatbin"
    "5_context_independent"
    "6_nvjitlink_policy"
    "7_binary_extraction"
)

for demo in "${demos[@]}"; do
    echo "========================================="
    echo "Building $demo"
    echo "========================================="
    cd "$demo"
    make clean && make
    if [ $? -eq 0 ]; then
        echo "✓ $demo built successfully"
    else
        echo "✗ $demo build failed"
    fi
    cd ..
done
```

---

## Use Cases by Industry

### Multi-Tenant Cloud Providers
- **Demos 5, 6, 7**: Share GPU code across tenants, inject isolation policies

### Security Software
- **Demos 2, 7**: Monitor and audit GPU operations, inject security checks

### HPC Centers
- **Demos 3, 6**: Apply scheduling policies, optimize resource usage

### ML/AI Platforms
- **Demos 1, 4**: Dynamic model loading, multi-arch support

### Game Engines
- **Demos 1, 5**: Plugin systems, fast context switching

---

## Future Enhancements

1. **Combine Demos 6 + 7** → Complete binary rewriter
2. **Add Demo 8** → CUPTI integration for runtime profiling
3. **Add Demo 9** → Multi-GPU policy coordination
4. **Add Demo 10** → WebAssembly GPU backend

---

## Key Takeaways

1. ✅ CUDA 12 provides modern APIs for dynamic code management
2. ✅ nvJitLink is simpler and more powerful than old cuLink
3. ✅ Context-independent loading saves massive memory
4. ✅ Binary extraction enables policy injection without source
5. ✅ These features enable production-ready policy frameworks

---

## Learning Path

**Beginner** → Demo 1 (Dynamic Loading)
**Intermediate** → Demos 3, 6 (nvJitLink)
**Advanced** → Demos 5, 7 (Caching + Extraction)
**Expert** → Build complete framework combining all

---

## Support

For issues or questions:
- Check individual demo READMEs
- See CUDA documentation: https://docs.nvidia.com/cuda/
- Report issues with specific demo name and error message

---

**All demos tested on**: NVIDIA GeForce RTX 5090 (sm_120), CUDA 12.9

**Status**: ✓ All 7 demos working and verified!
