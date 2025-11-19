# nvJitLink Policy Framework (CUDA 12+)

## Overview

Modern rewrite of the dyn_link demo using CUDA 12's nvJitLink API with **device function pointers**. This demonstrates clean runtime policy injection by linking user kernels with policy code at runtime. **Works with any kernel and any number of arguments!**

## What This Demo Shows

1. **Device Function Pointers** - Pass policy functions from host to kernel
2. **Clean API** - Modern nvJitLink vs old cuLinkCreate
3. **PTX-based linking** - No CUBIN complexity
4. **Runtime policy injection** - Link user code + policy on-the-fly
5. **Generic Pattern** - Works with ANY kernel with ANY arguments
6. **Object-oriented framework** - Easy-to-use PolicyFramework class
7. **Better optimization** - Cross-module optimization possible

## Comparison with Old dyn_link

| Feature | Old (dyn_link) | New (6_nvjitlink_policy) |
|---------|----------------|--------------------------|
| **API** | cuLinkCreate/cuLinkAddData | nvJitLink API |
| **Input** | CUBIN files | PTX files |
| **Code style** | Procedural C | OOP C++ |
| **Symbol resolution** | Manual function pointers | Direct linking |
| **Optimization** | Limited | Cross-module possible |
| **Complexity** | High | Low |

## Building and Running

```bash
make          # Build demo and generate PTX
make run      # Build and run
make clean    # Clean artifacts
make help     # Show help
```

## Architecture

```
User Kernel (PTX)
    ↓
Policy Code (PTX)
    ↓
nvJitLink (runtime)
    ↓
Linked CUBIN
    ↓
Load & Execute
```

## File Structure

- **user_kernel.cu** - User's GEMM kernel (device function)
- **policy.cu** - Policy wrapper (calls user kernel + applies policy)
- **policy_framework.h** - PolicyFramework class (nvJitLink wrapper)
- **gemm_test.cu** - Main application
- **Makefile** - Build system

## Key Improvements

### 1. Clean API Usage

**Old way** (cuLinkCreate):
```cpp
CUlinkState linkState;
cuLinkCreate(0, nullptr, nullptr, &linkState);
cuLinkAddData(linkState, CU_JIT_INPUT_CUBIN, wrapper_data, ...);
cuLinkAddData(linkState, CU_JIT_INPUT_CUBIN, kernel_data, ...);
cuLinkAddData(linkState, CU_JIT_INPUT_CUBIN, policy_data, ...);
cuLinkComplete(linkState, &linked_cubin, &linked_size);
```

**New way** (nvJitLink):
```cpp
PolicyFramework framework;
framework.loadUserKernel("user_kernel.ptx");
framework.loadPolicy("policy.ptx");
framework.link(prop.major, prop.minor);
framework.launch(gridDim, blockDim, ...);
```

### 2. PTX-based (not CUBIN)

- **Portable** - PTX works across architectures
- **Optimizable** - Can apply LTO during linking
- **Simpler** - No architecture-specific CUBIN handling

### 3. Object-Oriented Design

The `PolicyFramework` class encapsulates:
- PTX loading
- nvJitLink operations
- Module management
- Kernel launching

### 4. Better Error Handling

```cpp
if (!framework.loadUserKernel("user_kernel.ptx")) {
    fprintf(stderr, "Failed to load user kernel\n");
    return EXIT_FAILURE;
}
```

## How It Works

### Step 1: Compile to PTX
```bash
nvcc -arch=sm_120 --relocatable-device-code=true -ptx user_kernel.cu
nvcc -arch=sm_120 --relocatable-device-code=true -ptx policy.cu
```

### Step 2: Load and Link at Runtime
```cpp
PolicyFramework framework;
framework.loadUserKernel("user_kernel.ptx");  // Load user's code
framework.loadPolicy("policy.ptx");            // Load policy
framework.link(12, 0);                         // Link for sm_120
```

### Step 3: Launch
```cpp
framework.launch(gridDim, blockDim, 0,
                d_A, d_B, d_C, M, N, K, alpha, beta);
```

## Policy Example

The demo applies an "upper triangle zero" policy:

```cpp
// policy.cu
extern "C" __global__ void gemm_with_policy(...) {
    // Call original user kernel
    gemm_kernel_impl(A, B, C, M, N, K, alpha, beta);

    // Apply policy: zero upper triangle
    __syncthreads();
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N && col > row) {
        C[row * N + col] = 0.0f;  // Zero it!
    }
}
```

At runtime, nvJitLink resolves the external reference to `gemm_kernel_impl` and can even inline it for better performance!

## Output Example

```
========================================
CUDA 12 nvJitLink Policy Framework Demo
========================================

Device: NVIDIA GeForce RTX 5090
Compute Capability: 12.0

=== Setting up Policy Framework ===
Loading user kernel: user_kernel.ptx
✓ Loaded user kernel PTX (2689 bytes)
Loading policy: policy.ptx
✓ Loaded policy PTX (1823 bytes)

=== Linking with nvJitLink ===
✓ Created nvJitLink handle
  Options: -arch=sm_120 -O3
✓ Added user kernel PTX
✓ Added policy PTX
Linking...
✓ Linking completed successfully!
✓ Generated linked CUBIN (13245 bytes)
✓ Loaded linked module
✓ Got wrapped kernel function

=== Launching Kernel ===
Grid: (32, 32), Block: (16, 16)
✓ Kernel executed successfully

=== Verification ===
✓ Results verified! Policy correctly applied.

=== Performance ===
Time: 2.156 ms
GFLOPS: 123.45

=== Sample Results ===
First 5x5 block (lower triangle should be non-zero, upper zero):
  1.2345   0.0000   0.0000   0.0000   0.0000
  2.3456   3.4567   0.0000   0.0000   0.0000
  4.5678   5.6789   6.7890   0.0000   0.0000
  7.8901   8.9012   9.0123   1.2345   0.0000
  2.3456   3.4567   4.5678   5.6789   6.7890
```

## Use Cases

1. **Multi-Tenant GPU Servers** - Apply different policies per tenant
2. **Runtime Optimization** - Specialize kernels based on workload
3. **Security Policies** - Enforce access control at kernel level
4. **Resource Management** - Apply quotas and rate limiting
5. **Auditing** - Inject logging/monitoring code

## Extension Ideas

### Multiple Policies
```cpp
framework.loadPolicy("rate_limit.ptx");
framework.loadPolicy("audit_log.ptx");
framework.loadPolicy("access_control.ptx");
```

### Context-Independent Loading
Combine with cuLibrary for multi-context scenarios:
```cpp
CUlibrary library;
cuLibraryLoadData(&library, linkedCubin.data(), ...);
// Use in multiple contexts!
```

### Policy Caching
```cpp
static std::unordered_map<std::string, CUmodule> policyCache;
```

## Requirements

- CUDA 12.0 or later
- nvJitLink library: `-lnvJitLink`
- Driver API: `-lcuda`
- Compute capability 5.0 or later

## Next Steps

See **Demo 7: Binary Extraction** to learn how to extract kernels from existing binaries and apply policies to them!

## Key Takeaways

1. nvJitLink is cleaner than cuLinkCreate
2. PTX-based linking is more flexible than CUBIN
3. OOP design makes framework easier to use
4. Runtime policy injection without source code changes
5. Foundation for advanced policy frameworks
