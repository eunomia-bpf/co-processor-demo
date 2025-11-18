# nvJitLink Demo (CUDA 12+)

## Overview

Demonstrates **nvJitLink** - CUDA's runtime JIT linker that allows linking multiple PTX/LTOIR modules at runtime with Link-Time Optimization (LTO).

## What This Demo Shows

1. **nvJitLinkCreate()** - Create a JIT linker handle
2. **nvJitLinkAddData()** - Add PTX/LTOIR modules to link
3. **nvJitLinkComplete()** - Perform the link with optimization
4. **Runtime Linking** - Combine separate modules at runtime
5. **Policy Injection** - Link user kernels with policy wrappers

## Key Features

- Link multiple PTX modules at runtime
- Link-Time Optimization (LTO) for cross-module inlining
- Combine user code with policy/wrapper code dynamically
- Generate optimized CUBIN from separate sources
- Perfect for runtime policy enforcement

## Building and Running

```bash
make          # Build demo and generate PTX files
make run      # Build and run
make ptx      # Generate PTX files only
make ltoir    # Generate LTO-IR files (alternative)
make clean    # Clean build artifacts
```

## How It Works

### Step 1: Compile to PTX with Relocatable Device Code
```bash
nvcc -arch=sm_120 --relocatable-device-code=true -ptx user_kernel.cu
nvcc -arch=sm_120 --relocatable-device-code=true -ptx policy.cu
```

### Step 2: Create Linker and Add Modules
```c
const char* options[] = {"-arch=sm_120", "-O3"};
nvJitLinkCreate(&handle, 2, options);

nvJitLinkAddData(handle, NVJITLINK_INPUT_PTX,
                 userPTX.data(), userPTX.size(), "user_kernel");
nvJitLinkAddData(handle, NVJITLINK_INPUT_PTX,
                 policyPTX.data(), policyPTX.size(), "policy");
```

### Step 3: Link and Get Result
```c
nvJitLinkComplete(handle);  // Performs linking + LTO

size_t cubinSize;
nvJitLinkGetLinkedCubinSize(handle, &cubinSize);
nvJitLinkGetLinkedCubin(handle, cubin.data());
```

### Step 4: Load and Execute
```c
cuModuleLoadData(&module, cubin.data());
cuModuleGetFunction(&kernel, module, "user_kernel_with_policy");
cuLaunchKernel(kernel, ...);
```

## Demo Architecture

The demo shows a realistic policy injection pattern:

**user_kernel.cu**:
```c
// User's original computation
extern "C" __device__ void user_kernel_impl(...) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    c[idx] = compute_element(a[idx], b[idx]);
}
```

**policy.cu**:
```c
// Policy wrapper that calls user's implementation
extern "C" __device__ void user_kernel_impl(...);  // External reference

extern "C" __global__ void user_kernel_with_policy(...) {
    if (blockIdx.x >= global_policy.max_blocks) {
        return;  // Policy enforcement
    }
    user_kernel_impl(...);  // Call original
}
```

At runtime, nvJitLink resolves the external reference and can inline the user's implementation into the policy wrapper!

## Use Cases

- **Runtime Policy Injection**: Wrap user kernels with policy code
- **Dynamic Code Generation**: Combine separately compiled modules
- **Multi-Tenant Systems**: Apply different policies to different tenants
- **Kernel Specialization**: Link generic code with specialized implementations
- **Plugin Systems**: Allow external modules to link with core functionality

## Output Example

```
=== Part 3: Add PTX inputs and link ===
✓ Added user_kernel.ptx
✓ Added policy.ptx

Linking with LTO...
✓ Linking completed successfully!
✓ Generated linked CUBIN (11776 bytes)

=== Part 4: Load and execute linked module ===
✓ Got kernel function: user_kernel_with_policy
Result: [0, 9, 36, 81, 144, 225, ...]
✓ Results verified!
```

## Related APIs

- `nvJitLinkCreate()` - Create linker handle with options
- `nvJitLinkAddData()` - Add PTX/LTOIR/CUBIN input
- `nvJitLinkAddFile()` - Add input from file
- `nvJitLinkComplete()` - Perform the link
- `nvJitLinkGetLinkedCubin()` - Get result CUBIN
- `nvJitLinkGetErrorLog()` - Get error messages
- `nvJitLinkDestroy()` - Clean up linker

## Linker Options

Common options passed to nvJitLinkCreate:
- `-arch=sm_XX` - Target architecture (required)
- `-O0/-O1/-O2/-O3` - Optimization level
- `-lto` - Enable Link-Time Optimization (disabled for sm_120 in this demo)
- `-g` - Generate debug info
- `-lineinfo` - Generate line info for profiling

## LTO Note

Link-Time Optimization (`-lto`) is disabled in this demo for sm_120 compatibility. For older architectures (sm_80 and below), you can enable it for better performance through cross-module inlining.

## Input Formats

nvJitLink accepts multiple input formats:
- **PTX** - Portable intermediate code (NVJITLINK_INPUT_PTX)
- **LTOIR** - LTO intermediate representation (NVJITLINK_INPUT_LTOIR)
- **CUBIN** - Binary code (NVJITLINK_INPUT_CUBIN)
- **FATBIN** - Fat binary (NVJITLINK_INPUT_FATBIN)

## Requirements

- CUDA 12.0 or later
- nvJitLink library: `-lnvJitLink`
- Driver API: `-lcuda`

## Integration with CLC Framework

Perfect for CLC (Compile-Load-Check) frameworks:

```
User Application Launch
         ↓
    [Intercept Hook]
         ↓
    Extract User Kernel → PTX
         ↓
    Compile Policy → PTX
         ↓
    nvJitLink (combine + optimize)
         ↓
    Load Wrapped Kernel
         ↓
    Execute with Policy
```

This allows dynamic policy injection without requiring source code access to user kernels!
