# cuGetProcAddress Demo (CUDA 11.3+)

## Overview

Demonstrates **cuGetProcAddress()** - the official CUDA API for getting driver function pointers. This is the foundation for building proxy libraries and intercepting CUDA calls.

## What This Demo Shows

1. **cuGetProcAddress()** - Get driver function pointers at runtime
2. **cudaGetDriverEntryPoint()** - Runtime API version for getting driver functions
3. **Hook/Interception Pattern** - How to intercept CUDA driver calls
4. **Building Proxy libcuda.so** - Pattern for creating a complete proxy library

## Key Features

- Get driver function pointers without dlsym() hacks
- Official API that handles symbol versioning automatically
- Foundation for building interception/proxy layers
- No need for LD_PRELOAD tricks
- Perfect for policy enforcement and monitoring

## Building and Running

```bash
make          # Build the demo
make run      # Build and run
make clean    # Clean build artifacts
```

## How It Works

### Part 1: Basic cuGetProcAddress Usage
```c
void* funcPtr;
cuGetProcAddress("cuDeviceGet", &funcPtr, driverVersion, 0, &result);
auto realFunc = (PFN_cuDeviceGet)funcPtr;
```

### Part 2: Hook/Interception Pattern
```c
static PFN_cuDeviceGetAttribute real_func = nullptr;

void init_hook() {
    void* ptr;
    cuGetProcAddress("cuDeviceGetAttribute", &ptr, ...);
    real_func = (PFN_cuDeviceGetAttribute)ptr;
}

CUresult hooked_cuDeviceGetAttribute(...) {
    printf("[HOOK] Intercepted call!\n");
    // Apply policy/monitoring logic here
    return real_func(...);  // Call real function
}
```

### Part 3: Building a Proxy Library
The demo includes a complete template for building a proxy `libcuda.so` that:
1. Loads the real libcuda.so
2. Uses cuGetProcAddress to get real function pointers
3. Wraps functions with custom logic (policy, monitoring, etc.)
4. Forwards calls to the real driver

## Use Cases

- **Policy Enforcement**: Inject CLC/policy wrappers around kernel launches
- **Monitoring**: Track all CUDA API calls for debugging/profiling
- **Resource Management**: Control GPU resource allocation
- **Multi-Tenancy**: Enforce quotas and scheduling policies
- **Security**: Audit and restrict CUDA operations

## Building a Proxy libcuda.so

The demo shows the complete pattern:

```c
// 1. Load real libcuda.so
real_libcuda = dlopen("/usr/lib/x86_64-linux-gnu/libcuda.so.1", RTLD_LAZY);

// 2. Use cuGetProcAddress to get real functions
cuGetProcAddress("cuLaunchKernel", &funcPtr, driverVersion, 0, &result);
real_cuLaunchKernel = (PFN_cuLaunchKernel)funcPtr;

// 3. Export your hooked version
extern "C" CUresult cuLaunchKernel(...) {
    // YOUR POLICY LOGIC HERE
    return real_cuLaunchKernel(...);
}
```

Then use: `LD_LIBRARY_PATH=. ./your_cuda_app`

## Output Example

```
=== Part 2: Demonstrating Hook/Interception Pattern ===
✓ Hook initialized: cuDeviceGetAttribute
  [HOOK] Intercepted cuDeviceGetAttribute call
  [HOOK]   Attribute: 75, Device: 0
  [HOOK]   Result: 12
  Compute Capability: 12.0
```

## Related APIs

- `cuGetProcAddress()` - Get driver function pointer
- `cuGetProcAddress_v2()` - Extended version with flags
- `cudaGetDriverEntryPoint()` - Runtime API version
- `cuDriverGetVersion()` - Get driver version (needed for cuGetProcAddress)

## Advantages Over dlsym()

1. **Official API** - Supported and maintained by NVIDIA
2. **Version Handling** - Automatically handles symbol versioning
3. **Forward Compatible** - Works with future CUDA versions
4. **No Symbol Mangling** - Direct function pointer lookup
5. **Per-Version Symbols** - Can get different versions of same function

## Requirements

- CUDA 11.3 or later (for cuGetProcAddress)
- CUDA 12.0 or later (for cudaGetDriverEntryPoint)
- Driver API linking: `-lcuda`

## Integration with CLC Framework

This API is essential for building a CLC (Compile-Load-Check) framework:

1. Use cuGetProcAddress to hook `cuLaunchKernel`
2. Intercept kernel launches
3. Wrap user kernels with policy code at runtime
4. Apply scheduling/resource policies
5. Forward to real driver with wrapped kernel
