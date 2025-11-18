# Dynamic Loading Demo (CUDA 12+)

## Overview

Demonstrates CUDA 12's new **runtime API** for dynamic kernel loading without mixing runtime and driver APIs.

## What This Demo Shows

1. **cudaGetKernel()** - Get a handle from a statically compiled kernel
2. **cudaLibraryLoadFromFile()** - Load a fatbin file at runtime
3. **cudaLibraryGetKernel()** - Get kernels by name from the loaded library
4. **cudaLaunchKernel()** - Launch kernels using their handles
5. **Pure runtime API** - No need to mix runtime and driver code

## Key Features

- Load CUDA kernels dynamically at runtime from fatbin files
- Enumerate all kernels in a library
- Launch kernels by handle instead of template syntax
- Perfect foundation for plugin systems and dynamic code loading

## Building and Running

```bash
make          # Build the demo
make run      # Build and run
make clean    # Clean build artifacts
```

## How It Works

### Part 1: Static Kernel Launch
Traditional way of launching kernels using template syntax.

### Part 2: cudaGetKernel
Get a handle from a statically compiled kernel and launch it using `cudaLaunchKernel()`.

### Part 3: Dynamic Library Loading
1. Compile kernels to a fatbin: `nvcc --fatbin kernel.cu -o kernels.fatbin`
2. Load at runtime: `cudaLibraryLoadFromFile(&lib, "kernels.fatbin")`
3. Get kernels: `cudaLibraryGetKernel(&kernel, lib, "kernelName")`
4. Launch: `cudaLaunchKernel(kernel, ...)`

## Use Cases

- **Plugin Systems**: Load user-provided kernels at runtime
- **JIT Compilation**: Compile and load kernels on-the-fly
- **Policy Frameworks**: Dynamically load wrapper kernels with policies
- **Multi-Version Support**: Load different kernel versions based on hardware

## Output Example

```
=== Part 3: Dynamic Library Loading ===
✓ Library loaded successfully
✓ Kernel count in library: 2
✓ Got vectorAdd kernel
Result (dynamic): [0, 3, 6, 9, 12, 15, ...]
✓ Dynamic kernel executed successfully!
```

## Related APIs

- `cudaGetKernel()` - Get handle from static kernel
- `cudaLibraryLoadFromFile()` - Load fatbin file
- `cudaLibraryLoadData()` - Load fatbin from memory
- `cudaLibraryGetKernel()` - Get kernel by name
- `cudaLibraryEnumerateKernels()` - List all kernels
- `cudaLaunchKernel()` - Launch with handle
- `cudaLibraryUnload()` - Unload library

## Requirements

- CUDA 12.0 or later
- Compute capability 5.0 or later
