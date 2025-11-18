# nvFatbin Demo (CUDA 12.4+)

## Overview

Demonstrates **nvFatbin** API - build fat binary (fatbin) files at runtime by combining multiple PTX, CUBIN, and LTOIR modules for different architectures.

## What This Demo Shows

1. **nvFatbinCreate()** - Create a fatbin builder
2. **nvFatbinAddPTX()** - Add PTX code for an architecture
3. **nvFatbinAddCubin()** - Add compiled CUBIN code
4. **nvFatbinGet()** - Get the final fatbin binary
5. **cudaLibraryLoadData()** - Load the generated fatbin

## Key Features

- Build fatbin files at runtime without nvcc
- Combine multiple code formats (PTX, CUBIN, LTOIR)
- Support multiple architectures in one fatbin
- Compress fatbin contents automatically
- Perfect for binary rewriting and code injection

## Building and Running

```bash
make          # Build demo and kernel files
make kernels  # Build PTX and CUBIN files only
make run      # Build and run
make clean    # Clean build artifacts
```

## How It Works

### Step 1: Prepare Kernel Code
```bash
# Compile to PTX (portable)
nvcc -arch=sm_120 -ptx kernel1.cu -o kernel1.ptx
nvcc -arch=sm_120 -ptx kernel2.cu -o kernel2.ptx

# Compile to CUBIN (architecture-specific)
nvcc -arch=sm_120 -cubin kernel1.cu -o kernel1.cubin
nvcc -arch=sm_120 -cubin kernel2.cu -o kernel2.cubin
```

### Step 2: Create Fatbin Builder
```c
nvFatbinHandle handle;
const char* options[] = {
    "-64",              // 64-bit
    "-compress=true",   // Compress contents
    "-host=linux"       // Host platform
};
nvFatbinCreate(&handle, 3, options);
```

### Step 3: Add Entries
```c
// Add PTX for architecture 120
nvFatbinAddPTX(handle, ptx_data, ptx_size, 120, NULL);

// Add CUBIN for architecture 120
nvFatbinAddCubin(handle, cubin_data, cubin_size, 120, NULL);

// Can add multiple architectures
nvFatbinAddPTX(handle, ptx_data, ptx_size, 80, NULL);  // sm_80
nvFatbinAddPTX(handle, ptx_data, ptx_size, 90, NULL);  // sm_90
```

### Step 4: Finalize and Get Result
```c
// Finalize the fatbin
size_t fatbinSize;
nvFatbinSize(handle, &fatbinSize);

// Get the binary data
std::vector<char> fatbin(fatbinSize);
nvFatbinGet(handle, fatbin.data());

// Optionally save to file
FILE* f = fopen("output.fatbin", "wb");
fwrite(fatbin.data(), 1, fatbinSize, f);
fclose(f);

nvFatbinDestroy(handle);
```

### Step 5: Load and Use
```c
CUlibrary library;
cudaLibraryLoadData(&library, fatbin.data());

CUkernel kernel;
cudaLibraryGetKernel(&kernel, library, "myKernel");
cudaLaunchKernel(kernel, ...);
```

## Use Cases

- **Binary Rewriting**: Modify existing CUDA binaries
- **Code Injection**: Add wrapper/policy kernels to binaries
- **Multi-Arch Support**: Package code for multiple GPU generations
- **Runtime Code Generation**: Build complete binaries from generated code
- **CLC Frameworks**: Package user kernels + policy wrappers

## Output Example

```
=== Part 3: Add entries to fatbin ===
✓ Added kernel1.ptx (arch=120)
✓ Added kernel2.ptx (arch=120)
✓ Added kernel1.cubin (arch=120)
✓ Added kernel2.cubin (arch=120)

=== Part 4: Finalize and get fatbin ===
✓ Fatbin size: 21800 bytes
✓ Got fatbin data
✓ Saved to runtime_generated.fatbin
```

## Related APIs

- `nvFatbinCreate()` - Create fatbin builder
- `nvFatbinAddPTX()` - Add PTX code
- `nvFatbinAddCubin()` - Add CUBIN code
- `nvFatbinAddLTOIR()` - Add LTO intermediate representation
- `nvFatbinSize()` - Get final size
- `nvFatbinGet()` - Get fatbin data
- `nvFatbinDestroy()` - Clean up builder

## Fatbin Options

Common options for nvFatbinCreate:
- `-64` or `-32` - Address size
- `-compress=true/false` - Enable compression
- `-host=linux/windows/mac` - Host platform
- `-cuda` - CUDA mode (vs OpenCL)

## Supported Input Formats

- **PTX** - Portable assembly (via nvFatbinAddPTX)
- **CUBIN** - Native binary code (via nvFatbinAddCubin)
- **LTOIR** - Link-time optimization IR (via nvFatbinAddLTOIR)

## Architecture Codes

Specify compute capability as integer:
- `70` - Volta (sm_70)
- `75` - Turing (sm_75)
- `80` - Ampere (sm_80)
- `89` - Ada Lovelace (sm_89)
- `90` - Hopper (sm_90)
- `120` - Blackwell (sm_120)

## Multi-Architecture Example

```c
nvFatbinHandle handle;
nvFatbinCreate(&handle, ...);

// Add for multiple architectures
nvFatbinAddPTX(handle, ptx, size, 80, NULL);   // Ampere
nvFatbinAddPTX(handle, ptx, size, 89, NULL);   // Ada
nvFatbinAddPTX(handle, ptx, size, 90, NULL);   // Hopper
nvFatbinAddPTX(handle, ptx, size, 120, NULL);  // Blackwell

// CUDA will select the best match at load time
```

## Requirements

- CUDA 12.4 or later
- nvFatbin library: `-lnvfatbin`
- May show version warning (benign): `no version information available`

## Integration with CLC Framework

Perfect for building CLC (Compile-Load-Check) wrapper binaries:

```
1. Extract user kernel from original binary
   ↓
2. Compile policy wrapper to CUBIN
   ↓
3. Use nvFatbin to package together:
   - User kernel (PTX)
   - Policy wrapper (CUBIN)
   - Multiple architectures
   ↓
4. Load combined fatbin
   ↓
5. Execute policy-wrapped kernel
```

## Benefits Over Manual Fatbin Creation

1. **No nvcc dependency** - Build fatbins at runtime
2. **Dynamic architecture selection** - Add architectures based on runtime detection
3. **Simplified workflow** - No need to understand fatbin format
4. **Compression** - Automatic compression support
5. **Official API** - Supported and maintained by NVIDIA

## Binary Rewriting Pipeline

This API enables powerful binary rewriting:

```
Original App Binary
       ↓
Extract Kernels (via cubin tools)
       ↓
Add/Modify Kernels (your code)
       ↓
nvFatbin Rebuild
       ↓
Replace Original Binary
       ↓
Run Modified App
```

No source code required!
