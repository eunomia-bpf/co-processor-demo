# Binary Extraction + JIT Rewriting Demo

## Overview

Demonstrates extracting CUDA kernels from compiled binaries and injecting policy code at runtime - perfect for closed-source applications and binary rewriting scenarios.

## What This Demo Shows

1. **Binary Extraction** - Extract embedded CUDA kernels using cuobjdump
2. **Symbol Discovery** - Find kernel names and signatures
3. **Policy Injection** - Add new policy code to extracted kernels
4. **JIT Rewriting** - Rebuild binaries with injected policies
5. **Runtime Loading** - Execute modified kernels

## Real-World Use Case

**Scenario**: You have a closed-source CUDA application and want to:
- Add rate limiting policies
- Inject monitoring/auditing code
- Apply security restrictions
- Enforce resource quotas

**This demo shows how to do it without source code!**

## Building and Running

```bash
make              # Build all components
make run          # Run complete demo
make extract      # Just extract and inspect
make clean        # Clean artifacts
make help         # Show help
```

## Architecture

```
Original Binary
    ↓
cuobjdump (extract CUBIN)
    ↓
Extracted Kernels
    ↓
Policy Code (PTX)
    ↓
cudaLibraryLoad
    ↓
Execute with Policy
```

## File Structure

- **sample_app.cu** - Original application (target for extraction)
- **policy.cu** - Policy code to inject
- **extractor.h** - BinaryExtractor and JITRewriter classes
- **extract_and_rewrite.cu** - Main extraction and rewriting tool
- **Makefile** - Build system

## How It Works

### Step 1: Build Target Binary

```bash
nvcc -arch=sm_120 sample_app.cu -o sample_app
```

This creates a binary with embedded fatbin containing kernels.

### Step 2: Extract Kernels

```bash
# List embedded ELF files
cuobjdump -lelf sample_app

# Extract all CUBIN files
cuobjdump -xelf all sample_app

# List symbols
cuobjdump -symbols sample_app
```

Output:
```
ELF file 1: sample_app.1.sm_120.cubin
Symbols:
  vectorAdd
  gemm_kernel
```

### Step 3: Load Policy and Rewrite

```cpp
JITRewriter rewriter;
rewriter.loadPolicy("policy.ptx");
rewriter.linkAndLoad(12, 0);
```

### Step 4: Execute Wrapped Kernels

```cpp
CUkernel wrappedKernel;
rewriter.getKernel(&wrappedKernel, "vectorAdd_with_policy");
cuLaunchKernel(wrappedKernel, ...);
```

## BinaryExtractor Class

Handles extraction of kernels from binaries:

```cpp
class BinaryExtractor {
public:
    BinaryExtractor(const char* path);

    // Extract all CUBIN files from binary
    bool extractAllCubins();

    // Get list of extracted files
    const std::vector<std::string>& getExtractedFiles();

    // List symbols in extracted CUBIN
    void listSymbols(const char* cubinFile);
};
```

## JITRewriter Class

Handles policy injection and rewriting:

```cpp
class JITRewriter {
public:
    // Load extracted kernel
    bool loadExtractedCubin(const char* cubinFile);

    // Load policy code
    bool loadPolicy(const char* policyFile);

    // Link and load as library
    bool linkAndLoad(int major, int minor);

    // Get wrapped kernel
    bool getKernel(CUkernel* kernel, const char* name);
};
```

## Example Policies

### Rate Limiting Policy
```cpp
__device__ int g_max_blocks = 1024;

extern "C" __global__ void kernel_with_policy(...) {
    if (blockIdx.x >= g_max_blocks) {
        return;  // Rate limit exceeded
    }
    // Original kernel code...
}
```

### Upper Triangle Zero Policy
```cpp
extern "C" __global__ void gemm_with_policy(...) {
    // Original GEMM computation
    C[row * N + col] = alpha * sum + beta * C[row * N + col];

    // Policy: Zero upper triangle
    if (col > row) {
        C[row * N + col] = 0.0f;
    }
}
```

## Output Example

```
========================================
Binary Extraction + JIT Rewriting Demo
========================================
Device: NVIDIA GeForce RTX 5090
Compute Capability: 12.0

=== Extracting Kernels from Binary ===
Binary: ./sample_app
Found: sample_app.1.sm_120.cubin
Extracting: sample_app.1.sm_120.cubin
✓ Extracted: sample_app.1.sm_120.cubin
✓ Extracted 1 kernel file(s)

=== Analyzing Extracted Files ===
Symbols in sample_app.1.sm_120.cubin:
  STT_FUNC  vectorAdd
  STT_FUNC  gemm_kernel

=== Loading Policy ===
File: policy.ptx
✓ Loaded policy PTX (3421 bytes)

=== Linking with nvJitLink ===
Loading policy as library...
✓ Loaded policy library

=== Testing Rewritten Kernels ===

--- Vector Add with Policy ---
✓ Got kernel: vectorAdd_with_policy
[POLICY] vectorAdd executed with rate limit: 1024 blocks
Result[0] = 0.0 (expected 0.0)
Result[10] = 30.0 (expected 30.0)
✓ Vector add with policy executed

--- GEMM with Policy ---
✓ Got kernel: gemm_with_policy
[POLICY] GEMM executed with upper-triangle-zero policy
First 5x5 block:
 256.0    0.0    0.0    0.0    0.0
 256.0  256.0    0.0    0.0    0.0
 256.0  256.0  256.0    0.0    0.0
 256.0  256.0  256.0  256.0    0.0
 256.0  256.0  256.0  256.0  256.0
✓ GEMM with policy executed correctly
✓ Upper triangle zeroed by policy!

========================================
Summary:
1. Extracted kernels from binary using cuobjdump
2. Loaded policy code (PTX)
3. Executed wrapped versions with policy
4. Verified policy enforcement

Key takeaway: We injected policy into a
pre-compiled binary without source code!
========================================
```

## Use Cases

### 1. Multi-Tenant GPU Server
```
User uploads binary
    ↓
Extract all kernels
    ↓
Inject tenant isolation policy
    ↓
Run with guaranteed isolation
```

### 2. Security Auditing
```
Vendor provides closed-source app
    ↓
Extract kernels
    ↓
Inject audit logging
    ↓
Monitor all GPU operations
```

### 3. Resource Management
```
Production application
    ↓
Extract hot kernels
    ↓
Inject rate limiting/quotas
    ↓
Enforce resource policies
```

### 4. Dynamic Optimization
```
Running application
    ↓
Extract underperforming kernels
    ↓
JIT recompile with better flags
    ↓
Hot-swap optimized version
```

## cuobjdump Commands Reference

### List Contents
```bash
# List ELF files
cuobjdump -lelf binary

# List PTX files (if embedded)
cuobjdump -lptx binary

# List all sections
cuobjdump --all-fatbin binary
```

### Extract
```bash
# Extract specific ELF
cuobjdump -xelf "name.cubin" binary

# Extract all ELFs
cuobjdump -xelf all binary

# Extract PTX (if available)
cuobjdump -xptx all binary
```

### Inspect
```bash
# Show symbols
cuobjdump -symbols binary

# Show SASS disassembly
cuobjdump -sass binary

# Show PTX
cuobjdump -ptx binary

# Show resource usage
cuobjdump -res-usage binary
```

## Advanced: Manual ELF Parsing

For more control, you can parse the `.nv_fatbin` ELF section directly:

```cpp
// Open binary as ELF
int fd = open("app_binary", O_RDONLY);

// Parse ELF header
Elf64_Ehdr ehdr;
read(fd, &ehdr, sizeof(ehdr));

// Find .nv_fatbin section
// Extract fatbin data
// Use nvFatbin API to parse
```

This gives you full control but requires more code.

## Limitations

1. **Kernel Signatures** - Must know or discover function signatures
2. **Symbol Names** - Need to find correct symbol names
3. **Architecture** - Extracted CUBIN is architecture-specific
4. **PTX Availability** - Not all binaries embed PTX
5. **Obfuscation** - Won't work on obfuscated binaries

## Workarounds

- **PTX Decompilation** - Use `cuobjdump -ptx` to decompile CUBIN to PTX
- **Runtime Interception** - Hook `cudaLaunchKernel` to intercept at runtime
- **Symbol Discovery** - Use `cuobjdump -symbols` to enumerate

## Integration with Other CUDA 12 Features

### With nvJitLink (Demo 3)
```cpp
// Extract CUBIN
extractor.extractAllCubins();

// Link with policy using nvJitLink
nvJitLinkAddData(handle, NVJITLINK_INPUT_CUBIN, extracted, ...);
nvJitLinkAddData(handle, NVJITLINK_INPUT_PTX, policy, ...);
nvJitLinkComplete(handle);
```

### With Context-Independent Loading (Demo 5)
```cpp
// Extract and rewrite once
auto wrappedBinary = rewriter.linkAndLoad(...);

// Load as context-independent library
CUlibrary lib;
cuLibraryLoadData(&lib, wrappedBinary, ...);

// Use in multiple contexts!
```

### With Module Caching
```cpp
// Cache wrapped versions
std::unordered_map<std::string, CUlibrary> cache;

auto key = getBinaryHash(originalBinary);
if (cache.find(key) == cache.end()) {
    // Extract, wrap, cache
    cache[key] = rewriteAndLoad(originalBinary);
}

// Reuse cached version
useKernel(cache[key]);
```

## Requirements

- CUDA 12.0 or later
- cuobjdump tool (included with CUDA)
- nvJitLink library: `-lnvJitLink`
- Driver API: `-lcuda`

## Next Steps

- Combine with **Demo 6** for complete policy framework
- Add ELF parsing for more control
- Implement runtime interception hooks
- Build production-ready binary rewriter

## Key Takeaways

1. cuobjdump extracts kernels from any CUDA binary
2. Can inject policies without source code
3. Perfect for closed-source applications
4. Foundation for dynamic policy enforcement
5. Works with existing compiled applications
