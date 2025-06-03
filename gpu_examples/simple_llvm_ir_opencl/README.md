# LLVM IR to OpenCL via SPIR-V

This is a minimal example showing how to run LLVM IR on a GPU via SPIR-V and OpenCL.

## Files

- `vector_add.ll` - LLVM IR for vector addition (the source code)
- `vector_add_llvm14.ll` - LLVM IR formatted for LLVM 14 compatibility
- `vector_add.cl` - Equivalent OpenCL C code (for reference only)
- `opencl_runner.c` - C program to execute SPIR-V on OpenCL
- `run_llvm_on_opencl.sh` - Script to convert LLVM IR to SPIR-V and run it

## How It Works

The workflow demonstrates the full chain:

1. LLVM IR → LLVM Bitcode → SPIR-V → OpenCL Execution

## LLVM Version Compatibility

This example includes a fallback mechanism to handle LLVM version incompatibilities. The issue is that LLVM IR format changes between versions, particularly with the introduction of opaque pointers in newer LLVM versions.

In this system:
- LLVM bitcode generator (llvm-as): Version 19.1.7
- SPIR-V translator (llvm-spirv): Version 14.0.6

This version mismatch causes errors like:
```
Opaque pointers are only supported in -opaque-pointers mode (Producer: 'LLVM19.1.7' Reader: 'LLVM 14.0.6')
```

The script tries multiple approaches:
1. First, it tries to use LLVM IR compatible with your system's LLVM version
2. If that fails, it falls back to using a pre-compiled SPIR-V binary

## Running the Example

```bash
chmod +x run_llvm_on_opencl.sh
./run_llvm_on_opencl.sh
```

## Achieving Complete LLVM IR to SPIR-V Conversion

To make the direct LLVM IR to SPIR-V conversion work without fallbacks, you would need:

1. Matching LLVM and SPIRV-LLVM-Translator versions
2. LLVM IR syntax that matches your LLVM version

The most reliable approaches would be:
- Build SPIRV-LLVM-Translator against your LLVM 19.1.7 version
- Use an older LLVM version (14.0.6) to generate the bitcode

For simplicity, this example uses the fallback to a pre-compiled SPIR-V binary when version incompatibilities arise.

## LLVM IR Details

The LLVM IR file demonstrates:

1. Use of address space qualifiers (`addrspace(1)`) to map to OpenCL global memory
2. OpenCL built-in function calls like `get_global_id()`
3. Proper SPIR target triple: `spir-unknown-unknown`

The script includes a fallback to a pre-compiled SPIR-V binary if the conversion fails, ensuring the example works even when LLVM tools have version incompatibilities. 