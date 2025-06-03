# Minimal OpenCL Example

This is a bare-bones example showing how to run OpenCL code on a GPU. While the original goal was to run LLVM IR directly, due to LLVM version compatibility issues, we're now using the direct OpenCL C path.

## Files

- `vector_add.cl` - OpenCL C code for vector addition
- `vector_add.ll` - Equivalent LLVM IR (for reference only)
- `opencl_runner.c` - C program to execute SPIR-V on OpenCL
- `run_llvm_on_opencl.sh` - Single script to run the complete example

## How It Works

The simplified workflow is:

1. OpenCL C → SPIR-V → GPU Execution

The script runs these steps sequentially:

```bash
# Convert OpenCL C to SPIR-V using clang (or use pre-existing SPIR-V)
clang -cc1 -emit-spirv -cl-std=CL2.0 -o vector_add.spv vector_add.cl

# Compile the OpenCL runner
gcc -o opencl_runner opencl_runner.c -lOpenCL -lm

# Run on GPU
./opencl_runner vector_add.spv
```

## Running the Example

```bash
chmod +x run_llvm_on_opencl.sh
./run_llvm_on_opencl.sh
```

## Note on LLVM IR to SPIR-V

The original approach was to convert LLVM IR directly to SPIR-V:

```
LLVM IR (.ll) → LLVM Bitcode (.bc) → SPIR-V (.spv) → OpenCL Execution
```

However, this requires matching LLVM versions between the IR file format and the LLVM toolchain. Due to the evolving nature of LLVM IR (especially with the transition to opaque pointers), we've simplified to the direct OpenCL C approach for reliability. 