# LLVM IR to OpenCL via SPIR-V

This example demonstrates how to convert LLVM IR to SPIR-V and execute it on an OpenCL device.

## Overview

The original goal was to implement a library for converting LLVM IR to OpenCL code that can run on a GPU. However, due to compatibility issues with newer LLVM versions (specifically opaque pointers), we've adopted a more flexible approach:

1. A script-based solution (`convert_ir.sh`) that can handle different LLVM versions and pointer formats
2. Support for direct SPIR-V execution on OpenCL devices
3. Fallback to using pre-compiled SPIR-V binaries

## Files

- `llvm_to_opencl.c` - Main C program that loads and executes SPIR-V on an OpenCL device
- `convert_ir.sh` - Script to convert LLVM IR to SPIR-V or OpenCL C
- `convert_cl_to_spirv.sh` - Script to convert OpenCL C directly to SPIR-V
- `vector_add.cl` - OpenCL C kernel for vector addition
- `vector_add.ll` - LLVM IR for vector addition
- `vector_add.spv` - Pre-compiled SPIR-V binary for vector addition
- `SETUP_GUIDE.md` - Detailed guide for setting up the LLVM IR to SPIR-V conversion environment
- `run_example.sh` - Script to run the complete example

## How It Works

1. **LLVM IR to SPIR-V Conversion**:
   - Using `convert_ir.sh`, LLVM IR is converted to SPIR-V with options for handling different LLVM versions
   - The script uses `llvm-as` to convert LLVM IR to bitcode, and `llvm-spirv` to convert bitcode to SPIR-V

2. **Direct OpenCL to SPIR-V Conversion**:
   - Using `convert_cl_to_spirv.sh`, OpenCL C code is directly compiled to SPIR-V
   - This approach is more reliable for modern OpenCL implementations

3. **SPIR-V Execution**:
   - The C program loads the SPIR-V binary and executes it on an OpenCL device
   - It supports both OpenCL 2.1+ (`clCreateProgramWithIL`) and older versions (via binary loading)

## Requirements

- LLVM (preferably version 14 for typed pointers or newer with `-opaque-pointers=0` flag)
- SPIRV-LLVM-Translator (matching your LLVM version)
- OpenCL runtime and development libraries
- (Optional) SPIRV-Tools for validation

See `SETUP_GUIDE.md` for detailed setup instructions.

## Usage

### Converting LLVM IR to SPIR-V

```bash
./convert_ir.sh -s -o output.spv input.ll
```

For newer LLVM versions with opaque pointers:

```bash
./convert_ir.sh -n -s -o output.spv input.ll
```

### Converting OpenCL C to SPIR-V

```bash
./convert_cl_to_spirv.sh input.cl output.spv
```

### Running the Example

```bash
./run_example.sh
```

This will:
1. Convert `vector_add.ll` to SPIR-V (if possible)
2. Compile and run the C program using the generated SPIR-V

## Limitations

- LLVM version compatibility: LLVM IR format changes frequently, particularly with the transition to opaque pointers in LLVM 14+
- OpenCL compatibility: Not all OpenCL implementations support SPIR-V equally
- Complex kernels: More complex LLVM IR may require additional handling

## Future Improvements

- Support for more complex LLVM IR constructs
- Better error handling and diagnostics
- Integration with LLVM JIT compilation
- Support for more OpenCL features and extensions

## Acknowledgments

This example uses components from:
- LLVM project
- Khronos Group (OpenCL, SPIR-V)
- SPIRV-LLVM-Translator 