# LLVM IR to OpenCL Conversion Example

This example demonstrates a conceptual workflow for converting LLVM IR to OpenCL kernel code and running it on a GPU.

## Overview

The example consists of:

1. An LLVM IR file `vector_add.ll` that represents a vector addition kernel
2. A corresponding OpenCL kernel file `vector_add.cl` (used for demonstration)
3. A program that simulates the conversion from LLVM IR to OpenCL and executes the kernel

## Files

- `vector_add.ll`: A vector addition kernel written in LLVM IR format
- `vector_add.cl`: The equivalent OpenCL kernel implementation
- `llvm_to_opencl.c`: The main program that simulates LLVM IR conversion and executes the kernel
- `Makefile`: For building the example

## How it Works

1. The program simulates converting the LLVM IR to OpenCL code
   - In a real implementation, you would use Clang/LLVM libraries programmatically
   - For this demonstration, we use a pre-written OpenCL kernel file
2. The generated OpenCL code is written to `generated_kernel.cl` for inspection
3. The program initializes OpenCL, creates buffers, and compiles/executes the kernel
4. Results are verified by comparing with a CPU implementation

## Building and Running

To build:

```
cd gpu_examples/llvm_to_opencl
make
```

To run:

```
./llvm_to_opencl
```

## Implementing a Real LLVM IR to OpenCL Converter

In a real implementation, you would:

1. Use the LLVM C++ API to parse and manipulate the LLVM IR
2. Use Clang's CodeGen to convert LLVM IR to OpenCL C code
3. Alternatively, use the SPIRV-LLVM Translator to convert LLVM IR to SPIR-V binary
   - Then use an OpenCL runtime that supports SPIR-V (OpenCL 2.1+)

## Requirements

- OpenCL development libraries
- GCC or compatible compiler 