# Simplified LLVM IR to OpenCL Example

This is a simplified example demonstrating how to convert LLVM IR to SPIR-V and run it on an OpenCL device.

## Overview

The example demonstrates a simple vector addition kernel written in LLVM IR, which is then:
1. Converted to SPIR-V format
2. Loaded and executed on an OpenCL device

This shows the complete pipeline from LLVM IR → SPIR-V → GPU execution.

## Files

- `simple_vector_add.ll` - LLVM IR for a vector addition kernel
- `simple_converter.sh` - Script to convert LLVM IR to SPIR-V
- `simple_opencl_runner.c` - C program to execute SPIR-V on an OpenCL device
- `simple_runner.sh` - Script to run the complete example
- `simple_makefile` - Makefile for building and running the example

## Requirements

- LLVM (with llvm-as tool)
- SPIRV-LLVM-Translator (with llvm-spirv tool)
- OpenCL runtime and development libraries

## How It Works

### 1. LLVM IR to SPIR-V Conversion

The conversion process consists of two steps:
```
LLVM IR (.ll) → LLVM Bitcode (.bc) → SPIR-V (.spv)
```

The `simple_converter.sh` script automates this process:
- It first converts LLVM IR to LLVM bitcode using `llvm-as`
- Then converts the bitcode to SPIR-V using `llvm-spirv`

### 2. Running SPIR-V on OpenCL

The `simple_opencl_runner.c` program:
- Loads the SPIR-V binary
- Sets up OpenCL environment (context, queue, buffers)
- Creates a program from the SPIR-V binary
- Executes the kernel and verifies the results

## Running the Example

```bash
# Using the provided script
chmod +x simple_runner.sh
./simple_runner.sh

# Or using the makefile
make -f simple_makefile run
```

## Explanation of Vector Addition Kernel

The LLVM IR in `simple_vector_add.ll` represents a simple vector addition kernel that:
1. Gets the global ID (work-item index in OpenCL)
2. Loads values from input arrays A and B at that index
3. Adds them together
4. Stores the result in array C

This shows the essential structure of GPU kernels:
- Each work-item (thread) processes one element of the arrays
- Memory accesses use address spaces to identify global memory (addrspace(1))
- Built-in functions like get_global_id() provide thread identification

## Differences from Full Example

This simplified example:
- Removes complex error handling and compatibility code
- Focuses on the essential conversion and execution steps
- Provides clearer visualization of the LLVM IR → SPIR-V → OpenCL process 