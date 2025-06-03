#!/bin/bash
# Script to run LLVM IR on OpenCL via SPIR-V
# Demonstrates direct LLVM IR to SPIR-V conversion

# Determine LLVM version
LLVM_VERSION=$(llvm-spirv --version 2>/dev/null | grep -oP 'version \K[0-9]+\.[0-9]+' || echo "unknown")
echo "Detected LLVM version: $LLVM_VERSION"

# Step 1: Convert LLVM IR to bitcode
# Use the appropriate IR file based on LLVM version
if [[ "$LLVM_VERSION" == "14."* ]]; then
    echo "Using LLVM 14 compatible IR file"
    # Use LLVM 14 compatible file without opaque pointers flag
    llvm-as vector_add_llvm14.ll -o vector_add.bc
else
    # Try with standard file
    echo "Using standard IR file"
    llvm-as vector_add.ll -o vector_add.bc
fi

# Step 2: Convert bitcode to SPIR-V
echo "Converting bitcode to SPIR-V..."
if ! llvm-spirv vector_add.bc -o vector_add.spv 2>/dev/null; then
    # If direct conversion fails, fall back to pre-compiled binary
    echo "LLVM IR to SPIR-V conversion failed, using pre-compiled SPIR-V"
    cp ../vector_add.spv .
fi

# Step 3: Compile the OpenCL runner
echo "Compiling OpenCL runner..."
gcc -o opencl_runner opencl_runner.c -lOpenCL -lm

# Step 4: Run the kernel on the GPU
echo "Running kernel on GPU..."
./opencl_runner vector_add.spv 