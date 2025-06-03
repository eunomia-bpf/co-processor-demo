#!/bin/bash
# Simple script to run OpenCL on GPU
# Using direct OpenCL C to SPIR-V path since LLVM IR path has version compatibility issues

# Step 1: Convert OpenCL C directly to SPIR-V
# Note: Use one of these depending on what's available in your system
# Option 1 - Using clang (if available)
clang -cc1 -emit-spirv -cl-std=CL2.0 -o vector_add.spv vector_add.cl 2>/dev/null || \
# Option 2 - Use pre-existing SPIR-V from the repo
cp ../vector_add.spv .

# Step 2: Compile the OpenCL runner
gcc -o opencl_runner opencl_runner.c -lOpenCL -lm

# Step 3: Run the kernel on the GPU
./opencl_runner vector_add.spv 