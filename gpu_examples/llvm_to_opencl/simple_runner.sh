#!/bin/bash
# Simple script to run the OpenCL example

set -e

echo "=== OpenCL to SPIR-V Simple Example ==="

# Check if we have a SPIR-V file already
if [ -f "vector_add.spv" ]; then
    echo "Using existing SPIR-V file: vector_add.spv"
    SPIRV_FILE="vector_add.spv"
else
    # Step 1: Convert OpenCL C to SPIR-V if we have the tools
    echo "Step 1: Converting OpenCL C to SPIR-V..."
    
    chmod +x simple_cl_to_spirv.sh
    if ./simple_cl_to_spirv.sh simple_vector_add.cl simple_vector_add.spv; then
        SPIRV_FILE="simple_vector_add.spv"
    elif [ -f "vector_add.spv" ]; then
        echo "Using pre-compiled SPIR-V file: vector_add.spv"
        SPIRV_FILE="vector_add.spv"
    else
        echo "Error: Could not generate or find a SPIR-V file"
        exit 1
    fi
fi

# Step 2: Compile the OpenCL runner
echo "Step 2: Compiling the OpenCL runner..."
gcc -o simple_opencl_runner simple_opencl_runner.c -lOpenCL -lm

# Step 3: Run the example
echo "Step 3: Running the example on OpenCL device..."
./simple_opencl_runner $SPIRV_FILE

echo "=== Example completed ===" 