#!/bin/bash
# Run the LLVM IR to OpenCL example

set -e

echo "======================================================================"
echo "          LLVM IR to OpenCL via SPIR-V Example Runner                 "
echo "======================================================================"

# Ensure we are in the correct directory
cd "$(dirname "$0")"

# Ensure the script is executable
chmod +x convert_ir.sh
chmod +x convert_cl_to_spirv.sh

# Try to convert LLVM IR to SPIR-V using the script with non-opaque pointers mode
echo -e "\nAttempting to convert LLVM IR to SPIR-V..."
if ./convert_ir.sh -v -n -s -o vector_add.spv vector_add.ll; then
    echo "Successfully converted LLVM IR to SPIR-V"
    SPIRV_GENERATED=true
else
    echo "Failed to convert LLVM IR to SPIR-V via direct conversion"
    SPIRV_GENERATED=false
fi

# If the LLVM IR conversion failed, try to convert from OpenCL C
if [ "$SPIRV_GENERATED" = false ]; then
    echo -e "\nAttempting to convert OpenCL C to SPIR-V..."
    if ./convert_cl_to_spirv.sh vector_add.cl vector_add.spv; then
        echo "Successfully converted OpenCL C to SPIR-V"
        SPIRV_GENERATED=true
    else
        echo "Failed to convert OpenCL C to SPIR-V"
        echo "Continuing with pre-existing SPIR-V file (if available)"
    fi
fi

# Verify the SPIR-V file exists
if [ ! -f vector_add.spv ]; then
    echo "Error: SPIR-V file not found. Aborting."
    exit 1
fi

# Build the C program
echo -e "\nBuilding the C program..."
make

# Run the example
echo -e "\nRunning the example..."
./llvm_to_opencl

echo -e "\nExample completed successfully!"
exit 0

# Show the SPIR-V binary disassembly if spirv-dis is available
if command -v spirv-dis &> /dev/null; then
    echo -e "\nSPIR-V Disassembly:"
    spirv-dis vector_add.spv
elif [ -f "./spirv-tools/build/tools/spirv-dis" ]; then
    echo -e "\nSPIR-V Disassembly:"
    ./spirv-tools/build/tools/spirv-dis vector_add.spv
else
    echo -e "\nNote: Install spirv-dis to view SPIR-V disassembly"
fi 