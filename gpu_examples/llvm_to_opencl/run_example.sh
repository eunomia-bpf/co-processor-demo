#!/bin/bash

# Build the example
make

# Run the example
./llvm_to_opencl

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