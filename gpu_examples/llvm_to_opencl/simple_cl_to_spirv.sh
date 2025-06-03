#!/bin/bash
# Simple script to convert OpenCL C to SPIR-V

set -e

INPUT="$1"
OUTPUT="$2"

if [ -z "$INPUT" ] || [ -z "$OUTPUT" ]; then
    echo "Usage: $0 <input.cl> <output.spv>"
    exit 1
fi

# Check if input file exists
if [ ! -f "$INPUT" ]; then
    echo "Error: Input file '$INPUT' does not exist"
    exit 1
fi

# Try to find OpenCL compiler
if command -v clang >/dev/null 2>&1; then
    COMPILER="clang"
elif command -v clspv >/dev/null 2>&1; then
    COMPILER="clspv"
else
    echo "Error: No OpenCL compiler found (clang or clspv)"
    exit 1
fi

# Convert OpenCL C to SPIR-V
echo "Converting OpenCL C to SPIR-V using $COMPILER..."

if [ "$COMPILER" = "clang" ]; then
    clang -cc1 -emit-spirv -cl-std=CL2.0 -o "$OUTPUT" "$INPUT"
elif [ "$COMPILER" = "clspv" ]; then
    clspv -o "$OUTPUT" "$INPUT"
fi

echo "Successfully converted OpenCL C to SPIR-V: $INPUT -> $OUTPUT"

# Validate if spirv-val exists (optional)
if command -v spirv-val >/dev/null 2>&1; then
    echo "Validating SPIR-V..."
    if spirv-val "$OUTPUT"; then
        echo "SPIR-V validation successful!"
    else
        echo "Warning: SPIR-V validation failed. The binary may not work correctly."
    fi
fi

exit 0 