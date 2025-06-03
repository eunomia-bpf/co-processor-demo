#!/bin/bash
# Simple script to convert LLVM IR to SPIR-V

set -e

INPUT="$1"
OUTPUT="$2"

if [ -z "$INPUT" ] || [ -z "$OUTPUT" ]; then
    echo "Usage: $0 <input.ll> <output.spv>"
    exit 1
fi

# Check if input file exists
if [ ! -f "$INPUT" ]; then
    echo "Error: Input file '$INPUT' does not exist"
    exit 1
fi

# Create temporary directory
TEMP_DIR=$(mktemp -d)
trap 'rm -rf "$TEMP_DIR"' EXIT

# Find tools (simplified approach)
LLVM_AS="llvm-as"
LLVM_SPIRV="llvm-spirv"

# Check if tools exist
command -v $LLVM_AS >/dev/null 2>&1 || { echo "Error: $LLVM_AS not found"; exit 1; }
command -v $LLVM_SPIRV >/dev/null 2>&1 || { echo "Error: $LLVM_SPIRV not found"; exit 1; }

echo "Step 1: Converting LLVM IR to bitcode..."
# Try without opaque pointers flag - older LLVM versions don't need it
$LLVM_AS "$INPUT" -o "$TEMP_DIR/temp.bc"

echo "Step 2: Converting bitcode to SPIR-V..."
$LLVM_SPIRV "$TEMP_DIR/temp.bc" -o "$OUTPUT"

echo "Successfully converted LLVM IR to SPIR-V: $INPUT -> $OUTPUT"

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