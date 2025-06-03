#!/bin/bash
# Convert LLVM IR to SPIR-V and OpenCL
# This script replaces the functionality of the previously implemented C library

set -e

# Display usage information
usage() {
    echo "Usage: $0 [options] <input_file.ll>"
    echo "Options:"
    echo "  -o <output_file>    Specify output file name (default: based on input name)"
    echo "  -s                  Convert to SPIR-V binary (default)"
    echo "  -c                  Convert to OpenCL C source code"
    echo "  -t                  Output SPIR-V in textual format (for debugging)"
    echo "  -v                  Verbose output"
    echo "  -n                  Use non-opaque pointers mode (for LLVM 14+)"
    echo "  -h                  Show this help message"
    exit 1
}

# Parse arguments
OUTPUT=""
CONVERT_TO_SPIRV=true
CONVERT_TO_OPENCL=false
TEXTUAL_FORMAT=false
VERBOSE=false
NON_OPAQUE_POINTERS=false

while getopts "o:sctvnh" opt; do
    case $opt in
        o) OUTPUT="$OPTARG" ;;
        s) CONVERT_TO_SPIRV=true ;;
        c) CONVERT_TO_OPENCL=true ;;
        t) TEXTUAL_FORMAT=true ;;
        v) VERBOSE=true ;;
        n) NON_OPAQUE_POINTERS=true ;;
        h) usage ;;
        *) usage ;;
    esac
done

shift $((OPTIND - 1))
INPUT="$1"

# Check if input file is provided
if [ -z "$INPUT" ]; then
    echo "Error: No input file specified"
    usage
fi

# Check if input file exists
if [ ! -f "$INPUT" ]; then
    echo "Error: Input file '$INPUT' does not exist"
    exit 1
fi

# Set default output file name if not specified
if [ -z "$OUTPUT" ]; then
    if [ "$CONVERT_TO_SPIRV" = true ]; then
        OUTPUT="${INPUT%.*}.spv"
    else
        OUTPUT="${INPUT%.*}.cl"
    fi
fi

# Create temporary directory
TEMP_DIR=$(mktemp -d)
trap 'rm -rf "$TEMP_DIR"' EXIT

# Log command execution if verbose
run_cmd() {
    if [ "$VERBOSE" = true ]; then
        echo "Executing: $@"
    fi
    "$@" || return $?
}

# Find LLVM tools matching the preferred version
find_llvm_tool() {
    local tool_name="$1"
    local preferred_version="14"  # Default to LLVM 14 which handles typed pointers well
    
    # Try with version suffix first
    if command -v "${tool_name}-${preferred_version}" &>/dev/null; then
        echo "${tool_name}-${preferred_version}"
    # Then try without version suffix
    elif command -v "${tool_name}" &>/dev/null; then
        echo "${tool_name}"
    else
        echo "Error: ${tool_name} tool not found. Please install LLVM."
        exit 1
    fi
}

# Find llvm-spirv tool (either from SPIRV-LLVM-Translator or system installation)
find_llvm_spirv() {
    # Check if we have a local build of llvm-spirv
    if [ -x "./llvm-spirv/build/tools/llvm-spirv/llvm-spirv" ]; then
        echo "./llvm-spirv/build/tools/llvm-spirv/llvm-spirv"
    elif command -v llvm-spirv &>/dev/null; then
        echo "llvm-spirv"
    else
        echo "Error: llvm-spirv tool not found. Please install SPIRV-LLVM-Translator or build it locally."
        echo "See SETUP_GUIDE.md for detailed instructions."
        exit 1
    fi
}

# Check for required tools
check_required_tools() {
    find_llvm_tool "llvm-as" >/dev/null
    
    # Check if spirv-tools are available (optional)
    if command -v spirv-val &>/dev/null; then
        echo "Found SPIRV-Tools (spirv-val) - will validate generated SPIR-V"
        HAVE_SPIRV_TOOLS=true
    else
        HAVE_SPIRV_TOOLS=false
    fi
}

# Convert LLVM IR to SPIR-V
convert_to_spirv() {
    local input_file="$1"
    local output_file="$2"
    local textual="$3"
    local llvm_spirv=$(find_llvm_spirv)
    local llvm_as=$(find_llvm_tool "llvm-as")
    
    echo "Using LLVM tools: $llvm_as"
    echo "Using LLVM-SPIRV: $llvm_spirv"
    
    # Convert LLVM IR to bitcode
    local bc_file="$TEMP_DIR/temp.bc"
    local as_cmd=("$llvm_as")
    
    # Add opaque pointers flag if needed
    if [ "$NON_OPAQUE_POINTERS" = true ]; then
        # Check if llvm-as supports opaque pointers flag
        if "$llvm_as" --help 2>&1 | grep -q "opaque-pointers"; then
            echo "Using non-opaque pointers mode"
            as_cmd+=("-opaque-pointers=0")
        else
            echo "Warning: This version of llvm-as doesn't support the -opaque-pointers flag."
            echo "Using default pointer mode, which might cause errors if there's a mismatch."
        fi
    fi
    
    as_cmd+=("$input_file" "-o" "$bc_file")
    
    if ! run_cmd "${as_cmd[@]}"; then
        echo "Error: Failed to convert LLVM IR to bitcode."
        echo "This might be due to pointer type mismatch. Try using the -n flag for non-opaque pointers mode."
        return 1
    fi
    
    # Convert bitcode to SPIR-V
    local spirv_cmd=("$llvm_spirv" "$bc_file" "-o" "$output_file")
    if [ "$textual" = true ]; then
        spirv_cmd+=("-spirv-text")
    fi
    
    if ! run_cmd "${spirv_cmd[@]}"; then
        echo "Error: Failed to convert bitcode to SPIR-V."
        echo "This might be due to incompatibilities between your LLVM version and the SPIRV-LLVM-Translator."
        echo "See SETUP_GUIDE.md for detailed setup instructions."
        return 1
    fi
    
    echo "Successfully converted LLVM IR to SPIR-V: $input_file -> $output_file"
    
    # Validate SPIR-V if spirv-val is available
    if [ "$HAVE_SPIRV_TOOLS" = true ]; then
        echo "Validating generated SPIR-V..."
        if ! spirv-val "$output_file" 2>/dev/null; then
            echo "Warning: Generated SPIR-V failed validation. It may not work with all OpenCL implementations."
        else
            echo "SPIR-V validation successful!"
        fi
    fi
    
    return 0
}

# Convert LLVM IR to OpenCL C
convert_to_opencl() {
    local input_file="$1"
    local output_file="$2"
    local spv_file="$TEMP_DIR/temp.spv"
    local llvm_spirv=$(find_llvm_spirv)
    
    # First convert to SPIR-V
    if ! convert_to_spirv "$input_file" "$spv_file" false; then
        echo "Error: Failed to convert to SPIR-V, cannot proceed with OpenCL conversion."
        return 1
    fi
    
    # Then convert SPIR-V to OpenCL C (if possible)
    if command -v spirv-to-opencl &>/dev/null; then
        if ! run_cmd spirv-to-opencl "$spv_file" -o "$output_file"; then
            echo "Error: Failed to convert SPIR-V to OpenCL C."
            return 1
        fi
        echo "Successfully converted LLVM IR to OpenCL C: $input_file -> $output_file"
    else
        # Alternative: Use clang to disassemble SPIR-V to pseudo-OpenCL
        # This is not perfect but can be useful for reference
        local bc_file="$TEMP_DIR/temp.bc"
        if ! run_cmd "$llvm_spirv" -r "$spv_file" -o "$bc_file"; then
            echo "Error: Failed to convert SPIR-V back to LLVM IR."
            return 1
        fi
        
        local llvm_dis=$(find_llvm_tool "llvm-dis")
        if ! run_cmd "$llvm_dis" "$bc_file" -o "$output_file"; then
            echo "Error: Failed to disassemble LLVM IR."
            return 1
        fi
        
        echo "Note: Converted to LLVM IR text (not OpenCL C): $input_file -> $output_file"
        echo "For true OpenCL C conversion, install spirv-to-opencl from SPIRV-Tools"
    fi
    
    return 0
}

# Main execution
check_required_tools

if [ "$CONVERT_TO_OPENCL" = true ]; then
    convert_to_opencl "$INPUT" "$OUTPUT"
else
    convert_to_spirv "$INPUT" "$OUTPUT" "$TEXTUAL_FORMAT"
fi

exit 0 