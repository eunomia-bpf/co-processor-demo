#!/bin/bash
# Convert OpenCL to SPIR-V
# This script creates a simple wrapper program to convert OpenCL to SPIR-V binary

set -e

# Parse arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <input.cl> [output.spv]"
    exit 1
fi

INPUT="$1"
OUTPUT="${2:-${INPUT%.*}.spv}"

echo "Converting OpenCL to SPIR-V using a wrapper program: $INPUT -> $OUTPUT"

# Create a temporary directory
TEMP_DIR=$(mktemp -d)
trap 'rm -rf "$TEMP_DIR"' EXIT

# Create a simple C program to compile the OpenCL kernel to SPIR-V
cat > "$TEMP_DIR/compile_kernel.c" << 'EOF'
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CL/cl.h>

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <input.cl> <output.spv>\n", argv[0]);
        return 1;
    }
    
    const char* input_file = argv[1];
    const char* output_file = argv[2];
    
    // Read the OpenCL kernel source
    FILE* fp = fopen(input_file, "rb");
    if (!fp) {
        fprintf(stderr, "Error: Could not open input file %s\n", input_file);
        return 1;
    }
    
    // Get file size
    fseek(fp, 0, SEEK_END);
    size_t source_size = ftell(fp);
    rewind(fp);
    
    // Read the file
    char* source_str = (char*)malloc(source_size + 1);
    if (!source_str) {
        fprintf(stderr, "Error: Failed to allocate memory for kernel source\n");
        fclose(fp);
        return 1;
    }
    
    size_t read_size = fread(source_str, 1, source_size, fp);
    fclose(fp);
    if (read_size != source_size) {
        fprintf(stderr, "Error: Failed to read the kernel source\n");
        free(source_str);
        return 1;
    }
    source_str[source_size] = '\0';
    
    // Initialize OpenCL
    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    
    // Get platform
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error: Failed to get platform ID: %d\n", err);
        free(source_str);
        return 1;
    }
    
    // Get device
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error: Failed to get device ID: %d\n", err);
        free(source_str);
        return 1;
    }
    
    // Create context
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error: Failed to create context: %d\n", err);
        free(source_str);
        return 1;
    }
    
    // Create program from source
    cl_program program = clCreateProgramWithSource(context, 1, 
        (const char**)&source_str, &source_size, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error: Failed to create program: %d\n", err);
        clReleaseContext(context);
        free(source_str);
        return 1;
    }
    
    // Build program with SPIR-V output
    const char* options = "-cl-std=CL1.2 -cl-kernel-arg-info";
    err = clBuildProgram(program, 1, &device, options, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error: Failed to build program: %d\n", err);
        
        // Get build log
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        fprintf(stderr, "Build log:\n%s\n", log);
        free(log);
        
        clReleaseProgram(program);
        clReleaseContext(context);
        free(source_str);
        return 1;
    }
    
    // Get binary (could be SPIR-V or platform-specific binary)
    size_t binary_size;
    err = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &binary_size, NULL);
    if (err != CL_SUCCESS || binary_size == 0) {
        fprintf(stderr, "Error: Failed to get program binary size: %d\n", err);
        clReleaseProgram(program);
        clReleaseContext(context);
        free(source_str);
        return 1;
    }
    
    unsigned char* binary = (unsigned char*)malloc(binary_size);
    unsigned char* binaries[] = { binary };
    err = clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(unsigned char*), binaries, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error: Failed to get program binary: %d\n", err);
        free(binary);
        clReleaseProgram(program);
        clReleaseContext(context);
        free(source_str);
        return 1;
    }
    
    // Write binary to output file
    fp = fopen(output_file, "wb");
    if (!fp) {
        fprintf(stderr, "Error: Could not open output file %s\n", output_file);
        free(binary);
        clReleaseProgram(program);
        clReleaseContext(context);
        free(source_str);
        return 1;
    }
    
    size_t written = fwrite(binary, 1, binary_size, fp);
    fclose(fp);
    
    if (written != binary_size) {
        fprintf(stderr, "Error: Failed to write the binary data\n");
        free(binary);
        clReleaseProgram(program);
        clReleaseContext(context);
        free(source_str);
        return 1;
    }
    
    printf("Successfully compiled OpenCL kernel to binary: %s -> %s\n", input_file, output_file);
    printf("Binary size: %zu bytes\n", binary_size);
    
    // Clean up
    free(binary);
    clReleaseProgram(program);
    clReleaseContext(context);
    free(source_str);
    
    return 0;
}
EOF

# Compile the program
gcc -o "$TEMP_DIR/compile_kernel" "$TEMP_DIR/compile_kernel.c" -lOpenCL

# Run the program to compile the kernel
"$TEMP_DIR/compile_kernel" "$INPUT" "$OUTPUT"

echo "Conversion complete!"
exit 0 