/**
 * ir_to_opencl.c
 * Implementation of LLVM IR to OpenCL converter functions
 */

#include "ir_to_opencl.h"
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>

#define MAX_CMD_LEN 1024
#define MAX_PATH_LEN 256
#define LLVM_VERSION "14"

bool file_exists(const char* filename) {
    struct stat buffer;
    return (stat(filename, &buffer) == 0);
}

bool execute_command(const char* command) {
    printf("Executing: %s\n", command);
    int result = system(command);
    return (result == 0);
}

bool convert_ir_to_opencl(const char* ir_file, const char* cl_file) {
    if (!file_exists(ir_file)) {
        fprintf(stderr, "Error: Input file '%s' does not exist\n", ir_file);
        return false;
    }

    char temp_dir[MAX_PATH_LEN];
    snprintf(temp_dir, MAX_PATH_LEN, "%s.tmp", ir_file);
    mkdir(temp_dir, 0755);

    char temp_bc[MAX_PATH_LEN];
    snprintf(temp_bc, MAX_PATH_LEN, "%s/temp.bc", temp_dir);

    char temp_ll[MAX_PATH_LEN];
    snprintf(temp_ll, MAX_PATH_LEN, "%s/temp.ll", temp_dir);

    char temp_cl_ll[MAX_PATH_LEN];
    snprintf(temp_cl_ll, MAX_PATH_LEN, "%s/temp_cl.ll", temp_dir);

    // Step 1: Convert LLVM IR to bitcode
    char cmd[MAX_CMD_LEN];
    snprintf(cmd, MAX_CMD_LEN, "llvm-as-%s %s -o %s", LLVM_VERSION, ir_file, temp_bc);
    if (!execute_command(cmd)) {
        fprintf(stderr, "Error: Failed to convert IR to bitcode\n");
        return false;
    }

    // Step 2: Process with Clang
    snprintf(cmd, MAX_CMD_LEN, "clang-%s -S -emit-llvm %s -o %s", LLVM_VERSION, ir_file, temp_ll);
    if (!execute_command(cmd)) {
        fprintf(stderr, "Error: Failed to process IR with Clang\n");
        return false;
    }

    // Step 3: Convert to OpenCL with Clang headers
    snprintf(cmd, MAX_CMD_LEN, "clang-%s -x cl -Xclang -finclude-default-header -S -emit-llvm %s -o %s", 
             LLVM_VERSION, temp_ll, temp_cl_ll);
    if (!execute_command(cmd)) {
        fprintf(stderr, "Error: Failed to convert to OpenCL with headers\n");
        return false;
    }

    // Step 4: Convert to readable form
    snprintf(cmd, MAX_CMD_LEN, "llvm-dis-%s %s -o %s", LLVM_VERSION, temp_cl_ll, cl_file);
    if (!execute_command(cmd)) {
        fprintf(stderr, "Error: Failed to convert to readable form\n");
        return false;
    }

    // Cleanup temporary files
    snprintf(cmd, MAX_CMD_LEN, "rm -rf %s", temp_dir);
    execute_command(cmd);

    printf("Successfully converted LLVM IR to OpenCL C: %s -> %s\n", ir_file, cl_file);
    return true;
}

bool convert_ir_to_spirv(const char* ir_file, const char* spv_file) {
    if (!file_exists(ir_file)) {
        fprintf(stderr, "Error: Input file '%s' does not exist\n", ir_file);
        return false;
    }

    char temp_bc[MAX_PATH_LEN];
    snprintf(temp_bc, MAX_PATH_LEN, "build/temp_%ld.bc", (long)time(NULL));

    // Create build directory if it doesn't exist
    if (access("build", F_OK) != 0) {
        if (mkdir("build", 0755) != 0) {
            fprintf(stderr, "Error: Failed to create build directory\n");
            return false;
        }
    }

    // Step 1: Convert LLVM IR to bitcode
    char cmd[MAX_CMD_LEN];
    snprintf(cmd, MAX_CMD_LEN, "llvm-as-%s %s -o %s", LLVM_VERSION, ir_file, temp_bc);
    if (!execute_command(cmd)) {
        fprintf(stderr, "Error: Failed to convert IR to bitcode\n");
        return false;
    }

    // Step 2: Convert bitcode to SPIR-V using llvm-spirv
    // Try using the local build with the correct path
    snprintf(cmd, MAX_CMD_LEN, "./llvm-spirv/build/tools/llvm-spirv/llvm-spirv %s -o %s", temp_bc, spv_file);
    if (!execute_command(cmd)) {
        // Try with system installed llvm-spirv as fallback
        snprintf(cmd, MAX_CMD_LEN, "llvm-spirv %s -o %s", temp_bc, spv_file);
        if (!execute_command(cmd)) {
            fprintf(stderr, "Error: Failed to convert bitcode to SPIR-V. Make sure llvm-spirv is installed.\n");
            return false;
        }
    }

    // Cleanup temporary file
    if (file_exists(temp_bc)) {
        snprintf(cmd, MAX_CMD_LEN, "rm %s", temp_bc);
        execute_command(cmd);
    }

    printf("Successfully converted LLVM IR to SPIR-V: %s -> %s\n", ir_file, spv_file);
    return true;
} 