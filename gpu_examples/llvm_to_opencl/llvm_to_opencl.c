#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CL/cl.h>
#include <sys/time.h>
#include <math.h>

#define VECTOR_SIZE 1000000
#define KERNEL_NAME "vector_add"
#define PREGENERATED_SPIRV "vector_add.spv"

// Function to read binary file contents
unsigned char* read_binary_file(const char* filename, size_t* size) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Failed to open file: %s\n", filename);
        exit(1);
    }

    fseek(file, 0, SEEK_END);
    *size = ftell(file);
    fseek(file, 0, SEEK_SET);

    unsigned char* binary = (unsigned char*)malloc(*size);
    if (!binary) {
        printf("Memory allocation failed\n");
        fclose(file);
        exit(1);
    }

    size_t read = fread(binary, 1, *size, file);
    if (read != *size) {
        printf("Failed to read file: %s\n", filename);
        free(binary);
        fclose(file);
        exit(1);
    }

    fclose(file);
    return binary;
}

// Function to execute shell command
int execute_command(const char* command) {
    printf("Executing: %s\n", command);
    return system(command);
}

// Function to generate SPIR-V or check if it exists
int ensure_spirv_exists(const char* spv_file) {
    FILE* file = fopen(spv_file, "rb");
    if (file) {
        printf("Using existing SPIR-V file: %s\n", spv_file);
        fclose(file);
        return 1;
    }
    
    printf("Pre-generated SPIR-V file not found: %s\n", spv_file);
    printf("\nIMPORTANT: To create a valid SPIR-V file, you need:\n");
    printf("1. Compatible LLVM version with your IR file format\n");
    printf("2. SPIRV-LLVM-Translator for that LLVM version\n");
    printf("3. Run: llvm-as vector_add.ll -o vector_add.bc\n");
    printf("4. Run: llvm-spirv vector_add.bc -o %s\n\n", spv_file);
    
    // Try to run convert_ir.sh
    char command[256];
    snprintf(command, sizeof(command), "./convert_ir.sh -s -o %s vector_add.ll", spv_file);
    printf("Attempting conversion with: %s\n", command);
    int result = system(command);
    
    // Check if file was created
    file = fopen(spv_file, "rb");
    if (file) {
        printf("SPIR-V file created successfully!\n");
        fclose(file);
        return 1;
    }
    
    return 0;
}

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main() {
    printf("=================================================================\n");
    printf("                LLVM IR to OpenCL via SPIR-V Example             \n");
    printf("=================================================================\n\n");
    
    printf("This example demonstrates converting LLVM IR to SPIR-V and executing\n");
    printf("it on an OpenCL device. Due to version incompatibilities, you may\n");
    printf("need to generate a valid SPIR-V file separately.\n\n");
    
    // Check if the pre-generated SPIR-V file exists
    if (!ensure_spirv_exists(PREGENERATED_SPIRV)) {
        printf("\n=================================================================\n");
        printf("ERROR: SPIR-V file not found or could not be generated.\n");
        printf("Please follow the instructions above to create a valid SPIR-V file.\n");
        printf("For this demo, you can copy vector_add.cl to vector_add.spv to\n");
        printf("proceed, but the actual execution will fail as it's not valid SPIR-V.\n");
        printf("=================================================================\n");
        return 1;
    }
    
    // Generate a minimal OpenCL file for reference
    FILE* cl_file = fopen("generated_kernel.cl", "w");
    if (cl_file) {
        fprintf(cl_file, "// This is a simplified reference of the vector_add kernel\n");
        fprintf(cl_file, "kernel void vector_add(global float* a, global float* b, global float* c, uint n) {\n");
        fprintf(cl_file, "    size_t i = get_global_id(0);\n");
        fprintf(cl_file, "    if (i < n) {\n");
        fprintf(cl_file, "        c[i] = a[i] + b[i];\n");
        fprintf(cl_file, "    }\n");
        fprintf(cl_file, "}\n");
        fclose(cl_file);
    }
    
    // Initialize data
    float* a = (float*)malloc(sizeof(float) * VECTOR_SIZE);
    float* b = (float*)malloc(sizeof(float) * VECTOR_SIZE);
    float* c = (float*)malloc(sizeof(float) * VECTOR_SIZE);
    float* verify = (float*)malloc(sizeof(float) * VECTOR_SIZE);

    for(int i = 0; i < VECTOR_SIZE; i++) {
        a[i] = rand() / (float)RAND_MAX;
        b[i] = rand() / (float)RAND_MAX;
    }

    // Read SPIR-V binary from pre-generated file
    size_t binary_size = 0;
    unsigned char* binary = read_binary_file(PREGENERATED_SPIRV, &binary_size);
    printf("Read SPIR-V binary, size: %zu bytes\n", binary_size);
    
    printf("\n=================================================================\n");
    printf("                    OpenCL Initialization                        \n");
    printf("=================================================================\n\n");

    // Get platform
    cl_platform_id platform;
    cl_uint num_platforms;
    cl_int err = clGetPlatformIDs(1, &platform, &num_platforms);
    if (err != CL_SUCCESS) {
        printf("Failed to get OpenCL platform: %d\n", err);
        return 1;
    }
    printf("Found %d platform(s)\n", num_platforms);

    // Get device
    cl_device_id device;
    cl_uint num_devices;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &num_devices);
    if (err != CL_SUCCESS) {
        printf("Failed to get OpenCL device: %d\n", err);
        return 1;
    }
    printf("Found %d device(s)\n", num_devices);

    // Check if device supports SPIR-V
    char extensions[4096];
    clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, sizeof(extensions), extensions, NULL);
    if (strstr(extensions, "cl_khr_il_program") == NULL) {
        printf("Warning: Device does not explicitly support SPIR-V (cl_khr_il_program).\n");
        printf("Falling back to compatibility mode. This may not work on all devices.\n");
    }

    // Display device info
    char device_name[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    printf("Using device: %s\n", device_name);

    // Create context
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Failed to create OpenCL context: %d\n", err);
        return 1;
    }

    // Create command queue
    cl_queue_properties properties[] = {CL_QUEUE_PROPERTIES, 0, 0};
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, properties, &err);
    if (err != CL_SUCCESS) {
        printf("Failed to create command queue: %d\n", err);
        return 1;
    }
    
    printf("\n=================================================================\n");
    printf("                    Loading SPIR-V Program                       \n");
    printf("=================================================================\n\n");

    // Create memory buffers
    cl_mem a_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * VECTOR_SIZE, NULL, &err);
    cl_mem b_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * VECTOR_SIZE, NULL, &err);
    cl_mem c_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * VECTOR_SIZE, NULL, &err);

    // Copy input data to device
    err = clEnqueueWriteBuffer(queue, a_mem, CL_TRUE, 0, sizeof(float) * VECTOR_SIZE, a, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, b_mem, CL_TRUE, 0, sizeof(float) * VECTOR_SIZE, b, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Failed to write to device buffer: %d\n", err);
        return 1;
    }

    // Create program from SPIR-V binary
    cl_program program = NULL;
    
    // Try to use clCreateProgramWithIL for OpenCL 2.1+ devices
    cl_int il_error = CL_SUCCESS;
    // Get OpenCL version
    char version[256];
    clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(version), version, NULL);
    printf("OpenCL version: %s\n", version);
    
    // Check if we can use clCreateProgramWithIL
    // First try with direct function call if available
    void* clCreateProgramWithILFn = clGetExtensionFunctionAddressForPlatform(platform, "clCreateProgramWithIL");
    if (clCreateProgramWithILFn || strstr(extensions, "cl_khr_il_program")) {
        printf("Attempting to load SPIR-V using clCreateProgramWithIL...\n");
        program = clCreateProgramWithIL(context, binary, binary_size, &il_error);
        if (il_error == CL_SUCCESS) {
            printf("Created program from SPIR-V using clCreateProgramWithIL\n");
        } else {
            printf("Failed to create program with IL: %d\n", il_error);
        }
    }
    
    // Fallback to binary if IL loading fails or is not supported
    if (program == NULL || il_error != CL_SUCCESS) {
        printf("Falling back to clCreateProgramWithBinary...\n");
        program = clCreateProgramWithBinary(context, 1, &device, &binary_size, 
                                         (const unsigned char**)&binary, NULL, &err);
        if (err != CL_SUCCESS) {
            printf("\n=================================================================\n");
            printf("ERROR: Failed to create program with binary: %d\n", err);
            printf("This is expected if you don't have a valid SPIR-V binary file.\n");
            printf("=================================================================\n");
            return 1;
        }
        printf("Created program from binary using clCreateProgramWithBinary\n");
    }

    // Build the program
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        // Get build log
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        printf("\n=================================================================\n");
        printf("ERROR: Build failed with error: %d\n", err);
        printf("Build log:\n%s\n", log);
        printf("=================================================================\n");
        free(log);
        return 1;
    }
    printf("Program built successfully\n");
    
    printf("\n=================================================================\n");
    printf("                    Executing OpenCL Kernel                      \n");
    printf("=================================================================\n\n");

    // Create kernel
    cl_kernel kernel = clCreateKernel(program, KERNEL_NAME, &err);
    if (err != CL_SUCCESS) {
        printf("Failed to create kernel: %d\n", err);
        return 1;
    }
    
    // Get kernel info
    char kernel_name[128];
    size_t kernel_name_size;
    err = clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, sizeof(kernel_name), kernel_name, &kernel_name_size);
    if (err == CL_SUCCESS) {
        printf("Kernel name: %s\n", kernel_name);
    } else {
        printf("Could not get kernel name: %d\n", err);
    }
    
    // Get kernel argument info if supported
    cl_uint num_args;
    err = clGetKernelInfo(kernel, CL_KERNEL_NUM_ARGS, sizeof(num_args), &num_args, NULL);
    if (err == CL_SUCCESS) {
        printf("Kernel has %u arguments\n", num_args);
    } else {
        printf("Could not get kernel argument count: %d\n", err);
    }

    // Set kernel arguments
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_mem);
    // No 4th argument for array size in our updated kernel
    
    if (err != CL_SUCCESS) {
        printf("Failed to set kernel arguments: %d\n", err);
        return 1;
    }

    // Execute kernel
    size_t global_size = VECTOR_SIZE;
    double start_time = get_time();
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Failed to execute kernel: %d\n", err);
        return 1;
    }
    
    clFinish(queue);
    double gpu_time = get_time() - start_time;
    printf("OpenCL kernel executed\n");

    // Read result
    err = clEnqueueReadBuffer(queue, c_mem, CL_TRUE, 0, sizeof(float) * VECTOR_SIZE, c, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Failed to read from device: %d\n", err);
        return 1;
    }

    // Verify result
    start_time = get_time();
    for(int i = 0; i < VECTOR_SIZE; i++) {
        verify[i] = a[i] + b[i];
    }
    double cpu_time = get_time() - start_time;

    // Check results
    int correct = 1;
    for(int i = 0; i < VECTOR_SIZE; i++) {
        if(fabs(verify[i] - c[i]) > 1e-5) {
            correct = 0;
            printf("Mismatch at position %d: CPU=%f, GPU=%f\n", i, verify[i], c[i]);
            break;
        }
    }

    printf("\n=================================================================\n");
    printf("                         Results                                 \n");
    printf("=================================================================\n\n");
    printf("Vector addition of size %d\n", VECTOR_SIZE);
    printf("GPU Time: %f seconds\n", gpu_time);
    printf("CPU Time: %f seconds\n", cpu_time);
    printf("Verification: %s\n", correct ? "PASSED" : "FAILED");
    printf("Speedup: %fx\n", cpu_time / gpu_time);

    // Cleanup
    clReleaseMemObject(a_mem);
    clReleaseMemObject(b_mem);
    clReleaseMemObject(c_mem);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    
    free(a);
    free(b);
    free(c);
    free(verify);
    free(binary);

    return 0;
} 