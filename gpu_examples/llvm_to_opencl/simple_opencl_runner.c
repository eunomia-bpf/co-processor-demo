#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <CL/cl.h>

#define VECTOR_SIZE 1024
#define KERNEL_NAME "vector_add"

// Function to read binary file
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
    fread(binary, 1, *size, file);
    fclose(file);
    
    return binary;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <spir-v-file>\n", argv[0]);
        return 1;
    }
    
    // Read SPIR-V binary
    size_t binary_size = 0;
    unsigned char* binary = read_binary_file(argv[1], &binary_size);
    printf("Read SPIR-V binary, size: %zu bytes\n", binary_size);
    
    // Initialize input data
    float* a = (float*)malloc(sizeof(float) * VECTOR_SIZE);
    float* b = (float*)malloc(sizeof(float) * VECTOR_SIZE);
    float* c = (float*)malloc(sizeof(float) * VECTOR_SIZE);
    
    for (int i = 0; i < VECTOR_SIZE; i++) {
        a[i] = i;
        b[i] = VECTOR_SIZE - i;
    }
    
    // Initialize OpenCL
    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    
    // Get platform and device
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    
    // Print device info
    char device_name[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    printf("Using device: %s\n", device_name);
    
    // Create context and command queue
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    queue = clCreateCommandQueueWithProperties(context, device, NULL, &err);
    
    // Create buffers
    cl_mem a_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * VECTOR_SIZE, NULL, &err);
    cl_mem b_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * VECTOR_SIZE, NULL, &err);
    cl_mem c_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * VECTOR_SIZE, NULL, &err);
    
    // Copy input data to device
    clEnqueueWriteBuffer(queue, a_mem, CL_TRUE, 0, sizeof(float) * VECTOR_SIZE, a, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, b_mem, CL_TRUE, 0, sizeof(float) * VECTOR_SIZE, b, 0, NULL, NULL);
    
    // Create program from SPIR-V binary
    program = clCreateProgramWithIL(context, binary, binary_size, &err);
    if (err != CL_SUCCESS) {
        printf("Failed to create program with IL: %d\n", err);
        printf("Falling back to binary loading...\n");
        
        program = clCreateProgramWithBinary(context, 1, &device, &binary_size, 
                                           (const unsigned char**)&binary, NULL, &err);
        if (err != CL_SUCCESS) {
            printf("Failed to create program with binary: %d\n", err);
            return 1;
        }
    }
    
    // Build program
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        printf("Build error: %s\n", log);
        free(log);
        return 1;
    }
    
    // Create kernel
    kernel = clCreateKernel(program, KERNEL_NAME, &err);
    if (err != CL_SUCCESS) {
        printf("Failed to create kernel: %d\n", err);
        return 1;
    }
    
    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_mem);
    
    // Execute kernel
    size_t global_size = VECTOR_SIZE;
    printf("Executing kernel...\n");
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
    
    // Read result
    clEnqueueReadBuffer(queue, c_mem, CL_TRUE, 0, sizeof(float) * VECTOR_SIZE, c, 0, NULL, NULL);
    
    // Verify result
    printf("Verifying results...\n");
    int correct = 1;
    for (int i = 0; i < VECTOR_SIZE; i++) {
        float expected = a[i] + b[i];
        if (fabs(c[i] - expected) > 0.001) {
            printf("Error at index %d: %f != %f\n", i, c[i], expected);
            correct = 0;
            break;
        }
    }
    
    if (correct) {
        printf("SUCCESS: Vector addition computed correctly!\n");
    } else {
        printf("ERROR: Vector addition failed verification.\n");
    }
    
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
    free(binary);
    
    return 0;
} 