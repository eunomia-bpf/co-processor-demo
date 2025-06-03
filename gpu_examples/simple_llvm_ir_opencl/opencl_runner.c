#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <CL/cl.h>

#define VECTOR_SIZE 1024
#define KERNEL_NAME "vector_add"

// Read SPIR-V binary with error handling
unsigned char* read_binary_file(const char* filename, size_t* size) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error: Failed to open file %s\n", filename);
        exit(1);
    }
    
    fseek(file, 0, SEEK_END);
    *size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    if (*size == 0) {
        printf("Error: File %s is empty\n", filename);
        fclose(file);
        exit(1);
    }
    
    unsigned char* binary = (unsigned char*)malloc(*size);
    if (!binary) {
        printf("Error: Memory allocation failed\n");
        fclose(file);
        exit(1);
    }
    
    size_t read_size = fread(binary, 1, *size, file);
    fclose(file);
    
    if (read_size != *size) {
        printf("Error: Failed to read entire file\n");
        free(binary);
        exit(1);
    }
    
    return binary;
}

int main(int argc, char* argv[]) {
    // Check arguments
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
    
    if (!a || !b || !c) {
        printf("Error: Memory allocation failed\n");
        exit(1);
    }
    
    for (int i = 0; i < VECTOR_SIZE; i++) {
        a[i] = i;
        b[i] = VECTOR_SIZE - i;
    }
    
    // Initialize OpenCL
    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_uint num_platforms = 0;
    cl_uint num_devices = 0;
    
    // Get platform
    err = clGetPlatformIDs(1, &platform, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        printf("Error: Failed to find OpenCL platform\n");
        return 1;
    }
    
    // Get device
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &num_devices);
    if (err != CL_SUCCESS || num_devices == 0) {
        printf("Error: Failed to find OpenCL GPU device, trying CPU...\n");
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, &num_devices);
        if (err != CL_SUCCESS || num_devices == 0) {
            printf("Error: Failed to find any OpenCL device\n");
            return 1;
        }
    }
    
    // Print device info
    char device_name[256] = {0};
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    printf("Using device: %s\n", device_name);
    
    // Create context
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to create OpenCL context\n");
        return 1;
    }
    
    // Create command queue
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to create command queue\n");
        return 1;
    }
    
    // Create buffers
    cl_mem a_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * VECTOR_SIZE, NULL, &err);
    cl_mem b_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * VECTOR_SIZE, NULL, &err);
    cl_mem c_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * VECTOR_SIZE, NULL, &err);
    
    if (!a_mem || !b_mem || !c_mem) {
        printf("Error: Failed to create OpenCL buffers\n");
        return 1;
    }
    
    // Copy input data to device
    err = clEnqueueWriteBuffer(queue, a_mem, CL_TRUE, 0, sizeof(float) * VECTOR_SIZE, a, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, b_mem, CL_TRUE, 0, sizeof(float) * VECTOR_SIZE, b, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to write to device memory\n");
        return 1;
    }
    
    // Create program from SPIR-V binary
    cl_program program = NULL;
    
    // Try clCreateProgramWithIL first (OpenCL 2.1+)
    program = clCreateProgramWithIL(context, binary, binary_size, &err);
    if (err != CL_SUCCESS) {
        printf("Failed to create program with IL, trying binary format...\n");
        
        // Fall back to binary format for older OpenCL
        program = clCreateProgramWithBinary(context, 1, &device, &binary_size, 
                                          (const unsigned char**)&binary, NULL, &err);
        if (err != CL_SUCCESS) {
            printf("Error: Failed to create program from binary\n");
            return 1;
        }
    }
    
    // Build program
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size + 1);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        log[log_size] = '\0';
        printf("Error: Failed to build program. Build log:\n%s\n", log);
        free(log);
        return 1;
    }
    
    // Create kernel
    cl_kernel kernel = clCreateKernel(program, KERNEL_NAME, &err);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to create kernel\n");
        return 1;
    }
    
    // Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_mem);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to set kernel arguments\n");
        return 1;
    }
    
    // Execute kernel
    size_t global_size = VECTOR_SIZE;
    printf("Executing kernel...\n");
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to execute kernel\n");
        return 1;
    }
    
    // Read result
    err = clEnqueueReadBuffer(queue, c_mem, CL_TRUE, 0, sizeof(float) * VECTOR_SIZE, c, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to read results from device\n");
        return 1;
    }
    
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