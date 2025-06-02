#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <sys/time.h>
#include <math.h>

#define VECTOR_SIZE 1000000

// Function to read the kernel source code
char* read_kernel_source(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Failed to open kernel file\n");
        exit(1);
    }

    fseek(file, 0, SEEK_END);
    long length = ftell(file);
    fseek(file, 0, SEEK_SET);

    char* source = (char*)malloc(length + 1);
    size_t read = fread(source, 1, length, file);
    if (read != (size_t)length) {
        printf("Failed to read kernel file\n");
        exit(1);
    }
    source[length] = '\0';

    fclose(file);
    return source;
}

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main() {
    // Initialize data
    float* a = (float*)malloc(sizeof(float) * VECTOR_SIZE);
    float* b = (float*)malloc(sizeof(float) * VECTOR_SIZE);
    float* c = (float*)malloc(sizeof(float) * VECTOR_SIZE);
    float* verify = (float*)malloc(sizeof(float) * VECTOR_SIZE);

    for(int i = 0; i < VECTOR_SIZE; i++) {
        a[i] = rand() / (float)RAND_MAX;
        b[i] = rand() / (float)RAND_MAX;
    }

    // Get platform
    cl_platform_id platform;
    cl_uint num_platforms;
    clGetPlatformIDs(1, &platform, &num_platforms);

    // Get device
    cl_device_id device;
    cl_uint num_devices;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &num_devices);

    // Create context
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);

    // Create command queue
    cl_queue_properties properties[] = {CL_QUEUE_PROPERTIES, 0, 0};
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, properties, NULL);

    // Create memory buffers
    cl_mem a_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * VECTOR_SIZE, NULL, NULL);
    cl_mem b_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * VECTOR_SIZE, NULL, NULL);
    cl_mem c_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * VECTOR_SIZE, NULL, NULL);

    // Copy input data to device
    clEnqueueWriteBuffer(queue, a_mem, CL_TRUE, 0, sizeof(float) * VECTOR_SIZE, a, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, b_mem, CL_TRUE, 0, sizeof(float) * VECTOR_SIZE, b, 0, NULL, NULL);

    // Create and build program
    const char* source = read_kernel_source("vector_add.cl");
    cl_program program = clCreateProgramWithSource(context, 1, &source, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    // Create kernel
    cl_kernel kernel = clCreateKernel(program, "vector_add", NULL);

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_mem);
    unsigned int vector_size = VECTOR_SIZE;
    clSetKernelArg(kernel, 3, sizeof(unsigned int), &vector_size);

    // Execute kernel
    size_t global_size = VECTOR_SIZE;
    double start_time = get_time();
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
    clFinish(queue);
    double gpu_time = get_time() - start_time;

    // Read result
    clEnqueueReadBuffer(queue, c_mem, CL_TRUE, 0, sizeof(float) * VECTOR_SIZE, c, 0, NULL, NULL);

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
            break;
        }
    }

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
    free((void*)source);

    return 0;
} 