#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <sys/time.h>
#include <math.h>

#define MATRIX_SIZE 1024
#define BLOCK_SIZE 16

// OpenCL kernel for matrix multiplication
const char* kernel_source = 
"__kernel void matrix_mul(__global float* A, __global float* B, __global float* C, const int size) {\n"
"    int row = get_global_id(0);\n"
"    int col = get_global_id(1);\n"
"    if (row < size && col < size) {\n"
"        float sum = 0.0f;\n"
"        for (int k = 0; k < size; k++) {\n"
"            sum += A[row * size + k] * B[k * size + col];\n"
"        }\n"
"        C[row * size + col] = sum;\n"
"    }\n"
"}\n";

// Function to perform matrix multiplication on CPU for comparison
void matrix_multiply_cpu(float* A, float* B, float* C, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            float sum = 0.0f;
            for (int k = 0; k < size; k++) {
                sum += A[i * size + k] * B[k * size + j];
            }
            C[i * size + j] = sum;
        }
    }
}

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main() {
    // Allocate host memory for matrices
    float* A = (float*)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    float* B = (float*)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    float* C_gpu = (float*)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    float* C_cpu = (float*)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

    // Initialize matrices with random data
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
        A[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;  // Random values between -1 and 1
        B[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }

    printf("Performing %dx%d matrix multiplication...\n", MATRIX_SIZE, MATRIX_SIZE);

    // ---------------------------
    // Set up OpenCL
    // ---------------------------
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_int err;

    // Get platform and device information
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) {
        printf("Error getting platform ID: %d\n", err);
        return 1;
    }

    // Try to get GPU device first, fallback to CPU if not available
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        printf("GPU device not found, trying CPU...\n");
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
        if (err != CL_SUCCESS) {
            printf("Error getting device ID: %d\n", err);
            return 1;
        }
    }

    // Print device name
    char device_name[128];
    err = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    if (err == CL_SUCCESS) {
        printf("Using device: %s\n", device_name);
    }

    // Create OpenCL context
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Error creating context: %d\n", err);
        return 1;
    }

    // Create command queue
    queue = clCreateCommandQueue(context, device, 0, &err);
    if (err != CL_SUCCESS) {
        printf("Error creating command queue: %d\n", err);
        return 1;
    }

    // Create program from kernel source
    program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Error creating program: %d\n", err);
        return 1;
    }

    // Build program
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        printf("Error building program: %s\n", log);
        free(log);
        return 1;
    }

    // Create kernel
    kernel = clCreateKernel(program, "matrix_mul", &err);
    if (err != CL_SUCCESS) {
        printf("Error creating kernel: %d\n", err);
        return 1;
    }

    // Create buffers
    cl_mem A_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), NULL, &err);
    cl_mem B_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), NULL, &err);
    cl_mem C_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), NULL, &err);

    // Copy data to buffers
    err = clEnqueueWriteBuffer(queue, A_buf, CL_TRUE, 0, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), A, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, B_buf, CL_TRUE, 0, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), B, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error copying data to device: %d\n", err);
        return 1;
    }

    // Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &A_buf);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &B_buf);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &C_buf);
    int size = MATRIX_SIZE;
    err |= clSetKernelArg(kernel, 3, sizeof(int), &size);
    if (err != CL_SUCCESS) {
        printf("Error setting kernel arguments: %d\n", err);
        return 1;
    }

    // Execute kernel
    size_t global_work_size[2] = {MATRIX_SIZE, MATRIX_SIZE};
    size_t local_work_size[2] = {BLOCK_SIZE, BLOCK_SIZE};
    
    // Measure GPU execution time
    double gpu_start = get_time();
    
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error executing kernel: %d\n", err);
        return 1;
    }
    
    clFinish(queue);
    double gpu_end = get_time();
    double gpu_time = gpu_end - gpu_start;

    // Copy results back
    err = clEnqueueReadBuffer(queue, C_buf, CL_TRUE, 0, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), C_gpu, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error copying results from device: %d\n", err);
        return 1;
    }

    // Now compute on CPU for comparison
    double cpu_start = get_time();
    matrix_multiply_cpu(A, B, C_cpu, MATRIX_SIZE);
    double cpu_end = get_time();
    double cpu_time = cpu_end - cpu_start;

    // Verify results
    int correct = 1;
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
        if (fabs(C_gpu[i] - C_cpu[i]) > 1e-2) {
            printf("Verification failed at index %d: GPU=%f, CPU=%f\n", i, C_gpu[i], C_cpu[i]);
            correct = 0;
            break;
        }
    }

    // Print performance comparison
    printf("\nMatrix multiplication %dx%d\n", MATRIX_SIZE, MATRIX_SIZE);
    printf("GPU time: %.2f ms\n", gpu_time * 1000);
    printf("CPU time: %.2f ms\n", cpu_time * 1000);
    printf("Speedup: %.2fx\n", cpu_time / gpu_time);
    printf("Verification: %s\n", correct ? "PASSED" : "FAILED");

    // Note about NPU
    printf("\nNOTE: This demo uses OpenCL which typically targets the GPU.\n");
    printf("To specifically target the NPU on Intel Core Ultra processors,\n");
    printf("you would need to use OpenVINO with the 'NPU' device specified.\n");
    printf("The NPU is optimized for AI workloads rather than general matrix operations.\n");

    // Clean up
    clReleaseMemObject(A_buf);
    clReleaseMemObject(B_buf);
    clReleaseMemObject(C_buf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(A);
    free(B);
    free(C_gpu);
    free(C_cpu);

    return 0;
} 