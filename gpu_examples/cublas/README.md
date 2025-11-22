# cuBLAS Demo

This is a simple demonstration of using NVIDIA's cuBLAS library to perform matrix multiplication on the GPU.

## Overview

The demo performs matrix multiplication: `C = A × B` where:
- Matrix A: 3×4
- Matrix B: 4×2
- Result C: 3×2

The program:
1. Initializes matrices A and B with sequential values
2. Copies data to GPU memory
3. Uses `cublasSgemm` to perform single-precision matrix multiplication
4. Copies results back to host
5. Verifies correctness with CPU computation

## Features

- Error checking for both CUDA and cuBLAS operations
- Result verification against CPU computation
- Clear output showing input matrices and results
- Proper resource cleanup

## Building

```bash
make
```

This will compile the program using `nvcc` with cuBLAS linking.

## Running

```bash
make run
# or directly:
./cublas_demo
```

## Expected Output

The program will display:
- Input matrix A (3×4)
- Input matrix B (4×2)
- Result matrix C (3×2)
- Verification status

## Requirements

- CUDA Toolkit (with cuBLAS library)
- NVIDIA GPU with compute capability 7.0 or higher
- Linux environment

## Notes

- cuBLAS uses **column-major** ordering (Fortran-style), unlike C's row-major ordering
- The demo handles this by adjusting the order of operations
- Compute capability can be adjusted in the Makefile (`-arch=sm_70`)

## Customization

To change matrix dimensions, modify these constants in `cublas_demo.cu`:
```c
const int m = 3;  // rows of A and C
const int k = 4;  // cols of A, rows of B
const int n = 2;  // cols of B and C
```
