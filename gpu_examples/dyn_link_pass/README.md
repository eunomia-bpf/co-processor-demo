# GEMM Kernel - Policy Pass Demo

This directory contains a simple GEMM (General Matrix Multiply) kernel implementation.

## Structure

- `gemm_kernel.cu` - Basic GEMM kernel implementation
- `Makefile` - Build system

## Building

```bash
# Build the test
make

# Run the test
make run

# Clean
make clean
```

## About

This is a baseline GEMM kernel that can be used for demonstrating policy enforcement techniques. The kernel performs matrix multiplication: C = alpha * A * B + beta * C.
