# Co-Processor Demo

This repository contains examples of using different co-processors available on Intel Core Ultra 7 258V processors.

## Repository Structure

- `gpu_examples/`: Examples that use the Intel Arc GPU as a co-processor via OpenCL
- `npu_examples/`: Examples that use the Intel NPU (Neural Processing Unit) as a co-processor via OpenVINO

## GPU Examples

The GPU examples use OpenCL to offload computations to the Intel Arc GPU:

1. **Vector Addition**: A simple example that adds two vectors using the GPU
2. **Matrix Multiplication**: A more complex example that performs matrix multiplication using the GPU

### Building GPU Examples

```bash
cd gpu_examples
make
```

### Running GPU Examples

```bash
cd gpu_examples
./vector_add
./matrix_mul
```

## NPU Examples

The NPU examples use OpenVINO to offload computations to the Intel Neural Processing Unit:

1. **Matrix Multiplication**: Uses the NPU to perform matrix multiplication by creating a simple neural network model

### Building NPU Examples

**Note**: The NPU examples require OpenVINO to be installed. If OpenVINO is not found, the build process will provide instructions for installation.

```bash
cd npu_examples
make
```

### Running NPU Examples

```bash
cd npu_examples
./npu_matrix_mul
```

## Building All Examples

To build all examples (both GPU and NPU), simply run:

```bash
make
```

## System Requirements

- Intel Core Ultra 7 258V processor (or compatible)
- OpenCL runtime for GPU examples
- OpenVINO toolkit for NPU examples