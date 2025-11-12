# UVMBench: Comprehensive Benchmark Suite for Unified Virtual Memory Research

## Table of Contents
1. [Overview](#overview)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Directory Structure](#directory-structure)
5. [Building the Benchmarks](#building-the-benchmarks)
6. [Preparing Test Data](#preparing-test-data)
7. [Running Benchmarks](#running-benchmarks)
8. [Benchmark Categories](#benchmark-categories)
9. [Profiling and Analysis](#profiling-and-analysis)
10. [Troubleshooting](#troubleshooting)
11. [Citation](#citation)

---

## Overview

UVMBench is a comprehensive benchmark suite for researching Unified Virtual Memory (UVM) in GPUs. It consists of **32 representative benchmarks** from diverse application domains including:

- Machine Learning
- Linear Algebra
- Graph Theory
- Statistics
- Physics Simulation
- Image Processing
- Bioinformatics
- Data Mining

The suite features:
- **Three versions of each benchmark**: UVM, non-UVM, and UVM with memory oversubscription
- **Unified programming implementation** using CUDA
- **Diverse memory access patterns** (regular and irregular)
- **Memory oversubscription support**
- **Automated profiling tools** for performance analysis

### Key Features

- 32 benchmarks covering wide range of domains
- Support for both UVM and traditional CUDA memory management
- Memory oversubscription testing capabilities
- Python-based profiling wrapper for automated analysis
- Compatible with NVIDIA profiling tools (nvprof, Nsight)

---

## System Requirements

### Hardware
- NVIDIA GPU with Compute Capability 7.5 or higher (recommended)
- Minimum 4GB GPU memory
- For oversubscription tests: 8GB+ system RAM recommended

### Software
- **CUDA Toolkit**: Version 11.0 or higher (tested with CUDA 12.9)
- **GCC/G++**: Version 7.0 or higher
- **Python**: Version 3.6 or higher (for profiling scripts)
- **Operating System**: Linux (Ubuntu 18.04+, CentOS 7+, or similar)

### CUDA Driver
- Driver version compatible with installed CUDA toolkit
- Check compatibility: `nvidia-smi` should show matching CUDA version

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/OSU-STARLAB/UVM_benchmark.git
cd UVM_benchmark
```

### 2. Verify CUDA Installation

```bash
# Check CUDA compiler
nvcc --version

# Check GPU and driver
nvidia-smi

# Verify CUDA samples (optional)
cd /usr/local/cuda/samples/1_Utilities/deviceQuery
make && ./deviceQuery
```

### 3. Configure CUDA Paths

Edit `common/make.config` to match your CUDA installation:

```bash
# Default is /usr/local/cuda-12.9
CUDA_DIR = /usr/local/cuda-12.9

# Library path is auto-detected for 64-bit systems
# CUDA_LIB_DIR will be set to $(CUDA_DIR)/lib64 automatically
```

---

## Directory Structure

```
UVM_benchmark/
├── UVM_benchmarks/          # UVM version benchmarks
│   ├── bfs/                 # Breadth-First Search
│   ├── BN/                  # Bayesian Network
│   ├── CNN/                 # Convolutional Neural Network
│   ├── kmeans/              # K-means clustering
│   ├── knn/                 # K-Nearest Neighbors
│   ├── logistic-regression/ # Logistic Regression
│   ├── SVM/                 # Support Vector Machine
│   ├── polybench/           # Linear algebra benchmarks
│   │   ├── 2DCONV/         # 2D Convolution
│   │   ├── 2MM/            # 2 Matrix Multiplications
│   │   ├── 3DCONV/         # 3D Convolution
│   │   ├── 3MM/            # 3 Matrix Multiplications
│   │   ├── ATAX/           # Matrix transpose and vector multiplication
│   │   ├── BICG/           # BiCGStab linear solver
│   │   ├── CORR/           # Correlation computation
│   │   ├── COVAR/          # Covariance computation
│   │   ├── FDTD-2D/        # Finite Difference Time Domain
│   │   ├── GEMM/           # General Matrix Multiply
│   │   ├── GESUMMV/        # Scalar, vector, matrix multiplication
│   │   ├── GRAMSCHM/       # Gram-Schmidt decomposition
│   │   ├── MVT/            # Matrix vector product
│   │   ├── SYR2K/          # Symmetric rank-2k operations
│   │   └── SYRK/           # Symmetric rank-k operations
│   └── rodinia/             # Rodinia suite benchmarks
│       ├── backprop/       # Backpropagation
│       ├── dwt2d/          # Discrete Wavelet Transform 2D
│       ├── gaussian/       # Gaussian Elimination
│       ├── hotspot/        # HotSpot thermal simulation
│       ├── hotspot3D/      # 3D HotSpot
│       ├── nn/             # Nearest Neighbor
│       ├── nw/             # Needleman-Wunsch
│       ├── particlefilter/ # Particle Filter
│       ├── pathfinder/     # Pathfinder
│       ├── srad/           # SRAD
│       └── streamcluster/  # Stream Cluster
│
├── non_UVM_benchmarks/      # Traditional CUDA version (same structure as UVM)
│
├── UVM_benchmarks_oversub/  # UVM with memory oversubscription support
│
├── data/                    # Input data files
│   ├── bfs/
│   ├── nn/
│   ├── hotspot/
│   ├── hotspot3D/
│   └── SVM/
│
├── common/                  # Shared build configuration
│   ├── make.config         # CUDA paths and settings
│   ├── common.mk           # Common makefile rules
│   └── polybenchUtilFuncts.h
│
├── profiling_wrapper.py    # Automated profiling script
├── metric_list.py          # GPU metrics for profiling
└── Makefile                # Top-level makefile
```

---

## Building the Benchmarks

### Build All Benchmarks

```bash
# Build both UVM and non-UVM versions
make

# Build only UVM version
cd UVM_benchmarks && make

# Build only non-UVM version
cd non_UVM_benchmarks && make
```

### Build Individual Benchmarks

```bash
# Example: Build K-means
cd UVM_benchmarks/kmeans
make

# Example: Build GEMM from polybench
cd UVM_benchmarks/polybench/GEMM
make
```

### Clean Build Artifacts

```bash
# Clean all
make clean

# Clean specific benchmark
cd UVM_benchmarks/kmeans && make clean
```

### Build Issues and Solutions

**Issue 1: CUDA linking errors with device-link**

Some benchmarks in `UVM_benchmarks_oversub` may fail with linking errors. This is due to missing `--no-device-link` flag.

**Solution**: Use the regular `UVM_benchmarks` directory which has proper build configuration.

**Issue 2: Architecture mismatch**

If you see warnings about deprecated GPU targets:

```bash
# Edit the Makefile and update architecture flag
# Change from: -arch=sm_30
# To match your GPU: -arch=sm_75 (for Turing) or -arch=sm_90 (for Hopper)
```

Check your GPU architecture:
```bash
nvidia-smi --query-gpu=compute_cap --format=csv
```

---

## Preparing Test Data

Most benchmarks require input data files. Data must be generated or prepared before running benchmarks.

### BFS (Breadth-First Search)

```bash
cd data/bfs/inputGen

# Generate graph data (nodes, name)
./graphgen 1024 1k      # 1K nodes
./graphgen 8192 8k      # 8K nodes
./graphgen 1048576 1M   # 1M nodes

# Output files: graph1k.txt, graph8k.txt, graph1M.txt
```

### K-means

```bash
cd data/kmeans

# Create directory if it doesn't exist
mkdir -p /path/to/UVM_benchmark/data/kmeans

# Generate random 2D points using Python
python3 << 'EOF'
import random

# Generate data points
with open('1000_points.txt', 'w') as f:
    for i in range(1000):
        x = random.uniform(0, 100)
        y = random.uniform(0, 100)
        f.write(f'{x} {y}\n')

# Generate initial centroids
with open('initCoord.txt', 'w') as f:
    for i in range(2):
        x = random.uniform(0, 100)
        y = random.uniform(0, 100)
        f.write(f'{x} {y}\n')

print('Generated K-means test data')
EOF

# For larger datasets
python3 << 'EOF'
import random

for size in [5000, 10000, 50000, 100000]:
    filename = f'{size}_points.txt'
    with open(filename, 'w') as f:
        for i in range(size):
            x = random.uniform(0, 100)
            y = random.uniform(0, 100)
            f.write(f'{x} {y}\n')
    print(f'Generated {filename}')
EOF
```

### Nearest Neighbor (NN)

```bash
cd data/nn/inputGen

# Generate hurricane dataset
make
./hurricanegen 4  # Generates 4 database files
```

### HotSpot / HotSpot3D

Data files are pre-generated in `data/hotspot/` and `data/hotspot3D/` directories.

### Polybench Benchmarks

Polybench benchmarks typically generate data programmatically. No external data files needed.

---

## Running Benchmarks

### Running Individual Benchmarks

Each benchmark directory contains a `run` script with pre-configured parameters.

#### K-means Example

```bash
cd UVM_benchmarks/kmeans

# Create output directory
mkdir -p result/cuda

# Run with custom parameters
./kmeans_cuda <num_clusters> <input_file> <num_points>

# Example: 2 clusters, 1000 points
./kmeans_cuda 2 ../../data/kmeans/1000_points.txt 1000

# Using the run script (predefined parameters)
chmod +x run
./run
```

#### BFS Example

```bash
cd UVM_benchmarks/bfs

# Run with graph from file
./main <start_vertex> < ../../data/bfs/inputGen/graph1k.txt

# Example: start from vertex 0
./main 0 < ../../data/bfs/inputGen/graph8k.txt
```

#### Polybench GEMM Example

```bash
cd UVM_benchmarks/polybench/GEMM

# Run executable
./gemm.exe

# With run script
chmod +x run
./run
```

#### Rodinia NN Example

```bash
cd UVM_benchmarks/rodinia/nn

# Run with hurricane dataset
./nn ../../data/nn/cane4_0.db -r 5 -lat 30 -lng 90
```

### Comparing UVM vs Non-UVM Performance

```bash
# Run UVM version
cd UVM_benchmarks/kmeans
./kmeans_cuda 2 ../../data/kmeans/10000_points.txt 10000

# Run non-UVM version
cd ../../non_UVM_benchmarks/kmeans
./kmeans_cuda 2 ../../data/kmeans/10000_points.txt 10000

# Compare execution times from output
```

### Memory Oversubscription Tests

```bash
# The UVM_benchmarks_oversub directory contains benchmarks configured
# to test memory oversubscription scenarios

cd UVM_benchmarks_oversub/kmeans

# Note: Some may require build fixes for the --no-device-link flag
# Refer to regular UVM_benchmarks for working examples
```

---

## Benchmark Categories

### Machine Learning (12 benchmarks)
- **2DCONV**: 2D Convolution for image processing
- **3DCONV**: 3D Convolution for video/volumetric data
- **BACKPROP**: Neural network backpropagation
- **BN**: Bayesian Network learning
- **CNN**: Convolutional Neural Network
- **GEMM**: General Matrix Multiply (ML workloads)
- **GESUMMV**: Scalar, vector, matrix multiplication
- **KMEANS**: K-means clustering
- **KNN**: K-Nearest Neighbors classification
- **LR**: Logistic Regression
- **SVM**: Support Vector Machine

### Linear Algebra (10 benchmarks)
- **2MM**: 2 Matrix Multiplications
- **3MM**: 3 Matrix Multiplications
- **ATAX**: Matrix transpose and vector multiplication
- **BICG**: BiCGStab linear solver
- **GAUSSIAN**: Gaussian Elimination
- **GRAMSCHM**: Gram-Schmidt decomposition
- **MVT**: Matrix vector product
- **SYR2K**: Symmetric rank-2k operations
- **SYRK**: Symmetric rank-k operations

### Graph Theory (1 benchmark)
- **BFS**: Breadth-First Search

### Statistics (2 benchmarks)
- **CORR**: Correlation computation
- **COVAR**: Covariance computation

### Others
- **DWT2D**: Discrete Wavelet Transform (Media Compression)
- **FDTD-2D**: Finite Difference Time Domain (Electrodynamics)
- **HOTSPOT**: Thermal simulation (Physics)
- **HOTSPOT3D**: 3D Thermal simulation
- **NW**: Needleman-Wunsch (Bioinformatics)
- **PFILTER**: Particle Filter (Medical Imaging)
- **PATHFINDER**: Grid traversal
- **SRAD**: Speckle Reducing Anisotropic Diffusion (Image Processing)
- **SC**: Stream Cluster (Data Mining)

---

## Profiling and Analysis

### Using NVIDIA Profilers

#### Nsight Compute (Modern CUDA Profiler)

```bash
cd UVM_benchmarks/kmeans

# Basic profiling
ncu --set full ./kmeans_cuda 2 ../../data/kmeans/10000_points.txt 10000

# Memory-focused profiling
ncu --set memory ./kmeans_cuda 2 ../../data/kmeans/10000_points.txt 10000

# Export to file
ncu --export profile.ncu-rep ./kmeans_cuda 2 ../../data/kmeans/10000_points.txt 10000
```

#### nvprof (Legacy, deprecated but still useful)

**Note**: nvprof is deprecated as of CUDA 11, but the provided profiling_wrapper.py uses it. For newer CUDA versions, use Nsight Compute.

```bash
# Basic timeline
nvprof ./kmeans_cuda 2 ../../data/kmeans/10000_points.txt 10000

# With metrics
nvprof --metrics achieved_occupancy,gld_efficiency ./kmeans_cuda 2 ../../data/kmeans/10000_points.txt 10000
```

### Automated Profiling with Python Script

The suite includes `profiling_wrapper.py` for batch profiling:

```bash
# View available metrics
cat metric_list.py

# Edit profiling_wrapper.py to select metrics and benchmarks
vim profiling_wrapper.py

# Run profiling (creates results/ directory with timestamped outputs)
python3 profiling_wrapper.py
```

The script profiles:
- UVM vs non-UVM versions
- Multiple benchmarks in batch
- Exports CSV files with metrics
- Supports PCIe transfer analysis

**Profiling modes available**:
- `profile_general()`: General performance summary
- `profile_details()`: Detailed metrics
- `profile_PCIe()`: PCIe transfer analysis
- `profile_PCIe_UVM()`: UVM-specific PCIe tracking

### Manual Performance Analysis

```bash
# Time measurement is built into benchmarks
cd UVM_benchmarks/kmeans
./kmeans_cuda 2 ../../data/kmeans/10000_points.txt 10000

# Output shows:
# CUDA Took: 0.0206169s for 10000 points.
```

### Key Metrics to Monitor

1. **Execution Time**: Total kernel runtime
2. **Memory Transfer**: Host-to-Device and Device-to-Host bandwidth
3. **Page Faults**: UVM page fault frequency and handling time
4. **PCIe Utilization**: Data transfer efficiency
5. **Kernel Occupancy**: Thread utilization
6. **Memory Bandwidth**: DRAM bandwidth utilization

---

## Troubleshooting

### Common Issues

#### 1. CUDA Driver/Runtime Version Mismatch

**Error**: `CUDA driver version is insufficient for CUDA runtime version`

**Solution**:
```bash
# Check versions
nvidia-smi  # Shows driver CUDA version
nvcc --version  # Shows toolkit CUDA version

# Update driver or use compatible CUDA toolkit
# Driver version must be >= toolkit version
```

#### 2. Segmentation Fault on Benchmark Execution

**Causes**:
- Missing input data files
- Incorrect command-line arguments
- Invalid file paths

**Solutions**:

```bash
# Ensure data files exist
ls -lh data/kmeans/
ls -lh data/bfs/inputGen/

# Check the 'run' script for correct parameters
cat run

# Create required directories
mkdir -p result/cuda
```

#### 3. Build Failures with Linking Errors

**Error**: `#include REGISTERLINKBINARYFILE` errors

**Cause**: Missing `--no-device-link` compilation flag

**Solution**: Use benchmarks from `UVM_benchmarks/` (not `_oversub`) which have correct flags, or add to Makefile:

```makefile
NVCC_FLAGS = --no-device-link -arch=sm_90
```

#### 4. Architecture Warnings

**Warning**: `Support for offline compilation for architectures prior to sm_75 will be removed`

**Solution**: Update Makefiles with your GPU architecture:

```bash
# Find your GPU compute capability
nvidia-smi --query-gpu=compute_cap --format=csv,noheader

# Edit Makefile
# For RTX 3090 (sm_86):
-arch=sm_86

# For RTX 4090 (sm_89):
-arch=sm_89

# For RTX 5090/H100 (sm_90):
-arch=sm_90
```

#### 5. Missing Data Files

**Error**: File not found errors when running benchmarks

**Solution**: Generate data as described in [Preparing Test Data](#preparing-test-data)

#### 6. Permission Denied on Executables

**Error**: `Permission denied` when running `./run` or executables

**Solution**:
```bash
# Make executable
chmod +x run
chmod +x ./kmeans_cuda

# Or use make target
make get_permission
```

#### 7. Out of Memory Errors

**Symptoms**: CUDA out of memory errors

**Solutions**:
- Reduce input dataset size
- Use smaller problem sizes in run scripts
- Check GPU memory: `nvidia-smi`
- Close other GPU-using applications

### Debugging Tips

```bash
# Enable CUDA error checking
export CUDA_LAUNCH_BLOCKING=1

# Check CUDA errors
cuda-memcheck ./kmeans_cuda 2 ../../data/kmeans/1000_points.txt 1000

# Verify GPU is accessible
nvidia-smi

# Check compute capability
nvidia-smi --query-gpu=compute_cap --format=csv

# Monitor GPU during execution
watch -n 0.1 nvidia-smi
```

---

## Test Results Summary

### Environment
- **GPU**: NVIDIA GeForce RTX 5090 (32GB, Compute Capability 9.0)
- **CUDA**: Version 12.9
- **Driver**: 575.57.08
- **OS**: Linux 6.15.11

### Successful Test Run

```bash
$ cd UVM_benchmarks/kmeans
$ ./kmeans_cuda 2 ../../data/kmeans/1000_points.txt 1000

Output:
CUDA Took: 0.0206169s for 1000 points.

Generated outputs:
- result/cuda/1000_centroids.txt
- result/cuda/1000_group_members.txt
- CUDAtimes.txt
```

### Benchmark Categories Status

| Category | Status | Notes |
|----------|--------|-------|
| K-means | Working | Requires data generation |
| BFS | Partial | Data generation works, runtime has driver compatibility issue |
| Polybench | Build issues | Requires --no-device-link flag addition |
| Rodinia NN | Build issues | Linking errors with default config |
| Rodinia (others) | Not tested | Likely similar to NN |

**Recommendation**: Use `UVM_benchmarks/` directory with `--no-device-link` flag in Makefiles for best compatibility.

---

## Citation

If you use this benchmark suite in your research, please cite:

```bibtex
@article{gu2020uvmbench,
  title={UVMBench: A Comprehensive Benchmark Suite for Researching Unified Virtual Memory in GPUs},
  author={Gu, Yongbin and Wu, Wenxuan and Li, Yunfan and Chen, Lizhong},
  journal={arXiv preprint arXiv:2007.09822},
  year={2020}
}
```

**Paper**: Gu, Yongbin, et al. "UVMBench: A Comprehensive Benchmark Suite for Researching Unified Virtual Memory in GPUs." ArXiv:2007.09822 [Cs], July 2020. http://arxiv.org/abs/2007.09822.

---

## Additional Resources

- **Original Repository**: https://github.com/OSU-STARLAB/UVM_benchmark
- **CUDA Documentation**: https://docs.nvidia.com/cuda/
- **UVM Programming Guide**: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#unified-memory-programming-hd
- **Nsight Compute**: https://developer.nvidia.com/nsight-compute

---

## License

See LICENSE file in the repository.

---

## Contact

For issues, questions, or contributions, please refer to the original repository or contact the authors:

- Yongbin Gu: guyo@oregonstate.edu
- Wenxuan Wu: wuwen@oregonstate.edu
- Yunfan Li: liyunf@oregonstate.edu
- Lizhong Chen: chenliz@oregonstate.edu

Oregon State University
School of Electrical Engineering and Computer Science
