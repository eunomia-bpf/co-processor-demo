# UVM Benchmarking Methodology and Common Practices

A comprehensive guide to evaluating Unified Virtual Memory (UVM) performance based on academic research and industry practices.

---

## Table of Contents

1. [Overview](#overview)
2. [Creating Oversubscription Scenarios](#creating-oversubscription-scenarios)
3. [Core Evaluation Metrics](#core-evaluation-metrics)
4. [Hardware and Software Environments](#hardware-and-software-environments)
5. [Common Benchmarks by Access Pattern](#common-benchmarks-by-access-pattern)
6. [Widely-Used Benchmark Suites](#widely-used-benchmark-suites)
7. [Best Practices](#best-practices)
8. [References](#references)

---

## Overview

This document summarizes common methodologies used in UVM research and performance evaluation, based on practices from:

- **SUV** (Smart UVM) - CSA IISc Bangalore research
- **ETC** (Efficient Transfer Clustering) - People at Pitt
- **SC'21** UVM performance analysis - Tallendev
- **NVIDIA Developer** guidelines and best practices

### Key Questions Addressed

1. **How to create oversubscription scenarios?**
2. **What metrics should be measured?**
3. **Which benchmarks cover different memory access patterns?**
4. **What are the standard benchmark suites?**

---

## Creating Oversubscription Scenarios

### Method: Artificial HBM/GDDR Capacity Limitation

The most common approach is to **artificially limit available GPU memory** to force oversubscription.

### Standard Oversubscription Levels

| Oversubscription % | Available Memory | Common Use |
|-------------------|-----------------|------------|
| **15%** | 85% of GPU RAM | Light oversubscription |
| **30%** | 70% of GPU RAM | Moderate oversubscription |
| **50%** | 50% of GPU RAM | Heavy oversubscription |

### Implementation Approaches

#### 1. **CUDA Context API Limits** (Software)
```c
cudaDeviceSetLimit(cudaLimitStackSize, limited_size);
```

#### 2. **Memory Pre-allocation** (Simple)
```c
// Pre-allocate "wasted" memory to reduce available pool
void* waste;
size_t waste_size = total_gpu_memory * oversubscription_ratio;
cudaMalloc(&waste, waste_size);
// Don't free until test completes
```

#### 3. **UVM Hints and Limits** (Precise)
```c
cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
```

#### 4. **Kernel Module Parameters** (System-level)
```bash
# Limit via nvidia driver parameters
echo "limited_size" > /sys/module/nvidia/parameters/MemoryPoolSize
```

### Observation Points

When running the same workload under different oversubscription levels, observe:

- ✅ Performance degradation
- ✅ Page fault frequency
- ✅ PCIe/NVLink traffic volume
- ✅ Thrashing behavior
- ✅ Driver overhead changes

**Example Results (from SUV):**
- At 50% oversubscription: PCIe traffic reduced to ~23% of baseline
- Performance can degrade 100× in worst-case scenarios (NVIDIA warning)

---

## Core Evaluation Metrics

Research papers consistently report these five categories of metrics:

### 1. End-to-End Runtime and Speedup

**Metric:**
```
Speedup = T_baseline / T_optimized
Geometric Mean Speedup = (∏ speedup_i)^(1/n)
```

**Why Geometric Mean?**
- Handles wide performance variance
- Standard in SPEC/MLPerf benchmarks
- Avoids skew from outliers

**Typical Reporting:**
```
Baseline: No UVM optimizations
Optimized: With prefetch/eviction policies
Report: Geometric mean across all benchmarks
```

### 2. Page Fault Count and Batch Size

**Key Findings from SC'21:**

| Batch Size | Benefit | Trade-off |
|------------|---------|-----------|
| **Small (1-8)** | Low latency | High driver overhead |
| **Medium (64-256)** | Balanced | Optimal for most workloads |
| **Large (512+)** | Amortized overhead | DMA mapping serialization |

**Measured Metrics:**
- Page faults per kernel launch
- Average batch size
- Fault servicing latency
- Driver overhead percentage

**Formula:**
```
Overhead_ratio = Time_driver / Time_total
Amortization = Fault_count / Batch_count
```

### 3. PCIe/NVLink Transfer Volume

**What to Measure:**

| Metric | Description | Tool |
|--------|-------------|------|
| **Total Bytes Transferred** | Host↔Device traffic | `nvidia-smi dmon -s u` |
| **Remote Read/Write Count** | Number of remote access operations | UVM stats |
| **Bandwidth Utilization** | % of theoretical peak | nvprof/nsys |
| **Transfer Patterns** | Sequential vs random | Timeline analysis |

**Example (SUV Results):**
```
Baseline: 100% PCIe traffic
With TBP (Transfer Batch Prefetch): 23% PCIe traffic
Reduction: 77% fewer bytes transferred
```

### 4. Eviction and Prefetch Event Statistics

**Events to Track:**

#### Prefetch Events
```bash
# Check UVM prefetch stats
cat /proc/driver/nvidia-uvm/stats | grep prefetch
```

**Metrics:**
- Prefetch activation rate (% of memory accesses)
- Prefetch accuracy (useful prefetches / total prefetches)
- Prefetch range size (bytes per prefetch operation)

#### Eviction Events
```bash
# Check eviction statistics
cat /proc/driver/nvidia-uvm/stats | grep evict
```

**Metrics:**
- Eviction frequency
- Evicted page reuse rate (thrashing indicator)
- Cooperative eviction effectiveness

**Thrashing Detection:**
```
Thrashing_score = Pages_evicted_and_refaulted / Total_pages_evicted
If Thrashing_score > 0.3: High thrashing detected
```

### 5. UVM Driver Path Time Breakdown

**Components to Profile (from SC'21):**

| Component | Typical % | Description |
|-----------|-----------|-------------|
| **DMA Mapping** | 30-40% | Map pages for DMA transfer |
| **VABlock State Init** | 20-30% | Virtual address block initialization |
| **CPU Unmap** | 15-25% | Unmap pages from CPU page table |
| **TLB Invalidation** | 10-15% | Invalidate TLB entries |
| **Lock Contention** | 5-10% | Synchronization overhead |

**Profiling Command:**
```bash
nsys profile --trace=cuda,nvtx,osrt \
  --sample=cpu \
  --backtrace=dwarf \
  ./benchmark
```

**Analysis:**
```python
# Look for UVM driver symbols in timeline
uvm_migrate_*
uvm_populate_*
uvm_va_block_*
```

---

## Hardware and Software Environments

### Typical Hardware Configurations

#### 1. **Single GPU + PCIe (Most Common)**

**Why?**
- PCIe is the bottleneck in oversubscription scenarios
- Easier to control and measure
- Representative of most deployment scenarios

**Common Setup:**
```
GPU: NVIDIA Ampere/Lovelace/Hopper
PCIe: Gen3 x16 or Gen4 x16
CPU: Modern x86_64 (sufficient bandwidth)
RAM: 2-4× GPU memory (to handle oversubscription)
```

**Example Platforms:**
- NVIDIA A100 (40GB/80GB) + PCIe Gen4
- NVIDIA H100 (80GB) + PCIe Gen5
- NVIDIA RTX 4090 (24GB) + PCIe Gen4

#### 2. **NVLink/Multi-GPU (Advanced Studies)**

**Why?**
- Higher bandwidth (900 GB/s vs ~64 GB/s PCIe Gen4)
- Different bottleneck characteristics
- Multi-GPU migration patterns

**Common Setup:**
```
GPUs: 2-8× Tesla/A100/H100
Interconnect: NVLink 2.0/3.0/4.0
Topology: All-to-all or tree
```

#### 3. **Grace-Hopper Superchip (Future)**

**Why?**
- Coherent memory between CPU and GPU
- NVLink-C2C (900 GB/s)
- Different UVM characteristics

### Software Environment

#### UVM Driver Versions

**Open-Source UVM Branch:**
- Many research papers (SUV, SC'21) use open-source NVIDIA kernel modules
- Repository: https://github.com/NVIDIA/open-gpu-kernel-modules
- Benefits: Instrumentation, custom policies, detailed profiling

**Proprietary Driver:**
- Standard NVIDIA driver releases
- More stable but less customizable
- Used for production benchmarking

**Version Considerations:**
```
Driver 470.x: CUDA 11.4 - Basic UVM
Driver 515.x: CUDA 11.7 - Improved access counters
Driver 525.x: CUDA 12.0 - Better thrashing detection
Driver 535.x: CUDA 12.2 - Enhanced prefetch
Driver 550.x+: CUDA 12.4+ - Latest optimizations
```

#### Kernel and OS

**Recommended:**
```
Kernel: 5.15+ (better HMM support)
OS: Ubuntu 20.04/22.04, RHEL 8/9
Compiler: GCC 9-11, NVCC matching CUDA version
```

### NVIDIA Guidelines on Variability

**Warning from NVIDIA Developer:**
> "Oversubscription performance can vary by 100× depending on:
> - Access patterns (regular vs irregular)
> - UVM hint/strategy combinations
> - Dataset size relative to GPU memory"

**Implication:**
Rigorous evaluation must cover:
- ✅ Regular AND irregular access patterns
- ✅ Multiple dataset sizes (0.5×, 1×, 2×, 5× GPU memory)
- ✅ Multiple oversubscription levels (15%, 30%, 50%+)
- ✅ Different UVM parameter configurations

---

## Common Benchmarks by Access Pattern

### 1. Matrix/Linear Algebra & Stencil (Regular, Predictable)

**Why These?**
- Highly regular memory access
- "Magnifying glass" for TBP, batch size, driver overhead
- Used in ETC and SC'21 sensitivity analysis

**Common Benchmarks:**

#### GEMM (General Matrix Multiply)
```c
// C = alpha * A * B + beta * C
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            m, n, k, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc);
```

**Characteristics:**
- Sequential column-major or row-major access
- High arithmetic intensity
- Benefits greatly from prefetching

#### 2D/3D Convolution
```c
// Stencil pattern: each output depends on NxN neighborhood
for (int i = 1; i < n-1; i++)
  for (int j = 1; j < m-1; j++)
    output[i][j] = stencil(input[i-1:i+1][j-1:j+1]);
```

**Characteristics:**
- Halo region communication
- Spatial locality
- Regular access pattern

#### Other Linear Algebra Kernels
- **ATAX** (Matrix-vector multiply and transpose)
- **MVT** (Matrix-vector product and transpose)
- **2MM/3MM** (Multiple matrix multiplications)
- **FDTD** (Finite-difference time-domain)

**Usage in Research:**
- ETC: Used for TBP effectiveness analysis
- SC'21: Used for batch size sensitivity graphs
- SUV: Used for prefetch threshold tuning

### 2. Graph and Irregular Access Patterns

**Why These?**
- Trigger extreme prefetch/eviction behavior
- Test adaptive migration/threshold strategies
- Expose thrashing conditions

**Common Benchmarks:**

#### BFS (Breadth-First Search)
```c
// Irregular: neighbors vary per vertex
for each vertex v in frontier:
  for each neighbor n of v:
    if not visited[n]:
      visit(n)
      add_to_next_frontier(n)
```

**Access Pattern:**
- Random memory access
- Unpredictable prefetch effectiveness
- Tests thrashing mitigation

#### B+ Tree Traversal
```c
// Pointer chasing with unpredictable paths
node = root;
while (!node.isLeaf) {
  child_idx = search(node.keys, key);
  node = node.children[child_idx];  // Random access
}
```

**Characteristics:**
- Pointer chasing
- Poor spatial locality
- Challenges prefetchers

#### Sparse Matrix Operations (SpMV)
```c
// Compressed Sparse Row format
for (int i = 0; i < nrows; i++) {
  for (int j = row_ptr[i]; j < row_ptr[i+1]; j++) {
    y[i] += values[j] * x[col_idx[j]];  // Irregular access to x[]
  }
}
```

**Characteristics:**
- Irregular column access
- Sparse data structure
- Variable row lengths

**Usage in Research:**
- People at Pitt (ETC): Used to validate adaptive eviction
- SC'21: Used for worst-case thrashing analysis

### 3. Real/Proxy HPC Applications

**Why These?**
- Multi-level memory hierarchy
- Complex control flow
- Validate that "strategies work on realistic codes"

**Common Proxy Applications:**

#### HPGMG (High-Performance Geometric Multigrid)
```c
// Multigrid V-cycle
for level in range(finest, coarsest):
  smooth(level)           // Regular access
  restrict(level)         // Strided access
for level in range(coarsest, finest):
  interpolate(level)      // Strided access
  smooth(level)           // Regular access
```

**Characteristics:**
- Multiple grid levels
- Mix of regular and strided access
- SC'21 timeline analysis shows UVM behavior

#### Gauss-Seidel Iterative Solver
```c
// Iterative update with dependencies
for iter in range(max_iters):
  for i in range(1, n-1):
    for j in range(1, m-1):
      x[i][j] = f(x[i-1][j], x[i+1][j], x[i][j-1], x[i][j+1])
```

**Characteristics:**
- Sequential dependencies
- Regular stencil pattern
- Used in SC'21 prefetch/eviction timeline analysis

#### XSBench (Monte Carlo Particle Transport)
```c
// Random energy lookups in cross-section data
for particle in particles:
  energy = random()
  xs = lookup_cross_section(energy)  // Random access
  // ... Monte Carlo step
```

**Characteristics:**
- Random memory access
- Memory subsystem stress test
- Proxy for nuclear reactor simulations

**Usage in Research:**
- SC'21: XSBench used as memory pressure proxy
- HPGMG: Multi-level memory behavior analysis

### 4. Deep Learning Inference/Training Microbenchmarks

**Why These?**
- Increasingly important workload
- Mix of regular (convolution) and irregular (sparse) access
- Real-world UVM use case

#### Tango (DNN Inference Small Networks)

**What is Tango?**
- Collection of small DNN kernels
- Pure CUDA/OpenCL (no framework overhead)
- Easy to instrument for UVM evaluation

**Example Kernels:**
```c
// Conv2D layer (regular)
conv2d<<<grid, block>>>(input, weights, output);

// Fully-connected (regular)
gemm<<<grid, block>>>(input, weights, output);

// Activation (element-wise)
relu<<<grid, block>>>(input, output);
```

**Characteristics:**
- Mix of compute and memory bound
- Regular access patterns
- Used in SUV for DNN workload evaluation

#### Custom DNN Microbenchmarks

**Common Operations:**
- Matrix multiply (FC layers)
- Convolution (CONV layers)
- Batch normalization
- Pooling (max/avg)
- Sparse attention (irregular)

**Usage in Research:**
- SUV: Included DNN workloads in benchmark suite
- Focus on inference (lower memory, more oversubscription potential)

---

## Widely-Used Benchmark Suites

These suites are repeatedly used in UVM/oversubscription/migration research papers.

### 1. PolyBench/GPU (PolyBench-ACC)

**Source:** https://github.com/cavazos-lab/PolybenchGPU

**Description:**
- GPU versions of PolyBench kernels
- "Static Control Parts (SCoP)" - analyzable loop nests
- 30+ kernels from linear algebra, image processing, physics simulation

**Categories:**

| Category | Kernels | Access Pattern |
|----------|---------|---------------|
| **Linear Algebra** | GEMM, GEMVER, GESUMMV, SYMM, SYRK, SYR2K, TRMM | Regular, sequential |
| **Stencil** | 2DCONV, 3DCONV, FDTD-2D, JACOBI-1D/2D, SEIDEL-2D | Regular, halo |
| **Data Mining** | CORRELATION, COVARIANCE, ATAX, BICG, MVT | Regular, strided |

**Why Used in UVM Research?**
- ✅ Regular access patterns (good for prefetch analysis)
- ✅ Parameterizable sizes (easy to create oversubscription)
- ✅ Well-understood performance characteristics
- ✅ Standard in compiler/architecture research

**Usage Examples:**
- ETC: Used for access pattern sensitivity analysis
- SC'21: Used for driver overhead breakdown
- Multiple studies: Baseline performance comparisons

**Build and Run:**
```bash
git clone https://github.com/cavazos-lab/PolybenchGPU
cd PolybenchGPU
make
# Run individual benchmarks
./2DCONV -s LARGE
./GEMM -s EXTRALARGE
```

### 2. Rodinia Benchmark Suite

**Source:** http://rodinia.cs.virginia.edu/

**Description:**
- Diverse applications from various domains
- Both regular and irregular memory access
- OpenCL and CUDA versions available

**Key Benchmarks:**

| Benchmark | Domain | Access Pattern |
|-----------|--------|---------------|
| **BFS** | Graph | Irregular |
| **Hotspot** | Physics | Regular stencil |
| **Needleman-Wunsch** | Bioinformatics | Dynamic programming |
| **Pathfinder** | Grid traversal | Regular |
| **SRAD** | Image processing | Stencil |
| **LUD** | Linear algebra | Blocked |

**Why Used in UVM Research?**
- ✅ Mix of regular and irregular
- ✅ Real application domains
- ✅ Widely cited baseline
- ✅ UVM versions available (see UVM_benchmark)

### 3. Parboil Benchmark Suite

**Source:** http://impact.crhc.illinois.edu/parboil/parboil.aspx

**Description:**
- UIUC IMPACT research group
- Focus on throughput computing
- Multiple implementations (CUDA, OpenCL, OpenMP)

**Key Benchmarks:**
- **SPMV** - Sparse matrix-vector multiply (irregular)
- **STENCIL** - 3D stencil computation (regular)
- **TPACF** - Two-point angular correlation (astronomy)

### 4. SHOC (Scalable HeterOgeneous Computing)

**Source:** https://github.com/vetter/shoc

**Description:**
- Oak Ridge National Lab
- Focus on memory bandwidth and latency
- Good for UVM stress testing

**Key Microbenchmarks:**
- **BusBandwidth** - PCIe/NVLink bandwidth
- **DeviceMemory** - GPU memory bandwidth
- **MaxFlops** - Compute capability

### 5. CUDA Samples

**Source:** NVIDIA CUDA Toolkit samples

**Relevant for UVM:**
- **simpleCUDA2GL** - UVM with OpenGL interop
- **UnifiedMemoryStreams** - UVM with CUDA streams
- **vectorAdd** - Basic UVM example

### 6. Custom Research Suites

#### UVMBench (This Work)
**Source:** https://github.com/eunomia-bpf/UVM_benchmark

**Description:**
- Specifically designed for UVM evaluation
- Mix of ML, graph, and HPC benchmarks
- Three versions: UVM, UVM-oversub, non-UVM

#### SUV Benchmark Suite
**Components:**
- Microbenchmarks (GEMM, CONV, etc.)
- Graph algorithms (BFS, B+Tree)
- DNN inference kernels
- Custom memory access patterns

---

## Best Practices

### Experimental Design

#### 1. **Multi-Dimensional Parameter Sweep**

**Dimensions to Vary:**
```
1. Dataset Size: [0.5×, 1×, 2×, 5× GPU memory]
2. Oversubscription: [0%, 15%, 30%, 50%, 75%]
3. Access Pattern: [Sequential, Random, Mixed]
4. UVM Parameters: [Prefetch on/off, Threshold variations]
```

**Example Matrix:**
```
For each benchmark:
  For each dataset_size in sizes:
    For each oversub_level in levels:
      For each uvm_config in configs:
        Run(benchmark, dataset_size, oversub_level, uvm_config)
        Record(metrics)
```

#### 2. **Statistical Rigor**

**Required:**
- ✅ Minimum 5 runs per configuration
- ✅ Report median + std deviation or confidence intervals
- ✅ Test for statistical significance
- ✅ Identify and explain outliers

**Example:**
```python
import numpy as np
from scipy import stats

results = [run_benchmark() for _ in range(10)]
median = np.median(results)
std = np.std(results)
ci_95 = stats.t.interval(0.95, len(results)-1,
                         loc=median,
                         scale=stats.sem(results))
```

#### 3. **Baseline Comparison**

**Always Compare Against:**
1. **No UVM** - Explicit cudaMemcpy
2. **UVM Default** - No hints or prefetch
3. **UVM Optimized** - With your optimization

**Reporting:**
```
Speedup vs No-UVM:     X.XX×
Speedup vs UVM-Default: X.XX×
```

### Measurement Best Practices

#### 1. **Warmup Runs**
```c
// Warmup to populate caches and driver state
for (int i = 0; i < 3; i++) {
  kernel<<<...>>>();
  cudaDeviceSynchronize();
}

// Actual measurement
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start);

kernel<<<...>>>();

cudaEventRecord(stop);
cudaEventSynchronize(stop);
float ms;
cudaEventElapsedTime(&ms, start, stop);
```

#### 2. **Isolate System Effects**
```bash
# Disable CPU frequency scaling
sudo cpupower frequency-set -g performance

# Disable GPU boost
sudo nvidia-smi -pm 1  # Persistence mode
sudo nvidia-smi -lgc 1410,1410  # Lock GPU clock

# Minimize background processes
sudo systemctl stop <unnecessary services>
```

#### 3. **Monitor During Test**
```bash
# Terminal 1: Run benchmark
./benchmark

# Terminal 2: Monitor GPU
nvidia-smi dmon -s ucm -i 0

# Terminal 3: Monitor UVM
watch -n 1 'cat /proc/driver/nvidia-uvm/stats'
```

### Reporting Best Practices

#### Required Plots/Tables

1. **Performance vs Oversubscription**
   - X-axis: Oversubscription level (0%, 15%, 30%, 50%)
   - Y-axis: Speedup or runtime
   - Lines: Different benchmarks or configurations

2. **Metric Breakdown**
   - Stacked bar chart showing driver time components
   - Page fault counts per benchmark
   - PCIe traffic volume comparison

3. **Sensitivity Analysis**
   - Heatmap: benchmark × UVM parameter → performance
   - Identify which benchmarks are sensitive to which parameters

#### Required Tables

**Summary Table Example:**
| Benchmark | No-UVM (ms) | UVM-Default (ms) | UVM-Opt (ms) | Speedup |
|-----------|-------------|------------------|--------------|---------|
| GEMM      | 10.2 ± 0.3  | 15.4 ± 0.5       | 11.1 ± 0.2   | 1.39×   |
| BFS       | 45.2 ± 1.2  | 180.3 ± 5.4      | 52.1 ± 1.8   | 3.46×   |
| ...       | ...         | ...              | ...          | ...     |

---

## References

### Academic Papers

1. **SUV (Smart UVM)**
   - CSA - IISc Bangalore
   - Focus: Intelligent prefetch and eviction policies
   - Key contribution: 77% PCIe traffic reduction at 50% oversubscription

2. **ETC (Efficient Transfer Clustering)**
   - People at Pitt
   - Focus: Transfer batching and clustering
   - Key contribution: Adaptive thrashing mitigation

3. **SC'21 UVM Performance Analysis**
   - Tallendev
   - Focus: Driver overhead breakdown
   - Key contribution: DMA mapping and VABlock initialization analysis

### NVIDIA Resources

- [Unified Memory Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#unified-memory-programming)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [UVM Developer Blog](https://developer.nvidia.com/blog/unified-memory-cuda-beginners/)

### Benchmark Suites

- [PolyBench/GPU](https://github.com/cavazos-lab/PolybenchGPU)
- [Rodinia](http://rodinia.cs.virginia.edu/)
- [UVMBench](https://github.com/eunomia-bpf/UVM_benchmark)
- [Parboil](http://impact.crhc.illinois.edu/parboil/)
- [SHOC](https://github.com/vetter/shoc)

---

## Summary Checklist

When designing a UVM evaluation study, ensure:

- [ ] Multiple oversubscription levels tested (15%, 30%, 50%)
- [ ] Mix of regular and irregular access patterns
- [ ] Dataset sizes relative to GPU memory (0.5×, 1×, 2×, 5×)
- [ ] Statistical rigor (multiple runs, confidence intervals)
- [ ] Standard benchmarks (PolyBench, Rodinia, or equivalent)
- [ ] Core metrics reported (runtime, page faults, PCIe traffic, driver overhead)
- [ ] UVM parameter variations tested
- [ ] Baseline comparisons (no-UVM, UVM-default)
- [ ] System configuration documented (GPU, driver, kernel version)
- [ ] Reproducibility information provided (code, parameters, scripts)

---

**Document Version:** 1.0
**Date:** 2025-11-11
**Related Documents:**
- [UVM Tuning Guide](./NVIDIA_UVM_TUNING_GUIDE.md)
- [UVM Benchmark Results](./UVM_BENCHMARK_RESULTS.md)
- [UVM Test Process](./UVM_BENCHMARK_TEST_PROCESS.md)
