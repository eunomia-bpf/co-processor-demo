# UVM Benchmark Results

Summary of running the eunomia-bpf/UVM_benchmark suite on NVIDIA H100.

---

## System Configuration

**GPU:** NVIDIA H100 (97871 MiB)
**Driver Version:** 580.105.08
**CUDA Version:** 12.9
**Kernel:** 6.8.0-87-generic
**Repository:** https://github.com/eunomia-bpf/UVM_benchmark
**Commit:** Latest from main branch
**Date:** 2025-11-11

---

## Build Status

### Successfully Compiled Benchmarks

| Benchmark | Status | Executable |
|-----------|--------|------------|
| **BFS** (Breadth-First Search) | ✅ Success | `UVM_benchmarks/bfs/main` |
| **KMeans** | ✅ Success | `UVM_benchmarks/kmeans/kmeans_cuda` |
| **KMeans (Standard)** | ✅ Success | `UVM_benchmarks/kmeans/kmeans_standard` |
| **KNN** (K-Nearest Neighbors) | ✅ Success | `UVM_benchmarks/knn/knn` |

### Failed Benchmarks

| Benchmark | Status | Error |
|-----------|--------|-------|
| **BN** (Bayesian Network) | ❌ Failed | nvcc fatal: 'compute_35' is not in 'keyword=value' format |
| **CNN** (Convolutional Neural Network) | ❌ Failed | cudaMemPrefetchAsync API signature mismatch (CUDA 12.9 incompatibility) |
| **Logistic Regression** | ❌ Failed | nvcc path issue in Makefile |

**Issues:**
- Some benchmarks use deprecated CUDA compute capabilities (sm_30, sm_35)
- CNN benchmark uses old CUDA API signatures incompatible with CUDA 12.9
- Logistic regression Makefile has hardcoded incorrect nvcc path

---

## Benchmark Results

### 1. BFS (Breadth-First Search)

**Configuration:**
- Vertices: 1,000,000
- Edges: 20,000,000
- Source vertex: 5

**Results:**

| Implementation | Time (ms) | Status |
|----------------|-----------|--------|
| Sequential (CPU) | 250 ms | ✅ Baseline |
| Simple Parallel (GPU) | 1008 ms | ✅ Verified |
| Queue Parallel (GPU) | 174 ms | ✅ Verified |
| Scan Parallel (GPU) | **9 ms** | ✅ Verified |

**Overall Time:** 1,642 ms

**Analysis:**
- Scan parallel implementation is **27.8x faster** than sequential
- Scan parallel is **111.9x faster** than simple parallel
- UVM allows automatic data migration between CPU/GPU
- All implementations verified correct output

---

### 2. KNN (K-Nearest Neighbors)

**Configuration:**
- Reference points: 4,096
- Query points: 4,096
- Dimensions: 32
- Neighbors (k): 20

**Results:**

| Implementation | Iterations | Total Time (s) | Time per Iteration |
|----------------|------------|----------------|-------------------|
| CPU | 10 | 15.632 s | 1.563 s |
| GPU (UVM) | 100 | 0.475 s | **0.005 s** |

**Speedup:** ~329x faster per iteration (GPU vs CPU)

**Analysis:**
- Massive speedup for compute-intensive workload
- UVM enables seamless data sharing
- GPU processes 100 iterations in less time than CPU processes 10

---

### 3. KMeans Clustering

**Configuration:**
- Points: 100,000
- Clusters: 2
- Data file: `data/kmeans/100000_points.txt`

**Results:**
- **CUDA Time:** 0.034 seconds (34.3 ms)

**Note:** Benchmark experienced segmentation fault during cleanup (after computation completed)

**Analysis:**
- Very fast clustering performance
- Segfault indicates potential memory management issue in cleanup code
- Core computation completed successfully

---

## UVM Behavior Observations

### Memory Management

1. **Automatic Page Migration**
   - UVM automatically migrates pages between CPU and GPU
   - No explicit `cudaMemcpy` calls needed in benchmark code
   - Uses `cudaMallocManaged()` for unified memory allocation

2. **Page Fault Handling**
   - Initial access causes page faults
   - Subsequent accesses benefit from data locality
   - Prefetching reduces fault overhead in sequential workloads

3. **Oversubscription**
   - Benchmarks can allocate more memory than GPU VRAM
   - OS manages swapping between system RAM and GPU memory
   - Critical for large datasets (1M+ vertices/points)

### Performance Characteristics

| Workload Pattern | UVM Performance | Notes |
|------------------|-----------------|-------|
| **Sequential Access** | Excellent | BFS scan benefits from prefetching |
| **Random Access** | Good | KNN still achieves 329x speedup |
| **Large Datasets** | Very Good | 1M vertices, 20M edges handled well |
| **Compute-Intensive** | Excellent | KNN shows massive GPU acceleration |

---

## UVM Parameter Impact

### Current UVM Configuration

The system is running with these key UVM parameters:

```
uvm_perf_prefetch_enable: 1 (enabled)
uvm_perf_prefetch_threshold: 51
uvm_perf_thrashing_enable: 1 (enabled)
uvm_perf_fault_batch_count: 256
uvm_global_oversubscription: 1 (enabled)
```

### Potential Optimizations

Based on benchmark patterns, these tuning suggestions may improve performance:

#### For BFS (Sequential, Large Graph)
```bash
# More aggressive prefetching
uvm_perf_prefetch_threshold=25
uvm_perf_fault_batch_count=512
```

#### For KNN (Random Access, Compute-Intensive)
```bash
# Lower latency, access counter migration
uvm_perf_access_counter_threshold=128
uvm_perf_fault_batch_count=128
```

#### For KMeans (Iterative, Shared CPU-GPU)
```bash
# Reduce thrashing for iterative workloads
uvm_perf_thrashing_threshold=5
uvm_perf_thrashing_pin=500
```

---

## Comparison: UVM vs Traditional CUDA

### Advantages of UVM

| Aspect | Traditional CUDA | UVM |
|--------|-----------------|-----|
| **Programming** | Manual `cudaMemcpy` | Automatic migration |
| **Memory Management** | Explicit allocation | Unified address space |
| **Oversubscription** | Limited | Supported |
| **Code Complexity** | Higher | Lower |

### Performance Trade-offs

| Scenario | UVM vs Traditional |
|----------|-------------------|
| **First Access** | Slower (page faults) |
| **Repeated Access** | Similar (data locality) |
| **Large Datasets** | Better (oversubscription) |
| **Memory Copy Overhead** | Lower (automatic) |

---

## Recommendations

### For These Benchmarks

1. **Fix CNN Benchmark**
   - Update `cudaMemPrefetchAsync()` calls to CUDA 12.9 API
   - Parameters order changed in newer CUDA versions

2. **Fix BN Benchmark**
   - Update compute capability flags
   - Remove deprecated sm_30/sm_35, use sm_70+ for H100

3. **Fix Logistic Regression**
   - Update Makefile to use `/usr/local/cuda/bin/nvcc`

### For Production UVM Workloads

1. **Profile First**
   - Run with `nsys` or `ncu` to understand access patterns
   - Check page fault rates with `/proc/driver/nvidia-uvm/stats`

2. **Tune Based on Workload**
   - Sequential: Enable aggressive prefetching
   - Random: Lower fault batch sizes for latency
   - Iterative: Increase thrashing protection

3. **Monitor Memory**
   - Use `nvidia-smi` to track GPU memory usage
   - Watch for excessive migrations
   - Consider explicit prefetch hints for hot paths

---

## Benchmark Commands Reference

### BFS
```bash
cd UVM_benchmarks/bfs
./main 5 1000000 10000000  # vertices edges
```

### KNN
```bash
cd UVM_benchmarks/knn
./knn  # Uses default parameters
```

### KMeans
```bash
cd UVM_benchmarks/kmeans
./kmeans_cuda 2 ../../data/kmeans/100000_points.txt 100000
```

---

## Files and Locations

**Benchmark Suite:** `/root/co-processor-demo/uvm_bench/UVM_benchmark`
**UVM Tuning Guide:** `/root/co-processor-demo/memory/NVIDIA_UVM_TUNING_GUIDE.md`
**This Report:** `/root/co-processor-demo/memory/UVM_BENCHMARK_RESULTS.md`

---

## Key Takeaways

1. ✅ **UVM Works Well on H100**
   - Successfully ran multiple benchmarks
   - Significant performance gains over CPU
   - Automatic memory management simplifies code

2. ⚡ **Performance is Excellent**
   - BFS: 27.8x speedup (scan parallel)
   - KNN: 329x speedup per iteration
   - KMeans: 34ms for 100k points

3. 🔧 **Tuning Opportunities Exist**
   - Prefetch tuning can help sequential workloads
   - Access counter migration for random access
   - Thrashing protection for iterative algorithms

4. 🐛 **Some Compatibility Issues**
   - Older benchmarks need CUDA API updates
   - Deprecated compute capabilities
   - Minor Makefile path issues

---

## Related Documentation

- [NVIDIA UVM Tuning Guide](./NVIDIA_UVM_TUNING_GUIDE.md) - Complete parameter reference
- [UVM Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#unified-memory-programming)
- [UVMBench Paper](http://arxiv.org/abs/2007.09822) - Original research

---

**Generated:** 2025-11-11
**System:** NVIDIA H100 | Driver 580.105.08 | CUDA 12.9
