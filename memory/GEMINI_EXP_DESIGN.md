# Evaluation Design for UVM Performance Analysis

**Target Venue:** OSDI (USENIX Symposium on Operating Systems Design and Implementation)

**Design Status:** Draft v0.1 - Section-by-section construction

---

## Document Overview

This document designs a comprehensive evaluation methodology for analyzing Unified Virtual Memory (UVM) performance optimizations at the OSDI level. The evaluation framework includes:

1. **Research Questions (RQs)** - Key hypotheses to validate
2. **Experimental Testbed Design** - Automated Python-based framework
3. **Benchmark Selection** - Representative workloads across access patterns
4. **Metrics and Measurements** - Quantitative performance indicators
5. **Visualization Plan** - Figures and tables for paper
6. **Statistical Analysis** - Rigor and reproducibility

**Document Structure:** This will be built section by section to allow for iterative refinement.

---

## Table of Contents

1. [Research Questions](#1-research-questions)
2. [Experimental Methodology](#2-experimental-methodology)
3. [Testbed Architecture](#3-testbed-architecture)
4. [Benchmark Suite Design](#4-benchmark-suite-design)
5. [Metrics and Measurements](#5-metrics-and-measurements)
6. [Experimental Setup](#6-experimental-setup)
7. [Data Collection Framework](#7-data-collection-framework)
8. [Visualization and Presentation](#8-visualization-and-presentation)
9. [Statistical Analysis Plan](#9-statistical-analysis-plan)
10. [Reproducibility Package](#10-reproducibility-package)

---

## 1. Research Questions

### 1.1 Primary Research Questions

The evaluation is designed to answer the following research questions:

#### **RQ1: Performance Under Oversubscription**

> **How does UVM performance degrade under different levels of memory oversubscription, and can intelligent prefetching and eviction policies mitigate this degradation?**

**Hypothesis:** 
- H1.1: Performance degrades super-linearly with oversubscription (worse than O(n))
- H1.2: Intelligent policies can reduce degradation to near-linear
- H1.3: Impact varies significantly by access pattern (regular vs irregular)

**Measurements:**
- End-to-end execution time at 0%, 15%, 30%, 50%, 75% oversubscription
- Page fault counts and servicing latency
- PCIe/NVLink bandwidth utilization

**Success Criteria:**
- Geometric mean speedup ≥ 2.5× over baseline UVM at 50% oversubscription
- Performance degradation < 3× compared to no-oversubscription case

---

#### **RQ2: Access Pattern Sensitivity**

> **How do different memory access patterns (sequential, random, strided, irregular) respond to UVM optimizations?**

**Hypothesis:**
- H2.1: Sequential patterns benefit most from prefetching (5-10× improvement)
- H2.2: Random patterns benefit from intelligent eviction (2-3× improvement)
- H2.3: Mixed patterns require adaptive strategies

**Measurements:**
- Speedup per access pattern category
- Prefetch accuracy (useful prefetches / total prefetches)
- Cache hit rates and miss penalties

**Success Criteria:**
- Speedup ≥ 5× for sequential workloads
- Speedup ≥ 2× for random workloads
- No regression for any workload type

---

#### **RQ3: Driver Overhead Breakdown**

> **What are the dominant bottlenecks in the UVM driver path, and how do optimizations reduce this overhead?**

**Hypothesis:**
- H3.1: DMA mapping dominates (30-40% of driver time)
- H3.2: Batch processing reduces per-fault overhead by 10-50×
- H3.3: Lock contention is significant (10-15%) in multi-threaded workloads

**Measurements:**
- Time spent in each driver component (DMA, VABlock, TLB, locks)
- Fault batch size distribution
- Amortization effectiveness

**Success Criteria:**
- Reduce DMA mapping overhead by ≥ 30%
- Increase average batch size by ≥ 2×
- Reduce total driver overhead to < 15% of execution time

---

#### **RQ4: Thrashing Detection and Mitigation**

> **Can we effectively detect and mitigate memory thrashing in CPU-GPU collaborative workloads?**

**Hypothesis:**
- H4.1: Thrashing is detectable through eviction/refault patterns
- H4.2: Page pinning reduces thrashing by 3-5×
- H4.3: Adaptive thresholds outperform static policies

**Measurements:**
- Thrashing score (refaulted pages / evicted pages)
- Page ping-pong frequency
- Pin duration effectiveness

**Success Criteria:**
- Thrashing score < 0.2 (down from > 0.5 baseline)
- ≥ 3× reduction in page migrations for iterative workloads
- No more than 5% overhead for non-thrashing workloads

---

#### **RQ5: Scalability with Dataset Size**

> **How do UVM optimizations scale with increasing dataset sizes (0.5×, 1×, 2×, 5× GPU memory)?**

**Hypothesis:**
- H5.1: Benefits increase with dataset size (more oversubscription)
- H5.2: Performance remains within 2× of baseline when dataset ≤ 1× GPU memory
- H5.3: Optimizations maintain effectiveness even at 5× oversubscription

**Measurements:**
- Performance curves vs dataset size
- Memory transfer volume vs size
- Transition point where oversubscription becomes critical

**Success Criteria:**
- < 2× slowdown at 2× GPU memory
- < 5× slowdown at 5× GPU memory
- Maintain speedup advantage across all sizes

---

### 1.2 Secondary Research Questions

#### **RQ6: Parameter Sensitivity Analysis**

> **How sensitive is performance to UVM tuning parameters (prefetch threshold, batch size, pin duration)?**

**Goal:** Identify optimal parameter ranges and guide auto-tuning strategies

**Key Parameters to Vary:**
- `uvm_perf_prefetch_threshold`: [10, 25, 51, 75, 100]
- `uvm_perf_fault_batch_count`: [64, 128, 256, 512, 1024]
- `uvm_perf_thrashing_pin`: [100, 300, 500, 1000, 2000] ms
- `uvm_perf_access_counter_threshold`: [64, 128, 256, 512, 1024]

**Analysis:**
- Heatmaps showing parameter × benchmark performance
- Identify parameter interactions (e.g., batch size + prefetch threshold)
- Recommend default configurations per workload class

---

#### **RQ7: Hardware Platform Differences**

> **How do UVM optimizations perform across different interconnects (PCIe Gen3, Gen4, Gen5, NVLink)?**

**Motivation:** Bandwidth differences significantly impact migration cost

**Platforms:**
- PCIe Gen3 x16: ~16 GB/s
- PCIe Gen4 x16: ~32 GB/s  
- PCIe Gen5 x16: ~64 GB/s
- NVLink 4.0: ~900 GB/s

**Expected Results:**
- Higher bandwidth reduces migration penalties
- Optimizations more critical on lower-bandwidth platforms
- Prefetching effectiveness varies with bandwidth

---

#### **RQ8: Multi-GPU Scenarios**

> **How do UVM optimizations extend to multi-GPU workloads with peer-to-peer access?**

**Scenarios:**
- 2 GPUs with NVLink
- 4 GPUs with NVLink mesh
- 2 GPUs PCIe-only (no P2P)

**Measurements:**
- Peer migration efficiency
- Load balancing effectiveness
- Scalability vs number of GPUs

---

### 1.3 Research Question Summary Table

| RQ | Category | Priority | Complexity | Expected Figures |
|----|----------|----------|------------|------------------|
| RQ1 | Performance | Critical | Medium | Fig 1, 2, 3 |
| RQ2 | Workload | Critical | Medium | Fig 4, 5 |
| RQ3 | Overhead | Critical | High | Fig 6, 7 |
| RQ4 | Thrashing | High | High | Fig 8, 9 |
| RQ5 | Scalability | High | Medium | Fig 10, 11 |
| RQ6 | Tuning | Medium | Low | Fig 12 |
| RQ7 | Hardware | Low | High | Fig 13 |
| RQ8 | Multi-GPU | Low | Very High | Fig 14 |

**Evaluation Scope for Paper:**
- **Must Include:** RQ1-RQ5 (answers core contribution claims)
- **Should Include:** RQ6 (guides users on tuning)
- **May Include:** RQ7-RQ8 (if space allows, or appendix)

---

## 2. Experimental Methodology

### 2.1 Evaluation Philosophy

**Approach:** Systematic, multi-dimensional evaluation following OSDI best practices

**Key Principles:**
1. **Reproducibility First** - All experiments scripted and automated
2. **Statistical Rigor** - Multiple runs, confidence intervals, outlier analysis
3. **Real Workloads** - Use established benchmarks (PolyBench, Rodinia, UVMBench)
4. **Comprehensive Coverage** - Test diverse access patterns and scales
5. **Fair Comparison** - Consistent baselines, controlled environment

### 2.2 Experimental Dimensions

The evaluation spans multiple dimensions to ensure comprehensive coverage:

#### **Dimension 1: Oversubscription Levels**

```python
OVERSUBSCRIPTION_LEVELS = [
    0,     # No oversubscription (baseline)
    15,    # Light (85% available memory)
    30,    # Moderate (70% available)
    50,    # Heavy (50% available)
    75,    # Extreme (25% available)
]
```

**Implementation Method:** Pre-allocate "waste" memory to reduce available GPU memory

```python
def set_oversubscription(level_percent, total_gpu_memory_bytes):
    """
    Reduce available GPU memory by pre-allocating dummy memory
    
    Args:
        level_percent: Percentage of memory to make unavailable (0-100)
        total_gpu_memory_bytes: Total GPU memory in bytes
        
    Returns:
        Handle to waste allocation (must not be freed during test)
    """
    waste_bytes = int(total_gpu_memory_bytes * (level_percent / 100.0))
    # CUDA code will cudaMalloc this amount
    return waste_bytes
```

---

#### **Dimension 2: Access Patterns**

Classify benchmarks by their memory access characteristics:

| Pattern Class | Characteristics | Example Benchmarks | Expected Benefit |
|--------------|-----------------|-------------------|------------------|
| **Sequential** | Linear, predictable | GEMM, 2DCONV, FDTD | High (prefetch) |
| **Strided** | Regular but non-sequential | Matrix transpose | Medium (prefetch) |
| **Random** | Unpredictable | BFS, B+Tree, SpMV | Low (caching) |
| **Iterative** | Repeated access | KMeans, Jacobi | Medium (thrashing) |
| **Mixed** | Combination | HPGMG, CNN layers | Variable (adaptive) |

---

#### **Dimension 3: Dataset Sizes**

Test relative to GPU memory capacity:

```python
DATASET_SIZE_MULTIPLIERS = [
    0.5,   # Fits easily in GPU memory
    1.0,   # Exactly fills GPU memory
    2.0,   # 2× GPU memory (requires oversubscription)
    5.0,   # 5× GPU memory (heavy oversubscription)
]
```

**Rationale:** 
- 0.5× tests overhead when no migration needed
- 1.0× tests boundary conditions
- 2.0× tests moderate oversubscription effectiveness
- 5.0× tests extreme oversubscription handling

---

#### **Dimension 4: UVM Configurations**

Test multiple policy configurations:

```python
UVM_CONFIGURATIONS = {
    "baseline": {
        "uvm_perf_prefetch_enable": 0,
        "uvm_perf_thrashing_enable": 0,
        "description": "No UVM optimizations"
    },
    "default": {
        "uvm_perf_prefetch_enable": 1,
        "uvm_perf_prefetch_threshold": 51,
        "uvm_perf_thrashing_enable": 1,
        "description": "NVIDIA default settings"
    },
    "aggressive_prefetch": {
        "uvm_perf_prefetch_enable": 1,
        "uvm_perf_prefetch_threshold": 25,
        "uvm_perf_fault_batch_count": 512,
        "description": "Optimized for sequential access"
    },
    "thrashing_resistant": {
        "uvm_perf_thrashing_enable": 1,
        "uvm_perf_thrashing_threshold": 5,
        "uvm_perf_thrashing_pin": 500,
        "description": "Optimized for iterative workloads"
    },
    "proposed": {
        # Your research contribution parameters
        "description": "Your proposed optimization"
    }
}
```

---

### 2.3 Experimental Workflow

**Phase 1: Environment Setup**
1. Verify GPU configuration (nvidia-smi)
2. Check UVM module status
3. Set CPU governor to performance mode
4. Lock GPU clocks to prevent boost fluctuation
5. Clear system caches

**Phase 2: Benchmark Compilation**
1. Build all benchmarks with consistent flags
2. Verify successful compilation
3. Run smoke tests (single iteration, small size)

**Phase 3: Data Collection**
For each combination of (benchmark, oversubscription, dataset_size, uvm_config):
1. Apply UVM configuration
2. Set oversubscription level
3. Run warmup iterations (3×)
4. Run measurement iterations (10×)
5. Collect metrics (time, faults, transfers, etc.)
6. Store raw data to JSON
7. Clean up GPU state

**Phase 4: Statistical Analysis**
1. Compute median and confidence intervals
2. Identify outliers
3. Test for statistical significance
4. Generate summary tables

**Phase 5: Visualization**
1. Generate plots for each RQ
2. Create tables for paper
3. Export high-quality figures (PDF/PNG)

---

### 2.4 Controlled Variables

**Fixed During All Experiments:**
- GPU model (e.g., NVIDIA H100)
- Driver version (e.g., 580.105.08)
- CUDA version (e.g., 12.9)
- Kernel version (e.g., 6.8.0-87)
- CPU governor (performance mode)
- GPU clock frequency (locked)
- No other GPU processes running

**Varied Systematically:**
- Oversubscription level
- Dataset size
- Benchmark/access pattern
- UVM configuration

---

### 2.5 Measurement Techniques

#### **Primary Metrics: End-to-End Performance**

```python
# CUDA event-based timing (most accurate)
cudaEventCreate(&start)
cudaEventCreate(&stop)

cudaEventRecord(start)
kernel<<<grid, block>>>(args)
cudaEventRecord(stop)

cudaEventSynchronize(stop)
cudaEventElapsedTime(&elapsed_ms, start, stop)
```

#### **Secondary Metrics: UVM Statistics**

```bash
# Before benchmark
cat /proc/driver/nvidia-uvm/stats > before.txt

# Run benchmark
./benchmark

# After benchmark
cat /proc/driver/nvidia-uvm/stats > after.txt

# Compute deltas
python compute_uvm_metrics.py before.txt after.txt
```

#### **Tertiary Metrics: Profiler Data**

```bash
# Nsight Systems for timeline
nsys profile -o timeline.qdrep \
  --trace=cuda,nvtx,osrt \
  --sample=cpu \
  ./benchmark

# Nsight Compute for kernel details
ncu --set full -o kernel_metrics.ncu-rep ./benchmark
```

---

### 2.6 Baseline Comparisons

**Three baseline configurations:**

1. **No-UVM Baseline:**
   - Traditional explicit memory management
   - `cudaMalloc()` + `cudaMemcpy()`
   - Represents best-case scenario (no page faults)

2. **UVM-Default Baseline:**
   - NVIDIA driver defaults
   - No manual prefetch hints in code
   - Represents typical UVM usage

3. **UVM-Optimized (Proposed):**
   - Your research contribution
   - Intelligent policies or hints
   - Goal: Match or beat No-UVM

**Reporting Format:**
```
Speedup vs No-UVM:        X.XX× (shows competitiveness)
Speedup vs UVM-Default:   X.XX× (shows contribution)
```

---

### 2.7 Threat to Validity Mitigation

| Threat | Mitigation Strategy |
|--------|-------------------|
| **GPU boost affecting results** | Lock GPU clocks to base frequency |
| **CPU frequency scaling** | Set CPU governor to performance mode |
| **Background processes** | Run on isolated system, minimize services |
| **Thermal throttling** | Monitor GPU temperature, pause if > 80°C |
| **Memory fragmentation** | Reboot system between major test phases |
| **Driver state carryover** | Reset GPU between configurations |
| **Measurement noise** | 10+ runs per config, report median + CI |
| **Cache effects** | Warmup runs before measurement |
| **Dataset bias** | Use multiple established benchmark suites |

---

