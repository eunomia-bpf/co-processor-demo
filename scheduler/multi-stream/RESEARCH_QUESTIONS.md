# GPU Scheduler Research Questions

## Overview
Systematic exploration of CUDA scheduler behavior using micro-benchmarks to understand concurrency, fairness, and performance characteristics.

## Research Questions

### RQ1: Stream Scalability
**Question**: How does concurrent execution rate scale with the number of streams?

**Hypothesis**: Concurrent execution rate should increase with more streams until hitting hardware limits (e.g., SM count, memory bandwidth).

**Experiments**:
- Vary streams: 1, 2, 4, 8, 16, 32, 64
- Fixed: 20 kernels/stream, mixed workload
- Metrics: Concurrent execution rate, max concurrent kernels, throughput

**Expected Insight**: Identify optimal stream count and hardware saturation point.

---

### RQ2: Workload Characterization
**Question**: How do different kernel types affect scheduler behavior and concurrency?

**Hypothesis**: Compute-bound kernels should show better concurrency than memory-bound ones due to less contention for memory bandwidth.

**Experiments**:
- Kernel types: compute, memory, mixed, gemm
- Fixed: 8 streams, 20 kernels/stream
- Metrics: Concurrent execution rate, throughput, avg latency

**Expected Insight**: Understand which workloads benefit most from multi-stream scheduling.

---

### RQ3: Priority Scheduling Effectiveness
**Question**: Does CUDA priority scheduling reduce priority inversions and improve fairness?

**Hypothesis**: Priority-enabled streams should show fewer inversions and potentially better high-priority stream performance.

**Experiments**:
- Compare: with/without priorities
- Vary: 4, 8, 16 streams
- Metrics: Priority inversions, Jain's fairness index, per-stream latency

**Expected Insight**: Quantify effectiveness of CUDA priority mechanism.

---

### RQ4: Memory Pressure Impact
**Question**: How does memory allocation size affect scheduler concurrency and performance?

**Hypothesis**: Larger allocations increase memory pressure, reducing concurrent execution due to memory subsystem saturation.

**Experiments**:
- Workload sizes: 256KB, 1MB, 4MB, 16MB, 64MB
- Fixed: 8 streams, gemm workload
- Metrics: Concurrent execution rate, throughput, memory bandwidth utilization

**Expected Insight**: Identify memory bottlenecks in multi-stream execution.

---

### RQ5: Multi-Process Interference
**Question**: How do multiple concurrent processes sharing the GPU affect scheduler fairness and performance?

**Hypothesis**: Multiple processes will compete for GPU resources, reducing per-process throughput but may improve overall GPU utilization.

**Experiments**:
- Concurrent processes: 1, 2, 4, 8
- Each with: 4 streams, mixed workload
- Metrics: Per-process throughput, system-wide fairness, aggregate throughput

**Expected Insight**: Understand MPS (Multi-Process Service) or time-slicing scheduler behavior.

---

### RQ6: Load Imbalance and Fairness
**Question**: How does the scheduler handle imbalanced workloads across streams?

**Hypothesis**: Imbalanced workloads should show lower Jain's fairness index and potential head-of-line blocking.

**Experiments**:
- Patterns tested:
  - Balanced: 5,5,5,5,5,5,5,5 (baseline)
  - Imbalanced: 5,10,20,40 (exponential growth)
  - Bimodal: 10,10,10,10,20,20,20,20 (two workload classes)
  - Linear: 5,10,15,20,25,30,35,40 (gradual increase)
  - Outliers: 20,20,20,20,20,20,20,5 and 5,5,5,5,5,5,5,20
- Fixed: mixed workload (1MB per stream)
- Metrics: Jain's fairness index, per-stream completion time, load imbalance (stddev)

**Expected Insight**: Evaluate scheduler's work-conserving and fairness properties.

**Implementation**: Using `--load-imbalance` flag to specify custom kernel counts per stream.

---

### RQ7: Tail Latency Under Contention
**Question**: How does increasing contention affect tail latency (P99)?

**Hypothesis**: Higher contention (more streams/processes) increases P99 latency disproportionately compared to median.

**Experiments**:
- Vary contention: streams (2→64) and processes (1→8)
- Metrics: P50, P95, P99 latencies, P99/P50 ratio

**Expected Insight**: Characterize predictability and QoS under load.

---

### RQ8: Kernel Launch Overhead
**Question**: Does launch overhead increase with stream count or system load?

**Hypothesis**: Launch latency should remain constant per-stream but aggregate overhead increases linearly.

**Experiments**:
- Vary: streams (1→64)
- Small, fast kernels to isolate launch overhead
- Metrics: Avg/max launch latency, launch overhead vs execution time

**Expected Insight**: Quantify scheduler overhead and scalability limits.

---

## Meta-Analysis Questions

### MQ1: Hardware Bottleneck Identification
Across all experiments, identify primary bottleneck: compute, memory bandwidth, or scheduler.

**Key Factor: Working Set Size**
- **Definition**: Total memory actively used by all concurrent streams
- **Calculation**: `working_set_size = num_streams × workload_size × sizeof(element)`
- **Critical Threshold**: GPU L2 cache size (typically 96-128MB for modern GPUs)
- **Impact**:
  - If `working_set_size ≤ L2_cache_size`: Cache-resident, high throughput
  - If `working_set_size > L2_cache_size`: Cache thrashing, memory-bound, 3× throughput drop

**Why Working Set Size Matters**:
1. **L2 Cache Capacity**: RTX 5090 has 96MB L2 cache
2. **Example (RQ4 results)**:
   - 0.25MB/stream × 8 streams = 2MB total → 70,338 kernels/sec (fits in L2)
   - 1MB/stream × 8 streams = 8MB total → 23,473 kernels/sec (exceeds L2 working set)
   - **3× performance cliff** at the L2 boundary
3. **Cache Hierarchy**:
   - L1 cache: 128KB per SM (fast, but per-SM)
   - L2 cache: 96MB shared (critical for multi-stream)
   - HBM: 32GB (slow, ~200-400μs latency)

**Evaluation on Current Setup**:
- Each experiment now reports `working_set_mb` and `fits_in_l2` metrics
- Monitor: If working set exceeds L2, expect memory bandwidth bottleneck
- Optimization: Keep working set < L2 for optimal multi-stream performance

### MQ2: Optimal Configuration
For different workload types, recommend optimal stream count and configuration.

**Working Set Considerations**:
- **Cache-friendly**: working_set < 96MB → Use more streams for parallelism
- **Cache-thrashing**: working_set > 96MB → Use fewer streams or smaller workload sizes
- **Rule of thumb**: `workload_size_per_stream ≤ L2_cache_size / num_streams`

### MQ3: Predictability
Measure variance across runs to assess scheduler determinism and predictability.

---

## Experimental Methodology

### Controls
- GPU temperature stabilization between runs
- Consistent GPU clock frequencies
- Multiple trials (10+) per configuration
- Statistical significance testing

### Data Collection
- Raw CSV output from benchmark
- System metrics (nvidia-smi): temperature, power, utilization
- Timing reproducibility measurements

### Analysis Techniques
- Regression analysis for scalability trends
- ANOVA for workload comparison
- Heatmaps for multi-dimensional configuration space
- CDF plots for latency distributions
