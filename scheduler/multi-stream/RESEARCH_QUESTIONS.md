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
- Create imbalance: vary kernel count per stream (e.g., 5, 10, 20, 40)
- Fixed: 8 streams, mixed workload
- Metrics: Jain's fairness index, per-stream completion time, load imbalance

**Expected Insight**: Evaluate scheduler's work-conserving and fairness properties.

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

### MQ2: Optimal Configuration
For different workload types, recommend optimal stream count and configuration.

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
