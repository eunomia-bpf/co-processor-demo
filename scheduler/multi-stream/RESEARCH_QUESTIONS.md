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

**Implementation Enhancement**: Added heterogeneous workload tests (memory vs compute kernel types) alongside original homogeneous tests to create realistic contention patterns.

---

### RQ9: Priority-Based Tail Latency (XSched Comparison)
**Question**: Does CUDA stream priority mechanism effectively reduce tail latency for high-priority workloads?

**Motivation**: Compare with XSched's "fixed priority" policy that achieves 2-3× P99 latency reduction.

**Hypothesis**: High-priority streams should achieve lower P99 latency when competing with low-priority background work.

**Experiments**:
- **Workload model**: Dual-stream setup
  - **Front-end (high priority)**: Periodic latency-sensitive requests (small kernels, arrival rate λ)
  - **Back-end (low priority)**: Continuous throughput workload (large kernels)
- **Configurations**:
  - Front load intensity: 10%, 30%, 50%, 70% of GPU capacity
  - Stream counts: 2 (1 front + 1 back), 4 (2+2), 8 (4+4)
  - Priority modes: Disabled (baseline) vs Enabled (CUDA priority range)
- **Metrics**:
  - Front-end: **P50, P95, P99 latency** + full CDF
  - Back-end: Throughput (kernels/sec)
  - Priority effectiveness: P99_priority / P99_baseline
  - Comparison: CUDA Native (FCFS) vs Priority-enabled vs XSched (from paper)

**Expected Insight**:
- Priority should reduce front-end P99 by 2-5× (comparable to XSched)
- Higher front load → diminishing priority benefit (saturation)
- Validate CUDA priority as practical QoS mechanism

**XSched Comparison**: Maps to XSched §7.2 + Figure 9 (top) - "Fixed Priority" policy

**Implementation Enhancement**: Added heterogeneous tests with fast memory kernels (HIGH priority) vs slow compute kernels (LOW priority) to simulate front-end/back-end separation, alongside original load-imbalance tests.

---

### RQ10: Scheduler Preemption Latency
**Question**: How quickly can the CUDA scheduler preempt a running stream to service a higher-priority request?

**Motivation**: XSched achieves 10-50µs preemption latency with Lv3 (hardware-assisted). What is CUDA's baseline?

**Hypothesis**: CUDA's time-slicing introduces ms-level preemption latency, proportional to kernel execution time.

**Experiments**:
- **Setup**:
  1. Launch long-running kernel on low-priority stream A
  2. After T ms, inject high-priority kernel on stream B
  3. Measure: Δt = time from `cudaLaunchKernel(B)` to B's actual execution start
- **Variables**:
  - Command duration (A's kernel): 1ms, 10ms, 100ms, 1000ms
  - Priority modes: No priority (FCFS) vs High priority (B)
  - Kernel types: compute, memory, mixed
- **Metrics**:
  - **P50, P95, P99 preemption latency**
  - **Preemption latency vs command duration** (should be linear for FCFS, sub-linear for priority)
  - **Comparison**: CUDA baseline vs XSched Lv1/Lv2/Lv3 (from paper)

**Measurement Method**:
```cpp
// On stream A (low priority)
cudaEventRecord(A_start, streamA);
long_kernel<<<grid, block, 0, streamA>>>(data, 100ms);
cudaEventRecord(A_end, streamA);

// After 10ms, on stream B (high priority)
cudaEventRecord(B_enqueue, 0);  // Host-side timestamp
cudaEventRecord(B_start, streamB);
short_kernel<<<grid, block, 0, streamB>>>(data, 1ms);

// Calculate preemption latency
preemption_latency = B_start_time - B_enqueue_time;
```

**Expected Results**:
- **CUDA FCFS**: Preemption ≈ remaining time of A's kernel (0-100ms, linear)
- **CUDA Priority**: Preemption ~1-10ms (better but still coarse-grained)
- **XSched Lv3**: 10-50µs (from paper, hardware-assisted)

**Limitation**: CUDA lacks fine-grained preemption (no Lv2/Lv3 equivalent). This RQ quantifies the gap.

**XSched Comparison**: Maps to XSched §7.3 + Figure 11(a)(b) - "Preemption Latency vs Command Duration"

**Implementation**: Measured via contention analysis (fast memory kernel latency with/without blocking compute kernel) rather than direct GPU timestamps.

---

### RQ11: Bandwidth Partitioning and Quota Enforcement
**Question**: Can we achieve target throughput ratios (bandwidth partitioning) between competing streams?

**Motivation**: XSched's "bandwidth partition" achieves 75/25 split with 1.5% overhead. Can we do this in user-space?

**Hypothesis**: With kernel-level throttling, we can enforce quota ratios (e.g., 75/25) with <3% efficiency loss.

**Experiments**:
- **Workload**: Two stream groups (Front + Back), both running same kernel type
- **Target ratios**: 50/50, 75/25, 90/10, 95/5
- **Enforcement mechanism** (user-space implementation):
  ```cpp
  // Option 1: Kernel throttling
  while (front_quota_used >= front_quota_limit) {
      usleep(100); // Wait for quota window reset
  }
  launch_kernel(front_stream, ...);

  // Option 2: Dynamic kernel count adjustment
  int front_kernels = total_kernels * target_ratio;
  int back_kernels = total_kernels * (1 - target_ratio);
  ```
- **Variables**:
  - Enforcement granularity: 1ms, 10ms, 100ms windows
  - Workload types: compute, memory, mixed
  - Stream counts: 2 (1+1), 4 (2+2), 8 (4+4)
- **Metrics**:
  - **Quota accuracy**: |actual_ratio - target_ratio| (should be <5%)
  - **Front/Back throughput**: Normalized to solo execution (0-1.0)
  - **Total throughput**: Sum of both, normalized to baseline (measures efficiency loss)
  - **Overhead**: Runtime increase (%), CPU utilization (%)
  - **Comparison**: CUDA Native (no quota) vs Our mechanism vs XSched (from paper)

**Expected Results**:
| Target Ratio | Achieved Ratio | Total Throughput | Overhead |
|--------------|----------------|------------------|----------|
| 50/50        | 50.2/49.8      | 0.98            | 2.1%     |
| 75/25        | 74.8/25.2      | 0.97            | 2.8%     |
| 90/10        | 89.5/10.5      | 0.96            | 3.5%     |

**Comparison with XSched**:
- XSched achieves ~1.5% overhead (Lv1) with hardware support
- Our user-space approach targets <3% (acceptable for quota enforcement)

**XSched Comparison**: Maps to XSched §7.2 + Figure 9 (bottom) - "Bandwidth Partition" policy

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
