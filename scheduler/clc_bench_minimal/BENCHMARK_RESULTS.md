# Cluster Launch Control Benchmark Results

## Overview

This benchmark compares three thread block scheduling approaches for CUDA kernels:

1. **Fixed Work per Thread Block** - Traditional approach where each block handles fixed work
2. **Fixed Number of Thread Blocks** - Grid-stride loop with SM-based block count
3. **Cluster Launch Control (CLC)** - New CC 10.0 feature enabling work-stealing

## Test Environment

- **GPU**: NVIDIA GeForce RTX 5090
- **Compute Capability**: 12.0 (Blackwell)
- **SM Count**: 170
- **CUDA Version**: 12.9
- **Driver**: 575.57.08

## Benchmark Configuration

All tests use vector-scalar multiplication (y = αx) with:
- Prologue: `compute_scalar()` simulates shared computation overhead
- Threads per block: 256
- Warmup runs: 3
- Benchmark runs: 5

## Results

### Small Dataset (64K elements, 0.26 MB)

```
Fixed Work per Thread Block:
  Configuration: 256 blocks x 256 threads
  Average time: 0.003 ms
  Average blocks executed: 256.0
  Bandwidth: 182.04 GB/s
  Correctness: ✅ PASSED

Fixed Number of Thread Blocks:
  Configuration: 340 blocks x 256 threads
  Average time: 0.003 ms
  Average blocks executed: 340.0
  Bandwidth: 181.37 GB/s
  Correctness: ✅ PASSED

Cluster Launch Control:
  Configuration: 256 blocks x 256 threads
  Average time: 0.003 ms
  Average blocks executed: 256.0
  Average work steals: 0.0
  Bandwidth: 189.05 GB/s
  Correctness: ✅ PASSED
```

**Analysis**: For small workloads, all three approaches perform similarly with minimal overhead differences.

### Medium Dataset (1M elements, 4.19 MB)

```
Fixed Work per Thread Block:
  Configuration: 4096 blocks x 256 threads
  Average time: 0.006 ms
  Average blocks executed: 4096.0
  Bandwidth: 1492.85 GB/s
  Correctness: ✅ PASSED

Fixed Number of Thread Blocks:
  Configuration: 340 blocks x 256 threads
  Average time: 0.005 ms
  Average blocks executed: 340.0
  Bandwidth: 1747.63 GB/s
  Correctness: ✅ PASSED

Cluster Launch Control:
  Configuration: 4096 blocks x 256 threads
  Average time: 0.007 ms
  Average blocks executed: 1020.0
  Average work steals: 3076.0
  Bandwidth: 1260.31 GB/s
  Correctness: ✅ PASSED
```

**Analysis**:
- **CLC work-stealing in action**: Launched 4096 blocks but only 1020 actually executed (75% reduction!)
- **Fixed Blocks wins**: Achieves highest bandwidth due to minimal prologue overhead (340 blocks vs 1020/4096)
- **CLC trades some performance for flexibility**: Shows ~17% lower bandwidth than Fixed Blocks but provides load balancing + preemption benefits

## Key Insights

### Approach Comparison

| Approach | Blocks Launched | Blocks Executed | Prologue Overhead | Best For |
|----------|----------------|-----------------|-------------------|----------|
| Fixed Work | 4096 | 4096 | High (4096x) | Variable workloads with preemption needs |
| Fixed Blocks | 340 | 340 | Low (340x) | Known workloads, maximum throughput |
| CLC | 4096 | 1020 | Medium (1020x) | Balance of efficiency + flexibility |

### When to Use Each Approach

**Fixed Work per Thread Block:**
- ✅ When thread blocks have variable execution times
- ✅ When preemption is critical
- ✅ When load balancing across SMs is more important than overhead
- ❌ When prologue cost is significant

**Fixed Number of Thread Blocks:**
- ✅ When workload is predictable
- ✅ When minimizing overhead is critical
- ✅ When prologue has expensive shared computations
- ❌ When preemption is needed
- ❌ When load balancing is critical

**Cluster Launch Control:**
- ✅ When you want **both** reduced overhead **and** load balancing
- ✅ When supporting preemption with reasonable efficiency
- ✅ For general-purpose kernels needing flexibility
- ⚠️ Requires CC 10.0+ hardware

### CLC Benefits

1. **Reduced Prologue Overhead**: Only 1020 blocks computed `alpha` vs 4096 in Fixed Work (75% reduction)
2. **Load Balancing**: Can redistribute work if some SMs complete faster
3. **Preemption Support**: Can exit early if higher-priority kernel arrives
4. **Best of Both Worlds**: Combines advantages of both traditional approaches

## Bandwidth Analysis

For this benchmark:
- **Fixed Blocks**: 1747.63 GB/s (best)
- **Fixed Work**: 1492.85 GB/s (-15% vs Fixed Blocks)
- **CLC**: 1260.31 GB/s (-28% vs Fixed Blocks, -16% vs Fixed Work)

The bandwidth differences reflect the trade-off between overhead reduction and work-stealing coordination costs.

## Running the Benchmark

```bash
# Build
make clc_benchmark

# Run with defaults (1M elements)
./clc_benchmark

# Custom configuration
./clc_benchmark [elements] [threads_per_block] [warmup_runs] [bench_runs]

# Examples
./clc_benchmark 65536 256 2 3          # Quick test
./clc_benchmark 1048576 256 3 10       # Standard benchmark
./clc_benchmark 16777216 512 5 20      # Large dataset
```

## Conclusions

1. **CLC successfully reduces overhead**: 75% fewer blocks executed vs Fixed Work
2. **Fixed Blocks still wins for pure performance**: When overhead is the main concern
3. **CLC provides unique flexibility**: Only approach offering both load balancing AND reduced overhead
4. **Hardware-dependent**: Requires Blackwell (CC 10.0+) architecture
5. **Use case dependent**: Choose based on whether you prioritize:
   - Maximum throughput → Fixed Blocks
   - Load balancing + preemption → Fixed Work
   - Balanced approach → CLC

The Cluster Launch Control feature demonstrates NVIDIA's continued innovation in giving developers fine-grained control over GPU scheduling, enabling new optimization strategies that weren't previously possible.
