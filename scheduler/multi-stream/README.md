# Multi-Stream GPU Scheduler Micro-Benchmark

A minimal micro-benchmark for evaluating GPU kernel scheduling performance across multiple CUDA streams with automated experiment framework.

## Quick Start

```bash
# One command to run everything
make workflow-quick

# See all options
make help
```

See [QUICKSTART.md](QUICKSTART.md) for immediate usage.

## Overview

This benchmark measures the efficiency of concurrent kernel execution across multiple CUDA streams, providing insights into scheduler behavior, resource utilization, and parallel execution capabilities.

**Features:**
- ✅ Modern C++14 with RAII (no manual memory management)
- ✅ Comprehensive metrics (P50/P95/P99, Jain's fairness, priority inversions)
- ✅ Multiple kernel types (compute, memory, mixed, GEMM)
- ✅ Automated experiment framework with Python driver
- ✅ Multi-process scheduler testing
- ✅ Publication-ready analysis and visualizations

## Key Metrics

1. **Concurrent Execution Rate**: Percentage of kernels that execute simultaneously
2. **Aggregate Throughput**: Total operations completed per second across all streams
3. **Scheduler Overhead**: Time spent in stream management vs. kernel execution
4. **Stream Latency**: Average time from kernel enqueue to completion per stream
5. **Load Imbalance**: Standard deviation of completion times across streams
6. **GPU Utilization**: Percentage of theoretical peak compute utilized

## Architecture

```
┌─────────────────────────────────────────┐
│         Benchmark Controller            │
└─────────────────────────────────────────┘
              ┌───┴───┐
         ┌────┴───┐   └────┬───┐
    Stream 0   Stream 1   Stream N
         │         │          │
    ┌────▼───┐ ┌──▼────┐ ┌──▼────┐
    │Kernel 0│ │Kernel 1│ │Kernel N│
    │(Compute)│ │(Memory)│ │(Mixed) │
    └────────┘ └────────┘ └────────┘
```

## Configuration

Benchmark parameters (configurable via command line):

- `NUM_STREAMS`: Number of concurrent CUDA streams (default: 4)
- `NUM_KERNELS_PER_STREAM`: Kernels launched per stream (default: 10)
- `WORKLOAD_SIZE`: Problem size per kernel (default: 1M elements)
- `KERNEL_TYPE`: compute, memory, or mixed (default: mixed)

## Building

```bash
make
```

Requirements:
- CUDA Toolkit 11.0+
- GPU with compute capability 7.0+ (Volta or newer)
- GCC/G++ 9.0+

## Running

```bash
# Basic run with defaults
./multi_stream_bench

# Custom configuration
./multi_stream_bench --streams 8 --kernels 20 --size 4194304 --type compute

# Example output:
# ====================================
# Multi-Stream Scheduler Benchmark
# ====================================
# Configuration:
#   Streams: 8
#   Kernels per stream: 20
#   Workload size: 4194304 elements
#   Kernel type: compute
#
# Results:
#   Total execution time: 245.32 ms
#   Aggregate throughput: 652.14 kernels/sec
#   Avg latency per kernel: 12.27 ms
#   Concurrent execution rate: 87.3%
#   Scheduler overhead: 3.2%
#   Load imbalance (stddev): 1.45 ms
#   GPU utilization: 94.2%
# ====================================
```

## Output Format

Results are printed to stdout in both human-readable and CSV formats for easy integration with analysis scripts.

## Use Cases

- Evaluating GPU scheduler implementations
- Measuring multi-stream concurrency support
- Identifying scheduling bottlenecks
- Academic research on GPU resource management
- Performance regression testing

## Design Decisions

1. **Minimal Dependencies**: Only CUDA runtime, no external libraries
2. **Reproducible**: Fixed random seeds, deterministic execution
3. **Transparent**: All timing includes CUDA synchronization overhead
4. **Scalable**: Linear scaling with stream count up to hardware limits

## Automated Experiments

### Make Targets

The Makefile provides complete automation:

```bash
# Quick test workflow (~5 min)
make workflow-quick

# Full experiments (~1 hour)
make workflow-full

# Individual research questions
make experiment-rq1    # Stream scalability
make experiment-rq2    # Workload characterization
make experiment-rq3    # Priority effectiveness
make experiment-rq4    # Memory pressure
make experiment-rq5    # Multi-process interference
make experiment-rq7    # Tail latency

# Analysis
make analyze           # Analyze all results
make view-results      # Display summary
```

### Python Experiment Driver

Run custom experiments programmatically:

```python
from experiment_driver import BenchmarkRunner

runner = BenchmarkRunner()

# Single experiment
results = runner.run_single(
    streams=8, kernels=20,
    workload_size=1048576,
    kernel_type="gemm",
    trials=10
)

# Multi-process experiment
results = runner.run_multi_process(
    num_processes=4,
    streams_per_process=8,
    kernels=20,
    trials=10
)
```

### Research Questions

The framework systematically explores:

- **RQ1**: How does concurrent execution scale with stream count?
- **RQ2**: Which workload types benefit from multi-stream execution?
- **RQ3**: Does CUDA priority scheduling work effectively?
- **RQ4**: When does memory become the bottleneck?
- **RQ5**: How do multiple processes share the GPU?
- **RQ7**: How does tail latency degrade under contention?

See [RESEARCH_QUESTIONS.md](RESEARCH_QUESTIONS.md) for detailed methodology.

## Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Get started in 5 minutes
- **[EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md)** - Comprehensive experiment guide
- **[RESEARCH_QUESTIONS.md](RESEARCH_QUESTIONS.md)** - Research methodology
- **[requirements.txt](requirements.txt)** - Python dependencies

## Publications

Suitable for inclusion in systems conferences (OSDI, SOSP, ATC, EuroSys) focusing on:
- GPU scheduling algorithms
- Resource management in heterogeneous systems
- Performance characterization of concurrent GPU workloads
- Multi-process GPU scheduling and fairness
