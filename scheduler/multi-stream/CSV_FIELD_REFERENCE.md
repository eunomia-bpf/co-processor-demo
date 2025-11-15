# CSV Field Reference

This document describes all fields in the benchmark CSV output and how they map between the old and new formats.

## CSV Output Location

**IMPORTANT**: CSV output is now on **stderr** (not stdout) in standard CSV format (no prefixes).

```bash
# Capture CSV output
./multi_stream_bench -s 8 -k 20 2> results.csv

# Or use experiment driver which handles this automatically
python experiment_driver.py --rq RQ1
```

## Aggregated Metrics CSV Format

### Header Row

```
streams,kernels_per_stream,kernels_per_stream_detail,total_kernels,type,type_detail,wall_time_ms,e2e_wall_time_ms,throughput,svc_mean,svc_p50,svc_p95,svc_p99,e2e_mean,e2e_p50,e2e_p95,e2e_p99,avg_queue_wait,max_queue_wait,concurrent_rate,util,jains_index,max_concurrent,avg_concurrent,inversions,inversion_rate,working_set_mb,fits_in_l2,svc_stddev,grid_size,block_size,per_priority_avg,per_priority_p50,per_priority_p99
```

### Field Descriptions

#### Configuration Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `streams` | int | Number of CUDA streams | `8` |
| `kernels_per_stream` | int | Base number of kernels per stream | `20` |
| `kernels_per_stream_detail` | string | Actual kernels per stream (comma-separated) | `"20,20,20,20,20,20,20,20"` or `"5,10,40,80"` |
| `total_kernels` | int | Total number of kernels launched | `160` |
| `type` | string | Base kernel type | `"mixed"`, `"compute"`, `"memory"`, `"gemm"` |
| `type_detail` | string | Per-stream kernel types if heterogeneous | `""` or `"compute,compute,memory,mixed,gemm"` |
| `grid_size` | int | CUDA grid size (blocks) | `256` |
| `block_size` | int | CUDA block size (threads) | `256` |

#### Timing & Throughput

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `wall_time_ms` | float | ms | Total wall time from first launch to last completion |
| `e2e_wall_time_ms` | float | ms | Total E2E wall time (including final sync) |
| `throughput` | float | kernels/sec | Throughput = total_kernels / (wall_time_ms / 1000) |

#### Service Time Metrics (Kernel Execution Duration)

| Field | Type | Unit | Description | Old Field Name |
|-------|------|------|-------------|----------------|
| `svc_mean` | float | ms | Mean kernel service time | `mean_duration` |
| `svc_p50` | float | ms | Median kernel service time | - |
| `svc_p95` | float | ms | 95th percentile service time | - |
| `svc_p99` | float | ms | 99th percentile service time | - |
| `svc_stddev` | float | ms | Standard deviation of service time | - |

**Service time** = GPU execution time only (end_time - start_time)

#### End-to-End Latency Metrics (Total Latency)

| Field | Type | Unit | Description | Old Field Name |
|-------|------|------|-------------|----------------|
| `e2e_mean` | float | ms | Mean end-to-end latency | - |
| `e2e_p50` | float | ms | Median end-to-end latency | `p50` |
| `e2e_p95` | float | ms | 95th percentile E2E latency | `p95` |
| `e2e_p99` | float | ms | 99th percentile E2E latency | `p99` |

**End-to-end latency** = Total time from enqueue to completion (end_time - enqueue_time)

#### Queueing Metrics

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `avg_queue_wait` | float | ms | Average queue waiting time (launch latency) |
| `max_queue_wait` | float | ms | Maximum queue waiting time |

**Queue wait** = Time from enqueue to start (start_time - enqueue_time)

#### Concurrency Metrics

| Field | Type | Unit | Description | Old Field Name |
|-------|------|------|-------------|----------------|
| `concurrent_rate` | float | ratio (0-1) | Fraction of time with ≥2 kernels running | - |
| `util` | float | ratio (0-1) | GPU busy time / total time | `utilization` |
| `max_concurrent` | int | count | Maximum concurrent kernels at any point | - |
| `avg_concurrent` | float | count | Average number of concurrent kernels | - |

#### Fairness Metrics

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `jains_index` | float | 0-1 | Jain's fairness index based on per-stream total execution time |

**Jain's index** = 1.0 means perfect fairness, closer to 0 means unfair

#### Priority Metrics

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `inversions` | int | count | Number of priority inversions detected |
| `inversion_rate` | float | ratio (0-1) | Fraction of kernel pairs with priority inversion |
| `per_priority_avg` | string | ms | Average E2E latency per priority class (comma-separated) |
| `per_priority_p50` | string | ms | P50 E2E latency per priority class (comma-separated) |
| `per_priority_p99` | string | ms | P99 E2E latency per priority class (comma-separated) |

**Priority inversion** = A lower-priority kernel starts before a higher-priority kernel that was enqueued earlier

Priority values are sorted in ascending order (most negative = highest priority):
- Example: `-5,-4,-2,0` means 4 priority classes
- `per_priority_p99` = `"1.2,1.5,2.0,3.5"` means P99 latencies for each class

#### Memory Metrics

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `working_set_mb` | float | MB | Estimated total working set size |
| `fits_in_l2` | bool | 0/1 | Whether working set fits in L2 cache |

**Working set** = streams × workload_size × sizeof(float) / (1024²)

## Raw Per-Kernel CSV Format

When using `--csv-output <file>`, per-kernel data is written to a separate file:

### Header Row

```
stream_id,kernel_id,priority,kernel_type,enqueue_time_ms,start_time_ms,end_time_ms,duration_ms,launch_latency_ms,e2e_latency_ms,host_launch_us,host_sync_us
```

### Field Descriptions

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `stream_id` | int | - | CUDA stream identifier (0-indexed) |
| `kernel_id` | int | - | Kernel identifier within stream (0-indexed) |
| `priority` | int | - | CUDA stream priority (-10 to 0, lower = higher priority) |
| `kernel_type` | string | - | Kernel type: `"compute"`, `"memory"`, `"mixed"`, `"gemm"` |
| `enqueue_time_ms` | float | ms | Time when kernel was enqueued (relative to first launch) |
| `start_time_ms` | float | ms | Time when kernel started executing on GPU |
| `end_time_ms` | float | ms | Time when kernel completed |
| `duration_ms` | float | ms | Kernel execution time = end_time - start_time |
| `launch_latency_ms` | float | ms | Queue wait time = start_time - enqueue_time |
| `e2e_latency_ms` | float | ms | Total latency = end_time - enqueue_time |
| `host_launch_us` | float | us | Host-side cudaLaunchKernel time (microseconds) |
| `host_sync_us` | float | us | Host-side cudaEventSynchronize time (microseconds) |

## Field Mapping: Old vs New

### Changed Field Names

| Old Field | New Field | Notes |
|-----------|-----------|-------|
| `p50` | `e2e_p50` | Clarifies it's end-to-end latency |
| `p95` | `e2e_p95` | Clarifies it's end-to-end latency |
| `p99` | `e2e_p99` | Clarifies it's end-to-end latency |
| `mean_duration` | `svc_mean` | Now separate from E2E latency |
| `utilization` | `util` | Shorter name |

### New Fields Added

- `e2e_wall_time_ms`: Total E2E wall time including sync
- `kernels_per_stream_detail`: Actual per-stream kernel counts
- `type_detail`: Per-stream kernel types for heterogeneous workloads
- `svc_p50`, `svc_p95`, `svc_p99`, `svc_stddev`: Service time percentiles
- `e2e_mean`: Mean E2E latency (separate from service time)
- `avg_queue_wait`, `max_queue_wait`: Queue wait statistics
- `inversion_rate`: Priority inversion rate
- `per_priority_avg`, `per_priority_p50`, `per_priority_p99`: Per-priority latency stats

### Removed Prefixes

**Old format** (stdout):
```
CSV_HEADER: streams,kernels,total_kernels,...
CSV: 8,20,160,...
```

**New format** (stderr):
```
streams,kernels_per_stream,total_kernels,...
8,20,160,...
```

Clean CSV format with no prefixes!

## How to Use in Analysis Scripts

### Reading Aggregated CSV

```python
import pandas as pd

# Read aggregated metrics
df = pd.read_csv('results/rq1_stream_scalability.csv')

# Access fields
print(df['e2e_p99'])  # 99th percentile E2E latency
print(df['throughput'])  # Throughput in kernels/sec
print(df['concurrent_rate'])  # Fraction of time in concurrent execution
```

### Reading Raw CSV

```python
# Read raw per-kernel data
raw_df = pd.read_csv('results/rq3_1_raw_low_concurrency_run0.csv')

# Calculate custom metrics
import numpy as np
per_stream_p99 = raw_df.groupby('stream_id')['e2e_latency_ms'].apply(
    lambda x: np.percentile(x, 99)
)

# Plot CDF
sorted_latency = np.sort(raw_df['e2e_latency_ms'])
cdf = np.arange(1, len(sorted_latency) + 1) / len(sorted_latency)
plt.plot(sorted_latency, cdf)
```

### Parsing Per-Priority Fields

```python
# Parse per_priority_p99 (comma-separated string)
def parse_per_priority(row):
    if pd.isna(row['per_priority_p99']):
        return []
    return [float(x) for x in row['per_priority_p99'].split(',')]

df['priority_p99_list'] = df.apply(parse_per_priority, axis=1)

# Extract high vs low priority
df['high_prio_p99'] = df['priority_p99_list'].apply(lambda x: np.mean(x[:len(x)//2]))
df['low_prio_p99'] = df['priority_p99_list'].apply(lambda x: np.mean(x[len(x)//2:]))
```

## Command-Line Options Affecting CSV Output

### Basic Options

```bash
# Control number of streams and kernels
./multi_stream_bench -s 8 -k 20

# Control workload size (affects kernel duration and working set)
./multi_stream_bench -w 1048576

# Control kernel type
./multi_stream_bench -t compute  # or memory, mixed, gemm
```

### Advanced Options

```bash
# Priority (affects priority metrics)
./multi_stream_bench -p "-5,-5,-5,-5,0,0,0,0"

# Load imbalance (affects kernels_per_stream_detail)
./multi_stream_bench -l "5,10,40,80"

# Heterogeneous types (affects type_detail)
./multi_stream_bench -H "compute,compute,memory,gemm"

# Launch frequency (affects arrival pattern and queueing)
./multi_stream_bench -f "100,100,100,100"

# Random seed for jitter
./multi_stream_bench -S 42  # Non-zero = add jitter, 0 = periodic
```

### CSV Output Options

```bash
# Write raw per-kernel CSV to file
./multi_stream_bench -o raw_output.csv

# Skip header for batch runs (append mode)
./multi_stream_bench -n  # or --no-header

# Both aggregated and raw CSV
./multi_stream_bench -o raw.csv 2> aggregated.csv
```

## Batch Processing Example

```bash
#!/bin/bash
# Run multiple experiments and collect results

OUTPUT_FILE="batch_results.csv"

# First run with header
./multi_stream_bench -s 4 -k 20 2> $OUTPUT_FILE

# Subsequent runs without header
for streams in 8 16 32 64; do
    ./multi_stream_bench -s $streams -k 20 -n 2>> $OUTPUT_FILE
done
```

## Notes

1. **stderr vs stdout**: Aggregated CSV is on stderr, regular output on stdout
2. **No prefixes**: Standard CSV format for easy parsing
3. **Header control**: Use `-n` or `--no-header` for batch runs
4. **Service vs E2E**: Service time = GPU execution, E2E = total latency
5. **Priority sorting**: Priority metrics are sorted by priority value (ascending)
6. **Heterogeneous fields**: Detail fields contain comma-separated per-stream values
