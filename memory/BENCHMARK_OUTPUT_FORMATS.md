# Benchmark Output Formats and Metrics Extraction

**Purpose:** Document actual output formats from benchmarks to guide automation script development
**Date:** 2025-11-11

---

## Overview

This document catalogs the actual output formats from the UVMBench and PolyBench/GPU suites to enable reliable metric extraction for automated evaluation.

---

## 1. PolyBench/GPU Output Format

### Common Pattern

All PolyBench/GPU benchmarks follow this consistent format:

```
setting device 0 with name NVIDIA H100
GPU Time in seconds:
 0.019381
CPU Time in seconds:
 0.098326
Non-Matching CPU-GPU Outputs Beyond Error Threshold of 0.05 Percent: 0
```

### Key Metrics Available:
- **GPU Time:** Kernel execution time in seconds
- **CPU Time:** CPU reference implementation time
- **Validation:** Non-matching outputs count (should be 0)

### Extraction Regex Patterns:

```python
import re

# GPU timing
gpu_time_pattern = r"GPU Time in seconds:\s*\n\s*([0-9.]+)"

# CPU timing
cpu_time_pattern = r"CPU Time in seconds:\s*\n\s*([0-9.]+)"

# Validation
validation_pattern = r"Non-Matching.*?:\s*(\d+)"

# Example extraction
def parse_polybench_output(output: str) -> dict:
    gpu_match = re.search(gpu_time_pattern, output)
    cpu_match = re.search(cpu_time_pattern, output)
    val_match = re.search(validation_pattern, output)

    return {
        'gpu_time_sec': float(gpu_match.group(1)) if gpu_match else None,
        'cpu_time_sec': float(cpu_match.group(1)) if cpu_match else None,
        'validation_errors': int(val_match.group(1)) if val_match else None,
        'validation_passed': (val_match and int(val_match.group(1)) == 0)
    }
```

### Example Benchmarks:
- GEMM, 2DCONV, FDTD-2D, ATAX, JACOBI2D, CORR, MVT, 3MM, SYRK, LU

---

## 2. UVMBench - BFS Output Format

### Output Pattern:

```
Number of vertices 1000000
Number of edges 20000000

Starting sequential bfs.
Elapsed time in milliseconds : 250 ms.

Starting simple parallel bfs.
Elapsed time in milliseconds : 1008 ms.
Output OK!

Starting queue parallel bfs.
Elapsed time in milliseconds : 174 ms.
Output OK!

Starting scan parallel bfs.
Elapsed time in milliseconds : 9 ms.
Output OK!

Overall Elapsed time in milliseconds : 1642 ms.
```

### Key Metrics:
- Graph size (vertices, edges)
- Multiple implementation timings
- Validation status per implementation

### Extraction:

```python
def parse_bfs_output(output: str) -> dict:
    results = {}

    # Graph size
    vertices = re.search(r"Number of vertices (\d+)", output)
    edges = re.search(r"Number of edges (\d+)", output)

    # Individual timings
    sequential = re.search(r"sequential.*?\n.*?(\d+) ms", output, re.DOTALL)
    simple = re.search(r"simple parallel.*?\n.*?(\d+) ms", output, re.DOTALL)
    queue = re.search(r"queue parallel.*?\n.*?(\d+) ms", output, re.DOTALL)
    scan = re.search(r"scan parallel.*?\n.*?(\d+) ms", output, re.DOTALL)
    overall = re.search(r"Overall.*?(\d+) ms", output)

    # Validation
    validations = re.findall(r"Output OK!", output)

    return {
        'vertices': int(vertices.group(1)) if vertices else None,
        'edges': int(edges.group(1)) if edges else None,
        'sequential_ms': int(sequential.group(1)) if sequential else None,
        'simple_parallel_ms': int(simple.group(1)) if simple else None,
        'queue_parallel_ms': int(queue.group(1)) if queue else None,
        'scan_parallel_ms': int(scan.group(1)) if scan else None,
        'overall_ms': int(overall.group(1)) if overall else None,
        'validations_passed': len(validations)
    }
```

---

## 3. UVMBench - KNN Output Format

### Output Pattern:

```
Ground truth computation in progress...

Number of reference points      :   4096
Number of query points          :   4096
Dimension of points             :     32
Number of neighbors to consider :     20
Processing kNN search           :
On CPU:
1.000000, 1.000000
 done in 15.631555 s for 10 iterations (1.563155 s by iteration)
on GPU:
1.000000, 1.000000
 done in 0.475091 s for 100 iterations (0.004751 s by iteration)
```

### Key Metrics:
- Problem size (points, dimensions, k)
- CPU and GPU execution times
- Accuracy metrics (precision, index accuracy)
- Iterations performed

### Extraction:

```python
def parse_knn_output(output: str) -> dict:
    # Configuration
    ref_points = re.search(r"Number of reference points\s*:\s*(\d+)", output)
    query_points = re.search(r"Number of query points\s*:\s*(\d+)", output)
    dimensions = re.search(r"Dimension of points\s*:\s*(\d+)", output)
    k_neighbors = re.search(r"Number of neighbors.*?:\s*(\d+)", output)

    # CPU results
    cpu_accuracy = re.search(r"On CPU:.*?\n([\d.]+), ([\d.]+)", output, re.DOTALL)
    cpu_time = re.search(r"done in ([\d.]+) s for (\d+) iterations", output)

    # GPU results
    gpu_accuracy = re.search(r"on GPU:.*?\n([\d.]+), ([\d.]+)", output, re.DOTALL)
    gpu_time = re.search(r"on GPU:.*?done in ([\d.]+) s for (\d+) iterations", output, re.DOTALL)

    return {
        'ref_points': int(ref_points.group(1)) if ref_points else None,
        'query_points': int(query_points.group(1)) if query_points else None,
        'dimensions': int(dimensions.group(1)) if dimensions else None,
        'k': int(k_neighbors.group(1)) if k_neighbors else None,
        'cpu_precision': float(cpu_accuracy.group(1)) if cpu_accuracy else None,
        'cpu_index_acc': float(cpu_accuracy.group(2)) if cpu_accuracy else None,
        'cpu_time_sec': float(cpu_time.group(1)) if cpu_time else None,
        'cpu_iterations': int(cpu_time.group(2)) if cpu_time else None,
        'gpu_precision': float(gpu_accuracy.group(1)) if gpu_accuracy else None,
        'gpu_index_acc': float(gpu_accuracy.group(2)) if gpu_accuracy else None,
        'gpu_time_sec': float(gpu_time.group(1)) if gpu_time else None,
        'gpu_iterations': int(gpu_time.group(2)) if gpu_time else None,
    }
```

---

## 4. UVMBench - CNN Output Format

### Output Pattern:

```
Data loading done!
Learning
error: 2.425312e-01, time_on_gpu: 7.740000
Training complete

 Time - 7.740000

Testing!
Error Rate: 0.00%
```

### Key Metrics:
- Training error (loss)
- GPU training time
- Test error rate

### Extraction:

```python
def parse_cnn_output(output: str) -> dict:
    # Training metrics
    train_error = re.search(r"error:\s*([\d.e+-]+),\s*time_on_gpu:\s*([\d.]+)", output)

    # Total time
    total_time = re.search(r"Time - ([\d.]+)", output)

    # Test results
    test_error = re.search(r"Error Rate:\s*([\d.]+)%", output)

    return {
        'training_error': float(train_error.group(1)) if train_error else None,
        'training_time_sec': float(train_error.group(2)) if train_error else None,
        'total_time_sec': float(total_time.group(1)) if total_time else None,
        'test_error_percent': float(test_error.group(1)) if test_error else None,
        'test_passed': (test_error and float(test_error.group(1)) < 5.0)
    }
```

---

## 5. UVMBench - KMeans Output Format

### Output Pattern:

```
CUDA Took: 0.0343208s for 100000 points.
Segmentation fault (core dumped)
```

### Key Metrics:
- Execution time
- Point count
- **Note:** Segfault is post-computation (cleanup issue)

### Extraction:

```python
def parse_kmeans_output(output: str) -> dict:
    # Extract time and points
    match = re.search(r"CUDA Took:\s*([\d.]+)s for (\d+) points", output)

    # Check for segfault
    has_segfault = "Segmentation fault" in output

    return {
        'execution_time_sec': float(match.group(1)) if match else None,
        'num_points': int(match.group(2)) if match else None,
        'computation_successful': match is not None,
        'has_post_cleanup_error': has_segfault
    }
```

---

## 6. UVMBench - BN (Bayesian Network) Output Format

### Output Pattern:

```
Duration per iteration: 3.182 ms
Total duration: 439.545 ms
Preprocessing duration: 121.388 ms
```

### Key Metrics:
- Per-iteration time
- Total execution time
- Preprocessing overhead

### Extraction:

```python
def parse_bn_output(output: str) -> dict:
    per_iter = re.search(r"Duration per iteration:\s*([\d.]+) ms", output)
    total = re.search(r"Total duration:\s*([\d.]+) ms", output)
    preproc = re.search(r"Preprocessing duration:\s*([\d.]+) ms", output)

    return {
        'per_iteration_ms': float(per_iter.group(1)) if per_iter else None,
        'total_duration_ms': float(total.group(1)) if total else None,
        'preprocessing_ms': float(preproc.group(1)) if preproc else None,
        'num_iterations': int(float(total.group(1)) / float(per_iter.group(1))) if (total and per_iter) else None
    }
```

---

## 7. UVM Driver Statistics

### Available from `/proc/driver/nvidia-uvm/stats`

**Location:** `/proc/driver/nvidia-uvm/stats`

**Format:** Key-value pairs, one per line

```
cpu_pages_faulted: 12345
gpu_pages_faulted: 98765
migrations_cpu_to_gpu: 4567
migrations_gpu_to_cpu: 1234
thrashing_detected: 23
evictions: 456
...
```

### Key Metrics to Track:

```python
UVM_STATS_KEYS = [
    'cpu_pages_faulted',
    'gpu_pages_faulted',
    'migrations_cpu_to_gpu',
    'migrations_gpu_to_cpu',
    'migrations_gpu_to_gpu',
    'thrashing_detected',
    'evictions',
    'remote_mappings',
    'fault_batch_count',
    'prefetch_count',
    'prefetch_pages',
]

def parse_uvm_stats(stats_file: str) -> dict:
    stats = {}
    with open(stats_file) as f:
        for line in f:
            if ':' in line:
                key, value = line.strip().split(':', 1)
                try:
                    stats[key.strip()] = int(value.strip())
                except ValueError:
                    pass
    return stats

def compute_uvm_delta(before: dict, after: dict) -> dict:
    """Compute change in UVM stats during benchmark run"""
    delta = {}
    for key in UVM_STATS_KEYS:
        if key in after and key in before:
            delta[key] = after[key] - before[key]
        elif key in after:
            delta[key] = after[key]
    return delta
```

---

## 8. Unified Parsing Interface

### Benchmark Registry:

```python
from enum import Enum
from typing import Callable, Dict

class BenchmarkType(Enum):
    POLYBENCH = "polybench"
    BFS = "bfs"
    KNN = "knn"
    CNN = "cnn"
    KMEANS = "kmeans"
    BN = "bn"
    LOGREG = "logreg"

PARSER_REGISTRY: Dict[BenchmarkType, Callable] = {
    BenchmarkType.POLYBENCH: parse_polybench_output,
    BenchmarkType.BFS: parse_bfs_output,
    BenchmarkType.KNN: parse_knn_output,
    BenchmarkType.CNN: parse_cnn_output,
    BenchmarkType.KMEANS: parse_kmeans_output,
    BenchmarkType.BN: parse_bn_output,
}

def parse_benchmark_output(benchmark_type: BenchmarkType, output: str) -> dict:
    """Unified interface for parsing benchmark outputs"""
    parser = PARSER_REGISTRY.get(benchmark_type)
    if parser is None:
        raise ValueError(f"No parser registered for {benchmark_type}")
    return parser(output)
```

---

## 9. Validation Checks

### Standard Validation Rules:

```python
def validate_benchmark_result(result: dict, benchmark_type: BenchmarkType) -> tuple[bool, str]:
    """
    Validate benchmark result
    Returns: (is_valid, error_message)
    """

    # Check for required fields
    if benchmark_type == BenchmarkType.POLYBENCH:
        if result.get('gpu_time_sec') is None:
            return False, "Missing GPU time"
        if not result.get('validation_passed', False):
            return False, f"Validation failed with {result.get('validation_errors', 'unknown')} errors"

    elif benchmark_type == BenchmarkType.BFS:
        if result.get('overall_ms') is None:
            return False, "Missing overall time"
        expected_validations = 3  # simple, queue, scan
        if result.get('validations_passed', 0) < expected_validations:
            return False, f"Only {result.get('validations_passed')} of {expected_validations} validations passed"

    elif benchmark_type == BenchmarkType.KNN:
        if result.get('gpu_time_sec') is None:
            return False, "Missing GPU time"
        # Check accuracy
        if result.get('gpu_precision', 0) < 0.95:
            return False, f"Low precision: {result.get('gpu_precision')}"

    elif benchmark_type == BenchmarkType.CNN:
        if result.get('training_time_sec') is None:
            return False, "Missing training time"
        # Check convergence
        if result.get('test_error_percent', 100) > 10.0:
            return False, f"High test error: {result.get('test_error_percent')}%"

    elif benchmark_type == BenchmarkType.KMEANS:
        if not result.get('computation_successful', False):
            return False, "Computation did not complete successfully"
        # Segfault is OK if computation completed

    elif benchmark_type == BenchmarkType.BN:
        if result.get('total_duration_ms') is None:
            return False, "Missing total duration"

    return True, ""
```

---

## 10. Error Handling

### Common Issues and Solutions:

| Issue | Detection | Handling |
|-------|-----------|----------|
| Timeout | Process exceeds timeout | Mark as failed, log timeout duration |
| Segfault | "Segmentation fault" in stderr | Check if computation completed first |
| CUDA Error | "CUDA" or "cudaGetErrorString" in output | Extract error code, mark as failed |
| Validation Failure | Non-zero validation errors | Log specific error count |
| Missing Output | Cannot parse expected metrics | Mark as failed, save raw output |
| Incorrect Results | Accuracy below threshold | Mark as validation failure |

```python
class BenchmarkExecutionError(Exception):
    """Base exception for benchmark execution errors"""
    pass

class BenchmarkTimeoutError(BenchmarkExecutionError):
    """Benchmark exceeded timeout"""
    pass

class BenchmarkValidationError(BenchmarkExecutionError):
    """Benchmark produced incorrect results"""
    pass

class BenchmarkParseError(BenchmarkExecutionError):
    """Could not parse benchmark output"""
    pass
```

---

## Summary

**Key Takeaways:**

1. **Two main output patterns:** PolyBench (consistent) and UVMBench (varied)
2. **Essential metrics:** Execution time, validation status
3. **UVM stats:** Available via `/proc/driver/nvidia-uvm/stats` (capture before/after)
4. **Parsing strategy:** Regex-based, with type-specific parsers
5. **Validation:** Check both execution success and correctness
6. **Error handling:** Distinguish between different failure modes

**Implementation Priority:**
1. Implement parsers for all benchmark types
2. Add UVM stats collection wrapper
3. Create unified validation framework
4. Build error recovery mechanisms
5. Add comprehensive logging

---

**Version:** 1.0
**Last Updated:** 2025-11-11
**Related Documents:**
- `/root/co-processor-demo/memory/OSDI_EVALUATION_SECTION.md`
- `/root/co-processor-demo/memory/benchmark_automation_design.py`
