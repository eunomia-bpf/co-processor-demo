# Running UVMBench - Quick Start Guide

This guide explains how to run all benchmarks in the UVMBench suite using the provided Python script.

## Quick Start

### Run All Benchmarks (UVM and non-UVM)

```bash
python3 run_all_benchmarks.py
```

This will:
1. Automatically prepare test data (BFS graphs, K-means data)
2. Run all UVM benchmarks
3. Run all non-UVM benchmarks
4. Generate summary reports (CSV, JSON, comparison table)
5. Save all outputs to `results/TIMESTAMP/` directory

### Run Only UVM Benchmarks

```bash
python3 run_all_benchmarks.py --mode uvm
```

### Run Only non-UVM Benchmarks

```bash
python3 run_all_benchmarks.py --mode non-uvm
```

### Run with Profiling (using ncu)

```bash
python3 run_all_benchmarks.py --profile
```

This generates `.ncu-rep` files for each benchmark that can be opened in NVIDIA Nsight Compute.

### Verbose Mode (Show Errors)

```bash
python3 run_all_benchmarks.py --verbose
```

Shows detailed error messages when benchmarks fail.

### Custom Timeout

```bash
python3 run_all_benchmarks.py --timeout 600
```

Sets timeout to 600 seconds (10 minutes) per benchmark. Default is 300 seconds.

### Prepare Data Only

```bash
python3 run_all_benchmarks.py --prepare-data
```

Only generates test data without running benchmarks.

## Command Line Options

```
usage: run_all_benchmarks.py [-h] [--mode {uvm,non-uvm,both}] [--profile]
                             [--timeout TIMEOUT] [--verbose] [--prepare-data]

Options:
  --mode {uvm,non-uvm,both}
                        Run mode (default: both)
  --profile             Enable ncu profiling
  --timeout TIMEOUT     Timeout per benchmark in seconds (default: 300)
  --verbose             Show detailed output
  --prepare-data        Only prepare test data and exit
  -h, --help           Show help message
```

## Output Files

After running, the script creates a timestamped directory under `results/` containing:

### Main Output Files

- **`run_all.log`** - Complete execution log with all benchmark outputs
- **`summary.csv`** - CSV file with benchmark results:
  ```
  Benchmark,Version,Status,Time(s),Output
  kmeans,UVM,SUCCESS,1.070,/path/to/output.txt
  ...
  ```

- **`summary.json`** - JSON file with detailed results and statistics:
  ```json
  {
    "timestamp": "2025-11-11T12:33:00",
    "statistics": {
      "total": 32,
      "successful": 10,
      "failed": 15,
      "skipped": 7,
      "timeout": 0
    },
    "results": [...]
  }
  ```

- **`comparison.txt`** - Performance comparison table (UVM vs non-UVM):
  ```
  Benchmark                      UVM (s)   non-UVM (s)   Speedup
  ------------------------------------------------------------------------
  kmeans                          1.070        0.850      0.79x
  CNN                            11.660       10.230      0.88x
  ...
  ```

### Individual Benchmark Outputs

- **`UVM_<benchmark>.txt`** - stdout/stderr from UVM version
- **`non-UVM_<benchmark>.txt`** - stdout/stderr from non-UVM version
- **`UVM_<benchmark>.ncu-rep`** - Profiling data (if `--profile` used)

## Example Output

```
=================================================================
UVMBench - Comprehensive Benchmark Suite Runner
=================================================================
Run mode: both
Profiling: False
Timeout: 300s
Results directory: /path/to/results/2025-11-11_12-33-00
=================================================================

Preparing test data...
  Generating BFS graph data...
  ✓ BFS data generated
  ✓ K-means data generated
✓ Test data preparation complete

=================================================================
Running UVM Benchmarks
=================================================================

Running simple benchmarks...

────────────────────────────────────────────────────────────
Running: kmeans (UVM)
────────────────────────────────────────────────────────────
[SUCCESS] Completed in 1.07s

────────────────────────────────────────────────────────────
Running: CNN (UVM)
────────────────────────────────────────────────────────────
[SUCCESS] Completed in 11.66s

...

=================================================================
Benchmark Execution Summary
=================================================================
Total benchmarks attempted: 64
Successful: 15
Failed: 40
Timeout: 0
Skipped: 9
Results saved to: /path/to/results/2025-11-11_12-33-00
Log file: /path/to/results/2025-11-11_12-33-00/run_all.log
=================================================================

Summary CSV: /path/to/results/2025-11-11_12-33-00/summary.csv
Summary JSON: /path/to/results/2025-11-11_12-33-00/summary.json
Comparison table: /path/to/results/2025-11-11_12-33-00/comparison.txt

Total execution time: 245.3s
```

## Understanding Results

### Status Codes

- **SUCCESS** - Benchmark completed successfully (exit code 0)
- **FAILED** - Benchmark failed (non-zero exit code)
- **TIMEOUT** - Benchmark exceeded timeout limit
- **SKIP** - Benchmark was skipped (missing executable or directory)

### Common Failure Reasons

1. **Missing data files** - Some benchmarks need pre-generated data
2. **Build errors** - Executable not compiled or compilation failed
3. **CUDA errors** - Runtime CUDA errors (out of memory, etc.)
4. **Input file errors** - Missing or incorrect input file format

### Analyzing Failures

Check individual output files for details:

```bash
# View failed benchmark output
cat results/TIMESTAMP/UVM_bfs.txt

# Search for error messages
grep -i "error\|failed" results/TIMESTAMP/*.txt

# List all failed benchmarks
grep "FAILED" results/TIMESTAMP/run_all.log
```

## Advanced Usage

### Running Specific Benchmark Categories

The script automatically runs benchmarks in these categories:

1. **Simple benchmarks**: bfs, BN, CNN, kmeans, knn, logistic-regression, SVM
2. **Rodinia benchmarks**: backprop, dwt2d, gaussian, hotspot, hotspot3D, nn, nw, particlefilter, pathfinder, srad, streamcluster
3. **Polybench benchmarks**: 2DCONV, 2MM, 3DCONV, 3MM, ATAX, BICG, CORR, COVAR, FDTD-2D, GEMM, GESUMMV, GRAMSCHM, MVT, SYR2K, SYRK

To run only specific benchmarks, modify the lists in the script or run individual benchmarks manually.

### Parallel Execution

The script runs benchmarks sequentially. For parallel execution, you can use GNU parallel:

```bash
# Create a list of commands
python3 -c "
categories = ['kmeans', 'CNN', 'logistic-regression']
for cat in categories:
    print(f'cd UVM_benchmarks/{cat} && ./run')
" > commands.txt

# Run in parallel (4 at a time)
parallel -j 4 < commands.txt
```

### Integration with CI/CD

```bash
# Run benchmarks and exit with error code if any failed
python3 run_all_benchmarks.py --mode uvm --timeout 600

# Check exit code
if [ $? -eq 0 ]; then
    echo "All benchmarks passed"
else
    echo "Some benchmarks failed"
    exit 1
fi
```

### Profiling Multiple Metrics

For detailed profiling with specific metrics:

```bash
# The script uses ncu with --set full by default
# For custom metrics, modify the script or run manually:

cd UVM_benchmarks/kmeans
ncu --metrics sm__cycles_elapsed.avg,dram__bytes.sum ./run
```

## Troubleshooting

### Script Hangs

If a benchmark hangs, the timeout will kill it automatically. Increase timeout if needed:

```bash
python3 run_all_benchmarks.py --timeout 600
```

### Permission Errors

```bash
# Make sure run scripts are executable
cd UVM_benchmarks
make get_permission

cd ../non_UVM_benchmarks
make get_permission
```

### Out of Memory

Reduce the problem size in individual `run` scripts or skip memory-intensive benchmarks.

### CUDA Version Mismatch

Some benchmarks may fail with older CUDA versions. Check compatibility:

```bash
nvidia-smi  # Check driver CUDA version
nvcc --version  # Check toolkit version
```

## Benchmarks Known to Work

Based on testing with CUDA 12.9 and RTX 5090:

**Working UVM Benchmarks:**
- ✓ kmeans
- ✓ CNN
- ✓ logistic-regression
- ✓ rodinia_nn
- ✓ Several polybench benchmarks

**Require Data Generation:**
- bfs (needs graphgen)
- knn (needs data files)
- SVM (needs data files)

**May Have Build Issues:**
- Some polybench benchmarks (missing --no-device-link flag)
- Some rodinia benchmarks (compilation errors)

## Performance Analysis

After running both UVM and non-UVM versions, check the comparison table:

```bash
cat results/TIMESTAMP/comparison.txt
```

This shows which benchmarks benefit from UVM and which perform better with traditional CUDA memory management.

Typical observations:
- Memory-bound applications may benefit from UVM's automatic migration
- Compute-bound applications may see overhead from UVM page faults
- Applications with irregular access patterns may benefit most from UVM

## Further Reading

- See `README_DETAILED.md` for comprehensive documentation
- See `UVMBench_Paper.md` for research background
- Check individual benchmark directories for specific run scripts and parameters
