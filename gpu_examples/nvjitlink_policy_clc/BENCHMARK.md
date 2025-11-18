# GEMM Benchmark System

Comprehensive benchmarking and visualization system for evaluating GEMM performance with different matrix patterns and CLC scheduling policies.

## Quick Start

```bash
# Run full benchmark pipeline
make benchmark-full
```

This will:
1. Build all binaries and PTX files
2. Generate test matrices with different patterns
3. Run comprehensive benchmarks
4. Generate visualizations and HTML report

View results: `./plots/benchmark_report.html`

## Step-by-Step Usage

### 1. Generate Test Matrices

```bash
make matrices
```

Generates test matrices in `./test_matrices/` with patterns:
- **uniform**: Random values uniformly distributed
- **sparse_50**: 50% zero values
- **sparse_90**: 90% zero values
- **diagonal**: Identity-like diagonal matrices
- **block**: Block-structured patterns
- **imbalanced**: Heavily skewed distributions
- **identity**: Pure identity matrices

Sizes generated: `512x512x512`, `256x256x256`

### 2. Run Benchmarks

```bash
make benchmark
```

Runs each test case with:
- **Original** version (direct kernel launch)
- **Policy (greedy)** - GreedyPolicy (always executes)
- **Policy (maxsteals)** - MaxStealsPolicy (limits to 8 executions)

Each configuration is:
- Warmed up with 3 runs
- Benchmarked with 10 runs
- Statistically analyzed (mean, std dev, min, max)

Results saved to: `benchmark_results.csv`

### 3. Visualize Results

```bash
make visualize
```

Generates:
- **performance_comparison.png** - GFLOPS across patterns/policies
- **speedup_analysis.png** - Speedup factors vs baseline
- **latency_comparison.png** - Execution time comparisons
- **size_scaling.png** - Performance scaling across sizes
- **benchmark_report.html** - Interactive HTML report

Output directory: `./plots/`

## Manual Usage

### Generate Custom Matrices

```bash
python3 generate_matrices.py --sizes 1024x1024x1024 --output-dir ./custom_matrices
```

### Run Custom Benchmark

```bash
python3 benchmark_driver.py \
  --matrix-dir ./custom_matrices \
  --binary-original ./gemm_test_original \
  --binary-policy ./gemm_test_modify \
  --warmup 5 \
  --runs 20 \
  --output custom_results.csv
```

### Visualize Custom Results

```bash
python3 visualize_results.py \
  --input custom_results.csv \
  --output-dir ./custom_plots
```

## Key Results Summary

Based on the benchmark results (`benchmark_results.csv`):

### 512x512x512 Matrices
- **Baseline (Original)**: ~2800-2900 GFLOPS
- **Policy Framework**: ~3100-3180 GFLOPS
- **Speedup**: 1.05x - 1.15x (5-15% improvement)
- **Best pattern**: Imbalanced (1.15x speedup)

### 256x256x256 Matrices
- **Baseline (Original)**: ~535-560 GFLOPS
- **Policy Framework**: ~900-955 GFLOPS
- **Speedup**: 1.62x - 1.75x (62-75% improvement)
- **Best pattern**: Sparse_90 (1.75x speedup)

### Key Insights

1. **Smaller matrices benefit more** from policy framework (75% vs 15% speedup)
2. **All matrix patterns** show consistent improvements
3. **Greedy and MaxSteals policies** perform similarly (within 1-2%)
4. **No correctness issues** - all results verified successfully

## Files

### Python Scripts
- `generate_matrices.py` - Generate test matrices with various patterns
- `benchmark_driver.py` - Automated benchmark runner
- `visualize_results.py` - Generate plots and HTML report

### Data Files
- `benchmark_results.csv` - Raw benchmark results
- `test_matrices/manifest.txt` - Matrix file manifest
- `test_matrices/*/` - Binary matrix files

### Output
- `plots/*.png` - Visualization plots
- `plots/benchmark_report.html` - Interactive HTML report

## Cleanup

```bash
# Clean build artifacts only
make clean

# Clean everything (including matrices and results)
make clean-all
```

## Architecture

The benchmark system uses:

1. **Binary matrix format**: Efficient storage with dimension headers
2. **Manifest-driven testing**: Easy batch processing from `manifest.txt`
3. **Statistical analysis**: Multiple runs with mean/std dev calculations
4. **Comparative visualization**: Side-by-side performance comparisons
5. **HTML reporting**: Interactive results viewing

## Matrix File Format

Binary format for easy loading:
```
[4 bytes: rows (int)]
[4 bytes: cols (int)]
[rows * cols * 4 bytes: data (float)]
```

## Environment Variables

When running with policy framework:
```bash
WRAPPER_KERNEL_PATH=./wrapper_kernel.ptx \
POLICY_PTX_PATH=./policy_greedy.ptx \
./gemm_test_modify --matrix-a A.bin --matrix-b B.bin --size 512x512x512
```

## Performance Tips

- Use `--warmup` to eliminate cold-start effects
- Increase `--runs` for more stable statistics
- Test multiple matrix sizes to understand scaling
- Compare both policies to find best for your workload
