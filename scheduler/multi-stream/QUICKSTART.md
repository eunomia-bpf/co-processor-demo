# Quick Start Guide

Get started with GPU scheduler experiments in 5 minutes!

## One-Command Quick Test

```bash
make workflow-quick
```

This will:
1. Build the benchmark
2. Run pilot experiments (3 trials, ~5 min)
3. Analyze results automatically
4. Display summary report

## What You'll Get

After running, you'll have:

```
results/
├── *.csv                    # Raw experimental data
├── figures/*.png            # Visualizations
└── ANALYSIS_REPORT.md       # Summary of findings
```

## View Results

```bash
make view-results
```

## Full Experiment Suite

For publication-quality results:

```bash
# Takes ~1 hour with 10 trials
make workflow-full

# Or customize trial count
make TRIALS=20 workflow-full
```

## Individual Experiments

Run specific research questions:

```bash
# Stream scalability
make experiment-rq1 analyze-rq1

# Workload comparison
make experiment-rq2 analyze-rq2

# Multi-process interference
make experiment-rq5 analyze-rq5
```

## Common Commands

```bash
# See all available targets
make help

# Check if Python dependencies are installed
make check-deps

# Install Python dependencies if needed
make install-deps

# Archive your results
make archive-results

# Clean everything and start fresh
make clean-all
```

## Customize for Your GPU

```bash
# For Ampere (A100, RTX 30xx)
make CUDA_ARCH=80 workflow-quick

# For Ada (RTX 40xx)
make CUDA_ARCH=89 workflow-quick
```

## Example Output

```
=== RQ1: Stream Scalability Analysis ===

Stream Count vs Metrics:
        concurrent_rate       throughput
                   mean   std       mean  std
streams
1                 72.60  0.10   70153.92  2.1
2                 36.00  0.17   67584.51  3.5
4                 17.93  0.06   66692.71  1.8
8                  9.03  0.06   65507.90  2.3
...

✓ Optimal stream count: 1 (throughput: 70153.92 kernels/sec)
✓ Saturation point: ~2 streams
```

## Troubleshooting

**"Command not found"**
```bash
# Make sure you're in the right directory
cd scheduler/multi-stream
```

**"Missing dependencies"**
```bash
make install-deps
```

**"CUDA out of memory"**
```bash
# Run with smaller workloads
./multi_stream_bench --streams 4 --size 262144
```

## Next Steps

1. **Quick Test**: `make workflow-quick` (5 min)
2. **Review Results**: `make view-results`
3. **Read Research Questions**: See `RESEARCH_QUESTIONS.md`
4. **Full Experiments**: `make workflow-full` (1 hour)
5. **Custom Analysis**: Modify `analyze_results.py`

## Manual Control

If you prefer manual control:

```bash
# Build
make

# Run experiments manually
python3 experiment_driver.py --experiments RQ1 RQ2 --trials 10

# Analyze manually
python3 analyze_results.py --experiments RQ1 RQ2

# View
cat results/ANALYSIS_REPORT.md
ls results/figures/
```

## Pro Tips

**Lock GPU clocks for reproducibility:**
```bash
sudo nvidia-smi -pm 1
sudo nvidia-smi -lgc 1500
```

**Monitor GPU during experiments:**
```bash
watch -n 1 nvidia-smi
```

**Archive results with timestamp:**
```bash
make archive-results
# Creates: results_YYYYMMDD_HHMMSS.tar.gz
```

**Parallel experiment execution:**
```bash
# Run multiple RQs in parallel (different terminals)
make experiment-rq1 &
make experiment-rq2 &
make experiment-rq3 &
wait
make analyze
```

## Getting Help

- `make help` - Show all Makefile targets
- `EXPERIMENT_GUIDE.md` - Detailed experiment guide
- `RESEARCH_QUESTIONS.md` - Research question definitions
- `README.md` - Benchmark implementation details

Happy experimenting! 🚀
