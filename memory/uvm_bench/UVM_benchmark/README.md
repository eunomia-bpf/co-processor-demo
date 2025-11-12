# UVMBench - CUDA Unified Virtual Memory Benchmark Suite

A comprehensive benchmark suite for researching Unified Virtual Memory (UVM) in NVIDIA GPUs.

## Quick Start

### Prerequisites
- CUDA 11.0+ (tested with CUDA 12.9)
- NVIDIA GPU with Compute Capability 7.5+
- Python 3.6+
- GCC 7.0+

### One-Command Setup and Run

```bash
# 1. Fix and build all benchmarks
python3 fix_and_build_all.py

# 2. Run all working benchmarks
python3 run_all_benchmarks.py --mode uvm

# 3. View results
cat results/*/summary.csv
```

## What Works (12 benchmarks)

✅ **Machine Learning** (3):
- CNN - Convolutional Neural Network
- K-means - Clustering algorithm
- Logistic Regression

✅ **Linear Algebra** (9 Polybench):
- 2MM, 3MM - Matrix multiplications
- ATAX, BICG - Linear solvers
- GEMM, GESUMMV, MVT - Matrix operations
- SYR2K, SYRK - Symmetric rank operations

## Repository Structure

```
UVM_benchmark/
├── run_all_benchmarks.py      # Main runner script
├── fix_and_build_all.py       # Auto-fix and build
├── create_run_scripts.py      # Generate run scripts
│
├── README.md                  # This file (quick start)
├── README_DETAILED.md         # Comprehensive guide
├── BENCHMARK_STATUS.md        # Detailed status report
├── RUNNING_BENCHMARKS.md      # Usage instructions
├── KNOWN_BROKEN.txt           # Known issues
│
├── UVM_benchmarks/            # UVM version (cudaMallocManaged)
├── non_UVM_benchmarks/        # Traditional CUDA
└── data/                      # Input datasets
```

## Documentation Files

- **README.md** (this file) - Quick start
- **README_DETAILED.md** - Full documentation (726 lines)
- **BENCHMARK_STATUS.md** - Status of all benchmarks
- **RUNNING_BENCHMARKS.md** - Detailed usage guide
- **KNOWN_BROKEN.txt** - Known issues and workarounds

## Usage Examples

### Run Single Benchmark
```bash
cd UVM_benchmarks/kmeans
./run
# Output: CUDA Took: 0.326s for 10000 points.
```

### Run All with Profiling
```bash
python3 run_all_benchmarks.py --mode both --profile
```

### Compare UVM vs non-UVM
```bash
python3 run_all_benchmarks.py --mode both
cat results/*/comparison.txt
```

## Results

After running, find results in `results/TIMESTAMP/`:
- `summary.csv` - All benchmark results
- `summary.json` - Detailed JSON format
- `comparison.txt` - UVM vs non-UVM performance
- `UVM_*.txt` - Individual benchmark outputs

## Current Status

- **12/33 benchmarks working** (36%)
- Successfully fixed: Build issues, data generation, problem sizes
- Known broken: BN, KNN (segfaults), Rodinia suite (build errors)

See `BENCHMARK_STATUS.md` for complete details.

## Scripts

| Script | Purpose |
|--------|---------|
| `run_all_benchmarks.py` | Run all benchmarks with timeout & logging |
| `fix_and_build_all.py` | Fix common issues and rebuild |
| `create_run_scripts.py` | Create missing run scripts |
| `profiling_wrapper.py` | Legacy nvprof-based profiling |

## Key Features

- ✅ Automatic data generation (k-means, BFS)
- ✅ Timeout handling
- ✅ CSV/JSON output
- ✅ UVM vs non-UVM comparison
- ✅ Detailed logging
- ✅ Error recovery

## Citation

```bibtex
@article{gu2020uvmbench,
  title={UVMBench: A Comprehensive Benchmark Suite for 
         Researching Unified Virtual Memory in GPUs},
  author={Gu, Yongbin and Wu, Wenxuan and Li, Yunfan and Chen, Lizhong},
  journal={arXiv preprint arXiv:2007.09822},
  year={2020}
}
```

## Links

- Original Repository: https://github.com/OSU-STARLAB/UVM_benchmark
- Paper: http://arxiv.org/abs/2007.09822

## Support

For detailed instructions, see:
- `README_DETAILED.md` - Complete setup guide
- `RUNNING_BENCHMARKS.md` - Usage examples
- `BENCHMARK_STATUS.md` - Status of each benchmark
