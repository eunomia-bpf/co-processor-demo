# GPU Scheduler Experiment Framework Guide

Complete guide for running systematic GPU scheduler experiments and analysis.

## Quick Start

```bash
# 1. Build the benchmark
make

# 2. Run a quick test (3 trials)
python3 experiment_driver.py --experiments RQ1 --trials 3

# 3. Analyze results
python3 analyze_results.py --experiments RQ1

# 4. View figures
ls results/figures/
```

## Full Experiment Suite

### Running All Experiments

```bash
# Run all research questions with 10 trials each (recommended for publication)
python3 experiment_driver.py --experiments all --trials 10

# This will take ~30-60 minutes depending on your GPU
```

### Running Individual Experiments

```bash
# RQ1: Stream Scalability (tests 1, 2, 4, 8, 16, 32, 64 streams)
python3 experiment_driver.py --experiments RQ1 --trials 10

# RQ2: Workload Characterization (compute, memory, mixed, gemm)
python3 experiment_driver.py --experiments RQ2 --trials 10

# RQ3: Priority Effectiveness (with/without priorities)
python3 experiment_driver.py --experiments RQ3 --trials 10

# RQ4: Memory Pressure (256KB to 64MB)
python3 experiment_driver.py --experiments RQ4 --trials 10

# RQ5: Multi-Process Interference (1, 2, 4, 8 processes)
python3 experiment_driver.py --experiments RQ5 --trials 10

# RQ7: Tail Latency (P50, P95, P99 analysis)
python3 experiment_driver.py --experiments RQ7 --trials 10
```

### Running Multiple Experiments

```bash
# Run subset of experiments
python3 experiment_driver.py --experiments RQ1 RQ2 RQ5 --trials 10
```

## Analysis

### Analyze Results

```bash
# Analyze all available results
python3 analyze_results.py --experiments all

# Analyze specific experiments
python3 analyze_results.py --experiments RQ1 RQ2
```

### Output Files

After running experiments, you'll find:

```
results/
├── rq1_stream_scalability.csv          # Raw data
├── rq2_workload_characterization.csv   # Raw data
├── rq3_priority_effectiveness.csv      # Raw data
├── rq4_memory_pressure.csv             # Raw data
├── rq5_multi_process.csv               # Raw data
├── rq7_tail_latency.csv                # Raw data
├── experiment_metadata.json            # GPU info, timestamp
├── ANALYSIS_REPORT.md                  # Summary report
└── figures/                            # Visualizations
    ├── rq1_stream_scalability.png
    ├── rq2_workload_characterization.png
    ├── rq3_priority_effectiveness.png
    ├── rq4_memory_pressure.png
    ├── rq5_multi_process.png
    └── rq7_tail_latency.png
```

## Research Questions Explained

### RQ1: Stream Scalability
**Goal**: Find optimal number of streams for maximum concurrency.

**Key Metrics**:
- Concurrent execution rate vs stream count
- Throughput saturation point
- GPU utilization

**Expected Insight**: Identify hardware-imposed limits on parallel execution.

---

### RQ2: Workload Characterization
**Goal**: Understand how different workload types affect scheduler behavior.

**Workloads Tested**:
- `compute`: Heavy computation (sqrt, sin, cos)
- `memory`: Memory-bound strided access
- `mixed`: Balanced compute + memory
- `gemm`: Tiled matrix multiplication

**Expected Insight**: Which workloads benefit most from multi-stream execution?

---

### RQ3: Priority Scheduling
**Goal**: Evaluate CUDA stream priority effectiveness.

**Tests**:
- Priority disabled (default)
- Priority enabled (streams get different priorities)

**Key Metrics**:
- Priority inversion count
- Jain's fairness index
- Per-stream latency

**Expected Insight**: Does CUDA priority mechanism actually work?

---

### RQ4: Memory Pressure
**Goal**: Understand memory subsystem limits on concurrency.

**Memory Sizes**: 256KB, 1MB, 4MB, 16MB, 64MB per stream

**Expected Insight**: At what memory allocation size does the memory subsystem become the bottleneck?

---

### RQ5: Multi-Process Interference
**Goal**: Study how multiple processes sharing GPU affect each other.

**Tests**: 1, 2, 4, 8 concurrent processes, each running 4 streams

**Key Metrics**:
- Per-process throughput
- System-wide aggregate throughput
- Inter-process fairness

**Expected Insight**: Does the GPU scheduler provide fair sharing across processes?

---

### RQ7: Tail Latency
**Goal**: Characterize predictability and QoS under load.

**Key Metrics**:
- P50, P95, P99 latencies
- P99/P50 ratio (tail amplification)

**Expected Insight**: How does tail latency degrade with increased contention?

## Custom Experiments

### Manual Benchmark Invocation

```bash
# Direct benchmark call
./multi_stream_bench --streams 8 --kernels 20 --type gemm --priority

# Parse output
./multi_stream_bench --streams 8 --kernels 20 --type mixed | grep "CSV:"
```

### Writing Custom Experiments

```python
from experiment_driver import BenchmarkRunner

runner = BenchmarkRunner("./multi_stream_bench")

# Custom single experiment
results = runner.run_single(
    streams=16,
    kernels=50,
    workload_size=4194304,  # 16 MB
    kernel_type="gemm",
    priority=True,
    trials=5
)

# Custom multi-process experiment
results = runner.run_multi_process(
    num_processes=4,
    streams_per_process=8,
    kernels=20,
    workload_size=1048576,
    kernel_type="mixed",
    trials=5
)
```

## Advanced Usage

### Parameter Sweeps

Create your own parameter sweep:

```python
import pandas as pd
from experiment_driver import BenchmarkRunner

runner = BenchmarkRunner()
results = []

# Sweep workload size and stream count
for size in [262144, 1048576, 4194304]:
    for streams in [4, 8, 16]:
        data = runner.run_single(
            streams=streams,
            kernels=20,
            workload_size=size,
            kernel_type="gemm",
            trials=10
        )
        results.extend(data)

df = pd.DataFrame(results)
df.to_csv("custom_sweep.csv", index=False)
```

### Monitoring GPU State

```bash
# Monitor GPU while experiments run
watch -n 1 nvidia-smi

# Log GPU metrics
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used \
           --format=csv -l 1 > gpu_log.csv &

# Run experiments
python3 experiment_driver.py --experiments all --trials 10

# Kill monitoring
killall nvidia-smi
```

## Tips for Publication-Quality Results

### 1. Statistical Rigor
```bash
# Use 10+ trials for low variance
python3 experiment_driver.py --experiments all --trials 20

# Check for outliers in results/
# Remove outlier trials if justified
```

### 2. Thermal Stability
```bash
# Let GPU cool between heavy experiments
sleep 60

# Monitor temperature
nvidia-smi --query-gpu=temperature.gpu --format=csv -l 1
```

### 3. Reproducibility
```bash
# Lock GPU clocks (requires root)
sudo nvidia-smi -pm 1
sudo nvidia-smi -lgc 1500  # Lock GPU clock to 1500 MHz

# Record system state
nvidia-smi --query-gpu=name,driver_version,clocks.gr,clocks.mem \
           --format=csv > system_info.csv
```

### 4. Comprehensive Data Collection
```bash
# Run all experiments with high trial count
python3 experiment_driver.py --experiments all --trials 20 \
    > experiment_log.txt 2>&1

# Archive results with timestamp
tar -czf results_$(date +%Y%m%d_%H%M%S).tar.gz results/
```

## Troubleshooting

### "Binary not found"
```bash
# Make sure benchmark is built
make clean && make
ls -l multi_stream_bench
```

### "CUDA out of memory"
```bash
# Reduce workload size or stream count
./multi_stream_bench --streams 4 --size 262144

# Check available memory
nvidia-smi
```

### "Results file not found"
```bash
# Make sure experiments ran successfully
ls results/*.csv

# Check experiment log for errors
tail experiment_log.txt
```

### Analysis Fails
```bash
# Install required Python packages
pip3 install pandas numpy matplotlib seaborn

# Check data integrity
head -n 5 results/rq1_stream_scalability.csv
```

## Example Workflow

Complete workflow for an OSDI paper:

```bash
# 1. Prepare system
make clean && make
python3 -c "import pandas, numpy, matplotlib, seaborn" || \
    pip3 install pandas numpy matplotlib seaborn

# 2. Run pilot study (quick, 3 trials)
python3 experiment_driver.py --experiments all --trials 3
python3 analyze_results.py --experiments all

# 3. Review pilot results
cat results/ANALYSIS_REPORT.md
ls results/figures/

# 4. Run full study (20 trials for publication)
python3 experiment_driver.py --experiments all --trials 20 \
    > experiment_full.log 2>&1

# 5. Full analysis
python3 analyze_results.py --experiments all

# 6. Archive results
tar -czf paper_results_$(date +%Y%m%d).tar.gz results/ \
    experiment_full.log

# 7. View all figures
eog results/figures/*.png  # or your favorite image viewer
```

## Citation

If you use this benchmark framework in your research, please cite:

```bibtex
@misc{gpu-scheduler-benchmark,
  title={Multi-Stream GPU Scheduler Micro-Benchmark},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/yourrepo}}
}
```

## Further Reading

- [RESEARCH_QUESTIONS.md](RESEARCH_QUESTIONS.md) - Detailed research question definitions
- [README.md](README.md) - Benchmark implementation details
- CUDA Programming Guide: Stream Priorities
- NVIDIA Multi-Process Service (MPS) Documentation
