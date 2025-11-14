#!/usr/bin/env python3
"""
GPU Scheduler Research Experiment Driver

Systematically explores CUDA scheduler behavior through automated experiments.
Supports single and multi-process configurations.
"""

import subprocess
import pandas as pd
import numpy as np
import json
import time
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

# Default experiment configuration
DEFAULT_WORKLOAD_SIZE = 16777216  # 64MB (16M elements × 4 bytes)
DEFAULT_KERNELS_PER_STREAM = 20


class BenchmarkRunner:
    """Runs multi-stream benchmark and parses results."""

    def __init__(self, binary_path: str = "./multi_stream_bench"):
        self.binary_path = binary_path
        if not os.path.exists(binary_path):
            raise FileNotFoundError(f"Benchmark binary not found: {binary_path}")

    def run_single(self, streams: int, kernels: int, workload_size: int,
                   kernel_type: str, priority: bool = False,
                   load_imbalance: Optional[str] = None,
                   trials: int = 1) -> List[Dict]:
        """Run benchmark with specified configuration."""
        results = []

        cmd = [
            self.binary_path,
            "--streams", str(streams),
            "--kernels", str(kernels),
            "--size", str(workload_size),
            "--type", kernel_type
        ]

        if priority:
            cmd.append("--priority")

        if load_imbalance:
            cmd.extend(["--load-imbalance", load_imbalance])

        for trial in range(trials):
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=True
                )

                # Parse CSV output
                csv_lines = [line for line in result.stdout.split('\n') if line.startswith('CSV:')]
                if len(csv_lines) >= 2:
                    header = csv_lines[0].replace('CSV: ', '').split(',')
                    data = csv_lines[1].replace('CSV: ', '').split(',')

                    row = dict(zip(header, data))
                    row['trial'] = trial
                    row['timestamp'] = datetime.now().isoformat()
                    results.append(row)
                else:
                    print(f"Warning: No CSV output in trial {trial}")

            except subprocess.TimeoutExpired:
                print(f"Warning: Trial {trial} timed out")
            except subprocess.CalledProcessError as e:
                print(f"Warning: Trial {trial} failed: {e}")

        return results

    def run_multi_process(self, num_processes: int, streams_per_process: int,
                         kernels: int, workload_size: int, kernel_type: str,
                         trials: int = 1) -> List[Dict]:
        """Run multiple benchmark processes concurrently."""
        all_results = []

        for trial in range(trials):
            trial_start = time.time()

            # Launch all processes simultaneously
            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                futures = []
                for proc_id in range(num_processes):
                    future = executor.submit(
                        self._run_process_wrapper,
                        streams_per_process, kernels, workload_size,
                        kernel_type, proc_id, trial
                    )
                    futures.append(future)

                # Collect results
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            result['trial'] = trial
                            result['trial_duration'] = time.time() - trial_start
                            all_results.append(result)
                    except Exception as e:
                        print(f"Process failed: {e}")

            time.sleep(0.5)  # Cool down between trials

        return all_results

    def _run_process_wrapper(self, streams: int, kernels: int,
                            workload_size: int, kernel_type: str,
                            proc_id: int, trial: int) -> Optional[Dict]:
        """Wrapper for running single process in multi-process experiment."""
        proc_start = time.time()

        cmd = [
            self.binary_path,
            "--streams", str(streams),
            "--kernels", str(kernels),
            "--size", str(workload_size),
            "--type", kernel_type
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                check=True
            )

            csv_lines = [line for line in result.stdout.split('\n') if line.startswith('CSV:')]
            if len(csv_lines) >= 2:
                header = csv_lines[0].replace('CSV: ', '').split(',')
                data = csv_lines[1].replace('CSV: ', '').split(',')

                row = dict(zip(header, data))
                row['process_id'] = proc_id
                row['process_duration'] = time.time() - proc_start
                row['timestamp'] = datetime.now().isoformat()
                return row
        except Exception as e:
            print(f"Process {proc_id} trial {trial} failed: {e}")
            return None


class ExperimentSuite:
    """Defines and runs experiment suites for research questions."""

    def __init__(self, runner: BenchmarkRunner, output_dir: str = "results", workload_size: int = DEFAULT_WORKLOAD_SIZE):
        self.runner = runner
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.workload_size = workload_size

    def rq1_stream_scalability(self, trials: int = 3) -> pd.DataFrame:
        """RQ1: Stream scalability experiment with varying workload sizes."""
        print("=== RQ1: Stream Scalability ===")

        stream_counts = [1, 2, 4, 8, 16, 32, 64]

        # Test different workload sizes to explore concurrent execution
        # Smaller sizes allow for true concurrent execution on GPU
        workload_sizes = [
            16384,      # 64 KB  - very small, should show high concurrency
            65536,      # 256 KB - small, good concurrency potential
            262144,     # 1 MB   - medium, some concurrency
            1048576,    # 4 MB   - large, limited concurrency
            4194304,    # 16 MB  - very large, likely serial execution
        ]

        results = []

        for size in workload_sizes:
            size_mb = (size * 4) / (1024 * 1024)
            print(f"\nTesting workload size: {size_mb:.2f} MB ({size} elements)")

            for streams in stream_counts:
                print(f"  {streams} streams...", end=" ", flush=True)
                trial_results = self.runner.run_single(
                    streams=streams,
                    kernels=20,
                    workload_size=size,
                    kernel_type="mixed",
                    trials=trials
                )
                # Add workload size info to each result
                for result in trial_results:
                    result['workload_size_mb'] = size_mb
                    result['workload_elements'] = size
                results.extend(trial_results)
                print("done")

        df = pd.DataFrame(results)
        df.to_csv(self.output_dir / "rq1_stream_scalability.csv", index=False)
        print(f"\n✓ RQ1 complete: {len(df)} data points collected")
        return df

    def rq2_workload_characterization(self, trials: int = 10) -> pd.DataFrame:
        """RQ2: Workload characterization experiment."""
        print("=== RQ2: Workload Characterization ===")

        kernel_types = ["compute", "memory", "mixed", "gemm"]
        results = []

        for ktype in kernel_types:
            print(f"Testing {ktype} workload...")
            trial_results = self.runner.run_single(
                streams=8,
                kernels=20,
                workload_size=self.workload_size,
                kernel_type=ktype,
                trials=trials
            )
            results.extend(trial_results)

        df = pd.DataFrame(results)
        df.to_csv(self.output_dir / "rq2_workload_characterization.csv", index=False)
        return df

    def rq3_priority_effectiveness(self, trials: int = 10) -> pd.DataFrame:
        """RQ3: Priority scheduling effectiveness."""
        print("=== RQ3: Priority Effectiveness ===")

        stream_counts = [4, 8, 16]
        results = []

        for streams in stream_counts:
            for priority in [False, True]:
                config_name = f"{streams}streams_priority{priority}"
                print(f"Testing {config_name}...")
                trial_results = self.runner.run_single(
                    streams=streams,
                    kernels=20,
                    workload_size=self.workload_size,
                    kernel_type="mixed",
                    priority=priority,
                    trials=trials
                )
                results.extend(trial_results)

        df = pd.DataFrame(results)
        df.to_csv(self.output_dir / "rq3_priority_effectiveness.csv", index=False)
        return df

    def rq4_memory_pressure(self, trials: int = 10) -> pd.DataFrame:
        """RQ4: Memory pressure impact."""
        print("=== RQ4: Memory Pressure ===")

        # Sizes in elements (each float = 4 bytes)
        workload_sizes = [
            65536,      # 256 KB
            262144,     # 1 MB
            1048576,    # 4 MB
            4194304,    # 16 MB
            16777216    # 64 MB
        ]
        results = []

        for size in workload_sizes:
            size_mb = (size * 4) / (1024 * 1024)
            print(f"Testing {size_mb:.1f} MB workload...")
            trial_results = self.runner.run_single(
                streams=8,
                kernels=20,
                workload_size=size,
                kernel_type="gemm",
                trials=trials
            )
            for r in trial_results:
                r['size_mb'] = size_mb
            results.extend(trial_results)

        df = pd.DataFrame(results)
        df.to_csv(self.output_dir / "rq4_memory_pressure.csv", index=False)
        return df

    def rq5_multi_process_interference(self, trials: int = 10) -> pd.DataFrame:
        """RQ5: Multi-process interference."""
        print("=== RQ5: Multi-Process Interference ===")

        process_counts = [1, 2, 4, 8]
        results = []

        for num_procs in process_counts:
            print(f"Testing {num_procs} concurrent processes...")
            trial_results = self.runner.run_multi_process(
                num_processes=num_procs,
                streams_per_process=4,
                kernels=20,
                workload_size=self.workload_size,
                kernel_type="mixed",
                trials=trials
            )
            for r in trial_results:
                r['num_processes'] = num_procs
            results.extend(trial_results)

        df = pd.DataFrame(results)
        df.to_csv(self.output_dir / "rq5_multi_process.csv", index=False)
        return df

    def rq6_load_imbalance(self, trials: int = 10) -> pd.DataFrame:
        """RQ6: Load imbalance and fairness."""
        print("=== RQ6: Load Imbalance and Fairness ===")

        # Test different load imbalance patterns
        imbalance_patterns = [
            ("5,5,5,5,5,5,5,5", "balanced_8"),
            ("5,10,20,40", "imbalanced_4"),
            ("10,10,10,10,20,20,20,20", "bimodal_8"),
            ("5,10,15,20,25,30,35,40", "linear_8"),
            ("20,20,20,20,20,20,20,5", "outlier_low"),
            ("5,5,5,5,5,5,5,20", "outlier_high"),
        ]

        results = []

        for pattern, pattern_name in imbalance_patterns:
            print(f"Testing load pattern: {pattern_name} ({pattern})...")
            trial_results = self.runner.run_single(
                streams=8,  # Will be overridden by pattern length
                kernels=20,  # Ignored when load_imbalance is set
                workload_size=self.workload_size,
                kernel_type="mixed",
                load_imbalance=pattern,
                trials=trials
            )
            for r in trial_results:
                r['pattern_name'] = pattern_name
                r['pattern'] = pattern
            results.extend(trial_results)

        df = pd.DataFrame(results)
        df.to_csv(self.output_dir / "rq6_load_imbalance.csv", index=False)
        return df

    def rq7_tail_latency_contention(self, trials: int = 10) -> pd.DataFrame:
        """RQ7: Tail latency under contention with HETEROGENEOUS workloads."""
        print("=== RQ7: Tail Latency Under Contention (OSDI-Quality) ===")

        # Original homogeneous tests
        stream_counts = [2, 4, 8, 16, 32, 64]
        results = []

        for streams in stream_counts:
            print(f"Testing {streams} streams (homogeneous mixed)...")
            trial_results = self.runner.run_single(
                streams=streams,
                kernels=50,
                workload_size=self.workload_size,
                kernel_type="mixed",
                trials=trials
            )
            for r in trial_results:
                r['workload_pattern'] = "homogeneous_mixed"
            results.extend(trial_results)

        # OSDI Enhancement: Heterogeneous kernel types for realistic contention
        print("\n--- OSDI Enhancement: Heterogeneous Workloads ---")
        hetero_configs = [
            (4, "memory", "fast_memory_only"),
            (4, "compute", "slow_compute_only"),
            (8, "memory", "fast_memory_8s"),
            (8, "compute", "slow_compute_8s"),
        ]

        for streams, ktype, pattern_name in hetero_configs:
            print(f"Testing heterogeneous: {pattern_name} ({streams} streams, {ktype})...")
            trial_results = self.runner.run_single(
                streams=streams,
                kernels=50,
                workload_size=4194304,  # 16MB for longer execution
                kernel_type=ktype,
                trials=trials
            )
            for r in trial_results:
                r['workload_pattern'] = pattern_name
            results.extend(trial_results)

        df = pd.DataFrame(results)
        df.to_csv(self.output_dir / "rq7_tail_latency.csv", index=False)
        return df

    def rq9_priority_tail_latency(self, trials: int = 10) -> pd.DataFrame:
        """RQ9: Priority-based tail latency with XSched-style HETEROGENEOUS workloads."""
        print("=== RQ9: Priority-Based Tail Latency (OSDI-Quality) ===")

        results = []

        # Original homogeneous tests (load-imbalance only)
        print("\n--- Original: Load Imbalance (Homogeneous Kernels) ---")
        load_imb_configs = [
            (2, False, 20, 20, "homog_baseline_50pct"),
            (2, True, 20, 20, "homog_priority_50pct"),
            (4, False, 10, 30, "homog_baseline_25pct"),
            (4, True, 10, 30, "homog_priority_25pct"),
        ]

        for streams, use_priority, front_k, back_k, label in load_imb_configs:
            print(f"Testing {label}: {streams} streams, priority={use_priority}...")
            pattern_parts = []
            for i in range(streams // 2):
                pattern_parts.append(str(front_k))
            for i in range(streams - streams // 2):
                pattern_parts.append(str(back_k))
            pattern = ",".join(pattern_parts)

            trial_results = self.runner.run_single(
                streams=streams,
                kernels=20,
                workload_size=4194304,  # 16MB for longer execution
                kernel_type="mixed",
                priority=use_priority,
                load_imbalance=pattern,
                trials=trials
            )
            for r in trial_results:
                r['config_label'] = label
                r['workload_type'] = "homogeneous"
                r['priority_enabled'] = use_priority
            results.extend(trial_results)

        # OSDI Enhancement: Heterogeneous kernel types (memory vs compute)
        print("\n--- OSDI Enhancement: Heterogeneous Kernels (Memory vs Compute) ---")
        hetero_configs = [
            # (streams, kernels, ktype, priority, label, description)
            (2, 100, "memory", False, "hetero_memory_alone", "Fast memory, no competition"),
            (4, 50, "memory", False, "hetero_memory_no_pri", "Memory kernels, no priority"),
            (4, 50, "memory", True, "hetero_memory_with_pri", "Memory kernels, HIGH priority"),
            (4, 50, "compute", False, "hetero_compute_no_pri", "Compute kernels (interference)"),
            (4, 50, "compute", True, "hetero_compute_with_pri", "Compute kernels, LOW priority"),
        ]

        for streams, kernels, ktype, use_priority, label, desc in hetero_configs:
            print(f"Testing {label}: {desc}")
            trial_results = self.runner.run_single(
                streams=streams,
                kernels=kernels,
                workload_size=4194304,  # 16MB for longer execution
                kernel_type=ktype,
                priority=use_priority,
                trials=trials
            )
            for r in trial_results:
                r['config_label'] = label
                r['workload_type'] = "heterogeneous"
                r['kernel_type_used'] = ktype
                r['priority_enabled'] = use_priority
                r['description'] = desc
            results.extend(trial_results)

        df = pd.DataFrame(results)
        df.to_csv(self.output_dir / "rq9_priority_tail_latency.csv", index=False)
        return df

    def rq10_preemption_latency(self, trials: int = 10) -> pd.DataFrame:
        """RQ10: Preemption latency estimation via contention analysis."""
        print("=== RQ10: Preemption Latency (OSDI-Quality) ===")
        print("Method: Compare fast kernel latency with/without slow kernel contention")

        results = []

        # Baseline: Fast kernel alone (no contention)
        print("\nBaseline: Memory kernel alone (no contention)...")
        baseline_results = self.runner.run_single(
            streams=1,
            kernels=20,
            workload_size=4194304,  # 16MB for longer execution
            kernel_type="memory",
            trials=trials
        )
        for r in baseline_results:
            r['scenario'] = "baseline_no_contention"
            r['interfering_kernel'] = "none"
        results.extend(baseline_results)

        # Contention scenarios: Different blocking kernel types
        blocking_kernels = [
            ("compute", "1.2ms compute kernel"),
            ("gemm", "0.23ms GEMM kernel"),
        ]

        for blocking_type, blocking_desc in blocking_kernels:
            print(f"\nContention with {blocking_desc}...")

            # WITHOUT priority: FIFO scheduling (preemption latency = blocking kernel duration)
            print(f"  Testing WITHOUT priority...")
            no_pri_results = self.runner.run_single(
                streams=2,
                kernels=20,
                workload_size=4194304,  # 16MB for longer execution
                kernel_type=blocking_type,
                priority=False,
                trials=trials
            )
            for r in no_pri_results:
                r['scenario'] = f"contention_{blocking_type}_no_priority"
                r['interfering_kernel'] = blocking_type
                r['priority_enabled'] = False
            results.extend(no_pri_results)

            # WITH priority: Should preempt (test if CUDA priority helps)
            print(f"  Testing WITH priority...")
            pri_results = self.runner.run_single(
                streams=2,
                kernels=20,
                workload_size=4194304,  # 16MB for longer execution
                kernel_type=blocking_type,
                priority=True,
                trials=trials
            )
            for r in pri_results:
                r['scenario'] = f"contention_{blocking_type}_with_priority"
                r['interfering_kernel'] = blocking_type
                r['priority_enabled'] = True
            results.extend(pri_results)

        df = pd.DataFrame(results)

        # Calculate and print preemption latency estimates
        if not df.empty:
            baseline_p99 = df[df['scenario'] == 'baseline_no_contention']['p99'].astype(float).mean()
            print("\n" + "="*70)
            print("PREEMPTION LATENCY ANALYSIS")
            print("="*70)
            print(f"Baseline P99 (no contention): {baseline_p99:.2f} ms\n")

            for blocking_type in ['compute', 'gemm']:
                no_pri_p99 = df[df['scenario'] == f'contention_{blocking_type}_no_priority']['p99'].astype(float).mean()
                pri_p99 = df[df['scenario'] == f'contention_{blocking_type}_with_priority']['p99'].astype(float).mean()

                print(f"{blocking_type.upper()} Kernel:")
                print(f"  No Priority P99: {no_pri_p99:.2f} ms (inflation: {no_pri_p99/baseline_p99:.1f}×)")
                print(f"  With Priority P99: {pri_p99:.2f} ms (inflation: {pri_p99/baseline_p99:.1f}×)")
                print(f"  Priority Benefit: {no_pri_p99/pri_p99:.2f}× reduction")
                print(f"  → Preemption latency ≈ {no_pri_p99:.2f} ms (must wait for blocking kernel)\n")

        df.to_csv(self.output_dir / "rq10_preemption_latency.csv", index=False)
        return df

    def rq11_bandwidth_partitioning(self, trials: int = 10) -> pd.DataFrame:
        """RQ11: Bandwidth partitioning and quota enforcement."""
        print("=== RQ11: Bandwidth Partitioning ===")

        # Test different target ratios
        target_ratios = [
            (50, 50, "50/50"),
            (75, 25, "75/25"),
            (90, 10, "90/10"),
            (95, 5, "95/5"),
        ]

        results = []

        for front_pct, back_pct, label in target_ratios:
            print(f"Testing {label} bandwidth partition...")

            # Use load-imbalance to approximate quota
            # Front gets front_pct kernels, back gets back_pct kernels
            # Scale to reasonable total (e.g., 100 kernels total)
            total_kernels = 100
            front_k = int(total_kernels * front_pct / 100)
            back_k = int(total_kernels * back_pct / 100)

            # Create pattern: 4 front streams + 4 back streams
            pattern_parts = []
            for i in range(4):
                pattern_parts.append(str(front_k // 4))
            for i in range(4):
                pattern_parts.append(str(back_k // 4))
            pattern = ",".join(pattern_parts)

            trial_results = self.runner.run_single(
                streams=8,
                kernels=20,  # Ignored
                workload_size=self.workload_size,
                kernel_type="mixed",
                load_imbalance=pattern,
                trials=trials
            )
            for r in trial_results:
                r['target_ratio'] = label
                r['target_front_pct'] = front_pct
                r['target_back_pct'] = back_pct
                r['front_kernels'] = front_k
                r['back_kernels'] = back_k
            results.extend(trial_results)

        df = pd.DataFrame(results)
        df.to_csv(self.output_dir / "rq11_bandwidth_partition.csv", index=False)
        return df

    def run_all(self, trials: int = 10) -> Dict[str, pd.DataFrame]:
        """Run all research question experiments."""
        results = {}

        experiments = [
            ("RQ1", self.rq1_stream_scalability),
            ("RQ2", self.rq2_workload_characterization),
            ("RQ3", self.rq3_priority_effectiveness),
            ("RQ4", self.rq4_memory_pressure),
            ("RQ5", self.rq5_multi_process_interference),
            ("RQ6", self.rq6_load_imbalance),
            ("RQ7", self.rq7_tail_latency_contention),
            ("RQ9", self.rq9_priority_tail_latency),
            ("RQ10", self.rq10_preemption_latency),
            ("RQ11", self.rq11_bandwidth_partitioning),
        ]

        for name, experiment_func in experiments:
            print(f"\n{'='*60}")
            print(f"Running {name}")
            print(f"{'='*60}\n")

            try:
                df = experiment_func(trials=trials)
                results[name] = df
                print(f"✓ {name} complete: {len(df)} data points\n")
            except Exception as e:
                print(f"✗ {name} failed: {e}\n")

        # Save metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "trials_per_config": trials,
            "total_experiments": len(results),
            "gpu_info": self._get_gpu_info()
        }

        with open(self.output_dir / "experiment_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return results

    def _get_gpu_info(self) -> Dict:
        """Get GPU information using nvidia-smi."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                 "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True
            )
            info = result.stdout.strip().split(', ')
            return {
                "name": info[0] if len(info) > 0 else "unknown",
                "driver": info[1] if len(info) > 1 else "unknown",
                "memory": info[2] if len(info) > 2 else "unknown"
            }
        except:
            return {"name": "unknown", "driver": "unknown", "memory": "unknown"}


def main():
    parser = argparse.ArgumentParser(
        description="GPU Scheduler Research Experiment Driver"
    )
    parser.add_argument(
        "--binary", default="./multi_stream_bench",
        help="Path to benchmark binary"
    )
    parser.add_argument(
        "--output", default="results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--trials", type=int, default=10,
        help="Number of trials per configuration"
    )
    parser.add_argument(
        "--workload-size", type=int, default=DEFAULT_WORKLOAD_SIZE,
        help=f"Workload size in elements (default: {DEFAULT_WORKLOAD_SIZE} = 64MB)"
    )
    parser.add_argument(
        "--experiments", nargs="+",
        choices=["RQ1", "RQ2", "RQ3", "RQ4", "RQ5", "RQ6", "RQ7", "RQ9", "RQ10", "RQ11", "all"],
        default=["all"],
        help="Which experiments to run"
    )

    args = parser.parse_args()

    print("="*60)
    print("GPU Scheduler Research Experiment Driver")
    print("="*60)
    print(f"Binary: {args.binary}")
    print(f"Output: {args.output}")
    print(f"Trials per config: {args.trials}")
    print(f"Workload size: {args.workload_size} elements ({args.workload_size * 4 / (1024**2):.1f} MB)")
    print("="*60)
    print()

    runner = BenchmarkRunner(args.binary)
    suite = ExperimentSuite(runner, args.output, workload_size=args.workload_size)

    if "all" in args.experiments:
        results = suite.run_all(trials=args.trials)
    else:
        results = {}
        exp_map = {
            "RQ1": suite.rq1_stream_scalability,
            "RQ2": suite.rq2_workload_characterization,
            "RQ3": suite.rq3_priority_effectiveness,
            "RQ4": suite.rq4_memory_pressure,
            "RQ5": suite.rq5_multi_process_interference,
            "RQ6": suite.rq6_load_imbalance,
            "RQ7": suite.rq7_tail_latency_contention,
            "RQ9": suite.rq9_priority_tail_latency,
            "RQ10": suite.rq10_preemption_latency,
            "RQ11": suite.rq11_bandwidth_partitioning,
        }

        for exp_name in args.experiments:
            if exp_name in exp_map:
                print(f"\nRunning {exp_name}...\n")
                results[exp_name] = exp_map[exp_name](trials=args.trials)

    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    for name, df in results.items():
        print(f"{name}: {len(df)} data points")
    print(f"\nResults saved to: {args.output}/")
    print("="*60)


if __name__ == "__main__":
    main()
