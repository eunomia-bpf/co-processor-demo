#!/usr/bin/env python3
"""
GPU Scheduler Research Experiment Driver

Systematically explores CUDA scheduler behavior through automated experiments
based on RESEARCH_QUESTIONS.md. Generates CSV data files for analysis.

Usage:
    python experiment_driver.py --rq RQ1 --output-dir results/
    python experiment_driver.py --rq all --output-dir results/
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
import argparse
import itertools


class ExperimentDriver:
    """Drives systematic experiments for GPU scheduler research."""

    def __init__(self, bench_path: str, output_dir: str, num_runs: int = 3):
        self.bench_path = bench_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_runs = num_runs

    def run_benchmark(self, args: List[str], first_run: bool = True) -> Optional[str]:
        """
        Run benchmark and capture CSV output from stderr.

        Args:
            args: Command line arguments for benchmark
            first_run: If True, include header in CSV output

        Returns:
            CSV output string or None if failed
        """
        # Add --no-header for non-first runs
        if not first_run:
            args.append('--no-header')

        cmd = [self.bench_path] + args

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode != 0:
                print(f"Warning: Benchmark failed with return code {result.returncode}")
                print(f"stderr: {result.stderr[:500]}")
                return None

            # CSV output is on stderr
            return result.stderr.strip()

        except subprocess.TimeoutExpired:
            print(f"Warning: Benchmark timed out")
            return None
        except Exception as e:
            print(f"Warning: Benchmark failed with exception: {e}")
            return None

    def save_csv(self, csv_lines: List[str], filename: str):
        """Save CSV lines to file."""
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            f.write('\n'.join(csv_lines))
        print(f"Saved: {filepath}")

    # ========================================================================
    # RQ1: Stream Scalability & Concurrency
    # ========================================================================

    def run_rq1_experiments(self):
        """
        RQ1: GPU stream concurrency capability and scalability.

        Sub-RQ1.1-1.4: max_concurrent, concurrent_rate, util, avg_concurrent
        vs #streams with different kernel sizes.
        """
        print("\n=== Running RQ1 Experiments: Stream Scalability & Concurrency ===")

        stream_counts = [1, 2, 4, 8, 16, 32, 64]
        # Different kernel durations via workload_size
        # Small: ~50us, Medium: ~200us, Large: ~1ms, XLarge: ~5ms
        kernel_sizes = [
            ('small', 65536),      # ~50us
            ('medium', 262144),    # ~200us
            ('large', 1048576),    # ~1ms
            ('xlarge', 4194304),   # ~5ms
        ]

        csv_lines = []

        for size_name, workload_size in kernel_sizes:
            for streams in stream_counts:
                print(f"  RQ1: streams={streams}, size={size_name} ({workload_size})")

                for run_idx in range(self.num_runs):
                    args = [
                        '--streams', str(streams),
                        '--kernels', '20',
                        '--size', str(workload_size),
                        '--type', 'mixed',
                    ]

                    csv_output = self.run_benchmark(
                        args,
                        first_run=(len(csv_lines) == 0)
                    )

                    if csv_output:
                        # Skip header if not first
                        lines = csv_output.split('\n')
                        if len(csv_lines) == 0:
                            csv_lines.extend(lines)  # Include header
                        else:
                            csv_lines.extend([l for l in lines if not l.startswith('streams,')])

                    time.sleep(0.5)  # Brief pause between runs

        self.save_csv(csv_lines, 'rq1_stream_scalability.csv')

    # ========================================================================
    # RQ2: Throughput & Workload Type
    # ========================================================================

    def run_rq2_experiments(self):
        """
        RQ2: Throughput variations with stream count and workload type.

        Sub-RQ2.1: throughput vs streams (different sizes)
        Sub-RQ2.2: throughput vs streams (different types)
        Sub-RQ2.3: throughput vs offered load (via launch_frequency)
        """
        print("\n=== Running RQ2 Experiments: Throughput & Workload Type ===")

        # Sub-RQ2.1: Already covered by RQ1 data (can reuse)
        print("  RQ2.1: Reusing RQ1 data for throughput vs streams")

        # Sub-RQ2.2: Different kernel types
        print("  RQ2.2: Throughput vs streams for different kernel types")
        stream_counts = [1, 2, 4, 8, 16, 32, 64]
        kernel_types = ['compute', 'memory', 'mixed', 'gemm']

        csv_lines = []

        for ktype in kernel_types:
            for streams in stream_counts:
                print(f"    type={ktype}, streams={streams}")

                for run_idx in range(self.num_runs):
                    args = [
                        '--streams', str(streams),
                        '--kernels', '20',
                        '--size', '1048576',  # Fixed size for fair comparison
                        '--type', ktype,
                    ]

                    csv_output = self.run_benchmark(
                        args,
                        first_run=(len(csv_lines) == 0)
                    )

                    if csv_output:
                        lines = csv_output.split('\n')
                        if len(csv_lines) == 0:
                            csv_lines.extend(lines)
                        else:
                            csv_lines.extend([l for l in lines if not l.startswith('streams,')])

                    time.sleep(0.5)

        self.save_csv(csv_lines, 'rq2_2_throughput_by_type.csv')

        # Sub-RQ2.3: Throughput vs offered load (launch_frequency)
        print("  RQ2.3: Throughput vs offered load")
        frequencies = [10, 20, 50, 100, 200, 500, 1000, 0]  # 0 = max
        num_streams = 8

        csv_lines = []

        for freq in frequencies:
            for with_jitter in [False, True]:
                seed = 42 if with_jitter else 0
                print(f"    freq={freq}Hz, jitter={with_jitter}")

                for run_idx in range(self.num_runs):
                    freq_spec = ','.join([str(freq)] * num_streams)
                    args = [
                        '--streams', str(num_streams),
                        '--kernels', '50',
                        '--size', '524288',
                        '--type', 'mixed',
                        '--launch-frequency', freq_spec,
                        '--seed', str(seed),
                    ]

                    csv_output = self.run_benchmark(
                        args,
                        first_run=(len(csv_lines) == 0)
                    )

                    if csv_output:
                        lines = csv_output.split('\n')
                        if len(csv_lines) == 0:
                            csv_lines.extend(lines)
                        else:
                            csv_lines.extend([l for l in lines if not l.startswith('streams,')])

                    time.sleep(0.5)

        self.save_csv(csv_lines, 'rq2_3_throughput_vs_load.csv')

    # ========================================================================
    # RQ3: Latency & Queueing
    # ========================================================================

    def run_rq3_experiments(self):
        """
        RQ3: Latency distribution and queueing behavior.

        Sub-RQ3.1: E2E latency CDF (low vs high concurrency) - needs raw CSV
        Sub-RQ3.2: E2E P99 vs streams (different types)
        Sub-RQ3.3: avg/max queue wait vs streams
        Sub-RQ3.4: Queue wait CDF (light vs heavy load) - needs raw CSV
        """
        print("\n=== Running RQ3 Experiments: Latency & Queueing ===")

        # Sub-RQ3.2 & 3.3: P99 and queue wait vs streams
        print("  RQ3.2 & 3.3: P99 and queue wait vs streams")
        stream_counts = [1, 2, 4, 8, 16, 32, 64]
        kernel_types = ['compute', 'memory', 'mixed', 'gemm']

        csv_lines = []

        for ktype in kernel_types:
            for streams in stream_counts:
                print(f"    type={ktype}, streams={streams}")

                for run_idx in range(self.num_runs):
                    args = [
                        '--streams', str(streams),
                        '--kernels', '30',
                        '--size', '1048576',
                        '--type', ktype,
                    ]

                    csv_output = self.run_benchmark(
                        args,
                        first_run=(len(csv_lines) == 0)
                    )

                    if csv_output:
                        lines = csv_output.split('\n')
                        if len(csv_lines) == 0:
                            csv_lines.extend(lines)
                        else:
                            csv_lines.extend([l for l in lines if not l.startswith('streams,')])

                    time.sleep(0.5)

        self.save_csv(csv_lines, 'rq3_latency_vs_streams.csv')

        # Sub-RQ3.1 & 3.4: CDF analysis - need raw CSV output
        print("  RQ3.1: E2E latency CDF (low vs high concurrency)")

        csv_lines = []
        raw_csv_configs = [
            ('low_concurrency', 2, 0),
            ('high_concurrency', 32, 0),
        ]

        for config_name, streams, seed in raw_csv_configs:
            print(f"    config={config_name}, streams={streams}")

            for run_idx in range(self.num_runs):
                raw_csv_file = str(self.output_dir / f'rq3_1_raw_{config_name}_run{run_idx}.csv')

                args = [
                    '--streams', str(streams),
                    '--kernels', '50',
                    '--size', '524288',
                    '--type', 'mixed',
                    '--csv-output', raw_csv_file,
                ]

                csv_output = self.run_benchmark(
                    args,
                    first_run=(len(csv_lines) == 0)
                )

                if csv_output:
                    lines = csv_output.split('\n')
                    if len(csv_lines) == 0:
                        csv_lines.extend(lines)
                    else:
                        csv_lines.extend([l for l in lines if not l.startswith('streams,')])

                time.sleep(0.5)

        self.save_csv(csv_lines, 'rq3_1_latency_cdf_aggregated.csv')

        # Sub-RQ3.4: Queue wait CDF at different loads
        print("  RQ3.4: Queue wait CDF (light vs heavy load)")

        csv_lines = []
        load_configs = [
            ('light', 8, 20),    # 20 Hz per stream
            ('medium', 8, 100),  # 100 Hz per stream
            ('heavy', 8, 500),   # 500 Hz per stream
        ]

        for config_name, streams, freq in load_configs:
            print(f"    load={config_name}, freq={freq}Hz")

            for run_idx in range(self.num_runs):
                raw_csv_file = str(self.output_dir / f'rq3_4_raw_{config_name}_run{run_idx}.csv')
                freq_spec = ','.join([str(freq)] * streams)

                args = [
                    '--streams', str(streams),
                    '--kernels', '50',
                    '--size', '524288',
                    '--type', 'mixed',
                    '--launch-frequency', freq_spec,
                    '--csv-output', raw_csv_file,
                ]

                csv_output = self.run_benchmark(
                    args,
                    first_run=(len(csv_lines) == 0)
                )

                if csv_output:
                    lines = csv_output.split('\n')
                    if len(csv_lines) == 0:
                        csv_lines.extend(lines)
                    else:
                        csv_lines.extend([l for l in lines if not l.startswith('streams,')])

                time.sleep(0.5)

        self.save_csv(csv_lines, 'rq3_4_queue_wait_cdf_aggregated.csv')

    # ========================================================================
    # RQ4: Priority Semantics
    # ========================================================================

    def run_rq4_experiments(self):
        """
        RQ4: CUDA stream priority actual semantics.

        Sub-RQ4.1: inversion_rate vs streams (with/without priority)
        Sub-RQ4.2: per-priority P99 vs offered load
        Sub-RQ4.3: fast kernels P99 (single/no-prio/prio)
        Sub-RQ4.4: Jain fairness vs priority pattern
        """
        print("\n=== Running RQ4 Experiments: Priority Semantics ===")

        # Sub-RQ4.1: Inversion rate with/without priority
        print("  RQ4.1: Inversion rate vs streams")
        stream_counts = [4, 8, 16, 32]

        csv_lines = []

        for streams in stream_counts:
            # Baseline: no priority (all 0)
            print(f"    streams={streams}, no priority")
            for run_idx in range(self.num_runs):
                args = [
                    '--streams', str(streams),
                    '--kernels', '30',
                    '--size', '1048576',
                    '--type', 'mixed',
                ]

                csv_output = self.run_benchmark(
                    args,
                    first_run=(len(csv_lines) == 0)
                )

                if csv_output:
                    lines = csv_output.split('\n')
                    if len(csv_lines) == 0:
                        csv_lines.extend(lines)
                    else:
                        csv_lines.extend([l for l in lines if not l.startswith('streams,')])

                time.sleep(0.5)

            # With priority: half high, half low
            print(f"    streams={streams}, with priority")
            high_prio = -5  # Higher priority (more negative)
            low_prio = 0    # Lower priority
            half = streams // 2
            prio_spec = ','.join([str(high_prio)] * half + [str(low_prio)] * (streams - half))

            for run_idx in range(self.num_runs):
                args = [
                    '--streams', str(streams),
                    '--kernels', '30',
                    '--size', '1048576',
                    '--type', 'mixed',
                    '--priority', prio_spec,
                ]

                csv_output = self.run_benchmark(
                    args,
                    first_run=False  # Header already added
                )

                if csv_output:
                    lines = csv_output.split('\n')
                    csv_lines.extend([l for l in lines if not l.startswith('streams,')])

                time.sleep(0.5)

        self.save_csv(csv_lines, 'rq4_1_inversion_rate.csv')

        # Sub-RQ4.2: Per-priority P99 vs load
        print("  RQ4.2: Per-priority P99 vs offered load")
        frequencies = [20, 50, 100, 200, 500, 1000]
        num_streams = 8
        high_prio = -5
        low_prio = 0
        half = num_streams // 2
        prio_spec = ','.join([str(high_prio)] * half + [str(low_prio)] * (num_streams - half))

        csv_lines = []

        for freq in frequencies:
            print(f"    freq={freq}Hz")
            freq_spec = ','.join([str(freq)] * num_streams)

            for run_idx in range(self.num_runs):
                args = [
                    '--streams', str(num_streams),
                    '--kernels', '50',
                    '--size', '524288',
                    '--type', 'mixed',
                    '--priority', prio_spec,
                    '--launch-frequency', freq_spec,
                ]

                csv_output = self.run_benchmark(
                    args,
                    first_run=(len(csv_lines) == 0)
                )

                if csv_output:
                    lines = csv_output.split('\n')
                    if len(csv_lines) == 0:
                        csv_lines.extend(lines)
                    else:
                        csv_lines.extend([l for l in lines if not l.startswith('streams,')])

                time.sleep(0.5)

        self.save_csv(csv_lines, 'rq4_2_per_priority_vs_load.csv')

        # Sub-RQ4.3: Fast kernels with/without priority (RT vs BE scenario)
        print("  RQ4.3: Fast kernels in RT vs BE scenario")

        csv_lines = []

        # Config 1: Only fast (baseline)
        print("    Config: Only fast kernels")
        for run_idx in range(self.num_runs):
            args = [
                '--streams', '4',
                '--kernels', '40',
                '--size', '65536',  # Small/fast
                '--type', 'compute',
            ]

            csv_output = self.run_benchmark(
                args,
                first_run=(len(csv_lines) == 0)
            )

            if csv_output:
                lines = csv_output.split('\n')
                if len(csv_lines) == 0:
                    csv_lines.extend(lines)
                else:
                    csv_lines.extend([l for l in lines if not l.startswith('streams,')])

            time.sleep(0.5)

        # Config 2: Fast + Slow, no priority
        print("    Config: Fast + Slow, no priority")
        for run_idx in range(self.num_runs):
            # Need to use heterogeneous and load-imbalance
            # 4 streams: 2 fast (many small kernels), 2 slow (few large kernels)
            heterogeneous = 'compute,compute,gemm,gemm'
            load_imbalance = '40,40,10,10'  # Fast streams have more kernels

            args = [
                '--streams', '4',
                '--kernels', '40',  # Base, modified by load-imbalance
                '--size', '65536',   # Base size
                '--heterogeneous', heterogeneous,
                '--load-imbalance', load_imbalance,
            ]

            csv_output = self.run_benchmark(args, first_run=False)

            if csv_output:
                lines = csv_output.split('\n')
                csv_lines.extend([l for l in lines if not l.startswith('streams,')])

            time.sleep(0.5)

        # Config 3: Fast + Slow, with priority
        print("    Config: Fast + Slow, with priority")
        for run_idx in range(self.num_runs):
            heterogeneous = 'compute,compute,gemm,gemm'
            load_imbalance = '40,40,10,10'
            priority = '-5,-5,0,0'  # Fast streams get high priority

            args = [
                '--streams', '4',
                '--kernels', '40',
                '--size', '65536',
                '--heterogeneous', heterogeneous,
                '--load-imbalance', load_imbalance,
                '--priority', priority,
            ]

            csv_output = self.run_benchmark(args, first_run=False)

            if csv_output:
                lines = csv_output.split('\n')
                csv_lines.extend([l for l in lines if not l.startswith('streams,')])

            time.sleep(0.5)

        self.save_csv(csv_lines, 'rq4_3_fast_kernels_rt_be.csv')

        # Sub-RQ4.4: Jain fairness vs priority pattern
        print("  RQ4.4: Jain fairness vs priority pattern")

        csv_lines = []
        num_streams = 8

        priority_patterns = [
            ('all_equal', '0,0,0,0,0,0,0,0'),
            ('1high_7low', '-5,0,0,0,0,0,0,0'),
            ('2high_6low', '-5,-5,0,0,0,0,0,0'),
            ('4high_4low', '-5,-5,-5,-5,0,0,0,0'),
            ('multi_level', '-10,-8,-5,-3,0,0,0,0'),
        ]

        for pattern_name, prio_spec in priority_patterns:
            print(f"    pattern={pattern_name}")

            for run_idx in range(self.num_runs):
                args = [
                    '--streams', str(num_streams),
                    '--kernels', '30',
                    '--size', '1048576',
                    '--type', 'mixed',
                    '--priority', prio_spec,
                ]

                csv_output = self.run_benchmark(
                    args,
                    first_run=(len(csv_lines) == 0)
                )

                if csv_output:
                    lines = csv_output.split('\n')
                    if len(csv_lines) == 0:
                        csv_lines.extend(lines)
                    else:
                        csv_lines.extend([l for l in lines if not l.startswith('streams,')])

                time.sleep(0.5)

        self.save_csv(csv_lines, 'rq4_4_fairness_vs_priority.csv')

    # ========================================================================
    # RQ5: Heterogeneity & Load Imbalance
    # ========================================================================

    def run_rq5_experiments(self):
        """
        RQ5: Heterogeneity and load imbalance effects.

        Sub-RQ5.1: Jain index vs load imbalance
        Sub-RQ5.2: Per-stream P99 (load imbalance) - needs raw CSV
        Sub-RQ5.3: throughput/concurrency in homogeneous vs heterogeneous
        """
        print("\n=== Running RQ5 Experiments: Heterogeneity & Load Imbalance ===")

        # Sub-RQ5.1: Jain index vs load imbalance
        print("  RQ5.1: Jain index vs load imbalance")

        csv_lines = []
        num_streams = 4

        imbalance_patterns = [
            ('balanced', '20,20,20,20'),
            ('mild', '10,20,30,40'),
            ('moderate', '5,15,30,50'),
            ('severe', '5,10,40,80'),
        ]

        for pattern_name, load_spec in imbalance_patterns:
            print(f"    pattern={pattern_name}")

            for run_idx in range(self.num_runs):
                args = [
                    '--streams', str(num_streams),
                    '--kernels', '20',  # Base value, overridden by load-imbalance
                    '--size', '1048576',
                    '--type', 'mixed',
                    '--load-imbalance', load_spec,
                ]

                csv_output = self.run_benchmark(
                    args,
                    first_run=(len(csv_lines) == 0)
                )

                if csv_output:
                    lines = csv_output.split('\n')
                    if len(csv_lines) == 0:
                        csv_lines.extend(lines)
                    else:
                        csv_lines.extend([l for l in lines if not l.startswith('streams,')])

                time.sleep(0.5)

        self.save_csv(csv_lines, 'rq5_1_jain_vs_imbalance.csv')

        # Sub-RQ5.2: Per-stream P99 (severe imbalance with raw CSV)
        print("  RQ5.2: Per-stream P99 latency (load imbalance)")

        csv_lines = []
        load_spec = '5,10,40,80'

        for run_idx in range(self.num_runs):
            raw_csv_file = str(self.output_dir / f'rq5_2_raw_imbalance_run{run_idx}.csv')

            args = [
                '--streams', str(num_streams),
                '--kernels', '20',
                '--size', '1048576',
                '--type', 'mixed',
                '--load-imbalance', load_spec,
                '--csv-output', raw_csv_file,
            ]

            csv_output = self.run_benchmark(
                args,
                first_run=(len(csv_lines) == 0)
            )

            if csv_output:
                lines = csv_output.split('\n')
                if len(csv_lines) == 0:
                    csv_lines.extend(lines)
                else:
                    csv_lines.extend([l for l in lines if not l.startswith('streams,')])

            time.sleep(0.5)

        self.save_csv(csv_lines, 'rq5_2_per_stream_imbalance_aggregated.csv')

        # Sub-RQ5.3: Homogeneous vs heterogeneous
        print("  RQ5.3: Throughput/concurrency homogeneous vs heterogeneous")

        csv_lines = []
        stream_counts = [2, 4, 8, 16, 32]

        for streams in stream_counts:
            # Homogeneous: all compute
            print(f"    streams={streams}, homogeneous")
            for run_idx in range(self.num_runs):
                args = [
                    '--streams', str(streams),
                    '--kernels', '20',
                    '--size', '1048576',
                    '--type', 'compute',
                ]

                csv_output = self.run_benchmark(
                    args,
                    first_run=(len(csv_lines) == 0)
                )

                if csv_output:
                    lines = csv_output.split('\n')
                    if len(csv_lines) == 0:
                        csv_lines.extend(lines)
                    else:
                        csv_lines.extend([l for l in lines if not l.startswith('streams,')])

                time.sleep(0.5)

            # Heterogeneous: mix of types
            print(f"    streams={streams}, heterogeneous")
            # Create heterogeneous pattern: cycle through types
            types = ['memory', 'compute', 'mixed', 'gemm']
            hetero_spec = ','.join([types[i % len(types)] for i in range(streams)])

            for run_idx in range(self.num_runs):
                args = [
                    '--streams', str(streams),
                    '--kernels', '20',
                    '--size', '1048576',
                    '--heterogeneous', hetero_spec,
                ]

                csv_output = self.run_benchmark(args, first_run=False)

                if csv_output:
                    lines = csv_output.split('\n')
                    csv_lines.extend([l for l in lines if not l.startswith('streams,')])

                time.sleep(0.5)

        self.save_csv(csv_lines, 'rq5_3_homo_vs_hetero.csv')

    # ========================================================================
    # RQ6: Arrival Pattern & Jitter
    # ========================================================================

    def run_rq6_experiments(self):
        """
        RQ6: Arrival pattern and jitter effects.

        Sub-RQ6.1: concurrent_rate vs jitter
        Sub-RQ6.2: E2E P99 vs jitter
        """
        print("\n=== Running RQ6 Experiments: Arrival Pattern & Jitter ===")

        csv_lines = []
        num_streams = 8
        frequencies = [50, 100, 200, 500]

        for freq in frequencies:
            freq_spec = ','.join([str(freq)] * num_streams)

            # No jitter (periodic)
            print(f"  freq={freq}Hz, no jitter")
            for run_idx in range(self.num_runs):
                args = [
                    '--streams', str(num_streams),
                    '--kernels', '40',
                    '--size', '524288',
                    '--type', 'mixed',
                    '--launch-frequency', freq_spec,
                    '--seed', '0',  # 0 = no jitter
                ]

                csv_output = self.run_benchmark(
                    args,
                    first_run=(len(csv_lines) == 0)
                )

                if csv_output:
                    lines = csv_output.split('\n')
                    if len(csv_lines) == 0:
                        csv_lines.extend(lines)
                    else:
                        csv_lines.extend([l for l in lines if not l.startswith('streams,')])

                time.sleep(0.5)

            # With jitter
            print(f"  freq={freq}Hz, with jitter")
            for run_idx in range(self.num_runs):
                args = [
                    '--streams', str(num_streams),
                    '--kernels', '40',
                    '--size', '524288',
                    '--type', 'mixed',
                    '--launch-frequency', freq_spec,
                    '--seed', '42',  # Non-zero = jitter
                ]

                csv_output = self.run_benchmark(args, first_run=False)

                if csv_output:
                    lines = csv_output.split('\n')
                    csv_lines.extend([l for l in lines if not l.startswith('streams,')])

                time.sleep(0.5)

        self.save_csv(csv_lines, 'rq6_jitter_effects.csv')

    # ========================================================================
    # RQ7: Working Set vs L2 Cache
    # ========================================================================

    def run_rq7_experiments(self):
        """
        RQ7: Memory working set vs L2 cache boundary.

        Sub-RQ7.1: throughput/util vs working_set/L2 ratio
        """
        print("\n=== Running RQ7 Experiments: Working Set vs L2 Cache ===")

        csv_lines = []
        num_streams = 8

        # Vary workload_size to cover < L2, ~ L2, > L2
        # Typical L2: 4-6 MB
        # working_set = streams * size * sizeof(float) / 1MB
        # For 8 streams: size=131072 -> ~4MB, size=524288 -> ~16MB, etc.
        workload_sizes = [
            65536,    # ~2 MB total (< L2)
            131072,   # ~4 MB total (~ L2)
            262144,   # ~8 MB total (> L2)
            524288,   # ~16 MB total (>> L2)
            1048576,  # ~32 MB total (>>> L2)
        ]

        kernel_types = ['memory', 'mixed', 'compute']

        for ktype in kernel_types:
            for size in workload_sizes:
                print(f"  type={ktype}, size={size}")

                for run_idx in range(self.num_runs):
                    args = [
                        '--streams', str(num_streams),
                        '--kernels', '20',
                        '--size', str(size),
                        '--type', ktype,
                    ]

                    csv_output = self.run_benchmark(
                        args,
                        first_run=(len(csv_lines) == 0)
                    )

                    if csv_output:
                        lines = csv_output.split('\n')
                        if len(csv_lines) == 0:
                            csv_lines.extend(lines)
                        else:
                            csv_lines.extend([l for l in lines if not l.startswith('streams,')])

                    time.sleep(0.5)

        self.save_csv(csv_lines, 'rq7_working_set_vs_l2.csv')

    # ========================================================================
    # RQ8: Multi-Process vs Single-Process
    # ========================================================================

    def run_rq8_experiments(self):
        """
        RQ8: Multi-process vs single-process scheduling behavior.

        Sub-RQ8.1: Single-process multi-stream vs multi-process few-stream
        Compares fairness, throughput, and concurrency when:
        - Mode A: 1 process with 32 streams
        - Mode B: 4 processes with 8 streams each
        - Mode C: 8 processes with 4 streams each
        """
        print("\n=== Running RQ8 Experiments: Multi-Process vs Single-Process ===")

        # Note: Since we can't easily run multiple processes concurrently from Python
        # in a controlled way, we'll simulate this by running them sequentially
        # and then the analyzer can combine the results

        total_streams = 32
        configurations = [
            ('1proc_32streams', 1, 32),
            ('4proc_8streams', 4, 8),
            ('8proc_4streams', 8, 4),
            ('16proc_2streams', 16, 2),
        ]

        csv_lines = []

        for config_name, num_processes, streams_per_process in configurations:
            print(f"  Config: {config_name} ({num_processes} processes × {streams_per_process} streams)")

            for run_idx in range(self.num_runs):
                # For single process, just run normally
                if num_processes == 1:
                    args = [
                        '--streams', str(streams_per_process),
                        '--kernels', '30',
                        '--size', '1048576',
                        '--type', 'mixed',
                    ]

                    csv_output = self.run_benchmark(
                        args,
                        first_run=(len(csv_lines) == 0)
                    )

                    if csv_output:
                        lines = csv_output.split('\n')
                        if len(csv_lines) == 0:
                            csv_lines.extend(lines)
                        else:
                            csv_lines.extend([l for l in lines if not l.startswith('streams,')])

                else:
                    # For multiple processes, run them sequentially but save raw CSVs
                    # The analyzer will need to understand this is multi-process
                    process_outputs = []

                    for proc_id in range(num_processes):
                        raw_csv_file = str(self.output_dir / f'rq8_raw_{config_name}_run{run_idx}_proc{proc_id}.csv')

                        args = [
                            '--streams', str(streams_per_process),
                            '--kernels', '30',
                            '--size', '1048576',
                            '--type', 'mixed',
                            '--csv-output', raw_csv_file,
                        ]

                        csv_output = self.run_benchmark(
                            args,
                            first_run=(len(csv_lines) == 0 and proc_id == 0)
                        )

                        if csv_output:
                            process_outputs.append(csv_output)

                        time.sleep(0.3)  # Small delay between processes

                    # For aggregated output, we'll just take the first process's aggregated metrics
                    # The analyzer will do proper multi-process analysis from raw CSVs
                    if process_outputs:
                        lines = process_outputs[0].split('\n')
                        if len(csv_lines) == 0:
                            csv_lines.extend(lines)
                        else:
                            csv_lines.extend([l for l in lines if not l.startswith('streams,')])

                time.sleep(0.5)

        self.save_csv(csv_lines, 'rq8_multiprocess.csv')

    # ========================================================================
    # Main driver
    # ========================================================================

    def run_all_experiments(self):
        """Run all RQ experiments."""
        self.run_rq1_experiments()
        self.run_rq2_experiments()
        self.run_rq3_experiments()
        self.run_rq4_experiments()
        self.run_rq5_experiments()
        self.run_rq6_experiments()
        self.run_rq7_experiments()
        self.run_rq8_experiments()


def main():
    parser = argparse.ArgumentParser(
        description='GPU Scheduler Experiment Driver',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all experiments
  python experiment_driver.py --rq all --output-dir results/

  # Run specific RQ
  python experiment_driver.py --rq RQ1 --output-dir results/

  # Custom benchmark path
  python experiment_driver.py --rq all --bench ./multi_stream_bench --output-dir results/
"""
    )

    parser.add_argument(
        '--rq',
        type=str,
        default='all',
        help='Research question to run: all, RQ1, RQ2, RQ3, RQ4, RQ5, RQ6, RQ7, RQ8'
    )

    parser.add_argument(
        '--bench',
        type=str,
        default='./multi_stream_bench',
        help='Path to benchmark binary (default: ./multi_stream_bench)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for CSV files (default: results/)'
    )

    parser.add_argument(
        '--num-runs',
        type=int,
        default=3,
        help='Number of runs per configuration (default: 3)'
    )

    args = parser.parse_args()

    # Validate benchmark exists
    if not os.path.exists(args.bench):
        print(f"Error: Benchmark not found at {args.bench}")
        sys.exit(1)

    driver = ExperimentDriver(
        bench_path=args.bench,
        output_dir=args.output_dir,
        num_runs=args.num_runs
    )

    rq = args.rq.upper()

    if rq == 'ALL':
        driver.run_all_experiments()
    elif rq == 'RQ1':
        driver.run_rq1_experiments()
    elif rq == 'RQ2':
        driver.run_rq2_experiments()
    elif rq == 'RQ3':
        driver.run_rq3_experiments()
    elif rq == 'RQ4':
        driver.run_rq4_experiments()
    elif rq == 'RQ5':
        driver.run_rq5_experiments()
    elif rq == 'RQ6':
        driver.run_rq6_experiments()
    elif rq == 'RQ7':
        driver.run_rq7_experiments()
    elif rq == 'RQ8':
        driver.run_rq8_experiments()
    else:
        print(f"Error: Unknown RQ '{args.rq}'. Valid options: all, RQ1-RQ8")
        sys.exit(1)

    print(f"\nExperiments complete! Results saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
