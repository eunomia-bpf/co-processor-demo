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
            first_run: Not used anymore (kept for API compatibility)

        Returns:
            Clean CSV output string (header + data row) or None if failed
        """
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

            # Extract clean CSV from stderr
            # The benchmark outputs CSV in format:
            # streams,kernels_per_stream,...
            # 8,20,...
            lines = result.stderr.splitlines()
            header = None
            data_rows = []

            for line in lines:
                line = line.strip()
                # Look for the CSV header line (starts with "streams,")
                if line.startswith('streams,'):
                    header = line
                # Look for CSV data lines (start with a digit)
                elif line and line[0].isdigit():
                    data_rows.append(line)

            if header is None:
                print("Warning: No CSV header found in benchmark output")
                return None

            if not data_rows:
                print("Warning: No CSV data rows found in benchmark output")
                return None

            # Return header + all data rows
            return header + '\n' + '\n'.join(data_rows)

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
        # RTX 5090 L2: 96 MB
        # working_set = streams * size * sizeof(float) / 1MB
        # For 8 streams with sizeof(float)=4: size * 8 * 4 / 1048576 = size * 32 / 1048576 MB
        workload_sizes = [
            1048576,    # 32 MB total (< L2, ~0.33x)
            2097152,    # 64 MB total (< L2, ~0.67x)
            3145728,    # 96 MB total (≈ L2, ~1.0x)
            4194304,    # 128 MB total (> L2, ~1.33x)
            6291456,    # 192 MB total (>> L2, ~2.0x)
            8388608,    # 256 MB total (>>> L2, ~2.67x)
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
        - Mode D: 16 processes with 2 streams each

        This uses concurrent process execution to properly test multi-process GPU scheduling.
        """
        print("\n=== Running RQ8 Experiments: Multi-Process vs Single-Process ===")

        import multiprocessing
        import glob

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
                    # For multiple processes, run them CONCURRENTLY using subprocess.Popen
                    print(f"    Launching {num_processes} concurrent processes...")

                    # Start all processes concurrently
                    processes = []
                    raw_files = []

                    for proc_id in range(num_processes):
                        raw_csv_file = str(self.output_dir / f'rq8_raw_{config_name}_run{run_idx}_proc{proc_id}.csv')
                        raw_files.append(raw_csv_file)

                        cmd = [
                            self.bench_path,
                            '--streams', str(streams_per_process),
                            '--kernels', '30',
                            '--size', '1048576',
                            '--type', 'mixed',
                            '--csv-output', raw_csv_file,
                            '--no-header',  # Skip header in raw files
                        ]

                        proc = subprocess.Popen(
                            cmd,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL
                        )
                        processes.append(proc)

                    # Wait for all processes to complete
                    print(f"    Waiting for {num_processes} processes to complete...")
                    results = []
                    for proc_id, (proc, raw_file) in enumerate(zip(processes, raw_files)):
                        returncode = proc.wait(timeout=120)
                        success = (returncode == 0)
                        results.append((proc_id, success, raw_file))

                    # Check results
                    successful = sum(1 for _, success, _ in results if success)
                    print(f"    Completed: {successful}/{num_processes} processes succeeded")

                    # Aggregate metrics from all raw CSVs for this run
                    raw_files = [f for _, success, f in results if success and f]
                    if raw_files:
                        # Read and combine all per-kernel data
                        all_timings = []
                        for proc_id, raw_file in enumerate(raw_files):
                            if os.path.exists(raw_file):
                                df = pd.read_csv(raw_file)
                                # Assign global stream IDs: each process's streams get unique IDs
                                df['global_stream_id'] = df['stream_id'] + proc_id * streams_per_process
                                all_timings.append(df)

                        if all_timings:
                            # Combine all timings
                            combined_df = pd.concat(all_timings, ignore_index=True)

                            # Compute aggregate metrics across all processes
                            total_kernels = len(combined_df)
                            # Use actual number of successful processes, not nominal count
                            effective_procs = len(raw_files)
                            total_streams = effective_procs * streams_per_process

                            # Calculate throughput: total kernels / total wall time
                            global_start = combined_df['start_time_ms'].min()
                            global_end = combined_df['end_time_ms'].max()
                            wall_time_sec = (global_end - global_start) / 1000.0
                            throughput = total_kernels / wall_time_sec if wall_time_sec > 0 else 0

                            # Calculate P99 latency
                            e2e_p99 = combined_df['e2e_latency_ms'].quantile(0.99)

                            # Calculate Jain's fairness index per stream (using global stream IDs)
                            stream_throughputs = []
                            for stream_id in combined_df['global_stream_id'].unique():
                                stream_df = combined_df[combined_df['global_stream_id'] == stream_id]
                                stream_start = stream_df['start_time_ms'].min()
                                stream_end = stream_df['end_time_ms'].max()
                                stream_time = (stream_end - stream_start) / 1000.0
                                stream_tput = len(stream_df) / stream_time if stream_time > 0 else 0
                                stream_throughputs.append(stream_tput)

                            # Jain's index = (sum x)^2 / (n * sum x^2)
                            if stream_throughputs:
                                sum_tput = sum(stream_throughputs)
                                sum_tput_sq = sum(x*x for x in stream_throughputs)
                                n = len(stream_throughputs)
                                jains_index = (sum_tput * sum_tput) / (n * sum_tput_sq) if sum_tput_sq > 0 else 1.0
                            else:
                                jains_index = 1.0

                            # Compute GPU utilization (simplified: fraction of time with active kernels)
                            events = []
                            for _, row in combined_df.iterrows():
                                events.append((row['start_time_ms'], 1))
                                events.append((row['end_time_ms'], -1))
                            events.sort()

                            time_with_kernels = 0
                            current_count = 0
                            last_time = events[0][0]
                            for event_time, delta in events:
                                if current_count > 0:
                                    time_with_kernels += (event_time - last_time)
                                current_count += delta
                                last_time = event_time

                            util = (time_with_kernels / (global_end - global_start) * 100) if (global_end - global_start) > 0 else 0

                            # Create a full CSV line matching the header format
                            # Most fields will be empty for multiprocess aggregation
                            wall_time = (global_end - global_start)

                            csv_line_parts = [
                                str(total_streams),  # streams
                                str(total_kernels // total_streams),  # kernels_per_stream
                                f'{num_processes}proc_x_{streams_per_process}streams',  # kernels_per_stream_detail
                                str(total_kernels),  # total_kernels
                                'mixed',  # type
                                'multiprocess',  # type_detail
                                f'{wall_time:.2f}',  # wall_time_ms
                                f'{wall_time:.2f}',  # e2e_wall_time_ms
                                f'{throughput:.2f}',  # throughput
                                '',  # svc_mean
                                '',  # svc_p50
                                '',  # svc_p95
                                '',  # svc_p99
                                '',  # e2e_mean
                                '',  # e2e_p50
                                '',  # e2e_p95
                                f'{e2e_p99:.2f}',  # e2e_p99
                                '',  # avg_queue_wait
                                '',  # max_queue_wait
                                '',  # concurrent_rate
                                f'{util:.1f}',  # util
                                f'{jains_index:.4f}',  # jains_index
                                '',  # max_concurrent
                                '',  # avg_concurrent
                                '',  # inversions
                                '',  # inversion_rate
                                '',  # working_set_mb
                                '',  # fits_in_l2
                                '',  # svc_stddev
                                '',  # grid_size
                                '',  # block_size
                                '',  # per_priority_avg
                                '',  # per_priority_p50
                                '',  # per_priority_p99
                                '0',  # launch_freq
                                '0',  # seed
                            ]

                            csv_lines.append(','.join(csv_line_parts))

                time.sleep(1.0)  # Pause between runs

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
