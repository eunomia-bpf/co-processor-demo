"""
RQ5: Preemption Latency Analysis (formerly RQ4.5-4.8)

Experiments:
- RQ5.1 (formerly RQ4.5): Preemption latency vs kernel duration
- RQ5.2 (formerly RQ4.6): Preemption latency vs offered load
- RQ5.3 (formerly RQ4.7): CDF of preemption latency for small vs large background kernels
- RQ5.4 (formerly RQ4.8): CDF of preemption latency for low vs high offered load
"""

import time
import subprocess
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from .base import RQBase


class RQ5(RQBase):
    """RQ5: Preemption Latency Analysis - 4 plots."""

    def run_experiments(self):
        """Run RQ5.1-RQ5.4 experiments (preemption latency)."""
        print("\n=== Running RQ5 Experiments: Preemption Latency Analysis ===")

        self._run_rq5_1()
        self._run_rq5_2()

        print("\n=== RQ5 Experiments Complete ===")
        print("Note: RQ5.3 and RQ5.4 use raw CSV data from RQ5.1 and RQ5.2 for CDF plots")

    def _run_rq5_1(self):
        """
        RQ5.1: Preemption latency vs kernel duration.

        Measures how preemption latency (start - enqueue for high-priority tasks)
        varies with background kernel duration. Similar to XSched Fig.11(b).
        """
        print("  RQ5.1: Preemption latency vs kernel duration")

        # We'll use raw CSV output to calculate preemption latency
        # High priority: 4 streams with small fast kernels
        # Low priority: 4 streams with varying duration kernels

        num_streams = 8
        high_prio_streams = 4
        low_prio_streams = 4

        # Sweep background kernel durations
        bg_kernel_sizes = [
            (65536, '~0.1ms'),
            (262144, '~0.5ms'),
            (1048576, '~2ms'),
            (4194304, '~8ms'),
        ]

        for bg_size, bg_label in bg_kernel_sizes:
            print(f"    Background kernel size={bg_size} ({bg_label})")

            for run_idx in range(self.num_runs):
                # Priority spec: first 4 high (-5), last 4 low (0)
                prio_spec = ','.join(['-5'] * high_prio_streams + ['0'] * low_prio_streams)

                # Per-stream sizes: high-prio gets small fixed size, low-prio gets variable bg_size
                size_spec = ','.join(['65536'] * high_prio_streams + [str(bg_size)] * low_prio_streams)

                # Load imbalance: high-prio streams launch more frequently
                # We want high-prio to keep issuing work while low-prio runs long kernels
                load_spec = ','.join(['100'] * high_prio_streams + ['20'] * low_prio_streams)

                raw_csv_file = self.output_dir / f'rq5_1_raw_bg{bg_size}_run{run_idx}.csv'

                args = [
                    '--streams', str(num_streams),
                    '--kernels', '50',  # Base, modified by load-imbalance
                    '--type', 'mixed',  # All streams use mixed kernels
                    '--priority', prio_spec,
                    '--per-stream-sizes', size_spec,  # Per-stream workload sizes
                    '--load-imbalance', load_spec,
                    '--csv-output', str(raw_csv_file),
                    '--no-header',
                ]

                # Run benchmark (output goes to file, not stdout)
                result = subprocess.run(
                    [self.bench_path] + args,
                    capture_output=True,
                    text=True,
                    timeout=120
                )

                if result.returncode != 0:
                    print(f"      Warning: Run {run_idx} failed: {result.stderr}")

                time.sleep(0.5)

        # Now aggregate all raw CSV files into analysis-ready format
        print("    Aggregating raw CSV files...")
        self._aggregate_rq5_1_data(bg_kernel_sizes)

    def _aggregate_rq5_1_data(self, bg_kernel_sizes):
        """Aggregate RQ5.1 raw CSV files and compute preemption latency metrics."""
        aggregated_data = []

        for bg_size, bg_label in bg_kernel_sizes:
            pattern = str(self.output_dir / f'rq5_1_raw_bg{bg_size}_run*.csv')
            raw_files = glob.glob(pattern)

            if not raw_files:
                continue

            for raw_file in raw_files:
                if not os.path.exists(raw_file):
                    continue

                df = pd.read_csv(raw_file)

                # Filter high-priority tasks (priority < 0)
                high_prio = df[df['priority'] < 0].copy()

                if len(high_prio) == 0:
                    continue

                # Calculate preemption latency: start_time - enqueue_time
                high_prio['preempt_latency_ms'] = high_prio['start_time_ms'] - high_prio['enqueue_time_ms']

                # Aggregate statistics
                row = {
                    'bg_kernel_size': bg_size,
                    'bg_label': bg_label,
                    'preempt_latency_mean': high_prio['preempt_latency_ms'].mean(),
                    'preempt_latency_p50': high_prio['preempt_latency_ms'].quantile(0.50),
                    'preempt_latency_p95': high_prio['preempt_latency_ms'].quantile(0.95),
                    'preempt_latency_p99': high_prio['preempt_latency_ms'].quantile(0.99),
                    'preempt_latency_max': high_prio['preempt_latency_ms'].max(),
                    'high_prio_duration_mean': high_prio['duration_ms'].mean(),
                    'num_high_prio_kernels': len(high_prio),
                }
                aggregated_data.append(row)

        if aggregated_data:
            agg_df = pd.DataFrame(aggregated_data)

            # Group by bg_kernel_size and compute mean/std across runs
            final_df = agg_df.groupby(['bg_kernel_size', 'bg_label']).agg({
                'preempt_latency_mean': ['mean', 'std'],
                'preempt_latency_p50': ['mean', 'std'],
                'preempt_latency_p95': ['mean', 'std'],
                'preempt_latency_p99': ['mean', 'std'],
                'preempt_latency_max': ['mean', 'std'],
                'high_prio_duration_mean': 'mean',
                'num_high_prio_kernels': 'sum',
            }).reset_index()

            # Flatten column names
            final_df.columns = ['_'.join(col).strip('_') if col[1] else col[0]
                               for col in final_df.columns.values]

            csv_path = self.results_dir / 'rq5_1_preempt_vs_duration.csv'
            final_df.to_csv(csv_path, index=False)
            print(f"Saved: {csv_path}")

    def _run_rq5_2(self):
        """
        RQ5.2: Preemption latency vs offered load.

        Measures how preemption latency varies with system load (launch frequency).
        Similar to XSched Fig.11(c).
        """
        print("  RQ5.2: Preemption latency vs offered load")

        num_streams = 8
        high_prio_streams = 4
        low_prio_streams = 4

        # Fixed kernel duration, sweep launch frequencies
        kernel_size = 262144  # ~0.5ms

        launch_frequencies = [20, 50, 100, 200, 500, 1000]  # Hz

        for freq in launch_frequencies:
            print(f"    Launch frequency={freq}Hz")

            for run_idx in range(self.num_runs):
                # Priority spec: first 4 high (-5), last 4 low (0)
                prio_spec = ','.join(['-5'] * high_prio_streams + ['0'] * low_prio_streams)

                # All streams same frequency
                freq_spec = ','.join([str(freq)] * num_streams)

                raw_csv_file = self.output_dir / f'rq5_2_raw_freq{freq}_run{run_idx}.csv'

                args = [
                    '--streams', str(num_streams),
                    '--kernels', '50',
                    '--size', str(kernel_size),
                    '--type', 'mixed',
                    '--priority', prio_spec,
                    '--launch-frequency', freq_spec,
                    '--csv-output', str(raw_csv_file),
                    '--no-header',
                ]

                result = subprocess.run(
                    [self.bench_path] + args,
                    capture_output=True,
                    text=True,
                    timeout=120
                )

                if result.returncode != 0:
                    print(f"      Warning: Run {run_idx} failed: {result.stderr}")

                time.sleep(0.5)

        # Aggregate
        print("    Aggregating raw CSV files...")
        self._aggregate_rq5_2_data(launch_frequencies)

    def _aggregate_rq5_2_data(self, launch_frequencies):
        """Aggregate RQ5.2 raw CSV files and compute preemption latency vs load."""
        aggregated_data = []

        for freq in launch_frequencies:
            pattern = str(self.output_dir / f'rq5_2_raw_freq{freq}_run*.csv')
            raw_files = glob.glob(pattern)

            if not raw_files:
                continue

            for raw_file in raw_files:
                if not os.path.exists(raw_file):
                    continue

                df = pd.read_csv(raw_file)

                # Filter high-priority tasks
                high_prio = df[df['priority'] < 0].copy()

                if len(high_prio) == 0:
                    continue

                # Calculate preemption latency
                high_prio['preempt_latency_ms'] = high_prio['start_time_ms'] - high_prio['enqueue_time_ms']

                # Aggregate statistics
                row = {
                    'launch_freq': freq,
                    'offered_load': freq * len(df['stream_id'].unique()),  # approx kernels/sec
                    'preempt_latency_mean': high_prio['preempt_latency_ms'].mean(),
                    'preempt_latency_p50': high_prio['preempt_latency_ms'].quantile(0.50),
                    'preempt_latency_p95': high_prio['preempt_latency_ms'].quantile(0.95),
                    'preempt_latency_p99': high_prio['preempt_latency_ms'].quantile(0.99),
                    'preempt_latency_max': high_prio['preempt_latency_ms'].max(),
                    'num_high_prio_kernels': len(high_prio),
                }
                aggregated_data.append(row)

        if aggregated_data:
            agg_df = pd.DataFrame(aggregated_data)

            # Group by launch_freq and compute mean/std across runs
            final_df = agg_df.groupby(['launch_freq', 'offered_load']).agg({
                'preempt_latency_mean': ['mean', 'std'],
                'preempt_latency_p50': ['mean', 'std'],
                'preempt_latency_p95': ['mean', 'std'],
                'preempt_latency_p99': ['mean', 'std'],
                'preempt_latency_max': ['mean', 'std'],
                'num_high_prio_kernels': 'sum',
            }).reset_index()

            # Flatten column names
            final_df.columns = ['_'.join(col).strip('_') if col[1] else col[0]
                               for col in final_df.columns.values]

            csv_path = self.results_dir / 'rq5_2_preempt_vs_load.csv'
            final_df.to_csv(csv_path, index=False)
            print(f"Saved: {csv_path}")

    def analyze(self):
        """Generate RQ5 analysis figure (2x2 layout)."""
        print("\n=== Analyzing RQ5: Preemption Latency Analysis ===")

        # Create a 2x2 subplot figure (4 subplots)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        self._plot_rq5_1(ax1)
        self._plot_rq5_2(ax2)
        self._plot_rq5_3(ax3)
        self._plot_rq5_4(ax4)

        fig.suptitle('RQ5: Preemption Latency Analysis', fontsize=16, fontweight='bold')
        self.save_figure('rq5_preemption')
        print("=== RQ5 Analysis Complete ===")

    def _plot_rq5_1(self, ax):
        """Plot RQ5.1: Preemption latency vs kernel duration."""
        print("  RQ5.1: Preemption latency vs kernel duration")
        df = self.load_csv('rq5_1_preempt_vs_duration.csv')

        # Fallback to old rq4_5 naming for backward compatibility
        if df is None:
            df = self.load_csv('rq4_5_preempt_vs_duration.csv')

        if df is None:
            ax.text(0.5, 0.5, 'No RQ5.1 data\n(run experiments first)',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, color='gray', style='italic')
            ax.set_title('(a) Preemption Latency vs Kernel Duration')
            return

        # Sort by kernel size for proper line plotting
        df_sorted = df.sort_values('bg_kernel_size')

        # Plot P99 preemption latency vs background kernel size
        ax.plot(df_sorted['bg_kernel_size'], df_sorted['preempt_latency_p99_mean'],
                marker='o', linewidth=2, markersize=8, color='purple', label='P99')
        ax.plot(df_sorted['bg_kernel_size'], df_sorted['preempt_latency_mean_mean'],
                marker='s', linewidth=2, markersize=6, color='blue', alpha=0.6, label='Mean')
        ax.plot(df_sorted['bg_kernel_size'], df_sorted['preempt_latency_max_mean'],
                marker='^', linewidth=2, markersize=6, color='red', alpha=0.4, label='Max')

        # Add annotations with kernel duration labels
        for idx, row in df_sorted.iterrows():
            ax.annotate(f"{row['bg_label']}",
                       (row['bg_kernel_size'], row['preempt_latency_p99_mean']),
                       xytext=(0, 10), textcoords='offset points',
                       fontsize=8, ha='center', style='italic')

        ax.set_xlabel('Background Kernel Size (elements)')
        ax.set_ylabel('Preemption Latency (ms)')
        ax.set_title('(a) Preemption Latency vs Kernel Duration')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')  # Log scale for kernel sizes

    def _plot_rq5_2(self, ax):
        """Plot RQ5.2: Preemption latency vs offered load."""
        print("  RQ5.2: Preemption latency vs offered load")
        df = self.load_csv('rq5_2_preempt_vs_load.csv')

        # Fallback to old rq4_6 naming for backward compatibility
        if df is None:
            df = self.load_csv('rq4_6_preempt_vs_load.csv')

        if df is None:
            ax.text(0.5, 0.5, 'No RQ5.2 data\n(run experiments first)',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, color='gray', style='italic')
            ax.set_title('(b) Preemption Latency vs Offered Load')
            return

        # Sort by launch frequency for proper line plotting
        df_sorted = df.sort_values('launch_freq')

        # Plot P99 preemption latency vs launch frequency
        ax.plot(df_sorted['launch_freq'], df_sorted['preempt_latency_p99_mean'],
                marker='o', linewidth=2, markersize=8, color='darkred', label='P99')
        ax.plot(df_sorted['launch_freq'], df_sorted['preempt_latency_mean_mean'],
                marker='s', linewidth=2, markersize=6, color='orange', alpha=0.6, label='Mean')
        ax.plot(df_sorted['launch_freq'], df_sorted['preempt_latency_p95_mean'],
                marker='d', linewidth=2, markersize=6, color='green', alpha=0.5, label='P95')

        ax.set_xlabel('Launch Frequency (Hz)')
        ax.set_ylabel('Preemption Latency (ms)')
        ax.set_title('(b) Preemption Latency vs Offered Load')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')  # Log scale for frequency range

    def _plot_rq5_3(self, ax):
        """Plot RQ5.3: CDF of preemption latency for small vs large background kernels."""
        print("  RQ5.3: Preemption latency CDF vs kernel duration")

        # Load raw CSV files for small (65536) and large (4194304) background kernels
        # Check both results_dir and output_dir for backward compatibility
        # Also check for old rq4_5 naming
        small_kernel_files = glob.glob(str(self.results_dir / 'rq5_1_raw_bg65536_run*.csv'))
        if not small_kernel_files:
            small_kernel_files = glob.glob(str(self.output_dir / 'rq5_1_raw_bg65536_run*.csv'))
        if not small_kernel_files:
            small_kernel_files = glob.glob(str(self.results_dir / 'rq4_5_raw_bg65536_run*.csv'))

        large_kernel_files = glob.glob(str(self.results_dir / 'rq5_1_raw_bg4194304_run*.csv'))
        if not large_kernel_files:
            large_kernel_files = glob.glob(str(self.output_dir / 'rq5_1_raw_bg4194304_run*.csv'))
        if not large_kernel_files:
            large_kernel_files = glob.glob(str(self.results_dir / 'rq4_5_raw_bg4194304_run*.csv'))

        if not small_kernel_files or not large_kernel_files:
            ax.text(0.5, 0.5, 'No RQ5.3 CDF data\n(run experiments first)',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, color='gray', style='italic')
            ax.set_title('(c) Preemption Latency CDF vs Kernel Duration')
            return

        # Load and combine all runs for small kernels
        small_dfs = []
        for f in small_kernel_files:
            try:
                df = pd.read_csv(f)
                small_dfs.append(df)
            except Exception as e:
                print(f"    Warning: Error loading {f}: {e}")

        # Load and combine all runs for large kernels
        large_dfs = []
        for f in large_kernel_files:
            try:
                df = pd.read_csv(f)
                large_dfs.append(df)
            except Exception as e:
                print(f"    Warning: Error loading {f}: {e}")

        if not small_dfs or not large_dfs:
            ax.text(0.5, 0.5, 'Error loading CDF data',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, color='red', style='italic')
            ax.set_title('(c) Preemption Latency CDF vs Kernel Duration')
            return

        small_all = pd.concat(small_dfs, ignore_index=True)
        large_all = pd.concat(large_dfs, ignore_index=True)

        # Filter for high-priority tasks only (priority < 0)
        small_high_prio = small_all[small_all['priority'] < 0].copy()
        large_high_prio = large_all[large_all['priority'] < 0].copy()

        # Calculate preemption latency
        small_high_prio['preempt_latency'] = small_high_prio['start_time_ms'] - small_high_prio['enqueue_time_ms']
        large_high_prio['preempt_latency'] = large_high_prio['start_time_ms'] - large_high_prio['enqueue_time_ms']

        # Plot CDFs
        small_sorted = np.sort(small_high_prio['preempt_latency'].values)
        large_sorted = np.sort(large_high_prio['preempt_latency'].values)

        small_cdf = np.arange(1, len(small_sorted) + 1) / len(small_sorted)
        large_cdf = np.arange(1, len(large_sorted) + 1) / len(large_sorted)

        ax.plot(small_sorted, small_cdf, linewidth=2, color='blue',
               label=f'Small BG Kernel (65k, n={len(small_high_prio)})')
        ax.plot(large_sorted, large_cdf, linewidth=2, color='red',
               label=f'Large BG Kernel (4M, n={len(large_high_prio)})')

        ax.set_xlabel('Preemption Latency (ms)')
        ax.set_ylabel('CDF')
        ax.set_title('(c) Preemption Latency CDF vs Kernel Duration')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim([0, 1])

    def _plot_rq5_4(self, ax):
        """Plot RQ5.4: CDF of preemption latency for low vs high offered load."""
        print("  RQ5.4: Preemption latency CDF vs offered load")

        # Load raw CSV files for low (20Hz) and high (1000Hz) launch frequencies
        # Check both results_dir and output_dir for backward compatibility
        # Also check for old rq4_6 naming
        low_freq_files = glob.glob(str(self.results_dir / 'rq5_2_raw_freq20_run*.csv'))
        if not low_freq_files:
            low_freq_files = glob.glob(str(self.output_dir / 'rq5_2_raw_freq20_run*.csv'))
        if not low_freq_files:
            low_freq_files = glob.glob(str(self.results_dir / 'rq4_6_raw_freq20_run*.csv'))

        high_freq_files = glob.glob(str(self.results_dir / 'rq5_2_raw_freq1000_run*.csv'))
        if not high_freq_files:
            high_freq_files = glob.glob(str(self.output_dir / 'rq5_2_raw_freq1000_run*.csv'))
        if not high_freq_files:
            high_freq_files = glob.glob(str(self.results_dir / 'rq4_6_raw_freq1000_run*.csv'))

        if not low_freq_files or not high_freq_files:
            ax.text(0.5, 0.5, 'No RQ5.4 CDF data\n(run experiments first)',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, color='gray', style='italic')
            ax.set_title('(d) Preemption Latency CDF vs Offered Load')
            return

        # Load and combine all runs for low frequency
        low_dfs = []
        for f in low_freq_files:
            try:
                df = pd.read_csv(f)
                low_dfs.append(df)
            except Exception as e:
                print(f"    Warning: Error loading {f}: {e}")

        # Load and combine all runs for high frequency
        high_dfs = []
        for f in high_freq_files:
            try:
                df = pd.read_csv(f)
                high_dfs.append(df)
            except Exception as e:
                print(f"    Warning: Error loading {f}: {e}")

        if not low_dfs or not high_dfs:
            ax.text(0.5, 0.5, 'Error loading CDF data',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, color='red', style='italic')
            ax.set_title('(d) Preemption Latency CDF vs Offered Load')
            return

        low_all = pd.concat(low_dfs, ignore_index=True)
        high_all = pd.concat(high_dfs, ignore_index=True)

        # Filter for high-priority tasks only (priority < 0)
        low_high_prio = low_all[low_all['priority'] < 0].copy()
        high_high_prio = high_all[high_all['priority'] < 0].copy()

        # Calculate preemption latency
        low_high_prio['preempt_latency'] = low_high_prio['start_time_ms'] - low_high_prio['enqueue_time_ms']
        high_high_prio['preempt_latency'] = high_high_prio['start_time_ms'] - high_high_prio['enqueue_time_ms']

        # Plot CDFs
        low_sorted = np.sort(low_high_prio['preempt_latency'].values)
        high_sorted = np.sort(high_high_prio['preempt_latency'].values)

        low_cdf = np.arange(1, len(low_sorted) + 1) / len(low_sorted)
        high_cdf = np.arange(1, len(high_sorted) + 1) / len(high_sorted)

        ax.plot(low_sorted, low_cdf, linewidth=2, color='green',
               label=f'Low Load (20Hz, n={len(low_high_prio)})')
        ax.plot(high_sorted, high_cdf, linewidth=2, color='purple',
               label=f'High Load (1000Hz, n={len(high_high_prio)})')

        ax.set_xlabel('Preemption Latency (ms)')
        ax.set_ylabel('CDF')
        ax.set_title('(d) Preemption Latency CDF vs Offered Load')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim([0, 1])
