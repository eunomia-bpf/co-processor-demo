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

        DESIGN: To measure true preemption latency:
        - FEW high-priority streams (2) with sparse arrivals
        - MANY low-priority streams (6) saturating GPU with long kernels
        - This ensures high-prio kernels arrive when GPU is busy with low-prio work
        """
        print("  RQ5.1: Preemption latency vs kernel duration")

        # REVERSED: Few high-prio, many low-prio (opposite of before)
        num_streams = 8
        high_prio_streams = 2  # REDUCED from 4
        low_prio_streams = 6   # INCREASED from 4

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
                # Priority spec: first 2 high (-5), last 6 low (0)
                prio_spec = ','.join(['-5'] * high_prio_streams + ['0'] * low_prio_streams)

                # Per-stream sizes: high-prio gets small fixed size, low-prio gets variable bg_size
                size_spec = ','.join(['65536'] * high_prio_streams + [str(bg_size)] * low_prio_streams)

                # REVERSED load imbalance: low-prio launches MUCH MORE to saturate GPU
                # High-prio: sparse arrivals (20 kernels total)
                # Low-prio: heavy load (100 kernels per stream = 600 total)
                load_spec = ','.join(['20'] * high_prio_streams + ['100'] * low_prio_streams)

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
        """
        Aggregate RQ5.1 raw CSV files and compute preemption latency metrics.

        KEY CHANGE: Only measure preemption latency for high-priority kernels that
        arrived when GPU was busy with low-priority kernels (true preemption scenario).
        """
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

                # Separate high and low priority kernels
                hi = df[df['priority'] < 0].copy()
                lo = df[df['priority'] >= 0].copy()

                if len(hi) == 0 or len(lo) == 0:
                    continue

                # Build intervals when low-priority kernels were running
                lo_intervals = list(zip(lo['start_time_ms'], lo['end_time_ms']))

                def is_gpu_busy_with_low_prio(enqueue_t):
                    """Check if any low-priority kernel was running at time t."""
                    for start, end in lo_intervals:
                        if start <= enqueue_t < end:
                            return True
                    return False

                # Filter: only high-prio kernels that arrived when GPU was busy
                hi['gpu_busy'] = hi['enqueue_time_ms'].apply(is_gpu_busy_with_low_prio)
                hi_blocked = hi[hi['gpu_busy']].copy()

                # Calculate preemption latency for blocked kernels only
                if len(hi_blocked) == 0:
                    # If no kernels were blocked, skip this run
                    continue

                hi_blocked['preempt_latency_ms'] = hi_blocked['start_time_ms'] - hi_blocked['enqueue_time_ms']

                # Aggregate statistics
                row = {
                    'bg_kernel_size': bg_size,
                    'bg_label': bg_label,
                    'preempt_latency_mean': hi_blocked['preempt_latency_ms'].mean(),
                    'preempt_latency_p50': hi_blocked['preempt_latency_ms'].quantile(0.50),
                    'preempt_latency_p95': hi_blocked['preempt_latency_ms'].quantile(0.95),
                    'preempt_latency_p99': hi_blocked['preempt_latency_ms'].quantile(0.99),
                    'preempt_latency_max': hi_blocked['preempt_latency_ms'].max(),
                    'high_prio_duration_mean': hi_blocked['duration_ms'].mean(),
                    'num_high_prio_kernels': len(hi),
                    'num_blocked_kernels': len(hi_blocked),
                    'blocked_ratio': len(hi_blocked) / len(hi) if len(hi) > 0 else 0,
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
                'num_blocked_kernels': 'sum',
                'blocked_ratio': 'mean',
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

        Measures how preemption latency varies with high-priority arrival rate,
        while keeping GPU saturated with low-priority background work.

        DESIGN: Fixed heavy background load + sweep high-priority arrival rate
        - Low-prio: Fixed high frequency (500Hz) to saturate GPU
        - High-prio: Variable frequency (20-1000Hz) to measure preemption under load
        """
        print("  RQ5.2: Preemption latency vs offered load")

        num_streams = 8
        high_prio_streams = 2  # REDUCED: sparse high-prio streams
        low_prio_streams = 6   # INCREASED: saturate with low-prio

        # Use heterogeneous kernel sizes: high-prio small (RT), low-prio large (BE)
        hi_prio_size = 65536    # Small/fast (~0.1ms, RT kernels)
        lo_prio_size = 1048576  # Large/slow (~2ms, BE kernels)

        # Sweep HIGH-PRIORITY arrival rates (not global)
        hi_prio_frequencies = [20, 50, 100, 200, 500, 1000]  # Hz
        lo_prio_freq = 500  # Fixed: keep background saturated

        for hi_freq in hi_prio_frequencies:
            print(f"    High-prio freq={hi_freq}Hz, Low-prio freq={lo_prio_freq}Hz (background)")

            for run_idx in range(self.num_runs):
                # Priority spec: first 2 high (-5), last 6 low (0)
                prio_spec = ','.join(['-5'] * high_prio_streams + ['0'] * low_prio_streams)

                # Different frequencies: high-prio variable, low-prio fixed high
                freq_spec = ','.join([str(hi_freq)] * high_prio_streams +
                                    [str(lo_prio_freq)] * low_prio_streams)

                # Per-stream sizes: high-prio small, low-prio large
                sizes_spec = ','.join([str(hi_prio_size)] * high_prio_streams +
                                     [str(lo_prio_size)] * low_prio_streams)

                raw_csv_file = self.output_dir / f'rq5_2_raw_freq{hi_freq}_run{run_idx}.csv'

                args = [
                    '--streams', str(num_streams),
                    '--kernels', '50',
                    '--type', 'mixed',
                    '--per-stream-sizes', sizes_spec,
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
        self._aggregate_rq5_2_data(hi_prio_frequencies)

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

                # Separate high-priority and low-priority kernels
                hi = df[df['priority'] < 0].copy()
                lo = df[df['priority'] >= 0].copy()

                if len(hi) == 0:
                    continue

                # Build low-priority execution intervals to detect GPU busy periods
                lo_intervals = list(zip(lo['start_time_ms'], lo['end_time_ms']))

                def is_gpu_busy_with_low_prio(enqueue_t):
                    """Check if GPU was busy with low-priority work when kernel was enqueued."""
                    for start, end in lo_intervals:
                        if start <= enqueue_t < end:
                            return True
                    return False

                # Filter: only measure latency for kernels that arrived when GPU was busy
                hi['gpu_busy'] = hi['enqueue_time_ms'].apply(is_gpu_busy_with_low_prio)
                hi_blocked = hi[hi['gpu_busy']].copy()

                # Calculate preemption latency for blocked kernels
                if len(hi_blocked) > 0:
                    hi_blocked['preempt_latency_ms'] = hi_blocked['start_time_ms'] - hi_blocked['enqueue_time_ms']

                    preempt_mean = hi_blocked['preempt_latency_ms'].mean()
                    preempt_p50 = hi_blocked['preempt_latency_ms'].quantile(0.50)
                    preempt_p95 = hi_blocked['preempt_latency_ms'].quantile(0.95)
                    preempt_p99 = hi_blocked['preempt_latency_ms'].quantile(0.99)
                    preempt_max = hi_blocked['preempt_latency_ms'].max()
                    num_blocked = len(hi_blocked)
                else:
                    # No blocked kernels in this run
                    preempt_mean = 0.0
                    preempt_p50 = 0.0
                    preempt_p95 = 0.0
                    preempt_p99 = 0.0
                    preempt_max = 0.0
                    num_blocked = 0

                # Aggregate statistics
                # offered_load: total arrival rate = 2*hi_freq + 6*lo_freq (500Hz fixed)
                hi_prio_load = 2 * freq  # 2 high-prio streams
                lo_prio_load = 6 * 500   # 6 low-prio streams at 500Hz
                total_offered_load = hi_prio_load + lo_prio_load

                row = {
                    'hi_prio_freq': freq,  # High-priority arrival rate per stream
                    'hi_prio_load': hi_prio_load,  # Total high-prio arrival rate
                    'total_offered_load': total_offered_load,  # Total system load
                    'preempt_latency_mean': preempt_mean,
                    'preempt_latency_p50': preempt_p50,
                    'preempt_latency_p95': preempt_p95,
                    'preempt_latency_p99': preempt_p99,
                    'preempt_latency_max': preempt_max,
                    'num_high_prio_kernels': len(hi),
                    'num_blocked_kernels': num_blocked,
                    'blocked_ratio': num_blocked / len(hi) if len(hi) > 0 else 0.0,
                }
                aggregated_data.append(row)

        if aggregated_data:
            agg_df = pd.DataFrame(aggregated_data)

            # Group by hi_prio_freq and compute mean/std across runs
            final_df = agg_df.groupby(['hi_prio_freq', 'hi_prio_load', 'total_offered_load']).agg({
                'preempt_latency_mean': ['mean', 'std'],
                'preempt_latency_p50': ['mean', 'std'],
                'preempt_latency_p95': ['mean', 'std'],
                'preempt_latency_p99': ['mean', 'std'],
                'preempt_latency_max': ['mean', 'std'],
                'num_high_prio_kernels': 'sum',
                'num_blocked_kernels': 'sum',
                'blocked_ratio': 'mean',
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
        for _, row in df_sorted.iterrows():
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

        # Sort by high-priority frequency for proper line plotting
        df_sorted = df.sort_values('hi_prio_freq')

        # Plot P99 preemption latency vs high-priority launch frequency
        ax.plot(df_sorted['hi_prio_freq'], df_sorted['preempt_latency_p99_mean'],
                marker='o', linewidth=2, markersize=8, color='darkred', label='P99')
        ax.plot(df_sorted['hi_prio_freq'], df_sorted['preempt_latency_mean_mean'],
                marker='s', linewidth=2, markersize=6, color='orange', alpha=0.6, label='Mean')
        ax.plot(df_sorted['hi_prio_freq'], df_sorted['preempt_latency_p95_mean'],
                marker='d', linewidth=2, markersize=6, color='green', alpha=0.5, label='P95')

        ax.set_xlabel('High-Priority Launch Frequency (Hz)')
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

        # Filter for blocked high-priority tasks only
        def filter_blocked_kernels(df):
            """Filter high-priority kernels that arrived when GPU was busy with low-priority work."""
            hi = df[df['priority'] < 0].copy()
            lo = df[df['priority'] >= 0].copy()

            if len(hi) == 0 or len(lo) == 0:
                return hi

            # Build low-priority execution intervals
            lo_intervals = list(zip(lo['start_time_ms'], lo['end_time_ms']))

            def is_gpu_busy_with_low_prio(enqueue_t):
                for start, end in lo_intervals:
                    if start <= enqueue_t < end:
                        return True
                return False

            hi['gpu_busy'] = hi['enqueue_time_ms'].apply(is_gpu_busy_with_low_prio)
            return hi[hi['gpu_busy']].copy()

        small_blocked = filter_blocked_kernels(small_all)
        large_blocked = filter_blocked_kernels(large_all)

        # Calculate preemption latency for blocked kernels
        small_blocked['preempt_latency'] = small_blocked['start_time_ms'] - small_blocked['enqueue_time_ms']
        large_blocked['preempt_latency'] = large_blocked['start_time_ms'] - large_blocked['enqueue_time_ms']

        # Plot CDFs
        small_sorted = np.sort(small_blocked['preempt_latency'].values)
        large_sorted = np.sort(large_blocked['preempt_latency'].values)

        small_cdf = np.arange(1, len(small_sorted) + 1) / len(small_sorted)
        large_cdf = np.arange(1, len(large_sorted) + 1) / len(large_sorted)

        ax.plot(small_sorted, small_cdf, linewidth=2, color='blue',
               label=f'Small BG Kernel (65k, n={len(small_blocked)})')
        ax.plot(large_sorted, large_cdf, linewidth=2, color='red',
               label=f'Large BG Kernel (4M, n={len(large_blocked)})')

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

        # Filter for blocked high-priority tasks only
        def filter_blocked_kernels(df):
            """Filter high-priority kernels that arrived when GPU was busy with low-priority work."""
            hi = df[df['priority'] < 0].copy()
            lo = df[df['priority'] >= 0].copy()

            if len(hi) == 0 or len(lo) == 0:
                return hi

            # Build low-priority execution intervals
            lo_intervals = list(zip(lo['start_time_ms'], lo['end_time_ms']))

            def is_gpu_busy_with_low_prio(enqueue_t):
                for start, end in lo_intervals:
                    if start <= enqueue_t < end:
                        return True
                return False

            hi['gpu_busy'] = hi['enqueue_time_ms'].apply(is_gpu_busy_with_low_prio)
            return hi[hi['gpu_busy']].copy()

        low_blocked = filter_blocked_kernels(low_all)
        high_blocked = filter_blocked_kernels(high_all)

        # Calculate preemption latency for blocked kernels
        low_blocked['preempt_latency'] = low_blocked['start_time_ms'] - low_blocked['enqueue_time_ms']
        high_blocked['preempt_latency'] = high_blocked['start_time_ms'] - high_blocked['enqueue_time_ms']

        # Plot CDFs
        low_sorted = np.sort(low_blocked['preempt_latency'].values)
        high_sorted = np.sort(high_blocked['preempt_latency'].values)

        low_cdf = np.arange(1, len(low_sorted) + 1) / len(low_sorted)
        high_cdf = np.arange(1, len(high_sorted) + 1) / len(high_sorted)

        ax.plot(low_sorted, low_cdf, linewidth=2, color='green',
               label=f'Low Load (20Hz, n={len(low_blocked)})')
        ax.plot(high_sorted, high_cdf, linewidth=2, color='purple',
               label=f'High Load (1000Hz, n={len(high_blocked)})')

        ax.set_xlabel('Preemption Latency (ms)')
        ax.set_ylabel('CDF')
        ax.set_title('(d) Preemption Latency CDF vs Offered Load')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim([0, 1])
