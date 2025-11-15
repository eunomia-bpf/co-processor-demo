"""RQ7: Arrival Pattern & Jitter (formerly RQ6)"""
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from .base import RQBase


class RQ7(RQBase):
    """RQ7: Arrival pattern and jitter effects (formerly RQ6)."""

    def run_experiments(self):
        """
        RQ7: Arrival pattern and jitter effects.

        Sub-RQ7.1: concurrent_rate vs jitter
        Sub-RQ7.2: E2E P99 vs jitter
        """
        print("\n=== Running RQ7 Experiments: Arrival Pattern & Jitter ===")

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

        self.save_csv(csv_lines, 'rq7_jitter_effects.csv')

    def analyze(self):
        """
        Analyze RQ7: Arrival pattern and jitter effects - combined figure.

        NOTE: Jitter (randomized arrival times) REDUCES concurrent execution rate compared
        to periodic arrivals. This is EXPECTED behavior - periodic arrivals create bursts
        that enable more kernels to overlap, while random arrivals spread out submissions.
        """
        print("\n=== Analyzing RQ7: Arrival Pattern & Jitter ===")

        # Create a 1x2 subplot figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Load data once
        df = self.load_csv('rq7_jitter_effects.csv')
        if df is not None:
            # Use actual seed column if available
            # seed=0 means periodic (no jitter), seed!=0 means random (with jitter)
            df['has_jitter'] = df.get('seed', pd.Series([0]*len(df))) != 0

            print(f"    Data summary: {len(df[~df['has_jitter']])} periodic runs, "
                  f"{len(df[df['has_jitter']])} jittered runs")

            # Use launch_freq if available, otherwise estimate from throughput
            if 'launch_freq' in df.columns:
                df['freq_bin_val'] = df['launch_freq']
            else:
                df['freq_bin_val'] = df['total_kernels'] / (df['wall_time_ms'] / 1000.0) / df['streams']

            freq_bins = [0, 75, 150, 300, 600]
            freq_labels = ['50Hz', '100Hz', '200Hz', '500Hz']
            df['freq_bin'] = pd.cut(df['freq_bin_val'], bins=freq_bins, labels=freq_labels)

            # RQ7.1: concurrent_rate vs jitter
            print("  RQ7.1: Concurrent rate vs jitter")
            grouped = df.groupby(['freq_bin', 'has_jitter']).agg({
                'concurrent_rate': ['mean', 'std']
            }).reset_index()

            grouped.columns = ['freq_bin', 'has_jitter', 'concurrent_rate_mean', 'concurrent_rate_std']

            freq_bins_unique = grouped['freq_bin'].unique()
            x = np.arange(len(freq_bins_unique))
            width = 0.35

            no_jitter = grouped[grouped['has_jitter'] == False]
            with_jitter = grouped[grouped['has_jitter'] == True]

            if len(no_jitter) > 0:
                yerr = no_jitter['concurrent_rate_std'].fillna(0) if no_jitter['concurrent_rate_std'].notna().any() else None
                ax1.bar(x - width/2, no_jitter['concurrent_rate_mean'], width,
                        yerr=yerr,
                        label='No Jitter', capsize=5, alpha=0.7)
            if len(with_jitter) > 0:
                yerr = with_jitter['concurrent_rate_std'].fillna(0) if with_jitter['concurrent_rate_std'].notna().any() else None
                ax1.bar(x + width/2, with_jitter['concurrent_rate_mean'], width,
                        yerr=yerr,
                        label='With Jitter', capsize=5, alpha=0.7)

            ax1.set_xlabel('Launch Frequency')
            ax1.set_ylabel('Concurrent Rate (%)')
            ax1.set_title('(a) Concurrent Rate vs Jitter')
            ax1.set_xticks(x)
            ax1.set_xticklabels(freq_bins_unique)
            ax1.legend()
            ax1.grid(True, alpha=0.3, axis='y')

            # RQ7.2: E2E P99 vs jitter
            print("  RQ7.2: E2E P99 vs jitter")
            grouped = df.groupby(['freq_bin', 'has_jitter']).agg({
                'e2e_p99': ['mean', 'std']
            }).reset_index()

            grouped.columns = ['freq_bin', 'has_jitter', 'e2e_p99_mean', 'e2e_p99_std']

            no_jitter = grouped[grouped['has_jitter'] == False]
            with_jitter = grouped[grouped['has_jitter'] == True]

            if len(no_jitter) > 0:
                yerr = no_jitter['e2e_p99_std'].fillna(0) if no_jitter['e2e_p99_std'].notna().any() else None
                ax2.bar(x - width/2, no_jitter['e2e_p99_mean'], width,
                        yerr=yerr,
                        label='No Jitter (Periodic)', capsize=5, alpha=0.7)
            if len(with_jitter) > 0:
                yerr = with_jitter['e2e_p99_std'].fillna(0) if with_jitter['e2e_p99_std'].notna().any() else None
                ax2.bar(x + width/2, with_jitter['e2e_p99_mean'], width,
                        yerr=yerr,
                        label='With Jitter (Random)', capsize=5, alpha=0.7)

            ax2.set_xlabel('Launch Frequency')
            ax2.set_ylabel('E2E P99 Latency (ms)')
            ax2.set_title('(b) E2E P99 Latency vs Jitter')
            ax2.set_xticks(x)
            ax2.set_xticklabels(freq_bins_unique)
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')

        fig.suptitle('RQ7: Arrival Pattern & Jitter', fontsize=16, fontweight='bold')
        self.save_figure('rq7_jitter')
