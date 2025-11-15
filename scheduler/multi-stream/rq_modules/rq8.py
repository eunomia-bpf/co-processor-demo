"""RQ8: Working Set vs L2 Cache (formerly RQ7)"""
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from .base import RQBase


class RQ8(RQBase):
    """RQ8: Memory working set vs L2 cache boundary (formerly RQ7)."""

    def run_experiments(self):
        """
        RQ8: Memory working set vs L2 cache boundary.

        Sub-RQ8.1: throughput/util vs working_set/L2 ratio
        """
        print("\n=== Running RQ8 Experiments: Working Set vs L2 Cache ===")

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

        self.save_csv(csv_lines, 'rq8_working_set_vs_l2.csv')

    def analyze(self):
        """Analyze RQ8: Memory working set vs L2 cache boundary - combined figure."""
        print("\n=== Analyzing RQ8: Working Set vs L2 Cache ===")

        # RQ8.1: throughput/util vs working_set/L2 ratio
        print("  RQ8.1: Throughput/util vs working set")

        df = self.load_csv('rq8_working_set_vs_l2.csv')
        if df is not None:
            # Get L2 size from first row that has it, or use RTX 5090 default
            l2_size_mb = 96.0  # RTX 5090 L2 cache size
            if 'l2_cache_mb' in df.columns:
                l2_vals = df['l2_cache_mb'].dropna()
                if len(l2_vals) > 0:
                    l2_size_mb = l2_vals.iloc[0]
                    print(f"    Using L2 cache size from CSV: {l2_size_mb:.2f} MB")
            else:
                print(f"    Using detected L2 cache size: {l2_size_mb:.2f} MB")

            df['ws_l2_ratio'] = df['working_set_mb'] / l2_size_mb

            print(f"    Working set range: {df['working_set_mb'].min():.1f} - {df['working_set_mb'].max():.1f} MB")
            print(f"    WS/L2 ratio range: {df['ws_l2_ratio'].min():.3f} - {df['ws_l2_ratio'].max():.3f}")

            grouped = df.groupby(['ws_l2_ratio', 'type']).agg({
                'throughput': ['mean', 'std'],
                'util': ['mean', 'std']
            }).reset_index()

            grouped.columns = ['ws_l2_ratio', 'type', 'throughput_mean', 'throughput_std',
                               'util_mean', 'util_std']

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            # Throughput
            for ktype in ['memory', 'mixed', 'compute']:
                data = grouped[grouped['type'] == ktype].sort_values('ws_l2_ratio')
                if len(data) > 0:
                    ax1.plot(data['ws_l2_ratio'], data['throughput_mean'],
                             marker='o', label=ktype.upper(), linewidth=2)
                    ax1.fill_between(data['ws_l2_ratio'],
                                     data['throughput_mean'] - data['throughput_std'],
                                     data['throughput_mean'] + data['throughput_std'],
                                     alpha=0.2)

            ax1.axvline(x=1.0, color='r', linestyle='--', alpha=0.5, label='L2 Boundary')
            ax1.set_xlabel('Working Set / L2 Cache Ratio')
            ax1.set_ylabel('Throughput (kernels/sec)')
            ax1.set_title('(a) Throughput vs Working Set / L2 Ratio')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Utilization
            for ktype in ['memory', 'mixed', 'compute']:
                data = grouped[grouped['type'] == ktype].sort_values('ws_l2_ratio')
                if len(data) > 0:
                    ax2.plot(data['ws_l2_ratio'], data['util_mean'],
                             marker='o', label=ktype.upper(), linewidth=2)
                    ax2.fill_between(data['ws_l2_ratio'],
                                     data['util_mean'] - data['util_std'],
                                     data['util_mean'] + data['util_std'],
                                     alpha=0.2)

            ax2.axvline(x=1.0, color='r', linestyle='--', alpha=0.5, label='L2 Boundary')
            ax2.set_xlabel('Working Set / L2 Cache Ratio')
            ax2.set_ylabel('GPU Utilization (%)')
            ax2.set_title('(b) Utilization vs Working Set / L2 Ratio')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            fig.suptitle('RQ8: Working Set vs L2 Cache', fontsize=16, fontweight='bold')
            self.save_figure('rq8_memory')
        else:
            print("    Warning: No data found for RQ8")
