"""RQ1: Stream Scalability & Concurrency"""
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from .base import RQBase


class RQ1(RQBase):
    """RQ1: Stream scalability and concurrency capability."""

    def run_experiments(self):
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

    def analyze(self):
        """Analyze RQ1: Stream scalability and concurrency - combined figure."""
        print("\n=== Analyzing RQ1: Stream Scalability & Concurrency ===")

        df = self.load_csv('rq1_stream_scalability.csv')
        if df is None:
            return

        size_labels = {
            65536: 'Small (~50us)',
            262144: 'Medium (~200us)',
            1048576: 'Large (~1ms)',
            4194304: 'XLarge (~5ms)',
        }

        # Create a 2x2 subplot figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # RQ1.1: max_concurrent vs streams
        print("  RQ1.1: max_concurrent vs streams")
        grouped = df.groupby(['streams', 'size']).agg({
            'max_concurrent': ['mean', 'std']
        }).reset_index()
        grouped.columns = ['streams', 'size', 'max_concurrent_mean', 'max_concurrent_std']

        for size, label in size_labels.items():
            data = grouped[grouped['size'] == size]
            if len(data) > 0:
                ax1.plot(data['streams'], data['max_concurrent_mean'],
                         marker='o', label=label, linewidth=2)
                ax1.fill_between(data['streams'],
                                 data['max_concurrent_mean'] - data['max_concurrent_std'],
                                 data['max_concurrent_mean'] + data['max_concurrent_std'],
                                 alpha=0.2)

        ax1.set_xlabel('Number of Streams')
        ax1.set_ylabel('Max Concurrent Kernels')
        ax1.set_title('(a) Maximum Concurrent Kernels')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log', base=2)

        # RQ1.2: concurrent_rate vs streams
        print("  RQ1.2: concurrent_rate vs streams")
        grouped = df.groupby(['streams', 'size']).agg({
            'concurrent_rate': ['mean', 'std']
        }).reset_index()
        grouped.columns = ['streams', 'size', 'concurrent_rate_mean', 'concurrent_rate_std']

        for size, label in size_labels.items():
            data = grouped[grouped['size'] == size]
            if len(data) > 0:
                ax2.plot(data['streams'], data['concurrent_rate_mean'],
                         marker='o', label=label, linewidth=2)
                ax2.fill_between(data['streams'],
                                 data['concurrent_rate_mean'] - data['concurrent_rate_std'],
                                 data['concurrent_rate_mean'] + data['concurrent_rate_std'],
                                 alpha=0.2)

        ax2.set_xlabel('Number of Streams')
        ax2.set_ylabel('Concurrent Rate (%)')
        ax2.set_title('(b) Concurrent Execution Rate')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log', base=2)

        # RQ1.3: GPU utilization vs streams
        print("  RQ1.3: util vs streams")
        grouped = df.groupby(['streams', 'size']).agg({
            'util': ['mean', 'std']
        }).reset_index()
        grouped.columns = ['streams', 'size', 'util_mean', 'util_std']

        for size, label in size_labels.items():
            data = grouped[grouped['size'] == size]
            if len(data) > 0:
                ax3.plot(data['streams'], data['util_mean'],
                         marker='o', label=label, linewidth=2)
                ax3.fill_between(data['streams'],
                                 data['util_mean'] - data['util_std'],
                                 data['util_mean'] + data['util_std'],
                                 alpha=0.2)

        ax3.set_xlabel('Number of Streams')
        ax3.set_ylabel('GPU Utilization (%)')
        ax3.set_title('(c) GPU Utilization')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xscale('log', base=2)

        # RQ1.4: avg_concurrent vs streams
        print("  RQ1.4: avg_concurrent vs streams")
        grouped = df.groupby(['streams', 'size']).agg({
            'avg_concurrent': ['mean', 'std']
        }).reset_index()
        grouped.columns = ['streams', 'size', 'avg_concurrent_mean', 'avg_concurrent_std']

        for size, label in size_labels.items():
            data = grouped[grouped['size'] == size]
            if len(data) > 0:
                ax4.plot(data['streams'], data['avg_concurrent_mean'],
                         marker='o', label=label, linewidth=2)
                ax4.fill_between(data['streams'],
                                 data['avg_concurrent_mean'] - data['avg_concurrent_std'],
                                 data['avg_concurrent_mean'] + data['avg_concurrent_std'],
                                 alpha=0.2)

        ax4.set_xlabel('Number of Streams')
        ax4.set_ylabel('Average Concurrent Kernels')
        ax4.set_title('(d) Average Concurrent Kernels')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xscale('log', base=2)

        fig.suptitle('RQ1: Stream Scalability & Concurrency', fontsize=16, fontweight='bold')
        self.save_figure('rq1_stream_scalability')
