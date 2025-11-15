"""RQ2: Throughput & Workload Type"""
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from .base import RQBase


class RQ2(RQBase):
    """RQ2: Throughput variations with stream count and workload type."""

    def run_experiments(self):
        """
        RQ2: Throughput variations with stream count and workload type.

        Sub-RQ2.1: throughput vs streams (different sizes) - reuses RQ1 data
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

    def analyze(self):
        """Analyze RQ2: Throughput variations - combined figure."""
        print("\n=== Analyzing RQ2: Throughput & Workload Type ===")

        # Create a 1x3 subplot figure
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

        # RQ2.1: throughput vs streams (different sizes)
        print("  RQ2.1: throughput vs streams (different sizes)")
        df = self.load_csv('rq1_stream_scalability.csv')
        if df is not None:

            size_labels = {
                65536: 'Small (~50us)',
                262144: 'Medium (~200us)',
                1048576: 'Large (~1ms)',
                4194304: 'XLarge (~5ms)',
            }

            grouped = df.groupby(['streams', 'size']).agg({
                'throughput': ['mean', 'std']
            }).reset_index()

            grouped.columns = ['streams', 'size', 'throughput_mean', 'throughput_std']

            for size, label in size_labels.items():
                data = grouped[grouped['size'] == size]
                if len(data) > 0:
                    ax1.plot(data['streams'], data['throughput_mean'],
                             marker='o', label=label, linewidth=2)
                    ax1.fill_between(data['streams'],
                                     data['throughput_mean'] - data['throughput_std'],
                                     data['throughput_mean'] + data['throughput_std'],
                                     alpha=0.2)

            ax1.set_xlabel('Number of Streams')
            ax1.set_ylabel('Throughput (kernels/sec)')
            ax1.set_title('(a) Throughput vs Stream Count (Different Sizes)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_xscale('log', base=2)

        # RQ2.2: throughput vs streams (different types)
        print("  RQ2.2: throughput vs streams (different types)")
        df = self.load_csv('rq2_2_throughput_by_type.csv')
        if df is not None:
            grouped = df.groupby(['streams', 'type']).agg({
                'throughput': ['mean', 'std']
            }).reset_index()

            grouped.columns = ['streams', 'type', 'throughput_mean', 'throughput_std']

            for ktype in ['compute', 'memory', 'mixed', 'gemm']:
                data = grouped[grouped['type'] == ktype]
                if len(data) > 0:
                    ax2.plot(data['streams'], data['throughput_mean'],
                             marker='o', label=ktype.upper(), linewidth=2)
                    ax2.fill_between(data['streams'],
                                     data['throughput_mean'] - data['throughput_std'],
                                     data['throughput_mean'] + data['throughput_std'],
                                     alpha=0.2)

            ax2.set_xlabel('Number of Streams')
            ax2.set_ylabel('Throughput (kernels/sec)')
            ax2.set_title('(b) Throughput vs Stream Count (Different Types)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_xscale('log', base=2)

        # RQ2.3: throughput vs offered load
        print("  RQ2.3: throughput vs offered load")
        df = self.load_csv('rq2_3_throughput_vs_load.csv')
        if df is not None:
            # Use actual seed column if available
            df['has_jitter'] = df.get('seed', pd.Series([0]*len(df))) != 0

            # Calculate offered load from launch_frequency
            if 'launch_freq' in df.columns:
                df['offered_load'] = df.apply(
                    lambda row: row['streams'] * row['launch_freq']
                    if row['launch_freq'] > 0
                    else row['throughput'],  # freq=0 means max rate
                    axis=1
                )
            else:
                # Fallback for old CSV without launch_freq
                df['offered_load'] = df['total_kernels'] / (df['wall_time_ms'] / 1000.0)

            df_sorted = df.sort_values('offered_load')

            for has_jitter in [False, True]:
                data = df_sorted[df_sorted['has_jitter'] == has_jitter]
                if len(data) > 0:
                    label = 'With Jitter' if has_jitter else 'No Jitter (Periodic)'
                    ax3.scatter(data['offered_load'], data['throughput'],
                                label=label, alpha=0.6, s=50)

            ax3.set_xlabel('Offered Load (kernels/sec)')
            ax3.set_ylabel('Achieved Throughput (kernels/sec)')
            ax3.set_title('(c) Throughput vs Offered Load')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            # Add y=x reference line
            if len(df) > 0:
                max_val = max(df['offered_load'].max(), df['throughput'].max())
                ax3.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Ideal (100% efficiency)')

        fig.suptitle('RQ2: Throughput & Workload Type', fontsize=16, fontweight='bold')
        self.save_figure('rq2_throughput')
