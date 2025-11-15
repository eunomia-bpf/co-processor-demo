"""RQ3: Latency & Queueing"""
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from .base import RQBase


class RQ3(RQBase):
    """RQ3: Latency distribution and queueing behavior."""

    def run_experiments(self):
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

        # Sub-RQ3.1: CDF analysis - need raw CSV output
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

    def analyze(self):
        """Analyze RQ3: Latency and queueing - combined figure."""
        print("\n=== Analyzing RQ3: Latency & Queueing ===")

        # Create a 2x2 subplot figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # RQ3.1: E2E latency CDF (low vs high concurrency)
        print("  RQ3.1: E2E latency CDF")
        low_dfs = self._load_raw_csvs('rq3_1_raw_low_concurrency_*.csv')
        high_dfs = self._load_raw_csvs('rq3_1_raw_high_concurrency_*.csv')

        if low_dfs or high_dfs:
            # Plot CDF for low concurrency
            if low_dfs:
                all_latencies = pd.concat([df['e2e_latency_ms'] for df in low_dfs])
                sorted_lat = np.sort(all_latencies)
                cdf = np.arange(1, len(sorted_lat) + 1) / len(sorted_lat)
                ax1.plot(sorted_lat, cdf, label='Low Concurrency (2 streams)',
                         linewidth=2, color='blue')

            # Plot CDF for high concurrency
            if high_dfs:
                all_latencies = pd.concat([df['e2e_latency_ms'] for df in high_dfs])
                sorted_lat = np.sort(all_latencies)
                cdf = np.arange(1, len(sorted_lat) + 1) / len(sorted_lat)
                ax1.plot(sorted_lat, cdf, label='High Concurrency (32 streams)',
                         linewidth=2, color='red')

            ax1.set_xlabel('End-to-End Latency (ms)')
            ax1.set_ylabel('CDF')
            ax1.set_title('(a) E2E Latency CDF (Low vs High Concurrency)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        else:
            print("    Warning: No raw CSV files found for RQ3.1")

        # RQ3.2: E2E P99 vs streams (different types)
        print("  RQ3.2: E2E P99 vs streams")
        df = self.load_csv('rq3_latency_vs_streams.csv')
        if df is not None:
            grouped = df.groupby(['streams', 'type']).agg({
                'e2e_p99': ['mean', 'std']
            }).reset_index()

            grouped.columns = ['streams', 'type', 'e2e_p99_mean', 'e2e_p99_std']

            for ktype in ['compute', 'memory', 'mixed', 'gemm']:
                data = grouped[grouped['type'] == ktype]
                if len(data) > 0:
                    ax2.plot(data['streams'], data['e2e_p99_mean'],
                             marker='o', label=ktype.upper(), linewidth=2)
                    ax2.fill_between(data['streams'],
                                     data['e2e_p99_mean'] - data['e2e_p99_std'],
                                     data['e2e_p99_mean'] + data['e2e_p99_std'],
                                     alpha=0.2)

            ax2.set_xlabel('Number of Streams')
            ax2.set_ylabel('E2E P99 Latency (ms)')
            ax2.set_title('(b) E2E P99 Latency vs Stream Count')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_xscale('log', base=2)

        # RQ3.3: Queue wait vs streams (combined avg and max in one subplot)
        print("  RQ3.3: queue wait vs streams")
        df = self.load_csv('rq3_latency_vs_streams.csv')
        if df is not None:
            # Group across all types
            grouped = df.groupby('streams').agg({
                'avg_queue_wait': ['mean', 'std'],
                'max_queue_wait': ['mean', 'std']
            }).reset_index()

            grouped.columns = ['streams', 'avg_queue_wait_mean', 'avg_queue_wait_std',
                               'max_queue_wait_mean', 'max_queue_wait_std']

            # Plot both average and max on same subplot
            ax3.plot(grouped['streams'], grouped['avg_queue_wait_mean'],
                     marker='o', linewidth=2, color='blue', label='Average')
            ax3.fill_between(grouped['streams'],
                             grouped['avg_queue_wait_mean'] - grouped['avg_queue_wait_std'],
                             grouped['avg_queue_wait_mean'] + grouped['avg_queue_wait_std'],
                             alpha=0.2, color='blue')

            ax3.plot(grouped['streams'], grouped['max_queue_wait_mean'],
                     marker='s', linewidth=2, color='red', label='Maximum')
            ax3.fill_between(grouped['streams'],
                             grouped['max_queue_wait_mean'] - grouped['max_queue_wait_std'],
                             grouped['max_queue_wait_mean'] + grouped['max_queue_wait_std'],
                             alpha=0.2, color='red')

            ax3.set_xlabel('Number of Streams')
            ax3.set_ylabel('Queue Wait (ms)')
            ax3.set_title('(c) Queue Wait vs Stream Count')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_xscale('log', base=2)

        # RQ3.4: Queue wait CDF (light vs heavy load)
        print("  RQ3.4: Queue wait CDF")
        light_dfs = self._load_raw_csvs('rq3_4_raw_light_*.csv')
        medium_dfs = self._load_raw_csvs('rq3_4_raw_medium_*.csv')
        heavy_dfs = self._load_raw_csvs('rq3_4_raw_heavy_*.csv')

        if light_dfs or medium_dfs or heavy_dfs:
            # Plot CDF for each load level
            if light_dfs:
                all_latencies = pd.concat([df['launch_latency_ms'] for df in light_dfs])
                sorted_lat = np.sort(all_latencies)
                cdf = np.arange(1, len(sorted_lat) + 1) / len(sorted_lat)
                ax4.plot(sorted_lat, cdf, label='Light Load (20 Hz)',
                         linewidth=2, color='green')

            if medium_dfs:
                all_latencies = pd.concat([df['launch_latency_ms'] for df in medium_dfs])
                sorted_lat = np.sort(all_latencies)
                cdf = np.arange(1, len(sorted_lat) + 1) / len(sorted_lat)
                ax4.plot(sorted_lat, cdf, label='Medium Load (100 Hz)',
                         linewidth=2, color='orange')

            if heavy_dfs:
                all_latencies = pd.concat([df['launch_latency_ms'] for df in heavy_dfs])
                sorted_lat = np.sort(all_latencies)
                cdf = np.arange(1, len(sorted_lat) + 1) / len(sorted_lat)
                ax4.plot(sorted_lat, cdf, label='Heavy Load (500 Hz)',
                         linewidth=2, color='red')

            ax4.set_xlabel('Queue Wait / Launch Latency (ms)')
            ax4.set_ylabel('CDF')
            ax4.set_title('(d) Queue Wait CDF at Different Load Levels')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            print("    Warning: No raw CSV files found for RQ3.4")

        fig.suptitle('RQ3: Latency & Queueing', fontsize=16, fontweight='bold')
        self.save_figure('rq3_latency')

    def _load_raw_csvs(self, pattern: str):
        """Load multiple raw CSV files matching a pattern."""
        import glob
        # Raw CSV files are in output_dir, not results_dir
        files = glob.glob(str(self.output_dir / pattern))
        dfs = []
        for f in files:
            try:
                df = pd.read_csv(f)
                dfs.append(df)
            except Exception as e:
                print(f"Error loading {f}: {e}")
        return dfs
