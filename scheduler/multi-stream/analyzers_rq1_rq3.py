#!/usr/bin/env python3
"""
GPU Scheduler Experiment Analysis - RQ1-RQ3 Analyzer

Analyzes stream scalability, throughput, and latency research questions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional, Dict
import glob


class RQ1_RQ3_Analyzer:
    """Analyzer for RQ1 (Scalability), RQ2 (Throughput), and RQ3 (Latency)."""

    def __init__(self, results_dir: Path, output_dir: Path, fig_format: str = 'png', dpi: int = 300):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.fig_format = fig_format
        self.dpi = dpi

        # Set up plotting style
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=1.3)
        plt.rcParams['figure.figsize'] = (8, 6)

    def load_csv(self, filename: str) -> Optional[pd.DataFrame]:
        """Load CSV file from results directory."""
        filepath = self.results_dir / filename
        if not filepath.exists():
            print(f"Warning: File not found: {filepath}")
            return None

        try:
            df = pd.read_csv(filepath)
            # Add 'size' column derived from grid_size if not present
            # The CSV has grid_size which determines workload_size (grid_size * block_size)
            if 'grid_size' in df.columns and 'size' not in df.columns:
                df['size'] = df['grid_size'] * df.get('block_size', 256)
            elif 'workload_size' in df.columns and 'size' not in df.columns:
                df['size'] = df['workload_size']
            return df
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None

    def load_raw_csvs(self, pattern: str) -> List[pd.DataFrame]:
        """Load multiple raw CSV files matching a pattern."""
        files = glob.glob(str(self.results_dir / pattern))
        dfs = []
        for f in files:
            try:
                df = pd.read_csv(f)
                dfs.append(df)
            except Exception as e:
                print(f"Error loading {f}: {e}")
        return dfs

    def save_figure(self, filename: str):
        """Save current figure to output directory."""
        filepath = self.output_dir / f"{filename}.{self.fig_format}"
        plt.tight_layout()
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        print(f"Saved figure: {filepath}")
        plt.close()

    # ========================================================================
    # RQ1: Stream Scalability & Concurrency
    # ========================================================================

    def analyze_rq1(self, sub_rqs: Optional[List[str]] = None):
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

    # ========================================================================
    # RQ2: Throughput & Workload Type
    # ========================================================================

    def analyze_rq2(self, sub_rqs: Optional[List[str]] = None):
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
            df['has_jitter'] = df.get('seed', 0) != 0
            df['offered_load_est'] = df['total_kernels'] / (df['wall_time_ms'] / 1000.0)

            df_sorted = df.sort_values('offered_load_est')

            for has_jitter in [False, True]:
                data = df_sorted[df_sorted['has_jitter'] == has_jitter]
                if len(data) > 0:
                    label = 'With Jitter' if has_jitter else 'No Jitter'
                    ax3.scatter(data['offered_load_est'], data['throughput'],
                                label=label, alpha=0.6, s=50)

            ax3.set_xlabel('Offered Load (kernels/sec)')
            ax3.set_ylabel('Achieved Throughput (kernels/sec)')
            ax3.set_title('(c) Throughput vs Offered Load')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            # Add y=x reference line
            if len(df) > 0:
                max_val = max(df['offered_load_est'].max(), df['throughput'].max())
                ax3.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='y=x')

        fig.suptitle('RQ2: Throughput & Workload Type', fontsize=16, fontweight='bold')
        self.save_figure('rq2_throughput')

    # ========================================================================
    # RQ3: Latency & Queueing
    # ========================================================================

    def analyze_rq3(self, sub_rqs: Optional[List[str]] = None):
        """Analyze RQ3: Latency and queueing - combined figure."""
        print("\n=== Analyzing RQ3: Latency & Queueing ===")

        # Create a 2x2 subplot figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # RQ3.1: E2E latency CDF (low vs high concurrency)
        print("  RQ3.1: E2E latency CDF")
        low_dfs = self.load_raw_csvs('rq3_1_raw_low_concurrency_*.csv')
        high_dfs = self.load_raw_csvs('rq3_1_raw_high_concurrency_*.csv')

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
        light_dfs = self.load_raw_csvs('rq3_4_raw_light_*.csv')
        medium_dfs = self.load_raw_csvs('rq3_4_raw_medium_*.csv')
        heavy_dfs = self.load_raw_csvs('rq3_4_raw_heavy_*.csv')

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

    # ========================================================================
    # Main analysis driver
    # ========================================================================

    def analyze_all(self):
        """Run all RQ1-RQ3 analyses."""
        self.analyze_rq1()
        self.analyze_rq2()
        self.analyze_rq3()
