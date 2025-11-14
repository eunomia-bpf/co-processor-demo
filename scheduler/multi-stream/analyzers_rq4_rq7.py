#!/usr/bin/env python3
"""
RQ4-RQ7 Analyzers

Contains analysis methods for:
- RQ4: Memory Pressure Impact
- RQ5: Multi-Process Interference
- RQ6: Load Imbalance and Fairness
- RQ7: Tail Latency Under Contention
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
from analyzer_base import BaseAnalyzer


class RQ4_RQ7_Analyzer(BaseAnalyzer):
    """Analyzer for RQ4, RQ5, RQ6, and RQ7."""
    def analyze_rq4_memory_pressure(self) -> Dict:
        """Analyze RQ4: Memory Pressure."""
        print("\n=== RQ4: Memory Pressure Analysis ===")

        df = self.load_data("rq4_memory_pressure.csv")
        df['size_mb'] = pd.to_numeric(df.get('size_mb', df['wall_time_ms']))  # Fallback

        stats = df.groupby('size_mb').agg({
            'concurrent_rate': ['mean', 'std'],
            'throughput': ['mean', 'std'],
            'util': ['mean', 'std']
        }).round(2)

        print("\nMemory Size vs Metrics:")
        print(stats)

        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        self._plot_with_error(df, 'size_mb', 'concurrent_rate',
                             axes[0], 'Memory Size (MB)', 'Concurrent Rate (%)')

        self._plot_with_error(df, 'size_mb', 'throughput',
                             axes[1], 'Memory Size (MB)', 'Throughput (kernels/sec)')

        self._plot_with_error(df, 'size_mb', 'util',
                             axes[2], 'Memory Size (MB)', 'GPU Utilization (%)')

        plt.tight_layout()
        plt.savefig(self.figures_dir / "rq4_memory_pressure.png", dpi=300)
        plt.close()

        return {}

    def analyze_rq5_multi_process(self) -> Dict:
        """Analyze RQ5: Multi-Process Interference."""
        print("\n=== RQ5: Multi-Process Interference Analysis ===")

        df = self.load_data("rq5_multi_process.csv")
        df['num_processes'] = pd.to_numeric(df['num_processes'])

        # Aggregate throughput per process
        process_stats = df.groupby('num_processes').agg({
            'throughput': ['mean', 'std', 'count'],
            'jains_index': ['mean', 'std']
        }).round(2)

        print("\nPer-Process Metrics:")
        print(process_stats)

        # Calculate system-wide throughput
        system_throughput = df.groupby(['num_processes', 'trial']).agg({
            'throughput': 'sum'
        }).groupby('num_processes').agg({
            'throughput': ['mean', 'std']
        })

        print("\nSystem-Wide Throughput:")
        print(system_throughput)

        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Per-process throughput
        self._plot_with_error(df, 'num_processes', 'throughput',
                             axes[0], 'Number of Processes', 'Per-Process Throughput')

        # System throughput (summed)
        system_df = df.groupby(['num_processes', 'trial'])['throughput'].sum().reset_index()
        self._plot_with_error(system_df, 'num_processes', 'throughput',
                             axes[1], 'Number of Processes', 'System-Wide Throughput')

        # Fairness
        self._plot_with_error(df, 'num_processes', 'jains_index',
                             axes[2], 'Number of Processes', "Jain's Fairness Index")

        plt.tight_layout()
        plt.savefig(self.figures_dir / "rq5_multi_process.png", dpi=300)
        plt.close()

        return {}

    def analyze_rq6_load_imbalance(self) -> Dict:
        """Analyze RQ6: Load Imbalance and Fairness."""
        print("\n=== RQ6: Load Imbalance and Fairness Analysis ===")

        df = self.load_data("rq6_load_imbalance.csv")

        # Group by pattern
        stats = df.groupby('pattern_name').agg({
            'jains_index': ['mean', 'std'],
            'throughput': ['mean', 'std'],
            'overhead': ['mean', 'std'],
            'stddev': ['mean', 'std']
        }).round(4)

        print("\nLoad Imbalance Pattern Statistics:")
        print(stats)

        # Find most fair and most unfair
        pattern_fairness = df.groupby('pattern_name')['jains_index'].mean().sort_values(ascending=False)
        print(f"\n✓ Most fair pattern: {pattern_fairness.index[0]} (Jain's = {pattern_fairness.iloc[0]:.4f})")
        print(f"✓ Most unfair pattern: {pattern_fairness.index[-1]} (Jain's = {pattern_fairness.iloc[-1]:.4f})")

        # Fairness degradation
        balanced_fairness = df[df['pattern_name'] == 'balanced_8']['jains_index'].mean()
        imbalanced_fairness = df[df['pattern_name'] == 'imbalanced_4']['jains_index'].mean()
        degradation = ((balanced_fairness - imbalanced_fairness) / balanced_fairness) * 100
        print(f"✓ Fairness degradation (balanced → imbalanced): {degradation:.1f}%")

        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Fairness comparison
        patterns = df['pattern_name'].unique()
        fairness_means = [df[df['pattern_name'] == p]['jains_index'].mean() for p in patterns]
        fairness_stds = [df[df['pattern_name'] == p]['jains_index'].std() for p in patterns]

        axes[0, 0].barh(patterns, fairness_means, xerr=fairness_stds, capsize=5)
        axes[0, 0].set_xlabel("Jain's Fairness Index")
        axes[0, 0].set_ylabel('Load Pattern')
        axes[0, 0].set_title("Fairness Across Load Imbalance Patterns")
        axes[0, 0].axvline(x=1.0, color='r', linestyle='--', label='Perfect Fairness')
        axes[0, 0].axvline(x=0.8, color='orange', linestyle='--', label='Acceptable')
        axes[0, 0].legend()
        axes[0, 0].grid(True, axis='x')

        # Throughput comparison
        throughput_means = [df[df['pattern_name'] == p]['throughput'].mean() for p in patterns]
        throughput_stds = [df[df['pattern_name'] == p]['throughput'].std() for p in patterns]

        axes[0, 1].barh(patterns, throughput_means, xerr=throughput_stds, capsize=5, color='green', alpha=0.7)
        axes[0, 1].set_xlabel('Throughput (kernels/sec)')
        axes[0, 1].set_ylabel('Load Pattern')
        axes[0, 1].set_title("Throughput Across Load Imbalance Patterns")
        axes[0, 1].grid(True, axis='x')

        # Fairness vs Throughput scatter
        pattern_data = []
        for pattern in patterns:
            pattern_df = df[df['pattern_name'] == pattern]
            pattern_data.append({
                'pattern': pattern,
                'fairness': pattern_df['jains_index'].mean(),
                'throughput': pattern_df['throughput'].mean()
            })

        scatter_df = pd.DataFrame(pattern_data)
        axes[1, 0].scatter(scatter_df['fairness'], scatter_df['throughput'], s=100, alpha=0.7)

        for _, row in scatter_df.iterrows():
            axes[1, 0].annotate(row['pattern'],
                               (row['fairness'], row['throughput']),
                               fontsize=8, ha='right')

        axes[1, 0].set_xlabel("Jain's Fairness Index")
        axes[1, 0].set_ylabel('Throughput (kernels/sec)')
        axes[1, 0].set_title('Fairness-Throughput Tradeoff')
        axes[1, 0].grid(True)

        # Scheduler overhead comparison
        overhead_means = [df[df['pattern_name'] == p]['overhead'].mean() for p in patterns]

        axes[1, 1].barh(patterns, overhead_means, color='red', alpha=0.6)
        axes[1, 1].set_xlabel('Scheduler Overhead (%)')
        axes[1, 1].set_ylabel('Load Pattern')
        axes[1, 1].set_title('Scheduler Overhead Across Patterns')
        axes[1, 1].grid(True, axis='x')

        plt.tight_layout()
        plt.savefig(self.figures_dir / "rq6_load_imbalance.png", dpi=300)
        plt.close()

        return {
            'most_fair': pattern_fairness.index[0],
            'most_unfair': pattern_fairness.index[-1],
            'fairness_degradation_pct': degradation
        }

    def analyze_rq7_tail_latency(self) -> Dict:
        """Analyze RQ7: Tail Latency with heterogeneous workloads."""
        print("\n=== RQ7: Tail Latency Analysis ===")

        df = self.load_data("rq7_tail_latency.csv")
        df['streams'] = pd.to_numeric(df['streams'])
        df['p99_p50_ratio'] = df['p99'] / df['p50']

        # Separate homogeneous and heterogeneous workloads
        homog_df = df[df['workload_pattern'] == 'homogeneous_mixed'] if 'workload_pattern' in df.columns else df
        hetero_df = df[df['workload_pattern'] != 'homogeneous_mixed'] if 'workload_pattern' in df.columns else pd.DataFrame()

        stats = homog_df.groupby('streams').agg({
            'p50': ['mean', 'std'],
            'p95': ['mean', 'std'],
            'p99': ['mean', 'std'],
            'p99_p50_ratio': ['mean', 'std']
        }).round(3)

        print("\nHomogeneous Workload Tail Latency Statistics:")
        print(stats)

        if not hetero_df.empty:
            print("\nHeterogeneous Workload Comparison:")
            hetero_stats = hetero_df.groupby('workload_pattern').agg({
                'p50': ['mean', 'std'],
                'p99': ['mean', 'std'],
                'throughput': ['mean', 'std']
            }).round(3)
            print(hetero_stats)

        # Visualization - 3 subplots for heterogeneous analysis
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Plot 1: Homogeneous latency percentiles
        stream_vals = sorted(homog_df['streams'].unique())
        p50_means = [homog_df[homog_df['streams'] == s]['p50'].mean() for s in stream_vals]
        p95_means = [homog_df[homog_df['streams'] == s]['p95'].mean() for s in stream_vals]
        p99_means = [homog_df[homog_df['streams'] == s]['p99'].mean() for s in stream_vals]

        axes[0].plot(stream_vals, p50_means, 'o-', label='P50', linewidth=2)
        axes[0].plot(stream_vals, p95_means, 's-', label='P95', linewidth=2)
        axes[0].plot(stream_vals, p99_means, '^-', label='P99', linewidth=2)
        axes[0].set_xlabel('Stream Count')
        axes[0].set_ylabel('Latency (ms)')
        axes[0].set_title('Latency Percentiles vs Stream Count\n(Homogeneous Mixed Workload)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: P99/P50 ratio
        self._plot_with_error(homog_df, 'streams', 'p99_p50_ratio',
                             axes[1], 'Stream Count', 'P99/P50 Ratio')
        axes[1].set_title('Tail Latency Amplification\n(P99/P50 Ratio)')

        # Plot 3: Heterogeneous workload comparison
        if not hetero_df.empty:
            patterns = hetero_df['workload_pattern'].unique()
            p99_vals = [hetero_df[hetero_df['workload_pattern'] == p]['p99'].mean() for p in patterns]
            colors = ['green' if 'memory' in p else 'red' for p in patterns]

            axes[2].barh(patterns, p99_vals, color=colors, alpha=0.7)
            axes[2].set_xlabel('P99 Latency (ms)')
            axes[2].set_title('Heterogeneous Workload P99 Latency\n(Green=Memory, Red=Compute)')
            axes[2].grid(True, alpha=0.3, axis='x')
        else:
            axes[2].text(0.5, 0.5, 'No heterogeneous\nworkload data',
                        ha='center', va='center', transform=axes[2].transAxes)
            axes[2].set_title('Heterogeneous Workload Analysis')

        plt.tight_layout()
        plt.savefig(self.figures_dir / "rq7_tail_latency.png", dpi=300)
        plt.close()

        return {
            'max_p99_p50_ratio': homog_df['p99_p50_ratio'].max(),
            'has_heterogeneous_data': not hetero_df.empty
        }

