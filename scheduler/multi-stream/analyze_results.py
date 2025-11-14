#!/usr/bin/env python3
"""
GPU Scheduler Experiment Analysis and Visualization

Analyzes results from experiment_driver.py and generates insights.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse
import os
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class ResultAnalyzer:
    """Analyzes experimental results and answers research questions."""

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.figures_dir = self.results_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)

    def load_data(self, filename: str) -> pd.DataFrame:
        """Load experimental data."""
        filepath = self.results_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Results file not found: {filepath}")

        df = pd.read_csv(filepath)

        # Convert numeric columns
        numeric_cols = ['wall_time_ms', 'throughput', 'mean_lat', 'p50', 'p95', 'p99',
                       'concurrent_rate', 'overhead', 'util', 'jains_index',
                       'max_concurrent', 'avg_concurrent', 'inversions']

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    def analyze_rq1_stream_scalability(self) -> Dict:
        """Analyze RQ1: Stream Scalability with workload size variations."""
        print("\n=== RQ1: Stream Scalability Analysis ===")

        df = self.load_data("rq1_stream_scalability.csv")
        df['streams'] = pd.to_numeric(df['streams'])

        # Check if we have workload size variations
        has_workload_variations = 'workload_size_mb' in df.columns or 'workload_elements' in df.columns

        if has_workload_variations:
            # Ensure workload_size_mb exists
            if 'workload_size_mb' not in df.columns and 'workload_elements' in df.columns:
                df['workload_size_mb'] = (pd.to_numeric(df['workload_elements']) * 4) / (1024 * 1024)

            df['workload_size_mb'] = pd.to_numeric(df['workload_size_mb'])

            # Group by both stream count and workload size
            print("\nStream Count vs Workload Size Analysis:")
            for size_mb in sorted(df['workload_size_mb'].unique()):
                size_df = df[df['workload_size_mb'] == size_mb]
                stats = size_df.groupby('streams').agg({
                    'concurrent_rate': 'mean',
                    'throughput': 'mean',
                    'max_concurrent': 'mean',
                }).round(2)
                print(f"\nWorkload: {size_mb:.2f} MB")
                print(stats.head())

            # Find optimal configuration
            best_config = df.loc[df['throughput'].idxmax()]
            print(f"\n✓ Best configuration: {int(best_config['streams'])} streams, "
                  f"{best_config['workload_size_mb']:.2f} MB workload, "
                  f"throughput: {best_config['throughput']:.2f} kernels/sec")

            # Visualization with multiple lines per workload
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))

            # Plot each workload size as a separate line
            workload_sizes = sorted(df['workload_size_mb'].unique())
            colors = plt.cm.viridis(np.linspace(0, 1, len(workload_sizes)))

            # Get grid sizes for each workload (for legend)
            workload_grid_map = {}
            for size_mb in workload_sizes:
                size_df = df[df['workload_size_mb'] == size_mb]
                # Get the grid size (should be same for all streams with same workload)
                grid_size = int(size_df['grid_size'].iloc[0])
                workload_grid_map[size_mb] = grid_size

            # Plot 1: Concurrent execution rate
            for i, size_mb in enumerate(workload_sizes):
                size_df = df[df['workload_size_mb'] == size_mb]
                grouped = size_df.groupby('streams').agg({
                    'concurrent_rate': ['mean', 'std']
                }).reset_index()
                grid_size = workload_grid_map[size_mb]
                axes[0, 0].errorbar(grouped['streams'], grouped['concurrent_rate']['mean'],
                                   yerr=grouped['concurrent_rate']['std'],
                                   fmt='o-', linewidth=2, markersize=6,
                                   label=f'{size_mb:.2f} MB ({grid_size} blocks)', color=colors[i],
                                   capsize=4, alpha=0.8)
            axes[0, 0].set_xlabel('Stream Count')
            axes[0, 0].set_ylabel('Concurrent Execution Rate (%)')
            axes[0, 0].set_title('Concurrent Execution Rate vs Stream Count')
            axes[0, 0].legend(title='Workload Size', fontsize=8)
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_xscale('log', base=2)

            # Plot 2: Throughput
            for i, size_mb in enumerate(workload_sizes):
                size_df = df[df['workload_size_mb'] == size_mb]
                grouped = size_df.groupby('streams').agg({
                    'throughput': ['mean', 'std']
                }).reset_index()
                grid_size = workload_grid_map[size_mb]
                axes[0, 1].errorbar(grouped['streams'], grouped['throughput']['mean'],
                                   yerr=grouped['throughput']['std'],
                                   fmt='s-', linewidth=2, markersize=6,
                                   label=f'{size_mb:.2f} MB ({grid_size} blocks)', color=colors[i],
                                   capsize=4, alpha=0.8)
            axes[0, 1].set_xlabel('Stream Count')
            axes[0, 1].set_ylabel('Throughput (kernels/sec)')
            axes[0, 1].set_title('Throughput vs Stream Count')
            axes[0, 1].legend(title='Workload Size', fontsize=8)
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_xscale('log', base=2)

            # Plot 3: Max concurrent kernels
            for i, size_mb in enumerate(workload_sizes):
                size_df = df[df['workload_size_mb'] == size_mb]
                grouped = size_df.groupby('streams').agg({
                    'max_concurrent': ['mean', 'std']
                }).reset_index()
                grid_size = workload_grid_map[size_mb]
                axes[1, 0].errorbar(grouped['streams'], grouped['max_concurrent']['mean'],
                                   yerr=grouped['max_concurrent']['std'],
                                   fmt='^-', linewidth=2, markersize=6,
                                   label=f'{size_mb:.2f} MB ({grid_size} blocks)', color=colors[i],
                                   capsize=4, alpha=0.8)
            axes[1, 0].set_xlabel('Stream Count')
            axes[1, 0].set_ylabel('Max Concurrent Kernels')
            axes[1, 0].set_title('Max Concurrent Kernels vs Stream Count')
            axes[1, 0].legend(title='Workload Size', fontsize=8)
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_xscale('log', base=2)

            # Plot 4: GPU utilization
            for i, size_mb in enumerate(workload_sizes):
                size_df = df[df['workload_size_mb'] == size_mb]
                grouped = size_df.groupby('streams').agg({
                    'util': ['mean', 'std']
                }).reset_index()
                grid_size = workload_grid_map[size_mb]
                axes[1, 1].errorbar(grouped['streams'], grouped['util']['mean'],
                                   yerr=grouped['util']['std'],
                                   fmt='d-', linewidth=2, markersize=6,
                                   label=f'{size_mb:.2f} MB ({grid_size} blocks)', color=colors[i],
                                   capsize=4, alpha=0.8)
            axes[1, 1].set_xlabel('Stream Count')
            axes[1, 1].set_ylabel('GPU Utilization (%)')
            axes[1, 1].set_title('GPU Utilization vs Stream Count')
            axes[1, 1].legend(title='Workload Size', fontsize=8)
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_xscale('log', base=2)

            plt.tight_layout()
            plt.savefig(self.figures_dir / "rq1_stream_scalability.png", dpi=300)
            plt.close()

            return {
                "best_streams": int(best_config['streams']),
                "best_workload_mb": float(best_config['workload_size_mb']),
                "best_throughput": float(best_config['throughput']),
                "workload_variations": len(workload_sizes)
            }

        else:
            # Legacy code for single workload size
            stats = df.groupby('streams').agg({
                'concurrent_rate': ['mean', 'std'],
                'throughput': ['mean', 'std'],
                'max_concurrent': ['mean', 'std'],
                'util': ['mean', 'std']
            }).round(2)

            print("\nStream Count vs Metrics:")
            print(stats)

            optimal_streams = df.groupby('streams')['throughput'].mean().idxmax()
            optimal_throughput = df.groupby('streams')['throughput'].mean().max()

            print(f"\n✓ Optimal stream count: {optimal_streams} (throughput: {optimal_throughput:.2f} kernels/sec)")

            saturation_point = self._find_saturation_point(
                df.groupby('streams')['throughput'].mean()
            )
            print(f"✓ Saturation point: ~{saturation_point} streams")

            # Original single-line visualization
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            self._plot_with_error(df, 'streams', 'concurrent_rate',
                                 axes[0, 0], 'Stream Count', 'Concurrent Execution Rate (%)')
            self._plot_with_error(df, 'streams', 'throughput',
                                 axes[0, 1], 'Stream Count', 'Throughput (kernels/sec)')
            self._plot_with_error(df, 'streams', 'max_concurrent',
                                 axes[1, 0], 'Stream Count', 'Max Concurrent Kernels')
            self._plot_with_error(df, 'streams', 'util',
                                 axes[1, 1], 'Stream Count', 'GPU Utilization (%)')

            plt.tight_layout()
            plt.savefig(self.figures_dir / "rq1_stream_scalability.png", dpi=300)
            plt.close()

            return {
                "optimal_streams": int(optimal_streams),
                "optimal_throughput": float(optimal_throughput),
                "saturation_point": int(saturation_point)
            }

    def analyze_rq2_workload_characterization(self) -> Dict:
        """Analyze RQ2: Workload Characterization with varying stream counts."""
        print("\n=== RQ2: Workload Characterization Analysis ===")

        df = self.load_data("rq2_workload_characterization.csv")
        df['streams'] = pd.to_numeric(df['streams'])

        # Check if we have stream count variations
        has_stream_variations = len(df['streams'].unique()) > 1

        if has_stream_variations:
            # Multi-line plot analysis
            print("\nWorkload Type vs Stream Count Analysis:")

            kernel_types = sorted(df['type'].unique())
            colors = plt.cm.tab10(np.linspace(0, 1, len(kernel_types)))

            # Print statistics for each workload type
            for ktype in kernel_types:
                type_df = df[df['type'] == ktype]
                print(f"\n{ktype.upper()} workload:")
                stats = type_df.groupby('streams').agg({
                    'concurrent_rate': 'mean',
                    'throughput': 'mean',
                    'max_concurrent': 'mean',
                }).round(2)
                print(stats.head())

            # Find best configuration
            best_config = df.loc[df['concurrent_rate'].idxmax()]
            print(f"\n✓ Best concurrency: {best_config['type']} with {int(best_config['streams'])} streams")
            print(f"  concurrent_rate = {best_config['concurrent_rate']:.1f}%")
            print(f"  max_concurrent = {best_config['max_concurrent']:.1f}")

            # Visualization with multiple lines per workload type
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))

            # Plot 1: Concurrent execution rate
            for i, ktype in enumerate(kernel_types):
                type_df = df[df['type'] == ktype]
                grouped = type_df.groupby('streams').agg({
                    'concurrent_rate': ['mean', 'std']
                }).reset_index()
                axes[0, 0].errorbar(grouped['streams'], grouped['concurrent_rate']['mean'],
                                   yerr=grouped['concurrent_rate']['std'],
                                   fmt='o-', linewidth=2, markersize=6,
                                   label=ktype.upper(), color=colors[i],
                                   capsize=4, alpha=0.8)
            axes[0, 0].set_xlabel('Stream Count')
            axes[0, 0].set_ylabel('Concurrent Execution Rate (%)')
            axes[0, 0].set_title('Concurrent Execution Rate vs Stream Count')
            axes[0, 0].legend(title='Workload Type', fontsize=10)
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_xscale('log', base=2)

            # Plot 2: Throughput
            for i, ktype in enumerate(kernel_types):
                type_df = df[df['type'] == ktype]
                grouped = type_df.groupby('streams').agg({
                    'throughput': ['mean', 'std']
                }).reset_index()
                axes[0, 1].errorbar(grouped['streams'], grouped['throughput']['mean'],
                                   yerr=grouped['throughput']['std'],
                                   fmt='s-', linewidth=2, markersize=6,
                                   label=ktype.upper(), color=colors[i],
                                   capsize=4, alpha=0.8)
            axes[0, 1].set_xlabel('Stream Count')
            axes[0, 1].set_ylabel('Throughput (kernels/sec)')
            axes[0, 1].set_title('Throughput vs Stream Count')
            axes[0, 1].legend(title='Workload Type', fontsize=10)
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_xscale('log', base=2)

            # Plot 3: Max concurrent kernels
            for i, ktype in enumerate(kernel_types):
                type_df = df[df['type'] == ktype]
                grouped = type_df.groupby('streams').agg({
                    'max_concurrent': ['mean', 'std']
                }).reset_index()
                axes[1, 0].errorbar(grouped['streams'], grouped['max_concurrent']['mean'],
                                   yerr=grouped['max_concurrent']['std'],
                                   fmt='^-', linewidth=2, markersize=6,
                                   label=ktype.upper(), color=colors[i],
                                   capsize=4, alpha=0.8)
            axes[1, 0].set_xlabel('Stream Count')
            axes[1, 0].set_ylabel('Max Concurrent Kernels')
            axes[1, 0].set_title('Max Concurrent Kernels vs Stream Count')
            axes[1, 0].legend(title='Workload Type', fontsize=10)
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_xscale('log', base=2)

            # Plot 4: Mean latency
            for i, ktype in enumerate(kernel_types):
                type_df = df[df['type'] == ktype]
                grouped = type_df.groupby('streams').agg({
                    'mean_lat': ['mean', 'std']
                }).reset_index()
                axes[1, 1].errorbar(grouped['streams'], grouped['mean_lat']['mean'],
                                   yerr=grouped['mean_lat']['std'],
                                   fmt='d-', linewidth=2, markersize=6,
                                   label=ktype.upper(), color=colors[i],
                                   capsize=4, alpha=0.8)
            axes[1, 1].set_xlabel('Stream Count')
            axes[1, 1].set_ylabel('Mean Latency (ms)')
            axes[1, 1].set_title('Mean Latency vs Stream Count')
            axes[1, 1].legend(title='Workload Type', fontsize=10)
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_xscale('log', base=2)

            plt.tight_layout()
            plt.savefig(self.figures_dir / "rq2_workload_characterization.png", dpi=300)
            plt.close()

            return {
                "best_concurrent_workload": best_config['type'],
                "best_concurrent_streams": int(best_config['streams']),
                "best_throughput_workload": df.loc[df['throughput'].idxmax()]['type']
            }

        else:
            # Legacy single-stream-count analysis
            stats = df.groupby('type').agg({
                'concurrent_rate': ['mean', 'std'],
                'throughput': ['mean', 'std'],
                'mean_lat': ['mean', 'std'],
                'util': ['mean', 'std']
            }).round(2)

            print("\nWorkload Type vs Metrics:")
            print(stats)

            best_concurrent = df.groupby('type')['concurrent_rate'].mean().idxmax()
            best_throughput = df.groupby('type')['throughput'].mean().idxmax()

            print(f"\n✓ Best concurrency: {best_concurrent}")
            print(f"✓ Best throughput: {best_throughput}")

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            workload_order = ['compute', 'memory', 'mixed', 'gemm']

            sns.boxplot(data=df, x='type', y='concurrent_rate', order=workload_order, ax=axes[0, 0])
            axes[0, 0].set_title('Concurrent Execution Rate by Workload')
            axes[0, 0].set_ylabel('Concurrent Rate (%)')

            sns.boxplot(data=df, x='type', y='throughput', order=workload_order, ax=axes[0, 1])
            axes[0, 1].set_title('Throughput by Workload')
            axes[0, 1].set_ylabel('Throughput (kernels/sec)')

            sns.boxplot(data=df, x='type', y='mean_lat', order=workload_order, ax=axes[1, 0])
            axes[1, 0].set_title('Mean Latency by Workload')
            axes[1, 0].set_ylabel('Latency (ms)')

            sns.boxplot(data=df, x='type', y='util', order=workload_order, ax=axes[1, 1])
            axes[1, 1].set_title('GPU Utilization by Workload')
            axes[1, 1].set_ylabel('Utilization (%)')

            plt.tight_layout()
            plt.savefig(self.figures_dir / "rq2_workload_characterization.png", dpi=300)
            plt.close()

            return {
                "best_concurrent_workload": best_concurrent,
                "best_throughput_workload": best_throughput
            }

    def analyze_rq3_priority_effectiveness(self) -> Dict:
        """Analyze RQ3: CUDA Priority Mechanism Analysis (4 sub-questions)."""
        print("\n=== RQ3: CUDA Priority Mechanism Analysis ===")

        df = self.load_data("rq3_priority_effectiveness.csv")

        # Check if we have priority_enabled column
        if 'priority_enabled' not in df.columns:
            print("\nWarning: priority_enabled column not found, using inversions as proxy")
            df['priority_enabled'] = df['inversions'] > 0

        # Separate priority and non-priority runs
        no_pri_df = df[df['priority_enabled'] == 0]
        with_pri_df = df[df['priority_enabled'] == 1]

        print("\n--- Performance Comparison (Priority vs No Priority) ---")

        # Compare metrics across stream counts
        for streams in sorted(df['streams'].unique()):
            no_pri = no_pri_df[no_pri_df['streams'] == streams]
            with_pri = with_pri_df[with_pri_df['streams'] == streams]

            if not no_pri.empty and not with_pri.empty:
                no_pri_tput = no_pri['throughput'].mean()
                with_pri_tput = with_pri['throughput'].mean()
                no_pri_lat = no_pri['mean_lat'].mean()
                with_pri_lat = with_pri['mean_lat'].mean()
                no_pri_max_conc = no_pri['max_concurrent'].mean()
                with_pri_max_conc = with_pri['max_concurrent'].mean()
                no_pri_inv = no_pri['inversions'].mean()
                with_pri_inv = with_pri['inversions'].mean()

                tput_diff = ((with_pri_tput - no_pri_tput) / no_pri_tput) * 100
                lat_diff = ((with_pri_lat - no_pri_lat) / no_pri_lat) * 100

                print(f"\n{streams} streams:")
                print(f"  Throughput: {no_pri_tput:.2f} → {with_pri_tput:.2f} kernels/sec ({tput_diff:+.2f}%)")
                print(f"  Latency: {no_pri_lat:.2f} → {with_pri_lat:.2f} ms ({lat_diff:+.2f}%)")
                print(f"  Max Concurrent: {no_pri_max_conc:.2f} → {with_pri_max_conc:.2f}")
                print(f"  Priority Inversions: {no_pri_inv:.0f} → {with_pri_inv:.0f}")

        # Overall statistics
        print("\n--- Overall Statistics ---")
        if not no_pri_df.empty and not with_pri_df.empty:
            avg_tput_diff = ((with_pri_df['throughput'].mean() - no_pri_df['throughput'].mean())
                            / no_pri_df['throughput'].mean()) * 100
            avg_lat_diff = ((with_pri_df['mean_lat'].mean() - no_pri_df['mean_lat'].mean())
                           / no_pri_df['mean_lat'].mean()) * 100

            print(f"Average throughput difference: {avg_tput_diff:+.2f}%")
            print(f"Average latency difference: {avg_lat_diff:+.2f}%")
            print(f"Average inversions (no priority): {no_pri_df['inversions'].mean():.1f}")
            print(f"Average inversions (with priority): {with_pri_df['inversions'].mean():.1f}")

        # Try to load RQ10 data for RQ3.3 (preemption test)
        try:
            df_preempt = self.load_data("rq10_preemption_latency.csv")
            has_preempt_data = True
        except FileNotFoundError:
            print("\n⚠ RQ10 data not found, RQ3.3 (preemption test) will be skipped")
            has_preempt_data = False

        # Visualization - 2×2 grid (4 subplots)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        stream_counts = sorted(df['streams'].unique())
        x = np.arange(len(stream_counts))
        width = 0.35

        # RQ3.1: Queue ordering (inversions)
        no_pri_invs = [no_pri_df[no_pri_df['streams'] == s]['inversions'].mean() for s in stream_counts]
        with_pri_invs = [with_pri_df[with_pri_df['streams'] == s]['inversions'].mean() for s in stream_counts]

        axes[0, 0].bar(x - width/2, no_pri_invs, width, label='No Priority', alpha=0.7, color='blue', edgecolor='black', linewidth=1.5)
        axes[0, 0].bar(x + width/2, with_pri_invs, width, label='With Priority', alpha=0.7, color='orange', edgecolor='black', linewidth=1.5)
        axes[0, 0].set_xlabel('Stream Count', fontsize=10)
        axes[0, 0].set_ylabel('Priority Inversions Detected', fontsize=10)
        axes[0, 0].set_title('RQ3.1: Does priority affect queue ordering?', fontsize=11, fontweight='bold')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(stream_counts, fontsize=10)
        axes[0, 0].legend(fontsize=9)
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        axes[0, 0].set_ylim(bottom=0)  # Start Y-axis from 0 to show zero bars clearly

        # Add value labels on bars
        for i, (no_pri, with_pri) in enumerate(zip(no_pri_invs, with_pri_invs)):
            axes[0, 0].text(i - width/2, no_pri + 0.5, f'{no_pri:.0f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
            axes[0, 0].text(i + width/2, with_pri + 0.5, f'{with_pri:.0f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')

        # RQ3.2: Performance impact (throughput)
        no_pri_tputs = [no_pri_df[no_pri_df['streams'] == s]['throughput'].mean() for s in stream_counts]
        with_pri_tputs = [with_pri_df[with_pri_df['streams'] == s]['throughput'].mean() for s in stream_counts]

        axes[0, 1].bar(x - width/2, no_pri_tputs, width, label='No Priority', alpha=0.7, color='blue')
        axes[0, 1].bar(x + width/2, with_pri_tputs, width, label='With Priority', alpha=0.7, color='orange')
        axes[0, 1].set_xlabel('Stream Count', fontsize=10)
        axes[0, 1].set_ylabel('Throughput (kernels/sec)', fontsize=10)
        axes[0, 1].set_title('RQ3.2: Does priority affect execution performance?', fontsize=11, fontweight='bold')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(stream_counts, fontsize=10)
        axes[0, 1].legend(fontsize=9)
        axes[0, 1].grid(True, alpha=0.3, axis='y')

        # Add percentage difference annotations
        for i, (no_pri, with_pri) in enumerate(zip(no_pri_tputs, with_pri_tputs)):
            if not pd.isna(no_pri) and not pd.isna(with_pri) and no_pri > 0:
                diff_pct = ((with_pri - no_pri) / no_pri) * 100
                axes[0, 1].text(i, max(no_pri, with_pri) * 1.02, f'{diff_pct:+.1f}%',
                           ha='center', fontsize=9, fontweight='bold',
                           color='red' if abs(diff_pct) > 1 else 'green')

        # RQ3.3: Preemption capability (from RQ10 data)
        if has_preempt_data:
            baseline = df_preempt[df_preempt['scenario'] == 'baseline_no_contention']
            baseline_p99 = baseline['p99'].astype(float).mean() if not baseline.empty else 0

            blocking_types = ['compute', 'gemm']
            preempt_data = []

            for blocking_type in blocking_types:
                no_pri = df_preempt[df_preempt['scenario'] == f'contention_{blocking_type}_no_priority']
                with_pri = df_preempt[df_preempt['scenario'] == f'contention_{blocking_type}_with_priority']

                if not no_pri.empty and not with_pri.empty:
                    no_pri_p99 = no_pri['p99'].astype(float).mean()
                    with_pri_p99 = with_pri['p99'].astype(float).mean()
                    benefit = no_pri_p99 / with_pri_p99 if with_pri_p99 > 0 else 1.0

                    preempt_data.append({
                        'type': blocking_type.upper(),
                        'baseline': baseline_p99,
                        'no_pri': no_pri_p99,
                        'with_pri': with_pri_p99,
                        'benefit': benefit
                    })

            if preempt_data:
                x_preempt = np.arange(len(preempt_data))
                width_preempt = 0.25

                baseline_vals = [d['baseline'] for d in preempt_data]
                no_pri_vals = [d['no_pri'] for d in preempt_data]
                with_pri_vals = [d['with_pri'] for d in preempt_data]

                axes[1, 0].bar(x_preempt - width_preempt, baseline_vals, width_preempt,
                           label='Baseline (no contention)', alpha=0.7, color='green')
                axes[1, 0].bar(x_preempt, no_pri_vals, width_preempt,
                           label='Contention (no priority)', alpha=0.7, color='red')
                axes[1, 0].bar(x_preempt + width_preempt, with_pri_vals, width_preempt,
                           label='Contention (with priority)', alpha=0.7, color='orange')

                axes[1, 0].set_ylabel('Fast Kernel P99 Latency (ms)', fontsize=10)
                axes[1, 0].set_xlabel('Blocking Kernel Type', fontsize=10)
                axes[1, 0].set_title('RQ3.3: Can priority preempt running kernels?', fontsize=11, fontweight='bold')
                axes[1, 0].set_xticks(x_preempt)
                axes[1, 0].set_xticklabels([d['type'] for d in preempt_data], fontsize=10)
                axes[1, 0].legend(fontsize=9, loc='upper left')
                axes[1, 0].grid(True, alpha=0.3, axis='y')
                axes[1, 0].set_yscale('log')

                # Add benefit annotations
                for i, d in enumerate(preempt_data):
                    axes[1, 0].text(i, max(d['no_pri'], d['with_pri']) * 1.2,
                               f'Benefit: {d["benefit"]:.2f}×',
                               ha='center', fontsize=8,
                               color='red' if d['benefit'] < 1.1 else 'green')
        else:
            axes[1, 0].text(0.5, 0.5, 'RQ3.3: Preemption test data not available\n(Run RQ10 experiment)',
                       ha='center', va='center', transform=axes[1, 0].transAxes, fontsize=10)
            axes[1, 0].set_title('RQ3.3: Can priority preempt running kernels?', fontsize=11, fontweight='bold')

        # RQ3.4: Per-priority-class latency
        # Load detailed per-kernel data from CSV files
        detailed_files = [f for f in os.listdir(self.results_dir) if f.startswith('detailed_kernels_') and f.endswith('.csv')]

        if detailed_files:
            priority_latencies = {'high': [], 'low': []}

            for detail_file in detailed_files:
                try:
                    detail_df = pd.read_csv(os.path.join(self.results_dir, detail_file))
                    # High priority: -5, -4; Low priority: -2, 0
                    high_pri = detail_df[detail_df['priority'].isin([-5, -4])]['duration_ms']
                    low_pri = detail_df[detail_df['priority'].isin([-2, 0])]['duration_ms']

                    if not high_pri.empty:
                        priority_latencies['high'].extend(high_pri.tolist())
                    if not low_pri.empty:
                        priority_latencies['low'].extend(low_pri.tolist())
                except Exception as e:
                    print(f"Warning: Failed to load {detail_file}: {e}")

            if priority_latencies['high'] and priority_latencies['low']:
                # Calculate average latencies
                avg_high = np.mean(priority_latencies['high'])
                avg_low = np.mean(priority_latencies['low'])
                std_high = np.std(priority_latencies['high'])
                std_low = np.std(priority_latencies['low'])

                # Bar chart
                x_pri = np.arange(2)
                width = 0.6

                bars = axes[1, 1].bar(x_pri, [avg_high, avg_low], width,
                                     color=['#ff7f0e', '#1f77b4'],
                                     alpha=0.7, edgecolor='black', linewidth=1.5,
                                     yerr=[std_high, std_low], capsize=5)

                axes[1, 1].set_ylabel('Average Kernel Duration (ms)', fontsize=10)
                axes[1, 1].set_xlabel('Priority Class', fontsize=10)
                axes[1, 1].set_title('RQ3.4: Do high-priority kernels achieve lower latency?', fontsize=11, fontweight='bold')
                axes[1, 1].set_xticks(x_pri)
                axes[1, 1].set_xticklabels(['High Priority\n(-5, -4)', 'Low Priority\n(-2, 0)'], fontsize=10)
                axes[1, 1].grid(True, alpha=0.3, axis='y')
                axes[1, 1].set_ylim(bottom=0)

                # Add value labels
                axes[1, 1].text(0, avg_high + std_high + 0.05, f'{avg_high:.2f}ms',
                           ha='center', va='bottom', fontsize=9, fontweight='bold')
                axes[1, 1].text(1, avg_low + std_low + 0.05, f'{avg_low:.2f}ms',
                           ha='center', va='bottom', fontsize=9, fontweight='bold')

                # Add ratio annotation
                ratio = avg_high / avg_low if avg_low > 0 else 1.0
                axes[1, 1].text(0.5, max(avg_high, avg_low) + max(std_high, std_low) + 0.15,
                           f'Ratio: {ratio:.2f}×',
                           ha='center', fontsize=9,
                           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
            else:
                axes[1, 1].text(0.5, 0.5, 'RQ3.4: Insufficient data',
                           ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=10)
                axes[1, 1].set_title('RQ3.4: Do high-priority kernels achieve lower latency?', fontsize=11, fontweight='bold')
        else:
            axes[1, 1].text(0.5, 0.5, 'RQ3.4: No detailed data files found',
                       ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=10)
            axes[1, 1].set_title('RQ3.4: Do high-priority kernels achieve lower latency?', fontsize=11, fontweight='bold')
            axes[1, 1].set_xlabel('Priority Class', fontsize=10)
            axes[1, 1].set_ylabel('Average Latency (ms)', fontsize=10)

        plt.tight_layout()
        plt.savefig(self.figures_dir / "rq3_priority_effectiveness.png", dpi=300)
        plt.close()

        return {
            "avg_throughput_diff_pct": avg_tput_diff if not no_pri_df.empty and not with_pri_df.empty else 0,
            "avg_latency_diff_pct": avg_lat_diff if not no_pri_df.empty and not with_pri_df.empty else 0,
            "priority_affects_queue_only": True,
            "has_preempt_data": has_preempt_data
        }

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

    def analyze_rq9_priority_tail_latency(self) -> Dict:
        """Analyze RQ9: Priority-based tail latency (XSched comparison) with heterogeneous workloads."""
        print("\n=== RQ9: Priority-Based Tail Latency Analysis ===")

        df = self.load_data("rq9_priority_tail_latency.csv")

        # Separate homogeneous and heterogeneous workloads
        homog_df = df[df['workload_type'] == 'homogeneous'] if 'workload_type' in df.columns else df
        hetero_df = df[df['workload_type'] == 'heterogeneous'] if 'workload_type' in df.columns else pd.DataFrame()

        # Compare priority vs baseline for homogeneous
        baseline_df = homog_df[~homog_df['priority_enabled']] if not homog_df.empty else pd.DataFrame()
        priority_df = homog_df[homog_df['priority_enabled']] if not homog_df.empty else pd.DataFrame()

        print("\n--- Homogeneous Workload: Load Imbalance Analysis ---")
        if not baseline_df.empty and not priority_df.empty:
            for label in homog_df['config_label'].unique():
                if 'baseline' in label or 'priority' in label:
                    continue
                baseline_label = label.replace('priority', 'baseline')
                priority_label = label.replace('baseline', 'priority')

                baseline_p99 = homog_df[homog_df['config_label'] == baseline_label]['p99'].mean()
                priority_p99 = homog_df[homog_df['config_label'] == priority_label]['p99'].mean()

                if not pd.isna(baseline_p99) and not pd.isna(priority_p99) and priority_p99 > 0:
                    improvement = baseline_p99 / priority_p99
                    print(f"  {label}: Baseline P99={baseline_p99:.3f}ms, "
                          f"Priority P99={priority_p99:.3f}ms, Improvement={improvement:.2f}×")

        print("\n--- Heterogeneous Workload: Memory vs Compute Analysis ---")
        if not hetero_df.empty:
            for label in sorted(hetero_df['config_label'].unique()):
                p99_mean = hetero_df[hetero_df['config_label'] == label]['p99'].mean()
                throughput_mean = hetero_df[hetero_df['config_label'] == label]['throughput'].mean()
                print(f"  {label}: P99={p99_mean:.3f}ms, Throughput={throughput_mean:.1f} kernels/sec")

        # Visualization - 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Plot 1: Homogeneous P99 comparison (baseline vs priority)
        if not baseline_df.empty and not priority_df.empty:
            # Extract unique load patterns (extract the last part after 'baseline_' or 'priority_')
            all_labels = homog_df['config_label'].unique()
            label_patterns = set()
            for label in all_labels:
                if 'baseline_' in label:
                    pattern = label.split('baseline_')[1]
                    label_patterns.add(pattern)
                elif 'priority_' in label:
                    pattern = label.split('priority_')[1]
                    label_patterns.add(pattern)

            labels = sorted(list(label_patterns))
            baseline_p99s = []
            priority_p99s = []

            for label_part in labels:
                baseline_label = f'homog_baseline_{label_part}'
                priority_label = f'homog_priority_{label_part}'

                baseline_p99 = homog_df[homog_df['config_label'] == baseline_label]['p99'].astype(float).mean()
                priority_p99 = homog_df[homog_df['config_label'] == priority_label]['p99'].astype(float).mean()

                if not pd.isna(baseline_p99):
                    baseline_p99s.append(baseline_p99)
                else:
                    baseline_p99s.append(0)

                if not pd.isna(priority_p99):
                    priority_p99s.append(priority_p99)
                else:
                    priority_p99s.append(0)

            if len(baseline_p99s) > 0 and len(priority_p99s) > 0:
                x = np.arange(len(labels))
                width = 0.35

                axes[0].bar(x - width/2, baseline_p99s, width, label='Baseline (No Priority)', alpha=0.7, color='red')
                axes[0].bar(x + width/2, priority_p99s, width, label='Priority Enabled', alpha=0.7, color='green')
                axes[0].set_xlabel('Load Configuration')
                axes[0].set_ylabel('P99 Latency (ms)')
                axes[0].set_title('Homogeneous: Priority vs Baseline\n(Load Imbalance Patterns)')
                axes[0].set_xticks(x)
                axes[0].set_xticklabels(labels, rotation=15, ha='right')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3, axis='y')
            else:
                axes[0].text(0.5, 0.5, 'No matching\nbaseline/priority pairs',
                            ha='center', va='center', transform=axes[0].transAxes)
        else:
            axes[0].text(0.5, 0.5, 'No homogeneous\nworkload data',
                        ha='center', va='center', transform=axes[0].transAxes)

        # Plot 2: Heterogeneous workload P99 comparison
        if not hetero_df.empty:
            hetero_labels = sorted(hetero_df['config_label'].unique())
            hetero_p99s = [hetero_df[hetero_df['config_label'] == l]['p99'].mean() for l in hetero_labels]
            colors = ['green' if 'memory' in l else 'red' if 'compute' in l else 'blue' for l in hetero_labels]

            axes[1].barh(range(len(hetero_labels)), hetero_p99s, color=colors, alpha=0.7)
            axes[1].set_yticks(range(len(hetero_labels)))
            axes[1].set_yticklabels([l.replace('hetero_', '').replace('_', ' ') for l in hetero_labels], fontsize=8)
            axes[1].set_xlabel('P99 Latency (ms)')
            axes[1].set_title('Heterogeneous: Memory vs Compute\n(Green=Memory, Red=Compute)')
            axes[1].grid(True, alpha=0.3, axis='x')
        else:
            axes[1].text(0.5, 0.5, 'No heterogeneous\nworkload data',
                        ha='center', va='center', transform=axes[1].transAxes)

        # Plot 3: Priority improvement factor
        if not baseline_df.empty and not priority_df.empty and 'labels' in locals():
            improvements = []
            improvement_labels = []

            for label_part in labels:
                baseline_label = f'homog_baseline_{label_part}'
                priority_label = f'homog_priority_{label_part}'

                baseline_p99 = homog_df[homog_df['config_label'] == baseline_label]['p99'].astype(float).mean()
                priority_p99 = homog_df[homog_df['config_label'] == priority_label]['p99'].astype(float).mean()

                if not pd.isna(baseline_p99) and not pd.isna(priority_p99) and priority_p99 > 0:
                    improvement = baseline_p99 / priority_p99
                    improvements.append(improvement)
                    improvement_labels.append(label_part)

            if improvements:
                axes[2].bar(range(len(improvements)), improvements, color='green', alpha=0.7)
                axes[2].axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='No improvement')
                axes[2].axhline(y=2.0, color='orange', linestyle='--', linewidth=2, label='XSched target (2×)')
                axes[2].set_xticks(range(len(improvements)))
                axes[2].set_xticklabels(improvement_labels, rotation=15, ha='right')
                axes[2].set_ylabel('P99 Improvement Factor (×)')
                axes[2].set_xlabel('Load Configuration')
                axes[2].set_title('Priority Effectiveness\n(Higher is Better)')
                axes[2].legend()
                axes[2].grid(True, alpha=0.3, axis='y')

                max_improvement = max(improvements)
                avg_improvement = np.mean(improvements)
            else:
                axes[2].text(0.5, 0.5, 'No improvement\ndata available',
                            ha='center', va='center', transform=axes[2].transAxes)
                max_improvement = 1.0
                avg_improvement = 1.0
        else:
            axes[2].text(0.5, 0.5, 'No priority\ncomparison data',
                        ha='center', va='center', transform=axes[2].transAxes)
            max_improvement = 1.0
            avg_improvement = 1.0

        plt.tight_layout()
        plt.savefig(self.figures_dir / "rq9_priority_tail_latency.png", dpi=300)
        plt.close()

        return {
            'max_improvement': max_improvement,
            'avg_improvement': avg_improvement,
            'has_heterogeneous_data': not hetero_df.empty
        }

    # RQ10 has been merged into RQ3.3
    # Keeping this function for reference, but it's no longer called
    def analyze_rq10_preemption_latency_DEPRECATED(self) -> Dict:
        """Analyze RQ10: Preemption latency estimation via contention analysis."""
        print("\n=== RQ10: Preemption Latency Analysis ===")
        print("Method: Measure fast kernel latency inflation under contention")

        df = self.load_data("rq10_preemption_latency.csv")

        # Extract baseline and contention scenarios
        baseline = df[df['scenario'] == 'baseline_no_contention']
        baseline_p99 = baseline['p99'].astype(float).mean() if not baseline.empty else 0

        print(f"\nBaseline (no contention): P99 = {baseline_p99:.3f} ms")

        # Analyze each blocking kernel type
        blocking_types = ['compute', 'gemm']
        results = []

        for blocking_type in blocking_types:
            no_pri = df[df['scenario'] == f'contention_{blocking_type}_no_priority']
            with_pri = df[df['scenario'] == f'contention_{blocking_type}_with_priority']

            if not no_pri.empty:
                no_pri_p99 = no_pri['p99'].astype(float).mean()
                inflation_no_pri = no_pri_p99 / baseline_p99 if baseline_p99 > 0 else 0

                print(f"\n{blocking_type.upper()} Blocking Kernel:")
                print(f"  Without Priority: P99 = {no_pri_p99:.3f} ms (inflation: {inflation_no_pri:.1f}×)")

                if not with_pri.empty:
                    with_pri_p99 = with_pri['p99'].astype(float).mean()
                    inflation_with_pri = with_pri_p99 / baseline_p99 if baseline_p99 > 0 else 0
                    benefit = no_pri_p99 / with_pri_p99 if with_pri_p99 > 0 else 1.0

                    print(f"  With Priority: P99 = {with_pri_p99:.3f} ms (inflation: {inflation_with_pri:.1f}×)")
                    print(f"  Priority Benefit: {benefit:.2f}× reduction")
                    print(f"  Estimated Preemption Latency: {no_pri_p99:.3f} ms")

                    results.append({
                        'blocking_type': blocking_type,
                        'baseline_p99': baseline_p99,
                        'no_pri_p99': no_pri_p99,
                        'with_pri_p99': with_pri_p99,
                        'inflation_no_pri': inflation_no_pri,
                        'inflation_with_pri': inflation_with_pri,
                        'priority_benefit': benefit
                    })

        # Visualization - 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        if results:
            result_df = pd.DataFrame(results)

            # Plot 1: P99 Latency Comparison
            x = np.arange(len(result_df))
            width = 0.25

            axes[0].bar(x - width, [baseline_p99] * len(result_df), width,
                       label='Baseline (no contention)', alpha=0.7, color='green')
            axes[0].bar(x, result_df['no_pri_p99'], width,
                       label='With Contention (no priority)', alpha=0.7, color='red')
            axes[0].bar(x + width, result_df['with_pri_p99'], width,
                       label='With Contention (priority)', alpha=0.7, color='orange')

            axes[0].set_ylabel('P99 Latency (ms)')
            axes[0].set_xlabel('Blocking Kernel Type')
            axes[0].set_title('P99 Latency: Baseline vs Contention\n(Lower is Better)')
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(result_df['blocking_type'].str.upper())
            axes[0].legend()
            axes[0].grid(True, alpha=0.3, axis='y')
            axes[0].set_yscale('log')

            # Plot 2: Latency Inflation Factor
            axes[1].bar(x - width/2, result_df['inflation_no_pri'], width,
                       label='No Priority', alpha=0.7, color='red')
            axes[1].bar(x + width/2, result_df['inflation_with_pri'], width,
                       label='With Priority', alpha=0.7, color='orange')
            axes[1].axhline(y=1.0, color='green', linestyle='--', linewidth=2, label='Baseline (1×)')

            axes[1].set_ylabel('Latency Inflation Factor (×)')
            axes[1].set_xlabel('Blocking Kernel Type')
            axes[1].set_title('Latency Inflation Under Contention\n(Closer to 1× is Better)')
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(result_df['blocking_type'].str.upper())
            axes[1].legend()
            axes[1].grid(True, alpha=0.3, axis='y')

            # Plot 3: Preemption Latency Estimation
            axes[2].barh(result_df['blocking_type'].str.upper(),
                        result_df['no_pri_p99'],
                        color=['red', 'orange'], alpha=0.7)
            axes[2].axvline(x=baseline_p99, color='green', linestyle='--',
                           linewidth=2, label=f'Baseline ({baseline_p99:.3f} ms)')
            axes[2].axvline(x=0.05, color='blue', linestyle='--',
                           linewidth=2, label='XSched Lv3 (~0.05 ms)')

            axes[2].set_xlabel('Estimated Preemption Latency (ms)')
            axes[2].set_ylabel('Blocking Kernel Type')
            axes[2].set_title('Preemption Latency Estimation\n(Time to Preempt Blocking Kernel)')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3, axis='x')
            axes[2].set_xscale('log')

            max_inflation = result_df['inflation_no_pri'].max()
            avg_preemption = result_df['no_pri_p99'].mean()
        else:
            for ax in axes:
                ax.text(0.5, 0.5, 'No preemption\nlatency data',
                       ha='center', va='center', transform=ax.transAxes)
            max_inflation = 0
            avg_preemption = 0

        plt.tight_layout()
        plt.savefig(self.figures_dir / "rq10_preemption_latency.png", dpi=300)
        plt.close()

        print("\n" + "="*70)
        print(f"Key Findings:")
        print(f"  - Baseline P99: {baseline_p99:.3f} ms")
        print(f"  - Max Inflation: {max_inflation:.1f}×")
        print(f"  - Avg Preemption Latency: {avg_preemption:.3f} ms")
        print(f"  - CUDA shows LIMITED preemption capability (priority has minimal effect)")
        print(f"  - Comparison: XSched Lv3 achieves ~0.01-0.05 ms preemption latency")
        print("="*70)

        return {
            'baseline_p99': baseline_p99,
            'max_inflation': max_inflation,
            'avg_preemption_latency': avg_preemption
        }

    def analyze_rq11_bandwidth_partitioning(self) -> Dict:
        """Analyze RQ11: Bandwidth partitioning."""
        print("\n=== RQ11: Bandwidth Partitioning Analysis ===")

        df = self.load_data("rq11_bandwidth_partition.csv")

        print("\nQuota Accuracy (Target vs Achieved):")

        results = []
        for target_ratio in df['target_ratio'].unique():
            subset = df[df['target_ratio'] == target_ratio]

            # Calculate actual throughput ratio
            # Assuming first 4 streams are front, last 4 are back
            front_throughput = subset['throughput'].mean() * subset['front_kernels'].mean() / subset['total_kernels'].mean()
            back_throughput = subset['throughput'].mean() * subset['back_kernels'].mean() / subset['total_kernels'].mean()
            total = front_throughput + back_throughput

            actual_front_pct = (front_throughput / total) * 100 if total > 0 else 0
            target_front_pct = subset['target_front_pct'].mean()

            error = abs(actual_front_pct - target_front_pct)

            print(f"  {target_ratio}: Target={target_front_pct:.0f}%, "
                  f"Achieved={actual_front_pct:.1f}%, Error={error:.1f}%")

            results.append({
                'target_ratio': target_ratio,
                'target_pct': target_front_pct,
                'actual_pct': actual_front_pct,
                'error': error
            })

        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Target vs Achieved
        result_df = pd.DataFrame(results)
        x = np.arange(len(result_df))
        width = 0.35

        axes[0].bar(x - width/2, result_df['target_pct'], width, label='Target', alpha=0.8)
        axes[0].bar(x + width/2, result_df['actual_pct'], width, label='Achieved', alpha=0.8)
        axes[0].set_xlabel('Target Ratio')
        axes[0].set_ylabel('Front-end Bandwidth (%)')
        axes[0].set_title('Bandwidth Partition: Target vs Achieved')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(result_df['target_ratio'])
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')

        # Error analysis
        axes[1].bar(result_df['target_ratio'], result_df['error'], color='red', alpha=0.6)
        axes[1].axhline(y=5.0, color='orange', linestyle='--', label='Target <5% error')
        axes[1].set_xlabel('Target Ratio')
        axes[1].set_ylabel('Quota Error (%)')
        axes[1].set_title('Bandwidth Partition Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(self.figures_dir / "rq11_bandwidth_partition.png", dpi=300)
        plt.close()

        return {
            'max_error': result_df['error'].max(),
            'avg_error': result_df['error'].mean()
        }

    def generate_summary_report(self, analyses: Dict) -> str:
        """Generate markdown summary report."""
        report = "# GPU Scheduler Experiment Results\n\n"
        report += f"Generated: {pd.Timestamp.now()}\n\n"

        report += "## Key Findings\n\n"

        if 'RQ1' in analyses:
            report += "### RQ1: Stream Scalability\n"
            if 'best_streams' in analyses['RQ1']:
                # New multi-workload format
                report += f"- Best configuration: {analyses['RQ1']['best_streams']} streams, "
                report += f"{analyses['RQ1']['best_workload_mb']:.2f} MB workload\n"
                report += f"- Peak throughput: {analyses['RQ1']['best_throughput']:.2f} kernels/sec\n"
                report += f"- Workload variations tested: {analyses['RQ1']['workload_variations']}\n\n"
            else:
                # Legacy single-workload format
                report += f"- Optimal stream count: {analyses['RQ1']['optimal_streams']}\n"
                report += f"- Peak throughput: {analyses['RQ1']['optimal_throughput']:.2f} kernels/sec\n"
                report += f"- Saturation point: ~{analyses['RQ1']['saturation_point']} streams\n\n"

        if 'RQ2' in analyses:
            report += "### RQ2: Workload Characterization\n"
            report += f"- Best concurrency: {analyses['RQ2']['best_concurrent_workload']}\n"
            report += f"- Best throughput: {analyses['RQ2']['best_throughput_workload']}\n\n"

        report += "\n## Visualizations\n\n"
        report += "All figures saved to: `results/figures/`\n\n"

        # Save report
        with open(self.results_dir / "ANALYSIS_REPORT.md", "w") as f:
            f.write(report)

        return report

    def _plot_with_error(self, df, x_col, y_col, ax, xlabel, ylabel):
        """Plot with error bars."""
        grouped = df.groupby(x_col)[y_col].agg(['mean', 'std']).reset_index()
        ax.errorbar(grouped[x_col], grouped['mean'], yerr=grouped['std'],
                    fmt='o-', linewidth=2, markersize=8, capsize=5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    def _find_saturation_point(self, series, threshold=0.05):
        """Find saturation point where growth rate drops below threshold."""
        values = series.values
        for i in range(1, len(values)):
            growth_rate = (values[i] - values[i-1]) / values[i-1]
            if growth_rate < threshold:
                return series.index[i]
        return series.index[-1]


def main():
    parser = argparse.ArgumentParser(description="Analyze GPU scheduler experiments")
    parser.add_argument("--results", default="results", help="Results directory")
    parser.add_argument("--experiments", nargs="+",
                       choices=["RQ1", "RQ2", "RQ3", "RQ4", "RQ5", "RQ6", "RQ7", "RQ9", "RQ11", "all"],
                       default=["all"], help="Which experiments to analyze")

    args = parser.parse_args()

    print("="*60)
    print("GPU Scheduler Experiment Analysis")
    print("="*60)

    analyzer = ResultAnalyzer(args.results)
    analyses = {}

    exp_map = {
        "RQ1": analyzer.analyze_rq1_stream_scalability,
        "RQ2": analyzer.analyze_rq2_workload_characterization,
        "RQ3": analyzer.analyze_rq3_priority_effectiveness,
        "RQ4": analyzer.analyze_rq4_memory_pressure,
        "RQ5": analyzer.analyze_rq5_multi_process,
        "RQ6": analyzer.analyze_rq6_load_imbalance,
        "RQ7": analyzer.analyze_rq7_tail_latency,
        "RQ9": analyzer.analyze_rq9_priority_tail_latency,
        "RQ11": analyzer.analyze_rq11_bandwidth_partitioning,
    }

    experiments_to_run = list(exp_map.keys()) if "all" in args.experiments else args.experiments

    for exp_name in experiments_to_run:
        if exp_name in exp_map:
            try:
                print(f"\nAnalyzing {exp_name}...")
                result = exp_map[exp_name]()
                analyses[exp_name] = result
            except FileNotFoundError as e:
                print(f"✗ {exp_name}: {e}")
            except Exception as e:
                print(f"✗ {exp_name} failed: {e}")

    # Generate summary
    report = analyzer.generate_summary_report(analyses)
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(report)


if __name__ == "__main__":
    main()
