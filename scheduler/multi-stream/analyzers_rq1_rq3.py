#!/usr/bin/env python3
"""
RQ1-RQ3 Analyzers

Contains analysis methods for:
- RQ1: Stream Scalability
- RQ2: Workload Characterization  
- RQ3: Priority Effectiveness
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict
from analyzer_base import BaseAnalyzer


class RQ1_RQ3_Analyzer(BaseAnalyzer):
    """Analyzer for RQ1, RQ2, and RQ3."""
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

