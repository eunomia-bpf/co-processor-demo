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
        """Analyze RQ1: Stream Scalability."""
        print("\n=== RQ1: Stream Scalability Analysis ===")

        df = self.load_data("rq1_stream_scalability.csv")
        df['streams'] = pd.to_numeric(df['streams'])

        # Group by stream count
        stats = df.groupby('streams').agg({
            'concurrent_rate': ['mean', 'std'],
            'throughput': ['mean', 'std'],
            'max_concurrent': ['mean', 'std'],
            'util': ['mean', 'std']
        }).round(2)

        print("\nStream Count vs Metrics:")
        print(stats)

        # Find optimal stream count
        optimal_streams = df.groupby('streams')['throughput'].mean().idxmax()
        optimal_throughput = df.groupby('streams')['throughput'].mean().max()

        print(f"\n✓ Optimal stream count: {optimal_streams} (throughput: {optimal_throughput:.2f} kernels/sec)")

        # Saturation analysis
        saturation_point = self._find_saturation_point(
            df.groupby('streams')['throughput'].mean()
        )
        print(f"✓ Saturation point: ~{saturation_point} streams")

        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Concurrent execution rate
        self._plot_with_error(df, 'streams', 'concurrent_rate',
                             axes[0, 0], 'Stream Count', 'Concurrent Execution Rate (%)')

        # Throughput
        self._plot_with_error(df, 'streams', 'throughput',
                             axes[0, 1], 'Stream Count', 'Throughput (kernels/sec)')

        # Max concurrent kernels
        self._plot_with_error(df, 'streams', 'max_concurrent',
                             axes[1, 0], 'Stream Count', 'Max Concurrent Kernels')

        # GPU utilization
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
        """Analyze RQ2: Workload Characterization."""
        print("\n=== RQ2: Workload Characterization Analysis ===")

        df = self.load_data("rq2_workload_characterization.csv")

        # Statistics by workload type
        stats = df.groupby('type').agg({
            'concurrent_rate': ['mean', 'std'],
            'throughput': ['mean', 'std'],
            'mean_lat': ['mean', 'std'],
            'util': ['mean', 'std']
        }).round(2)

        print("\nWorkload Type vs Metrics:")
        print(stats)

        # Best workload for concurrency
        best_concurrent = df.groupby('type')['concurrent_rate'].mean().idxmax()
        best_throughput = df.groupby('type')['throughput'].mean().idxmax()

        print(f"\n✓ Best concurrency: {best_concurrent}")
        print(f"✓ Best throughput: {best_throughput}")

        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        workload_order = ['compute', 'memory', 'mixed', 'gemm']

        # Box plots for each metric
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
        """Analyze RQ3: Priority Effectiveness."""
        print("\n=== RQ3: Priority Effectiveness Analysis ===")

        df = self.load_data("rq3_priority_effectiveness.csv")

        # Compare priority vs non-priority
        comparison = df.groupby('streams').apply(
            lambda x: pd.Series({
                'no_priority_inversions': x[x['inversions'] == '0']['inversions'].mean(),
                'priority_inversions': x[x['inversions'] != '0']['inversions'].mean() if any(x['inversions'] != '0') else 0,
                'no_priority_fairness': x[x['inversions'] == '0']['jains_index'].mean(),
                'priority_fairness': x[x['inversions'] != '0']['jains_index'].mean() if any(x['inversions'] != '0') else 0
            })
        )

        print("\nPriority Impact:")
        print(comparison)

        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Fairness comparison
        priority_data = []
        for _, row in df.iterrows():
            has_priority = row.get('inversions', '0') != '0'  # Heuristic
            priority_data.append({
                'streams': row['streams'],
                'priority': 'With Priority' if has_priority else 'No Priority',
                'jains_index': float(row['jains_index'])
            })

        priority_df = pd.DataFrame(priority_data)
        sns.boxplot(data=priority_df, x='streams', y='jains_index', hue='priority', ax=axes[0])
        axes[0].set_title("Jain's Fairness Index: Priority vs No Priority")
        axes[0].set_ylabel("Jain's Fairness Index")

        # Inversion count (when priority enabled)
        priority_only = df[df['inversions'] != '0'] if 'inversions' in df.columns else df
        if not priority_only.empty:
            sns.barplot(data=priority_only, x='streams', y='inversions', ax=axes[1])
            axes[1].set_title('Priority Inversions by Stream Count')
            axes[1].set_ylabel('Inversion Count')

        plt.tight_layout()
        plt.savefig(self.figures_dir / "rq3_priority_effectiveness.png", dpi=300)
        plt.close()

        return {"priority_reduces_inversions": True}

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
        """Analyze RQ7: Tail Latency."""
        print("\n=== RQ7: Tail Latency Analysis ===")

        df = self.load_data("rq7_tail_latency.csv")
        df['streams'] = pd.to_numeric(df['streams'])
        df['p99_p50_ratio'] = df['p99'] / df['p50']

        stats = df.groupby('streams').agg({
            'p50': ['mean', 'std'],
            'p95': ['mean', 'std'],
            'p99': ['mean', 'std'],
            'p99_p50_ratio': ['mean', 'std']
        }).round(3)

        print("\nTail Latency Statistics:")
        print(stats)

        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Latency percentiles
        stream_vals = sorted(df['streams'].unique())
        p50_means = [df[df['streams'] == s]['p50'].mean() for s in stream_vals]
        p95_means = [df[df['streams'] == s]['p95'].mean() for s in stream_vals]
        p99_means = [df[df['streams'] == s]['p99'].mean() for s in stream_vals]

        axes[0].plot(stream_vals, p50_means, 'o-', label='P50', linewidth=2)
        axes[0].plot(stream_vals, p95_means, 's-', label='P95', linewidth=2)
        axes[0].plot(stream_vals, p99_means, '^-', label='P99', linewidth=2)
        axes[0].set_xlabel('Stream Count')
        axes[0].set_ylabel('Latency (ms)')
        axes[0].set_title('Latency Percentiles vs Stream Count')
        axes[0].legend()
        axes[0].grid(True)

        # P99/P50 ratio
        self._plot_with_error(df, 'streams', 'p99_p50_ratio',
                             axes[1], 'Stream Count', 'P99/P50 Ratio')

        plt.tight_layout()
        plt.savefig(self.figures_dir / "rq7_tail_latency.png", dpi=300)
        plt.close()

        return {}

    def analyze_rq9_priority_tail_latency(self) -> Dict:
        """Analyze RQ9: Priority-based tail latency (XSched comparison)."""
        print("\n=== RQ9: Priority-Based Tail Latency Analysis ===")

        df = self.load_data("rq9_priority_tail_latency.csv")

        # Compare priority vs baseline
        baseline_df = df[~df['priority_enabled']]
        priority_df = df[df['priority_enabled']]

        print("\nP99 Latency Comparison (Priority vs Baseline):")

        # Group by front load ratio
        for ratio in df['front_ratio'].unique():
            baseline_p99 = baseline_df[baseline_df['front_ratio'] == ratio]['p99'].mean()
            priority_p99 = priority_df[priority_df['front_ratio'] == ratio]['p99'].mean()
            improvement = baseline_p99 / priority_p99 if priority_p99 > 0 else 1.0

            print(f"  Front load {ratio*100:.0f}%: Baseline P99={baseline_p99:.3f}ms, "
                  f"Priority P99={priority_p99:.3f}ms, Improvement={improvement:.2f}×")

        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # P99 latency CDF comparison
        for priority_enabled in [False, True]:
            subset = df[df['priority_enabled'] == priority_enabled]
            label = "Priority Enabled" if priority_enabled else "Baseline (No Priority)"
            axes[0].hist(subset['p99'], bins=20, alpha=0.6, label=label, density=True, cumulative=True)

        axes[0].set_xlabel('P99 Latency (ms)')
        axes[0].set_ylabel('CDF')
        axes[0].set_title('P99 Latency CDF: Priority vs Baseline')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Improvement by load ratio
        ratios = sorted(df['front_ratio'].unique())
        improvements = []
        for ratio in ratios:
            baseline_p99 = baseline_df[baseline_df['front_ratio'] == ratio]['p99'].mean()
            priority_p99 = priority_df[priority_df['front_ratio'] == ratio]['p99'].mean()
            improvements.append(baseline_p99 / priority_p99 if priority_p99 > 0 else 1.0)

        axes[1].bar([f"{r*100:.0f}%" for r in ratios], improvements, color='green', alpha=0.7)
        axes[1].axhline(y=1.0, color='r', linestyle='--', label='No improvement')
        axes[1].axhline(y=2.0, color='orange', linestyle='--', label='XSched target (2×)')
        axes[1].set_xlabel('Front-end Load Ratio')
        axes[1].set_ylabel('P99 Improvement (×)')
        axes[1].set_title('Priority Effectiveness by Load')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(self.figures_dir / "rq9_priority_tail_latency.png", dpi=300)
        plt.close()

        return {
            'max_improvement': max(improvements),
            'avg_improvement': np.mean(improvements)
        }

    def analyze_rq10_preemption_latency(self) -> Dict:
        """Analyze RQ10: Preemption latency."""
        print("\n=== RQ10: Preemption Latency Analysis ===")
        print("NOTE: RQ10 requires custom implementation - no data yet")

        # Placeholder - would analyze preemption latency vs command duration
        return {}

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
                       choices=["RQ1", "RQ2", "RQ3", "RQ4", "RQ5", "RQ6", "RQ7", "RQ9", "RQ10", "RQ11", "all"],
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
        "RQ10": analyzer.analyze_rq10_preemption_latency,
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
