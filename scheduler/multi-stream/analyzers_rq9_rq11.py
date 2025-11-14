#!/usr/bin/env python3
"""
RQ9-RQ11 Analyzers

Contains analysis methods for:
- RQ9: Priority-Based Tail Latency
- RQ10: Preemption Latency (DEPRECATED)
- RQ11: Bandwidth Partitioning
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
from analyzer_base import BaseAnalyzer


class RQ9_RQ11_Analyzer(BaseAnalyzer):
    """Analyzer for RQ9, RQ10, and RQ11."""
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

