#!/usr/bin/env python3
"""
Result Analyzer for CLC Policy Benchmark Exploration

Analyzes the comprehensive CSV from explore_configurations.py and generates:
- Summary statistics
- Best/worst configurations for each policy
- Performance trends
- Comparison tables
"""

import pandas as pd
import sys
import os
from typing import Dict, List


def load_results(filename: str) -> pd.DataFrame:
    """Load results CSV file."""

    if not os.path.exists(filename):
        print(f"ERROR: File '{filename}' not found", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {filename}...", flush=True)
    df = pd.read_csv(filename)
    print(f"  Loaded {len(df)} results", flush=True)
    print()

    return df


def print_section(title: str):
    """Print a section header."""
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)
    print()


def analyze_overall_statistics(df: pd.DataFrame):
    """Print overall statistics."""

    print_section("Overall Statistics")

    print(f"Total configurations tested: {df['config_n'].nunique() * df['config_threads'].nunique() * df['config_imbalance_scale'].nunique() * df['config_workload_scale'].nunique()}")
    print(f"Total workloads: {df['Workload'].nunique()}")
    print(f"Total results: {len(df)}")
    print()

    print("Configurations explored:")
    print(f"  Elements (n):        {sorted(df['config_n'].unique())}")
    print(f"  Threads per block:   {sorted(df['config_threads'].unique())}")
    print(f"  Imbalance scales:    {sorted(df['config_imbalance_scale'].unique())}")
    print(f"  Workload scales:     {sorted(df['config_workload_scale'].unique())}")
    print()

    print("Workloads:")
    for i, workload in enumerate(df['Workload'].unique(), 1):
        count = len(df[df['Workload'] == workload])
        print(f"  {i}. {workload} ({count} configs)")


def analyze_clustered_heavy(df: pd.DataFrame):
    """Analyze ClusteredHeavy workload specifically (our key result)."""

    print_section("ClusteredHeavy Workload Analysis")

    clustered = df[df['Workload'].str.contains('Clustered Heavy', na=False)].copy()

    if len(clustered) == 0:
        print("No ClusteredHeavy results found")
        return

    print(f"Total ClusteredHeavy configurations: {len(clustered)}")
    print()

    # Best LatencyBudget vs Greedy
    if 'LatencyBudget_vs_Greedy_pct' in clustered.columns:
        best_idx = clustered['LatencyBudget_vs_Greedy_pct'].idxmax()
        best = clustered.loc[best_idx]

        print("BEST: LatencyBudget vs Greedy")
        print(f"  Speedup: {best['LatencyBudget_vs_Greedy_pct']:.2f}%")
        print(f"  Greedy time: {best['Greedy_ms']:.3f} ms ({best['Greedy_steals']:.0f} steals)")
        print(f"  LatencyBudget time: {best['LatencyBudget_ms']:.3f} ms ({best['LatencyBudget_steals']:.0f} steals)")
        print(f"  Configuration: n={best['config_n']:.0f}, threads={best['config_threads']:.0f}, imb={best['config_imbalance_scale']:.1f}, work={best['config_workload_scale']:.1f}")
        print()

    # Worst policy vs Greedy
    speedup_cols = [col for col in clustered.columns if col.endswith('_vs_Greedy_pct')]

    if speedup_cols:
        worst_speedup = float('inf')
        worst_policy = None
        worst_row = None

        for col in speedup_cols:
            min_val = clustered[col].min()
            if min_val < worst_speedup:
                worst_speedup = min_val
                worst_policy = col.replace('_vs_Greedy_pct', '')
                worst_row = clustered.loc[clustered[col].idxmin()]

        if worst_policy:
            print(f"WORST: {worst_policy} vs Greedy")
            print(f"  Speedup: {worst_speedup:.2f}%")
            print(f"  Greedy time: {worst_row['Greedy_ms']:.3f} ms")
            print(f"  {worst_policy} time: {worst_row[f'{worst_policy}_ms']:.3f} ms")
            print(f"  Configuration: n={worst_row['config_n']:.0f}, threads={worst_row['config_threads']:.0f}, imb={worst_row['config_imbalance_scale']:.1f}, work={worst_row['config_workload_scale']:.1f}")
            print()

    # Average speedups across all configs
    print("AVERAGE speedups across all configurations (vs Greedy):")
    for col in speedup_cols:
        policy = col.replace('_vs_Greedy_pct', '')
        avg_speedup = clustered[col].mean()
        print(f"  {policy:20s}: {avg_speedup:6.2f}%")


def analyze_policy_consistency(df: pd.DataFrame):
    """Analyze how consistent policies are across different configurations."""

    print_section("Policy Consistency Analysis")

    # For each workload, calculate stddev of speedups
    workloads = df['Workload'].unique()

    print("Standard deviation of speedups (lower = more consistent):")
    print()

    for workload in workloads[:3]:  # Top 3 workloads only to keep output manageable
        workload_df = df[df['Workload'] == workload]

        print(f"{workload}:")

        speedup_cols = [col for col in workload_df.columns if col.endswith('_vs_Greedy_pct')]

        for col in speedup_cols:
            policy = col.replace('_vs_Greedy_pct', '')
            std = workload_df[col].std()
            mean = workload_df[col].mean()
            print(f"  {policy:20s}: mean={mean:6.2f}%, std={std:5.2f}%")

        print()


def analyze_parameter_effects(df: pd.DataFrame):
    """Analyze how parameters affect performance."""

    print_section("Parameter Effects on ClusteredHeavy")

    clustered = df[df['Workload'].str.contains('Clustered Heavy', na=False)].copy()

    if len(clustered) == 0:
        return

    # Effect of imbalance_scale on LatencyBudget speedup
    print("Effect of imbalance_scale on LatencyBudget vs Greedy:")
    if 'LatencyBudget_vs_Greedy_pct' in clustered.columns:
        for imb in sorted(clustered['config_imbalance_scale'].unique()):
            subset = clustered[clustered['config_imbalance_scale'] == imb]
            avg_speedup = subset['LatencyBudget_vs_Greedy_pct'].mean()
            print(f"  imbalance={imb:.1f}: {avg_speedup:6.2f}% average speedup")
        print()

    # Effect of workload_scale
    print("Effect of workload_scale on LatencyBudget vs Greedy:")
    if 'LatencyBudget_vs_Greedy_pct' in clustered.columns:
        for work in sorted(clustered['config_workload_scale'].unique()):
            subset = clustered[clustered['config_workload_scale'] == work]
            avg_speedup = subset['LatencyBudget_vs_Greedy_pct'].mean()
            print(f"  workload={work:.1f}: {avg_speedup:6.2f}% average speedup")
        print()

    # Effect of problem size (n)
    print("Effect of problem size on LatencyBudget vs Greedy:")
    if 'LatencyBudget_vs_Greedy_pct' in clustered.columns:
        for n in sorted(clustered['config_n'].unique()):
            subset = clustered[clustered['config_n'] == n]
            avg_speedup = subset['LatencyBudget_vs_Greedy_pct'].mean()
            print(f"  n={n/1e6:.1f}M: {avg_speedup:6.2f}% average speedup")


def generate_comparison_table(df: pd.DataFrame):
    """Generate a comparison table for the paper/documentation."""

    print_section("Comparison Table (for Documentation)")

    # Get default configuration results
    default = df[
        (df['config_n'] == 2097152) &
        (df['config_threads'] == 256) &
        (df['config_imbalance_scale'] == 1.0) &
        (df['config_workload_scale'] == 1.0) &
        (df['Workload'].str.contains('Clustered Heavy', na=False))
    ]

    if len(default) == 0:
        print("No results for default configuration")
        return

    row = default.iloc[0]

    policies = ['FixedWork', 'LatencyBudget', 'NeverSteal', 'Greedy', 'LightHelpsHeavy',
                'AdaptiveSteal', 'ClusterAware']

    print("ClusteredHeavy (imbalance=1.0, default config):")
    print()
    print(f"{'Policy':<20s} | {'Time (ms)':>10s} | {'Steals':>10s} | {'vs Greedy':>10s}")
    print("-" * 60)

    for policy in policies:
        time_col = f'{policy}_ms'
        steals_col = f'{policy}_steals'
        speedup_col = f'{policy}_vs_Greedy_pct'

        if time_col in row:
            time_val = row[time_col]
            steals_val = row.get(steals_col, 'N/A')
            speedup_val = row.get(speedup_col, 0)

            if steals_val == 'N/A' or pd.isna(steals_val):
                steals_str = '-'
            else:
                steals_str = f"{float(steals_val):.0f}"

            print(f"{policy:<20s} | {float(time_val):10.3f} | {steals_str:>10s} | {float(speedup_val):9.1f}%")


def main():
    """Main execution function."""

    if len(sys.argv) < 2:
        print("Usage: python3 analyze_results.py <results_csv_file>")
        print()
        print("Example: python3 analyze_results.py exploration_results_20250105_143022.csv")
        sys.exit(1)

    filename = sys.argv[1]
    df = load_results(filename)

    # Run all analyses
    analyze_overall_statistics(df)
    analyze_clustered_heavy(df)
    analyze_policy_consistency(df)
    analyze_parameter_effects(df)
    generate_comparison_table(df)

    print()
    print("=" * 70)
    print("Analysis Complete!")
    print("=" * 70)
    print()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...", file=sys.stderr)
        sys.exit(1)
