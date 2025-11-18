#!/usr/bin/env python3
"""
Visualization script for GEMM benchmark results
Generates plots showing performance comparisons across different matrix patterns and policies
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from pathlib import Path

def load_results(csv_file):
    """Load benchmark results from CSV file"""
    if not os.path.exists(csv_file):
        print(f"Error: Results file not found: {csv_file}")
        return None

    df = pd.read_csv(csv_file)
    return df

def plot_performance_comparison(df, output_dir):
    """Plot GFLOPS performance comparison across patterns and policies"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Get unique sizes
    sizes = df['size'].unique()

    for idx, size in enumerate(sizes):
        df_size = df[df['size'] == size]

        # Prepare data for plotting
        patterns = df_size['pattern'].unique()
        configs = []

        for pattern in patterns:
            df_pattern = df_size[df_size['pattern'] == pattern]
            for _, row in df_pattern.iterrows():
                config_name = f"{row['policy']}" if row['config'] == 'policy' else 'original'
                configs.append({
                    'pattern': pattern,
                    'config': config_name,
                    'gflops': row['gflops_avg'],
                    'gflops_std': row['gflops_std']
                })

        df_plot = pd.DataFrame(configs)

        # Create grouped bar plot
        ax = axes[idx] if len(sizes) > 1 else axes

        # Pivot for grouped bar chart
        pivot_data = df_plot.pivot(index='pattern', columns='config', values='gflops')
        pivot_std = df_plot.pivot(index='pattern', columns='config', values='gflops_std')

        pivot_data.plot(kind='bar', ax=ax, yerr=pivot_std, capsize=4, width=0.8)
        ax.set_title(f'GFLOPS Performance - {size}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Matrix Pattern', fontsize=12)
        ax.set_ylabel('GFLOPS', fontsize=12)
        ax.legend(title='Configuration', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'performance_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

def plot_speedup_analysis(df, output_dir):
    """Plot speedup factors for each policy vs original"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sizes = df['size'].unique()

    for idx, size in enumerate(sizes):
        df_size = df[df['size'] == size]
        df_policy = df_size[df_size['config'] == 'policy']

        if df_policy.empty:
            continue

        patterns = df_policy['pattern'].unique()
        speedups = []

        for pattern in patterns:
            df_pattern = df_policy[df_policy['pattern'] == pattern]
            for _, row in df_pattern.iterrows():
                speedups.append({
                    'pattern': pattern,
                    'policy': row['policy'],
                    'speedup': row['speedup_vs_original']
                })

        df_speedup = pd.DataFrame(speedups)

        # Create grouped bar plot
        ax = axes[idx] if len(sizes) > 1 else axes

        pivot_speedup = df_speedup.pivot(index='pattern', columns='policy', values='speedup')
        pivot_speedup.plot(kind='bar', ax=ax, width=0.8)

        # Add horizontal line at 1.0 (no speedup)
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Baseline (1.0x)')

        ax.set_title(f'Speedup vs Original - {size}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Matrix Pattern', fontsize=12)
        ax.set_ylabel('Speedup Factor', fontsize=12)
        ax.legend(title='Policy', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'speedup_analysis.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

def plot_latency_comparison(df, output_dir):
    """Plot execution time (latency) comparison"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sizes = df['size'].unique()

    for idx, size in enumerate(sizes):
        df_size = df[df['size'] == size]

        patterns = df_size['pattern'].unique()
        times = []

        for pattern in patterns:
            df_pattern = df_size[df_size['pattern'] == pattern]
            for _, row in df_pattern.iterrows():
                config_name = f"{row['policy']}" if row['config'] == 'policy' else 'original'
                times.append({
                    'pattern': pattern,
                    'config': config_name,
                    'time': row['time_avg'],
                    'time_std': row['time_std']
                })

        df_time = pd.DataFrame(times)

        ax = axes[idx] if len(sizes) > 1 else axes

        pivot_time = df_time.pivot(index='pattern', columns='config', values='time')
        pivot_std = df_time.pivot(index='pattern', columns='config', values='time_std')

        pivot_time.plot(kind='bar', ax=ax, yerr=pivot_std, capsize=4, width=0.8)
        ax.set_title(f'Execution Time - {size}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Matrix Pattern', fontsize=12)
        ax.set_ylabel('Time (ms)', fontsize=12)
        ax.legend(title='Configuration', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'latency_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

def plot_size_scaling(df, output_dir):
    """Plot performance scaling across different matrix sizes"""
    patterns = df['pattern'].unique()

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for idx, pattern in enumerate(patterns):
        if idx >= len(axes):
            break

        df_pattern = df[df['pattern'] == pattern]

        ax = axes[idx]

        # Group by size and config
        for config in ['original', 'policy']:
            if config == 'original':
                df_config = df_pattern[df_pattern['config'] == 'original']
                label = 'Original'
                marker = 'o'
            else:
                # Show each policy separately
                for policy in df_pattern[df_pattern['config'] == 'policy']['policy'].unique():
                    df_config = df_pattern[(df_pattern['config'] == 'policy') & (df_pattern['policy'] == policy)]
                    sizes = []
                    gflops = []
                    for size in df['size'].unique():
                        df_size = df_config[df_config['size'] == size]
                        if not df_size.empty:
                            sizes.append(size)
                            gflops.append(df_size['gflops_avg'].values[0])

                    if sizes:
                        ax.plot(range(len(sizes)), gflops, marker='s', label=f'Policy-{policy}', linewidth=2)
                continue

            sizes = []
            gflops = []
            for size in df['size'].unique():
                df_size = df_config[df_config['size'] == size]
                if not df_size.empty:
                    sizes.append(size)
                    gflops.append(df_size['gflops_avg'].values[0])

            if sizes:
                ax.plot(range(len(sizes)), gflops, marker=marker, label=label, linewidth=2)

        ax.set_title(f'Pattern: {pattern}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Matrix Size', fontsize=10)
        ax.set_ylabel('GFLOPS', fontsize=10)
        ax.set_xticks(range(len(df['size'].unique())))
        ax.set_xticklabels(df['size'].unique(), rotation=45, ha='right')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # Hide unused subplots
    for idx in range(len(patterns), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'size_scaling.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

def print_summary_table(df):
    """Print summary statistics table"""
    print("\n" + "=" * 100)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 100)

    for size in df['size'].unique():
        print(f"\n{'=' * 100}")
        print(f"Matrix Size: {size}")
        print(f"{'=' * 100}")
        print(f"{'Pattern':<15} {'Config':<12} {'Policy':<12} {'Time (ms)':<15} {'GFLOPS':<15} {'Speedup':<10}")
        print("-" * 100)

        df_size = df[df['size'] == size]

        for pattern in df_size['pattern'].unique():
            df_pattern = df_size[df_size['pattern'] == pattern]

            for _, row in df_pattern.iterrows():
                config = row['config']
                policy = row['policy'] if row['config'] == 'policy' else '-'
                time_str = f"{row['time_avg']:.3f} ± {row['time_std']:.3f}"
                gflops_str = f"{row['gflops_avg']:.2f}"
                speedup = row.get('speedup_vs_original', '-')
                speedup_str = f"{speedup:.2f}x" if isinstance(speedup, (int, float)) else '-'

                print(f"{pattern:<15} {config:<12} {policy:<12} {time_str:<15} {gflops_str:<15} {speedup_str:<10}")

    print("\n" + "=" * 100)

def generate_html_report(df, output_dir):
    """Generate HTML report with embedded images"""
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>GEMM Benchmark Results</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
            text-align: center;
        }
        h2 {
            color: #555;
            margin-top: 30px;
            border-bottom: 2px solid #ddd;
            padding-bottom: 10px;
        }
        .plot {
            text-align: center;
            margin: 20px 0;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .plot img {
            max-width: 100%;
            height: auto;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #4CAF50;
            color: white;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .speedup-good {
            color: green;
            font-weight: bold;
        }
        .speedup-neutral {
            color: orange;
        }
    </style>
</head>
<body>
    <h1>GEMM Benchmark Results</h1>
    <p style="text-align: center; color: #666;">
        Comparison of GEMM performance with different matrix patterns and scheduling policies
    </p>

    <h2>Performance Comparison (GFLOPS)</h2>
    <div class="plot">
        <img src="performance_comparison.png" alt="Performance Comparison">
    </div>

    <h2>Speedup Analysis</h2>
    <div class="plot">
        <img src="speedup_analysis.png" alt="Speedup Analysis">
    </div>

    <h2>Latency Comparison (Execution Time)</h2>
    <div class="plot">
        <img src="latency_comparison.png" alt="Latency Comparison">
    </div>

    <h2>Performance Scaling Across Matrix Sizes</h2>
    <div class="plot">
        <img src="size_scaling.png" alt="Size Scaling">
    </div>

    <h2>Detailed Results Table</h2>
"""

    # Add table for each size
    for size in df['size'].unique():
        df_size = df[df['size'] == size]
        html_content += f"""
    <h3>Matrix Size: {size}</h3>
    <table>
        <tr>
            <th>Pattern</th>
            <th>Config</th>
            <th>Policy</th>
            <th>Time (ms)</th>
            <th>GFLOPS</th>
            <th>Speedup</th>
            <th>Verified</th>
        </tr>
"""

        for _, row in df_size.iterrows():
            policy = row['policy'] if row['config'] == 'policy' else '-'
            speedup = row.get('speedup_vs_original', '-')
            speedup_class = 'speedup-good' if isinstance(speedup, (int, float)) and speedup > 1.0 else 'speedup-neutral'
            speedup_str = f"{speedup:.2f}x" if isinstance(speedup, (int, float)) else '-'
            verified_str = '✓' if row['verified'] else '✗'

            html_content += f"""
        <tr>
            <td>{row['pattern']}</td>
            <td>{row['config']}</td>
            <td>{policy}</td>
            <td>{row['time_avg']:.3f} ± {row['time_std']:.3f}</td>
            <td>{row['gflops_avg']:.2f}</td>
            <td class="{speedup_class}">{speedup_str}</td>
            <td>{verified_str}</td>
        </tr>
"""

        html_content += """
    </table>
"""

    html_content += """
</body>
</html>
"""

    output_file = os.path.join(output_dir, 'benchmark_report.html')
    with open(output_file, 'w') as f:
        f.write(html_content)

    print(f"✓ Saved: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Visualize GEMM benchmark results')
    parser.add_argument('--input', default='benchmark_results.csv',
                        help='Input CSV file with benchmark results (default: benchmark_results.csv)')
    parser.add_argument('--output-dir', default='./plots',
                        help='Output directory for plots (default: ./plots)')

    args = parser.parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load results
    print("=" * 80)
    print("GEMM Benchmark Visualization")
    print("=" * 80)
    print(f"Input file: {args.input}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)

    df = load_results(args.input)
    if df is None:
        return 1

    print(f"\nLoaded {len(df)} benchmark results")

    # Set plotting style
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'

    # Generate plots
    print("\nGenerating visualizations...")
    plot_performance_comparison(df, args.output_dir)
    plot_speedup_analysis(df, args.output_dir)
    plot_latency_comparison(df, args.output_dir)
    plot_size_scaling(df, args.output_dir)

    # Generate HTML report
    print("\nGenerating HTML report...")
    generate_html_report(df, args.output_dir)

    # Print summary table
    print_summary_table(df)

    print("\n" + "=" * 80)
    print("Visualization Complete!")
    print("=" * 80)
    print(f"\nView the results:")
    print(f"  HTML Report: {os.path.join(args.output_dir, 'benchmark_report.html')}")
    print(f"  Plots: {args.output_dir}/")
    print("=" * 80)

if __name__ == '__main__':
    main()
