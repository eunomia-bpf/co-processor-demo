#!/usr/bin/env python3
"""
GPU Scheduler Experiment Analysis - RQ4-RQ8 Analyzer

Analyzes priority, heterogeneity, arrival pattern, memory, and multi-process research questions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional, Dict
import glob


class RQ4_RQ8_Analyzer:
    """Analyzer for RQ4 (Priority), RQ5 (Heterogeneity), RQ6 (Jitter), RQ7 (Memory), and RQ8 (Multi-Process)."""

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
    # RQ4: Priority Semantics
    # ========================================================================

    def analyze_rq4(self, sub_rqs: Optional[List[str]] = None):
        """Analyze RQ4: CUDA stream priority semantics - combined figure."""
        print("\n=== Analyzing RQ4: Priority Semantics ===")

        # Create a 2x2 subplot figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # RQ4.1: Priority inversion rate vs streams
        print("  RQ4.1: Inversion rate vs streams")
        df = self.load_csv('rq4_1_inversion_rate.csv')
        if df is not None:
            df['has_priority'] = df.get('type_detail', '').str.len() > 0

            grouped = df.groupby(['streams', 'has_priority']).agg({
                'inversion_rate': ['mean', 'std']
            }).reset_index()

            grouped.columns = ['streams', 'has_priority', 'inversion_rate_mean', 'inversion_rate_std']

            for has_prio in [False, True]:
                data = grouped[grouped['has_priority'] == has_prio]
                if len(data) > 0:
                    label = 'With Priority' if has_prio else 'No Priority (Baseline)'
                    # inversion_rate is already in 0-1 range, multiply by 100 for percentage display
                    ax1.plot(data['streams'], data['inversion_rate_mean'] * 100,
                             marker='o', label=label, linewidth=2)
                    ax1.fill_between(data['streams'],
                                     (data['inversion_rate_mean'] - data['inversion_rate_std']) * 100,
                                     (data['inversion_rate_mean'] + data['inversion_rate_std']) * 100,
                                     alpha=0.2)

            ax1.set_xlabel('Number of Streams')
            ax1.set_ylabel('Priority Inversion Rate (%)')
            ax1.set_title('(a) Priority Inversion Rate vs Stream Count')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # RQ4.2: Per-priority P99 vs offered load
        print("  RQ4.2: Per-priority P99 vs load")
        df = self.load_csv('rq4_2_per_priority_vs_load.csv')
        if df is not None:
            df['offered_load'] = df['total_kernels'] / (df['wall_time_ms'] / 1000.0)

            def parse_priorities(row):
                if pd.isna(row['per_priority_p99']):
                    return None, None
                # Handle both colon-separated (new format) and comma-separated (old format)
                val_str = str(row['per_priority_p99'])
                if ':' in val_str:
                    vals = [float(x) for x in val_str.split(':') if x.strip()]
                else:
                    vals = [float(x) for x in val_str.split(',') if x.strip()]
                if len(vals) >= 2:
                    mid = len(vals) // 2
                    high_p99 = np.mean(vals[:mid]) if mid > 0 else vals[0]
                    low_p99 = np.mean(vals[mid:]) if mid < len(vals) else vals[-1]
                    return high_p99, low_p99
                return None, None

            df[['high_prio_p99', 'low_prio_p99']] = df.apply(
                lambda row: pd.Series(parse_priorities(row)), axis=1
            )

            df_sorted = df.dropna(subset=['high_prio_p99', 'low_prio_p99'])
            if len(df_sorted) > 0:
                grouped = df_sorted.groupby(pd.cut(df_sorted['offered_load'], bins=6)).agg({
                    'high_prio_p99': ['mean', 'std'],
                    'low_prio_p99': ['mean', 'std'],
                    'offered_load': 'mean'
                }).reset_index(drop=True)

                grouped.columns = ['high_p99_mean', 'high_p99_std', 'low_p99_mean',
                                   'low_p99_std', 'offered_load']

                ax2.plot(grouped['offered_load'], grouped['high_p99_mean'],
                         marker='o', label='High Priority', linewidth=2, color='blue')
                ax2.fill_between(grouped['offered_load'],
                                 grouped['high_p99_mean'] - grouped['high_p99_std'],
                                 grouped['high_p99_mean'] + grouped['high_p99_std'],
                                 alpha=0.2, color='blue')

                ax2.plot(grouped['offered_load'], grouped['low_p99_mean'],
                         marker='s', label='Low Priority', linewidth=2, color='red')
                ax2.fill_between(grouped['offered_load'],
                                 grouped['low_p99_mean'] - grouped['low_p99_std'],
                                 grouped['low_p99_mean'] + grouped['low_p99_std'],
                                 alpha=0.2, color='red')

                ax2.set_xlabel('Offered Load (kernels/sec)')
                ax2.set_ylabel('E2E P99 Latency (ms)')
                ax2.set_title('(b) Per-Priority P99 Latency vs Offered Load')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            else:
                print("    Warning: No valid priority data found")

        # RQ4.3: Fast kernels P99 in RT vs BE scenario
        print("  RQ4.3: Fast kernels RT vs BE scenario")
        df = self.load_csv('rq4_3_fast_kernels_rt_be.csv')
        if df is not None:
            df['config'] = 'Unknown'

            mask_only_fast = (df.get('type_detail', '') == '') & (df.get('kernels_per_stream_detail', '').str.count(',') == 0)
            df.loc[mask_only_fast, 'config'] = 'Only Fast'

            num_runs = 3
            total_rows = len(df)

            if total_rows >= num_runs * 3:
                df.iloc[0:num_runs, df.columns.get_loc('config')] = 'Only Fast'
                df.iloc[num_runs:2*num_runs, df.columns.get_loc('config')] = 'Fast+Slow, No Priority'
                df.iloc[2*num_runs:3*num_runs, df.columns.get_loc('config')] = 'Fast+Slow, With Priority'

            grouped = df.groupby('config').agg({
                'e2e_p99': ['mean', 'std']
            }).reset_index()

            grouped.columns = ['config', 'e2e_p99_mean', 'e2e_p99_std']

            config_order = ['Only Fast', 'Fast+Slow, No Priority', 'Fast+Slow, With Priority']
            grouped['config'] = pd.Categorical(grouped['config'], categories=config_order, ordered=True)
            grouped = grouped.sort_values('config')

            x = np.arange(len(grouped))
            ax3.bar(x, grouped['e2e_p99_mean'], yerr=grouped['e2e_p99_std'],
                    capsize=5, color=['green', 'orange', 'blue'], alpha=0.7)

            ax3.set_xticks(x)
            ax3.set_xticklabels(grouped['config'])
            ax3.set_ylabel('Fast Kernels E2E P99 (ms)')
            ax3.set_title('(c) Fast Kernel Latency in RT vs BE Scenario')
            ax3.grid(True, alpha=0.3, axis='y')

        # RQ4.4: Jain fairness vs priority pattern
        print("  RQ4.4: Jain fairness vs priority pattern")
        df = self.load_csv('rq4_4_fairness_vs_priority.csv')
        if df is not None:
            num_runs = 3
            patterns = ['All Equal', '1H-7L', '2H-6L', '4H-4L', 'Multi-Level']

            df['pattern'] = 'Unknown'
            total_rows = len(df)

            for i, pattern in enumerate(patterns):
                start = i * num_runs
                end = start + num_runs
                if end <= total_rows:
                    df.iloc[start:end, df.columns.get_loc('pattern')] = pattern

            grouped = df.groupby('pattern').agg({
                'jains_index': ['mean', 'std']
            }).reset_index()

            grouped.columns = ['pattern', 'jains_index_mean', 'jains_index_std']

            grouped['pattern'] = pd.Categorical(grouped['pattern'], categories=patterns, ordered=True)
            grouped = grouped.sort_values('pattern')
            grouped = grouped[grouped['pattern'] != 'Unknown']

            x = np.arange(len(grouped))
            ax4.bar(x, grouped['jains_index_mean'], yerr=grouped['jains_index_std'],
                    capsize=5, alpha=0.7)

            ax4.set_xticks(x)
            ax4.set_xticklabels(grouped['pattern'], rotation=15, ha='right')
            ax4.set_ylabel("Jain's Fairness Index")
            ax4.set_title('(d) Fairness vs Priority Pattern')
            ax4.set_ylim([0, 1.1])
            ax4.grid(True, alpha=0.3, axis='y')
            ax4.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Perfect Fairness')
            ax4.legend()

        fig.suptitle('RQ4: Priority Semantics', fontsize=16, fontweight='bold')
        self.save_figure('rq4_priority')

    # ========================================================================
    # RQ5: Heterogeneity & Load Imbalance
    # ========================================================================

    def analyze_rq5(self, sub_rqs: Optional[List[str]] = None):
        """Analyze RQ5: Heterogeneity and load imbalance - combined figure."""
        print("\n=== Analyzing RQ5: Heterogeneity & Load Imbalance ===")

        # Create a 1x3 subplot figure
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

        # RQ5.1: Jain index vs load imbalance
        print("  RQ5.1: Jain index vs imbalance")
        df = self.load_csv('rq5_1_jain_vs_imbalance.csv')
        if df is not None:
            num_runs = 3
            patterns = ['Balanced', 'Mild', 'Moderate', 'Severe']

            df['pattern'] = 'Unknown'
            total_rows = len(df)

            for i, pattern in enumerate(patterns):
                start = i * num_runs
                end = start + num_runs
                if end <= total_rows:
                    df.iloc[start:end, df.columns.get_loc('pattern')] = pattern

            def calc_cv(detail_str):
                if pd.isna(detail_str) or detail_str == '':
                    return 0
                # Handle both colon-separated (new format) and comma-separated (old format)
                val_str = str(detail_str)
                if ':' in val_str:
                    vals = [float(x) for x in val_str.split(':') if x.strip()]
                else:
                    vals = [float(x) for x in val_str.split(',') if x.strip()]
                if len(vals) > 0:
                    return np.std(vals) / np.mean(vals) if np.mean(vals) > 0 else 0
                return 0

            df['imbalance_cv'] = df.get('kernels_per_stream_detail', '').apply(calc_cv)

            grouped = df.groupby('pattern').agg({
                'jains_index': ['mean', 'std'],
                'imbalance_cv': 'mean'
            }).reset_index()

            grouped.columns = ['pattern', 'jains_index_mean', 'jains_index_std', 'imbalance_cv']

            grouped['pattern'] = pd.Categorical(grouped['pattern'], categories=patterns, ordered=True)
            grouped = grouped.sort_values('pattern')
            grouped = grouped[grouped['pattern'] != 'Unknown']

            ax1.plot(grouped['imbalance_cv'], grouped['jains_index_mean'],
                     marker='o', linewidth=2, markersize=10)
            ax1.errorbar(grouped['imbalance_cv'], grouped['jains_index_mean'],
                         yerr=grouped['jains_index_std'], fmt='none', capsize=5)

            for i, row in grouped.iterrows():
                ax1.annotate(row['pattern'], (row['imbalance_cv'], row['jains_index_mean']),
                             xytext=(5, 5), textcoords='offset points')

            ax1.set_xlabel('Load Imbalance (CV of kernels per stream)')
            ax1.set_ylabel("Jain's Fairness Index")
            ax1.set_title('(a) Fairness vs Load Imbalance')
            ax1.grid(True, alpha=0.3)

        # RQ5.2: Per-stream P99 latency (load imbalance)
        print("  RQ5.2: Per-stream P99 (load imbalance)")
        raw_dfs = self.load_raw_csvs('rq5_2_raw_imbalance_*.csv')

        if raw_dfs:
            all_data = pd.concat(raw_dfs, ignore_index=True)

            per_stream = all_data.groupby('stream_id').agg({
                'e2e_latency_ms': lambda x: np.percentile(x, 99)
            }).reset_index()

            per_stream.columns = ['stream_id', 'p99_latency']

            ax2.bar(per_stream['stream_id'], per_stream['p99_latency'], alpha=0.7)

            ax2.set_xlabel('Stream ID')
            ax2.set_ylabel('E2E P99 Latency (ms)')
            ax2.set_title('(b) Per-Stream P99 Latency (Imbalance: 5,10,40,80)')
            ax2.grid(True, alpha=0.3, axis='y')
        else:
            print("    Warning: No raw CSV files found for RQ5.2")

        # RQ5.3: Throughput/concurrency in homogeneous vs heterogeneous
        print("  RQ5.3: Homogeneous vs Heterogeneous")
        df = self.load_csv('rq5_3_homo_vs_hetero.csv')
        if df is not None:
            df['is_hetero'] = df.get('type_detail', '').str.len() > 0

            grouped = df.groupby(['streams', 'is_hetero']).agg({
                'throughput': ['mean', 'std'],
                'concurrent_rate': ['mean', 'std']
            }).reset_index()

            grouped.columns = ['streams', 'is_hetero', 'throughput_mean', 'throughput_std',
                               'concurrent_rate_mean', 'concurrent_rate_std']

            # Plot throughput on ax3
            for is_hetero in [False, True]:
                data = grouped[grouped['is_hetero'] == is_hetero]
                if len(data) > 0:
                    label = 'Heterogeneous' if is_hetero else 'Homogeneous'
                    ax3.plot(data['streams'], data['throughput_mean'],
                             marker='o', label=label, linewidth=2)
                    ax3.fill_between(data['streams'],
                                     data['throughput_mean'] - data['throughput_std'],
                                     data['throughput_mean'] + data['throughput_std'],
                                     alpha=0.2)

            ax3.set_xlabel('Number of Streams')
            ax3.set_ylabel('Throughput (kernels/sec)')
            ax3.set_title('(c) Throughput: Homogeneous vs Heterogeneous')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_xscale('log', base=2)

        fig.suptitle('RQ5: Heterogeneity & Load Imbalance', fontsize=16, fontweight='bold')
        self.save_figure('rq5_heterogeneity')

    # ========================================================================
    # RQ6: Arrival Pattern & Jitter
    # ========================================================================

    def analyze_rq6(self, sub_rqs: Optional[List[str]] = None):
        """Analyze RQ6: Arrival pattern and jitter effects - combined figure."""
        print("\n=== Analyzing RQ6: Arrival Pattern & Jitter ===")

        # Create a 1x2 subplot figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Load data once
        df = self.load_csv('rq6_jitter_effects.csv')
        if df is not None:
            df['has_jitter'] = df.get('seed', 0) != 0
            df['freq_est'] = df['total_kernels'] / (df['wall_time_ms'] / 1000.0) / df['streams']

            freq_bins = [0, 75, 150, 300, 600]
            freq_labels = ['50Hz', '100Hz', '200Hz', '500Hz']
            df['freq_bin'] = pd.cut(df['freq_est'], bins=freq_bins, labels=freq_labels)

            # RQ6.1: concurrent_rate vs jitter
            print("  RQ6.1: Concurrent rate vs jitter")
            grouped = df.groupby(['freq_bin', 'has_jitter']).agg({
                'concurrent_rate': ['mean', 'std']
            }).reset_index()

            grouped.columns = ['freq_bin', 'has_jitter', 'concurrent_rate_mean', 'concurrent_rate_std']

            freq_bins_unique = grouped['freq_bin'].unique()
            x = np.arange(len(freq_bins_unique))
            width = 0.35

            no_jitter = grouped[grouped['has_jitter'] == False]
            with_jitter = grouped[grouped['has_jitter'] == True]

            if len(no_jitter) > 0:
                ax1.bar(x - width/2, no_jitter['concurrent_rate_mean'], width,
                        yerr=no_jitter['concurrent_rate_std'],
                        label='No Jitter', capsize=5, alpha=0.7)
            if len(with_jitter) > 0:
                ax1.bar(x + width/2, with_jitter['concurrent_rate_mean'], width,
                        yerr=with_jitter['concurrent_rate_std'],
                        label='With Jitter', capsize=5, alpha=0.7)

            ax1.set_xlabel('Launch Frequency')
            ax1.set_ylabel('Concurrent Rate (%)')
            ax1.set_title('(a) Concurrent Rate vs Jitter')
            ax1.set_xticks(x)
            ax1.set_xticklabels(freq_bins_unique)
            ax1.legend()
            ax1.grid(True, alpha=0.3, axis='y')

            # RQ6.2: E2E P99 vs jitter
            print("  RQ6.2: E2E P99 vs jitter")
            grouped = df.groupby(['freq_bin', 'has_jitter']).agg({
                'e2e_p99': ['mean', 'std']
            }).reset_index()

            grouped.columns = ['freq_bin', 'has_jitter', 'e2e_p99_mean', 'e2e_p99_std']

            no_jitter = grouped[grouped['has_jitter'] == False]
            with_jitter = grouped[grouped['has_jitter'] == True]

            if len(no_jitter) > 0:
                ax2.bar(x - width/2, no_jitter['e2e_p99_mean'], width,
                        yerr=no_jitter['e2e_p99_std'],
                        label='No Jitter (Periodic)', capsize=5, alpha=0.7)
            if len(with_jitter) > 0:
                ax2.bar(x + width/2, with_jitter['e2e_p99_mean'], width,
                        yerr=with_jitter['e2e_p99_std'],
                        label='With Jitter (Random)', capsize=5, alpha=0.7)

            ax2.set_xlabel('Launch Frequency')
            ax2.set_ylabel('E2E P99 Latency (ms)')
            ax2.set_title('(b) E2E P99 Latency vs Jitter')
            ax2.set_xticks(x)
            ax2.set_xticklabels(freq_bins_unique)
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')

        fig.suptitle('RQ6: Arrival Pattern & Jitter', fontsize=16, fontweight='bold')
        self.save_figure('rq6_jitter')

    # ========================================================================
    # RQ7: Working Set vs L2 Cache
    # ========================================================================

    def analyze_rq7(self, sub_rqs: Optional[List[str]] = None):
        """Analyze RQ7: Memory working set vs L2 cache boundary - combined figure."""
        print("\n=== Analyzing RQ7: Working Set vs L2 Cache ===")

        # RQ7.1: throughput/util vs working_set/L2 ratio
        print("  RQ7.1: Throughput/util vs working set")

        df = self.load_csv('rq7_working_set_vs_l2.csv')
        if df is not None:
            # Get L2 size from first row that has it, or assume 6MB
            l2_size_mb = 6.0  # Default
            if 'l2_cache_mb' in df.columns:
                l2_vals = df['l2_cache_mb'].dropna()
                if len(l2_vals) > 0:
                    l2_size_mb = l2_vals.iloc[0]

            df['ws_l2_ratio'] = df['working_set_mb'] / l2_size_mb

            grouped = df.groupby(['ws_l2_ratio', 'type']).agg({
                'throughput': ['mean', 'std'],
                'util': ['mean', 'std']
            }).reset_index()

            grouped.columns = ['ws_l2_ratio', 'type', 'throughput_mean', 'throughput_std',
                               'util_mean', 'util_std']

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            # Throughput
            for ktype in ['memory', 'mixed', 'compute']:
                data = grouped[grouped['type'] == ktype].sort_values('ws_l2_ratio')
                if len(data) > 0:
                    ax1.plot(data['ws_l2_ratio'], data['throughput_mean'],
                             marker='o', label=ktype.upper(), linewidth=2)
                    ax1.fill_between(data['ws_l2_ratio'],
                                     data['throughput_mean'] - data['throughput_std'],
                                     data['throughput_mean'] + data['throughput_std'],
                                     alpha=0.2)

            ax1.axvline(x=1.0, color='r', linestyle='--', alpha=0.5, label='L2 Boundary')
            ax1.set_xlabel('Working Set / L2 Cache Ratio')
            ax1.set_ylabel('Throughput (kernels/sec)')
            ax1.set_title('(a) Throughput vs Working Set / L2 Ratio')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Utilization
            for ktype in ['memory', 'mixed', 'compute']:
                data = grouped[grouped['type'] == ktype].sort_values('ws_l2_ratio')
                if len(data) > 0:
                    ax2.plot(data['ws_l2_ratio'], data['util_mean'],
                             marker='o', label=ktype.upper(), linewidth=2)
                    ax2.fill_between(data['ws_l2_ratio'],
                                     data['util_mean'] - data['util_std'],
                                     data['util_mean'] + data['util_std'],
                                     alpha=0.2)

            ax2.axvline(x=1.0, color='r', linestyle='--', alpha=0.5, label='L2 Boundary')
            ax2.set_xlabel('Working Set / L2 Cache Ratio')
            ax2.set_ylabel('GPU Utilization (%)')
            ax2.set_title('(b) Utilization vs Working Set / L2 Ratio')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            fig.suptitle('RQ7: Working Set vs L2 Cache', fontsize=16, fontweight='bold')
            self.save_figure('rq7_memory')
        else:
            print("    Warning: No data found for RQ7")

    # ========================================================================
    # RQ8: Multi-Process vs Single-Process
    # ========================================================================

    def analyze_rq8(self, sub_rqs: Optional[List[str]] = None):
        """
        Analyze RQ8: Multi-process vs single-process fairness and throughput.

        Sub-RQ8.1: Fairness and throughput comparison across process configurations
        """
        print("\n=== Analyzing RQ8: Multi-Process vs Single-Process ===")

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

        # Load aggregated data
        df = self.load_csv('rq8_multiprocess.csv')
        if df is not None and len(df) > 0:
            print("  RQ8.1: Fairness and throughput vs process configuration")

            # Parse configuration from data
            # We need to infer the configuration from streams count
            # Single process: 32 streams
            # 4 processes: 8 streams each
            # 8 processes: 4 streams each
            # 16 processes: 2 streams each

            def infer_config(row):
                s = row['streams']
                if s == 32:
                    return '1×32'
                elif s == 8:
                    return '4×8'
                elif s == 4:
                    return '8×4'
                elif s == 2:
                    return '16×2'
                else:
                    return f'{s} streams'

            df['config'] = df.apply(infer_config, axis=1)

            # Group by configuration
            grouped = df.groupby('config').agg({
                'throughput': ['mean', 'std'],
                'jains_index': ['mean', 'std'],
                'concurrent_rate': ['mean', 'std'],
                'e2e_p99': ['mean', 'std'],
            }).reset_index()

            grouped.columns = ['config', 'throughput_mean', 'throughput_std',
                             'jains_mean', 'jains_std',
                             'concurrent_rate_mean', 'concurrent_rate_std',
                             'e2e_p99_mean', 'e2e_p99_std']

            # Sort by configuration order
            config_order = ['1×32', '4×8', '8×4', '16×2']
            grouped['config'] = pd.Categorical(grouped['config'], categories=config_order, ordered=True)
            grouped = grouped.sort_values('config')

            x = np.arange(len(grouped))
            width = 0.6

            # Subplot 1: Throughput
            ax1.bar(x, grouped['throughput_mean'], width,
                   yerr=grouped['throughput_std'],
                   capsize=5, alpha=0.7, color='steelblue')
            ax1.set_xlabel('Configuration (processes × streams/proc)')
            ax1.set_ylabel('Throughput (kernels/sec)')
            ax1.set_title('(a) Throughput vs Process Configuration')
            ax1.set_xticks(x)
            ax1.set_xticklabels(grouped['config'])
            ax1.grid(True, alpha=0.3, axis='y')

            # Subplot 2: Jain's Fairness Index
            ax2.bar(x, grouped['jains_mean'], width,
                   yerr=grouped['jains_std'],
                   capsize=5, alpha=0.7, color='seagreen')
            ax2.set_xlabel('Configuration (processes × streams/proc)')
            ax2.set_ylabel("Jain's Fairness Index")
            ax2.set_title('(b) Fairness vs Process Configuration')
            ax2.set_xticks(x)
            ax2.set_xticklabels(grouped['config'])
            ax2.set_ylim([0, 1.1])
            ax2.grid(True, alpha=0.3, axis='y')

            # Subplot 3: P99 Latency
            ax3.bar(x, grouped['e2e_p99_mean'], width,
                   yerr=grouped['e2e_p99_std'],
                   capsize=5, alpha=0.7, color='coral')
            ax3.set_xlabel('Configuration (processes × streams/proc)')
            ax3.set_ylabel('E2E P99 Latency (ms)')
            ax3.set_title('(c) P99 Latency vs Process Configuration')
            ax3.set_xticks(x)
            ax3.set_xticklabels(grouped['config'])
            ax3.grid(True, alpha=0.3, axis='y')

            fig.suptitle('RQ8: Multi-Process vs Single-Process Scheduling', fontsize=16, fontweight='bold')
            self.save_figure('rq8_multiprocess')
        else:
            print("    Warning: No data found for RQ8")

    # ========================================================================
    # Main analysis driver
    # ========================================================================

    def analyze_all(self):
        """Run all RQ4-RQ8 analyses."""
        self.analyze_rq4()
        self.analyze_rq5()
        self.analyze_rq6()
        self.analyze_rq7()
        self.analyze_rq8()
