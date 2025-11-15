"""
RQ4: Priority Semantics

Experiments:
- RQ4.1: Inversion rate with/without priority across different stream counts
- RQ4.2: Per-priority P99 latency vs offered load
- RQ4.3: Fast kernel latency in RT vs BE scenario (only fast, fast+slow no-prio, fast+slow with-prio)
- RQ4.4: Jain fairness index vs priority pattern (all equal, 1H-7L, 2H-6L, 4H-4L, multi-level)
"""

import time
import glob
import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from .base import RQBase


class RQ4(RQBase):
    """RQ4: Priority Semantics - 4 original plots."""

    def run_experiments(self):
        """Run RQ4.1-RQ4.4 experiments."""
        print("\n=== Running RQ4 Experiments: Priority Semantics ===")

        self._run_rq4_1()
        self._run_rq4_2()
        self._run_rq4_3()
        self._run_rq4_4()

        print("\n=== RQ4 Experiments Complete ===")

    def _run_rq4_1(self):
        """RQ4.1: Inversion rate vs streams."""
        print("  RQ4.1: Inversion rate vs streams")
        stream_counts = [4, 8, 16, 32]

        csv_lines = []

        for streams in stream_counts:
            # Baseline: no priority (all 0)
            print(f"    streams={streams}, no priority")
            for run_idx in range(self.num_runs):
                args = [
                    '--streams', str(streams),
                    '--kernels', '30',
                    '--size', '1048576',
                    '--type', 'mixed',
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

            # With priority: half high, half low
            print(f"    streams={streams}, with priority")
            high_prio = -5  # Higher priority (more negative)
            low_prio = 0    # Lower priority
            half = streams // 2
            prio_spec = ','.join([str(high_prio)] * half + [str(low_prio)] * (streams - half))

            for run_idx in range(self.num_runs):
                args = [
                    '--streams', str(streams),
                    '--kernels', '30',
                    '--size', '1048576',
                    '--type', 'mixed',
                    '--priority', prio_spec,
                ]

                csv_output = self.run_benchmark(
                    args,
                    first_run=False  # Header already added
                )

                if csv_output:
                    lines = csv_output.split('\n')
                    csv_lines.extend([l for l in lines if not l.startswith('streams,')])

                time.sleep(0.5)

        self.save_csv(csv_lines, 'rq4_1_inversion_rate.csv')

    def _run_rq4_2(self):
        """RQ4.2: Per-priority P99 vs offered load."""
        print("  RQ4.2: Per-priority P99 vs offered load")
        frequencies = [20, 50, 100, 200, 500, 1000]
        num_streams = 8
        high_prio = -5
        low_prio = 0
        half = num_streams // 2
        prio_spec = ','.join([str(high_prio)] * half + [str(low_prio)] * (num_streams - half))

        csv_lines = []

        for freq in frequencies:
            print(f"    freq={freq}Hz")
            freq_spec = ','.join([str(freq)] * num_streams)

            for run_idx in range(self.num_runs):
                args = [
                    '--streams', str(num_streams),
                    '--kernels', '50',
                    '--size', '524288',
                    '--type', 'mixed',
                    '--priority', prio_spec,
                    '--launch-frequency', freq_spec,
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

        self.save_csv(csv_lines, 'rq4_2_per_priority_vs_load.csv')

    def _run_rq4_3(self):
        """
        RQ4.3: Fast kernels in RT vs BE scenario.

        KEY FIX: Use raw per-kernel CSV to measure ONLY fast kernels' P99,
        not the whole workload's P99 (which would be dominated by slow kernels).
        """
        print("  RQ4.3: Fast kernels in RT vs BE scenario")

        # Config 1: Only fast (baseline)
        print("    Config: Only fast kernels")
        for run_idx in range(self.num_runs):
            raw_csv_file = self.output_dir / f'rq4_3_raw_only_fast_run{run_idx}.csv'

            args = [
                '--streams', '4',
                '--kernels', '40',
                '--size', '65536',  # Small/fast
                '--type', 'compute',
                '--csv-output', str(raw_csv_file),
                '--no-header',
            ]

            result = subprocess.run(
                [self.bench_path] + args,
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode != 0:
                print(f"      Warning: Run {run_idx} failed: {result.stderr}")

            time.sleep(0.5)

        # Config 2: Fast + Slow, no priority
        print("    Config: Fast + Slow, no priority")
        for run_idx in range(self.num_runs):
            raw_csv_file = self.output_dir / f'rq4_3_raw_mixed_noprio_run{run_idx}.csv'

            # 4 streams: 2 fast (stream 0,1), 2 slow (stream 2,3)
            heterogeneous = 'compute,compute,memory,memory'
            load_imbalance = '40,40,10,10'  # Fast streams have more kernels
            sizes = '65536,65536,4194304,4194304'  # Fast: small, Slow: large

            args = [
                '--streams', '4',
                '--kernels', '40',  # Base, modified by load-imbalance
                '--type', 'mixed',
                '--heterogeneous', heterogeneous,
                '--load-imbalance', load_imbalance,
                '--per-stream-sizes', sizes,
                '--csv-output', str(raw_csv_file),
                '--no-header',
            ]

            result = subprocess.run(
                [self.bench_path] + args,
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode != 0:
                print(f"      Warning: Run {run_idx} failed: {result.stderr}")

            time.sleep(0.5)

        # Config 3: Fast + Slow, with priority
        print("    Config: Fast + Slow, with priority")
        for run_idx in range(self.num_runs):
            raw_csv_file = self.output_dir / f'rq4_3_raw_mixed_prio_run{run_idx}.csv'

            heterogeneous = 'compute,compute,memory,memory'
            load_imbalance = '40,40,10,10'
            sizes = '65536,65536,4194304,4194304'  # Fast: small, Slow: large
            priority = '-5,-5,0,0'  # Fast streams get high priority

            args = [
                '--streams', '4',
                '--kernels', '40',
                '--type', 'mixed',
                '--heterogeneous', heterogeneous,
                '--load-imbalance', load_imbalance,
                '--per-stream-sizes', sizes,
                '--priority', priority,
                '--csv-output', str(raw_csv_file),
                '--no-header',
            ]

            result = subprocess.run(
                [self.bench_path] + args,
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode != 0:
                print(f"      Warning: Run {run_idx} failed: {result.stderr}")

            time.sleep(0.5)

        # Aggregate: Extract fast kernels' P99 from raw CSVs
        print("    Aggregating fast kernels' P99 from raw CSV files...")
        self._aggregate_rq4_3_data()

    def _aggregate_rq4_3_data(self):
        """
        Aggregate RQ4.3 raw CSV files and compute P99 for FAST KERNELS ONLY.

        Fast kernels are identified by stream_id in {0, 1} (first 2 streams).
        This ensures we measure RT kernels' tail latency, not the whole workload.
        """
        aggregated_data = []

        configs = [
            ('only_fast', 'Only Fast'),
            ('mixed_noprio', 'Fast+Slow, No Priority'),
            ('mixed_prio', 'Fast+Slow, Priority'),
        ]

        for cfg_name, cfg_label in configs:
            pattern = str(self.output_dir / f'rq4_3_raw_{cfg_name}_run*.csv')
            raw_files = glob.glob(pattern)

            if not raw_files:
                print(f"      Warning: No files found for {cfg_name}")
                continue

            for raw_file in raw_files:
                if not os.path.exists(raw_file):
                    continue

                df = pd.read_csv(raw_file)

                # Filter for FAST kernels only (stream 0 and 1)
                # In "only_fast" config, all streams are fast
                # In mixed configs, streams 0-1 are fast (small kernels)
                if cfg_name == 'only_fast':
                    fast = df.copy()
                else:
                    fast = df[df['stream_id'].isin([0, 1])].copy()

                if len(fast) == 0:
                    continue

                # Calculate metrics on FAST kernels only
                e2e_p99 = fast['e2e_latency_ms'].quantile(0.99)
                e2e_mean = fast['e2e_latency_ms'].mean()
                e2e_p50 = fast['e2e_latency_ms'].quantile(0.50)
                num_fast = len(fast)

                aggregated_data.append({
                    'config': cfg_label,
                    'e2e_p99': e2e_p99,
                    'e2e_mean': e2e_mean,
                    'e2e_p50': e2e_p50,
                    'num_fast_kernels': num_fast,
                })

        if aggregated_data:
            agg_df = pd.DataFrame(aggregated_data)

            # Group by config and compute mean/std across runs
            final_df = agg_df.groupby('config').agg({
                'e2e_p99': ['mean', 'std'],
                'e2e_mean': ['mean', 'std'],
                'e2e_p50': ['mean', 'std'],
                'num_fast_kernels': 'sum',
            }).reset_index()

            # Flatten column names
            final_df.columns = ['_'.join(col).strip('_') if col[1] else col[0]
                               for col in final_df.columns.values]

            csv_path = self.results_dir / 'rq4_3_fast_only_p99.csv'
            final_df.to_csv(csv_path, index=False)
            print(f"    Saved: {csv_path}")

    def _run_rq4_4(self):
        """
        RQ4.4: Jain fairness vs priority pattern.

        FIX: Store pattern name with each run to avoid hardcoded num_runs mapping.
        """
        print("  RQ4.4: Jain fairness vs priority pattern")

        num_streams = 8

        priority_patterns = [
            ('all_equal', 'All Equal', '0,0,0,0,0,0,0,0'),
            ('1high_7low', '1H-7L', '-5,0,0,0,0,0,0,0'),
            ('2high_6low', '2H-6L', '-5,-5,0,0,0,0,0,0'),
            ('4high_4low', '4H-4L', '-5,-5,-5,-5,0,0,0,0'),
            ('multi_level', 'Multi-Level', '-10,-8,-5,-3,0,0,0,0'),
        ]

        aggregated_data = []

        for pattern_key, pattern_label, prio_spec in priority_patterns:
            print(f"    pattern={pattern_label}")

            for run_idx in range(self.num_runs):
                args = [
                    '--streams', str(num_streams),
                    '--kernels', '30',
                    '--size', '1048576',
                    '--type', 'mixed',
                    '--priority', prio_spec,
                ]

                csv_output = self.run_benchmark(args, first_run=False)

                if csv_output:
                    # Parse the aggregate CSV line
                    lines = [l for l in csv_output.split('\n') if l.strip() and not l.startswith('streams,')]
                    if lines:
                        # Parse first data line (should only be one for aggregate output)
                        parts = lines[0].split(',')
                        if len(parts) > 0:
                            # Extract jains_index from CSV (column index depends on CSV format)
                            # Safer: use load_csv on this one line
                            header = csv_output.split('\n')[0]
                            full_csv = header + '\n' + lines[0]
                            import io
                            df_run = pd.read_csv(io.StringIO(full_csv))

                            if 'jains_index' in df_run.columns:
                                jains = df_run['jains_index'].values[0]
                                aggregated_data.append({
                                    'pattern': pattern_label,
                                    'pattern_key': pattern_key,
                                    'jains_index': jains,
                                    'run_idx': run_idx,
                                })

                time.sleep(0.5)

        # Save to CSV with pattern column
        if aggregated_data:
            agg_df = pd.DataFrame(aggregated_data)

            # Group by pattern and calculate mean/std
            final_df = agg_df.groupby('pattern').agg({
                'jains_index': ['mean', 'std'],
            }).reset_index()

            final_df.columns = ['pattern', 'jains_index_mean', 'jains_index_std']

            csv_path = self.results_dir / 'rq4_4_fairness_vs_priority.csv'
            final_df.to_csv(csv_path, index=False)
            print(f"    Saved: {csv_path}")

    def analyze(self):
        """Generate RQ4 analysis figure (2x2 layout)."""
        print("\n=== Analyzing RQ4: Priority Semantics ===")

        # Create a 2x2 subplot figure (4 subplots)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        self._plot_rq4_1(ax1)
        self._plot_rq4_2(ax2)
        self._plot_rq4_3(ax3)
        self._plot_rq4_4(ax4)

        self.save_figure('rq4_priority')
        print("=== RQ4 Analysis Complete ===")

    def _plot_rq4_1(self, ax):
        """Plot RQ4.1: Inversion rate vs streams."""
        print("  RQ4.1: Inversion rate vs streams")
        df = self.load_csv('rq4_1_inversion_rate.csv')

        if df is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('(a) Priority Inversion Rate vs Stream Count')
            return

        # Detect priority enabled: check if per_priority_avg has meaningful content
        # More robust than just checking for ':' - also handles ',' and other formats
        df['priority_enabled'] = df['per_priority_avg'].notna() & (df['per_priority_avg'].astype(str).str.strip() != '')

        # Group by streams and priority, calculate mean and std
        grouped = df.groupby(['streams', 'priority_enabled']).agg({
            'inversion_rate': ['mean', 'std']
        }).reset_index()

        grouped.columns = ['streams', 'priority_enabled', 'inversion_rate_mean', 'inversion_rate_std']

        # Split by priority
        no_prio = grouped[grouped['priority_enabled'] == False]
        with_prio = grouped[grouped['priority_enabled'] == True]

        # Plot
        ax.plot(no_prio['streams'], no_prio['inversion_rate_mean'],
                marker='o', linewidth=2, label='Without Priority', color='red')
        ax.fill_between(no_prio['streams'],
                        no_prio['inversion_rate_mean'] - no_prio['inversion_rate_std'],
                        no_prio['inversion_rate_mean'] + no_prio['inversion_rate_std'],
                        alpha=0.3, color='red')

        ax.plot(with_prio['streams'], with_prio['inversion_rate_mean'],
                marker='s', linewidth=2, label='With Priority', color='blue')
        ax.fill_between(with_prio['streams'],
                        with_prio['inversion_rate_mean'] - with_prio['inversion_rate_std'],
                        with_prio['inversion_rate_mean'] + with_prio['inversion_rate_std'],
                        alpha=0.3, color='blue')

        ax.set_xlabel('Number of Streams')
        ax.set_ylabel('Inversion Rate')
        ax.set_title('(a) Priority Inversion Rate vs Stream Count')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    def _plot_rq4_2(self, ax):
        """Plot RQ4.2: Per-priority P99 vs load."""
        print("  RQ4.2: Per-priority P99 vs offered load")
        df = self.load_csv('rq4_2_per_priority_vs_load.csv')

        if df is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('(b) Per-Priority P99 Latency vs Launch Frequency')
            return

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
        if len(df_sorted) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('(b) Per-Priority P99 Latency vs Launch Frequency')
            return

        # Group by launch frequency
        grouped = df_sorted.groupby('launch_freq').agg({
            'high_prio_p99': ['mean', 'std'],
            'low_prio_p99': ['mean', 'std']
        }).reset_index()

        grouped.columns = ['launch_freq', 'high_p99_mean', 'high_p99_std',
                           'low_p99_mean', 'low_p99_std']

        # Sort by launch frequency
        grouped = grouped.sort_values('launch_freq')

        # Plot
        ax.plot(grouped['launch_freq'], grouped['high_p99_mean'],
                marker='o', linewidth=2, label='High Priority', color='blue')
        ax.fill_between(grouped['launch_freq'],
                        grouped['high_p99_mean'] - grouped['high_p99_std'],
                        grouped['high_p99_mean'] + grouped['high_p99_std'],
                        alpha=0.3, color='blue')

        ax.plot(grouped['launch_freq'], grouped['low_p99_mean'],
                marker='s', linewidth=2, label='Low Priority', color='red')
        ax.fill_between(grouped['launch_freq'],
                        grouped['low_p99_mean'] - grouped['low_p99_std'],
                        grouped['low_p99_mean'] + grouped['low_p99_std'],
                        alpha=0.3, color='red')

        ax.set_xlabel('Launch Frequency (Hz)')
        ax.set_ylabel('P99 Latency (ms)')
        ax.set_title('(b) Per-Priority P99 Latency vs Launch Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_rq4_3(self, ax):
        """Plot RQ4.3: Fast kernel P99 in RT vs BE (REDESIGNED - only fast kernels)."""
        print("  RQ4.3: Fast kernel P99 in RT vs BE scenario")
        df = self.load_csv('rq4_3_fast_only_p99.csv')

        if df is None:
            ax.text(0.5, 0.5, 'No RQ4.3 data\n(run experiments first)',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, color='gray', style='italic')
            ax.set_title('(c) Fast Kernel P99 at Mixed Duration')
            return

        # Ensure correct ordering
        config_order = ['Only Fast', 'Fast+Slow, No Priority', 'Fast+Slow, Priority']
        df['order'] = df['config'].map({c: i for i, c in enumerate(config_order)})
        df = df.sort_values('order')

        # Plot bar chart using FAST KERNELS' P99 ONLY
        x_pos = np.arange(len(df))
        colors = ['green', 'orange', 'blue']

        # Use yerr if we have valid std values
        yerr = df['e2e_p99_std'].fillna(0) if len(df) > 0 and 'e2e_p99_std' in df.columns else None

        ax.bar(x_pos, df['e2e_p99_mean'], yerr=yerr,
               capsize=5, alpha=0.7, color=[colors[int(o)] for o in df['order']])

        ax.set_xticks(x_pos)
        ax.set_xticklabels(df['config'], rotation=15, ha='right')
        ax.set_ylabel('P99 Latency (ms)')
        ax.set_title('(c) Fast Kernel P99 at Mixed Duration')
        ax.grid(True, alpha=0.3, axis='y')

    def _plot_rq4_4(self, ax):
        """Plot RQ4.4: Jain fairness vs priority pattern (FIXED - no hardcoded num_runs)."""
        print("  RQ4.4: Jain fairness vs priority pattern")
        df = self.load_csv('rq4_4_fairness_vs_priority.csv')

        if df is None:
            ax.text(0.5, 0.5, 'No RQ4.4 data\n(run experiments first)',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, color='gray', style='italic')
            ax.set_title('(d) Fairness vs Priority Pattern')
            return

        # Ensure proper ordering (pattern names are already in the CSV)
        patterns = ['All Equal', '1H-7L', '2H-6L', '4H-4L', 'Multi-Level']
        pattern_order = {p: i for i, p in enumerate(patterns)}
        df['order'] = df['pattern'].map(pattern_order)
        df = df.sort_values('order')

        # Plot
        x_pos = np.arange(len(df))
        # Only use yerr if we have valid std values
        yerr = df['jains_index_std'].fillna(0) if len(df) > 0 and 'jains_index_std' in df.columns else None

        ax.bar(x_pos, df['jains_index_mean'], yerr=yerr,
               capsize=5, alpha=0.7, color='steelblue')

        ax.set_xticks(x_pos)
        ax.set_xticklabels(df['pattern'], rotation=15, ha='right')
        ax.set_ylabel('Jain Fairness Index')
        ax.set_title('(d) Fairness vs Priority Pattern')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
