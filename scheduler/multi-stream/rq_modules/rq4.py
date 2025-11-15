"""
RQ4: Priority Semantics

Experiments:
- RQ4.1: Inversion rate with/without priority across different stream counts
- RQ4.2: Per-priority P99 latency vs offered load
- RQ4.3: Fast kernel latency in RT vs BE scenario (only fast, fast+slow no-prio, fast+slow with-prio)
- RQ4.4: Jain fairness index vs priority pattern (all equal, 1H-7L, 2H-6L, 4H-4L, multi-level)
"""

import time
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
        """RQ4.3: Fast kernels in RT vs BE scenario."""
        print("  RQ4.3: Fast kernels in RT vs BE scenario")

        csv_lines = []

        # Config 1: Only fast (baseline)
        print("    Config: Only fast kernels")
        for run_idx in range(self.num_runs):
            args = [
                '--streams', '4',
                '--kernels', '40',
                '--size', '65536',  # Small/fast
                '--type', 'compute',
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

        # Config 2: Fast + Slow, no priority
        print("    Config: Fast + Slow, no priority")
        for run_idx in range(self.num_runs):
            # 4 streams: 2 fast (small kernels), 2 slow (large kernels)
            # Use per-stream-sizes to create true duration difference
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
            ]

            csv_output = self.run_benchmark(args, first_run=False)

            if csv_output:
                lines = csv_output.split('\n')
                csv_lines.extend([l for l in lines if not l.startswith('streams,')])

            time.sleep(0.5)

        # Config 3: Fast + Slow, with priority
        print("    Config: Fast + Slow, with priority")
        for run_idx in range(self.num_runs):
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
            ]

            csv_output = self.run_benchmark(args, first_run=False)

            if csv_output:
                lines = csv_output.split('\n')
                csv_lines.extend([l for l in lines if not l.startswith('streams,')])

            time.sleep(0.5)

        self.save_csv(csv_lines, 'rq4_3_fast_kernels_rt_be.csv')

    def _run_rq4_4(self):
        """RQ4.4: Jain fairness vs priority pattern."""
        print("  RQ4.4: Jain fairness vs priority pattern")

        csv_lines = []
        num_streams = 8

        priority_patterns = [
            ('all_equal', '0,0,0,0,0,0,0,0'),
            ('1high_7low', '-5,0,0,0,0,0,0,0'),
            ('2high_6low', '-5,-5,0,0,0,0,0,0'),
            ('4high_4low', '-5,-5,-5,-5,0,0,0,0'),
            ('multi_level', '-10,-8,-5,-3,0,0,0,0'),
        ]

        for pattern_name, prio_spec in priority_patterns:
            print(f"    pattern={pattern_name}")

            for run_idx in range(self.num_runs):
                args = [
                    '--streams', str(num_streams),
                    '--kernels', '30',
                    '--size', '1048576',
                    '--type', 'mixed',
                    '--priority', prio_spec,
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

        self.save_csv(csv_lines, 'rq4_4_fairness_vs_priority.csv')

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

        # Detect priority enabled from per_priority_avg column (contains ':' when priority is used)
        df['priority_enabled'] = df['per_priority_avg'].astype(str).str.contains(':', na=False)

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

        # Group by launch frequency
        grouped = df.groupby('launch_freq').agg({
            'per_priority_p99': lambda x: list(x)
        }).reset_index()

        # Extract high-priority and low-priority p99 from per_priority_p99 list
        high_p99_list = []
        low_p99_list = []

        for _, row in grouped.iterrows():
            p99_lists = row['per_priority_p99']
            high_vals = []
            low_vals = []

            for p99_str in p99_lists:
                parts = p99_str.split(';')
                if len(parts) >= 2:
                    # Format: "prio:-5,p99:X.XX;prio:0,p99:Y.YY"
                    for part in parts:
                        if 'prio:-5' in part:
                            high_vals.append(float(part.split('p99:')[1]))
                        elif 'prio:0,' in part:
                            low_vals.append(float(part.split('p99:')[1]))

            high_p99_list.append({
                'launch_freq': row['launch_freq'],
                'mean': np.mean(high_vals) if high_vals else 0,
                'std': np.std(high_vals) if high_vals else 0
            })
            low_p99_list.append({
                'launch_freq': row['launch_freq'],
                'mean': np.mean(low_vals) if low_vals else 0,
                'std': np.std(low_vals) if low_vals else 0
            })

        high_df = pd.DataFrame(high_p99_list)
        low_df = pd.DataFrame(low_p99_list)

        # Plot
        ax.plot(high_df['launch_freq'], high_df['mean'],
                marker='o', linewidth=2, label='High Priority', color='blue')
        ax.fill_between(high_df['launch_freq'],
                        high_df['mean'] - high_df['std'],
                        high_df['mean'] + high_df['std'],
                        alpha=0.3, color='blue')

        ax.plot(low_df['launch_freq'], low_df['mean'],
                marker='s', linewidth=2, label='Low Priority', color='red')
        ax.fill_between(low_df['launch_freq'],
                        low_df['mean'] - low_df['std'],
                        low_df['mean'] + low_df['std'],
                        alpha=0.3, color='red')

        ax.set_xlabel('Launch Frequency (Hz)')
        ax.set_ylabel('P99 Latency (ms)')
        ax.set_title('(b) Per-Priority P99 Latency vs Launch Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_rq4_3(self, ax):
        """Plot RQ4.3: Fast kernel P99 in RT vs BE."""
        print("  RQ4.3: Fast kernel P99 in RT vs BE scenario")
        df = self.load_csv('rq4_3_fast_kernels_rt_be.csv')

        if df is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('(c) Fast Kernel P99 at Mixed Duration')
            return

        # Infer configuration from type_detail and per_priority_p99
        df['config'] = 'Unknown'
        df['has_priority'] = df['per_priority_p99'].astype(str).str.contains(':', na=False)
        df['is_mixed'] = df['type_detail'].astype(str).str.contains(':', na=False)

        # Config 1: uniform type (all fast, no mixing)
        # Config 2: mixed types, no priority (no ':' in per_priority_p99)
        # Config 3: mixed types, with priority (':' in per_priority_p99)

        for idx, row in df.iterrows():
            if not row['is_mixed']:
                df.at[idx, 'config'] = 'Only Fast'
            elif not row['has_priority']:
                df.at[idx, 'config'] = 'Fast+Slow, No Priority'
            else:
                df.at[idx, 'config'] = 'Fast+Slow, Priority'

        # Group by config and calculate mean/std of overall P99
        grouped = df.groupby('config').agg({
            'svc_p99': ['mean', 'std']
        }).reset_index()

        grouped.columns = ['config', 'p99_mean', 'p99_std']

        # Ensure correct ordering
        config_order = ['Only Fast', 'Fast+Slow, No Priority', 'Fast+Slow, Priority']
        grouped['order'] = grouped['config'].map({c: i for i, c in enumerate(config_order)})
        grouped = grouped.sort_values('order')

        # Plot bar chart
        x_pos = np.arange(len(grouped))
        colors = ['green', 'orange', 'blue']
        # Only use yerr if we have valid std values
        yerr = grouped['p99_std'].fillna(0) if len(grouped) > 0 else None
        ax.bar(x_pos, grouped['p99_mean'], yerr=yerr,
               capsize=5, alpha=0.7, color=[colors[int(o)] for o in grouped['order']])

        ax.set_xticks(x_pos)
        ax.set_xticklabels(grouped['config'], rotation=15, ha='right')
        ax.set_ylabel('P99 Latency (ms)')
        ax.set_title('(c) Fast Kernel P99 at Mixed Duration')
        ax.grid(True, alpha=0.3, axis='y')

    def _plot_rq4_4(self, ax):
        """Plot RQ4.4: Jain fairness vs priority pattern."""
        print("  RQ4.4: Jain fairness vs priority pattern")
        df = self.load_csv('rq4_4_fairness_vs_priority.csv')

        if df is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('(d) Fairness vs Priority Pattern')
            return

        # Count priority levels from per_priority_p99
        def count_priority_levels(s):
            """Count number of unique priorities in per_priority_p99 string."""
            if pd.isna(s) or s == '':
                return 0
            # Format: "prio:X,p99:Y;prio:Z,p99:W;..."
            parts = s.split(';')
            return len(parts)

        df['num_priority_levels'] = df['per_priority_p99'].apply(count_priority_levels)

        # Assign patterns based on priority levels
        patterns = ['All Equal', '1H-7L', '2H-6L', '4H-4L', 'Multi-Level']
        df['pattern'] = 'Unknown'

        # Pattern assignment logic
        one_level_rows = df[df['num_priority_levels'] == 1]
        two_level_rows = df[df['num_priority_levels'] == 2]
        five_level_rows = df[df['num_priority_levels'] == 5]

        # All Equal: 1 level
        df.loc[one_level_rows.index, 'pattern'] = 'All Equal'

        # Multi-Level: 5 levels
        df.loc[five_level_rows.index, 'pattern'] = 'Multi-Level'

        # For 2-level patterns, assign in order (1H-7L, 2H-6L, 4H-4L)
        if len(two_level_rows) > 0:
            two_level_indices = sorted(two_level_rows.index.tolist())
            n_per_pattern = len(two_level_indices) // 3
            if n_per_pattern > 0:
                df.loc[two_level_indices[:n_per_pattern], 'pattern'] = '1H-7L'
                df.loc[two_level_indices[n_per_pattern:2*n_per_pattern], 'pattern'] = '2H-6L'
                df.loc[two_level_indices[2*n_per_pattern:], 'pattern'] = '4H-4L'

        # Group by pattern and calculate mean/std
        grouped = df.groupby('pattern').agg({
            'jains_index': ['mean', 'std']
        }).reset_index()

        grouped.columns = ['pattern', 'fairness_mean', 'fairness_std']

        # Ensure proper ordering
        pattern_order = {p: i for i, p in enumerate(patterns)}
        grouped['order'] = grouped['pattern'].map(pattern_order)
        grouped = grouped.sort_values('order')

        # Plot
        x_pos = np.arange(len(grouped))
        # Only use yerr if we have valid std values
        yerr = grouped['fairness_std'].fillna(0) if len(grouped) > 0 else None
        ax.bar(x_pos, grouped['fairness_mean'], yerr=yerr,
               capsize=5, alpha=0.7, color='steelblue')

        ax.set_xticks(x_pos)
        ax.set_xticklabels(grouped['pattern'], rotation=15, ha='right')
        ax.set_ylabel('Jain Fairness Index')
        ax.set_title('(d) Fairness vs Priority Pattern')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
