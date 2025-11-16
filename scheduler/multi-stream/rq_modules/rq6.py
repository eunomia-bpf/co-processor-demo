"""RQ6: Heterogeneity & Load Imbalance (formerly RQ5)"""
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from .base import RQBase


class RQ6(RQBase):
    """RQ6: Heterogeneity and load imbalance effects (formerly RQ5)."""

    def run_experiments(self):
        """
        RQ6: Heterogeneity and load imbalance effects.

        Sub-RQ6.1: Jain index vs load imbalance
        Sub-RQ6.2: Per-stream P99 (load imbalance) - needs raw CSV
        Sub-RQ6.3: throughput/concurrency in homogeneous vs heterogeneous
        """
        print("\n=== Running RQ6 Experiments: Heterogeneity & Load Imbalance ===")

        # Sub-RQ6.1: Jain index vs load imbalance
        print("  RQ6.1: Jain index vs load imbalance")

        csv_lines = []
        num_streams = 4

        imbalance_patterns = [
            ('balanced', '20,20,20,20'),
            ('mild', '10,20,30,40'),
            ('moderate', '5,15,30,50'),
            ('severe', '5,10,40,80'),
        ]

        for pattern_name, load_spec in imbalance_patterns:
            print(f"    pattern={pattern_name}")

            for run_idx in range(self.num_runs):
                args = [
                    '--streams', str(num_streams),
                    '--kernels', '20',  # Base value, overridden by load-imbalance
                    '--size', '1048576',
                    '--type', 'mixed',
                    '--load-imbalance', load_spec,
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

        self.save_csv(csv_lines, 'rq6_1_jain_vs_imbalance.csv')

        # Sub-RQ6.2: Per-stream P99 (severe imbalance with raw CSV)
        print("  RQ6.2: Per-stream P99 latency (load imbalance)")

        csv_lines = []
        load_spec = '5,10,40,80'

        for run_idx in range(self.num_runs):
            raw_csv_file = str(self.output_dir / f'rq6_2_raw_imbalance_run{run_idx}.csv')

            args = [
                '--streams', str(num_streams),
                '--kernels', '20',
                '--size', '1048576',
                '--type', 'mixed',
                '--load-imbalance', load_spec,
                '--csv-output', raw_csv_file,
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

        self.save_csv(csv_lines, 'rq6_2_per_stream_imbalance_aggregated.csv')

        # Sub-RQ6.3: Homogeneous vs heterogeneous
        print("  RQ6.3: Throughput/concurrency homogeneous vs heterogeneous")

        csv_lines = []
        stream_counts = [2, 4, 8, 16, 32]

        for streams in stream_counts:
            # Homogeneous: all compute
            print(f"    streams={streams}, homogeneous")
            for run_idx in range(self.num_runs):
                args = [
                    '--streams', str(streams),
                    '--kernels', '20',
                    '--size', '1048576',
                    '--type', 'compute',
                ]

                csv_output = self.run_benchmark(
                    args,
                    first_run=(len(csv_lines) == 0)
                )

                if csv_output:
                    lines = csv_output.split('\n')
                    if len(csv_lines) == 0:
                        # Add 'pattern' column to header
                        header = lines[0] + ',pattern'
                        csv_lines.append(header)
                        data_lines = lines[1:] if len(lines) > 1 else []
                    else:
                        data_lines = [l for l in lines if not l.startswith('streams,')]

                    # Append 'homogeneous' to each data line
                    for line in data_lines:
                        if line.strip():
                            csv_lines.append(line + ',homogeneous')

                time.sleep(0.5)

            # Heterogeneous: mix of types
            print(f"    streams={streams}, heterogeneous")
            # Create heterogeneous pattern: cycle through types (avoid gemm due to CUDA errors)
            types = ['memory', 'compute', 'mixed']
            hetero_spec = ','.join([types[i % len(types)] for i in range(streams)])

            for run_idx in range(self.num_runs):
                args = [
                    '--streams', str(streams),
                    '--kernels', '20',
                    '--size', '1048576',
                    '--type', 'mixed',  # Add explicit type
                    '--heterogeneous', hetero_spec,
                ]

                csv_output = self.run_benchmark(args, first_run=False)

                if csv_output:
                    lines = csv_output.split('\n')
                    data_lines = [l for l in lines if l.strip() and not l.startswith('streams,')]

                    # Append 'heterogeneous' to each data line
                    for line in data_lines:
                        if line.strip():
                            csv_lines.append(line + ',heterogeneous')

                time.sleep(0.5)

        self.save_csv(csv_lines, 'rq6_3_homo_vs_hetero.csv')

    def analyze(self):
        """Analyze RQ6: Heterogeneity and load imbalance - combined figure."""
        print("\n=== Analyzing RQ6: Heterogeneity & Load Imbalance ===")

        # Create a 1x3 subplot figure
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

        # RQ6.1: Jain index vs load imbalance
        print("  RQ6.1: Jain index vs imbalance")
        df = self.load_csv('rq6_1_jain_vs_imbalance.csv')
        if df is not None:
            def calc_cv(detail_str):
                if pd.isna(detail_str) or detail_str == '':
                    return 0
                val_str = str(detail_str)
                if ':' in val_str:
                    vals = [float(x) for x in val_str.split(':') if x.strip()]
                else:
                    vals = [float(x) for x in val_str.split(',') if x.strip()]
                if len(vals) > 0:
                    return np.std(vals) / np.mean(vals) if np.mean(vals) > 0 else 0
                return 0

            df['imbalance_cv'] = df.get('kernels_per_stream_detail', '').apply(calc_cv)

            def classify_pattern(detail_str, cv):
                if cv < 0.05:
                    return 'Balanced'
                elif cv < 0.55:
                    return 'Mild'
                elif cv < 0.75:
                    return 'Moderate'
                else:
                    return 'Severe'

            df['pattern'] = df.apply(lambda row: classify_pattern(
                row.get('kernels_per_stream_detail', ''), row['imbalance_cv']), axis=1)

            grouped = df.groupby('pattern').agg({
                'jains_index': ['mean', 'std', 'count'],
                'imbalance_cv': 'mean'
            }).reset_index()

            grouped.columns = ['pattern', 'jains_index_mean', 'jains_index_std', 'sample_size', 'imbalance_cv']

            expected_patterns = ['Balanced', 'Mild', 'Moderate', 'Severe']
            grouped['pattern'] = pd.Categorical(grouped['pattern'], categories=expected_patterns, ordered=True)
            grouped = grouped.sort_values('pattern')

            ax1.plot(grouped['imbalance_cv'], grouped['jains_index_mean'],
                     marker='o', linewidth=2, markersize=10, color='steelblue')
            # Only use yerr if we have valid std values
            yerr = grouped['jains_index_std'].fillna(0) if len(grouped) > 0 and grouped['jains_index_std'].notna().any() else None
            if yerr is not None:
                ax1.errorbar(grouped['imbalance_cv'], grouped['jains_index_mean'],
                             yerr=yerr, fmt='none', capsize=5, color='steelblue')

            for idx, row in grouped.iterrows():
                y_offset = 10 + (idx * 3) if idx < 2 else -15 - ((idx-2) * 3)
                x_offset = 8
                ax1.annotate(f"{row['pattern']}\n(n={int(row['sample_size'])})",
                             (row['imbalance_cv'], row['jains_index_mean']),
                             xytext=(x_offset, y_offset), textcoords='offset points',
                             fontsize=9, ha='left',
                             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                      edgecolor='steelblue', alpha=0.7),
                             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0',
                                           color='steelblue', lw=1))

            ax1.set_xlabel('Load Imbalance (CV of kernels per stream)')
            ax1.set_ylabel("Jain's Fairness Index")
            ax1.set_title('(a) Fairness vs Load Imbalance')
            ax1.grid(True, alpha=0.3)

        # RQ6.2: Per-stream P99 latency (load imbalance)
        print("  RQ6.2: Per-stream P99 (load imbalance)")
        raw_dfs = self._load_raw_csvs('rq6_2_raw_imbalance_*.csv')

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
            print("    Warning: No raw CSV files found for RQ6.2")

        # RQ6.3: Throughput/concurrency in homogeneous vs heterogeneous
        print("  RQ6.3: Homogeneous vs Heterogeneous")
        df = self.load_csv('rq6_3_homo_vs_hetero.csv')
        if df is not None:
            # Use the 'pattern' column directly instead of heuristics
            if 'pattern' not in df.columns:
                print("    Warning: 'pattern' column not found, using fallback heuristic")
                df['pattern'] = df.get('type_detail', '').apply(
                    lambda x: 'heterogeneous' if ':' in str(x) else 'homogeneous')

            grouped = df.groupby(['streams', 'pattern']).agg({
                'throughput': ['mean', 'std', 'count'],
                'concurrent_rate': ['mean', 'std']
            }).reset_index()

            grouped.columns = ['streams', 'pattern', 'throughput_mean', 'throughput_std', 'sample_size',
                               'concurrent_rate_mean', 'concurrent_rate_std']

            for pattern in ['homogeneous', 'heterogeneous']:
                data = grouped[grouped['pattern'] == pattern]
                if len(data) > 0:
                    label = f'Heterogeneous (n={len(data)})' if pattern == 'heterogeneous' else f'Homogeneous (n={len(data)})'
                    ax3.plot(data['streams'], data['throughput_mean'],
                             marker='o', label=label, linewidth=2, markersize=8)
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

        fig.suptitle('RQ6: Heterogeneity & Load Imbalance', fontsize=16, fontweight='bold')
        self.save_figure('rq6_heterogeneity')

    def _load_raw_csvs(self, pattern: str):
        """Load multiple raw CSV files matching a pattern."""
        import glob
        files = glob.glob(str(self.results_dir / pattern))
        # Also check output_dir for backward compatibility
        files.extend(glob.glob(str(self.output_dir / pattern)))
        dfs = []
        for f in files:
            try:
                df = pd.read_csv(f)
                dfs.append(df)
            except Exception as e:
                print(f"Error loading {f}: {e}")
        return dfs
