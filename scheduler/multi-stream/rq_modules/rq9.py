"""RQ9: Multi-Process vs Single-Process (formerly RQ8)"""
import time
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from .base import RQBase


class RQ9(RQBase):
    """RQ9: Multi-process vs single-process scheduling behavior (formerly RQ8)."""

    def run_experiments(self):
        """
        RQ9: Multi-process vs single-process scheduling behavior.

        Sub-RQ9.1: Single-process multi-stream vs multi-process few-stream
        Compares fairness, throughput, and concurrency when:
        - Mode A: 1 process with 32 streams
        - Mode B: 4 processes with 8 streams each
        - Mode C: 8 processes with 4 streams each
        - Mode D: 16 processes with 2 streams each

        This uses concurrent process execution to properly test multi-process GPU scheduling.
        """
        print("\n=== Running RQ9 Experiments: Multi-Process vs Single-Process ===")

        total_streams = 32
        configurations = [
            ('1proc_32streams', 1, 32),
            ('4proc_8streams', 4, 8),
            ('8proc_4streams', 8, 4),
            ('16proc_2streams', 16, 2),
        ]

        csv_lines = []

        for config_name, num_processes, streams_per_process in configurations:
            print(f"  Config: {config_name} ({num_processes} processes × {streams_per_process} streams)")

            for run_idx in range(self.num_runs):
                # For single process, just run normally
                if num_processes == 1:
                    args = [
                        '--streams', str(streams_per_process),
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

                else:
                    # For multiple processes, run them CONCURRENTLY using subprocess.Popen
                    print(f"    Launching {num_processes} concurrent processes...")

                    # Start all processes concurrently
                    processes = []
                    raw_files = []

                    for proc_id in range(num_processes):
                        raw_csv_file = str(self.output_dir / f'rq9_raw_{config_name}_run{run_idx}_proc{proc_id}.csv')
                        raw_files.append(raw_csv_file)

                        cmd = [
                            self.bench_path,
                            '--streams', str(streams_per_process),
                            '--kernels', '30',
                            '--size', '1048576',
                            '--type', 'mixed',
                            '--csv-output', raw_csv_file,
                            '--no-header',  # Skip header in raw files
                        ]

                        proc = subprocess.Popen(
                            cmd,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL
                        )
                        processes.append(proc)

                    # Wait for all processes to complete
                    print(f"    Waiting for {num_processes} processes to complete...")
                    results = []
                    for proc_id, (proc, raw_file) in enumerate(zip(processes, raw_files)):
                        returncode = proc.wait(timeout=120)
                        success = (returncode == 0)
                        results.append((proc_id, success, raw_file))

                    # Check results
                    successful = sum(1 for _, success, _ in results if success)
                    print(f"    Completed: {successful}/{num_processes} processes succeeded")

                    # Aggregate metrics from all raw CSVs for this run
                    raw_files = [f for _, success, f in results if success and f]
                    if raw_files:
                        # Read and combine all per-kernel data
                        all_timings = []
                        for proc_id, raw_file in enumerate(raw_files):
                            if os.path.exists(raw_file):
                                df = pd.read_csv(raw_file)
                                # Assign global stream IDs: each process's streams get unique IDs
                                df['global_stream_id'] = df['stream_id'] + proc_id * streams_per_process
                                all_timings.append(df)

                        if all_timings:
                            # Combine all timings
                            combined_df = pd.concat(all_timings, ignore_index=True)

                            # Compute aggregate metrics across all processes
                            total_kernels = len(combined_df)
                            effective_procs = len(raw_files)
                            total_streams = effective_procs * streams_per_process

                            # Calculate throughput: total kernels / total wall time
                            global_start = combined_df['start_time_ms'].min()
                            global_end = combined_df['end_time_ms'].max()
                            wall_time_sec = (global_end - global_start) / 1000.0
                            throughput = total_kernels / wall_time_sec if wall_time_sec > 0 else 0

                            # Calculate P99 latency
                            e2e_p99 = combined_df['e2e_latency_ms'].quantile(0.99)

                            # Calculate Jain's fairness index per stream (using global stream IDs)
                            stream_throughputs = []
                            for stream_id in combined_df['global_stream_id'].unique():
                                stream_df = combined_df[combined_df['global_stream_id'] == stream_id]
                                stream_start = stream_df['start_time_ms'].min()
                                stream_end = stream_df['end_time_ms'].max()
                                stream_time = (stream_end - stream_start) / 1000.0
                                stream_tput = len(stream_df) / stream_time if stream_time > 0 else 0
                                stream_throughputs.append(stream_tput)

                            # Jain's index = (sum x)^2 / (n * sum x^2)
                            if stream_throughputs:
                                sum_tput = sum(stream_throughputs)
                                sum_tput_sq = sum(x*x for x in stream_throughputs)
                                n = len(stream_throughputs)
                                jains_index = (sum_tput * sum_tput) / (n * sum_tput_sq) if sum_tput_sq > 0 else 1.0
                            else:
                                jains_index = 1.0

                            # Compute GPU utilization (simplified: fraction of time with active kernels)
                            events = []
                            for _, row in combined_df.iterrows():
                                events.append((row['start_time_ms'], 1))
                                events.append((row['end_time_ms'], -1))
                            events.sort()

                            time_with_kernels = 0
                            current_count = 0
                            last_time = events[0][0]
                            for event_time, delta in events:
                                if current_count > 0:
                                    time_with_kernels += (event_time - last_time)
                                current_count += delta
                                last_time = event_time

                            util = (time_with_kernels / (global_end - global_start) * 100) if (global_end - global_start) > 0 else 0

                            # Create a full CSV line matching the header format
                            wall_time = (global_end - global_start)

                            csv_line_parts = [
                                str(total_streams),  # streams
                                str(total_kernels // total_streams),  # kernels_per_stream
                                f'{num_processes}proc_x_{streams_per_process}streams',  # kernels_per_stream_detail
                                str(total_kernels),  # total_kernels
                                'mixed',  # type
                                'multiprocess',  # type_detail
                                f'{wall_time:.2f}',  # wall_time_ms
                                f'{wall_time:.2f}',  # e2e_wall_time_ms
                                f'{throughput:.2f}',  # throughput
                                '',  # svc_mean
                                '',  # svc_p50
                                '',  # svc_p95
                                '',  # svc_p99
                                '',  # e2e_mean
                                '',  # e2e_p50
                                '',  # e2e_p95
                                f'{e2e_p99:.2f}',  # e2e_p99
                                '',  # avg_queue_wait
                                '',  # max_queue_wait
                                '',  # concurrent_rate
                                f'{util:.1f}',  # util
                                f'{jains_index:.4f}',  # jains_index
                                '',  # max_concurrent
                                '',  # avg_concurrent
                                '',  # inversions
                                '',  # inversion_rate
                                '',  # working_set_mb
                                '',  # fits_in_l2
                                '',  # svc_stddev
                                '',  # grid_size
                                '',  # block_size
                                '',  # per_priority_avg
                                '',  # per_priority_p50
                                '',  # per_priority_p99
                                '0',  # launch_freq
                                '0',  # seed
                            ]

                            csv_lines.append(','.join(csv_line_parts))

                time.sleep(1.0)  # Pause between runs

        self.save_csv(csv_lines, 'rq9_multiprocess.csv')

    def analyze(self):
        """
        Analyze RQ9: Multi-process vs single-process fairness and throughput.

        Sub-RQ9.1: Fairness and throughput comparison across process configurations
        """
        print("\n=== Analyzing RQ9: Multi-Process vs Single-Process ===")

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

        # Load aggregated data
        df = self.load_csv('rq9_multiprocess.csv')
        if df is not None and len(df) > 0:
            print("  RQ9.1: Fairness and throughput vs process configuration")

            # Parse configuration from kernels_per_stream_detail
            def infer_config(row):
                detail = str(row.get('kernels_per_stream_detail', ''))

                if 'proc_x' in detail:
                    # Format: "4proc_x_8streams" -> "4×8"
                    parts = detail.replace('proc_x_', '×').replace('streams', '')
                    return parts
                else:
                    # Single process: uniform with 32 streams
                    return '1×32'

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
            yerr = grouped['throughput_std'].fillna(0) if grouped['throughput_std'].notna().any() else None
            ax1.bar(x, grouped['throughput_mean'], width,
                   yerr=yerr,
                   capsize=5, alpha=0.7, color='steelblue')
            ax1.set_xlabel('Configuration (processes × streams/proc)')
            ax1.set_ylabel('Throughput (kernels/sec)')
            ax1.set_title('(a) Throughput vs Process Configuration')
            ax1.set_xticks(x)
            ax1.set_xticklabels(grouped['config'])
            ax1.grid(True, alpha=0.3, axis='y')

            # Subplot 2: Jain's Fairness Index
            yerr = grouped['jains_std'].fillna(0) if grouped['jains_std'].notna().any() else None
            ax2.bar(x, grouped['jains_mean'], width,
                   yerr=yerr,
                   capsize=5, alpha=0.7, color='seagreen')
            ax2.set_xlabel('Configuration (processes × streams/proc)')
            ax2.set_ylabel("Jain's Fairness Index")
            ax2.set_title('(b) Fairness vs Process Configuration')
            ax2.set_xticks(x)
            ax2.set_xticklabels(grouped['config'])
            ax2.set_ylim([0, 1.1])
            ax2.grid(True, alpha=0.3, axis='y')

            # Subplot 3: P99 Latency
            yerr = grouped['e2e_p99_std'].fillna(0) if grouped['e2e_p99_std'].notna().any() else None
            ax3.bar(x, grouped['e2e_p99_mean'], width,
                   yerr=yerr,
                   capsize=5, alpha=0.7, color='coral')
            ax3.set_xlabel('Configuration (processes × streams/proc)')
            ax3.set_ylabel('E2E P99 Latency (ms)')
            ax3.set_title('(c) P99 Latency vs Process Configuration')
            ax3.set_xticks(x)
            ax3.set_xticklabels(grouped['config'])
            ax3.grid(True, alpha=0.3, axis='y')

            fig.suptitle('RQ9: Multi-Process vs Single-Process Scheduling', fontsize=16, fontweight='bold')
            self.save_figure('rq9_multiprocess')
        else:
            print("    Warning: No data found for RQ9")
