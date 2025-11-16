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
                            # Add per_proc_p99 to header
                            header_line = lines[0]
                            modified_header = header_line + ',per_proc_p99'
                            csv_lines.append(modified_header)

                            # Find e2e_p99 column index from header (avoid magic number)
                            header_parts = header_line.split(',')
                            try:
                                e2e_p99_idx = header_parts.index('e2e_p99')
                            except ValueError:
                                print("    Warning: e2e_p99 column not found in header, using index 16")
                                e2e_p99_idx = 16

                            # Add per_proc_p99 = e2e_p99 for single process to data rows
                            for line in lines[1:]:
                                if line.strip() and not line.startswith('streams,'):
                                    parts = line.split(',')
                                    e2e_p99_val = parts[e2e_p99_idx] if len(parts) > e2e_p99_idx else '0'
                                    modified_line = line + f',{e2e_p99_val}'
                                    csv_lines.append(modified_line)
                        else:
                            # Reuse the e2e_p99 index from first run (already stored in header)
                            # Parse header to find e2e_p99 index
                            first_header = csv_lines[0]
                            header_parts = first_header.rstrip(',per_proc_p99').split(',')
                            try:
                                e2e_p99_idx = header_parts.index('e2e_p99')
                            except ValueError:
                                e2e_p99_idx = 16

                            # Add per_proc_p99 = e2e_p99 for single process to data rows
                            for line in lines:
                                if line.strip() and not line.startswith('streams,'):
                                    parts = line.split(',')
                                    e2e_p99_val = parts[e2e_p99_idx] if len(parts) > e2e_p99_idx else '0'
                                    modified_line = line + f',{e2e_p99_val}'
                                    csv_lines.append(modified_line)

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

                    # Skip this run if not all processes succeeded
                    if successful != num_processes:
                        print(f"    Skipping run {run_idx} due to failed processes")
                        continue

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

                            # Calculate P99 latency (global)
                            e2e_p99 = combined_df['e2e_latency_ms'].quantile(0.99)

                            # Calculate per-process (tenant) P99 latency
                            proc_p99_values = []
                            for proc_id in range(effective_procs):
                                proc_df = combined_df[combined_df['global_stream_id'].between(
                                    proc_id * streams_per_process,
                                    (proc_id + 1) * streams_per_process - 1
                                )]
                                if len(proc_df) > 0:
                                    proc_p99_values.append(proc_df['e2e_latency_ms'].quantile(0.99))

                            # Worst-case tenant P99 (max across all tenants)
                            # This shows SLO violation for the most affected tenant
                            per_proc_p99 = np.max(proc_p99_values) if proc_p99_values else 0

                            # Calculate Jain's fairness index per PROCESS (tenant fairness)
                            # Each process is a tenant - measure GPU time fairness across tenants
                            proc_throughputs = []
                            for proc_id in range(effective_procs):
                                proc_df = combined_df[combined_df['global_stream_id'].between(
                                    proc_id * streams_per_process,
                                    (proc_id + 1) * streams_per_process - 1
                                )]
                                if len(proc_df) > 0:
                                    # Total GPU time consumed by this process (tenant)
                                    proc_gpu_time = proc_df['duration_ms'].sum()
                                    proc_throughputs.append(proc_gpu_time)

                            # Jain's index = (sum x)^2 / (n * sum x^2)
                            if proc_throughputs:
                                sum_tput = sum(proc_throughputs)
                                sum_tput_sq = sum(x*x for x in proc_throughputs)
                                n = len(proc_throughputs)
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
                                f'{jains_index:.4f}',  # jains_index (now per-process)
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
                                f'{per_proc_p99:.2f}',  # per_proc_p99 (avg across tenants)
                            ]

                            csv_lines.append(','.join(csv_line_parts))

                time.sleep(1.0)  # Pause between runs

        self.save_csv(csv_lines, 'rq9_multiprocess.csv')

    def analyze(self):
        """
        Analyze RQ9: Multi-process vs single-process fairness and throughput.

        Sub-RQ9.1: Fairness and throughput comparison across process configurations

        Shows 4 metrics:
        - Throughput
        - Per-process (tenant) fairness
        - Global P99 latency
        - Worst-case tenant P99 latency (max across all tenants)
        """
        print("\n=== Analyzing RQ9: Multi-Process vs Single-Process ===")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

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
                'e2e_p99': ['mean', 'std'],
                'per_proc_p99': ['mean', 'std'],
            }).reset_index()

            grouped.columns = ['config', 'throughput_mean', 'throughput_std',
                             'jains_mean', 'jains_std',
                             'e2e_p99_mean', 'e2e_p99_std',
                             'per_proc_p99_mean', 'per_proc_p99_std']

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
            ax1.set_xlabel('Configuration (processes × streams/proc)', fontsize=11)
            ax1.set_ylabel('Throughput (kernels/sec)', fontsize=11)
            ax1.set_title('(a) Throughput vs Process Configuration', fontsize=12, fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels(grouped['config'])
            ax1.grid(True, alpha=0.3, axis='y')

            # Subplot 2: Per-Process (Tenant) Fairness
            yerr = grouped['jains_std'].fillna(0) if grouped['jains_std'].notna().any() else None
            ax2.bar(x, grouped['jains_mean'], width,
                   yerr=yerr,
                   capsize=5, alpha=0.7, color='seagreen')
            ax2.set_xlabel('Configuration (processes × streams/proc)', fontsize=11)
            ax2.set_ylabel("Jain's Fairness Index (per-process)", fontsize=11)
            ax2.set_title('(b) Per-Process Fairness vs Configuration', fontsize=12, fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels(grouped['config'])
            ax2.set_ylim([0, 1.1])
            ax2.grid(True, alpha=0.3, axis='y')

            # Subplot 3: Global P99 Latency
            yerr = grouped['e2e_p99_std'].fillna(0) if grouped['e2e_p99_std'].notna().any() else None
            ax3.bar(x, grouped['e2e_p99_mean'], width,
                   yerr=yerr,
                   capsize=5, alpha=0.7, color='coral')
            ax3.set_xlabel('Configuration (processes × streams/proc)', fontsize=11)
            ax3.set_ylabel('Global E2E P99 Latency (ms)', fontsize=11)
            ax3.set_title('(c) Global P99 Latency vs Configuration', fontsize=12, fontweight='bold')
            ax3.set_xticks(x)
            ax3.set_xticklabels(grouped['config'])
            ax3.grid(True, alpha=0.3, axis='y')

            # Subplot 4: Worst-Tenant P99 Latency
            yerr = grouped['per_proc_p99_std'].fillna(0) if grouped['per_proc_p99_std'].notna().any() else None
            ax4.bar(x, grouped['per_proc_p99_mean'], width,
                   yerr=yerr,
                   capsize=5, alpha=0.7, color='orchid')
            ax4.set_xlabel('Configuration (processes × streams/proc)', fontsize=11)
            ax4.set_ylabel('Worst-Tenant P99 Latency (ms)', fontsize=11)
            ax4.set_title('(d) Worst-Tenant P99 vs Configuration', fontsize=12, fontweight='bold')
            ax4.set_xticks(x)
            ax4.set_xticklabels(grouped['config'])
            ax4.grid(True, alpha=0.3, axis='y')

            fig.suptitle('RQ9: Multi-Process vs Single-Process Scheduling', fontsize=16, fontweight='bold')
            plt.tight_layout()
            self.save_figure('rq9_multiprocess')
        else:
            print("    Warning: No data found for RQ9")
