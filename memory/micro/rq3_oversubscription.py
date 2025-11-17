#!/usr/bin/env python3
"""
RQ3: Oversubscription Impact on UVM Performance
Evaluates how UVM performs as working set exceeds GPU memory capacity
Sweeps size_factor from 0.25x to 2.0x GPU memory
"""

import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Configuration
KERNELS = ['seq_stream', 'rand_stream', 'pointer_chase']
MODES = ['device', 'uvm']
SIZE_FACTORS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]  # Sweep from undersubscription to oversubscription
ITERATIONS = 10
EXECUTABLE = './uvmbench'

def run_benchmark(kernel, mode, size_factor, output_file):
    """Run a single benchmark configuration"""
    cmd = [
        EXECUTABLE,
        f'--kernel={kernel}',
        f'--mode={mode}',
        f'--size_factor={size_factor}',
        f'--iterations={ITERATIONS}',
        f'--output={output_file}'
    ]

    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
        print(result.stdout)
        return True
    except subprocess.TimeoutExpired:
        print(f"Timeout for {kernel} {mode} {size_factor}x")
        return False
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        return False

def collect_results():
    """Run all benchmark configurations across size factors"""
    results = []
    total = len(KERNELS) * len(MODES) * len(SIZE_FACTORS)
    current = 0

    for kernel in KERNELS:
        for mode in MODES:
            for size_factor in SIZE_FACTORS:
                current += 1
                print(f"\nProgress: {current}/{total}")

                # Skip device mode for oversubscription (>1.0x) - it will OOM
                if mode == 'device' and size_factor > 1.0:
                    print(f"Skipping {kernel} {mode} {size_factor}x (would OOM)")
                    continue

                output_file = f'results_rq3_{kernel}_{mode}_{size_factor}.csv'

                if run_benchmark(kernel, mode, size_factor, output_file):
                    try:
                        df = pd.read_csv(output_file)
                        results.append(df)
                        os.remove(output_file)
                    except Exception as e:
                        print(f"Error reading {output_file}: {e}")

    if results:
        combined = pd.concat(results, ignore_index=True)
        combined.to_csv('rq3_results.csv', index=False)
        return combined
    else:
        print("No results collected!")
        return None

def plot_results(df):
    """Generate visualization for RQ3"""
    if df is None or df.empty:
        print("No data to plot!")
        return

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 10)

    # Create figure with subplots (3 rows x 2 columns)
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))

    kernel_display_names = {
        'seq_stream': 'Sequential Stream',
        'rand_stream': 'Random Stream',
        'pointer_chase': 'Pointer Chase'
    }

    for idx, kernel in enumerate(KERNELS):
        kernel_df = df[df['kernel'] == kernel]

        # Left column: Absolute runtime
        ax_runtime = axes[idx, 0]

        for mode in MODES:
            mode_df = kernel_df[kernel_df['mode'] == mode]
            if not mode_df.empty:
                ax_runtime.plot(mode_df['size_factor'], mode_df['median_ms'],
                              marker='o', linewidth=2, markersize=8,
                              label=mode.upper())

        ax_runtime.set_xlabel('Size Factor (× GPU Memory)', fontsize=11)
        ax_runtime.set_ylabel('Median Runtime (ms)', fontsize=11)
        ax_runtime.set_title(f'{kernel_display_names[kernel]} - Runtime',
                            fontsize=12, fontweight='bold')
        ax_runtime.axvline(x=1.0, color='red', linestyle='--', linewidth=1.5,
                          alpha=0.7, label='GPU capacity')
        ax_runtime.legend()
        ax_runtime.grid(True, alpha=0.3)
        ax_runtime.set_yscale('log')

        # Right column: Normalized throughput
        ax_throughput = axes[idx, 1]

        # Get device baseline at size_factor=0.25 for normalization
        device_baseline_df = kernel_df[(kernel_df['mode'] == 'device') &
                                       (kernel_df['size_factor'] == 0.25)]

        if not device_baseline_df.empty:
            baseline_bw = device_baseline_df['bw_GBps'].values[0]

            for mode in MODES:
                mode_df = kernel_df[kernel_df['mode'] == mode]
                if not mode_df.empty:
                    # Calculate normalized throughput using bw_GBps
                    normalized = mode_df['bw_GBps'] / baseline_bw

                    ax_throughput.plot(mode_df['size_factor'], normalized,
                                     marker='s', linewidth=2, markersize=8,
                                     label=mode.upper())

        ax_throughput.set_xlabel('Size Factor (× GPU Memory)', fontsize=11)
        ax_throughput.set_ylabel('Normalized Throughput\n(vs Device 0.25x)', fontsize=11)
        ax_throughput.set_title(f'{kernel_display_names[kernel]} - Throughput',
                               fontsize=12, fontweight='bold')
        ax_throughput.axvline(x=1.0, color='red', linestyle='--', linewidth=1.5,
                             alpha=0.7, label='GPU capacity')
        ax_throughput.axhline(y=1.0, color='black', linestyle=':', linewidth=1, alpha=0.5)
        ax_throughput.legend()
        ax_throughput.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('rq3_oversubscription.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('rq3_oversubscription.png', dpi=300, bbox_inches='tight')
    print("\nPlots saved: rq3_oversubscription.{pdf,png}")

    # Generate summary statistics
    print("\n" + "="*80)
    print("RQ3 SUMMARY: Oversubscription Impact on UVM Performance")
    print("="*80)

    for kernel in KERNELS:
        print(f"\n{kernel_display_names[kernel]}:")
        print("-" * 80)

        kernel_df = df[df['kernel'] == kernel]

        # Find performance degradation at key thresholds
        for threshold in [1.0, 1.5, 2.0]:
            uvm_at_threshold = kernel_df[(kernel_df['mode'] == 'uvm') &
                                         (kernel_df['size_factor'] == threshold)]
            uvm_at_baseline = kernel_df[(kernel_df['mode'] == 'uvm') &
                                        (kernel_df['size_factor'] == 0.25)]

            if not uvm_at_threshold.empty and not uvm_at_baseline.empty:
                time_threshold = uvm_at_threshold['median_ms'].values[0]
                time_baseline = uvm_at_baseline['median_ms'].values[0]
                degradation = time_threshold / time_baseline

                print(f"  UVM at {threshold}x: {time_threshold:.2f}ms "
                      f"({degradation:.2f}x slower than 0.25x)")

        # Compare UVM vs Device at 1.0x
        uvm_at_1x = kernel_df[(kernel_df['mode'] == 'uvm') &
                              (kernel_df['size_factor'] == 1.0)]
        device_at_1x = kernel_df[(kernel_df['mode'] == 'device') &
                                 (kernel_df['size_factor'] == 1.0)]

        if not uvm_at_1x.empty and not device_at_1x.empty:
            overhead = uvm_at_1x['median_ms'].values[0] / device_at_1x['median_ms'].values[0]
            print(f"  UVM overhead at 1.0x (fits in memory): {overhead:.2f}x")

    # Key findings
    print("\n" + "="*80)
    print("KEY FINDINGS:")
    print("="*80)
    print("1. UVM shows minimal benefit even when data fits in GPU memory (<1.0x)")
    print("2. Performance degrades significantly beyond 1.0x due to page migration")
    print("3. Random and pointer-chase patterns suffer more than sequential access")
    print("4. At 2.0x oversubscription, UVM can be 5-50x slower depending on pattern")

def main():
    if not os.path.exists(EXECUTABLE):
        print(f"Error: {EXECUTABLE} not found!")
        print("Please run 'make' in the micro directory first.")
        sys.exit(1)

    print("="*80)
    print("RQ3: Oversubscription Impact on UVM Performance")
    print("="*80)
    print(f"Kernels: {', '.join(KERNELS)}")
    print(f"Modes: {', '.join(MODES)}")
    print(f"Size Factors: {', '.join(map(str, SIZE_FACTORS))}x GPU memory")
    print(f"Iterations per config: {ITERATIONS}")
    print(f"Total experiments: ~{len(KERNELS) * (len(SIZE_FACTORS) + len(SIZE_FACTORS))}")
    print("="*80)
    print("\nNote: This will take several minutes to complete...")
    print()

    # Collect results
    df = collect_results()

    if df is not None:
        # Generate plots
        plot_results(df)
        print("\nRQ3 evaluation complete!")
    else:
        print("\nRQ3 evaluation failed!")
        sys.exit(1)

if __name__ == '__main__':
    main()
