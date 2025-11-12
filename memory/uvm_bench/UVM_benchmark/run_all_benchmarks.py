#!/usr/bin/env python3
"""
UVMBench - Comprehensive Benchmark Runner

This script runs all available benchmarks in the UVMBench suite
and collects results with timing information.

Usage:
    python3 run_all_benchmarks.py [--mode MODE] [--profile] [--timeout SECONDS]

Arguments:
    --mode MODE         Run mode: 'uvm', 'non-uvm', or 'both' (default: both)
    --profile          Enable ncu profiling (saves .ncu-rep files)
    --timeout SECONDS  Timeout per benchmark in seconds (default: 300)
    --verbose          Show detailed output
    --help             Show this help message
"""

import os
import sys
import subprocess
import time
import argparse
from datetime import datetime
from pathlib import Path
import json

# Color codes for terminal output
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'  # No Color
    BOLD = '\033[1m'

class BenchmarkRunner:
    def __init__(self, root_dir, results_dir, timeout=300, profile=False, verbose=False):
        self.root_dir = Path(root_dir)
        self.results_dir = Path(results_dir)
        self.timeout = timeout
        self.profile = profile
        self.verbose = verbose

        # Statistics
        self.stats = {
            'total': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'timeout': 0
        }

        # Results storage
        self.results = []

        # Create results directory
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.results_dir / 'run_all.log'

        # Initialize log file
        with open(self.log_file, 'w') as f:
            f.write(f"UVMBench Run - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Profile enabled: {profile}\n")
            f.write(f"Timeout: {timeout}s\n")
            f.write("=" * 60 + "\n\n")

    def log(self, message, to_console=True, to_file=True):
        """Log message to console and/or file"""
        if to_console:
            print(message)
        if to_file:
            # Strip color codes for file output
            clean_msg = message
            for color in [Colors.RED, Colors.GREEN, Colors.YELLOW, Colors.BLUE,
                         Colors.CYAN, Colors.NC, Colors.BOLD]:
                clean_msg = clean_msg.replace(color, '')
            with open(self.log_file, 'a') as f:
                f.write(clean_msg + '\n')

    def run_command(self, cmd, cwd, timeout=None):
        """Run a command with timeout and capture output"""
        if timeout is None:
            timeout = self.timeout

        try:
            start_time = time.time()
            result = subprocess.run(
                cmd,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                shell=True,
                text=True
            )
            elapsed = time.time() - start_time
            return {
                'success': result.returncode == 0,
                'output': result.stdout,
                'elapsed': elapsed,
                'timeout': False
            }
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            return {
                'success': False,
                'output': 'Timeout expired',
                'elapsed': elapsed,
                'timeout': True
            }
        except Exception as e:
            elapsed = time.time() - start_time
            return {
                'success': False,
                'output': str(e),
                'elapsed': elapsed,
                'timeout': False
            }

    def run_benchmark(self, bench_path, bench_name, version):
        """Run a single benchmark"""
        self.log(f"\n{Colors.BLUE}{'─' * 60}{Colors.NC}")
        self.log(f"Running: {Colors.GREEN}{bench_name}{Colors.NC} ({version})")
        self.log(f"{Colors.BLUE}{'─' * 60}{Colors.NC}")

        self.stats['total'] += 1

        bench_path = self.root_dir / bench_path

        # Check if directory exists
        if not bench_path.exists():
            self.log(f"{Colors.YELLOW}[SKIP]{Colors.NC} Directory not found: {bench_path}")
            self.stats['skipped'] += 1
            self.results.append({
                'name': bench_name,
                'version': version,
                'status': 'SKIP',
                'reason': 'Directory not found',
                'time': 0
            })
            return

        # Check for executable or run script
        run_script = bench_path / 'run'
        main_exec = bench_path / 'main'

        if not run_script.exists() and not main_exec.exists():
            self.log(f"{Colors.YELLOW}[SKIP]{Colors.NC} No executable found")
            self.stats['skipped'] += 1
            self.results.append({
                'name': bench_name,
                'version': version,
                'status': 'SKIP',
                'reason': 'No executable',
                'time': 0
            })
            return

        # Prepare output files
        output_file = self.results_dir / f"{version}_{bench_name}.txt"

        # Run benchmark
        if self.profile and run_script.exists():
            profile_file = self.results_dir / f"{version}_{bench_name}.ncu-rep"
            cmd = f"ncu --set full --export {profile_file} ./run"
        elif run_script.exists():
            cmd = "./run"
        else:
            cmd = "./main"

        if self.verbose:
            self.log(f"  Command: {cmd}", to_file=True)
            self.log(f"  Working directory: {bench_path}", to_file=True)

        result = self.run_command(cmd, cwd=bench_path)

        # Save output
        with open(output_file, 'w') as f:
            f.write(result['output'])

        # Record result
        if result['timeout']:
            status_str = f"{Colors.RED}[TIMEOUT]{Colors.NC}"
            status = 'TIMEOUT'
            self.stats['timeout'] += 1
        elif result['success']:
            status_str = f"{Colors.GREEN}[SUCCESS]{Colors.NC}"
            status = 'SUCCESS'
            self.stats['successful'] += 1
        else:
            status_str = f"{Colors.RED}[FAILED]{Colors.NC}"
            status = 'FAILED'
            self.stats['failed'] += 1

        self.log(f"{status_str} Completed in {result['elapsed']:.2f}s")

        if self.verbose and not result['success']:
            self.log(f"  Error output (first 10 lines):")
            for line in result['output'].split('\n')[:10]:
                self.log(f"    {line}")

        self.results.append({
            'name': bench_name,
            'version': version,
            'status': status,
            'time': result['elapsed'],
            'output_file': str(output_file)
        })

    def run_all_benchmarks(self, mode='both'):
        """Run all benchmarks based on mode"""

        # Benchmark lists
        simple_benchmarks = ['bfs', 'BN', 'CNN', 'kmeans', 'knn', 'logistic-regression', 'SVM']
        rodinia_benchmarks = ['backprop', 'dwt2d', 'gaussian', 'hotspot', 'hotspot3D',
                             'nn', 'nw', 'particlefilter', 'pathfinder', 'srad', 'streamcluster']
        polybench_benchmarks = ['2DCONV', '2MM', '3DCONV', '3MM', 'ATAX', 'BICG', 'CORR',
                               'COVAR', 'FDTD-2D', 'GEMM', 'GESUMMV', 'GRAMSCHM', 'MVT',
                               'SYR2K', 'SYRK']

        # Run UVM benchmarks
        if mode in ['uvm', 'both']:
            self.log(f"\n{Colors.GREEN}{'=' * 65}{Colors.NC}")
            self.log(f"{Colors.GREEN}{Colors.BOLD}Running UVM Benchmarks{Colors.NC}")
            self.log(f"{Colors.GREEN}{'=' * 65}{Colors.NC}\n")

            base_dir = 'UVM_benchmarks'

            # Simple benchmarks
            self.log(f"{Colors.CYAN}Running simple benchmarks...{Colors.NC}")
            for bench in simple_benchmarks:
                self.run_benchmark(f"{base_dir}/{bench}", bench, "UVM")

            # Rodinia benchmarks
            self.log(f"\n{Colors.CYAN}Running Rodinia benchmarks...{Colors.NC}")
            for bench in rodinia_benchmarks:
                self.run_benchmark(f"{base_dir}/rodinia/{bench}", f"rodinia_{bench}", "UVM")

            # Polybench benchmarks
            self.log(f"\n{Colors.CYAN}Running Polybench benchmarks...{Colors.NC}")
            for bench in polybench_benchmarks:
                self.run_benchmark(f"{base_dir}/polybench/{bench}", f"polybench_{bench}", "UVM")

        # Run non-UVM benchmarks
        if mode in ['non-uvm', 'both']:
            self.log(f"\n{Colors.GREEN}{'=' * 65}{Colors.NC}")
            self.log(f"{Colors.GREEN}{Colors.BOLD}Running non-UVM Benchmarks{Colors.NC}")
            self.log(f"{Colors.GREEN}{'=' * 65}{Colors.NC}\n")

            base_dir = 'non_UVM_benchmarks'

            # Simple benchmarks
            self.log(f"{Colors.CYAN}Running simple benchmarks...{Colors.NC}")
            for bench in simple_benchmarks:
                self.run_benchmark(f"{base_dir}/{bench}", bench, "non-UVM")

            # Rodinia benchmarks
            self.log(f"\n{Colors.CYAN}Running Rodinia benchmarks...{Colors.NC}")
            for bench in rodinia_benchmarks:
                self.run_benchmark(f"{base_dir}/rodinia/{bench}", f"rodinia_{bench}", "non-UVM")

            # Polybench benchmarks
            self.log(f"\n{Colors.CYAN}Running Polybench benchmarks...{Colors.NC}")
            for bench in polybench_benchmarks:
                self.run_benchmark(f"{base_dir}/polybench/{bench}", f"polybench_{bench}", "non-UVM")

    def generate_summary(self):
        """Generate summary report"""
        self.log(f"\n{Colors.BLUE}{'=' * 65}{Colors.NC}")
        self.log(f"{Colors.BLUE}{Colors.BOLD}Benchmark Execution Summary{Colors.NC}")
        self.log(f"{Colors.BLUE}{'=' * 65}{Colors.NC}")
        self.log(f"Total benchmarks attempted: {self.stats['total']}")
        self.log(f"{Colors.GREEN}Successful: {self.stats['successful']}{Colors.NC}")
        self.log(f"{Colors.RED}Failed: {self.stats['failed']}{Colors.NC}")
        self.log(f"{Colors.YELLOW}Timeout: {self.stats['timeout']}{Colors.NC}")
        self.log(f"{Colors.YELLOW}Skipped: {self.stats['skipped']}{Colors.NC}")
        self.log(f"Results saved to: {self.results_dir}")
        self.log(f"Log file: {self.log_file}")
        self.log(f"{Colors.BLUE}{'=' * 65}{Colors.NC}\n")

        # Generate CSV summary
        csv_file = self.results_dir / 'summary.csv'
        with open(csv_file, 'w') as f:
            f.write("Benchmark,Version,Status,Time(s),Output\n")
            for result in self.results:
                f.write(f"{result['name']},{result['version']},{result['status']},"
                       f"{result['time']:.3f},{result.get('output_file', '')}\n")

        self.log(f"Summary CSV: {csv_file}")

        # Generate JSON summary
        json_file = self.results_dir / 'summary.json'
        with open(json_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'statistics': self.stats,
                'results': self.results
            }, f, indent=2)

        self.log(f"Summary JSON: {json_file}")

        # Generate comparison table for UVM vs non-UVM
        uvm_results = {r['name']: r for r in self.results if r['version'] == 'UVM' and r['status'] == 'SUCCESS'}
        non_uvm_results = {r['name']: r for r in self.results if r['version'] == 'non-UVM' and r['status'] == 'SUCCESS'}

        common_benchmarks = set(uvm_results.keys()) & set(non_uvm_results.keys())

        if common_benchmarks:
            comparison_file = self.results_dir / 'comparison.txt'
            with open(comparison_file, 'w') as f:
                f.write("UVM vs non-UVM Performance Comparison\n")
                f.write("=" * 80 + "\n")
                f.write(f"{'Benchmark':<30} {'UVM (s)':>12} {'non-UVM (s)':>12} {'Speedup':>10}\n")
                f.write("-" * 80 + "\n")

                for bench in sorted(common_benchmarks):
                    uvm_time = uvm_results[bench]['time']
                    non_uvm_time = non_uvm_results[bench]['time']
                    speedup = non_uvm_time / uvm_time if uvm_time > 0 else 0
                    f.write(f"{bench:<30} {uvm_time:>12.3f} {non_uvm_time:>12.3f} {speedup:>10.2f}x\n")

            self.log(f"Comparison table: {comparison_file}")


def prepare_test_data(root_dir):
    """Prepare necessary test data for benchmarks"""
    root = Path(root_dir)

    print(f"{Colors.CYAN}Preparing test data...{Colors.NC}")

    # Generate BFS data
    bfs_gen = root / 'data' / 'bfs' / 'inputGen' / 'graphgen'
    if bfs_gen.exists():
        print(f"  Generating BFS graph data...")
        subprocess.run([str(bfs_gen), '1024', '1k'],
                      cwd=bfs_gen.parent,
                      stdout=subprocess.DEVNULL,
                      stderr=subprocess.DEVNULL)
        subprocess.run([str(bfs_gen), '8192', '8k'],
                      cwd=bfs_gen.parent,
                      stdout=subprocess.DEVNULL,
                      stderr=subprocess.DEVNULL)
        print(f"  {Colors.GREEN}✓{Colors.NC} BFS data generated")

    # Generate K-means data
    kmeans_data_dir = root / 'data' / 'kmeans'
    kmeans_data_dir.mkdir(parents=True, exist_ok=True)

    if not (kmeans_data_dir / '1000_points.txt').exists():
        print(f"  Generating K-means data...")
        import random

        # Generate point files
        for size in [1000, 10000, 50000, 100000]:
            with open(kmeans_data_dir / f'{size}_points.txt', 'w') as f:
                for _ in range(size):
                    x = random.uniform(0, 100)
                    y = random.uniform(0, 100)
                    f.write(f'{x} {y}\n')

        # Generate initial centroids
        with open(kmeans_data_dir / 'initCoord.txt', 'w') as f:
            for _ in range(2):
                x = random.uniform(0, 100)
                y = random.uniform(0, 100)
                f.write(f'{x} {y}\n')

        print(f"  {Colors.GREEN}✓{Colors.NC} K-means data generated")

    # Create result directories
    for version in ['UVM_benchmarks', 'non_UVM_benchmarks']:
        result_dir = root / version / 'kmeans' / 'result' / 'cuda'
        result_dir.mkdir(parents=True, exist_ok=True)

    print(f"{Colors.GREEN}✓ Test data preparation complete{Colors.NC}\n")


def main():
    parser = argparse.ArgumentParser(
        description='UVMBench - Comprehensive Benchmark Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all benchmarks (UVM and non-UVM)
  python3 run_all_benchmarks.py

  # Run only UVM benchmarks
  python3 run_all_benchmarks.py --mode uvm

  # Run with profiling enabled
  python3 run_all_benchmarks.py --profile

  # Run with custom timeout and verbose output
  python3 run_all_benchmarks.py --timeout 600 --verbose
        """
    )

    parser.add_argument('--mode', choices=['uvm', 'non-uvm', 'both'],
                       default='both', help='Run mode (default: both)')
    parser.add_argument('--profile', action='store_true',
                       help='Enable ncu profiling')
    parser.add_argument('--timeout', type=int, default=300,
                       help='Timeout per benchmark in seconds (default: 300)')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed output')
    parser.add_argument('--prepare-data', action='store_true',
                       help='Only prepare test data and exit')

    args = parser.parse_args()

    # Get root directory
    root_dir = Path.cwd()

    # Prepare data if requested
    if args.prepare_data:
        prepare_test_data(root_dir)
        return 0

    # Create results directory with timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    results_dir = root_dir / 'results' / timestamp

    # Print header
    print(f"{Colors.BLUE}{'=' * 65}{Colors.NC}")
    print(f"{Colors.BLUE}{Colors.BOLD}UVMBench - Comprehensive Benchmark Suite Runner{Colors.NC}")
    print(f"{Colors.BLUE}{'=' * 65}{Colors.NC}")
    print(f"Run mode: {Colors.YELLOW}{args.mode}{Colors.NC}")
    print(f"Profiling: {Colors.YELLOW}{args.profile}{Colors.NC}")
    print(f"Timeout: {Colors.YELLOW}{args.timeout}s{Colors.NC}")
    print(f"Results directory: {Colors.YELLOW}{results_dir}{Colors.NC}")
    print(f"{Colors.BLUE}{'=' * 65}{Colors.NC}\n")

    # Prepare test data
    prepare_test_data(root_dir)

    # Create runner and execute
    runner = BenchmarkRunner(root_dir, results_dir,
                            timeout=args.timeout,
                            profile=args.profile,
                            verbose=args.verbose)

    start_time = time.time()
    runner.run_all_benchmarks(mode=args.mode)
    total_time = time.time() - start_time

    runner.generate_summary()

    print(f"\nTotal execution time: {Colors.YELLOW}{total_time:.1f}s{Colors.NC}\n")

    return 0 if runner.stats['failed'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
