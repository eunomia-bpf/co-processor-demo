#!/usr/bin/env python3
"""
Benchmark driver for GEMM testing with different matrix patterns and policies
Runs comprehensive performance tests and generates reports
"""

import subprocess
import os
import re
import csv
import argparse
import json
from datetime import datetime

class BenchmarkRunner:
    def __init__(self, matrix_dir, binary_original, binary_policy, warmup=3, runs=10):
        self.matrix_dir = matrix_dir
        self.binary_original = binary_original
        self.binary_policy = binary_policy
        self.warmup = warmup
        self.runs = runs
        self.results = []

    def parse_output(self, output):
        """Parse benchmark output to extract metrics"""
        time_match = re.search(r'Time:\s+([\d.]+)\s+ms', output)
        gflops_match = re.search(r'GFLOPS:\s+([\d.]+)', output)
        verified_match = re.search(r'✓ Results verified successfully', output)

        return {
            'time_ms': float(time_match.group(1)) if time_match else None,
            'gflops': float(gflops_match.group(1)) if gflops_match else None,
            'verified': verified_match is not None
        }

    def run_benchmark(self, binary, env_vars, matrix_files, size):
        """Run a single benchmark"""
        cmd = [binary]

        if matrix_files:
            cmd.extend(['--matrix-a', matrix_files['A']])
            cmd.extend(['--matrix-b', matrix_files['B']])
            cmd.extend(['--matrix-c', matrix_files['C']])
            cmd.extend(['--size', size])

        env = os.environ.copy()
        env.update(env_vars)

        times = []
        gflops_list = []
        verified = True

        # Warmup + runs
        for run in range(self.warmup + self.runs):
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=60)
                metrics = self.parse_output(result.stdout)

                if not metrics['verified']:
                    verified = False

                if run >= self.warmup:  # Skip warmup runs
                    if metrics['time_ms']:
                        times.append(metrics['time_ms'])
                    if metrics['gflops']:
                        gflops_list.append(metrics['gflops'])

            except subprocess.TimeoutExpired:
                print(f"    TIMEOUT!")
                return None

        if not times:
            return None

        import statistics
        return {
            'time_avg': statistics.mean(times),
            'time_std': statistics.stdev(times) if len(times) > 1 else 0,
            'time_min': min(times),
            'time_max': max(times),
            'gflops_avg': statistics.mean(gflops_list),
            'gflops_std': statistics.stdev(gflops_list) if len(gflops_list) > 1 else 0,
            'verified': verified,
            'samples': len(times)
        }

    def load_manifest(self):
        """Load matrix manifest file"""
        manifest_file = os.path.join(self.matrix_dir, 'manifest.txt')
        if not os.path.exists(manifest_file):
            return []

        tests = []
        with open(manifest_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split(',')
                if len(parts) == 5:
                    tests.append({
                        'size': parts[0],
                        'pattern': parts[1],
                        'A': parts[2],
                        'B': parts[3],
                        'C': parts[4]
                    })
        return tests

    def run_all_benchmarks(self, policies):
        """Run all benchmarks with all policies"""
        tests = self.load_manifest()

        if not tests:
            print("No test matrices found. Generating default size...")
            tests = [{'size': '512x512x512', 'pattern': 'default', 'A': None, 'B': None, 'C': None}]

        print("=" * 80)
        print("GEMM Benchmark Suite")
        print("=" * 80)
        print(f"Matrix directory: {self.matrix_dir}")
        print(f"Warmup runs: {self.warmup}")
        print(f"Benchmark runs: {self.runs}")
        print(f"Total tests: {len(tests)} matrices x {len(policies) + 1} configs")
        print("=" * 80)

        for idx, test in enumerate(tests):
            print(f"\n[{idx+1}/{len(tests)}] Size: {test['size']}, Pattern: {test['pattern']}")

            matrix_files = {
                'A': test['A'],
                'B': test['B'],
                'C': test['C']
            } if test['A'] else None

            # Run original version
            print("  Running: Original (no policy)...", end=' ', flush=True)
            result_orig = self.run_benchmark(
                self.binary_original, {}, matrix_files, test['size']
            )
            if result_orig:
                print(f"✓ {result_orig['time_avg']:.3f}ms, {result_orig['gflops_avg']:.2f} GFLOPS")
                self.results.append({
                    'size': test['size'],
                    'pattern': test['pattern'],
                    'config': 'original',
                    'policy': 'none',
                    **result_orig
                })
            else:
                print("✗ FAILED")

            # Run policy versions
            for policy_name, policy_ptx in policies.items():
                print(f"  Running: Policy ({policy_name})...", end=' ', flush=True)
                env = {
                    'WRAPPER_KERNEL_PATH': './wrapper_kernel.ptx',
                    'POLICY_PTX_PATH': policy_ptx
                }
                result_policy = self.run_benchmark(
                    self.binary_policy, env, matrix_files, test['size']
                )
                if result_policy:
                    speedup = result_orig['time_avg'] / result_policy['time_avg'] if result_orig else 1.0
                    print(f"✓ {result_policy['time_avg']:.3f}ms, {result_policy['gflops_avg']:.2f} GFLOPS (speedup: {speedup:.2f}x)")
                    self.results.append({
                        'size': test['size'],
                        'pattern': test['pattern'],
                        'config': 'policy',
                        'policy': policy_name,
                        'speedup_vs_original': speedup,
                        **result_policy
                    })
                else:
                    print("✗ FAILED")

    def save_results(self, output_file):
        """Save results to CSV file"""
        if not self.results:
            print("No results to save")
            return

        with open(output_file, 'w', newline='') as f:
            fieldnames = ['size', 'pattern', 'config', 'policy', 'time_avg', 'time_std',
                          'time_min', 'time_max', 'gflops_avg', 'gflops_std',
                          'speedup_vs_original', 'verified', 'samples']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for result in self.results:
                row = {k: result.get(k, '') for k in fieldnames}
                writer.writerow(row)

        print(f"\n✓ Results saved to {output_file}")

    def print_summary(self):
        """Print benchmark summary"""
        if not self.results:
            return

        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)

        # Group by pattern
        patterns = {}
        for result in self.results:
            pattern = result['pattern']
            if pattern not in patterns:
                patterns[pattern] = []
            patterns[pattern].append(result)

        for pattern, results in patterns.items():
            print(f"\nPattern: {pattern}")
            print("-" * 80)
            print(f"{'Config':<20} {'Policy':<15} {'Time (ms)':<12} {'GFLOPS':<12} {'Speedup':<10}")
            print("-" * 80)

            for result in results:
                config = result['config']
                policy = result['policy']
                time_str = f"{result['time_avg']:.3f} ± {result['time_std']:.3f}"
                gflops_str = f"{result['gflops_avg']:.2f}"
                speedup_str = f"{result.get('speedup_vs_original', 1.0):.2f}x" if 'speedup_vs_original' in result else "-"

                print(f"{config:<20} {policy:<15} {time_str:<12} {gflops_str:<12} {speedup_str:<10}")

def main():
    parser = argparse.ArgumentParser(description='GEMM Benchmark Driver')
    parser.add_argument('--matrix-dir', default='./test_matrices',
                        help='Directory containing test matrices (default: ./test_matrices)')
    parser.add_argument('--binary-original', default='./gemm_test_original',
                        help='Original binary (default: ./gemm_test_original)')
    parser.add_argument('--binary-policy', default='./gemm_test_modify',
                        help='Policy framework binary (default: ./gemm_test_modify)')
    parser.add_argument('--warmup', type=int, default=3,
                        help='Number of warmup runs (default: 3)')
    parser.add_argument('--runs', type=int, default=10,
                        help='Number of benchmark runs (default: 10)')
    parser.add_argument('--output', default='benchmark_results.csv',
                        help='Output CSV file (default: benchmark_results.csv)')

    args = parser.parse_args()

    # Check binaries exist
    if not os.path.exists(args.binary_original):
        print(f"Error: Binary not found: {args.binary_original}")
        return 1

    if not os.path.exists(args.binary_policy):
        print(f"Error: Binary not found: {args.binary_policy}")
        return 1

    # Define policies to test
    policies = {
        'greedy': './policy_greedy.ptx',
        'maxsteals': './policy_maxsteals.ptx',
    }

    # Check policy files exist
    for name, ptx in policies.items():
        if not os.path.exists(ptx):
            print(f"Warning: Policy PTX not found: {ptx}")

    # Run benchmarks
    runner = BenchmarkRunner(
        args.matrix_dir,
        args.binary_original,
        args.binary_policy,
        args.warmup,
        args.runs
    )

    runner.run_all_benchmarks(policies)
    runner.print_summary()
    runner.save_results(args.output)

    print("\n" + "=" * 80)
    print("Benchmark Complete!")
    print("=" * 80)

if __name__ == '__main__':
    main()
