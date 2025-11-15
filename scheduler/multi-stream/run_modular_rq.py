#!/usr/bin/env python3
"""
Modular GPU Scheduler Research Questions Driver

This script runs experiments and analyses for GPU scheduler research questions
using the new modular structure.
"""

import argparse
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from rq_modules import RQ1, RQ2, RQ3, RQ4, RQ5, RQ6, RQ7, RQ8, RQ9


def main():
    parser = argparse.ArgumentParser(
        description='Run GPU scheduler research question experiments and analysis'
    )

    parser.add_argument(
        '--rq',
        type=str,
        default='all',
        help='Which RQ to run: 4, 5, or "all" (default: all)'
    )

    parser.add_argument(
        '--mode',
        type=str,
        default='both',
        choices=['experiments', 'analyze', 'both'],
        help='Run experiments, analysis, or both (default: both)'
    )

    parser.add_argument(
        '--bench-path',
        type=str,
        default='./multi_stream_bench',
        help='Path to multi_stream_bench executable (default: ./multi_stream_bench)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Directory for raw output files (default: output)'
    )

    parser.add_argument(
        '--results-dir',
        type=str,
        default='results',
        help='Directory for results CSV files (default: results)'
    )

    parser.add_argument(
        '--figures-dir',
        type=str,
        default='figures',
        help='Directory for generated figures (default: figures)'
    )

    parser.add_argument(
        '--num-runs',
        type=int,
        default=3,
        help='Number of runs per experiment (default: 3)'
    )

    args = parser.parse_args()

    # Convert paths
    bench_path = args.bench_path
    output_dir = Path(args.output_dir)
    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)

    # Determine which RQs to run
    rqs_to_run = []
    if args.rq == 'all':
        rqs_to_run = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
    else:
        rqs_to_run = args.rq.split(',')

    print(f"\n{'='*70}")
    print(f"Modular GPU Scheduler Research Questions")
    print(f"{'='*70}")
    print(f"Benchmark: {bench_path}")
    print(f"Output dir: {output_dir}")
    print(f"Results dir: {results_dir}")
    print(f"Figures dir: {figures_dir}")
    print(f"Num runs: {args.num_runs}")
    print(f"Mode: {args.mode}")
    print(f"RQs: {', '.join(rqs_to_run)}")
    print(f"{'='*70}\n")

    # Create RQ instances
    rq_instances = {}
    rq_classes = {
        '1': (RQ1, "Stream Scalability & Concurrency"),
        '2': (RQ2, "Throughput & Workload Type"),
        '3': (RQ3, "Latency & Queueing"),
        '4': (RQ4, "Priority Semantics"),
        '5': (RQ5, "Preemption Latency Analysis"),
        '6': (RQ6, "Heterogeneity & Load Imbalance"),
        '7': (RQ7, "Arrival Pattern & Jitter"),
        '8': (RQ8, "Working Set vs L2 Cache"),
        '9': (RQ9, "Multi-Process vs Single-Process"),
    }

    for rq_num in rqs_to_run:
        if rq_num in rq_classes:
            rq_class, description = rq_classes[rq_num]
            print(f"Initializing RQ{rq_num}: {description}...")
            rq_instances[rq_num] = rq_class(
                bench_path=bench_path,
                output_dir=output_dir,
                results_dir=results_dir,
                figures_dir=figures_dir,
                num_runs=args.num_runs
            )
        else:
            print(f"Warning: Unknown RQ number: {rq_num}")

    print()

    # Run experiments
    if args.mode in ['experiments', 'both']:
        for rq_num, rq in rq_instances.items():
            print(f"\n{'='*70}")
            print(f"Running RQ{rq_num} Experiments")
            print(f"{'='*70}\n")
            try:
                rq.run_experiments()
            except Exception as e:
                print(f"\n❌ Error running RQ{rq_num} experiments: {e}")
                import traceback
                traceback.print_exc()

    # Run analysis
    if args.mode in ['analyze', 'both']:
        for rq_num, rq in rq_instances.items():
            print(f"\n{'='*70}")
            print(f"Analyzing RQ{rq_num}")
            print(f"{'='*70}\n")
            try:
                rq.analyze()
            except Exception as e:
                print(f"\n❌ Error analyzing RQ{rq_num}: {e}")
                import traceback
                traceback.print_exc()

    print(f"\n{'='*70}")
    print("✅ All requested tasks completed!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
