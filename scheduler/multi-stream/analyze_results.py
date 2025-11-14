#!/usr/bin/env python3
"""
GPU Scheduler Experiment Analysis - Main Script

Analyzes results from experiment_driver.py and generates insights.
"""

import argparse
from analyzer_base import BaseAnalyzer
from analyzers_rq1_rq3 import RQ1_RQ3_Analyzer
from analyzers_rq4_rq7 import RQ4_RQ7_Analyzer
from analyzers_rq9_rq11 import RQ9_RQ11_Analyzer


class ResultAnalyzer(RQ1_RQ3_Analyzer, RQ4_RQ7_Analyzer, RQ9_RQ11_Analyzer):
    """Main analyzer that combines all RQ analyzers via multiple inheritance."""
    pass


def main():
    parser = argparse.ArgumentParser(description="Analyze GPU scheduler experiments")
    parser.add_argument("--results", default="results", help="Results directory")
    parser.add_argument("--experiments", "--experiment", nargs="+",
                       choices=["RQ1", "RQ2", "RQ3", "RQ4", "RQ5", "RQ6", "RQ7", "RQ9", "RQ11", "all"],
                       default=["all"], help="Which experiments to analyze")

    args = parser.parse_args()

    print("="*60)
    print("GPU Scheduler Experiment Analysis")
    print("="*60)

    analyzer = ResultAnalyzer(args.results)
    analyses = {}

    exp_map = {
        "RQ1": analyzer.analyze_rq1_stream_scalability,
        "RQ2": analyzer.analyze_rq2_workload_characterization,
        "RQ3": analyzer.analyze_rq3_priority_effectiveness,
        "RQ4": analyzer.analyze_rq4_memory_pressure,
        "RQ5": analyzer.analyze_rq5_multi_process,
        "RQ6": analyzer.analyze_rq6_load_imbalance,
        "RQ7": analyzer.analyze_rq7_tail_latency,
        "RQ9": analyzer.analyze_rq9_priority_tail_latency,
        "RQ11": analyzer.analyze_rq11_bandwidth_partitioning,
    }

    experiments_to_run = list(exp_map.keys()) if "all" in args.experiments else args.experiments

    for exp_name in experiments_to_run:
        if exp_name in exp_map:
            try:
                print(f"\nAnalyzing {exp_name}...")
                result = exp_map[exp_name]()
                analyses[exp_name] = result
            except FileNotFoundError as e:
                print(f"✗ {exp_name}: {e}")
            except Exception as e:
                print(f"✗ {exp_name} failed: {e}")

    # Generate summary
    report = analyzer.generate_summary_report(analyses)
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(report)


if __name__ == "__main__":
    main()
