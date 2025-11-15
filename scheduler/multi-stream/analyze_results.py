#!/usr/bin/env python3
"""
GPU Scheduler Experiment Analysis - Main Script

Analyzes results from experiment_driver.py and generates insights and figures
for all research questions based on RESEARCH_QUESTIONS.md.

Usage:
    python analyze_results.py --results-dir results/ --output-dir figures/
    python analyze_results.py --results-dir results/ --rq RQ1
"""

import argparse
import sys
from pathlib import Path
from analyzers_rq1_rq3 import RQ1_RQ3_Analyzer
from analyzers_rq4_rq8 import RQ4_RQ8_Analyzer


def main():
    parser = argparse.ArgumentParser(
        description='GPU Scheduler Experiment Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all RQs
  python analyze_results.py --results-dir results/ --output-dir figures/

  # Analyze specific RQ
  python analyze_results.py --results-dir results/ --rq RQ1

  # Generate only specific sub-RQs
  python analyze_results.py --results-dir results/ --rq RQ1 --sub-rq 1.1,1.2
"""
    )

    parser.add_argument(
        '--results-dir',
        type=str,
        default='results',
        help='Directory containing experiment CSV files (default: results/)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='figures',
        help='Output directory for figures (default: figures/)'
    )

    parser.add_argument(
        '--rq',
        type=str,
        default='all',
        help='Research question to analyze: all, RQ1, RQ2, RQ3, RQ4, RQ5, RQ6, RQ7, RQ8'
    )

    parser.add_argument(
        '--sub-rq',
        type=str,
        default=None,
        help='Specific sub-RQs to run (comma-separated, e.g., "1.1,1.2")'
    )

    parser.add_argument(
        '--format',
        type=str,
        default='png',
        choices=['png', 'pdf', 'svg'],
        help='Output figure format (default: png)'
    )

    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='Figure DPI for PNG output (default: 300)'
    )

    args = parser.parse_args()

    # Validate directories
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse sub-RQs if specified
    sub_rqs = None
    if args.sub_rq:
        sub_rqs = [s.strip() for s in args.sub_rq.split(',')]

    rq = args.rq.upper()

    # Create analyzers
    rq1_3_analyzer = RQ1_RQ3_Analyzer(
        results_dir=results_dir,
        output_dir=output_dir,
        fig_format=args.format,
        dpi=args.dpi
    )

    rq4_8_analyzer = RQ4_RQ8_Analyzer(
        results_dir=results_dir,
        output_dir=output_dir,
        fig_format=args.format,
        dpi=args.dpi
    )

    # Run analysis based on RQ selection
    if rq == 'ALL':
        print("Analyzing all research questions...")
        rq1_3_analyzer.analyze_all()
        rq4_8_analyzer.analyze_all()
    elif rq == 'RQ1':
        rq1_3_analyzer.analyze_rq1(sub_rqs)
    elif rq == 'RQ2':
        rq1_3_analyzer.analyze_rq2(sub_rqs)
    elif rq == 'RQ3':
        rq1_3_analyzer.analyze_rq3(sub_rqs)
    elif rq == 'RQ4':
        rq4_8_analyzer.analyze_rq4(sub_rqs)
    elif rq == 'RQ5':
        rq4_8_analyzer.analyze_rq5(sub_rqs)
    elif rq == 'RQ6':
        rq4_8_analyzer.analyze_rq6(sub_rqs)
    elif rq == 'RQ7':
        rq4_8_analyzer.analyze_rq7(sub_rqs)
    elif rq == 'RQ8':
        rq4_8_analyzer.analyze_rq8(sub_rqs)
    else:
        print(f"Error: Unknown RQ '{args.rq}'. Valid options: all, RQ1-RQ8")
        sys.exit(1)

    print(f"\nAnalysis complete! Figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
