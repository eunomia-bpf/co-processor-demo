"""
Base class for Research Question modules.

Each RQ module should inherit from this and implement:
- run_experiments(): Run the experiments and save data
- analyze(): Analyze data and generate figures
"""

import subprocess
import time
from pathlib import Path
from typing import List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class RQBase:
    """Base class for Research Question modules."""

    def __init__(self, bench_path: str, output_dir: Path, results_dir: Path,
                 figures_dir: Path, num_runs: int = 3):
        self.bench_path = bench_path
        self.output_dir = Path(output_dir)
        self.results_dir = Path(results_dir)
        self.figures_dir = Path(figures_dir)
        self.num_runs = num_runs

        # Ensure directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        # Set up plotting style
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=1.3)

    def run_benchmark(self, args: List[str], first_run: bool = True, timeout: int = 120) -> Optional[str]:
        """
        Run benchmark and capture CSV output from stderr.

        Args:
            args: Command line arguments for benchmark
            first_run: Not used anymore (kept for API compatibility)
            timeout: Timeout in seconds

        Returns:
            Clean CSV output string (header + data row) or None if failed
        """
        cmd = [self.bench_path] + args

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            if result.returncode != 0:
                print(f"Warning: Benchmark failed with return code {result.returncode}")
                print(f"stderr: {result.stderr[:500]}")
                return None

            # Extract clean CSV from stderr
            # The benchmark outputs CSV in format:
            # streams,kernels_per_stream,...
            # 8,20,...
            lines = result.stderr.splitlines()
            header = None
            data_rows = []

            for line in lines:
                line = line.strip()
                # Look for the CSV header line (starts with "streams,")
                if line.startswith('streams,'):
                    header = line
                # Look for CSV data lines (start with a digit)
                elif line and line[0].isdigit():
                    data_rows.append(line)

            if header is None:
                print("Warning: No CSV header found in benchmark output")
                return None

            if not data_rows:
                print("Warning: No CSV data rows found in benchmark output")
                return None

            # Return header + all data rows
            return header + '\n' + '\n'.join(data_rows)

        except subprocess.TimeoutExpired:
            print(f"Warning: Benchmark timed out after {timeout}s")
            return None
        except Exception as e:
            print(f"Warning: Benchmark failed with exception: {e}")
            return None

    def save_csv(self, csv_lines: List[str], filename: str):
        """Save CSV lines to file."""
        csv_path = self.results_dir / filename
        with open(csv_path, 'w') as f:
            f.write('\n'.join(csv_lines))
        print(f"Saved: {csv_path}")

    def load_csv(self, filename: str) -> Optional[pd.DataFrame]:
        """Load CSV file from results directory."""
        filepath = self.results_dir / filename
        if not filepath.exists():
            print(f"Warning: File not found: {filepath}")
            return None

        try:
            df = pd.read_csv(filepath)
            # Add 'size' column derived from grid_size if not present
            # The CSV has grid_size which determines workload_size (grid_size * block_size)
            if 'grid_size' in df.columns and 'size' not in df.columns:
                df['size'] = df['grid_size'] * df.get('block_size', 256)
            elif 'workload_size' in df.columns and 'size' not in df.columns:
                df['size'] = df['workload_size']
            return df
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None

    def save_figure(self, filename: str, dpi: int = 300):
        """Save current figure to figures directory."""
        filepath = self.figures_dir / f"{filename}.png"
        plt.tight_layout()
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
        print(f"Saved figure: {filepath}")
        plt.close()

    def run_experiments(self):
        """Run experiments for this RQ. Must be implemented by subclass."""
        raise NotImplementedError("Subclass must implement run_experiments()")

    def analyze(self):
        """Analyze results and generate figures. Must be implemented by subclass."""
        raise NotImplementedError("Subclass must implement analyze()")
