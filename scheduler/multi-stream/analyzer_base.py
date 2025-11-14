#!/usr/bin/env python3
"""
Base Analyzer Class

Provides common functionality for all RQ analyzers.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class BaseAnalyzer:
    """Base class for analyzing experimental results."""

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.figures_dir = self.results_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)

    def load_data(self, filename: str) -> pd.DataFrame:
        """Load experimental data."""
        filepath = self.results_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Results file not found: {filepath}")

        df = pd.read_csv(filepath)

        # Convert numeric columns
        numeric_cols = ['wall_time_ms', 'throughput', 'mean_lat', 'p50', 'p95', 'p99',
                       'concurrent_rate', 'overhead', 'util', 'jains_index',
                       'max_concurrent', 'avg_concurrent', 'inversions']

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    def _plot_with_error(self, df, x_col, y_col, ax, xlabel, ylabel):
        """Plot with error bars."""
        grouped = df.groupby(x_col)[y_col].agg(['mean', 'std']).reset_index()
        ax.errorbar(grouped[x_col], grouped['mean'], yerr=grouped['std'],
                    fmt='o-', linewidth=2, markersize=8, capsize=5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    def _find_saturation_point(self, series, threshold=0.05):
        """Find saturation point where growth rate drops below threshold."""
        values = series.values
        for i in range(1, len(values)):
            growth_rate = (values[i] - values[i-1]) / values[i-1]
            if growth_rate < threshold:
                return series.index[i]
        return series.index[-1]

    def generate_summary_report(self, analyses: Dict) -> str:
        """Generate markdown summary report."""
        report = "# GPU Scheduler Experiment Results\n\n"
        report += f"Generated: {pd.Timestamp.now()}\n\n"
        report += "## Key Findings\n\n"

        for exp_name, results in analyses.items():
            if results:
                report += f"### {exp_name}\n"
                for key, value in results.items():
                    if isinstance(value, (int, float)):
                        report += f"- {key}: {value:.2f}\n"
                    else:
                        report += f"- {key}: {value}\n"
                report += "\n"

        report += "## Visualizations\n\n"
        report += f"All figures saved to: `{self.figures_dir}/`\n"

        return report
