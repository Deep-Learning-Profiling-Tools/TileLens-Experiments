#!/usr/bin/env python3
"""
Plot overhead CDF for kernel_time and e2e_time metrics.
Generates two PDF files:
  - kernel_time_overhead.pdf
  - e2e_time_overhead.pdf
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Script directory
SCRIPT_DIR = Path(__file__).parent.resolve()

# --- 设置符合OSDI/学术风格的绘图参数 ---
plt.rcParams.update({
    'font.size': 14,
    'font.family': 'sans-serif',
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 22,
    'axes.linewidth': 1.5,
    'lines.linewidth': 2.5,
    'grid.linewidth': 1.0,
    'grid.linestyle': '--',
    'grid.color': '#aaaaaa',
    'figure.figsize': (10, 5.5),
    'figure.autolayout': True,
})


def calculate_cdf(data):
    """Calculate CDF for given data."""
    sorted_data = np.sort(data)
    yvals = np.arange(1, len(sorted_data) + 1) / float(len(sorted_data))
    return sorted_data, yvals


def plot_overhead_cdf(csv_path, output_pdf, title_suffix="", x_min_limit=None, x_max_limit=None):
    """Plot overhead CDF from a CSV file and save to PDF.

    Args:
        x_min_limit: Optional minimum x-axis value (None = auto).
        x_max_limit: Optional maximum x-axis value (None = auto).
    """
    if not csv_path.exists():
        print(f"Warning: CSV file not found: {csv_path}")
        return False

    # Read CSV
    df = pd.read_csv(csv_path)

    # Get column names (they contain units like "ms")
    baseline_col = [c for c in df.columns if 'baseline_mean' in c][0]
    profiler_col = [c for c in df.columns if 'profiler_mean' in c][0]
    sanitizer_col = [c for c in df.columns if 'sanitizer_mean' in c][0]

    # Filter out rows with missing or zero baseline
    df = df.dropna(subset=[baseline_col, profiler_col, sanitizer_col])
    df = df[df[baseline_col] > 0]

    if len(df) == 0:
        print(f"Warning: No valid data in {csv_path}")
        return False

    # Calculate normalized overhead
    sanitizer_overhead = df[sanitizer_col].values / df[baseline_col].values
    profiler_overhead = df[profiler_col].values / df[baseline_col].values

    print(f"\n{'=' * 60}")
    print(f"Processing: {csv_path.name}")
    print(f"{'=' * 60}")
    print(f"Loaded {len(df)} workloads")
    print(f"| {'Tool':<12} | {'Median':<12} | {'90th Pctl':<12} | {'Range':<28} |")
    print(f"|{'-'*14}|{'-'*14}|{'-'*14}|{'-'*30}|")
    san_median = np.median(sanitizer_overhead)
    san_p90 = np.percentile(sanitizer_overhead, 90)
    pro_median = np.median(profiler_overhead)
    pro_p90 = np.percentile(profiler_overhead, 90)
    print(f"| {'Sanitizer':<12} | {san_median:<12.2f} | {san_p90:<12.2f} | {sanitizer_overhead.min():.2f}x–{sanitizer_overhead.max():.2f}x |")
    print(f"| {'Profiler':<12} | {pro_median:<12.2f} | {pro_p90:<12.2f} | {profiler_overhead.min():.2f}x–{profiler_overhead.max():.2f}x |")

    # Calculate CDF
    san_x, san_y = calculate_cdf(sanitizer_overhead)
    pro_x, pro_y = calculate_cdf(profiler_overhead)

    # Create plot
    fig, ax = plt.subplots()

    # Plot lines
    ax.plot(san_x, san_y, label='Sanitizer', color='#005EB8', linestyle='-')
    ax.plot(pro_x, pro_y, label='Profiler', color='#C41E3A', linestyle='--')

    # Add baseline reference line
    ax.axvline(x=1.0, color='black', linestyle='-', linewidth=1.5, label='1.0x')

    # Style adjustments
    ax.set_xlabel('Overhead', fontsize=42)
    ax.set_ylabel('CDF', fontsize=42)
    ax.tick_params(axis='both', labelsize=39)
    ax.set_ylim(-0.05, 1.05)

    # Auto-adjust x-axis based on data range (or use limits if specified)
    max_overhead = max(sanitizer_overhead.max(), profiler_overhead.max())
    min_overhead = min(sanitizer_overhead.min(), profiler_overhead.min())
    if x_min_limit is not None:
        x_min = x_min_limit
    else:
        x_min = max(0, min_overhead * 0.9)
    if x_max_limit is not None:
        x_max = x_max_limit
    else:
        x_max = max_overhead * 1.1
    ax.set_xlim(x_min, x_max)

    # Set Y-axis to percentage
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])

    ax.grid(True)
    ax.legend(loc='lower right', frameon=True, framealpha=0.9, edgecolor='gray', fontsize=42)

    # Save to PDF
    plt.savefig(output_pdf, bbox_inches='tight')
    plt.close()

    print(f"Saved to: {output_pdf}")

    # Generate separate PDFs for sanitizer and profiler
    output_pdf_path = Path(output_pdf)
    base_name = output_pdf_path.stem  # e.g., "e2e_time_overhead"

    # Sanitizer plot
    fig, ax = plt.subplots()
    ax.plot(san_x, san_y, color='#005EB8', linestyle='-')
    ax.axvline(x=1.0, color='black', linestyle='-', linewidth=1.5)
    ax.set_xlabel('Overhead', fontsize=42)
    ax.set_ylabel('CDF', fontsize=42)
    ax.tick_params(axis='both', labelsize=39)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(x_min, sanitizer_overhead.max() * 1.1 if x_max_limit is None else x_max_limit)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
    ax.grid(True)
    sanitizer_pdf = output_pdf_path.parent / f"{base_name}_sanitizer.pdf"
    plt.savefig(sanitizer_pdf, bbox_inches='tight')
    plt.close()
    print(f"Saved to: {sanitizer_pdf}")

    # Profiler plot
    fig, ax = plt.subplots()
    ax.plot(pro_x, pro_y, color='#C41E3A', linestyle='--')
    ax.axvline(x=1.0, color='black', linestyle='-', linewidth=1.5)
    ax.set_xlabel('Overhead', fontsize=42)
    ax.set_ylabel('CDF', fontsize=42)
    ax.tick_params(axis='both', labelsize=39)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(x_min, profiler_overhead.max() * 1.1 if x_max_limit is None else x_max_limit)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
    ax.grid(True)
    profiler_pdf = output_pdf_path.parent / f"{base_name}_profiler.pdf"
    plt.savefig(profiler_pdf, bbox_inches='tight')
    plt.close()
    print(f"Saved to: {profiler_pdf}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Plot overhead CDF for kernel_time and e2e_time metrics"
    )
    parser.add_argument(
        "--kernel-xmin", type=float, default=None,
        help="X-axis minimum for kernel_time plot (default: auto)"
    )
    parser.add_argument(
        "--kernel-xmax", type=float, default=None,
        help="X-axis maximum for kernel_time plot (default: auto)"
    )
    parser.add_argument(
        "--e2e-xmin", type=float, default=None,
        help="X-axis minimum for e2e_time plot (default: auto)"
    )
    parser.add_argument(
        "--e2e-xmax", type=float, default=None,
        help="X-axis maximum for e2e_time plot (default: auto)"
    )
    args = parser.parse_args()

    results_dir = SCRIPT_DIR / "results"

    # Define metrics and their corresponding files
    metrics = [
        {
            "name": "kernel_time",
            "csv": results_dir / "kernel_time.csv",
            "pdf": SCRIPT_DIR / "kernel_time_overhead.pdf",
            "x_min": args.kernel_xmin,
            "x_max": 2500,
        },
        {
            "name": "e2e_time",
            "csv": results_dir / "e2e_time.csv",
            "pdf": SCRIPT_DIR / "e2e_time_overhead.pdf",
            "x_min": args.e2e_xmin,
            "x_max": args.e2e_xmax,
        }
    ]

    success_count = 0
    for metric in metrics:
        if plot_overhead_cdf(
            metric["csv"], metric["pdf"], metric["name"],
            x_min_limit=metric["x_min"], x_max_limit=metric["x_max"]
        ):
            success_count += 1

    print(f"\n{'=' * 60}")
    print(f"Generated {success_count * 3}/{len(metrics) * 3} PDF files")
    print(f"{'=' * 60}")

    return 0 if success_count > 0 else 1


if __name__ == "__main__":
    exit(main())
