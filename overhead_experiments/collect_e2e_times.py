#!/usr/bin/env python3
"""
Collect times from e2e_runner.py output logs and generate CSV files.
Supports two metrics:
  - kernel_time: Kernel execution time (from triton_profiler or ENABLE_TIMING)
  - e2e_time: End-to-end wall-clock time

Reads from results/metric/mode/runN/ folders and outputs:
  - kernel_time.csv
  - e2e_time.csv
"""

import re
import csv
from pathlib import Path
import argparse
import statistics


# Script directory
SCRIPT_DIR = Path(__file__).parent.resolve()


def check_exit_code(content):
    """Check if exit code is 0 in log content."""
    match = re.search(r'Exit code:\s*(\d+)', content)
    if match:
        return int(match.group(1)) == 0
    return False


def extract_kernel_time_from_log(log_file):
    """Extract kernel time from a log file (only if exit code is 0)."""
    if not log_file.exists():
        return None

    with open(log_file, 'r') as f:
        content = f.read()

    # Check exit code first
    if not check_exit_code(content):
        return None

    # Look for "Kernel time: X.XXX ms"
    match = re.search(r'Kernel time:\s*([\d.]+)\s*ms', content)
    if match:
        return float(match.group(1))

    return None


def extract_e2e_time_from_log(log_file):
    """Extract E2E time from a log file (only if exit code is 0)."""
    if not log_file.exists():
        return None

    with open(log_file, 'r') as f:
        content = f.read()

    # Check exit code first
    if not check_exit_code(content):
        return None

    # Look for "E2E time: X.XXX ms"
    match = re.search(r'E2E time:\s*([\d.]+)\s*ms', content)
    if match:
        return float(match.group(1))

    return None


def extract_case_name_from_log(log_filename):
    """Extract case name from log filename (e.g., '01_matmul_triton1.log' -> 'matmul_triton1')."""
    name = log_filename.stem
    match = re.match(r'^\d+_(.+)$', name)
    if match:
        return match.group(1)
    return name


def discover_runs(mode_dir):
    """Discover run directories under a mode directory."""
    runs = []
    if mode_dir.exists():
        for run_dir in sorted(mode_dir.glob("run*")):
            if run_dir.is_dir():
                runs.append(run_dir)
    return runs


def collect_times_for_metric(base_dir, metric):
    """Collect times for a specific metric from all three modes across multiple runs."""
    modes = ["baseline", "profiler", "sanitizer"]

    # Select the appropriate extraction function
    if metric == "kernel_time":
        extract_func = extract_kernel_time_from_log
    else:  # e2e_time
        extract_func = extract_e2e_time_from_log

    # metric is in path: base_dir/metric/mode/runN/
    metric_dir = base_dir / metric

    # Discover all case names from all runs
    all_cases = set()

    for mode in modes:
        mode_dir = metric_dir / mode
        for run_dir in discover_runs(mode_dir):
            for log_file in run_dir.glob("*.log"):
                case_name = extract_case_name_from_log(log_file)
                all_cases.add(case_name)

    if not all_cases:
        print(f"No log files found for metric '{metric}' in {metric_dir}")
        return []

    # Sort case names
    all_cases = sorted(all_cases)

    # Collect times for each case
    results = []
    for case_name in all_cases:
        row = {"case": case_name}

        # For each mode, collect times from all runs
        for mode in modes:
            mode_dir = metric_dir / mode
            times = []

            for run_dir in discover_runs(mode_dir):
                # Find log file matching this case
                for log_file in run_dir.glob("*.log"):
                    if extract_case_name_from_log(log_file) == case_name:
                        time_value = extract_func(log_file)
                        if time_value is not None:
                            times.append(time_value)
                        break

            # Calculate statistics
            if times:
                row[f"{mode}_mean"] = statistics.mean(times)
                row[f"{mode}_min"] = min(times)
                row[f"{mode}_max"] = max(times)
                row[f"{mode}_runs"] = len(times)
            else:
                row[f"{mode}_mean"] = None
                row[f"{mode}_min"] = None
                row[f"{mode}_max"] = None
                row[f"{mode}_runs"] = 0

        results.append(row)

    return results


def write_csv(results, output_file, metric):
    """Write results to CSV file."""
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)

        # Write header
        unit = "ms"
        writer.writerow([
            "case",
            f"baseline_mean ({unit})",
            f"profiler_mean ({unit})",
            f"sanitizer_mean ({unit})",
            "profiler_overhead",
            "sanitizer_overhead"
        ])

        # Write data
        for row in results:
            baseline = row["baseline_mean"]
            profiler = row["profiler_mean"]
            sanitizer = row["sanitizer_mean"]

            # Calculate overhead (profiler/baseline, sanitizer/baseline)
            profiler_overhead = ""
            sanitizer_overhead = ""
            if baseline is not None and baseline > 0:
                if profiler is not None:
                    profiler_overhead = f"{profiler/baseline:.2f}x"
                if sanitizer is not None:
                    sanitizer_overhead = f"{sanitizer/baseline:.2f}x"

            writer.writerow([
                row["case"],
                f"{baseline:.3f}" if baseline is not None else "",
                f"{profiler:.3f}" if profiler is not None else "",
                f"{sanitizer:.3f}" if sanitizer is not None else "",
                profiler_overhead,
                sanitizer_overhead
            ])

    print(f"CSV written to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect times from e2e experiment logs and generate CSV files"
    )
    parser.add_argument(
        "--input-dir", "-i",
        type=str,
        default=str(SCRIPT_DIR / "results"),
        help=f"Base directory containing results (default: {SCRIPT_DIR / 'results'})"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=str(SCRIPT_DIR / "results"),
        help=f"Output directory for CSV files (default: {SCRIPT_DIR / 'results'})"
    )
    parser.add_argument(
        "--metric",
        nargs="+",
        choices=["kernel_time", "e2e_time", "all"],
        default=["all"],
        help="Metric(s) to collect: kernel_time, e2e_time, or all (default: all)"
    )
    args = parser.parse_args()

    base_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine metrics to collect
    if "all" in args.metric:
        metrics_to_collect = ["kernel_time", "e2e_time"]
    else:
        metrics_to_collect = args.metric

    print(f"Collecting times from: {base_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Metrics: {', '.join(metrics_to_collect)}")
    print()

    # First pass: collect results for all metrics and find valid cases per metric
    all_metric_results = {}
    valid_cases_per_metric = {}

    for metric in metrics_to_collect:
        print(f"{'=' * 60}")
        print(f"Processing metric: {metric}")
        print(f"{'=' * 60}")
        print(f"  Looking in: {base_dir / metric}/")
        print(f"    - baseline/run*/")
        print(f"    - profiler/run*/")
        print(f"    - sanitizer/run*/")
        print()

        results = collect_times_for_metric(base_dir, metric)

        if not results:
            print(f"  No results found for metric '{metric}'")
            continue

        # Print summary before filtering
        print(f"  Found {len(results)} test cases (before filtering)")
        for mode in ["baseline", "profiler", "sanitizer"]:
            count = sum(1 for r in results if r[f"{mode}_mean"] is not None)
            print(f"    - {mode}: {count} results")

        # Filter to only keep cases where all three modes have valid data (time > 0)
        filtered_results = [
            r for r in results
            if r["baseline_mean"] is not None and r["baseline_mean"] > 0
            and r["profiler_mean"] is not None and r["profiler_mean"] > 0
            and r["sanitizer_mean"] is not None and r["sanitizer_mean"] > 0
        ]

        print(f"  After filtering (all modes passed): {len(filtered_results)} test cases")
        print()

        all_metric_results[metric] = {r["case"]: r for r in filtered_results}
        valid_cases_per_metric[metric] = set(r["case"] for r in filtered_results)

    # Find intersection of valid cases across all metrics
    if len(valid_cases_per_metric) > 1:
        common_cases = set.intersection(*valid_cases_per_metric.values())
        print(f"{'=' * 60}")
        print(f"Cross-metric filtering")
        print(f"{'=' * 60}")
        for metric, cases in valid_cases_per_metric.items():
            print(f"  {metric}: {len(cases)} valid cases")
        print(f"  Common cases (intersection): {len(common_cases)}")
        print()
    else:
        common_cases = list(valid_cases_per_metric.values())[0] if valid_cases_per_metric else set()

    # Write CSVs with only common cases
    for metric in metrics_to_collect:
        if metric not in all_metric_results:
            continue

        # Filter to only common cases
        final_results = [
            all_metric_results[metric][case]
            for case in sorted(common_cases)
            if case in all_metric_results[metric]
        ]

        csv_file = output_dir / f"{metric}.csv"
        write_csv(final_results, csv_file, metric)
        print()

    print(f"{'=' * 60}")
    print("Done!")
    print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    exit(main())
