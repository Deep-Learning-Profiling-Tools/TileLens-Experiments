#!/usr/bin/env python3
"""
Profiler K-Scaling experiment runner.
Tests different values of PROFILER_BLOCK_SAMPLING_K (0, 1, 5, 10) and measures
both e2e_time and kernel_time for each configuration.

K=0 means block sampling is disabled (PROFILER_ENABLE_BLOCK_SAMPLING=0).

Only runs profiler mode (no baseline or sanitizer).
"""

import os
import subprocess
import time
import re
from pathlib import Path
from datetime import datetime
import sys
import argparse
from typing import List, Tuple, Dict, Any

# Add utils to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from test_registry import (
    load_registry, discover_tests, REPO_CONFIGS,
    TRITONBENCH_DIR, DEFAULT_REGISTRY
)

# Script directory
SCRIPT_DIR = Path(__file__).parent.resolve()

# K values to test
K_VALUES = [0, 1, 5, 10]


def parse_triton_viz_timing(output: str) -> Tuple[float, int]:
    """
    Parse Triton-Viz execution times from output.
    Example line: Triton-Viz: execution time for _kernel_name: 3.326 ms

    Returns:
        Tuple of (total_ms, kernel_count)
    """
    pattern = r'Triton-Viz:\s+execution time for\s+(\S+):\s+([\d.]+)\s+ms'

    total_ms = 0.0
    count = 0

    for line in output.splitlines():
        match = re.search(pattern, line)
        if match:
            exec_time = float(match.group(2))
            total_ms += exec_time
            count += 1

    return total_ms, count


def run_profiler_with_k(
    test: Dict[str, Any],
    k_value: int,
    output_base_dir: Path,
    global_id: int,
    total_registry: int,
    current: int,
    total_current: int,
    run_number: int
) -> Tuple[bool, float, float]:
    """
    Run triton-profiler with a specific PROFILER_BLOCK_SAMPLING_K value.
    Measures both kernel_time and e2e_time in a single run.

    Returns:
        Tuple of (success, kernel_time_ms, e2e_time_ms)
    """
    file_path = test["file_path"]
    test_name = test["name"]
    is_pytest = test["is_pytest"]
    test_function = test.get("test_function")

    safe_name = test_name.replace("::", "__")
    id_str = str(global_id).zfill(len(str(total_registry)))
    output_filename = f"{id_str}_{safe_name}.log"

    # Create output directory: results/k{k_value}/run{run_number}/
    output_dir = output_base_dir / f"k{k_value}" / f"run{run_number}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Environment variables for profiler
    # Note: PROFILER_ENABLE_LOAD_STORE_SKIPPING is disabled to match ablation study baseline
    # K=0 means disable block sampling entirely
    env = os.environ.copy()
    env.update({
        "TRITON_INTERPRET": "1",
        "ENABLE_TIMING": "1",
        "PROFILER_ENABLE_LOAD_STORE_SKIPPING": "0",
        "PROFILER_DISABLE_BUFFER_LOAD_CHECK": "1",
        "SANITIZER_ENABLE_FAKE_TENSOR": "0"
    })

    if k_value == 0:
        # K=0: disable block sampling
        env["PROFILER_ENABLE_BLOCK_SAMPLING"] = "0"
    else:
        # K>0: set the sampling K value (block sampling is enabled by default)
        env["PROFILER_BLOCK_SAMPLING_K"] = str(k_value)

    if is_pytest:
        test_spec = f"{file_path.name}::{test_function}" if test_function else file_path.name
        cmd = ["triton-profiler", "pytest", "-s", "--assert=plain", test_spec]
        cwd = file_path.parent
    else:
        cmd = ["triton-profiler", str(file_path)]
        cwd = TRITONBENCH_DIR

    print(f"  [ID:{id_str}] ({current}/{total_current}) Running {test_name} with K={k_value}...")

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=300
        )

        elapsed_time = time.time() - start_time
        elapsed_time_ms = elapsed_time * 1000

        output = result.stdout + "\n" + result.stderr
        kernel_time_ms, kernel_count = parse_triton_viz_timing(output)

        # Write single log file with both metrics
        with open(output_dir / output_filename, 'w') as f:
            f.write(f"Test: {test_name}\n")
            f.write(f"Mode: profiler (triton-profiler)\n")
            f.write(f"PROFILER_BLOCK_SAMPLING_K: {k_value}\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Start time: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")
            f.write(output)
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Exit code: {result.returncode}\n")
            f.write(f"Kernel time: {kernel_time_ms:.3f} ms ({kernel_count} kernel calls)\n")
            f.write(f"E2E time: {elapsed_time_ms:.3f} ms ({elapsed_time:.3f} s)\n")

        if result.returncode == 0:
            print(f"    [OK] Kernel: {kernel_time_ms:.3f} ms ({kernel_count} kernels), E2E: {elapsed_time_ms:.3f} ms")
        else:
            print(f"    [FAIL] Exit code {result.returncode}, Kernel: {kernel_time_ms:.3f} ms, E2E: {elapsed_time_ms:.3f} ms")

        return result.returncode == 0, kernel_time_ms, elapsed_time_ms

    except subprocess.TimeoutExpired:
        elapsed_time = time.time() - start_time
        elapsed_time_ms = elapsed_time * 1000
        with open(output_dir / output_filename, 'w') as f:
            f.write(f"Test: {test_name}\n")
            f.write(f"Mode: profiler (triton-profiler)\n")
            f.write(f"PROFILER_BLOCK_SAMPLING_K: {k_value}\n")
            f.write("=" * 80 + "\n")
            f.write(f"TIMEOUT: Test exceeded 300 seconds\n")
            f.write(f"E2E time: {elapsed_time_ms:.3f} ms\n")
        print(f"    [TIMEOUT] Timeout")
        return False, 0.0, elapsed_time_ms

    except Exception as e:
        elapsed_time = time.time() - start_time
        elapsed_time_ms = elapsed_time * 1000
        with open(output_dir / output_filename, 'w') as f:
            f.write(f"Test: {test_name}\n")
            f.write(f"Mode: profiler (triton-profiler)\n")
            f.write(f"PROFILER_BLOCK_SAMPLING_K: {k_value}\n")
            f.write("=" * 80 + "\n")
            f.write(f"ERROR: {str(e)}\n")
            f.write(f"E2E time: {elapsed_time_ms:.3f} ms\n")
        print(f"    [ERROR] Error: {str(e)}")
        return False, 0.0, elapsed_time_ms


def run_k_value(
    k_value: int,
    tests: List[Dict[str, Any]],
    output_base_dir: Path,
    run_number: int,
    total_runs: int,
    total_registry_tests: int
) -> List[Dict[str, Any]]:
    """Run tests for a specific K value."""
    print(f"\n{'=' * 60}")
    print(f"Running with PROFILER_BLOCK_SAMPLING_K={k_value} (run {run_number}/{total_runs})")
    print(f"Output directory: {output_base_dir}")
    print(f"{'=' * 60}\n")

    results = []
    num_tests = len(tests)

    for i, test in enumerate(tests, 1):
        global_id = test["global_id"]
        success, kernel_time_ms, e2e_time_ms = run_profiler_with_k(
            test, k_value, output_base_dir, global_id,
            total_registry_tests, i, num_tests, run_number
        )
        results.append({
            "name": test["name"],
            "global_id": global_id,
            "k_value": k_value,
            "success": success,
            "kernel_time_ms": kernel_time_ms,
            "e2e_time_ms": e2e_time_ms
        })

    # Print summary for this K value
    successful = sum(1 for r in results if r["success"])
    print(f"\n  K={k_value} summary: {successful}/{num_tests} tests passed")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Profiler K-Scaling experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all K values on TritonBench
  python k_scaling_runner.py --repo tritonbench

  # Run specific K values
  python k_scaling_runner.py --k 1 5

  # Run single case
  python k_scaling_runner.py --case matmul_triton1

  # Run 3 times instead of default 5
  python k_scaling_runner.py --runs 3
        """
    )
    parser.add_argument(
        "--repo",
        choices=["tritonbench", "liger_kernel"],
        default="tritonbench",
        help="Repository to run tests from (default: tritonbench)"
    )
    parser.add_argument(
        "--k",
        nargs="+",
        type=int,
        default=K_VALUES,
        help=f"K values to test (default: {K_VALUES})"
    )
    parser.add_argument(
        "--registry", "-w",
        type=str,
        help=f"Path to test registry file (default: {DEFAULT_REGISTRY})"
    )
    parser.add_argument(
        "--case", "-c",
        type=str,
        help="Run a single test case (e.g., matmul_triton1)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        help="Base output directory for results (default: results/)"
    )
    parser.add_argument(
        "--runs", "-r",
        type=int,
        default=1,
        help="Number of runs for each K value (default: 1)"
    )
    args = parser.parse_args()

    k_values = args.k

    # Setup output directory
    if args.output_dir:
        output_base_dir = Path(args.output_dir)
    else:
        output_base_dir = SCRIPT_DIR / "results"

    # Load test registry
    registry_file = Path(args.registry) if args.registry else DEFAULT_REGISTRY
    if not registry_file.exists():
        print(f"Error: Test registry file not found: {registry_file}")
        return 1

    registry = load_registry(registry_file)
    if not registry:
        print(f"Error: No tests found in registry: {registry_file}")
        return 1

    print(f"Loaded test registry from {registry_file}: {len(registry)} total entries")

    # Discover tests for the specified repo
    tests = discover_tests(args.repo, registry, case=args.case)
    if not tests:
        print("No tests to process")
        return 1

    # Total tests in registry (for consistent filename width)
    total_registry_tests = len(registry)

    # Print summary
    id_range = f"{tests[0]['global_id']}-{tests[-1]['global_id']}" if tests else "N/A"
    print(f"\n{'=' * 60}")
    print(f"Profiler K-Scaling Experiment Runner")
    print(f"{'=' * 60}")
    print(f"Repository: {args.repo}")
    print(f"Tests to run: {len(tests)} (ID range: {id_range})")
    print(f"Total tests in registry: {total_registry_tests}")
    print(f"K values to test: {k_values}")
    print(f"Runs per K value: {args.runs}")
    print(f"Output base directory: {output_base_dir}")
    print(f"{'=' * 60}")

    # Print environment variables for each K value
    print(f"\nEnvironment Variables Configuration:")
    print(f"{'-' * 60}")
    print(f"  (PROFILER_ENABLE_LOAD_STORE_SKIPPING is disabled to match ablation baseline)")
    for k in k_values:
        print(f"\n  K={k}:")
        print(f"    TRITON_INTERPRET=1")
        print(f"    ENABLE_TIMING=1")
        print(f"    PROFILER_ENABLE_LOAD_STORE_SKIPPING=0")
        if k == 0:
            print(f"    PROFILER_ENABLE_BLOCK_SAMPLING=0  (disable block sampling)")
        else:
            print(f"    PROFILER_BLOCK_SAMPLING_K={k}")
        print(f"    PROFILER_DISABLE_BUFFER_LOAD_CHECK=1")
        print(f"    SANITIZER_ENABLE_FAKE_TENSOR=1")

    # Print total experiment count
    total_experiments = len(tests) * len(k_values) * args.runs
    print(f"\nTotal experiments: {total_experiments} runs")
    print(f"  ({len(tests)} tests x {len(k_values)} K values x {args.runs} runs)")
    print(f"\n{'=' * 60}")

    # Run tests for each K value
    all_results = {k: {"kernel_time": [], "e2e_time": []} for k in k_values}

    for k_value in k_values:
        for run_num in range(1, args.runs + 1):
            results = run_k_value(
                k_value, tests, output_base_dir,
                run_num, args.runs, total_registry_tests
            )
            for r in results:
                all_results[k_value]["kernel_time"].append({
                    "name": r["name"],
                    "global_id": r["global_id"],
                    "success": r["success"],
                    "time_ms": r["kernel_time_ms"]
                })
                all_results[k_value]["e2e_time"].append({
                    "name": r["name"],
                    "global_id": r["global_id"],
                    "success": r["success"],
                    "time_ms": r["e2e_time_ms"]
                })

    # Print overall summary
    print(f"\n{'=' * 60}")
    print("Overall Summary")
    print(f"{'=' * 60}")

    for k_value in k_values:
        print(f"\n  K={k_value}:")
        print(f"  {'-' * 50}")
        for metric in ["kernel_time", "e2e_time"]:
            results_list = all_results[k_value][metric]
            total_successful = sum(1 for r in results_list if r["success"])
            total_tests = len(results_list)
            total_time = sum(r["time_ms"] for r in results_list)
            if total_tests > 0:
                print(f"    {metric:15} {total_successful:3}/{total_tests:3} passed ({total_successful*100/total_tests:.1f}%) - Total: {total_time:.3f} ms")
            else:
                print(f"    {metric:15} No results")

    print(f"\n{'=' * 60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
