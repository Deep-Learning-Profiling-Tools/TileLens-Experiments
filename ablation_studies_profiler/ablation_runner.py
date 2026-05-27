#!/usr/bin/env python3
"""
Profiler ablation runner for TritonBench and Liger-Kernel files.
Runs triton-profiler on selected files with configurable environment variables.

Supports two repositories:
  - tritonbench: Run Python files directly
  - liger_kernel: Run pytest tests (test_file.py::test_function format)
"""

import os
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
import sys
import argparse
from collections import OrderedDict
from typing import Dict, Any, List

# Add utils to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from test_registry import (
    load_registry, discover_tests, get_test_dir, list_available_repos,
    TRITONBENCH_DIR
)

# Test name prefixes to exclude
EXCLUDED_TEST_PREFIXES = ["destindex_copy"]

# Profiler configurations with different environment variables
# Environment variables are aligned with k_scaling_runner.py for consistency
PROFILER_CONFIGS = OrderedDict([
    ("both_enabled", {
        "name": "both_enabled",
        "description": "Both load/store skipping and block sampling enabled",
        "env": {
            "TRITON_INTERPRET": "1",
            "ENABLE_TIMING": "1",
            "PROFILER_ENABLE_LOAD_STORE_SKIPPING": "1",
            "PROFILER_ENABLE_BLOCK_SAMPLING": "1",
            "PROFILER_BLOCK_SAMPLING_K": "1",
            "PROFILER_DISABLE_BUFFER_LOAD_CHECK": "1",
            "SANITIZER_ENABLE_FAKE_TENSOR": "0"
        }
    }),
    ("only_load_store_skipping", {
        "name": "only_load_store_skipping",
        "description": "Only load/store skipping enabled, block sampling disabled",
        "env": {
            "TRITON_INTERPRET": "1",
            "ENABLE_TIMING": "1",
            "PROFILER_ENABLE_LOAD_STORE_SKIPPING": "1",
            "PROFILER_ENABLE_BLOCK_SAMPLING": "0",
            "PROFILER_DISABLE_BUFFER_LOAD_CHECK": "1",
            "SANITIZER_ENABLE_FAKE_TENSOR": "0"
        }
    }),
    ("only_block_sampling", {
        "name": "only_block_sampling",
        "description": "Only block sampling enabled (K=1), load/store skipping disabled",
        "env": {
            "TRITON_INTERPRET": "1",
            "ENABLE_TIMING": "1",
            "PROFILER_ENABLE_LOAD_STORE_SKIPPING": "0",
            "PROFILER_ENABLE_BLOCK_SAMPLING": "1",
            "PROFILER_BLOCK_SAMPLING_K": "1",
            "PROFILER_DISABLE_BUFFER_LOAD_CHECK": "1",
            "SANITIZER_ENABLE_FAKE_TENSOR": "0"
        }
    }),
    ("both_disabled", {
        "name": "both_disabled",
        "description": "Both features disabled (baseline)",
        "env": {
            "TRITON_INTERPRET": "1",
            "ENABLE_TIMING": "1",
            "PROFILER_ENABLE_LOAD_STORE_SKIPPING": "0",
            "PROFILER_ENABLE_BLOCK_SAMPLING": "0",
            "PROFILER_DISABLE_BUFFER_LOAD_CHECK": "1",
            "SANITIZER_ENABLE_FAKE_TENSOR": "0"
        }
    })
])


def run_profiler(test: Dict[str, Any], config: Dict[str, Any], output_dir: Path,
                 test_number: int, total_tests: int) -> bool:
    """Run triton-profiler on a single test with given configuration."""

    file_path = test["file_path"]
    test_name = test["name"]
    is_pytest = test["is_pytest"]
    test_function = test.get("test_function")

    # Create output directory for this configuration
    config_output_dir = output_dir / config["name"]
    config_output_dir.mkdir(parents=True, exist_ok=True)

    # Create safe filename (replace :: with __)
    safe_name = test_name.replace("::", "__")
    test_num_str = str(test_number).zfill(len(str(total_tests)))
    output_filename = f"{test_num_str}_{safe_name}.log"
    output_file = config_output_dir / output_filename

    # Setup environment
    env = os.environ.copy()
    env.update(config["env"])

    # Construct command based on test type
    if is_pytest:
        # Liger-Kernel style: run pytest with triton-profiler
        test_spec = f"{file_path.name}::{test_function}" if test_function else file_path.name
        cmd = ["triton-profiler", "pytest", "-s", "--assert=plain", test_spec]
        cwd = file_path.parent
    else:
        # TritonBench style: run Python file directly
        cmd = ["triton-profiler", str(file_path)]
        cwd = file_path.parent

    # Print status
    print(f"  [{test_number}/{total_tests}] Running {test_name} with {config['name']}...")

    # Run the command
    start_time = time.time()

    with open(output_file, 'w') as f:
        # Write header information
        f.write(f"Test: {test_name}\n")
        f.write(f"Configuration: {config['name']}\n")
        f.write(f"Environment: {json.dumps(config['env'], indent=2)}\n")
        f.write(f"Command: {' '.join(cmd)}\n")
        f.write(f"Working directory: {cwd}\n")
        f.write(f"Start time: {datetime.now().isoformat()}\n")
        f.write("=" * 80 + "\n\n")
        f.flush()

        try:
            result = subprocess.run(
                cmd,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=cwd,
                timeout=300  # 5 minute timeout
            )

            elapsed_time = time.time() - start_time

            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Exit code: {result.returncode}\n")
            f.write(f"Elapsed time: {elapsed_time:.2f} seconds\n")

            if result.returncode == 0:
                print(f"    [OK] Completed in {elapsed_time:.2f}s")
            else:
                print(f"    [FAIL] Exit code {result.returncode}")

            return result.returncode == 0

        except subprocess.TimeoutExpired:
            elapsed_time = time.time() - start_time
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"TIMEOUT: Test exceeded 300 seconds\n")
            f.write(f"Elapsed time: {elapsed_time:.2f} seconds\n")
            print(f"    [TIMEOUT] {elapsed_time:.2f}s")
            return False

        except Exception as e:
            elapsed_time = time.time() - start_time
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"ERROR: {str(e)}\n")
            f.write(f"Elapsed time: {elapsed_time:.2f} seconds\n")
            print(f"    [ERROR] {str(e)}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Run profiler ablation study on TritonBench or Liger-Kernel files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tritonbench tests with all configurations
  python ablation_runner.py --repo tritonbench

  # Run liger_kernel tests
  python ablation_runner.py --repo liger_kernel

  # Run a single test case
  python ablation_runner.py --repo tritonbench -c matmul_triton1

  # Run with specific configurations
  python ablation_runner.py --repo tritonbench --configs both_enabled both_disabled
        """
    )
    parser.add_argument(
        "--repo",
        choices=list_available_repos(),
        default="tritonbench",
        help="Repository to run tests from (default: tritonbench)"
    )
    parser.add_argument(
        "-c", "--case",
        type=str,
        help="Run a single test case (e.g., matmul_triton1 or test_rms_norm.py::test_correctness)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for logs"
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=list(PROFILER_CONFIGS.keys()) + ["all"],
        default=["all"],
        help="Configurations to run"
    )
    parser.add_argument(
        "--registry",
        type=str,
        help="Path to custom test registry file (default: utils/test_registry.txt)"
    )
    args = parser.parse_args()

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("results")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load test registry
    registry_file = Path(args.registry) if args.registry else None
    registry = load_registry(registry_file)
    if not registry:
        print("Error: No tests found in registry")
        return 1

    # Discover tests for the specified repo
    tests = discover_tests(args.repo, registry, case=args.case)
    if not tests:
        print("No tests to process")
        return 1

    # Filter out excluded tests
    original_count = len(tests)
    tests = [t for t in tests if not any(t["name"].startswith(prefix) for prefix in EXCLUDED_TEST_PREFIXES)]
    if len(tests) < original_count:
        print(f"Excluded {original_count - len(tests)} tests matching prefixes: {EXCLUDED_TEST_PREFIXES}")

    # Determine configurations to run
    if "all" in args.configs:
        configs_to_run = list(PROFILER_CONFIGS.keys())
    else:
        configs_to_run = args.configs

    # Calculate total number of tests
    total_tests = len(tests) * len(configs_to_run)

    print(f"\n{'=' * 60}")
    print(f"Profiler Ablation Study")
    print(f"{'=' * 60}")
    print(f"Repository: {args.repo}")
    print(f"Tests to profile: {len(tests)}")
    print(f"Configurations: {', '.join(configs_to_run)}")
    print(f"Total runs: {total_tests}")
    print(f"Output directory: {output_dir}")
    print(f"{'=' * 60}\n")

    # Run tests
    results = {}

    for config_name in configs_to_run:
        config = PROFILER_CONFIGS[config_name]
        print(f"\n--- Configuration: {config['name']} ---")
        print(f"    {config['description']}")
        print()

        config_results = []
        test_counter = 0

        for test in tests:
            test_counter += 1
            success = run_profiler(test, config, output_dir, test_counter, len(tests))
            config_results.append({
                "name": test["name"],
                "global_id": test["global_id"],
                "success": success
            })

        results[config_name] = config_results

        # Print summary for this configuration
        successful = sum(1 for r in config_results if r["success"])
        print(f"\n  Configuration summary: {successful}/{len(tests)} tests passed\n")

    # Print overall summary
    print(f"\n{'=' * 60}")
    print("Overall Summary")
    print(f"{'=' * 60}")

    for config_name, config_results in results.items():
        successful = sum(1 for r in config_results if r["success"])
        total = len(config_results)
        if total > 0:
            print(f"{config_name:25} {successful:3}/{total:3} passed ({successful*100/total:.1f}%)")
        else:
            print(f"{config_name:25} No results")

    print(f"{'=' * 60}")
    print(f"Results saved in: {output_dir}/")
    print()

    # Write summary JSON
    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "repository": args.repo,
            "configurations": configs_to_run,
            "tests": [{"name": t["name"], "global_id": t["global_id"]} for t in tests],
            "results": results
        }, f, indent=2)

    print(f"Summary written to: {summary_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
