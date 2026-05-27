#!/usr/bin/env python3
"""
End-to-end experiment runner for TritonBench and Liger-Kernel files.
Supports two metrics:
  - kernel_time: Measure kernel execution time
    - baseline: Use triton_profiler.py hooks
    - profiler/sanitizer: Use ENABLE_TIMING=1
  - e2e_time: Measure end-to-end wall-clock time

Supports three modes:
  - baseline: Run with plain Python (+ triton_profiler for kernel_time)
  - profiler: Run with triton-profiler
  - sanitizer: Run with triton-sanitizer

Supports two repositories:
  - tritonbench: Run Python files directly
  - liger_kernel: Run pytest tests (test_file.py::test_function format)
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

# Script directory (where this script is located)
SCRIPT_DIR = Path(__file__).parent.resolve()

# Triton profiler wrapper script location (relative to script directory)
TRITON_PROFILER_WRAPPER = SCRIPT_DIR.parent / "utils" / "tritonbench_profiler_wrapper.py"


def parse_triton_profiler_timing(output: str) -> Tuple[float, int]:
    """
    Parse triton_profiler.py output for kernel GPU times.
    Example line: [triton-profiler] kernel=matmul_kernel cpu_launch_ms=0.123 gpu_time_ms=0.456

    Returns:
        Tuple of (total_gpu_time_ms, kernel_count)
    """
    pattern = r'\[triton-profiler\]\s+kernel=(\S+)\s+cpu_launch_ms=[\d.]+\s+gpu_time_ms=([\d.]+)'

    total_ms = 0.0
    count = 0

    for line in output.splitlines():
        match = re.search(pattern, line)
        if match:
            gpu_time = float(match.group(2))
            total_ms += gpu_time
            count += 1

    return total_ms, count


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


# =============================================================================
# Kernel Time Mode Functions
# =============================================================================

def run_baseline_kernel_time(test: Dict[str, Any], output_dir, global_id, total_registry, current, total_current):
    """Run baseline with triton_profiler.py to measure kernel GPU time."""
    output_dir.mkdir(parents=True, exist_ok=True)

    file_path = test["file_path"]
    test_name = test["name"]
    is_pytest = test["is_pytest"]
    test_function = test.get("test_function")

    # Create safe filename (replace :: with __)
    safe_name = test_name.replace("::", "__")
    id_str = str(global_id).zfill(len(str(total_registry)))
    output_filename = f"{id_str}_{safe_name}.log"
    output_file = output_dir / output_filename

    # Use wrapper script with ENABLE_TRITON_PROFILER=1
    env = os.environ.copy()
    env["ENABLE_TRITON_PROFILER"] = "1"

    if is_pytest:
        # For pytest-based tests (Liger-Kernel)
        test_spec = f"{file_path.name}::{test_function}" if test_function else file_path.name
        cmd = ["pytest", "-s", "--assert=plain", test_spec]
        cwd = file_path.parent
    else:
        # For direct Python tests (TritonBench)
        cmd = ["python", str(TRITON_PROFILER_WRAPPER), str(file_path)]
        cwd = TRITONBENCH_DIR

    print(f"  [ID:{id_str}] ({current}/{total_current}) Running {test_name} with baseline (triton_profiler)...")

    # Warmup run (not timed)
    print(f"    [WARMUP] Running warmup...")
    try:
        subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=300
        )
    except Exception:
        pass  # Ignore warmup errors

    # Actual timed run
    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=300
        )

        output = result.stdout + "\n" + result.stderr
        total_time_ms, kernel_count = parse_triton_profiler_timing(output)

        with open(output_file, 'w') as f:
            f.write(f"Test: {test_name}\n")
            f.write(f"Mode: baseline (triton_profiler)\n")
            f.write(f"Metric: kernel_time\n")
            f.write(f"Command: ENABLE_TRITON_PROFILER=1 {' '.join(cmd)}\n")
            f.write(f"Start time: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")
            f.write(output)
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Exit code: {result.returncode}\n")
            f.write(f"Kernel time: {total_time_ms:.3f} ms ({kernel_count} kernel calls)\n")

        if result.returncode == 0:
            print(f"    [OK] Kernel time: {total_time_ms:.3f} ms ({kernel_count} kernels)")
        else:
            print(f"    [FAIL] Exit code {result.returncode}, kernel time: {total_time_ms:.3f} ms")

        return result.returncode == 0, total_time_ms

    except subprocess.TimeoutExpired:
        with open(output_file, 'w') as f:
            f.write(f"Test: {test_name}\n")
            f.write(f"Mode: baseline (triton_profiler)\n")
            f.write("=" * 80 + "\n")
            f.write(f"TIMEOUT: Test exceeded 300 seconds\n")
        print(f"    [TIMEOUT] Timeout")
        return False, 0.0

    except Exception as e:
        with open(output_file, 'w') as f:
            f.write(f"Test: {test_name}\n")
            f.write(f"Mode: baseline (triton_profiler)\n")
            f.write("=" * 80 + "\n")
            f.write(f"ERROR: {str(e)}\n")
        print(f"    [ERROR] Error: {str(e)}")
        return False, 0.0


def run_profiler_kernel_time(test: Dict[str, Any], output_dir, global_id, total_registry, current, total_current):
    """Run triton-profiler with ENABLE_TIMING=1 to measure kernel time."""
    output_dir.mkdir(parents=True, exist_ok=True)

    file_path = test["file_path"]
    test_name = test["name"]
    is_pytest = test["is_pytest"]
    test_function = test.get("test_function")

    safe_name = test_name.replace("::", "__")
    id_str = str(global_id).zfill(len(str(total_registry)))
    output_filename = f"{id_str}_{safe_name}.log"
    output_file = output_dir / output_filename

    env = os.environ.copy()
    env.update({
        "TRITON_INTERPRET": "1",
        "ENABLE_TIMING": "1",
        "PROFILER_ENABLE_LOAD_STORE_SKIPPING": "1",
        "PROFILER_ENABLE_BLOCK_SAMPLING": "1",
        "PROFILER_DISABLE_BUFFER_LOAD_CHECK": "1",
        "SANITIZER_ENABLE_FAKE_TENSOR": "1"
    })

    if is_pytest:
        test_spec = f"{file_path.name}::{test_function}" if test_function else file_path.name
        cmd = ["triton-profiler", "pytest", "-s", "--assert=plain", test_spec]
        cwd = file_path.parent
    else:
        cmd = ["triton-profiler", str(file_path)]
        cwd = TRITONBENCH_DIR

    print(f"  [ID:{id_str}] ({current}/{total_current}) Running {test_name} with triton-profiler...")

    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=300
        )

        output = result.stdout + "\n" + result.stderr
        total_time_ms, kernel_count = parse_triton_viz_timing(output)

        with open(output_file, 'w') as f:
            f.write(f"Test: {test_name}\n")
            f.write(f"Mode: profiler (triton-profiler)\n")
            f.write(f"Metric: kernel_time\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Start time: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")
            f.write(output)
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Exit code: {result.returncode}\n")
            f.write(f"Kernel time: {total_time_ms:.3f} ms ({kernel_count} kernel calls)\n")

        if result.returncode == 0:
            print(f"    [OK] Kernel time: {total_time_ms:.3f} ms ({kernel_count} kernels)")
        else:
            print(f"    [FAIL] Exit code {result.returncode}, kernel time: {total_time_ms:.3f} ms")

        return result.returncode == 0, total_time_ms

    except subprocess.TimeoutExpired:
        with open(output_file, 'w') as f:
            f.write(f"Test: {test_name}\n")
            f.write(f"Mode: profiler (triton-profiler)\n")
            f.write("=" * 80 + "\n")
            f.write(f"TIMEOUT: Test exceeded 300 seconds\n")
        print(f"    [TIMEOUT] Timeout")
        return False, 0.0

    except Exception as e:
        with open(output_file, 'w') as f:
            f.write(f"Test: {test_name}\n")
            f.write(f"Mode: profiler (triton-profiler)\n")
            f.write("=" * 80 + "\n")
            f.write(f"ERROR: {str(e)}\n")
        print(f"    [ERROR] Error: {str(e)}")
        return False, 0.0


def run_sanitizer_kernel_time(test: Dict[str, Any], output_dir, global_id, total_registry, current, total_current):
    """Run triton-sanitizer with ENABLE_TIMING=1 to measure kernel time."""
    output_dir.mkdir(parents=True, exist_ok=True)

    file_path = test["file_path"]
    test_name = test["name"]
    is_pytest = test["is_pytest"]
    test_function = test.get("test_function")

    safe_name = test_name.replace("::", "__")
    id_str = str(global_id).zfill(len(str(total_registry)))
    output_filename = f"{id_str}_{safe_name}.log"
    output_file = output_dir / output_filename

    env = os.environ.copy()
    env.update({
        "TRITON_INTERPRET": "1",
        "ENABLE_TIMING": "1"
    })

    if is_pytest:
        test_spec = f"{file_path.name}::{test_function}" if test_function else file_path.name
        cmd = ["triton-sanitizer", "pytest", "-s", "--assert=plain", test_spec]
        cwd = file_path.parent
    else:
        cmd = ["triton-sanitizer", str(file_path)]
        cwd = TRITONBENCH_DIR

    print(f"  [ID:{id_str}] ({current}/{total_current}) Running {test_name} with triton-sanitizer...")

    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=300
        )

        output = result.stdout + "\n" + result.stderr
        total_time_ms, kernel_count = parse_triton_viz_timing(output)

        with open(output_file, 'w') as f:
            f.write(f"Test: {test_name}\n")
            f.write(f"Mode: sanitizer (triton-sanitizer)\n")
            f.write(f"Metric: kernel_time\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Start time: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")
            f.write(output)
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Exit code: {result.returncode}\n")
            f.write(f"Kernel time: {total_time_ms:.3f} ms ({kernel_count} kernel calls)\n")

        if result.returncode == 0:
            print(f"    [OK] Kernel time: {total_time_ms:.3f} ms ({kernel_count} kernels)")
        else:
            print(f"    [FAIL] Exit code {result.returncode}, kernel time: {total_time_ms:.3f} ms")

        return result.returncode == 0, total_time_ms

    except subprocess.TimeoutExpired:
        with open(output_file, 'w') as f:
            f.write(f"Test: {test_name}\n")
            f.write(f"Mode: sanitizer (triton-sanitizer)\n")
            f.write("=" * 80 + "\n")
            f.write(f"TIMEOUT: Test exceeded 300 seconds\n")
        print(f"    [TIMEOUT] Timeout")
        return False, 0.0

    except Exception as e:
        with open(output_file, 'w') as f:
            f.write(f"Test: {test_name}\n")
            f.write(f"Mode: sanitizer (triton-sanitizer)\n")
            f.write("=" * 80 + "\n")
            f.write(f"ERROR: {str(e)}\n")
        print(f"    [ERROR] Error: {str(e)}")
        return False, 0.0


# =============================================================================
# End-to-End Time Mode Functions
# =============================================================================

def run_baseline_e2e_time(test: Dict[str, Any], output_dir, global_id, total_registry, current, total_current):
    """Run baseline with plain Python and measure wall-clock time."""
    output_dir.mkdir(parents=True, exist_ok=True)

    file_path = test["file_path"]
    test_name = test["name"]
    is_pytest = test["is_pytest"]
    test_function = test.get("test_function")

    safe_name = test_name.replace("::", "__")
    id_str = str(global_id).zfill(len(str(total_registry)))
    output_filename = f"{id_str}_{safe_name}.log"
    output_file = output_dir / output_filename

    if is_pytest:
        test_spec = f"{file_path.name}::{test_function}" if test_function else file_path.name
        cmd = ["pytest", "-s", "--assert=plain", test_spec]
        cwd = file_path.parent
    else:
        cmd = ["python", str(file_path)]
        cwd = TRITONBENCH_DIR

    print(f"  [ID:{id_str}] ({current}/{total_current}) Running {test_name} with baseline (python)...")

    # Warmup run (not timed)
    print(f"    [WARMUP] Running warmup...")
    try:
        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=300
        )
    except Exception:
        pass  # Ignore warmup errors

    # Actual timed run
    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=300
        )

        elapsed_time = time.time() - start_time
        elapsed_time_ms = elapsed_time * 1000

        output = result.stdout + "\n" + result.stderr

        with open(output_file, 'w') as f:
            f.write(f"Test: {test_name}\n")
            f.write(f"Mode: baseline (python)\n")
            f.write(f"Metric: e2e_time\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Start time: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")
            f.write(output)
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Exit code: {result.returncode}\n")
            f.write(f"E2E time: {elapsed_time_ms:.3f} ms ({elapsed_time:.3f} s)\n")

        if result.returncode == 0:
            print(f"    [OK] E2E time: {elapsed_time_ms:.3f} ms")
        else:
            print(f"    [FAIL] Exit code {result.returncode}, E2E time: {elapsed_time_ms:.3f} ms")

        return result.returncode == 0, elapsed_time_ms

    except subprocess.TimeoutExpired:
        elapsed_time = time.time() - start_time
        elapsed_time_ms = elapsed_time * 1000
        with open(output_file, 'w') as f:
            f.write(f"Test: {test_name}\n")
            f.write(f"Mode: baseline (python)\n")
            f.write("=" * 80 + "\n")
            f.write(f"TIMEOUT: Test exceeded 300 seconds\n")
            f.write(f"E2E time: {elapsed_time_ms:.3f} ms\n")
        print(f"    [TIMEOUT] Timeout after {elapsed_time_ms:.3f} ms")
        return False, elapsed_time_ms

    except Exception as e:
        elapsed_time = time.time() - start_time
        elapsed_time_ms = elapsed_time * 1000
        with open(output_file, 'w') as f:
            f.write(f"Test: {test_name}\n")
            f.write(f"Mode: baseline (python)\n")
            f.write("=" * 80 + "\n")
            f.write(f"ERROR: {str(e)}\n")
            f.write(f"E2E time: {elapsed_time_ms:.3f} ms\n")
        print(f"    [ERROR] Error: {str(e)}")
        return False, elapsed_time_ms


def run_profiler_e2e_time(test: Dict[str, Any], output_dir, global_id, total_registry, current, total_current):
    """Run triton-profiler and measure wall-clock time."""
    output_dir.mkdir(parents=True, exist_ok=True)

    file_path = test["file_path"]
    test_name = test["name"]
    is_pytest = test["is_pytest"]
    test_function = test.get("test_function")

    safe_name = test_name.replace("::", "__")
    id_str = str(global_id).zfill(len(str(total_registry)))
    output_filename = f"{id_str}_{safe_name}.log"
    output_file = output_dir / output_filename

    env = os.environ.copy()
    env.update({
        "TRITON_INTERPRET": "1",
        "PROFILER_ENABLE_LOAD_STORE_SKIPPING": "1",
        "PROFILER_ENABLE_BLOCK_SAMPLING": "1",
        "PROFILER_DISABLE_BUFFER_LOAD_CHECK": "1",
        "SANITIZER_ENABLE_FAKE_TENSOR": "1"
    })

    if is_pytest:
        test_spec = f"{file_path.name}::{test_function}" if test_function else file_path.name
        cmd = ["triton-profiler", "pytest", "-s", "--assert=plain", test_spec]
        cwd = file_path.parent
    else:
        cmd = ["triton-profiler", str(file_path)]
        cwd = TRITONBENCH_DIR

    print(f"  [ID:{id_str}] ({current}/{total_current}) Running {test_name} with triton-profiler...")

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

        with open(output_file, 'w') as f:
            f.write(f"Test: {test_name}\n")
            f.write(f"Mode: profiler (triton-profiler)\n")
            f.write(f"Metric: e2e_time\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Start time: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")
            f.write(output)
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Exit code: {result.returncode}\n")
            f.write(f"E2E time: {elapsed_time_ms:.3f} ms ({elapsed_time:.3f} s)\n")

        if result.returncode == 0:
            print(f"    [OK] E2E time: {elapsed_time_ms:.3f} ms")
        else:
            print(f"    [FAIL] Exit code {result.returncode}, E2E time: {elapsed_time_ms:.3f} ms")

        return result.returncode == 0, elapsed_time_ms

    except subprocess.TimeoutExpired:
        elapsed_time = time.time() - start_time
        elapsed_time_ms = elapsed_time * 1000
        with open(output_file, 'w') as f:
            f.write(f"Test: {test_name}\n")
            f.write(f"Mode: profiler (triton-profiler)\n")
            f.write("=" * 80 + "\n")
            f.write(f"TIMEOUT: Test exceeded 300 seconds\n")
            f.write(f"E2E time: {elapsed_time_ms:.3f} ms\n")
        print(f"    [TIMEOUT] Timeout after {elapsed_time_ms:.3f} ms")
        return False, elapsed_time_ms

    except Exception as e:
        elapsed_time = time.time() - start_time
        elapsed_time_ms = elapsed_time * 1000
        with open(output_file, 'w') as f:
            f.write(f"Test: {test_name}\n")
            f.write(f"Mode: profiler (triton-profiler)\n")
            f.write("=" * 80 + "\n")
            f.write(f"ERROR: {str(e)}\n")
            f.write(f"E2E time: {elapsed_time_ms:.3f} ms\n")
        print(f"    [ERROR] Error: {str(e)}")
        return False, elapsed_time_ms


def run_sanitizer_e2e_time(test: Dict[str, Any], output_dir, global_id, total_registry, current, total_current):
    """Run triton-sanitizer and measure wall-clock time."""
    output_dir.mkdir(parents=True, exist_ok=True)

    file_path = test["file_path"]
    test_name = test["name"]
    is_pytest = test["is_pytest"]
    test_function = test.get("test_function")

    safe_name = test_name.replace("::", "__")
    id_str = str(global_id).zfill(len(str(total_registry)))
    output_filename = f"{id_str}_{safe_name}.log"
    output_file = output_dir / output_filename

    env = os.environ.copy()
    env.update({
        "TRITON_INTERPRET": "1",
        "SANITIZER_ENABLE_FAKE_TENSOR": "1"
    })

    if is_pytest:
        test_spec = f"{file_path.name}::{test_function}" if test_function else file_path.name
        cmd = ["triton-sanitizer", "pytest", "-s", "--assert=plain", test_spec]
        cwd = file_path.parent
    else:
        cmd = ["triton-sanitizer", str(file_path)]
        cwd = TRITONBENCH_DIR

    print(f"  [ID:{id_str}] ({current}/{total_current}) Running {test_name} with triton-sanitizer...")

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

        with open(output_file, 'w') as f:
            f.write(f"Test: {test_name}\n")
            f.write(f"Mode: sanitizer (triton-sanitizer)\n")
            f.write(f"Metric: e2e_time\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Start time: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")
            f.write(output)
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Exit code: {result.returncode}\n")
            f.write(f"E2E time: {elapsed_time_ms:.3f} ms ({elapsed_time:.3f} s)\n")

        if result.returncode == 0:
            print(f"    [OK] E2E time: {elapsed_time_ms:.3f} ms")
        else:
            print(f"    [FAIL] Exit code {result.returncode}, E2E time: {elapsed_time_ms:.3f} ms")

        return result.returncode == 0, elapsed_time_ms

    except subprocess.TimeoutExpired:
        elapsed_time = time.time() - start_time
        elapsed_time_ms = elapsed_time * 1000
        with open(output_file, 'w') as f:
            f.write(f"Test: {test_name}\n")
            f.write(f"Mode: sanitizer (triton-sanitizer)\n")
            f.write("=" * 80 + "\n")
            f.write(f"TIMEOUT: Test exceeded 300 seconds\n")
            f.write(f"E2E time: {elapsed_time_ms:.3f} ms\n")
        print(f"    [TIMEOUT] Timeout after {elapsed_time_ms:.3f} ms")
        return False, elapsed_time_ms

    except Exception as e:
        elapsed_time = time.time() - start_time
        elapsed_time_ms = elapsed_time * 1000
        with open(output_file, 'w') as f:
            f.write(f"Test: {test_name}\n")
            f.write(f"Mode: sanitizer (triton-sanitizer)\n")
            f.write("=" * 80 + "\n")
            f.write(f"ERROR: {str(e)}\n")
            f.write(f"E2E time: {elapsed_time_ms:.3f} ms\n")
        print(f"    [ERROR] Error: {str(e)}")
        return False, elapsed_time_ms


# =============================================================================
# Combined Mode Functions (collect both kernel_time and e2e_time in single run)
# =============================================================================

def run_baseline_combined(test: Dict[str, Any], output_base_dir, global_id, total_registry, current, total_current, run_number):
    """Run baseline with triton_profiler and measure both kernel time and wall-clock time."""
    file_path = test["file_path"]
    test_name = test["name"]
    is_pytest = test["is_pytest"]
    test_function = test.get("test_function")

    safe_name = test_name.replace("::", "__")
    id_str = str(global_id).zfill(len(str(total_registry)))
    output_filename = f"{id_str}_{safe_name}.log"

    # Create output directories for both metrics
    kernel_time_dir = output_base_dir / "kernel_time" / "baseline" / f"run{run_number}"
    e2e_time_dir = output_base_dir / "e2e_time" / "baseline" / f"run{run_number}"
    kernel_time_dir.mkdir(parents=True, exist_ok=True)
    e2e_time_dir.mkdir(parents=True, exist_ok=True)

    # Use wrapper script with ENABLE_TRITON_PROFILER=1
    env = os.environ.copy()
    env["ENABLE_TRITON_PROFILER"] = "1"

    if is_pytest:
        test_spec = f"{file_path.name}::{test_function}" if test_function else file_path.name
        cmd = ["pytest", "-s", "--assert=plain", test_spec]
        cwd = file_path.parent
    else:
        cmd = ["python", str(TRITON_PROFILER_WRAPPER), str(file_path)]
        cwd = TRITONBENCH_DIR

    print(f"  [ID:{id_str}] ({current}/{total_current}) Running {test_name} with baseline (triton_profiler)...")

    # Warmup run (not timed)
    print(f"    [WARMUP] Running warmup...")
    try:
        subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=cwd, timeout=300)
    except Exception:
        pass

    # Actual timed run
    start_time = time.time()

    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=cwd, timeout=300)

        elapsed_time = time.time() - start_time
        elapsed_time_ms = elapsed_time * 1000

        output = result.stdout + "\n" + result.stderr
        kernel_time_ms, kernel_count = parse_triton_profiler_timing(output)

        # Write kernel_time log
        with open(kernel_time_dir / output_filename, 'w') as f:
            f.write(f"Test: {test_name}\n")
            f.write(f"Mode: baseline (triton_profiler)\n")
            f.write(f"Metric: kernel_time\n")
            f.write(f"Command: ENABLE_TRITON_PROFILER=1 {' '.join(cmd)}\n")
            f.write(f"Start time: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")
            f.write(output)
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Exit code: {result.returncode}\n")
            f.write(f"Kernel time: {kernel_time_ms:.3f} ms ({kernel_count} kernel calls)\n")

        # Write e2e_time log
        with open(e2e_time_dir / output_filename, 'w') as f:
            f.write(f"Test: {test_name}\n")
            f.write(f"Mode: baseline (triton_profiler)\n")
            f.write(f"Metric: e2e_time\n")
            f.write(f"Command: ENABLE_TRITON_PROFILER=1 {' '.join(cmd)}\n")
            f.write(f"Start time: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")
            f.write(output)
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Exit code: {result.returncode}\n")
            f.write(f"E2E time: {elapsed_time_ms:.3f} ms ({elapsed_time:.3f} s)\n")

        if result.returncode == 0:
            print(f"    [OK] Kernel: {kernel_time_ms:.3f} ms ({kernel_count} kernels), E2E: {elapsed_time_ms:.3f} ms")
        else:
            print(f"    [FAIL] Exit code {result.returncode}, Kernel: {kernel_time_ms:.3f} ms, E2E: {elapsed_time_ms:.3f} ms")

        return result.returncode == 0, kernel_time_ms, elapsed_time_ms

    except subprocess.TimeoutExpired:
        elapsed_time = time.time() - start_time
        elapsed_time_ms = elapsed_time * 1000
        for out_dir in [kernel_time_dir, e2e_time_dir]:
            with open(out_dir / output_filename, 'w') as f:
                f.write(f"Test: {test_name}\n")
                f.write(f"Mode: baseline (triton_profiler)\n")
                f.write("=" * 80 + "\n")
                f.write(f"TIMEOUT: Test exceeded 300 seconds\n")
        print(f"    [TIMEOUT] Timeout")
        return False, 0.0, elapsed_time_ms

    except Exception as e:
        elapsed_time = time.time() - start_time
        elapsed_time_ms = elapsed_time * 1000
        for out_dir in [kernel_time_dir, e2e_time_dir]:
            with open(out_dir / output_filename, 'w') as f:
                f.write(f"Test: {test_name}\n")
                f.write(f"Mode: baseline (triton_profiler)\n")
                f.write("=" * 80 + "\n")
                f.write(f"ERROR: {str(e)}\n")
        print(f"    [ERROR] Error: {str(e)}")
        return False, 0.0, elapsed_time_ms


def run_profiler_combined(test: Dict[str, Any], output_base_dir, global_id, total_registry, current, total_current, run_number):
    """Run triton-profiler and measure both kernel time and wall-clock time."""
    file_path = test["file_path"]
    test_name = test["name"]
    is_pytest = test["is_pytest"]
    test_function = test.get("test_function")

    safe_name = test_name.replace("::", "__")
    id_str = str(global_id).zfill(len(str(total_registry)))
    output_filename = f"{id_str}_{safe_name}.log"

    # Create output directories for both metrics
    kernel_time_dir = output_base_dir / "kernel_time" / "profiler" / f"run{run_number}"
    e2e_time_dir = output_base_dir / "e2e_time" / "profiler" / f"run{run_number}"
    kernel_time_dir.mkdir(parents=True, exist_ok=True)
    e2e_time_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update({
        "TRITON_INTERPRET": "1",
        "ENABLE_TIMING": "1",
        "PROFILER_ENABLE_LOAD_STORE_SKIPPING": "1",
        "PROFILER_ENABLE_BLOCK_SAMPLING": "1",
        "PROFILER_DISABLE_BUFFER_LOAD_CHECK": "1",
        "SANITIZER_ENABLE_FAKE_TENSOR": "1"
    })

    if is_pytest:
        test_spec = f"{file_path.name}::{test_function}" if test_function else file_path.name
        cmd = ["triton-profiler", "pytest", "-s", "--assert=plain", test_spec]
        cwd = file_path.parent
    else:
        cmd = ["triton-profiler", str(file_path)]
        cwd = TRITONBENCH_DIR

    print(f"  [ID:{id_str}] ({current}/{total_current}) Running {test_name} with triton-profiler...")

    start_time = time.time()

    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=cwd, timeout=300)

        elapsed_time = time.time() - start_time
        elapsed_time_ms = elapsed_time * 1000

        output = result.stdout + "\n" + result.stderr
        kernel_time_ms, kernel_count = parse_triton_viz_timing(output)

        # Write kernel_time log
        with open(kernel_time_dir / output_filename, 'w') as f:
            f.write(f"Test: {test_name}\n")
            f.write(f"Mode: profiler (triton-profiler)\n")
            f.write(f"Metric: kernel_time\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Start time: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")
            f.write(output)
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Exit code: {result.returncode}\n")
            f.write(f"Kernel time: {kernel_time_ms:.3f} ms ({kernel_count} kernel calls)\n")

        # Write e2e_time log
        with open(e2e_time_dir / output_filename, 'w') as f:
            f.write(f"Test: {test_name}\n")
            f.write(f"Mode: profiler (triton-profiler)\n")
            f.write(f"Metric: e2e_time\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Start time: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")
            f.write(output)
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Exit code: {result.returncode}\n")
            f.write(f"E2E time: {elapsed_time_ms:.3f} ms ({elapsed_time:.3f} s)\n")

        if result.returncode == 0:
            print(f"    [OK] Kernel: {kernel_time_ms:.3f} ms ({kernel_count} kernels), E2E: {elapsed_time_ms:.3f} ms")
        else:
            print(f"    [FAIL] Exit code {result.returncode}, Kernel: {kernel_time_ms:.3f} ms, E2E: {elapsed_time_ms:.3f} ms")

        return result.returncode == 0, kernel_time_ms, elapsed_time_ms

    except subprocess.TimeoutExpired:
        elapsed_time = time.time() - start_time
        elapsed_time_ms = elapsed_time * 1000
        for out_dir in [kernel_time_dir, e2e_time_dir]:
            with open(out_dir / output_filename, 'w') as f:
                f.write(f"Test: {test_name}\n")
                f.write(f"Mode: profiler (triton-profiler)\n")
                f.write("=" * 80 + "\n")
                f.write(f"TIMEOUT: Test exceeded 300 seconds\n")
        print(f"    [TIMEOUT] Timeout")
        return False, 0.0, elapsed_time_ms

    except Exception as e:
        elapsed_time = time.time() - start_time
        elapsed_time_ms = elapsed_time * 1000
        for out_dir in [kernel_time_dir, e2e_time_dir]:
            with open(out_dir / output_filename, 'w') as f:
                f.write(f"Test: {test_name}\n")
                f.write(f"Mode: profiler (triton-profiler)\n")
                f.write("=" * 80 + "\n")
                f.write(f"ERROR: {str(e)}\n")
        print(f"    [ERROR] Error: {str(e)}")
        return False, 0.0, elapsed_time_ms


def run_sanitizer_combined(test: Dict[str, Any], output_base_dir, global_id, total_registry, current, total_current, run_number):
    """Run triton-sanitizer and measure both kernel time and wall-clock time."""
    file_path = test["file_path"]
    test_name = test["name"]
    is_pytest = test["is_pytest"]
    test_function = test.get("test_function")

    safe_name = test_name.replace("::", "__")
    id_str = str(global_id).zfill(len(str(total_registry)))
    output_filename = f"{id_str}_{safe_name}.log"

    # Create output directories for both metrics
    kernel_time_dir = output_base_dir / "kernel_time" / "sanitizer" / f"run{run_number}"
    e2e_time_dir = output_base_dir / "e2e_time" / "sanitizer" / f"run{run_number}"
    kernel_time_dir.mkdir(parents=True, exist_ok=True)
    e2e_time_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update({
        "TRITON_INTERPRET": "1",
        "ENABLE_TIMING": "1"
    })

    if is_pytest:
        test_spec = f"{file_path.name}::{test_function}" if test_function else file_path.name
        cmd = ["triton-sanitizer", "pytest", "-s", "--assert=plain", test_spec]
        cwd = file_path.parent
    else:
        cmd = ["triton-sanitizer", str(file_path)]
        cwd = TRITONBENCH_DIR

    print(f"  [ID:{id_str}] ({current}/{total_current}) Running {test_name} with triton-sanitizer...")

    start_time = time.time()

    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=cwd, timeout=300)

        elapsed_time = time.time() - start_time
        elapsed_time_ms = elapsed_time * 1000

        output = result.stdout + "\n" + result.stderr
        kernel_time_ms, kernel_count = parse_triton_viz_timing(output)

        # Write kernel_time log
        with open(kernel_time_dir / output_filename, 'w') as f:
            f.write(f"Test: {test_name}\n")
            f.write(f"Mode: sanitizer (triton-sanitizer)\n")
            f.write(f"Metric: kernel_time\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Start time: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")
            f.write(output)
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Exit code: {result.returncode}\n")
            f.write(f"Kernel time: {kernel_time_ms:.3f} ms ({kernel_count} kernel calls)\n")

        # Write e2e_time log
        with open(e2e_time_dir / output_filename, 'w') as f:
            f.write(f"Test: {test_name}\n")
            f.write(f"Mode: sanitizer (triton-sanitizer)\n")
            f.write(f"Metric: e2e_time\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Start time: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")
            f.write(output)
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Exit code: {result.returncode}\n")
            f.write(f"E2E time: {elapsed_time_ms:.3f} ms ({elapsed_time:.3f} s)\n")

        if result.returncode == 0:
            print(f"    [OK] Kernel: {kernel_time_ms:.3f} ms ({kernel_count} kernels), E2E: {elapsed_time_ms:.3f} ms")
        else:
            print(f"    [FAIL] Exit code {result.returncode}, Kernel: {kernel_time_ms:.3f} ms, E2E: {elapsed_time_ms:.3f} ms")

        return result.returncode == 0, kernel_time_ms, elapsed_time_ms

    except subprocess.TimeoutExpired:
        elapsed_time = time.time() - start_time
        elapsed_time_ms = elapsed_time * 1000
        for out_dir in [kernel_time_dir, e2e_time_dir]:
            with open(out_dir / output_filename, 'w') as f:
                f.write(f"Test: {test_name}\n")
                f.write(f"Mode: sanitizer (triton-sanitizer)\n")
                f.write("=" * 80 + "\n")
                f.write(f"TIMEOUT: Test exceeded 300 seconds\n")
        print(f"    [TIMEOUT] Timeout")
        return False, 0.0, elapsed_time_ms

    except Exception as e:
        elapsed_time = time.time() - start_time
        elapsed_time_ms = elapsed_time * 1000
        for out_dir in [kernel_time_dir, e2e_time_dir]:
            with open(out_dir / output_filename, 'w') as f:
                f.write(f"Test: {test_name}\n")
                f.write(f"Mode: sanitizer (triton-sanitizer)\n")
                f.write("=" * 80 + "\n")
                f.write(f"ERROR: {str(e)}\n")
        print(f"    [ERROR] Error: {str(e)}")
        return False, 0.0, elapsed_time_ms


# =============================================================================
# Run Mode Functions
# =============================================================================

def run_mode_combined(mode, tests, output_base_dir, run_number, total_runs, total_registry_tests):
    """Run tests for a specific mode, collecting both kernel_time and e2e_time in single run."""
    run_funcs = {
        "baseline": run_baseline_combined,
        "profiler": run_profiler_combined,
        "sanitizer": run_sanitizer_combined
    }

    if mode not in run_funcs:
        print(f"Unknown mode: {mode}")
        return []

    run_func = run_funcs[mode]

    print(f"\n{'=' * 60}")
    print(f"Running mode: {mode} (combined metrics, run {run_number}/{total_runs})")
    print(f"Output directory: {output_base_dir}")
    print(f"{'=' * 60}\n")

    results = []
    num_tests = len(tests)

    for i, test in enumerate(tests, 1):
        global_id = test["global_id"]
        success, kernel_time_ms, e2e_time_ms = run_func(
            test, output_base_dir, global_id, total_registry_tests, i, num_tests, run_number
        )
        results.append({
            "name": test["name"],
            "global_id": global_id,
            "success": success,
            "kernel_time_ms": kernel_time_ms,
            "e2e_time_ms": e2e_time_ms
        })

    # Print summary for this mode
    successful = sum(1 for r in results if r["success"])
    print(f"\n  Mode summary: {successful}/{num_tests} tests passed")

    return results


def run_mode(mode, metric, tests, output_base_dir, run_number, total_runs, total_registry_tests):
    """Run tests for a specific mode, metric, and run number."""
    # Select the appropriate function based on mode and metric
    if metric == "kernel_time":
        run_funcs = {
            "baseline": run_baseline_kernel_time,
            "profiler": run_profiler_kernel_time,
            "sanitizer": run_sanitizer_kernel_time
        }
    else:  # e2e_time
        run_funcs = {
            "baseline": run_baseline_e2e_time,
            "profiler": run_profiler_e2e_time,
            "sanitizer": run_sanitizer_e2e_time
        }

    if mode not in run_funcs:
        print(f"Unknown mode: {mode}")
        return []

    run_func = run_funcs[mode]

    # Output to metric/mode/runN/ subdirectory
    output_dir = output_base_dir / metric / mode / f"run{run_number}"

    print(f"\n{'=' * 60}")
    print(f"Running mode: {mode} (metric: {metric}, run {run_number}/{total_runs})")
    print(f"Output directory: {output_dir}")
    print(f"{'=' * 60}\n")

    results = []
    num_tests = len(tests)

    for i, test in enumerate(tests, 1):
        # Use global_id from registry for consistent numbering
        global_id = test["global_id"]
        success, measured_time = run_func(test, output_dir, global_id, total_registry_tests, i, num_tests)
        results.append({
            "name": test["name"],
            "global_id": global_id,
            "success": success,
            "time_ms": measured_time
        })

    # Print summary for this mode
    successful = sum(1 for r in results if r["success"])
    print(f"\n  Mode summary: {successful}/{num_tests} tests passed")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end experiment runner for TritonBench and Liger-Kernel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run TritonBench with all modes and kernel_time metric
  python e2e_runner.py --repo tritonbench --metric kernel_time --mode all

  # Run Liger-Kernel tests
  python e2e_runner.py --repo liger_kernel --metric all --mode all

  # Run both metrics for TritonBench
  python e2e_runner.py --metric all --mode all

  # Run a single case (TritonBench)
  python e2e_runner.py --metric kernel_time --mode all --case matmul_triton1

  # Run a single case (Liger-Kernel)
  python e2e_runner.py --repo liger_kernel --case test_rms_norm.py::test_correctness

  # Run 3 times instead of default 5
  python e2e_runner.py --metric all --mode all --runs 3
        """
    )
    parser.add_argument(
        "--repo",
        choices=["tritonbench", "liger_kernel"],
        default="tritonbench",
        help="Repository to run tests from (default: tritonbench)"
    )
    parser.add_argument(
        "--metric",
        nargs="+",
        choices=["kernel_time", "e2e_time", "all"],
        default=["all"],
        help="Metric(s) to measure: kernel_time, e2e_time, or all (default: all)"
    )
    parser.add_argument(
        "--mode", "-m",
        nargs="+",
        choices=["baseline", "profiler", "sanitizer", "all"],
        default=["all"],
        help="Mode(s) to run: baseline, profiler, sanitizer, or all (default: all)"
    )
    parser.add_argument(
        "--registry", "-w",
        type=str,
        help=f"Path to test registry file (default: {DEFAULT_REGISTRY})"
    )
    parser.add_argument(
        "--case", "-c",
        type=str,
        help="Run a single test case (e.g., matmul_triton1 or test_rms_norm.py::test_correctness)"
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
        help="Number of runs for each mode (default: 1)"
    )
    args = parser.parse_args()

    # Determine metrics to run
    use_combined_mode = "all" in args.metric
    if use_combined_mode:
        metrics_to_run = ["kernel_time", "e2e_time"]
    else:
        metrics_to_run = args.metric

    # Determine modes to run
    if "all" in args.mode:
        modes_to_run = ["baseline", "profiler", "sanitizer"]
    else:
        modes_to_run = args.mode

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
    print(f"End-to-End Experiment Runner")
    print(f"{'=' * 60}")
    print(f"Repository: {args.repo}")
    print(f"Tests to run: {len(tests)} (ID range: {id_range})")
    print(f"Total tests in registry: {total_registry_tests}")
    print(f"Metrics: {', '.join(metrics_to_run)}" + (" (combined collection)" if use_combined_mode else ""))
    print(f"Modes: {', '.join(modes_to_run)}")
    print(f"Runs per mode: {args.runs}")
    print(f"Output base directory: {output_base_dir}")
    print(f"{'=' * 60}")

    # Print environment variables for each mode
    print(f"\nEnvironment Variables Configuration:")
    print(f"{'-' * 60}")

    if "baseline" in modes_to_run:
        print(f"\n  baseline:")
        if "kernel_time" in metrics_to_run:
            print(f"    (kernel_time) ENABLE_TRITON_PROFILER=1")
        if "e2e_time" in metrics_to_run:
            print(f"    (e2e_time) (no special env vars)")

    if "profiler" in modes_to_run:
        print(f"\n  profiler:")
        print(f"    TRITON_INTERPRET=1")
        if "kernel_time" in metrics_to_run:
            print(f"    ENABLE_TIMING=1  (for kernel_time)")
        print(f"    PROFILER_DISABLE_BUFFER_LOAD_CHECK=1")
        print(f"    SANITIZER_ENABLE_FAKE_TENSOR=1")
        print(f"    (PROFILER_ENABLE_LOAD_STORE_SKIPPING and PROFILER_ENABLE_BLOCK_SAMPLING")
        print(f"     are enabled by default)")

    if "sanitizer" in modes_to_run:
        print(f"\n  sanitizer:")
        print(f"    TRITON_INTERPRET=1")
        if "kernel_time" in metrics_to_run:
            print(f"    ENABLE_TIMING=1  (for kernel_time)")
        if "e2e_time" in metrics_to_run:
            print(f"    SANITIZER_ENABLE_FAKE_TENSOR=1  (for e2e_time)")

    # Print total experiment count
    total_experiments = len(tests) * len(modes_to_run) * args.runs
    print(f"\nTotal experiments: {total_experiments} runs")
    print(f"  ({len(tests)} tests x {len(modes_to_run)} modes x {args.runs} runs)")
    print(f"\n{'=' * 60}")

    # Run tests for each metric and mode
    all_results = {}

    if use_combined_mode:
        # Combined mode: collect both metrics in a single run
        for metric in metrics_to_run:
            all_results[metric] = {}
            for mode in modes_to_run:
                all_results[metric][mode] = []

        for mode in modes_to_run:
            for run_num in range(1, args.runs + 1):
                results = run_mode_combined(mode, tests, output_base_dir, run_num, args.runs, total_registry_tests)
                # Split results into kernel_time and e2e_time
                for r in results:
                    all_results["kernel_time"][mode].append({
                        "name": r["name"],
                        "global_id": r["global_id"],
                        "success": r["success"],
                        "time_ms": r["kernel_time_ms"]
                    })
                    all_results["e2e_time"][mode].append({
                        "name": r["name"],
                        "global_id": r["global_id"],
                        "success": r["success"],
                        "time_ms": r["e2e_time_ms"]
                    })
    else:
        # Separate mode: run each metric independently
        for metric in metrics_to_run:
            all_results[metric] = {}
            for mode in modes_to_run:
                all_results[metric][mode] = []
                for run_num in range(1, args.runs + 1):
                    results = run_mode(mode, metric, tests, output_base_dir, run_num, args.runs, total_registry_tests)
                    all_results[metric][mode].extend(results)

    # Print overall summary
    print(f"\n{'=' * 60}")
    print("Overall Summary")
    print(f"{'=' * 60}")

    for metric in metrics_to_run:
        print(f"\n  Metric: {metric}")
        print(f"  {'-' * 50}")
        for mode, results_list in all_results[metric].items():
            total_successful = sum(1 for r in results_list if r["success"])
            total_tests = len(results_list)
            total_time = sum(r["time_ms"] for r in results_list)
            if total_tests > 0:
                print(f"  {mode:15} {total_successful:3}/{total_tests:3} passed ({total_successful*100/total_tests:.1f}%) - Total: {total_time:.3f} ms")
            else:
                print(f"  {mode:15} No results")

    print(f"\n{'=' * 60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
