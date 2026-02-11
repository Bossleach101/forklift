"""
Level 3: Functional Equivalence Tests

Compiles both ground-truth C and predicted LLVM IR into executables,
runs them with the same test driver and inputs, and compares outputs.

This is the strongest correctness test — if the outputs match,
the predicted IR is functionally equivalent to the original code.
"""

import pytest
import subprocess
import tempfile
import os
from pathlib import Path
from .conftest import discover_test_cases, CompilerHelper, TEST_DATA_DIR


# ---------------------------------------------------------------------------
# Discover test cases that have both predicted IR AND C source (with main)
# ---------------------------------------------------------------------------

def _has_main(c_path: Path) -> bool:
    """Check if a C file contains a main() function."""
    if not c_path or not c_path.exists():
        return False
    content = c_path.read_text()
    return "int main" in content or "void main" in content


EQUIVALENCE_CASES = [
    tc for tc in discover_test_cases()
    if tc.predicted_ll_path and tc.predicted_ll_path.exists()
    and tc.c_source_path and tc.c_source_path.exists()
    and _has_main(tc.c_source_path)
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_function_source(c_source: str) -> str:
    """
    Extract just the function definition (not main) from a C file.
    Returns the C source with main() removed, suitable for compiling 
    as a library to link against a test driver.
    
    Heuristic: removes everything from 'int main' onwards.
    """
    import re
    # Remove the main function — find 'int main' and remove from there
    # This is a simple heuristic; works for our test files
    match = re.search(r'\n(int|void)\s+main\s*\(', c_source)
    if match:
        return c_source[:match.start()]
    return c_source


def _build_test_driver(func_name: str, c_source: str) -> str:
    """
    Build a minimal C test driver that:
    1. Declares the function as extern
    2. Has a main() that calls it with known inputs and prints outputs

    The driver is used for BOTH ground-truth and predicted IR, ensuring
    the comparison is fair (same calling convention, same inputs).
    """
    if func_name == "f":
        return """
#include <stdio.h>
#include <stdlib.h>

extern int f(int *list, int val, int n);

int main() {
    int data[5] = {1, 2, 3, 4, 5};
    int result = f(data, 10, 5);
    printf("result=%d\\n", result);
    for (int i = 0; i < 5; ++i) {
        printf("data[%d]=%d\\n", i, data[i]);
    }
    return 0;
}
"""
    # Generic fallback
    return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFunctionalEquivalence:
    """Compare runtime output of ground-truth C vs predicted LLVM IR."""

    @pytest.mark.parametrize(
        "test_case",
        EQUIVALENCE_CASES,
        ids=[tc.name for tc in EQUIVALENCE_CASES],
    )
    def test_output_matches(self, test_case, require_clang, tmp_path):
        """
        Ground truth C function and predicted-IR function should 
        produce identical output when called from the same test driver.
        """
        # Try to build a test driver for this function
        fname = test_case.name.split("/")[-1].replace("_ground_truth", "")
        driver_source = _build_test_driver(fname, test_case.c_source)

        if driver_source is None:
            pytest.skip(f"No test driver available for {test_case.name}")

        # Write driver to temp file
        driver_path = str(tmp_path / "driver.c")
        with open(driver_path, "w") as f:
            f.write(driver_source)

        # Step 1: Extract the function (no main) from ground truth C,
        # compile it + the driver into an executable
        func_source = _extract_function_source(test_case.c_source)
        gt_func_path = str(tmp_path / "gt_func.c")
        with open(gt_func_path, "w") as f:
            f.write(func_source)

        gt_exe = str(tmp_path / "gt.x")
        r = subprocess.run(
            ["clang", gt_func_path, driver_path, "-o", gt_exe],
            capture_output=True, text=True, timeout=30,
        )
        assert r.returncode == 0, f"Failed to compile ground truth C:\n{r.stderr}"

        # Step 2: Run ground truth executable
        gt_run = CompilerHelper.run_executable(gt_exe)

        # Step 3: Compile predicted IR + driver to executable
        ir_content = test_case.predicted_ll
        pred_exe = str(tmp_path / "pred.x")
        r = CompilerHelper.compile_and_link(ir_content, driver_path, pred_exe)
        if r.returncode != 0:
            pytest.fail(
                f"Failed to compile predicted IR + driver for {test_case.name}:\n{r.stderr}"
            )

        # Step 4: Run predicted executable
        pred_run = CompilerHelper.run_executable(pred_exe)

        # Step 5: Compare stdout
        assert pred_run.stdout == gt_run.stdout, (
            f"Output mismatch for {test_case.name}:\n"
            f"  Ground truth output: {gt_run.stdout!r}\n"
            f"  Predicted output:    {pred_run.stdout!r}"
        )

        # Also compare return codes
        assert pred_run.returncode == gt_run.returncode, (
            f"Return code mismatch for {test_case.name}: "
            f"GT={gt_run.returncode}, Predicted={pred_run.returncode}"
        )

    def test_f_equivalence_direct(self, require_clang, tmp_path, f_test_case):
        """
        Direct test for the 'f' function — the primary test case.
        Compiles ground truth C function and predicted IR, links both 
        against the same test driver, runs both, compares output.
        """
        if not f_test_case.predicted_ll_path.exists():
            pytest.skip("No predicted IR for f test case")

        # Build driver
        driver_source = _build_test_driver("f", f_test_case.c_source)
        driver_path = str(tmp_path / "f_driver.c")
        with open(driver_path, "w") as f:
            f.write(driver_source)

        # Compile ground truth function (no main) + driver
        func_source = _extract_function_source(f_test_case.c_source)
        gt_func_path = str(tmp_path / "f_gt_func.c")
        with open(gt_func_path, "w") as f:
            f.write(func_source)

        gt_exe = str(tmp_path / "f_gt.x")
        r = subprocess.run(
            ["clang", gt_func_path, driver_path, "-o", gt_exe],
            capture_output=True, text=True, timeout=30,
        )
        assert r.returncode == 0, f"Ground truth compilation failed:\n{r.stderr}"
        gt_run = CompilerHelper.run_executable(gt_exe)

        # Compile predicted IR + driver
        pred_exe = str(tmp_path / "f_pred.x")
        r = CompilerHelper.compile_and_link(
            f_test_case.predicted_ll, driver_path, pred_exe
        )
        if r.returncode != 0:
            pytest.fail(f"Predicted IR compilation failed:\n{r.stderr}")

        pred_run = CompilerHelper.run_executable(pred_exe)

        assert pred_run.stdout == gt_run.stdout, (
            f"Output mismatch:\n"
            f"  GT: {gt_run.stdout!r}\n"
            f"  Pred: {pred_run.stdout!r}"
        )
        assert pred_run.returncode == gt_run.returncode, (
            f"Return code mismatch: GT={gt_run.returncode}, Pred={pred_run.returncode}"
        )


# ---------------------------------------------------------------------------
# Standalone equivalence checker (for use outside pytest)
# ---------------------------------------------------------------------------

def check_equivalence(
    predicted_ll: str,
    ground_truth_c_path: str,
    func_name: str = "f",
) -> dict:
    """
    Check functional equivalence between predicted IR and ground truth C.
    
    Returns:
        {
            'equivalent': bool or None (if compilation fails),
            'gt_output': str,
            'pred_output': str,
            'gt_returncode': int,
            'pred_returncode': int,
            'error': str or None,
        }
    """
    import tempfile
    results = {
        "equivalent": None,
        "gt_output": None,
        "pred_output": None,
        "gt_returncode": None,
        "pred_returncode": None,
        "error": None,
    }

    with tempfile.TemporaryDirectory() as tmp:
        # Compile ground truth
        gt_exe = os.path.join(tmp, "gt.x")
        r = CompilerHelper.compile_c_to_executable(ground_truth_c_path, gt_exe)
        if r.returncode != 0:
            results["error"] = f"GT compilation failed: {r.stderr}"
            return results

        gt_run = CompilerHelper.run_executable(gt_exe)
        results["gt_output"] = gt_run.stdout
        results["gt_returncode"] = gt_run.returncode

        # Build driver + predicted IR
        driver_source = _build_test_driver(func_name, Path(ground_truth_c_path).read_text())
        if driver_source is None:
            results["error"] = f"No test driver for function '{func_name}'"
            return results

        driver_path = os.path.join(tmp, "driver.c")
        with open(driver_path, "w") as f:
            f.write(driver_source)

        pred_exe = os.path.join(tmp, "pred.x")
        r = CompilerHelper.compile_and_link(predicted_ll, driver_path, pred_exe)
        if r.returncode != 0:
            results["error"] = f"Predicted IR compilation failed: {r.stderr}"
            return results

        pred_run = CompilerHelper.run_executable(pred_exe)
        results["pred_output"] = pred_run.stdout
        results["pred_returncode"] = pred_run.returncode
        results["equivalent"] = (
            pred_run.stdout == gt_run.stdout
            and pred_run.returncode == gt_run.returncode
        )

    return results
