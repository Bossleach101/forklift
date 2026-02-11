"""
Shared fixtures and helpers for the LLVM IR testing framework.

Provides:
  - Paths to test data directory
  - Compiler helpers (compile C, assemble IR, link, run)
  - Ground-truth test case discovery
"""

import pytest
import subprocess
import tempfile
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

TEST_DATA_DIR = Path(__file__).parent / "test_data"
MYTESTS_DIR = TEST_DATA_DIR / "mytests"


@pytest.fixture
def test_data_dir():
    """Return path to the tests/test_data/ directory."""
    return TEST_DATA_DIR


@pytest.fixture
def mytests_dir():
    """Return path to the tests/test_data/mytests/ directory."""
    return MYTESTS_DIR


# ---------------------------------------------------------------------------
# Data classes for test cases
# ---------------------------------------------------------------------------

@dataclass
class IRTestCase:
    """A ground-truth test case: C source + expected LLVM IR + optional assembly."""
    name: str
    c_source_path: Optional[Path] = None
    ground_truth_ll_path: Optional[Path] = None
    predicted_ll_path: Optional[Path] = None
    asm_path: Optional[Path] = None
    test_driver_path: Optional[Path] = None  # C file with main() that calls the function

    @property
    def c_source(self) -> Optional[str]:
        if self.c_source_path and self.c_source_path.exists():
            return self.c_source_path.read_text()
        return None

    @property
    def ground_truth_ll(self) -> Optional[str]:
        if self.ground_truth_ll_path and self.ground_truth_ll_path.exists():
            return self.ground_truth_ll_path.read_text()
        return None

    @property
    def predicted_ll(self) -> Optional[str]:
        if self.predicted_ll_path and self.predicted_ll_path.exists():
            return self.predicted_ll_path.read_text()
        return None


# ---------------------------------------------------------------------------
# Test case fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def f_test_case():
    """The primary 'f' test case (array add loop)."""
    return IRTestCase(
        name="f_ground_truth",
        c_source_path=TEST_DATA_DIR / "f_ground_truth.c",
        ground_truth_ll_path=TEST_DATA_DIR / "f_ground_truth.ll",
        predicted_ll_path=TEST_DATA_DIR / "f_predicted.ll",
        asm_path=TEST_DATA_DIR / "f_ground_truth.s",
    )


def discover_test_cases() -> List[IRTestCase]:
    """Discover all available test cases from test_data/."""
    cases = []

    # Main f test case
    if (TEST_DATA_DIR / "f_ground_truth.c").exists():
        cases.append(IRTestCase(
            name="f_ground_truth",
            c_source_path=TEST_DATA_DIR / "f_ground_truth.c",
            ground_truth_ll_path=TEST_DATA_DIR / "f_ground_truth.ll",
            predicted_ll_path=TEST_DATA_DIR / "f_predicted.ll",
            asm_path=TEST_DATA_DIR / "f_ground_truth.s",
        ))

    # mytests cases
    if MYTESTS_DIR.exists():
        for ll_file in sorted(MYTESTS_DIR.glob("*_predicted.ll")):
            base = ll_file.stem.replace("_predicted", "")
            cases.append(IRTestCase(
                name=f"mytests/{base}",
                c_source_path=MYTESTS_DIR / f"{base}.c" if (MYTESTS_DIR / f"{base}.c").exists() else None,
                ground_truth_ll_path=MYTESTS_DIR / f"{base}.ll" if (MYTESTS_DIR / f"{base}.ll").exists() else None,
                predicted_ll_path=ll_file,
                asm_path=MYTESTS_DIR / f"{base}.s" if (MYTESTS_DIR / f"{base}.s").exists() else None,
            ))

    return cases


# ---------------------------------------------------------------------------
# Compiler / Tool helpers
# ---------------------------------------------------------------------------

class CompilerHelper:
    """Wraps clang / llvm-as for compiling and validating LLVM IR."""

    @staticmethod
    def check_tool_available(tool: str) -> bool:
        """Check if a command-line tool is available."""
        try:
            subprocess.run([tool, "--version"], capture_output=True, timeout=10)
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    @staticmethod
    def validate_ir_syntax(ll_content: str) -> subprocess.CompletedProcess:
        """
        Run llvm-as on LLVM IR text to check syntax validity.
        Returns the CompletedProcess — check .returncode for success.
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ll", delete=False) as f:
            f.write(ll_content)
            f.flush()
            try:
                result = subprocess.run(
                    ["llvm-as", f.name, "-o", "/dev/null"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                return result
            finally:
                os.unlink(f.name)

    @staticmethod
    def compile_ir_to_object(ll_content: str, output_path: Optional[str] = None) -> subprocess.CompletedProcess:
        """
        Compile LLVM IR text to an object file using clang.
        Returns the CompletedProcess.
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ll", delete=False) as f:
            f.write(ll_content)
            f.flush()
            if output_path is None:
                output_path = f.name.replace(".ll", ".o")
            try:
                result = subprocess.run(
                    ["clang", "-c", f.name, "-o", output_path],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                return result
            finally:
                os.unlink(f.name)
                if os.path.exists(output_path):
                    os.unlink(output_path)

    @staticmethod
    def compile_and_link(
        ll_content: str,
        test_driver_path: str,
        output_path: Optional[str] = None,
    ) -> subprocess.CompletedProcess:
        """
        Compile LLVM IR + a C test driver into an executable.
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ll", delete=False) as f:
            f.write(ll_content)
            f.flush()
            if output_path is None:
                output_path = f.name.replace(".ll", ".x")
            try:
                result = subprocess.run(
                    ["clang", test_driver_path, f.name, "-o", output_path],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                return result
            finally:
                os.unlink(f.name)

    @staticmethod
    def run_executable(exe_path: str, timeout: int = 10) -> subprocess.CompletedProcess:
        """Run a compiled executable and capture output."""
        return subprocess.run(
            [exe_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    @staticmethod
    def compile_c_to_executable(
        c_source_path: str, output_path: Optional[str] = None
    ) -> subprocess.CompletedProcess:
        """Compile a C source file (with main) to an executable."""
        if output_path is None:
            output_path = c_source_path.replace(".c", ".x")
        return subprocess.run(
            ["clang", c_source_path, "-o", output_path],
            capture_output=True,
            text=True,
            timeout=30,
        )

    @staticmethod
    def compile_c_to_ir(
        c_source_path: str,
        output_path: Optional[str] = None,
        opt_level: str = "0",
        target: Optional[str] = None,
    ) -> subprocess.CompletedProcess:
        """
        Compile C source to LLVM IR using clang -S -emit-llvm.
        Useful for generating new ground-truth IR from C files.
        """
        if output_path is None:
            output_path = c_source_path.replace(".c", ".ll")
        cmd = ["clang", "-S", "-emit-llvm", f"-O{opt_level}", c_source_path, "-o", output_path]
        if target:
            cmd.insert(1, f"--target={target}")
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30)

    @staticmethod
    def compile_c_to_asm(
        c_source_path: str,
        output_path: Optional[str] = None,
        opt_level: str = "0",
        target: Optional[str] = None,
    ) -> subprocess.CompletedProcess:
        """
        Compile C source to assembly using clang -S.
        Useful for generating AArch64 assembly from C files.
        """
        if output_path is None:
            output_path = c_source_path.replace(".c", ".s")
        cmd = ["clang", "-S", f"-O{opt_level}", c_source_path, "-o", output_path]
        if target:
            cmd.insert(1, f"--target={target}")
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30)


@pytest.fixture
def compiler():
    """Provide a CompilerHelper instance."""
    return CompilerHelper()


@pytest.fixture
def require_clang():
    """Skip test if clang is not available."""
    if not CompilerHelper.check_tool_available("clang"):
        pytest.skip("clang not available")


@pytest.fixture
def require_llvm_as():
    """Skip test if llvm-as is not available."""
    if not CompilerHelper.check_tool_available("llvm-as"):
        pytest.skip("llvm-as not available")
