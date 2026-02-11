"""
Level 1 & 2: LLVM IR Validation Tests

Level 1 - Syntax: Runs llvm-as to verify the predicted IR is syntactically valid LLVM IR.
Level 2 - Compilation: Runs clang to verify the IR can be compiled to object code.
"""

import pytest
from pathlib import Path
from .conftest import discover_test_cases, CompilerHelper


# ---------------------------------------------------------------------------
# Discover all test cases that have predicted IR
# ---------------------------------------------------------------------------

PREDICTED_IR_CASES = [tc for tc in discover_test_cases() if tc.predicted_ll_path and tc.predicted_ll_path.exists()]


# ---------------------------------------------------------------------------
# Level 1: Syntax validation with llvm-as
# ---------------------------------------------------------------------------

class TestIRSyntax:
    """Verify predicted LLVM IR is syntactically valid using llvm-as."""

    @pytest.mark.parametrize(
        "test_case",
        PREDICTED_IR_CASES,
        ids=[tc.name for tc in PREDICTED_IR_CASES],
    )
    def test_predicted_ir_syntax(self, test_case, require_llvm_as):
        """Predicted LLVM IR should parse without errors."""
        ir_content = test_case.predicted_ll
        assert ir_content is not None, f"No predicted IR for {test_case.name}"
        assert len(ir_content.strip()) > 0, f"Predicted IR is empty for {test_case.name}"

        result = CompilerHelper.validate_ir_syntax(ir_content)
        assert result.returncode == 0, (
            f"llvm-as failed for {test_case.name}:\n{result.stderr}"
        )

    @pytest.mark.parametrize(
        "test_case",
        [tc for tc in PREDICTED_IR_CASES if tc.ground_truth_ll_path and tc.ground_truth_ll_path.exists()],
        ids=[tc.name for tc in PREDICTED_IR_CASES if tc.ground_truth_ll_path and tc.ground_truth_ll_path.exists()],
    )
    def test_ground_truth_ir_syntax(self, test_case, require_llvm_as):
        """Ground truth LLVM IR should also parse (sanity check)."""
        ir_content = test_case.ground_truth_ll
        assert ir_content is not None
        result = CompilerHelper.validate_ir_syntax(ir_content)
        assert result.returncode == 0, (
            f"llvm-as failed for ground truth {test_case.name}:\n{result.stderr}"
        )


# ---------------------------------------------------------------------------
# Level 2: Compilation to object code with clang
# ---------------------------------------------------------------------------

class TestIRCompilation:
    """Verify predicted LLVM IR can be compiled to object code."""

    @pytest.mark.parametrize(
        "test_case",
        PREDICTED_IR_CASES,
        ids=[tc.name for tc in PREDICTED_IR_CASES],
    )
    def test_predicted_ir_compiles(self, test_case, require_clang):
        """Predicted LLVM IR should compile to an object file without errors."""
        ir_content = test_case.predicted_ll
        assert ir_content is not None, f"No predicted IR for {test_case.name}"

        result = CompilerHelper.compile_ir_to_object(ir_content)
        assert result.returncode == 0, (
            f"clang compilation failed for {test_case.name}:\n{result.stderr}"
        )


# ---------------------------------------------------------------------------
# Standalone validation function (for use outside pytest)
# ---------------------------------------------------------------------------

def validate_ir_string(ir_content: str) -> dict:
    """
    Validate an LLVM IR string and return a results dict.
    Useful for calling from inference scripts.

    Returns:
        {
            'syntax_valid': bool,
            'syntax_error': str or None,
            'compiles': bool,
            'compile_error': str or None,
        }
    """
    results = {
        "syntax_valid": False,
        "syntax_error": None,
        "compiles": False,
        "compile_error": None,
    }

    # Level 1: syntax
    if CompilerHelper.check_tool_available("llvm-as"):
        r = CompilerHelper.validate_ir_syntax(ir_content)
        results["syntax_valid"] = r.returncode == 0
        if r.returncode != 0:
            results["syntax_error"] = r.stderr
            return results  # No point trying to compile if syntax is broken
    else:
        results["syntax_error"] = "llvm-as not available"

    # Level 2: compilation
    if CompilerHelper.check_tool_available("clang"):
        r = CompilerHelper.compile_ir_to_object(ir_content)
        results["compiles"] = r.returncode == 0
        if r.returncode != 0:
            results["compile_error"] = r.stderr
    else:
        results["compile_error"] = "clang not available"

    return results
