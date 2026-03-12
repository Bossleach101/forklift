"""
Tests for forklift.ir_checker – LLVM IR validation and functional correctness.

Covers Level 1 (syntax / llvm-as), Level 2 (compilation / clang -c),
Level 3 (functional correctness via IO pairs), and helper utilities.
"""

import pytest
from forklift.ir_checker import (
    validate_ir_syntax,
    compile_ir_to_object,
    check_ir,
    diff_io,
    exebench_dict_to_dict,
    CompilabilityStats,
    IRCheckResult,
    _tool_available,
    _inject_missing_declares,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_IR = """\
define dso_local i32 @add(i32 %0, i32 %1) {
  %3 = add nsw i32 %0, %1
  ret i32 %3
}
"""

INVALID_IR = "this is not valid llvm ir at all"

# Syntactically valid but references undefined function (fails llvm-as)
UNDEF_FUNC_IR = """\
define dso_local i32 @foo(i32 %0) {
  %2 = call i32 @undefined_func(i32 %0)
  ret i32 %2
}
"""

# Valid IR that compiles but does nothing (for functional failure tests)
TRIVIAL_VOID_IR = """\
define dso_local void @noop() {
  ret void
}
"""


# Skip if LLVM tools not available
requires_llvm_as = pytest.mark.skipif(
    not _tool_available("llvm-as"),
    reason="llvm-as not available",
)
requires_clang = pytest.mark.skipif(
    not _tool_available("clang"),
    reason="clang not available",
)


# ---------------------------------------------------------------------------
# Level 1: Syntax (llvm-as)
# ---------------------------------------------------------------------------

class TestValidateIRSyntax:
    """Test validate_ir_syntax function."""

    @requires_llvm_as
    def test_valid_ir(self):
        ok, err = validate_ir_syntax(VALID_IR)
        assert ok is True
        assert err is None

    @requires_llvm_as
    def test_invalid_ir(self):
        ok, err = validate_ir_syntax(INVALID_IR)
        assert ok is False
        assert err is not None
        assert "error" in err.lower()

    @requires_llvm_as
    def test_undefined_function_fails_syntax(self):
        ok, err = validate_ir_syntax(UNDEF_FUNC_IR)
        assert ok is False
        assert "undefined" in err.lower()

    @requires_llvm_as
    def test_undefined_function_passes_with_auto_declare(self):
        ok, err = validate_ir_syntax(UNDEF_FUNC_IR, auto_declare=True)
        assert ok is True
        assert err is None

    @requires_llvm_as
    def test_auto_declare_does_not_fix_real_errors(self):
        ok, err = validate_ir_syntax(INVALID_IR, auto_declare=True)
        assert ok is False

    @requires_llvm_as
    def test_empty_ir(self):
        ok, err = validate_ir_syntax("")
        # Empty file is valid (no errors, just no content)
        assert ok is True

    @requires_llvm_as
    def test_ir_with_types_and_globals(self):
        ir = """\
%struct.Point = type { i32, i32 }
@g = global %struct.Point { i32 1, i32 2 }

define dso_local i32 @get_x(%struct.Point* %0) {
  %2 = getelementptr inbounds %struct.Point, %struct.Point* %0, i32 0, i32 0
  %3 = load i32, i32* %2
  ret i32 %3
}
"""
        ok, err = validate_ir_syntax(ir)
        assert ok is True


# ---------------------------------------------------------------------------
# Level 2: Compilation (clang -c)
# ---------------------------------------------------------------------------

class TestCompileIRToObject:
    """Test compile_ir_to_object function."""

    @requires_clang
    def test_valid_ir_compiles(self):
        ok, err = compile_ir_to_object(VALID_IR)
        assert ok is True
        assert err is None

    @requires_clang
    def test_invalid_ir_does_not_compile(self):
        ok, err = compile_ir_to_object(INVALID_IR)
        assert ok is False
        assert err is not None

    @requires_clang
    def test_trivial_void_compiles(self):
        ok, err = compile_ir_to_object(TRIVIAL_VOID_IR)
        assert ok is True


# ---------------------------------------------------------------------------
# check_ir (multi-level)
# ---------------------------------------------------------------------------

class TestCheckIR:
    """Test the combined check_ir function."""

    @requires_llvm_as
    def test_level1_valid(self):
        result = check_ir(VALID_IR, level=1)
        assert result.syntax_valid is True
        assert result.compiles is False  # not tested at level 1

    @requires_llvm_as
    def test_level1_invalid(self):
        result = check_ir(INVALID_IR, level=1)
        assert result.syntax_valid is False

    @requires_clang
    def test_level2_valid(self):
        result = check_ir(VALID_IR, level=2)
        assert result.syntax_valid is True
        assert result.compiles is True

    @requires_clang
    def test_level2_syntax_fail_stops_early(self):
        result = check_ir(INVALID_IR, level=2)
        assert result.syntax_valid is False
        assert result.compiles is False  # never reached

    @requires_clang
    def test_level3_without_metadata(self):
        result = check_ir(VALID_IR, level=3)
        assert result.syntax_valid is True
        assert result.compiles is True
        assert result.links is False  # no wrapper provided
        assert result.functional_pass is None


# ---------------------------------------------------------------------------
# diff_io
# ---------------------------------------------------------------------------

class TestDiffIO:
    """Test IO comparison function."""

    def test_equal_dicts(self):
        assert diff_io({"a": 1, "b": 2}, {"a": 1, "b": 2}) is True

    def test_unequal_dicts(self):
        assert diff_io({"a": 1}, {"a": 2}) is False

    def test_missing_key(self):
        assert diff_io({"a": 1}, {"a": 1, "b": 2}) is False

    def test_extra_key(self):
        assert diff_io({"a": 1, "b": 2}, {"a": 1}) is False

    def test_nested_dicts(self):
        a = {"x": {"y": [1, 2, 3]}}
        b = {"x": {"y": [1, 2, 3]}}
        assert diff_io(a, b) is True

    def test_nested_dicts_differ(self):
        a = {"x": {"y": [1, 2, 3]}}
        b = {"x": {"y": [1, 2, 4]}}
        assert diff_io(a, b) is False

    def test_lists(self):
        assert diff_io([1, 2, 3], [1, 2, 3]) is True
        assert diff_io([1, 2, 3], [1, 2, 4]) is False
        assert diff_io([1, 2], [1, 2, 3]) is False

    def test_floats_close(self):
        assert diff_io(1.0, 1.0001) is True  # rel_tol=1e-3

    def test_floats_far(self):
        assert diff_io(1.0, 2.0) is False

    def test_int_float_cross(self):
        assert diff_io(1, 1.0) is True
        assert diff_io(1, 1.5) is False

    def test_type_mismatch(self):
        assert diff_io("hello", 42) is False

    def test_strings(self):
        assert diff_io("hello", "hello") is True
        assert diff_io("hello", "world") is False


# ---------------------------------------------------------------------------
# exebench_dict_to_dict
# ---------------------------------------------------------------------------

class TestExebenchDictToDict:
    """Test ExeBench format conversion."""

    def test_simple(self):
        eb = {"var": ["a", "b"], "value": ["1", "2"]}
        result = exebench_dict_to_dict(eb)
        assert result == {"a": 1, "b": 2}

    def test_list_value(self):
        eb = {"var": ["arr"], "value": ["[1, 2, 3]"]}
        result = exebench_dict_to_dict(eb)
        assert result == {"arr": [1, 2, 3]}

    def test_nested_dict_value(self):
        eb = {"var": ["obj"], "value": ['{"x": 1, "y": 2}']}
        result = exebench_dict_to_dict(eb)
        assert result == {"obj": {"x": 1, "y": 2}}

    def test_string_value(self):
        eb = {"var": ["name"], "value": ["'hello'"]}
        result = exebench_dict_to_dict(eb)
        assert result == {"name": "hello"}


# ---------------------------------------------------------------------------
# CompilabilityStats
# ---------------------------------------------------------------------------

class TestCompilabilityStats:
    """Test the stats aggregation class."""

    def test_empty(self):
        stats = CompilabilityStats()
        d = stats.to_dict()
        assert d["total"] == 0
        assert d["syntax_valid_pct"] == 0.0

    def test_update_syntax_valid(self):
        stats = CompilabilityStats()
        r = IRCheckResult(syntax_valid=True, compiles=True)
        stats.update(r, fname="test")
        d = stats.to_dict()
        assert d["total"] == 1
        assert d["syntax_valid"] == 1
        assert d["compiles"] == 1
        assert d["syntax_valid_pct"] == 100.0

    def test_update_mixed(self):
        stats = CompilabilityStats()
        stats.update(IRCheckResult(syntax_valid=True, compiles=True), "a")
        stats.update(IRCheckResult(syntax_valid=True, compiles=False), "b")
        stats.update(IRCheckResult(syntax_valid=False), "c")
        d = stats.to_dict()
        assert d["total"] == 3
        assert d["syntax_valid"] == 2
        assert d["compiles"] == 1
        assert d["syntax_valid_pct"] == pytest.approx(66.67, abs=0.01)
        assert d["compiles_pct"] == pytest.approx(33.33, abs=0.01)

    def test_functional_stats(self):
        stats = CompilabilityStats()
        r = IRCheckResult(
            syntax_valid=True, compiles=True, links=True,
            functional_pass=True, io_pass_count=5, io_total_count=5,
        )
        stats.update(r, "fn1")
        d = stats.to_dict()
        assert d["functional_pass"] == 1
        assert d["functional_tested"] == 1
        assert d["io_pass_total"] == 5
        assert d["io_tested_total"] == 5
        assert d["io_pass_pct"] == 100.0

    def test_errors_capped(self):
        stats = CompilabilityStats()
        for i in range(30):
            stats.update(IRCheckResult(syntax_valid=False, syntax_error=f"err{i}"), f"fn{i}")
        assert len(stats.errors) == 20  # capped at 20


# ---------------------------------------------------------------------------
# IRCheckResult
# ---------------------------------------------------------------------------

class TestIRCheckResult:
    """Test result dataclass."""

    def test_functional_pct_zero(self):
        r = IRCheckResult()
        assert r.functional_pct == 0.0

    def test_functional_pct(self):
        r = IRCheckResult(io_pass_count=3, io_total_count=5)
        assert r.functional_pct == 60.0


# ---------------------------------------------------------------------------
# _inject_missing_declares (auto_declare)
# ---------------------------------------------------------------------------

class TestInjectMissingDeclares:
    """Test the auto-declare stub injection."""

    @requires_llvm_as
    def test_injects_stub_for_undefined_call(self):
        """Calling an undefined function should get a declare stub."""
        patched = _inject_missing_declares(UNDEF_FUNC_IR)
        assert "declare" in patched
        assert "@undefined_func" in patched

    @requires_llvm_as
    def test_valid_ir_unchanged(self):
        """Valid IR should not be modified."""
        patched = _inject_missing_declares(VALID_IR)
        assert patched == VALID_IR

    @requires_llvm_as
    def test_patched_ir_passes_syntax(self):
        """After injection, the previously-invalid IR should parse."""
        patched = _inject_missing_declares(UNDEF_FUNC_IR)
        ok, err = validate_ir_syntax(patched)
        assert ok is True, f"Expected valid after auto-declare, got: {err}"

    @requires_llvm_as
    def test_multiple_undefined_functions(self):
        """Multiple undefined functions should all get stubs."""
        ir = """\
define dso_local i32 @bar(i32 %0) {
  %2 = call i32 @alpha(i32 %0)
  %3 = call i32 @beta(i32 %2)
  ret i32 %3
}
"""
        patched = _inject_missing_declares(ir)
        assert "@alpha" in patched
        assert "@beta" in patched
        ok, _ = validate_ir_syntax(patched)
        assert ok is True

    @requires_llvm_as
    def test_does_not_fix_non_declare_errors(self):
        """Totally broken IR stays broken — we only fix undefined values."""
        patched = _inject_missing_declares(INVALID_IR)
        ok, _ = validate_ir_syntax(patched)
        assert ok is False

    @requires_llvm_as
    def test_compile_with_auto_declare(self):
        """compile_ir_to_object should also support auto_declare."""
        ok, err = compile_ir_to_object(UNDEF_FUNC_IR, auto_declare=True)
        assert ok is True, f"Expected compilation with auto_declare, got: {err}"

    @requires_llvm_as
    def test_check_ir_auto_declare_default(self):
        """check_ir uses auto_declare=True by default."""
        result = check_ir(UNDEF_FUNC_IR, level=1)
        assert result.syntax_valid is True
