"""Tests for IR target cleaning and post-processing utilities."""
import pytest

from forklift.utils import truncate_ir_output
from neurel_deob.training.data import strip_ir_noise


# ══════════════════════════════════════════════════════════════════════
# strip_ir_noise
# ══════════════════════════════════════════════════════════════════════

class TestStripIrNoise:
    """Tests for removing noise from IR training targets."""

    def test_removes_declare_lines(self):
        ir = (
            "define dso_local i32 @f(i32 %0) {\n"
            "  ret i32 %0\n"
            "}\n"
            "declare dso_local i32 @printf(i8*, ...)\n"
            "declare dso_local i32 @scanf(i8*, float*)\n"
        )
        cleaned = strip_ir_noise(ir)
        assert "declare" not in cleaned
        assert "define dso_local i32 @f" in cleaned
        assert "ret i32 %0" in cleaned
        assert "}" in cleaned

    def test_removes_attributes(self):
        ir = (
            "define dso_local void @f() {\n"
            "  ret void\n"
            "}\n"
            'attributes #0 = { noinline nounwind optnone "frame-pointer"="all" }\n'
        )
        cleaned = strip_ir_noise(ir)
        assert "attributes" not in cleaned
        assert "ret void" in cleaned

    def test_removes_metadata(self):
        ir = (
            "define dso_local void @f() {\n"
            "  ret void\n"
            "}\n"
            "!0 = !{i32 1, !\"wchar_size\", i32 4}\n"
            "!1 = distinct !{!1, !2}\n"
        )
        cleaned = strip_ir_noise(ir)
        assert "!0 =" not in cleaned
        assert "!1 =" not in cleaned
        assert "ret void" in cleaned

    def test_keeps_struct_definitions(self):
        ir = (
            "%struct.foo = type { i32, i32 }\n"
            "define dso_local void @f(%struct.foo* %0) {\n"
            "  ret void\n"
            "}\n"
            "declare dso_local void @bar()\n"
        )
        cleaned = strip_ir_noise(ir)
        assert "%struct.foo = type" in cleaned
        assert "declare" not in cleaned

    def test_keeps_global_constants(self):
        ir = (
            "@.str = external hidden unnamed_addr constant [4 x i8], align 1\n"
            "define dso_local void @f() {\n"
            "  ret void\n"
            "}\n"
            "declare dso_local i32 @printf(i8*, ...)\n"
        )
        cleaned = strip_ir_noise(ir)
        assert "@.str" in cleaned
        assert "declare" not in cleaned

    def test_empty_input(self):
        assert strip_ir_noise("") == ""
        assert strip_ir_noise(None) is None

    def test_no_noise_unchanged(self):
        ir = (
            "define dso_local i32 @f(i32 %0) {\n"
            "  %2 = add nsw i32 %0, 1\n"
            "  ret i32 %2\n"
            "}"
        )
        cleaned = strip_ir_noise(ir)
        assert cleaned.strip() == ir.strip()

    def test_mixed_noise(self):
        """Realistic ExeBench IR with all noise types."""
        ir = (
            "%struct.TYPE_5__ = type { i64, i64, i32* }\n"
            "@.str = external hidden unnamed_addr constant [27 x i8], align 1\n"
            "define dso_local void @sacaC(%struct.TYPE_5__* %0, i32* %1) {\n"
            "  %3 = alloca %struct.TYPE_5__*, align 8\n"
            "  store %struct.TYPE_5__* %0, %struct.TYPE_5__** %3, align 8\n"
            "  ret void\n"
            "}\n"
            "declare dso_local i32 @vaciaC(%struct.TYPE_5__* byval(%struct.TYPE_5__) align 8)\n"
            "declare dso_local i32 @iniciaC(%struct.TYPE_5__*)\n"
            'attributes #0 = { noinline nounwind optnone "frame-pointer"="all" }\n'
            "!0 = !{i32 1, !\"wchar_size\", i32 4}\n"
        )
        cleaned = strip_ir_noise(ir)
        # Kept
        assert "%struct.TYPE_5__" in cleaned
        assert "@.str" in cleaned
        assert "define dso_local void @sacaC" in cleaned
        assert "ret void" in cleaned
        # Removed
        assert "declare" not in cleaned
        assert "attributes" not in cleaned
        assert "!0 =" not in cleaned


# ══════════════════════════════════════════════════════════════════════
# truncate_ir_output
# ══════════════════════════════════════════════════════════════════════

class TestTruncateIrOutput:
    """Tests for post-processing truncation of generated IR."""

    def test_truncates_after_closing_brace(self):
        ir = (
            "define dso_local i32 @f(i32 %0) {\n"
            "  ret i32 %0\n"
            "}\n"
            "declare dso_local i32 @free(i32)\n"
            "declare dso_local i32 @free(i32)\n"
            "declare dso_local i32 @free(i32)\n"
        )
        truncated = truncate_ir_output(ir)
        assert truncated.strip().endswith("}")
        assert "declare" not in truncated

    def test_keeps_preamble(self):
        ir = (
            "%struct.struct0 = type { i32, i32 }\n"
            "@.str = external constant [4 x i8]\n"
            "define dso_local i32 @f(i32 %0) {\n"
            "  ret i32 %0\n"
            "}\n"
            "garbage after\n"
        )
        truncated = truncate_ir_output(ir)
        assert "%struct.struct0" in truncated
        assert "@.str" in truncated
        assert "garbage" not in truncated

    def test_no_function_body(self):
        """If there's no define/}, return the original."""
        ir = "%struct.foo = type { i32 }\n"
        truncated = truncate_ir_output(ir)
        assert truncated == ir

    def test_empty_input(self):
        assert truncate_ir_output("") == ""
        assert truncate_ir_output(None) is None

    def test_brace_inside_body_not_truncated(self):
        """A } inside a label (e.g., in a comment) shouldn't trigger truncation."""
        ir = (
            "define dso_local void @f() {\n"
            "  br label %1\n"
            "1:\n"
            "  ret void\n"
            "}\n"
        )
        truncated = truncate_ir_output(ir)
        assert "ret void" in truncated
        assert truncated.strip().endswith("}")

    def test_realistic_degenerate_output(self):
        """Simulates what the fine-tuned model actually produces."""
        ir = (
            "%struct.struct0 = type { i32 }\n"
            "define dso_local i32 @f(i32 %0) {\n"
            "  %2 = alloca i32, align 4\n"
            "  store i32 %0, i32* %2, align 4\n"
            "  %3 = load i32, i32* %2, align 4\n"
            "  %4 = add nsw i32 %3, 1\n"
            "  ret i32 %4\n"
            "}\n"
            "declare dso_local i32 @free(i32)\n"
            "declare dso_local i32 @free(i32)\n"
            "declare dso_local i32 @free(i32)\n"
            "declare dso_local i32 @free(i32)\n"
            "declare dso_local i32 @free(i32)\n"
            "declare dso_local i32)\n"
            "declare dso_local i32 @free(i32)\n"
            "declare964 noalias = external964, i32)\n"
            "i32, i32, i32, i32)\n"
        )
        truncated = truncate_ir_output(ir)
        lines = truncated.strip().splitlines()
        assert lines[-1].strip() == "}"
        assert "add nsw" in truncated
        assert "declare" not in truncated
        assert len(lines) == 8  # struct + define + 5 body + }
