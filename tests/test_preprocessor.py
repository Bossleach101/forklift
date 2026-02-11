"""
Tests for the AsmPreprocessor module.

Validates architecture detection, function extraction, comment stripping,
directive removal, header normalisation, and end-to-end parity with the
Forklift pipeline (asm.py).
"""
import subprocess
import tempfile
import textwrap

import pytest

from neurel_deob.asm_preprocessor import Arch, AsmPreprocessor


# =====================================================================
# Fixtures: inline assembly snippets
# =====================================================================


AARCH64_ASM = textwrap.dedent("""\
    .text
    .file	"test.c"
    .globl	f                               // -- Begin function f
    .p2align	2
    .type	f,@function
    f:                                      // @f
    	.cfi_startproc
    // %bb.0:
    	sub	sp, sp, #16
    	.cfi_def_cfa_offset 16
    	str	w0, [sp, #12]
    	ldr	w8, [sp, #12]
    	add	w0, w8, #1
    	add	sp, sp, #16
    	.cfi_def_cfa_offset 0
    	ret
    .Lfunc_end0:
    	.size	f, .Lfunc_end0-f
    	.cfi_endproc
                                            // -- End function
    .ident	"clang version 19"
    .section	".note.GNU-stack","",@progbits
    .addrsig
""")


X86_ASM = textwrap.dedent("""\
    .text
    .file	"test.c"
    .globl	f                               # -- Begin function f
    .p2align	4, 0x90
    .type	f,@function
    f:                                      # @f
    	.cfi_startproc
    # %bb.0:
    	pushq	%rbp
    	.cfi_def_cfa_offset 16
    	.cfi_offset %rbp, -16
    	movq	%rsp, %rbp
    	.cfi_def_cfa_register %rbp
    	movl	%edi, -4(%rbp)
    	movl	-4(%rbp), %eax
    	addl	$1, %eax
    	popq	%rbp
    	.cfi_def_cfa %rsp, 8
    	retq
    .Lfunc_end0:
    	.size	f, .Lfunc_end0-f
    	.cfi_endproc
                                            # -- End function
    .ident	"clang version 19"
    .section	".note.GNU-stack","",@progbits
    .addrsig
""")


# AArch64 file with NO .cfi_endproc (tests fallback)
AARCH64_NO_CFI = textwrap.dedent("""\
    .text
    .globl	g
    .type	g,@function
    g:
    	sub	sp, sp, #16
    	str	w0, [sp, #12]
    	ldr	w0, [sp, #12]
    	add	sp, sp, #16
    	ret
    .Lfunc_end0:
    	.size	g, .Lfunc_end0-g
""")


# Multi-function file
MULTI_FUNC_ASM = textwrap.dedent("""\
    .text
    .globl	helper                          // -- Begin function helper
    .p2align	2
    .type	helper,@function
    helper:
    	.cfi_startproc
    	mov	w0, #42
    	ret
    	.cfi_endproc

    .globl	target                          // -- Begin function target
    .p2align	2
    .type	target,@function
    target:
    	.cfi_startproc
    	sub	sp, sp, #16
    	.cfi_def_cfa_offset 16
    	str	w0, [sp, #12]
    	ldr	w8, [sp, #12]
    	add	w0, w8, #10
    	add	sp, sp, #16
    	.cfi_def_cfa_offset 0
    	ret
    .Lfunc_end1:
    	.size	target, .Lfunc_end1-target
    	.cfi_endproc

    .ident	"clang version 19"
""")


# =====================================================================
# Architecture detection
# =====================================================================


class TestArchDetection:
    def test_detect_aarch64(self):
        p = AsmPreprocessor(asm_code=AARCH64_ASM, target_function='f')
        assert p.arch == Arch.AARCH64

    def test_detect_x86(self):
        p = AsmPreprocessor(asm_code=X86_ASM, target_function='f')
        assert p.arch == Arch.X86

    def test_explicit_arch_override(self):
        """Explicit arch should override detection."""
        p = AsmPreprocessor(asm_code=AARCH64_ASM, target_function='f', arch=Arch.X86)
        assert p.arch == Arch.X86

    def test_multi_func_aarch64(self):
        """Multi-function file should still detect AArch64 from // comments."""
        p = AsmPreprocessor(asm_code=MULTI_FUNC_ASM, target_function='target')
        assert p.arch == Arch.AARCH64


# =====================================================================
# Function extraction
# =====================================================================


class TestFunctionExtraction:
    def test_extract_aarch64_function(self):
        p = AsmPreprocessor(asm_code=AARCH64_ASM, target_function='f')
        assert 'f:' in p.func_asm
        assert '.cfi_startproc' in p.func_asm
        assert '.cfi_endproc' in p.func_asm
        assert 'ret' in p.func_asm

    def test_extract_x86_function(self):
        p = AsmPreprocessor(asm_code=X86_ASM, target_function='f')
        assert 'f:' in p.func_asm
        assert '.cfi_startproc' in p.func_asm
        assert '.cfi_endproc' in p.func_asm
        assert 'retq' in p.func_asm

    def test_extract_second_function(self):
        """Should extract 'target' not 'helper' from multi-function file."""
        p = AsmPreprocessor(asm_code=MULTI_FUNC_ASM, target_function='target')
        assert 'target:' in p.func_asm
        assert 'add\tw0, w8, #10' in p.func_asm
        # Should NOT contain helper's body
        assert 'mov\tw0, #42' not in p.func_asm

    def test_extract_first_function(self):
        """Should extract 'helper' correctly."""
        p = AsmPreprocessor(asm_code=MULTI_FUNC_ASM, target_function='helper')
        assert 'helper:' in p.func_asm
        assert 'mov\tw0, #42' in p.func_asm
        # Should NOT contain target's body
        assert 'add\tw0, w8, #10' not in p.func_asm

    def test_fallback_no_cfi_endproc(self):
        """When .cfi_endproc is missing, should use .size/.Lfunc_end fallback."""
        p = AsmPreprocessor(asm_code=AARCH64_NO_CFI, target_function='g', arch=Arch.AARCH64)
        assert 'g:' in p.func_asm
        assert 'ret' in p.func_asm

    def test_missing_function_raises(self):
        with pytest.raises(ValueError, match="not found"):
            AsmPreprocessor(asm_code=AARCH64_ASM, target_function='nonexistent')

    def test_file_path_loading(self, tmp_path):
        """Should load from file path."""
        f = tmp_path / "test.s"
        f.write_text(AARCH64_ASM)
        p = AsmPreprocessor(file_path=str(f), target_function='f')
        assert 'f:' in p.func_asm


# =====================================================================
# Comment stripping
# =====================================================================


class TestCommentStripping:
    def test_aarch64_comments_stripped(self):
        p = AsmPreprocessor(asm_code=AARCH64_ASM, target_function='f')
        # // comments should be gone
        assert '//' not in p.func_asm
        # But instructions should remain
        assert 'sub\tsp, sp, #16' in p.func_asm

    def test_x86_comments_stripped(self):
        p = AsmPreprocessor(asm_code=X86_ASM, target_function='f')
        # # comments should be gone
        for line in p.func_asm.splitlines():
            assert '#' not in line

    def test_comment_only_lines_removed(self):
        """Lines that are ONLY comments (like // %bb.0:) should be removed."""
        p = AsmPreprocessor(asm_code=AARCH64_ASM, target_function='f')
        for line in p.func_asm.splitlines():
            assert '%bb' not in line


# =====================================================================
# Directive removal
# =====================================================================


class TestDirectiveRemoval:
    def test_file_directive_removed(self):
        p = AsmPreprocessor(asm_code=AARCH64_ASM, target_function='f')
        assert '.file' not in p.func_asm

    def test_ident_directive_removed(self):
        p = AsmPreprocessor(asm_code=AARCH64_ASM, target_function='f')
        assert '.ident' not in p.func_asm

    def test_addrsig_removed(self):
        p = AsmPreprocessor(asm_code=AARCH64_ASM, target_function='f')
        assert '.addrsig' not in p.func_asm

    def test_section_removed(self):
        p = AsmPreprocessor(asm_code=AARCH64_ASM, target_function='f')
        assert '.section' not in p.func_asm

    def test_p2align_removed(self):
        p = AsmPreprocessor(asm_code=AARCH64_ASM, target_function='f')
        assert '.p2align' not in p.func_asm

    def test_text_removed(self):
        p = AsmPreprocessor(asm_code=AARCH64_ASM, target_function='f')
        for line in p.func_asm.splitlines():
            assert line.strip() != '.text'

    def test_lfunc_end_removed(self):
        p = AsmPreprocessor(asm_code=AARCH64_ASM, target_function='f')
        assert '.Lfunc_end' not in p.func_asm

    def test_size_removed(self):
        p = AsmPreprocessor(asm_code=AARCH64_ASM, target_function='f')
        assert '.size' not in p.func_asm

    def test_cfi_directives_kept(self):
        """CFI directives MUST be preserved — they're in the training data."""
        p = AsmPreprocessor(asm_code=AARCH64_ASM, target_function='f')
        assert '.cfi_startproc' in p.func_asm
        assert '.cfi_endproc' in p.func_asm
        assert '.cfi_def_cfa_offset 16' in p.func_asm


# =====================================================================
# Header normalisation
# =====================================================================


class TestHeaderNormalisation:
    def test_aarch64_uses_global(self):
        """AArch64 should use .global (not .globl)."""
        p = AsmPreprocessor(asm_code=AARCH64_ASM, target_function='f')
        lines = p.func_asm.splitlines()
        assert '.global f' in lines

    def test_aarch64_uses_percent_function(self):
        """AArch64 should use %function (not @function)."""
        p = AsmPreprocessor(asm_code=AARCH64_ASM, target_function='f')
        lines = p.func_asm.splitlines()
        assert '.type f, %function' in lines

    def test_x86_uses_globl(self):
        """x86 should use .globl (not .global)."""
        p = AsmPreprocessor(asm_code=X86_ASM, target_function='f')
        lines = p.func_asm.splitlines()
        assert '.globl f' in lines

    def test_x86_uses_at_function(self):
        """x86 should use @function (not %function)."""
        p = AsmPreprocessor(asm_code=X86_ASM, target_function='f')
        lines = p.func_asm.splitlines()
        assert '.type f, @function' in lines


# =====================================================================
# Whitespace / format
# =====================================================================


class TestWhitespace:
    def test_no_blank_lines(self):
        p = AsmPreprocessor(asm_code=AARCH64_ASM, target_function='f')
        for line in p.func_asm.splitlines():
            assert line.strip(), f"Blank line found: {line!r}"

    def test_no_trailing_whitespace(self):
        p = AsmPreprocessor(asm_code=AARCH64_ASM, target_function='f')
        for line in p.func_asm.splitlines():
            assert line == line.rstrip(), f"Trailing whitespace: {line!r}"

    def test_ends_with_newline(self):
        p = AsmPreprocessor(asm_code=AARCH64_ASM, target_function='f')
        assert p.func_asm.endswith('\n')


# =====================================================================
# End-to-end: match Forklift pipeline (asm.py)
# =====================================================================


class TestForkliftParity:
    """Verify preprocessor output exactly matches Forklift pipeline output."""

    @pytest.fixture(autouse=True)
    def _check_clang(self):
        """Skip if cross-compiler not available."""
        try:
            subprocess.run(
                ['clang', '--target=aarch64-linux-gnu', '--version'],
                capture_output=True, check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            pytest.skip("clang aarch64 cross-compiler not available")

    @pytest.mark.parametrize("code,fname", [
        ('int f(int x) { return x + 1; }', 'f'),
        ('void g(int *a, int n) { for(int i=0;i<n;i++) a[i]++; }', 'g'),
        ('int h(int a, int b) { return a > b ? a : b; }', 'h'),
    ])
    def test_matches_forklift_pipeline(self, code, fname):
        """
        Compile C → .s with clang, then compare:
          1) Forklift pipeline (Compiler.factory + get_func_asm)
          2) Our preprocessor (AsmPreprocessor from the .s file)
        """
        from forklift.asm import Compiler

        # Forklift pipeline
        c = Compiler.factory('clang', arch='arm', o='0')
        result = c.get_func_asm(code, fname)
        forklift_lines = [
            l.strip() for l in result.val.func_asm.strip().splitlines() if l.strip()
        ]

        # Generate .s file
        with tempfile.NamedTemporaryFile(suffix='.c', mode='w', delete=False) as cf:
            cf.write(code)
            c_path = cf.name
        s_path = c_path.replace('.c', '.s')
        subprocess.run(
            ['clang', '--target=aarch64-linux-gnu', '-O0', '-S', '-o', s_path, c_path],
            check=True,
        )

        # Preprocessor
        p = AsmPreprocessor(file_path=s_path, target_function=fname)
        prep_lines = [
            l.strip() for l in p.func_asm.strip().splitlines() if l.strip()
        ]

        assert forklift_lines == prep_lines, (
            f"Mismatch for {fname}:\n"
            f"Forklift ({len(forklift_lines)} lines) vs "
            f"Preprocessor ({len(prep_lines)} lines)"
        )


# =====================================================================
# Repr
# =====================================================================


class TestRepr:
    def test_repr_aarch64(self):
        p = AsmPreprocessor(asm_code=AARCH64_ASM, target_function='f')
        r = repr(p)
        assert 'aarch64' in r
        assert "'f'" in r

    def test_repr_x86(self):
        p = AsmPreprocessor(asm_code=X86_ASM, target_function='f')
        r = repr(p)
        assert 'x86' in r
