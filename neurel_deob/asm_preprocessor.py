"""
Assembly Preprocessor for Forklift Neural Lifter

Cleans up raw assembly files into the format expected by the Forklift 
model's tokenizer. Supports both x86-64 and AArch64 architectures.

The Forklift training data format:
  x86:    .globl fname / .type fname, @function / # comments stripped
  AArch64: .global fname / .type fname, %function / // comments stripped
  Both:   .cfi_* directives KEPT (part of training data)
"""
import re
from enum import Enum
from typing import Optional


class Arch(Enum):
    X86 = "x86"
    AARCH64 = "aarch64"


# Architecture detection heuristics
_AARCH64_INSTRUCTIONS = re.compile(
    r'^\s+(sub|add|str|ldr|ldrsw|subs|mov|b\.|b\s|bl\s|ret|stp|ldp|adrp|cmp|cbz|cbnz|tbnz|tbz|mrs|msr)\b',
    re.IGNORECASE,
)
_AARCH64_REGISTERS = re.compile(r'\b(x[0-9]|x[12][0-9]|x30|w[0-9]|w[12][0-9]|w30|sp|xzr|wzr)\b')
_X86_REGISTERS = re.compile(r'\b(%[re]?[abcd]x|%[re]?[sd]i|%[re]?[sb]p|%r[0-9]+|%[re]?ip)\b')


class AsmPreprocessor:
    """
    Preprocesses raw assembly into Forklift model input format.

    Usage:
        # Auto-detect architecture
        p = AsmPreprocessor(file_path='func.s', target_function='f')
        print(p.func_asm)
        print(p.arch)

        # Explicit architecture
        p = AsmPreprocessor(asm_code=code, target_function='f', arch=Arch.AARCH64)
    """

    def __init__(
        self,
        asm_code: Optional[str] = None,
        target_function: Optional[str] = None,
        file_path: Optional[str] = None,
        arch: Optional[Arch] = None,
    ):
        if target_function is None:
            print("Warning: target_function is None, defaulting to 'f'")
            target_function = 'f'

        if asm_code is None and file_path is not None:
            with open(file_path, 'r') as f:
                asm_code = f.read()
        elif asm_code is None:
            raise ValueError("Either asm_code or file_path must be provided.")

        # Detect or use provided architecture
        if arch is None:
            self.arch = self._detect_arch(asm_code)
        else:
            self.arch = arch

        self.target_function = target_function
        self.func_asm = self._process(asm_code, target_function)

    # ------------------------------------------------------------------
    # Architecture detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_arch(asm_code: str) -> Arch:
        """Auto-detect architecture from assembly content."""
        aarch64_score = 0
        x86_score = 0

        for line in asm_code.splitlines()[:50]:  # Sample first 50 lines
            if _AARCH64_INSTRUCTIONS.match(line):
                aarch64_score += 1
            if _AARCH64_REGISTERS.search(line):
                aarch64_score += 1
            if _X86_REGISTERS.search(line):
                x86_score += 1
            # AArch64 clang uses // for comments, x86 uses #
            stripped = line.strip()
            if stripped.startswith('//') or stripped.endswith('//'):
                aarch64_score += 1
            if stripped.startswith('#') and not stripped.startswith('#include'):
                x86_score += 1

        if aarch64_score > x86_score:
            return Arch.AARCH64
        return Arch.X86

    # ------------------------------------------------------------------
    # Main processing pipeline
    # ------------------------------------------------------------------

    def _process(self, asm_code: str, target_function: str) -> str:
        """Extract function and clean into Forklift model format."""
        raw_func = self._extract_function(asm_code, target_function)
        cleaned = self._strip_comments(raw_func)
        cleaned = self._remove_unwanted_directives(cleaned)
        cleaned = self._normalise_header(cleaned, target_function)
        cleaned = self._clean_whitespace(cleaned)
        return cleaned

    # ------------------------------------------------------------------
    # Function extraction
    # ------------------------------------------------------------------

    def _extract_function(self, asm_code: str, target_function: str) -> str:
        """
        Extract a single function from a full assembly file.
        Handles both x86 (.globl) and AArch64 (.globl/.global) conventions.

        Extraction ends at .cfi_endproc (matching Forklift's convention in
        _gas_get_func_asm_from_all_asm which uses .cfi_endproc as the
        end-of-function marker). Falls back to .size or .Lfunc_end if
        .cfi_endproc is not found.
        """
        lines = asm_code.split('\n')
        in_function = False
        func_lines = []

        # Match either .globl or .global, with optional trailing comment
        globl_pattern = re.compile(
            rf'^\s*\.glob[al]l?\s+{re.escape(target_function)}\b'
        )

        for line in lines:
            # Start: .globl/.global directive for target function
            if not in_function and globl_pattern.match(line.strip()):
                in_function = True
                func_lines.append(line)
                continue

            if in_function:
                func_lines.append(line)
                # End at .cfi_endproc (Forklift convention)
                if '.cfi_endproc' in line:
                    break

        # Fallback: if no .cfi_endproc found, re-extract using .size or .Lfunc_end
        if in_function and not any('.cfi_endproc' in l for l in func_lines):
            func_lines = []
            in_function = False
            size_pattern = re.compile(
                rf'^\s*\.size\s+{re.escape(target_function)}\s*,'
            )
            func_end_pattern = re.compile(r'^\s*\.Lfunc_end\d+:')

            for line in lines:
                if not in_function and globl_pattern.match(line.strip()):
                    in_function = True
                    func_lines.append(line)
                    continue
                if in_function:
                    if size_pattern.match(line.strip()) or func_end_pattern.match(line.strip()):
                        break
                    func_lines.append(line)

        if not func_lines:
            raise ValueError(
                f"Function '{target_function}' not found in assembly. "
                f"Looked for .globl/.global directive."
            )

        return '\n'.join(func_lines)

    # ------------------------------------------------------------------
    # Comment stripping
    # ------------------------------------------------------------------

    def _strip_comments(self, asm: str) -> str:
        """
        Strip comments based on architecture.
        Matches Forklift's strip_comments() in asm.py.
        """
        comment_sym = '//' if self.arch == Arch.AARCH64 else '#'
        result = []
        for line in asm.splitlines():
            without_comment = line.split(comment_sym)[0]
            # Keep the line if it has any content (preserves indentation)
            if without_comment.strip():
                result.append(without_comment.rstrip())
            # Also keep empty label lines like ".LBB0_4:"
        return '\n'.join(result)

    # ------------------------------------------------------------------
    # Directive removal
    # ------------------------------------------------------------------

    def _remove_unwanted_directives(self, asm: str) -> str:
        """
        Remove directives that are NOT part of the Forklift training data.
        
        IMPORTANT: .cfi_* directives are KEPT — they appear in the training data.
        
        Removed:
          - .file, .ident, .addrsig directives
          - .section directives (except within function body)
          - .p2align directives (alignment hints, not semantic)
          - .Lfunc_end labels and .size directives (post-function metadata)
          - Bare .text directives
        """
        remove_patterns = [
            r'^\s*\.file\s+',
            r'^\s*\.ident\s+',
            r'^\s*\.addrsig',
            r'^\s*\.section\s+',
            r'^\s*\.p2align\s+',
            r'^\s*\.text\s*$',
            r'^\s*\.Lfunc_end\d+:',
            r'^\s*\.size\s+',
        ]
        compiled = [re.compile(p) for p in remove_patterns]

        result = []
        for line in asm.splitlines():
            if any(p.match(line) for p in compiled):
                continue
            result.append(line)
        return '\n'.join(result)

    # ------------------------------------------------------------------
    # Header normalisation
    # ------------------------------------------------------------------

    def _normalise_header(self, asm: str, target_function: str) -> str:
        """
        Normalise the function header to match Forklift training format.

        AArch64 training data uses:
          .global fname
          .type fname, %function

        x86 training data uses:
          .globl fname
          .type fname, @function
        """
        lines = asm.splitlines()
        normalised = []

        for line in lines:
            stripped = line.strip()

            # Normalise .globl/.global
            if re.match(rf'\.glob[al]l?\s+{re.escape(target_function)}', stripped):
                if self.arch == Arch.AARCH64:
                    normalised.append(f'.global {target_function}')
                else:
                    normalised.append(f'.globl {target_function}')
                continue

            # Normalise .type directive
            type_match = re.match(
                rf'\.type\s+{re.escape(target_function)}\s*,\s*[%@]function',
                stripped,
            )
            if type_match:
                if self.arch == Arch.AARCH64:
                    normalised.append(f'.type {target_function}, %function')
                else:
                    normalised.append(f'.type {target_function}, @function')
                continue

            normalised.append(line)

        return '\n'.join(normalised)

    # ------------------------------------------------------------------
    # Whitespace cleanup
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_whitespace(asm: str) -> str:
        """Remove blank lines and trailing whitespace, add final newline."""
        lines = []
        for line in asm.splitlines():
            stripped = line.rstrip()
            if stripped:  # skip blank lines
                lines.append(stripped)
        if lines:
            return '\n'.join(lines) + '\n'
        return ''

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self):
        return (
            f"AsmPreprocessor(arch={self.arch.value}, "
            f"function='{self.target_function}', "
            f"lines={len(self.func_asm.splitlines())})"
        )


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        path = sys.argv[1]
        fname = sys.argv[2] if len(sys.argv) > 2 else 'f'
    else:
        path = '../tests/test_data/f_ground_truth.s'
        fname = 'f'

    p = AsmPreprocessor(file_path=path, target_function=fname)
    print(f"Arch: {p.arch.value}")
    print(f"---")
    print(p.func_asm)
