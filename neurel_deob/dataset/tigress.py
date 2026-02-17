"""
Tigress obfuscator wrapper for AArch64 dataset generation.

Wraps the Tigress C obfuscator to apply source-level obfuscation
transforms (CFF, EncodeArithmetic, etc.) to individual C functions,
then cross-compiles the result to AArch64 assembly.

Requirements
------------
- Tigress 4.x installed with ``TIGRESS_HOME`` set or at ``/usr/local/bin/tigresspkg/*/``
- ``aarch64-linux-gnu-gcc`` cross-compiler available on ``$PATH``
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Tigress transform definitions
# ──────────────────────────────────────────────────────────────────────

class TigressTransform(str, Enum):
    """Tigress transforms that work reliably without complex init.

    We deliberately exclude transforms that require InitEntropy /
    InitOpaque / extra system headers (e.g. AddOpaque, EncodeLiterals,
    Virtualize) because ExeBench functions are isolated and don't come
    with the necessary runtime scaffolding.
    """
    FLATTEN = "Flatten"           # Control-flow flattening (CFF)
    ENCODE_ARITH = "EncodeArithmetic"  # MBA-based arithmetic encoding
    FLATTEN_ENCODE_ARITH = "Flatten+EncodeArithmetic"  # Combined


# Default set of standard includes that most ExeBench functions need
_DEFAULT_INCLUDES = [
    "#include <stdlib.h>",
    "#include <string.h>",
    "#include <stdio.h>",
    "#include <math.h>",
    "#include <stdint.h>",
    "#include <stdbool.h>",
    "#include <limits.h>",
    "#include <float.h>",
    "#include <ctype.h>",
    "#include <assert.h>",
]


@dataclass
class TigressResult:
    """Result of a single Tigress obfuscation + cross-compilation."""
    success: bool
    obfuscated_c: Optional[str] = None
    obfuscated_asm: Optional[str] = None
    error: Optional[str] = None
    transform: Optional[TigressTransform] = None


# ──────────────────────────────────────────────────────────────────────
# synth_deps / source cleaning helpers
# ──────────────────────────────────────────────────────────────────────

# Lines from the ExeBench synth_deps preamble that conflict with our
# standard includes.  We strip these because we already provide the
# proper headers (<stdbool.h>, <stddef.h>, etc.) in prepare_source().
_SYNTH_DEPS_STRIP_PATTERNS = [
    # Conflicts with C23 bool keyword in GCC 15+ and <stdbool.h>
    re.compile(r'^\s*typedef\s+int\s+bool\s*;'),
    re.compile(r'^\s*#\s*define\s+false\s+0\s*$'),
    re.compile(r'^\s*#\s*define\s+true\s+1\s*$'),
]


def _clean_synth_deps(synth_deps: str) -> str:
    """Remove ExeBench preamble lines that conflict with standard headers."""
    out_lines = []
    for line in synth_deps.split("\n"):
        if any(pat.match(line) for pat in _SYNTH_DEPS_STRIP_PATTERNS):
            continue
        out_lines.append(line)
    return "\n".join(out_lines)


def _clean_for_gcc(source: str) -> str:
    """
    Prepare Tigress-obfuscated C source for cross-compilation with
    modern GCC (15+, C23 default).

    Tigress preprocesses all ``#include`` directives internally and
    outputs a self-contained C file with everything expanded inline.
    The ExeBench synth_deps preamble (``typedef int bool;``,
    ``#define false/true``, etc.) is preserved verbatim by Tigress.

    With ``-std=c11`` passed to GCC, all of these are valid C11
    constructs, so **no source-level cleaning is needed**.

    We do NOT add any ``#include`` directives because the Tigress
    output already has everything expanded.  Adding includes would
    cause conflicting type definitions between host and cross-compilation
    target headers.
    """
    return source


class TigressObfuscator:
    """
    Wraps Tigress + aarch64-linux-gnu-gcc for function-level obfuscation.

    Parameters
    ----------
    tigress_home : str or None
        Path to Tigress installation.  Auto-detected if not provided.
    gcc_path : str
        Path to the AArch64 cross-compiler.
    gcc_flags : list[str]
        Additional flags for gcc compilation.
    timeout : int
        Maximum seconds for each Tigress / gcc invocation.
    seed : int or None
        Tigress ``--Seed`` value.  ``None`` means random per invocation.
    """

    def __init__(
        self,
        tigress_home: Optional[str] = None,
        gcc_path: str = "aarch64-linux-gnu-gcc",
        gcc_flags: Optional[list[str]] = None,
        timeout: int = 60,
        seed: Optional[int] = None,
    ):
        self.tigress_home = tigress_home or self._detect_tigress_home()
        self.gcc_path = gcc_path
        self.gcc_flags = gcc_flags or ["-S", "-O0", "-std=c11", "-w"]
        self.timeout = timeout
        self.seed = seed

        # Validate toolchain
        self._validate()

    # ------------------------------------------------------------------
    # Toolchain validation
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_tigress_home() -> str:
        """Auto-detect TIGRESS_HOME from environment or known paths."""
        # Check env var first
        if "TIGRESS_HOME" in os.environ:
            return os.environ["TIGRESS_HOME"]

        # Check common install locations
        tigress_bin = shutil.which("tigress")
        if tigress_bin:
            resolved = os.path.realpath(tigress_bin)
            return str(Path(resolved).parent)

        # Check /usr/local/bin/tigresspkg/
        pkg_dir = Path("/usr/local/bin/tigresspkg")
        if pkg_dir.exists():
            versions = sorted(pkg_dir.iterdir(), reverse=True)
            if versions:
                return str(versions[0])

        raise RuntimeError(
            "Cannot detect Tigress installation. Set TIGRESS_HOME or "
            "ensure 'tigress' is on PATH."
        )

    def _validate(self):
        """Ensure Tigress and gcc are available."""
        tigress_bin = Path(self.tigress_home) / "tigress"
        if not tigress_bin.exists():
            # Maybe it's the parent dir
            if not shutil.which("tigress"):
                raise RuntimeError(
                    f"Tigress binary not found at {tigress_bin} and not on PATH."
                )

        if not shutil.which(self.gcc_path):
            raise RuntimeError(
                f"AArch64 cross-compiler '{self.gcc_path}' not found on PATH."
            )

    @staticmethod
    def is_available() -> bool:
        """Check if both Tigress and aarch64-linux-gnu-gcc are available."""
        try:
            TigressObfuscator()
            return True
        except RuntimeError:
            return False

    # ------------------------------------------------------------------
    # C source preparation
    # ------------------------------------------------------------------

    @staticmethod
    def prepare_source(
        func_def: str,
        fname: str,
        synth_deps: Optional[str] = None,
    ) -> str:
        """
        Prepare a C function for Tigress consumption.

        Tigress requires a ``main()`` function in the source file.
        ExeBench ``func_def`` typically includes type definitions and
        the function definition.  We add common includes, synthetic
        dependencies (type stubs), and a dummy ``main()``.

        Parameters
        ----------
        func_def : str
            The C function definition from ExeBench (may include structs,
            typedefs, etc.).
        fname : str
            The function name to obfuscate.
        synth_deps : str or None
            Synthetic dependencies from ExeBench (type definitions,
            forward declarations, stub functions).  Prepended before
            the function definition to help Tigress parse the code.

        Returns
        -------
        str
            Complete C source file suitable for Tigress.
        """
        # Check if main() already exists
        has_main = re.search(r'\bint\s+main\s*\(', func_def) or \
                   re.search(r'\bvoid\s+main\s*\(', func_def)

        lines = list(_DEFAULT_INCLUDES)
        lines.append("")

        # Add Tigress-incompatible type defines
        lines.append("#ifndef _Float64")
        lines.append("  #define _Float64 double")
        lines.append("#endif")
        lines.append("#ifndef _Float128")
        lines.append("  #define _Float128 double")
        lines.append("#endif")
        lines.append("#ifndef _Float32x")
        lines.append("  #define _Float32x double")
        lines.append("#endif")
        lines.append("#ifndef _Float64x")
        lines.append("  #define _Float64x double")
        lines.append("#endif")
        lines.append("")

        # Strip __attribute__((used)) which confuses Tigress
        clean_def = re.sub(
            r'__attribute__\s*\(\s*\(\s*used\s*\)\s*\)\s*', '', func_def
        )
        # Strip 'static inline' → just the function (Tigress sometimes
        # chokes on inline functions)
        clean_def = re.sub(r'\bstatic\s+inline\b', '', clean_def)

        # Add synthetic dependencies if available
        if synth_deps and isinstance(synth_deps, str) and synth_deps.strip():
            clean_deps = _clean_synth_deps(synth_deps)
            if clean_deps.strip():
                lines.append("/* --- ExeBench synth_deps --- */")
                lines.append(clean_deps)
                lines.append("/* --- end synth_deps --- */")
                lines.append("")

        # Add the function definition
        lines.append(clean_def)

        # Add dummy main if needed
        if not has_main:
            lines.append("")
            lines.append("int main(void) { return 0; }")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Core obfuscation pipeline
    # ------------------------------------------------------------------

    def obfuscate(
        self,
        func_def: str,
        fname: str,
        transform: TigressTransform = TigressTransform.FLATTEN,
        seed: Optional[int] = None,
        synth_deps: Optional[str] = None,
    ) -> TigressResult:
        """
        Obfuscate a single C function and cross-compile to AArch64.

        Parameters
        ----------
        func_def : str
            C function source code (from ExeBench ``func_def``).
        fname : str
            Name of the function to obfuscate.
        transform : TigressTransform
            Which Tigress transform(s) to apply.
        seed : int or None
            Random seed for Tigress.  Uses instance seed if None.
        synth_deps : str or None
            Synthetic dependencies from ExeBench for type resolution.

        Returns
        -------
        TigressResult
            Result with obfuscated C and AArch64 assembly, or error info.
        """
        effective_seed = seed if seed is not None else self.seed

        with tempfile.TemporaryDirectory(prefix="forklift_obfu_") as tmpdir:
            tmpdir = Path(tmpdir)
            input_c = tmpdir / "input.c"
            output_c = tmpdir / "obfuscated.c"
            output_s = tmpdir / "obfuscated.s"

            # Prepare source
            try:
                prepared = self.prepare_source(func_def, fname, synth_deps)
                input_c.write_text(prepared)
            except Exception as e:
                return TigressResult(
                    success=False,
                    error=f"Source preparation failed: {e}",
                    transform=transform,
                )

            # Run Tigress
            tigress_ok = self._run_tigress(
                input_c, output_c, fname, transform, effective_seed
            )
            if not tigress_ok:
                return TigressResult(
                    success=False,
                    error="Tigress obfuscation failed",
                    transform=transform,
                )

            # Read obfuscated C
            try:
                obfu_c = output_c.read_text()
            except Exception as e:
                return TigressResult(
                    success=False,
                    error=f"Failed to read Tigress output: {e}",
                    transform=transform,
                )

            # Clean obfuscated C for GCC: Tigress copies the full input
            # verbatim (including synth_deps preamble) so problematic
            # typedefs can reappear in the output.
            gcc_input_c = tmpdir / "gcc_input.c"
            gcc_input_c.write_text(_clean_for_gcc(obfu_c))

            # Cross-compile to AArch64
            gcc_ok = self._run_gcc(gcc_input_c, output_s)
            if not gcc_ok:
                return TigressResult(
                    success=False,
                    obfuscated_c=obfu_c,
                    error="AArch64 cross-compilation failed",
                    transform=transform,
                )

            # Read assembly output
            try:
                obfu_asm = output_s.read_text()
            except Exception as e:
                return TigressResult(
                    success=False,
                    obfuscated_c=obfu_c,
                    error=f"Failed to read assembly output: {e}",
                    transform=transform,
                )

            # Extract just the target function from the assembly
            extracted_asm = self._extract_function(obfu_asm, fname)
            if not extracted_asm:
                return TigressResult(
                    success=False,
                    obfuscated_c=obfu_c,
                    error=f"Could not extract function '{fname}' from assembly",
                    transform=transform,
                )

            return TigressResult(
                success=True,
                obfuscated_c=obfu_c,
                obfuscated_asm=extracted_asm,
                transform=transform,
            )

    # ------------------------------------------------------------------
    # Tigress invocation
    # ------------------------------------------------------------------

    def _run_tigress(
        self,
        input_c: Path,
        output_c: Path,
        fname: str,
        transform: TigressTransform,
        seed: Optional[int],
    ) -> bool:
        """Run Tigress on *input_c* and write result to *output_c*."""
        env = os.environ.copy()
        env["TIGRESS_HOME"] = self.tigress_home

        cmd = ["tigress", "--Environment=x86_64:Linux:Gcc:4.6"]

        if seed is not None:
            cmd.append(f"--Seed={seed}")

        # Build transform-specific arguments
        if transform == TigressTransform.FLATTEN:
            cmd += [
                "--Transform=Flatten",
                f"--Functions={fname}",
            ]
        elif transform == TigressTransform.ENCODE_ARITH:
            cmd += [
                "--Transform=EncodeArithmetic",
                f"--Functions={fname}",
            ]
        elif transform == TigressTransform.FLATTEN_ENCODE_ARITH:
            cmd += [
                "--Transform=Flatten",
                f"--Functions={fname}",
                "--Transform=EncodeArithmetic",
                f"--Functions={fname}",
            ]

        cmd += [f"--out={output_c}", str(input_c)]

        logger.debug("Tigress cmd: %s", " ".join(cmd))

        try:
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            logger.warning("Tigress timed out after %ds", self.timeout)
            return False
        except FileNotFoundError:
            logger.error("Tigress binary not found")
            return False

        if result.returncode != 0:
            logger.debug("Tigress failed (rc=%d): %s", result.returncode, result.stderr)
            return False

        # Check the output file is non-trivial (Tigress sometimes exits 0
        # but produces an empty/stub file on internal errors)
        if not output_c.exists():
            return False
        content = output_c.read_text()
        # If the obfuscated C doesn't contain the function name, Tigress
        # silently failed
        if fname not in content:
            logger.debug("Tigress output does not contain function '%s'", fname)
            return False

        return True

    # ------------------------------------------------------------------
    # Cross-compilation
    # ------------------------------------------------------------------

    def _run_gcc(self, input_c: Path, output_s: Path) -> bool:
        """Cross-compile C source to AArch64 assembly."""
        cmd = [self.gcc_path] + self.gcc_flags + ["-o", str(output_s), str(input_c)]

        logger.debug("GCC cmd: %s", " ".join(cmd))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            logger.warning("GCC timed out after %ds", self.timeout)
            return False
        except FileNotFoundError:
            logger.error("GCC binary '%s' not found", self.gcc_path)
            return False

        if result.returncode != 0:
            logger.debug("GCC failed (rc=%d): %s", result.returncode, result.stderr)
            return False

        # Sanity check: output should be more than just headers
        if output_s.exists() and output_s.stat().st_size < 100:
            logger.debug("GCC output suspiciously small (%d bytes)", output_s.stat().st_size)
            return False

        return True

    # ------------------------------------------------------------------
    # Assembly extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_function(asm_text: str, fname: str) -> Optional[str]:
        """
        Extract a single function from a full assembly file.

        Looks for the function's ``.global`` / ``.type`` header and
        captures everything up to ``.size fname, .-fname``.
        This is the same format ExeBench stores AArch64 assembly in.
        """
        lines = asm_text.split("\n")
        in_function = False
        func_lines: list[str] = []
        found_type = False

        for line in lines:
            stripped = line.strip()

            # Look for function entry markers
            if not in_function:
                # Match .type fname, %function  or  .type fname, @function
                if re.match(
                    rf'\.type\s+{re.escape(fname)}\s*,\s*[%@]function',
                    stripped,
                ):
                    found_type = True
                    # Include the .global directive if we haven't yet
                    if not func_lines:
                        func_lines.append(f"\t.global\t{fname}")
                    func_lines.append(line)
                    continue

                # Match .global fname or .globl fname
                if re.match(
                    rf'\.(global|globl)\s+{re.escape(fname)}\s*$',
                    stripped,
                ):
                    func_lines.append(line)
                    continue

                # Match .align directive right before function
                if found_type and re.match(r'\.align\s+\d+', stripped):
                    func_lines.append(line)
                    continue

                # Match function label
                if found_type and stripped == f"{fname}:":
                    in_function = True
                    func_lines.append(line)
                    continue
            else:
                # Check for function end marker
                if re.match(
                    rf'\.size\s+{re.escape(fname)}\s*,',
                    stripped,
                ):
                    func_lines.append(line)
                    break

                func_lines.append(line)

        if not func_lines or not in_function:
            return None

        return "\n".join(func_lines)
