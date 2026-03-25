"""
LLVM IR validation and functional correctness checking.

Three levels of verification for model-predicted LLVM IR:

    Level 1 – Syntax:      ``llvm-as``  parses the IR bitcode.
    Level 2 – Compilation:  ``clang -c`` compiles the IR to an object file.
    Level 3 – Functional:   Compile IR + ExeBench wrapper, run with IO pairs,
                            compare observed vs expected outputs.

The module is designed to be used both as a library (from evaluation scripts)
and via the CLI (``python -m forklift.ir_checker``).
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CMD_TIMEOUT = 30  # seconds per subprocess call
_RUN_TIMEOUT = 10  # seconds for running a compiled executable

# ExeBench repo checkout that ships synthesizer.h / nlohmann/json.hpp.
# Set via EXEBENCH_CLIB_DIR env-var, or we'll try to auto-detect.
_EXEBENCH_CLIB_DIR: Optional[str] = os.environ.get("EXEBENCH_CLIB_DIR")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_exebench_clib() -> str:
    """Locate the ExeBench ``exebench/`` package directory containing
    ``clib/`` and ``nlohmann/``.  Searched in order:

    1. ``$EXEBENCH_CLIB_DIR`` environment variable.
    2. ``exebench`` importable package.
    3. ``/tmp/exebench_repo/exebench`` (cloned copy).
    """
    global _EXEBENCH_CLIB_DIR
    if _EXEBENCH_CLIB_DIR:
        return _EXEBENCH_CLIB_DIR

    # Try importable package
    try:
        import exebench as _eb
        p = Path(_eb.__file__).parent
        if (p / "clib").is_dir():
            _EXEBENCH_CLIB_DIR = str(p)
            return _EXEBENCH_CLIB_DIR
    except ImportError:
        pass

    # Try /tmp clone
    p = Path("/tmp/exebench_repo/exebench")
    if (p / "clib").is_dir():
        _EXEBENCH_CLIB_DIR = str(p)
        return _EXEBENCH_CLIB_DIR

    raise FileNotFoundError(
        "Cannot locate ExeBench clib directory. "
        "Clone https://github.com/jordiae/exebench.git somewhere and set "
        "EXEBENCH_CLIB_DIR to point to the `exebench/` sub-directory."
    )


def _tool_available(tool: str) -> bool:
    try:
        subprocess.run(
            [tool, "--version"],
            capture_output=True,
            timeout=10,
        )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError):
        return False


@contextlib.contextmanager
def _tmp_file(content: Optional[str] = None, suffix: str = ".tmp", delete: bool = True):
    """Context-manager that yields a path to a named tmp file."""
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="forklift_")
    try:
        if content is not None:
            with os.fdopen(fd, "w") as f:
                f.write(content)
        else:
            os.close(fd)
        yield path
    finally:
        if delete and os.path.exists(path):
            os.unlink(path)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class IRCheckResult:
    """Result of a multi-level IR check."""

    # Level 1
    syntax_valid: bool = False
    syntax_error: Optional[str] = None

    # Level 2
    compiles: bool = False
    compile_error: Optional[str] = None

    # Level 3
    links: bool = False
    link_error: Optional[str] = None
    functional_pass: Optional[bool] = None  # None = not tested
    functional_detail: Optional[str] = None
    io_pass_count: int = 0
    io_total_count: int = 0

    @property
    def functional_pct(self) -> float:
        if self.io_total_count == 0:
            return 0.0
        return 100.0 * self.io_pass_count / self.io_total_count


# ---------------------------------------------------------------------------
# Level 1: Syntax validation via llvm-as
# ---------------------------------------------------------------------------

def _inject_missing_declares(ir_text: str, max_retries: int = 5) -> str:
    """Try to add ``declare`` stubs for undefined function references.

    The model output (and stripped ground-truth) typically lacks ``declare``
    statements for external functions, causing ``llvm-as`` to fail with
    "use of undefined value '@foo'" errors.  This helper iteratively runs
    ``llvm-as``, parses the missing symbol from stderr, and injects a
    generic ``declare void @foo(...)`` stub.  It is intentionally
    conservative — we only fix "undefined value" errors.
    """
    if not _tool_available("llvm-as"):
        return ir_text

    patched = ir_text
    for _ in range(max_retries):
        with _tmp_file(patched, suffix=".ll") as ll_path:
            r = subprocess.run(
                ["llvm-as", ll_path, "-o", "/dev/null"],
                capture_output=True,
                text=True,
                timeout=_CMD_TIMEOUT,
            )
        if r.returncode == 0:
            break  # valid!

        # Parse error: "use of undefined value '@name'"
        # Improved regex to handle quoted/unquoted and more characters (dots, dashes)
        import re as _re
        # Matches: use of undefined value '@foo'  OR  use of undefined value @foo
        m = _re.search(r"use of undefined value\s+(?:['\"])(@[-\w$.]+)(?:['\"])?", r.stderr)
        if not m:
            # Try without quotes
            m = _re.search(r"use of undefined value\s+(@[-\w$.]+)", r.stderr)

        if not m:
            # Log the failure to parse if we can't fix it, to help debugging
            if _ == 0: # Only log on first retry to avoid spam if we loop
                logger.warning(f"llvm-as failed with unparseable error: {r.stderr.strip()}")
            break  # different error, can't fix

        sym = m.group(1)
        
        # Determine strict declaration type based on usage in source line
        # Scan stderr for the source line containing the symbol
        decl = f"declare void {sym}(...)" # default to function
        
        # Look for line in stderr: "llvm-as: file:line:col: error: ..."
        # follow by the source code line.
        # We search for the symbol in the lines provided in stderr.
        lines = r.stderr.splitlines()
        source_line = None
        for i, ln in enumerate(lines):
            if "error:" in ln and "use of undefined value" in ln:
                # The next line(s) usually show the source
                # Or sometimes the caret line. We want the line above caret?
                # Actually llvm-as usually output:
                # llvm-as: <file>:<line>:<col>: error: use of undefined value '@var'
                #   %1 = load i32, i32* @var
                #                       ^
                if i + 1 < len(lines):
                    potential_code = lines[i+1].strip()
                    if sym in potential_code:
                        source_line = potential_code
                        break
        
        if source_line:
            # Heuristics for global variables vs functions
            is_func = False
            # Check for call/invoke
            if _re.search(rf"(?:call|invoke)\s+.+\s+{_re.escape(sym)}\(", source_line):
                is_func = True
            
            if not is_func:
                # Check for load: load <ty>, <ty>* @sym
                m_load = _re.search(r"load\s+([\w%.*\[\]\(\)\s]+?)\s*,\s*([\w%.*\[\]\(\)\s]+?)\s+" + _re.escape(sym), source_line)
                if m_load:
                    ty = m_load.group(1).strip()
                    decl = f"{sym} = external global {ty}"
                else:
                    # Check for store: store <ty> %val, <ty>* @sym
                    m_store = _re.search(r"store\s+([\w%.*\[\]\(\)\s]+?)\s+[%\d@\-\w.]+\s*,\s*([\w%.*\[\]\(\)\s]+?)\s+" + _re.escape(sym), source_line)
                    if m_store:
                        ty = m_store.group(1).strip()
                        decl = f"{sym} = external global {ty}"
                    else:
                        # Check for getelementptr
                        # Heuristic: look for type before symbol in getelementptr line
                        # "getelementptr inbounds (%struct.struct0, %struct.struct0* @foo, ...)"
                        # "getelementptr %struct.struct0, %struct.struct0* @foo, ..."
                        # Key: type, type* @sym
                        m_gep = _re.search(r"getelementptr\s+(?:inbounds\s+)?(?:\([^\)]+\)\s*,)?\s*([\w%.*\[\]\(\)\s]+?)\s*,\s*([\w%.*\[\]\(\)\s]+?)[\s,]+" + _re.escape(sym), source_line)
                        if m_gep:
                            ty = m_gep.group(1).strip()
                            if ty.startswith('(') and ty.endswith(')'):
                                ty = ty[1:-1]
                            decl = f"{sym} = external global {ty}"

        # Inject declaration
        patched = f"{decl}\n" + patched

    return patched


def validate_ir_syntax(
    ir_text: str,
    *,
    auto_declare: bool = False,
) -> Tuple[bool, Optional[str]]:
    """Return ``(valid, error_msg)``.

    Parameters
    ----------
    auto_declare : bool
        If True, try to inject ``declare`` stubs for undefined function
        references before validating.  This is useful when checking
        model predictions that lack ``declare`` lines (e.g. after
        ``strip_ir_noise``).
    """
    if not _tool_available("llvm-as"):
        logger.error("CRITICAL ERROR: 'llvm-as' is not installed or not in PATH! Run 'module load llvm' or install clang first.")
        return False, "llvm-as not available"
    text = _inject_missing_declares(ir_text) if auto_declare else ir_text
    with _tmp_file(text, suffix=".ll") as ll_path:
        r = subprocess.run(
            ["llvm-as", ll_path, "-o", "/dev/null"],
            capture_output=True,
            text=True,
            timeout=_CMD_TIMEOUT,
        )
        if r.returncode == 0:
            return True, None
        return False, r.stderr.strip()


# ---------------------------------------------------------------------------
# Level 2: Compilation via clang -c
# ---------------------------------------------------------------------------

def compile_ir_to_object(
    ir_text: str,
    *,
    auto_declare: bool = False,
) -> Tuple[bool, Optional[str]]:
    """Compile IR to an object file. Return ``(success, error_msg)``.

    Parameters
    ----------
    auto_declare : bool
        If True, inject ``declare`` stubs for undefined function references.
    """
    if not _tool_available("clang"):
        return False, "clang not available"
    text = _inject_missing_declares(ir_text) if auto_declare else ir_text
    with _tmp_file(text, suffix=".ll") as ll_path:
        obj_path = ll_path.replace(".ll", ".o")
        try:
            r = subprocess.run(
                ["clang", "-c", ll_path, "-o", obj_path],
                capture_output=True,
                text=True,
                timeout=_CMD_TIMEOUT,
            )
            if r.returncode == 0:
                return True, None
            return False, r.stderr.strip()
        finally:
            if os.path.exists(obj_path):
                os.unlink(obj_path)


# ---------------------------------------------------------------------------
# Level 3: Functional correctness via ExeBench wrapper
# ---------------------------------------------------------------------------

def _compile_ir_with_wrapper(
    ir_text: str,
    c_deps: str,
    func_c_signature: str,
    cpp_wrapper: str,
    dummy_funcs: str = "",
) -> Tuple[bool, Optional[str], Optional[str]]:
    """Compile predicted IR together with ExeBench wrapper into executable.

    Returns ``(success, error_msg, exe_path)``.  The caller is responsible
    for deleting *exe_path* when done.
    """
    exebench_dir = _find_exebench_clib()

    # Prepare deps: add dummy_funcs + extern declaration
    full_deps = c_deps
    if dummy_funcs:
        full_deps += "\n" + dummy_funcs + "\n"
    full_deps += f"\nextern {func_c_signature};\n"

    with _tmp_file(full_deps, suffix=".c") as deps_path:
        # Rewrite the cpp_wrapper's #include to point at our deps file
        patched_wrapper = re.sub(
            r'extern\s"C"\s\{\s.*?\s\}',
            f'extern "C" \n{{\n#include "{deps_path}"\n}}\n',
            cpp_wrapper,
            flags=re.DOTALL,
        )

        with _tmp_file(patched_wrapper, suffix=".cpp", delete=False) as cpp_path, \
             _tmp_file(ir_text, suffix=".ll", delete=False) as ll_path:
            # First compile IR → object
            obj_path = ll_path + ".o"
            exe_path = ll_path + ".x"
            try:
                # Step 1: IR → object
                r1 = subprocess.run(
                    ["clang", "-c", ll_path, "-o", obj_path],
                    capture_output=True,
                    text=True,
                    timeout=_CMD_TIMEOUT,
                )
                if r1.returncode != 0:
                    return False, f"IR compile failed: {r1.stderr.strip()}", None

                # Step 2: Link object + wrapper → executable
                # We compile the C++ wrapper separately and link together
                cmd = [
                    "g++", "-fpermissive", "-O0",
                    "-o", exe_path,
                    cpp_path, obj_path,
                    f"-I{exebench_dir}",         # nlohmann/json.hpp
                    f"-I{exebench_dir}/clib",     # clib/synthesizer.h — but we skip linking synthesizer.c unless needed
                ]
                # Link synthesizer.c if it exists (some wrappers use facc_malloc etc.)
                synth_c = os.path.join(exebench_dir, "clib", "synthesizer.c")
                fft_c = os.path.join(exebench_dir, "clib", "fft_synth", "lib.c")
                if os.path.exists(synth_c):
                    cmd.append(synth_c)
                if os.path.exists(fft_c):
                    cmd.append(fft_c)

                r2 = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=_CMD_TIMEOUT,
                )
                if r2.returncode != 0:
                    return False, f"Link failed: {r2.stderr.strip()}", None

                return True, None, exe_path

            finally:
                # Clean up intermediates (but NOT exe_path on success)
                for p in [obj_path, cpp_path, ll_path]:
                    if os.path.exists(p):
                        os.unlink(p)


def _run_executable_with_io(
    exe_path: str,
    input_dict: dict,
    timeout: int = _RUN_TIMEOUT,
) -> Tuple[bool, Optional[dict], Optional[str]]:
    """Run the compiled executable with a single IO input.

    Returns ``(success, output_dict, error_msg)``.
    """
    with _tmp_file(json.dumps(input_dict), suffix=".json") as inp_path:
        out_path = inp_path.replace(".json", "-out.json")
        try:
            r = subprocess.run(
                [exe_path, inp_path, out_path],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if r.returncode != 0:
                return False, None, f"Runtime error (rc={r.returncode}): {r.stderr.strip()[:200]}"
            if not os.path.exists(out_path):
                return False, None, "No output file produced"
            with open(out_path) as f:
                output = json.load(f)
            return True, output, None
        except subprocess.TimeoutExpired:
            return False, None, f"Execution timed out ({timeout}s)"
        except json.JSONDecodeError as e:
            return False, None, f"Invalid JSON output: {e}"
        finally:
            if os.path.exists(out_path):
                os.unlink(out_path)


# ---------------------------------------------------------------------------
# IO comparison (from ExeBench)
# ---------------------------------------------------------------------------

def diff_io(observed: Any, expected: Any) -> bool:
    """Recursively compare observed and expected outputs.

    Tolerant of float imprecision (uses ``math.isclose``).
    Ported from ``exebench.diff_io``.
    """
    if type(observed) is not type(expected):
        # Allow int/float cross-comparison
        if isinstance(observed, (int, float)) and isinstance(expected, (int, float)):
            if isinstance(observed, float) or isinstance(expected, float):
                return math.isclose(float(observed), float(expected), rel_tol=1e-3)
            return observed == expected
        return False
    if isinstance(observed, list):
        if len(observed) != len(expected):
            return False
        return all(diff_io(o, e) for o, e in zip(observed, expected))
    if isinstance(observed, dict):
        if set(observed.keys()) != set(expected.keys()):
            return False
        return all(diff_io(observed[k], expected[k]) for k in observed)
    if isinstance(observed, float):
        return math.isclose(observed, expected, rel_tol=1e-3)
    return observed == expected


def exebench_dict_to_dict(exebench_dict: dict) -> dict:
    """Convert ExeBench's ``{var: [...], value: [...]}`` format to a plain dict.

    Ported from ``exebench.exebench_dict_to_dict``.
    """
    from ast import literal_eval

    def _fix(val):
        if isinstance(val, dict):
            return {k: _fix(v) for k, v in val.items()}
        if isinstance(val, list):
            return [_fix(v) for v in val]
        return literal_eval(val)

    keys = exebench_dict["var"]
    values = exebench_dict["value"]
    return _fix({k: v for k, v in zip(keys, values)})


# ---------------------------------------------------------------------------
# High-level check functions
# ---------------------------------------------------------------------------

def check_ir(
    ir_text: str,
    *,
    level: int = 2,
    auto_declare: bool = True,
    c_deps: Optional[str] = None,
    func_c_signature: Optional[str] = None,
    cpp_wrapper: Optional[str] = None,
    dummy_funcs: str = "",
    io_pairs: Optional[dict] = None,
    max_io_tests: int = 5,
) -> IRCheckResult:
    """Run up to *level* checks on an LLVM IR string.

    Parameters
    ----------
    ir_text : str
        The predicted LLVM IR.
    level : int
        1 = syntax only, 2 = + compilation, 3 = + functional.
    auto_declare : bool
        If True (default), automatically inject ``declare`` stubs for
        undefined function references.  This is essential when checking
        model predictions that lack ``declare`` lines.
    c_deps, func_c_signature, cpp_wrapper, dummy_funcs :
        ExeBench row fields needed for Level 3.
    io_pairs : dict
        ExeBench ``synth_io_pairs`` dict (keys: input, output, dummy_funcs).
    max_io_tests : int
        Maximum number of IO test cases to run (default 5).
    """
    result = IRCheckResult()

    # Level 1: syntax
    ok, err = validate_ir_syntax(ir_text, auto_declare=auto_declare)
    result.syntax_valid = ok
    result.syntax_error = err
    if not ok or level < 2:
        return result

    # Level 2: compilation (use patched text if auto_declare)
    ir_for_compile = _inject_missing_declares(ir_text) if auto_declare else ir_text
    ok, err = compile_ir_to_object(ir_for_compile)
    result.compiles = ok
    result.compile_error = err
    if not ok or level < 3:
        return result

    # Level 3: functional
    if not all([c_deps, func_c_signature, cpp_wrapper]):
        result.functional_detail = "Missing ExeBench metadata for functional test"
        return result

    # Resolve dummy_funcs from io_pairs if needed
    if io_pairs and io_pairs.get("dummy_funcs") and not dummy_funcs:
        dummy_funcs = io_pairs["dummy_funcs"][0]

    ok, err, exe_path = _compile_ir_with_wrapper(
        ir_text, c_deps, func_c_signature, cpp_wrapper, dummy_funcs,
    )
    result.links = ok
    result.link_error = err

    if not ok or not exe_path:
        return result

    try:
        if io_pairs is None:
            result.functional_detail = "No IO pairs provided"
            return result

        inputs = io_pairs.get("input", [])
        outputs = io_pairs.get("output", [])
        n_tests = min(len(inputs), len(outputs), max_io_tests)
        result.io_total_count = n_tests

        for i in range(n_tests):
            try:
                inp = exebench_dict_to_dict(inputs[i])
                expected = exebench_dict_to_dict(outputs[i])
            except Exception as e:
                logger.debug("IO conversion error at index %d: %s", i, e)
                continue

            ok, observed, err = _run_executable_with_io(exe_path, inp)
            if not ok:
                logger.debug("Execution failed at IO %d: %s", i, err)
                continue

            if diff_io(observed, expected):
                result.io_pass_count += 1

        result.functional_pass = (
            result.io_pass_count == result.io_total_count
            and result.io_total_count > 0
        )
        result.functional_detail = (
            f"{result.io_pass_count}/{result.io_total_count} IO tests passed"
        )

    finally:
        if exe_path and os.path.exists(exe_path):
            os.unlink(exe_path)

    return result


# ---------------------------------------------------------------------------
# Batch evaluation helpers (for use in evaluation scripts)
# ---------------------------------------------------------------------------

@dataclass
class CompilabilityStats:
    """Aggregate stats from batch checking."""

    total: int = 0
    syntax_valid: int = 0
    compiles: int = 0
    links: int = 0
    functional_pass: int = 0
    functional_tested: int = 0
    io_pass_total: int = 0
    io_tested_total: int = 0
    errors: List[dict] = field(default_factory=list)

    def update(self, result: IRCheckResult, fname: str = "?"):
        self.total += 1
        if result.syntax_valid:
            self.syntax_valid += 1
        if result.compiles:
            self.compiles += 1
        if result.links:
            self.links += 1
        if result.functional_pass is not None:
            self.functional_tested += 1
            if result.functional_pass:
                self.functional_pass += 1
            self.io_pass_total += result.io_pass_count
            self.io_tested_total += result.io_total_count

        # Log first few errors
        if not result.syntax_valid and len(self.errors) < 20:
            self.errors.append({
                "fname": fname,
                "level": "syntax",
                "error": result.syntax_error,
            })
        elif not result.compiles and result.syntax_valid and len(self.errors) < 20:
            self.errors.append({
                "fname": fname,
                "level": "compile",
                "error": result.compile_error,
            })

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "syntax_valid": self.syntax_valid,
            "syntax_valid_pct": round(100 * self.syntax_valid / max(self.total, 1), 2),
            "compiles": self.compiles,
            "compiles_pct": round(100 * self.compiles / max(self.total, 1), 2),
            "links": self.links,
            "links_pct": round(100 * self.links / max(self.total, 1), 2),
            "functional_tested": self.functional_tested,
            "functional_pass": self.functional_pass,
            "functional_pass_pct": round(
                100 * self.functional_pass / max(self.functional_tested, 1), 2
            ),
            "io_pass_total": self.io_pass_total,
            "io_tested_total": self.io_tested_total,
            "io_pass_pct": round(
                100 * self.io_pass_total / max(self.io_tested_total, 1), 2
            ),
        }

    def log_summary(self):
        d = self.to_dict()
        logger.info("=" * 60)
        logger.info("COMPILABILITY / FUNCTIONAL RESULTS")
        logger.info("=" * 60)
        logger.info("  %-30s %d", "Total predictions", d["total"])
        logger.info("  %-30s %d / %d  (%.1f%%)",
                     "Syntax valid (llvm-as)",
                     d["syntax_valid"], d["total"], d["syntax_valid_pct"])
        logger.info("  %-30s %d / %d  (%.1f%%)",
                     "Compiles (clang -c)",
                     d["compiles"], d["total"], d["compiles_pct"])
        if d["links"] or d["functional_tested"]:
            logger.info("  %-30s %d / %d  (%.1f%%)",
                         "Links (with wrapper)",
                         d["links"], d["total"], d["links_pct"])
        if d["functional_tested"]:
            logger.info("  %-30s %d / %d  (%.1f%%)",
                         "Functional pass (all IO)",
                         d["functional_pass"], d["functional_tested"],
                         d["functional_pass_pct"])
            logger.info("  %-30s %d / %d  (%.1f%%)",
                         "IO tests passed",
                         d["io_pass_total"], d["io_tested_total"],
                         d["io_pass_pct"])
        logger.info("=" * 60)
