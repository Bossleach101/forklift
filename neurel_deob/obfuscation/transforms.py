"""
AArch64 assembly-level obfuscation transforms.

Each transform is a callable that takes a list of assembly lines (the
function body, excluding header/footer directives) and returns a
modified list.  Transforms are designed to mimic patterns produced by
real obfuscation tools (OLLVM, Tigress) at the assembly level:

* **DeadCodeInsertion** – inserts semantically dead instructions
  (nops, zero-register ops, dead stores to scratch registers).
* **InstructionSubstitution** – replaces simple instructions with
  equivalent but more complex sequences (e.g. ``add`` → ``sub neg``).
* **OpaquePredicate** – wraps basic blocks in always-true/false
  conditional branches, adding unreachable junk code.
* **JunkComputation** – inserts multi-instruction sequences that
  compute a value but never use it (mimicking encodearith noise).

All transforms use only the "dead" registers x9–x15 (caller-saved,
not used for argument passing or special purposes in AArch64 AAPCS)
for any inserted computations to avoid corrupting program semantics.

Usage::

    from neurel_deob.obfuscation.transforms import DeadCodeInsertion
    transform = DeadCodeInsertion(intensity=0.3, seed=42)
    obfuscated_lines = transform(clean_lines)
"""

from __future__ import annotations

import abc
import random
import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

# Registers safe to clobber in inserted junk (caller-saved temporaries
# that are NOT argument registers and NOT reserved).
# x9–x15 are caller-saved temporaries in AAPCS64.
SCRATCH_X_REGS = [f"x{i}" for i in range(9, 16)]
SCRATCH_W_REGS = [f"w{i}" for i in range(9, 16)]

# Pattern to identify actual instructions (not labels, directives, etc.)
_INSTRUCTION_RE = re.compile(
    r"^\s+[a-z]",  # instruction lines are indented and start with a mnemonic
)

# Label pattern
_LABEL_RE = re.compile(r"^\.?[A-Za-z_]\w*:")

# Directive pattern (lines starting with .)
_DIRECTIVE_RE = re.compile(r"^\s*\.")

# CFI directive (must be preserved in order)
_CFI_RE = re.compile(r"^\s*\.cfi_")

# Branch / control-flow instructions
_BRANCH_RE = re.compile(
    r"^\s+(b|bl|br|blr|ret|b\.\w+|cbz|cbnz|tbz|tbnz)\b", re.IGNORECASE
)


def _is_instruction(line: str) -> bool:
    """True if line is an actual instruction (not label/directive/blank)."""
    return bool(_INSTRUCTION_RE.match(line)) and not _DIRECTIVE_RE.match(line)


def _is_label(line: str) -> bool:
    return bool(_LABEL_RE.match(line.strip()))


def _is_cfi(line: str) -> bool:
    return bool(_CFI_RE.match(line))


def _is_branch(line: str) -> bool:
    return bool(_BRANCH_RE.match(line))


def _indent(instruction: str, width: int = 8) -> str:
    """Add standard indentation to an instruction."""
    return " " * width + instruction


# ──────────────────────────────────────────────────────────────────────
# Abstract base
# ──────────────────────────────────────────────────────────────────────


class ObfuscationTransform(abc.ABC):
    """
    Base class for assembly-level obfuscation transforms.

    Parameters
    ----------
    intensity : float
        Probability (0.0–1.0) that the transform is applied at each
        eligible insertion point.  Higher = more obfuscation.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(self, intensity: float = 0.3, seed: Optional[int] = None):
        if not 0.0 <= intensity <= 1.0:
            raise ValueError(f"intensity must be in [0, 1], got {intensity}")
        self.intensity = intensity
        self.rng = random.Random(seed)

    @abc.abstractmethod
    def apply(self, lines: List[str]) -> List[str]:
        """Transform a list of assembly lines and return the result."""
        ...

    def __call__(self, lines: List[str]) -> List[str]:
        return self.apply(list(lines))  # copy to avoid mutation

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Short identifier for this transform (used in logging)."""
        ...


# ──────────────────────────────────────────────────────────────────────
# Transform: Dead Code Insertion
# ──────────────────────────────────────────────────────────────────────


class DeadCodeInsertion(ObfuscationTransform):
    """
    Insert semantically dead instructions at random points.

    Mimics OLLVM's bogus code insertion.  Inserted instructions include:
    * ``nop``
    * ``mov xN, xN`` (identity move)
    * ``mov xN, #imm`` followed by no use  (dead store to scratch reg)
    * ``add xN, xN, #0``
    * ``eor xN, xN, xzr``
    """

    @property
    def name(self) -> str:
        return "dead_code"

    def _random_dead_instructions(self) -> List[str]:
        """Generate 1–3 dead instructions."""
        reg_x = self.rng.choice(SCRATCH_X_REGS)
        reg_w = self.rng.choice(SCRATCH_W_REGS)
        imm = self.rng.randint(0, 4095)

        pool = [
            f"nop",
            f"mov {reg_x}, {reg_x}",
            f"mov {reg_x}, #{imm}",
            f"add {reg_x}, {reg_x}, #0",
            f"eor {reg_x}, {reg_x}, xzr",
            f"sub {reg_x}, {reg_x}, #0",
            f"and {reg_w}, {reg_w}, {reg_w}",
            f"orr {reg_x}, {reg_x}, xzr",
            f"mov {reg_w}, {reg_w}",
        ]
        count = self.rng.randint(1, 3)
        return [_indent(self.rng.choice(pool)) for _ in range(count)]

    def apply(self, lines: List[str]) -> List[str]:
        result: List[str] = []
        for line in lines:
            result.append(line)
            # Only insert after real instructions, not after branches/cfi/labels
            if (_is_instruction(line)
                    and not _is_branch(line)
                    and not _is_cfi(line)
                    and self.rng.random() < self.intensity):
                result.extend(self._random_dead_instructions())
        return result


# ──────────────────────────────────────────────────────────────────────
# Transform: Instruction Substitution
# ──────────────────────────────────────────────────────────────────────


class InstructionSubstitution(ObfuscationTransform):
    """
    Replace instructions with semantically equivalent but more complex
    sequences.

    Mimics OLLVM's ``-sub`` (instruction substitution) pass.

    Supported substitutions:
    * ``add Xd, Xn, #imm``  →  ``sub Xd, Xn, #-imm``  (when imm fits)
    * ``sub Xd, Xn, #imm``  →  ``add Xd, Xn, #-imm``
    * ``mov Xd, #imm``      →  ``movz Xd, #imm``
    * ``mov Xd, Xn``        →  ``orr Xd, xzr, Xn``
    * ``cmp Xn, #imm``      →  ``subs xzr, Xn, #imm``
    """

    @property
    def name(self) -> str:
        return "insn_sub"

    # Patterns and their replacements
    _PATTERNS = [
        # add Xd, Xn, #imm  →  sub Xd, Xn, #(-imm)  (mod 4096)
        (
            re.compile(r"^(\s+)add\s+(x\d+|w\d+|sp),\s*(x\d+|w\d+|sp),\s*#(\d+)\s*$"),
            lambda m: (
                f"{m.group(1)}mov {m.group(4+0) if False else _sub_scratch(m)}, #{m.group(4)}\n"
                f"{m.group(1)}sub {m.group(2)}, {m.group(3)}, {_sub_scratch(m)}"
                if int(m.group(4)) > 0
                else None
            ),
        ),
        # mov Xd, Xn  →  orr Xd, xzr, Xn
        (
            re.compile(r"^(\s+)mov\s+(x\d+),\s*(x\d+)\s*$"),
            lambda m: f"{m.group(1)}orr {m.group(2)}, xzr, {m.group(3)}",
        ),
        # mov Wd, Wn  →  orr Wd, wzr, Wn
        (
            re.compile(r"^(\s+)mov\s+(w\d+),\s*(w\d+)\s*$"),
            lambda m: f"{m.group(1)}orr {m.group(2)}, wzr, {m.group(3)}",
        ),
        # mov Xd, #imm  →  movz Xd, #imm
        (
            re.compile(r"^(\s+)mov\s+(x\d+|w\d+),\s*(#-?\d+)\s*$"),
            lambda m: f"{m.group(1)}movz {m.group(2)}, {m.group(3)}",
        ),
        # cmp Xn, #imm  →  subs xzr, Xn, #imm  (for x registers)
        (
            re.compile(r"^(\s+)cmp\s+(x\d+),\s*(#\d+)\s*$"),
            lambda m: f"{m.group(1)}subs xzr, {m.group(2)}, {m.group(3)}",
        ),
        # cmp Wn, #imm  →  subs wzr, Wn, #imm
        (
            re.compile(r"^(\s+)cmp\s+(w\d+),\s*(#\d+)\s*$"),
            lambda m: f"{m.group(1)}subs wzr, {m.group(2)}, {m.group(3)}",
        ),
    ]

    def apply(self, lines: List[str]) -> List[str]:
        result: List[str] = []
        for line in lines:
            if _is_instruction(line) and self.rng.random() < self.intensity:
                replaced = self._try_substitute(line)
                if replaced is not None:
                    # A substitution may produce multiple lines
                    result.extend(replaced.split("\n"))
                    continue
            result.append(line)
        return result

    def _try_substitute(self, line: str) -> Optional[str]:
        """Try each pattern; return replacement or None."""
        for pattern, replacer in self._PATTERNS:
            m = pattern.match(line)
            if m:
                try:
                    result = replacer(m)
                    if result is not None:
                        return result
                except (ValueError, IndexError):
                    pass
        return None


def _sub_scratch(m) -> str:
    """Pick a scratch register matching the register class of the match."""
    reg = m.group(2)  # destination register
    if reg.startswith("w"):
        return "w15"
    return "x15"


# ──────────────────────────────────────────────────────────────────────
# Transform: Opaque Predicates
# ──────────────────────────────────────────────────────────────────────


class OpaquePredicate(ObfuscationTransform):
    """
    Insert opaque predicates — conditional branches that always go one
    way, wrapping real code in a seemingly conditional block.

    Mimics OLLVM ``-bcf`` (bogus control flow).

    Strategy: before a non-branch instruction, insert a sequence like::

        mov  x14, #7          ; scratch = known constant
        cmp  x14, #7          ; always true
        b.ne .Ljunk_N         ; never taken
        <original instruction>
        b    .Lreal_N
        .Ljunk_N:
        <junk instruction>
        .Lreal_N:

    The label counter is class-level to avoid collisions.
    """

    _label_counter: int = 0

    @property
    def name(self) -> str:
        return "opaque_pred"

    def apply(self, lines: List[str]) -> List[str]:
        result: List[str] = []
        for line in lines:
            if (_is_instruction(line)
                    and not _is_branch(line)
                    and not _is_cfi(line)
                    and self.rng.random() < self.intensity):
                result.extend(self._wrap_with_opaque(line))
            else:
                result.append(line)
        return result

    def _wrap_with_opaque(self, original_line: str) -> List[str]:
        """Wrap a single instruction in an opaque predicate."""
        OpaquePredicate._label_counter += 1
        n = OpaquePredicate._label_counter

        junk_label = f".Ljunk_{n}"
        real_label = f".Lreal_{n}"

        scratch = self.rng.choice(SCRATCH_X_REGS)
        const_val = self.rng.randint(1, 4095)

        # Pick an always-true or always-false predicate
        if self.rng.random() < 0.5:
            # Always-true: scratch == const → b.ne to junk (never taken)
            predicate_lines = [
                _indent(f"mov {scratch}, #{const_val}"),
                _indent(f"cmp {scratch}, #{const_val}"),
                _indent(f"b.ne {junk_label}"),
            ]
        else:
            # Always-false: scratch != (const+1) when scratch == const
            # → b.eq to junk (never taken since const != const+1)
            other_val = (const_val + 1) & 0xFFF  # keep in 12-bit range
            predicate_lines = [
                _indent(f"mov {scratch}, #{const_val}"),
                _indent(f"cmp {scratch}, #{other_val}"),
                _indent(f"b.eq {junk_label}"),
            ]

        junk_reg = self.rng.choice(SCRATCH_X_REGS)
        junk_imm = self.rng.randint(0, 4095)

        return [
            *predicate_lines,
            original_line,
            _indent(f"b {real_label}"),
            f"{junk_label}:",
            _indent(f"mov {junk_reg}, #{junk_imm}"),
            _indent(f"nop"),
            f"{real_label}:",
        ]


# ──────────────────────────────────────────────────────────────────────
# Transform: Junk Computation
# ──────────────────────────────────────────────────────────────────────


class JunkComputation(ObfuscationTransform):
    """
    Insert multi-instruction junk computations that compute a value
    in a scratch register but never use it for anything meaningful.

    Mimics OLLVM ``-sub`` (encode-arithmetic) noise — complex arithmetic
    sequences that bloat the code.

    Example inserted sequence::

        mov  x12, #42
        add  x12, x12, #17
        eor  x12, x12, #0xFF
        lsl  x12, x12, #2
        ; x12 is never read again
    """

    @property
    def name(self) -> str:
        return "junk_comp"

    def _random_junk_sequence(self) -> List[str]:
        """Generate 2–5 instructions computing junk in a scratch register."""
        reg = self.rng.choice(SCRATCH_X_REGS)
        reg_w = reg.replace("x", "w")
        imm1 = self.rng.randint(1, 4095)

        ops = [
            f"mov {reg}, #{imm1}",
        ]

        n_extra = self.rng.randint(1, 4)
        for _ in range(n_extra):
            imm = self.rng.randint(1, 255)
            op = self.rng.choice([
                f"add {reg}, {reg}, #{imm}",
                f"sub {reg}, {reg}, #{imm}",
                f"eor {reg}, {reg}, #{imm}",
                f"orr {reg}, {reg}, #{imm}",
                f"and {reg}, {reg}, #{imm}",
                f"lsl {reg}, {reg}, #{self.rng.randint(1, 4)}",
                f"lsr {reg}, {reg}, #{self.rng.randint(1, 4)}",
                f"add {reg}, {reg}, {reg}",
                f"mul {reg}, {reg}, {reg}",
            ])
            ops.append(op)

        return [_indent(op) for op in ops]

    def apply(self, lines: List[str]) -> List[str]:
        result: List[str] = []
        for line in lines:
            # Insert junk before some instructions
            if (_is_instruction(line)
                    and not _is_branch(line)
                    and not _is_cfi(line)
                    and self.rng.random() < self.intensity):
                result.extend(self._random_junk_sequence())
            result.append(line)
        return result
