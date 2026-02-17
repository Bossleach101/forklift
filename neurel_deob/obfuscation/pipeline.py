"""
Obfuscation composition pipeline.

Randomly selects and composes multiple obfuscation transforms to
produce varied, realistic-looking obfuscated assembly.  This is applied
to clean AArch64 assembly at data-loading time so that each epoch sees
different obfuscation patterns (data augmentation effect).

Usage::

    from neurel_deob.obfuscation.pipeline import ObfuscationPipeline, ObfuscationConfig

    config = ObfuscationConfig(
        techniques=["dead_code", "insn_sub", "opaque_pred", "junk_comp"],
        intensity_range=(0.1, 0.4),
    )
    pipeline = ObfuscationPipeline(config, seed=42)
    obfuscated_asm = pipeline(clean_asm_string)
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Type

from neurel_deob.obfuscation.transforms import (
    DeadCodeInsertion,
    InstructionSubstitution,
    JunkComputation,
    ObfuscationTransform,
    OpaquePredicate,
)

# Registry mapping short names to transform classes
TRANSFORM_REGISTRY: Dict[str, Type[ObfuscationTransform]] = {
    "dead_code": DeadCodeInsertion,
    "insn_sub": InstructionSubstitution,
    "opaque_pred": OpaquePredicate,
    "junk_comp": JunkComputation,
}


@dataclass
class ObfuscationConfig:
    """
    Configuration for the obfuscation pipeline.

    Parameters
    ----------
    techniques : list of str
        Which transforms to apply.  Valid names:
        ``"dead_code"``, ``"insn_sub"``, ``"opaque_pred"``, ``"junk_comp"``.
        If empty/None, all techniques are used.
    intensity_range : tuple of (float, float)
        For each sample, the per-transform intensity is drawn uniformly
        from this range.  Lower = subtle, higher = heavy obfuscation.
    min_transforms : int
        Minimum number of transforms to apply per sample.
    max_transforms : int
        Maximum number of transforms to apply per sample.
        Set min == max to always apply a fixed number.
    """

    techniques: List[str] = field(
        default_factory=lambda: ["dead_code", "insn_sub", "opaque_pred", "junk_comp"]
    )
    intensity_range: tuple = (0.1, 0.4)
    min_transforms: int = 1
    max_transforms: int = 4

    def __post_init__(self):
        # Validate technique names
        for t in self.techniques:
            if t not in TRANSFORM_REGISTRY:
                raise ValueError(
                    f"Unknown technique '{t}'. "
                    f"Available: {list(TRANSFORM_REGISTRY.keys())}"
                )
        if self.min_transforms < 1:
            raise ValueError("min_transforms must be >= 1")
        if self.max_transforms < self.min_transforms:
            raise ValueError("max_transforms must be >= min_transforms")


# ──────────────────────────────────────────────────────────────────────
# Header / footer splitting
# ──────────────────────────────────────────────────────────────────────

# Patterns that delimit the function prologue / header
_HEADER_END_MARKERS = [
    re.compile(r"^\s*\.cfi_startproc"),
]

# Patterns that delimit the function epilogue / footer
_FOOTER_START_MARKERS = [
    re.compile(r"^\s*\.cfi_endproc"),
]


def _split_function(asm_lines: List[str]) -> tuple:
    """
    Split assembly lines into (header, body, footer).

    Header  = everything up to and including .cfi_startproc
    Body    = the actual instructions between startproc and endproc
    Footer  = .cfi_endproc and anything after

    Transforms are only applied to the body.
    """
    header: List[str] = []
    body: List[str] = []
    footer: List[str] = []

    state = "header"  # header → body → footer

    for line in asm_lines:
        if state == "header":
            header.append(line)
            if any(p.match(line) for p in _HEADER_END_MARKERS):
                state = "body"
        elif state == "body":
            if any(p.match(line) for p in _FOOTER_START_MARKERS):
                footer.append(line)
                state = "footer"
            else:
                body.append(line)
        else:  # footer
            footer.append(line)

    # If no cfi_startproc found, treat everything as body
    if state == "header":
        return [], header, []

    return header, body, footer


# ──────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────


class ObfuscationPipeline:
    """
    Applies a random mixture of obfuscation transforms to AArch64 assembly.

    Each call to :meth:`__call__` randomly selects a subset of transforms
    (between ``min_transforms`` and ``max_transforms``) and applies them
    sequentially.  Intensity is randomised per transform.  This produces
    diverse obfuscation patterns across the training set.

    Parameters
    ----------
    config : ObfuscationConfig
        Pipeline configuration.
    seed : int or None
        Base seed.  Each call uses a derived seed so that results are
        reproducible when the same seed and call order are used.
    """

    def __init__(
        self,
        config: Optional[ObfuscationConfig] = None,
        seed: Optional[int] = None,
    ):
        self.config = config or ObfuscationConfig()
        self.rng = random.Random(seed)
        self._call_count = 0

    def __call__(self, asm: str) -> str:
        """
        Obfuscate a complete AArch64 assembly function string.

        Parameters
        ----------
        asm : str
            Clean AArch64 assembly text (e.g. from ExeBench
            ``angha_gcc_arm_O0``).

        Returns
        -------
        str
            Obfuscated assembly with the same function structure
            (header/footer preserved).
        """
        self._call_count += 1
        call_seed = self.rng.randint(0, 2**31)

        lines = asm.split("\n")
        header, body, footer = _split_function(lines)

        if not body:
            return asm  # nothing to obfuscate

        # Select which transforms to apply
        n_transforms = self.rng.randint(
            self.config.min_transforms,
            min(self.config.max_transforms, len(self.config.techniques)),
        )
        chosen = self.rng.sample(self.config.techniques, n_transforms)

        # Apply transforms sequentially
        current_body = body
        for i, technique_name in enumerate(chosen):
            transform_cls = TRANSFORM_REGISTRY[technique_name]
            intensity = self.rng.uniform(*self.config.intensity_range)
            transform = transform_cls(
                intensity=intensity,
                seed=call_seed + i,
            )
            current_body = transform(current_body)

        result_lines = header + current_body + footer
        return "\n".join(result_lines)

    @property
    def techniques_available(self) -> List[str]:
        """Return names of all registered transforms."""
        return list(TRANSFORM_REGISTRY.keys())

    def get_stats(self) -> dict:
        """Return pipeline usage statistics."""
        return {
            "calls": self._call_count,
            "config": {
                "techniques": self.config.techniques,
                "intensity_range": self.config.intensity_range,
                "min_transforms": self.config.min_transforms,
                "max_transforms": self.config.max_transforms,
            },
        }
