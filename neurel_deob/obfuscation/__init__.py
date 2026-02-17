"""
Assembly-level obfuscation transforms for AArch64.

Provides synthetic obfuscation transforms that can be applied to clean
AArch64 assembly at data-loading time, producing (obfuscated_asm, clean_ir)
training pairs for neural deobfuscation.
"""

from neurel_deob.obfuscation.transforms import (
    ObfuscationTransform,
    DeadCodeInsertion,
    InstructionSubstitution,
    OpaquePredicate,
    JunkComputation,
)
from neurel_deob.obfuscation.pipeline import (
    ObfuscationPipeline,
    ObfuscationConfig,
)

__all__ = [
    "ObfuscationTransform",
    "DeadCodeInsertion",
    "InstructionSubstitution",
    "OpaquePredicate",
    "JunkComputation",
    "ObfuscationPipeline",
    "ObfuscationConfig",
]
