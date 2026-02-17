"""
Obfuscation dataset generation for Forklift.

Generates paired (obfuscated_asm, clean_ir) training data using Tigress
compiler-level obfuscation and AArch64 cross-compilation.
"""

from neurel_deob.dataset.tigress import TigressObfuscator, TigressTransform
from neurel_deob.dataset.generator import DatasetGenerator, GeneratorConfig

__all__ = [
    "TigressObfuscator",
    "TigressTransform",
    "DatasetGenerator",
    "GeneratorConfig",
]
