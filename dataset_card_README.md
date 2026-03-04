---
license: mit
task_categories:
  - translation
language:
  - en
tags:
  - assembly
  - deobfuscation
  - llvm-ir
  - aarch64
  - binary-analysis
  - reverse-engineering
  - code
  - obfuscation
  - tigress
  - exebench
pretty_name: Obfuscated ExeBench (AArch64)
size_categories:
  - 100K<n<1M
configs:
  - config_name: default
    data_files:
      - split: train
        path: "data/train/*.parquet"
---

# Obfuscated ExeBench (AArch64)

A large-scale dataset of **obfuscated AArch64 assembly** functions paired with their **clean LLVM IR**, original **C source code**, and **clean assembly**.  Designed for training neural models that can **deobfuscate** and **lift** obfuscated binary code.

## Dataset Summary

| Property | Value |
|---|---|
| **Total samples** | ~980,000 |
| **Source** | [jordiae/exebench](https://huggingface.co/datasets/jordiae/exebench) (`clang` revision, `train_synth_compilable` split) |
| **Architecture** | AArch64 (ARM64) |
| **Obfuscator** | [Tigress 4.0.11](https://tigress.wtf/) |
| **Compiler** | `aarch64-linux-gnu-gcc 15.2.0` (`-S -O0 -std=c11 -w`) |
| **Techniques** | Control-Flow Flattening, Arithmetic Encoding, Combined |
| **Format** | Parquet with Snappy compression |

## Columns

| Column | Type | Description |
|---|---|---|
| `fname` | `string` | Function name |
| `func_def` | `string` | Original C source code of the function |
| `technique` | `string` | Obfuscation technique applied (see below) |
| `clean_asm` | `string` | Clean AArch64 assembly from ExeBench (`angha_gcc_arm_O0`) |
| `obfuscated_asm` | `string` | Obfuscated AArch64 assembly (after Tigress → GCC) |
| `clean_ir` | `string` | Clean LLVM IR from ExeBench (`angha_clang_ir_O0`) |
| `obfuscated_c` | `string` | Tigress-obfuscated C source (target function only, runtime stripped) |
| `tigress_seed` | `int32` | Random seed used for Tigress (for reproducibility) |
| `exebench_split` | `string` | Source ExeBench split name |

## Obfuscation Techniques

Each function is independently obfuscated with one of three Tigress transformation pipelines:

| Technique | Tigress Flags | Description |
|---|---|---|
| `Flatten` | `--Transform=Flatten` | **Control-Flow Flattening (CFF)** — replaces structured control flow with a dispatcher loop and switch statement, obscuring the original program structure. |
| `EncodeArithmetic` | `--Transform=EncodeArithmetic` | **Arithmetic Encoding** — replaces simple arithmetic and boolean operations with equivalent but complex expressions using Mixed Boolean-Arithmetic (MBA) identities. |
| `Flatten+EncodeArithmetic` | `--Transform=Flatten --Transform=EncodeArithmetic` | **Combined** — applies both CFF and arithmetic encoding for maximum obfuscation. |

The techniques are applied in roughly equal proportions (~⅓ each).

## Intended Uses

### Primary: Neural Deobfuscation / Lifting
Train sequence-to-sequence models that translate **obfuscated assembly → clean LLVM IR**:
- Input: `obfuscated_asm`
- Target: `clean_ir`

### Secondary: Obfuscation Detection / Classification
Build classifiers that identify which obfuscation technique was applied:
- Input: `obfuscated_asm` or `obfuscated_c`
- Target: `technique`

### Other Uses
- **Source-level deobfuscation**: `obfuscated_c` → `func_def`
- **Assembly comparison studies**: `clean_asm` vs `obfuscated_asm`
- **Benchmarking binary analysis tools** against known obfuscation
- **Understanding Tigress transformations** at the assembly level

## Quick Start

```python
from datasets import load_dataset

# Stream (recommended for large datasets)
ds = load_dataset("leachl/obfuscated-exebench", streaming=True, split="train")

for sample in ds:
    print(sample["fname"])
    print(sample["technique"])
    print(sample["obfuscated_asm"][:200])
    print(sample["clean_ir"][:200])
    break
```

```python
# Load fully into memory (requires ~1 GB RAM)
ds = load_dataset("leachl/obfuscated-exebench", split="train")
print(f"Total samples: {len(ds)}")
```

## Generation Pipeline

```
ExeBench (C source + synth_deps)
         │
         ▼
┌─────────────────┐
│   Tigress 4.0.11│  ← Obfuscation at C source level
│   --Env=aarch64  │
│   --Transform=…  │
└────────┬────────┘
         │ obfuscated C
         ▼
┌─────────────────────────┐
│ aarch64-linux-gnu-gcc   │  ← Cross-compilation to AArch64 asm
│ -S -O0 -std=c11 -w     │
└────────┬────────────────┘
         │ obfuscated assembly
         ▼
┌─────────────────────────────────┐
│  Paired with ExeBench originals │
│  clean_asm, clean_ir, func_def  │
└─────────────────────────────────┘
```

Each ExeBench function is processed through up to 3 obfuscation pipelines independently. Functions that fail Tigress transformation or GCC cross-compilation are silently skipped (~50% Tigress failure rate is expected).

## Tigress Runtime

The Tigress obfuscator prepends a large runtime (~7,400 lines / ~480 KB) to every output file. This runtime is **stripped** from the `obfuscated_c` column to save space — only the obfuscated target function body is stored.

A representative copy of the full Tigress runtime is available in [`tigress_runtime.c`](tigress_runtime.c). If you need to recompile the obfuscated C, prepend this runtime to the `obfuscated_c` field.

## Dataset Statistics

- **Source split**: `train_synth_compilable` from ExeBench
- **Technique distribution**: ~⅓ Flatten, ~⅓ EncodeArithmetic, ~⅓ Flatten+EncodeArithmetic
- **Obfuscated assembly is always different** from clean assembly (verified)
- **94%** of obfuscated functions produce longer assembly than the original
- **Clean IR** is valid LLVM IR (100% verified on sampled subset)

## Limitations

- Only AArch64 architecture (no x86, RISC-V, etc.)
- Only `train_synth_compilable` split from ExeBench (not `train_real_compilable`)
- Only 3 Tigress transformations (Tigress supports many more: Virtualize, JIT, etc.)
- Compiled at `-O0` only (no optimisation levels)
- The `obfuscated_c` field contains only the target function — helper functions and the Tigress runtime are stripped
- ~50% of ExeBench functions fail Tigress processing (complex dependencies, unsupported constructs)

## Citation

If you use this dataset, please cite:

```bibtex
@misc{obfuscated-exebench-2026,
  title={Obfuscated ExeBench: A Large-Scale Dataset for Neural Deobfuscation},
  author={Leach, L.},
  year={2026},
  howpublished={\url{https://huggingface.co/datasets/leachl/obfuscated-exebench}},
}
```

### Related Work

- **ExeBench**: Armengol-Estapé, J., et al. (2024). [ExeBench: an ML-scale dataset of executable C functions](https://huggingface.co/datasets/jordiae/exebench).
- **Tigress**: Collberg, C. (2024). [The Tigress C Diversifier/Obfuscator](https://tigress.wtf/).

## License

This dataset is released under the [MIT License](LICENSE). The underlying ExeBench data is subject to its own license terms.
