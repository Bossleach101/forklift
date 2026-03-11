#!/usr/bin/env python3
"""
Restructure leachl/obfuscated-exebench:
  1. Restore the original 980K training data from a known commit
  2. Add 3 test splits (test_flatten, test_encode_arithmetic, test_combined)
     from locally generated + cleaned parquet files
  3. Update the README with all 4 splits

Usage
-----
    python scripts/restructure_obfu_dataset.py
    python scripts/restructure_obfu_dataset.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ID = "leachl/obfuscated-exebench"
PRE_SPLIT_COMMIT = "077f3002b78ce6f43eba5e91b4e54eaa69ff68c7"

# Local cleaned test set directories (produced by upload_obfu_tests.py --dry-run)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_SETS = {
    "test_flatten": {
        "local_dir": PROJECT_ROOT / "obfu_test_flatten_clean",
        "technique": "Flatten",
    },
    "test_encode_arithmetic": {
        "local_dir": PROJECT_ROOT / "obfu_test_encode_arithmetic_clean",
        "technique": "EncodeArithmetic",
    },
    "test_combined": {
        "local_dir": PROJECT_ROOT / "obfu_test_combined_clean",
        "technique": "Flatten+EncodeArithmetic",
    },
}


def build_readme(test_counts: dict[str, int]) -> str:
    total_test = sum(test_counts.values())
    return f"""---
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
      - split: test_flatten
        path: "data/test_flatten/*.parquet"
      - split: test_encode_arithmetic
        path: "data/test_encode_arithmetic/*.parquet"
      - split: test_combined
        path: "data/test_combined/*.parquet"
---

# Obfuscated ExeBench (AArch64)

A large-scale dataset of **obfuscated AArch64 assembly** functions paired with their **clean LLVM IR**, original **C source code**, and **clean assembly**.  Designed for training neural models that can **deobfuscate** and **lift** obfuscated binary code.

## Dataset Summary

| Property | Value |
|---|---|
| **Training samples** | ~980,000 (from ExeBench `train_synth_compilable`) |
| **Test samples** | {total_test:,} ({test_counts.get('test_flatten', 0):,} Flatten / {test_counts.get('test_encode_arithmetic', 0):,} EncodeArithmetic / {test_counts.get('test_combined', 0):,} Combined) |
| **Test source** | ExeBench `test_synth` (unseen functions) |
| **Architecture** | AArch64 (ARM64) |
| **Obfuscator** | [Tigress 4.0.11](https://tigress.wtf/) |
| **Compiler** | `aarch64-linux-gnu-gcc 15.2.0` (`-S -O0 -std=c11 -w`) |
| **Techniques** | Control-Flow Flattening, Arithmetic Encoding, Combined |
| **Format** | Parquet with Snappy compression |

## Splits

| Split | Rows | Source | Techniques |
|---|---|---|---|
| `train` | ~980,000 | ExeBench `train_synth_compilable` | All three (balanced ⅓ each) |
| `test_flatten` | {test_counts.get('test_flatten', 0):,} | ExeBench `test_synth` | Flatten only |
| `test_encode_arithmetic` | {test_counts.get('test_encode_arithmetic', 0):,} | ExeBench `test_synth` | EncodeArithmetic only |
| `test_combined` | {test_counts.get('test_combined', 0):,} | ExeBench `test_synth` | Flatten+EncodeArithmetic |

## Columns

| Column | Type | Description |
|---|---|---|
| `fname` | `string` | Function name |
| `func_def` | `string` | Original C source code of the function |
| `technique` | `string` | Obfuscation technique applied |
| `clean_asm` | `string` | Clean AArch64 assembly from ExeBench (`angha_gcc_arm_O0`) |
| `obfuscated_asm` | `string` | Obfuscated AArch64 assembly (after Tigress → GCC) |
| `clean_ir` | `string` | Clean LLVM IR from ExeBench (`angha_clang_ir_O0`) |
| `obfuscated_c` | `string` | Tigress-obfuscated C source (target function only, runtime stripped) |
| `tigress_seed` | `int32` | Random seed used for Tigress (for reproducibility) |
| `exebench_split` | `string` | Source ExeBench split name |

## Usage

```python
from datasets import load_dataset

# Training data (all techniques)
train = load_dataset("{REPO_ID}", split="train", streaming=True)

# Test sets (one per technique, from unseen test_synth functions)
test_flat = load_dataset("{REPO_ID}", split="test_flatten")
test_ea   = load_dataset("{REPO_ID}", split="test_encode_arithmetic")
test_comb = load_dataset("{REPO_ID}", split="test_combined")
```

## Obfuscation Techniques

Each function is independently obfuscated with one of three Tigress transformations:

| Technique | Tigress Flag | Description |
|---|---|---|
| `Flatten` | `--Transform=Flatten` | Control-flow flattening — replaces structured control flow with a switch-in-a-loop dispatcher |
| `EncodeArithmetic` | `--Transform=EncodeArithmetic` | Replaces simple arithmetic/boolean expressions with equivalent but complex MBA expressions |
| `Flatten+EncodeArithmetic` | Both transforms | Combined: flattening + arithmetic encoding applied sequentially |

## Tigress Runtime

A representative Tigress runtime (~480 KB, ~7400 lines of C) is stored in `tigress_runtime.c`.
The `obfuscated_c` column contains **only** the target function body (runtime stripped).

## License

MIT — same as the underlying ExeBench dataset.
"""


def main():
    parser = argparse.ArgumentParser(
        description="Restructure obfuscated-exebench: restore train + add test splits"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from huggingface_hub import HfApi
    import pyarrow.parquet as pq

    api = HfApi()

    # ── Step 1: Delete current data/ directory ────────────────────────
    logger.info("Step 1: Deleting current data/ directory on Hub...")
    if not args.dry_run:
        try:
            api.delete_folder(
                repo_id=REPO_ID, path_in_repo="data",
                repo_type="dataset",
                commit_message="Clear data/ for restructuring",
            )
            logger.info("  Deleted data/")
        except Exception as e:
            logger.warning("  Could not delete data/: %s", e)

        # Also delete dataset_info.json if present
        try:
            api.delete_file(
                "dataset_info.json", repo_id=REPO_ID,
                repo_type="dataset",
                commit_message="Remove stale dataset_info.json",
            )
            logger.info("  Deleted dataset_info.json")
        except Exception:
            pass

    # ── Step 2: Restore train shards from pre-split commit ────────────
    logger.info("Step 2: Restoring train shards from commit %s...", PRE_SPLIT_COMMIT)
    old_files = api.list_repo_files(
        REPO_ID, repo_type="dataset", revision=PRE_SPLIT_COMMIT,
    )
    train_shards = sorted([f for f in old_files if f.startswith("data/train/")])
    logger.info("  Found %d train shards to restore", len(train_shards))

    if not args.dry_run:
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, shard_path in enumerate(train_shards):
                # Download from old commit
                local = api.hf_hub_download(
                    REPO_ID, shard_path,
                    repo_type="dataset",
                    revision=PRE_SPLIT_COMMIT,
                    local_dir=tmpdir,
                )
                # Re-upload to current HEAD
                api.upload_file(
                    path_or_fileobj=local,
                    path_in_repo=shard_path,
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    commit_message=f"Restore train shard {i+1}/{len(train_shards)}",
                )
                if (i + 1) % 10 == 0:
                    logger.info("  Restored %d/%d train shards", i + 1, len(train_shards))
        logger.info("  All %d train shards restored", len(train_shards))

    # ── Step 3: Upload test splits ────────────────────────────────────
    logger.info("Step 3: Uploading test splits...")
    test_counts: dict[str, int] = {}

    for split_name, info in TEST_SETS.items():
        local_dir = info["local_dir"]
        parquets = sorted(local_dir.glob("*.parquet"))
        if not parquets:
            logger.warning("  No parquets in %s — skipping %s", local_dir, split_name)
            continue

        # Count rows
        total_rows = sum(pq.read_metadata(p).num_rows for p in parquets)
        test_counts[split_name] = total_rows
        logger.info("  %s: %d rows from %s", split_name, total_rows, local_dir)

        if not args.dry_run:
            for p in parquets:
                api.upload_file(
                    path_or_fileobj=str(p),
                    path_in_repo=f"data/{split_name}/{p.name}",
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    commit_message=f"Add {split_name} test split",
                )
            logger.info("  Uploaded %s", split_name)

    # ── Step 4: Upload tigress_runtime.c ──────────────────────────────
    runtime_path = PROJECT_ROOT / "tigress_runtime.c"
    if runtime_path.exists() and not args.dry_run:
        api.upload_file(
            path_or_fileobj=str(runtime_path),
            path_in_repo="tigress_runtime.c",
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message="Restore tigress_runtime.c",
        )
        logger.info("  Uploaded tigress_runtime.c")

    # ── Step 5: Upload updated README ─────────────────────────────────
    logger.info("Step 4: Uploading README...")
    readme_content = build_readme(test_counts)

    if not args.dry_run:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(readme_content)
            readme_tmp = f.name
        api.upload_file(
            path_or_fileobj=readme_tmp,
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message="Update README with train + 3 test splits",
        )
        logger.info("  README uploaded")
    else:
        logger.info("  [dry-run] README would contain %d chars", len(readme_content))

    # ── Step 6: Clean up separate test repos ──────────────────────────
    logger.info("Step 5: Deleting separate test repos...")
    for repo in [
        "leachl/obfuscated-exebench-test-flatten",
        "leachl/obfuscated-exebench-test-encode-arithmetic",
        "leachl/obfuscated-exebench-test-combined",
    ]:
        if not args.dry_run:
            try:
                api.delete_repo(repo, repo_type="dataset")
                logger.info("  Deleted %s", repo)
            except Exception as e:
                logger.warning("  Could not delete %s: %s", repo, e)
        else:
            logger.info("  [dry-run] Would delete %s", repo)

    logger.info("Done!")


if __name__ == "__main__":
    main()
