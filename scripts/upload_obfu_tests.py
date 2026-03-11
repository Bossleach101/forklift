#!/usr/bin/env python3
"""
Clean and upload the 3 obfuscated test sets to HF Hub.

Strips the Tigress runtime from obfuscated_c (extracts only the
target function) and uploads each technique as a separate dataset.

Usage
-----
    python scripts/upload_obfu_tests.py
    python scripts/upload_obfu_tests.py --dry-run   # just clean, don't upload
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Test sets to process ─────────────────────────────────────────────
TEST_SETS = {
    "obfu_test_flatten": {
        "repo_id": "leachl/obfuscated-exebench-test-flatten",
        "technique": "Flatten",
    },
    "obfu_test_encode_arithmetic": {
        "repo_id": "leachl/obfuscated-exebench-test-encode-arithmetic",
        "technique": "EncodeArithmetic",
    },
    "obfu_test_combined": {
        "repo_id": "leachl/obfuscated-exebench-test-combined",
        "technique": "Flatten+EncodeArithmetic",
    },
}


def extract_obfuscated_function(obf_c: str, fname: str) -> str | None:
    """Extract the target function body from full Tigress output."""
    lines = obf_c.splitlines()
    start_idx = None
    end_idx = None

    begin_marker = f"BEGIN FUNCTION-DEF {fname} "
    end_marker = f"END FUNCTION-DEF {fname} "

    for i, line in enumerate(lines):
        if begin_marker in line:
            start_idx = i
        if end_marker in line:
            end_idx = i
            break

    if start_idx is not None and end_idx is not None and end_idx >= start_idx:
        return "\n".join(lines[start_idx : end_idx + 1])
    return None


def clean_and_upload(
    local_dir: str,
    repo_id: str,
    technique: str,
    project_root: Path,
    dry_run: bool = False,
):
    """Clean obfuscated_c in a local parquet dir and upload to Hub."""
    local_path = project_root / local_dir
    parquet_files = sorted(local_path.glob("*.parquet"))

    if not parquet_files:
        logger.warning("No parquet files in %s — skipping", local_path)
        return

    # Read all shards
    tables = [pq.read_table(p) for p in parquet_files]
    table = pa.concat_tables(tables)
    logger.info("[%s] Loaded %d rows from %s", technique, table.num_rows, local_path)

    # Clean obfuscated_c
    rows = table.to_pylist()
    cleaned = 0
    failed = 0
    for row in rows:
        obf_c = row.get("obfuscated_c", "")
        fname = row["fname"]
        if obf_c:
            extracted = extract_obfuscated_function(obf_c, fname)
            if extracted:
                row["obfuscated_c"] = extracted
                cleaned += 1
            else:
                # Keep original if extraction fails
                failed += 1
        else:
            failed += 1

    logger.info("[%s] Cleaned %d, extraction failed %d", technique, cleaned, failed)

    # Write cleaned parquet
    clean_dir = project_root / f"{local_dir}_clean"
    clean_dir.mkdir(parents=True, exist_ok=True)
    clean_table = pa.Table.from_pylist(rows)
    clean_path = clean_dir / "test-00000-of-00001.parquet"
    pq.write_table(clean_table, clean_path)

    # Log size comparison
    raw_size = sum(p.stat().st_size for p in parquet_files)
    clean_size = clean_path.stat().st_size
    logger.info(
        "[%s] Size: %.1f MB → %.1f MB (%.0f%% reduction)",
        technique, raw_size / 1e6, clean_size / 1e6,
        (1 - clean_size / raw_size) * 100,
    )

    if dry_run:
        logger.info("[%s] Dry run — not uploading", technique)
        return

    # Upload to Hub
    from huggingface_hub import HfApi
    api = HfApi()

    # Create repo if it doesn't exist
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)

    # Upload parquet under data/test/
    api.upload_file(
        path_or_fileobj=str(clean_path),
        path_in_repo="data/test/test-00000-of-00001.parquet",
        repo_id=repo_id,
        repo_type="dataset",
    )
    logger.info("[%s] Uploaded to %s", technique, repo_id)

    # Upload a README
    readme = f"""---
license: mit
task_categories:
  - translation
tags:
  - assembly
  - deobfuscation
  - llvm-ir
  - aarch64
  - obfuscation
  - tigress
  - exebench
pretty_name: "Obfuscated ExeBench Test Set — {technique}"
size_categories:
  - 1K<n<10K
configs:
  - config_name: default
    data_files:
      - split: test
        path: "data/test/*.parquet"
---

# Obfuscated ExeBench Test Set — {technique}

Test set of **obfuscated AArch64 assembly** functions paired with **clean LLVM IR**,
generated from ExeBench `test_synth` split using Tigress `{technique}` obfuscation.

| Property | Value |
|---|---|
| **Samples** | {len(rows)} |
| **Source split** | `test_synth` (ExeBench) |
| **Technique** | `{technique}` |
| **Architecture** | AArch64 (ARM64) |

## Columns

| Column | Type | Description |
|---|---|---|
| `fname` | `string` | Function name |
| `func_def` | `string` | Original C source |
| `technique` | `string` | Obfuscation technique (`{technique}`) |
| `clean_asm` | `string` | Clean AArch64 assembly |
| `obfuscated_asm` | `string` | Obfuscated AArch64 assembly |
| `clean_ir` | `string` | Clean LLVM IR |
| `obfuscated_c` | `string` | Obfuscated C source (runtime stripped) |
| `tigress_seed` | `int32` | Tigress random seed |
| `exebench_split` | `string` | Source ExeBench split |

## Usage

```python
from datasets import load_dataset
ds = load_dataset("{repo_id}", split="test")
```
"""
    readme_path = clean_dir / "README.md"
    readme_path.write_text(readme)
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
    logger.info("[%s] README uploaded", technique)


def main():
    parser = argparse.ArgumentParser(
        description="Clean and upload obfuscated test sets to HF Hub"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Clean locally but don't upload")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    for local_dir, info in TEST_SETS.items():
        clean_and_upload(
            local_dir=local_dir,
            repo_id=info["repo_id"],
            technique=info["technique"],
            project_root=project_root,
            dry_run=args.dry_run,
        )

    logger.info("Done!")


if __name__ == "__main__":
    main()
