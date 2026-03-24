#!/usr/bin/env python3
"""
Strip the Tigress runtime from obfuscated_c in the obfuscated-exebench dataset.

The raw dataset stores the *entire* Tigress output (~480 KB per sample),
but >99 % of that is a shared Tigress runtime (system headers, helper
functions, etc.).  This script extracts only:

  1. The target obfuscated function body  (between
     ``BEGIN FUNCTION-DEF <fname>`` / ``END FUNCTION-DEF <fname>``).

It writes cleaned Parquet shards to ``--output-dir`` and, optionally,
uploads the result to the Hugging Face Hub.

A representative copy of the Tigress runtime is saved as
``tigress_runtime.c`` alongside this script (or in the output dir).

Usage
-----
    python scripts/clean_obfu_dataset.py \
        --output-dir ./obfu_dataset_clean \
        --upload --repo-id leachl/obfuscated-exebench
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Columns in the output dataset ────────────────────────────────────
OUTPUT_COLUMNS = [
    "fname",
    "func_def",
    "technique",
    "clean_asm",
    "obfuscated_asm",
    "clean_ir",
    "obfuscated_c",        # now contains only the target function
    "tigress_seed",
    "exebench_split",
    # Level 3 constraints
    "c_deps",
    "func_c_signature",
    "cpp_wrapper",
    "dummy_funcs",
    "io_pairs",
]


def extract_obfuscated_function(obf_c: str, fname: str) -> str | None:
    """Extract the target function body from the full Tigress output.

    Looks for the markers::

        /* BEGIN FUNCTION-DEF <fname> ... */
        ...
        /* END FUNCTION-DEF <fname> ... */

    Returns the text *between* the markers (inclusive), or ``None`` if
    the markers are not found.
    """
    lines = obf_c.splitlines()
    start_idx: int | None = None
    end_idx: int | None = None

    begin_marker = f"BEGIN FUNCTION-DEF {fname} "
    end_marker = f"END FUNCTION-DEF {fname} "

    for i, line in enumerate(lines):
        if begin_marker in line:
            start_idx = i
        if end_marker in line:
            end_idx = i
            break          # stop at first match (there should be only one)

    if start_idx is not None and end_idx is not None and end_idx >= start_idx:
        return "\n".join(lines[start_idx : end_idx + 1])

    return None


def clean_dataset(
    *,
    repo_id: str,
    output_dir: Path,
    shard_size: int,
    upload: bool,
    upload_repo_id: str | None,
) -> None:
    """Stream the raw dataset, clean obfuscated_c, write shards."""

    output_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(repo_id, streaming=True, split="train")

    shard_idx = 0
    buf: list[dict] = []
    total = 0
    stripped = 0
    failed_extract = 0
    start = time.time()

    for row in ds:
        fname = row["fname"]
        obf_c = row.get("obfuscated_c", "")

        # Try to extract just the target function
        if obf_c:
            extracted = extract_obfuscated_function(obf_c, fname)
            if extracted is not None:
                row["obfuscated_c"] = extracted
                stripped += 1
            else:
                # Fallback: keep the full obfuscated_c (shouldn't happen)
                failed_extract += 1
                logger.warning(
                    "Could not extract function %s (sample %d), keeping full obfuscated_c",
                    fname, total,
                )

        buf.append({col: row[col] for col in OUTPUT_COLUMNS})
        total += 1

        if len(buf) >= shard_size:
            _write_shard(buf, shard_idx, output_dir)
            shard_idx += 1
            buf = []

        if total % 10_000 == 0:
            elapsed = time.time() - start
            rate = total / elapsed if elapsed > 0 else 0
            logger.info(
                "Processed %d samples (%.0f/s) | stripped=%d failed=%d",
                total, rate, stripped, failed_extract,
            )

    # Final shard
    if buf:
        _write_shard(buf, shard_idx, output_dir)
        shard_idx += 1

    elapsed = time.time() - start
    logger.info(
        "Done: %d samples in %d shards (%.0fs). "
        "Stripped: %d, failed: %d",
        total, shard_idx, elapsed, stripped, failed_extract,
    )

    # Write metadata
    info = {
        "description": (
            "Obfuscated AArch64 assembly dataset for neural deobfuscation. "
            "Generated from ExeBench (jordiae/exebench, clang revision) using "
            "Tigress compiler-level obfuscation and aarch64-linux-gnu-gcc "
            "cross-compilation. The obfuscated_c field contains only the "
            "obfuscated target function (Tigress runtime stripped). "
            "A representative Tigress runtime is stored in tigress_runtime.c."
        ),
        "features": {
            "fname":           {"dtype": "string", "_type": "Value"},
            "func_def":        {"dtype": "string", "_type": "Value"},
            "technique":       {"dtype": "string", "_type": "Value"},
            "clean_asm":       {"dtype": "string", "_type": "Value"},
            "obfuscated_asm":  {"dtype": "string", "_type": "Value"},
            "clean_ir":        {"dtype": "string", "_type": "Value"},
            "obfuscated_c":    {"dtype": "string", "_type": "Value"},
            "tigress_seed":    {"dtype": "int32",  "_type": "Value"},
            "exebench_split":  {"dtype": "string", "_type": "Value"},
            # New columns
            "c_deps":          {"dtype": "string", "_type": "Value"},
            "func_c_signature":{"dtype": "string", "_type": "Value"},
            "cpp_wrapper":     {"dtype": "string", "_type": "Value"},
            "dummy_funcs":     {"dtype": "string", "_type": "Value"},
            "io_pairs":        {"dtype": "string", "_type": "Value"}, # This is a JSON string or struct? ExeBench stores it as struct, load_dataset might return dict.
        },
        "techniques": ["Flatten", "EncodeArithmetic", "Flatten+EncodeArithmetic"],
        "source": "jordiae/exebench",
        "architecture": "AArch64",
        "total_samples": total,
        "stripped_ok": stripped,
        "strip_failed": failed_extract,
    }
    info_path = output_dir / "dataset_info.json"
    info_path.write_text(json.dumps(info, indent=2))
    logger.info("Wrote %s", info_path)

    if upload:
        _upload(output_dir, upload_repo_id or repo_id)


def _write_shard(samples: list[dict], shard_idx: int, output_dir: Path) -> None:
    """Write a list of sample dicts to a Parquet file."""
    path = output_dir / f"shard-{shard_idx:05d}.parquet"
    table = pa.table(
        {col: pa.array([s[col] for s in samples]) for col in OUTPUT_COLUMNS}
    )
    pq.write_table(table, path, compression="snappy")
    size_mb = path.stat().st_size / 1e6
    logger.info("Shard %d: %d rows → %s (%.1f MB)", shard_idx, len(samples), path, size_mb)


def _upload(output_dir: Path, repo_id: str) -> None:
    """Upload cleaned shards + metadata to Hugging Face Hub."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)

    # Upload all parquet files
    parquet_files = sorted(output_dir.glob("shard-*.parquet"))
    logger.info("Uploading %d shards to %s …", len(parquet_files), repo_id)

    api.upload_folder(
        folder_path=str(output_dir),
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=["shard-*.parquet", "dataset_info.json"],
        commit_message="Upload cleaned dataset (Tigress runtime stripped)",
    )

    # Upload tigress_runtime.c if it exists alongside the script
    runtime_path = Path(__file__).parent.parent / "tigress_runtime.c"
    if runtime_path.exists():
        api.upload_file(
            path_or_fileobj=str(runtime_path),
            path_in_repo="tigress_runtime.c",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Add Tigress runtime file",
        )
        logger.info("Uploaded tigress_runtime.c")

    logger.info("Upload complete: https://huggingface.co/datasets/%s", repo_id)


def main():
    parser = argparse.ArgumentParser(
        description="Strip Tigress runtime from obfuscated-exebench dataset"
    )
    parser.add_argument(
        "--repo-id",
        default="leachl/obfuscated-exebench",
        help="Source HF dataset repo (default: leachl/obfuscated-exebench)",
    )
    parser.add_argument(
        "--output-dir",
        default="./obfu_dataset_clean",
        help="Local output directory for cleaned shards",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=10_000,
        help="Rows per Parquet shard (default: 10000)",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload cleaned dataset to HF Hub after processing",
    )
    parser.add_argument(
        "--upload-repo-id",
        default=None,
        help="HF repo to upload to (default: same as --repo-id)",
    )
    args = parser.parse_args()

    clean_dataset(
        repo_id=args.repo_id,
        output_dir=Path(args.output_dir),
        shard_size=args.shard_size,
        upload=args.upload,
        upload_repo_id=args.upload_repo_id,
    )


if __name__ == "__main__":
    main()
