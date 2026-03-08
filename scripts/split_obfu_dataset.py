#!/usr/bin/env python3
"""
Split leachl/obfuscated-exebench into train / test with stratified
sampling so every obfuscation technique is equally represented in both
splits.

Usage
-----
    # Default 90/10 split, upload to Hub
    python scripts/split_obfu_dataset.py --upload

    # Custom split ratio, local only
    python scripts/split_obfu_dataset.py --test-ratio 0.05 --output-dir ./obfu_split

    # Dry-run: just print stats, don't write anything
    python scripts/split_obfu_dataset.py --dry-run
"""

from __future__ import annotations

import argparse
import collections
import logging
import os
import random
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

REPO_ID = "leachl/obfuscated-exebench"


def main():
    parser = argparse.ArgumentParser(
        description="Stratified train/test split of obfuscated-exebench"
    )
    parser.add_argument(
        "--repo-id", default=REPO_ID,
        help=f"Source HF dataset (default: {REPO_ID})",
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.10,
        help="Fraction of data for the test set (default: 0.10)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output-dir", default="./obfu_dataset_split",
        help="Local directory for output Parquet files",
    )
    parser.add_argument(
        "--shard-size", type=int, default=10_000,
        help="Rows per Parquet shard (default: 10,000)",
    )
    parser.add_argument(
        "--upload", action="store_true",
        help="Upload the split dataset back to the Hub repo",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only print stats, don't write files",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # ── Load full dataset into memory (grouped by technique) ──────────
    logger.info("Loading dataset from %s ...", args.repo_id)
    ds = load_dataset(args.repo_id, split="train", streaming=True)

    # Group rows by technique
    by_technique: dict[str, list[dict]] = collections.defaultdict(list)
    total = 0
    for row in ds:
        by_technique[row["technique"]].append(row)
        total += 1
        if total % 100_000 == 0:
            logger.info("  loaded %d rows ...", total)

    logger.info("Loaded %d rows total, %d techniques", total, len(by_technique))
    for t, rows in sorted(by_technique.items()):
        logger.info("  %-30s  %d rows", t, len(rows))

    # ── Stratified split ──────────────────────────────────────────────
    train_rows: list[dict] = []
    test_rows: list[dict] = []

    for technique, rows in sorted(by_technique.items()):
        rng.shuffle(rows)
        n_test = max(1, int(len(rows) * args.test_ratio))
        test_rows.extend(rows[:n_test])
        train_rows.extend(rows[n_test:])

    # Shuffle each split so shards are mixed
    rng.shuffle(train_rows)
    rng.shuffle(test_rows)

    # ── Report ────────────────────────────────────────────────────────
    train_tech = collections.Counter(r["technique"] for r in train_rows)
    test_tech = collections.Counter(r["technique"] for r in test_rows)

    logger.info("=" * 60)
    logger.info("SPLIT SUMMARY  (test_ratio=%.2f, seed=%d)", args.test_ratio, args.seed)
    logger.info("  Train: %d rows", len(train_rows))
    for t, c in sorted(train_tech.items()):
        logger.info("    %-30s  %d  (%.1f%%)", t, c, c / len(train_rows) * 100)
    logger.info("  Test:  %d rows", len(test_rows))
    for t, c in sorted(test_tech.items()):
        logger.info("    %-30s  %d  (%.1f%%)", t, c, c / len(test_rows) * 100)
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("Dry run — not writing files.")
        return

    # ── Write Parquet shards ──────────────────────────────────────────
    out_dir = Path(args.output_dir)

    def write_split(rows: list[dict], split_name: str):
        split_dir = out_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        n_shards = (len(rows) + args.shard_size - 1) // args.shard_size
        paths = []
        for i in range(n_shards):
            chunk = rows[i * args.shard_size : (i + 1) * args.shard_size]
            table = pa.Table.from_pylist(chunk)
            shard_path = split_dir / f"{split_name}-{i:05d}-of-{n_shards:05d}.parquet"
            pq.write_table(table, shard_path)
            paths.append(shard_path)
        logger.info("Wrote %d shards to %s", n_shards, split_dir)
        return paths

    train_paths = write_split(train_rows, "train")
    test_paths = write_split(test_rows, "test")

    # ── Upload to Hub ─────────────────────────────────────────────────
    if args.upload:
        from huggingface_hub import HfApi
        api = HfApi()

        logger.info("Uploading to %s ...", args.repo_id)

        # Delete old data/ directory first
        try:
            api.delete_folder(
                repo_id=args.repo_id,
                path_in_repo="data",
                repo_type="dataset",
            )
            logger.info("Deleted old data/ folder on Hub")
        except Exception as e:
            logger.warning("Could not delete old data/ folder: %s", e)

        # Upload train shards
        for p in train_paths:
            api.upload_file(
                path_or_fileobj=str(p),
                path_in_repo=f"data/train/{p.name}",
                repo_id=args.repo_id,
                repo_type="dataset",
            )
        logger.info("Uploaded %d train shards", len(train_paths))

        # Upload test shards
        for p in test_paths:
            api.upload_file(
                path_or_fileobj=str(p),
                path_in_repo=f"data/test/{p.name}",
                repo_id=args.repo_id,
                repo_type="dataset",
            )
        logger.info("Uploaded %d test shards", len(test_paths))

        logger.info("Done! Dataset at https://huggingface.co/datasets/%s", args.repo_id)


if __name__ == "__main__":
    main()
