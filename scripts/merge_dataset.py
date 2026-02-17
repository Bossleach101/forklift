#!/usr/bin/env python3
"""
Merge Parquet shards from SLURM job array chunks into a unified dataset.

After ``generate_dataset_csf.sh`` completes all array tasks, this script:

1. Discovers all ``shard-*.parquet`` files across chunk directories
2. Merges them into consolidated shards of configurable size
3. Writes a unified ``dataset_info.json`` with aggregated statistics
4. Optionally uploads the result to HuggingFace Hub

Examples
--------
    # Merge synth split
    python scripts/merge_dataset.py \\
        --input-dir ./obfu_dataset/train_synth_compilable \\
        --output-dir ./obfu_dataset_merged/train_synth_compilable

    # Merge and upload to HuggingFace Hub
    python scripts/merge_dataset.py \\
        --input-dir ./obfu_dataset/train_synth_compilable \\
        --output-dir ./obfu_dataset_merged/train_synth_compilable \\
        --upload-to leachl/obfuscated-exebench

    # Merge both splits and upload
    python scripts/merge_dataset.py \\
        --input-dir ./obfu_dataset/train_synth_compilable \\
               ./obfu_dataset/train_real_compilable \\
        --output-dir ./obfu_dataset_merged \\
        --upload-to leachl/obfuscated-exebench
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


def discover_shards(input_dirs: list[Path]) -> list[Path]:
    """Find all Parquet shards across chunk directories."""
    shards = []
    for input_dir in input_dirs:
        if not input_dir.exists():
            logger.warning("Input directory does not exist: %s", input_dir)
            continue
        # Look in chunk_* subdirectories and also top-level
        for parquet_file in sorted(input_dir.rglob("shard-*.parquet")):
            shards.append(parquet_file)
    logger.info("Discovered %d Parquet shards", len(shards))
    return shards


def discover_metadata(input_dirs: list[Path]) -> list[dict]:
    """Find and load all dataset_info.json files."""
    metadata_files = []
    for input_dir in input_dirs:
        for meta_file in sorted(input_dir.rglob("dataset_info.json")):
            try:
                data = json.loads(meta_file.read_text())
                metadata_files.append(data)
            except Exception as e:
                logger.warning("Failed to read %s: %s", meta_file, e)
    return metadata_files


def aggregate_stats(metadata_list: list[dict]) -> dict:
    """Aggregate generation stats from all chunks."""
    agg = {
        "total_rows_seen": 0,
        "rows_skipped_no_data": 0,
        "rows_skipped_too_long": 0,
        "rows_skipped_too_short": 0,
        "rows_skipped_chunk": 0,
        "tigress_failures": 0,
        "gcc_failures": 0,
        "samples_generated": 0,
        "samples_per_transform": defaultdict(int),
        "num_chunks": len(metadata_list),
    }

    for meta in metadata_list:
        stats = meta.get("generation_stats", {})
        for key in [
            "total_rows_seen", "rows_skipped_no_data",
            "rows_skipped_too_long", "rows_skipped_too_short",
            "rows_skipped_chunk", "tigress_failures",
            "gcc_failures", "samples_generated",
        ]:
            agg[key] += stats.get(key, 0)

        for tech, count in stats.get("samples_per_transform", {}).items():
            agg["samples_per_transform"][tech] += count

    agg["samples_per_transform"] = dict(agg["samples_per_transform"])
    return agg


def merge_shards(
    shard_paths: list[Path],
    output_dir: Path,
    target_shard_size: int = 50000,
) -> int:
    """
    Merge input shards into consolidated output shards.

    Returns the total number of rows written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    output_shard_idx = 0
    accumulated_tables = []
    accumulated_rows = 0

    for shard_path in shard_paths:
        try:
            table = pq.read_table(shard_path)
        except Exception as e:
            logger.warning("Failed to read %s: %s", shard_path, e)
            continue

        accumulated_tables.append(table)
        accumulated_rows += len(table)

        # Flush when we have enough rows
        while accumulated_rows >= target_shard_size:
            merged = pa.concat_tables(accumulated_tables)
            accumulated_tables = []
            accumulated_rows = 0

            # Split off target_shard_size rows
            if len(merged) > target_shard_size:
                write_table = merged.slice(0, target_shard_size)
                remainder = merged.slice(target_shard_size)
                accumulated_tables.append(remainder)
                accumulated_rows = len(remainder)
            else:
                write_table = merged

            out_path = output_dir / f"shard-{output_shard_idx:05d}.parquet"
            pq.write_table(write_table, out_path, compression="snappy")
            total_rows += len(write_table)
            logger.info(
                "Wrote merged shard %d: %d rows → %s (%.1f MB)",
                output_shard_idx,
                len(write_table),
                out_path,
                out_path.stat().st_size / 1e6,
            )
            output_shard_idx += 1

    # Write any remaining rows
    if accumulated_tables:
        merged = pa.concat_tables(accumulated_tables)
        out_path = output_dir / f"shard-{output_shard_idx:05d}.parquet"
        pq.write_table(merged, out_path, compression="snappy")
        total_rows += len(merged)
        logger.info(
            "Wrote merged shard %d: %d rows → %s (%.1f MB)",
            output_shard_idx,
            len(merged),
            out_path,
            out_path.stat().st_size / 1e6,
        )

    return total_rows


def upload_to_hub(output_dir: Path, repo_id: str):
    """Upload merged dataset to HuggingFace Hub."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.error("huggingface_hub not installed. Run: pip install huggingface_hub")
        return

    api = HfApi()

    # Create repo if it doesn't exist
    try:
        api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    except Exception as e:
        logger.warning("Could not create repo %s: %s", repo_id, e)

    # Upload the entire directory
    logger.info("Uploading %s to %s ...", output_dir, repo_id)
    api.upload_folder(
        folder_path=str(output_dir),
        repo_id=repo_id,
        repo_type="dataset",
    )
    logger.info("Upload complete: https://huggingface.co/datasets/%s", repo_id)


def main():
    parser = argparse.ArgumentParser(
        description="Merge Parquet shards from SLURM job array into unified dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input-dir", "-i",
        nargs="+",
        required=True,
        help="Input directories containing chunk_* subdirectories with shards",
    )
    parser.add_argument(
        "--output-dir", "-o",
        required=True,
        help="Output directory for merged Parquet shards",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=50000,
        help="Target rows per merged shard (default: 50000)",
    )
    parser.add_argument(
        "--upload-to",
        default=None,
        help="HuggingFace Hub repo ID to upload to (e.g. leachl/obfuscated-exebench)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    input_dirs = [Path(d) for d in args.input_dir]
    output_dir = Path(args.output_dir)

    # Discover shards and metadata
    shard_paths = discover_shards(input_dirs)
    if not shard_paths:
        logger.error("No Parquet shards found in %s", input_dirs)
        sys.exit(1)

    metadata_list = discover_metadata(input_dirs)
    agg_stats = aggregate_stats(metadata_list)

    # Merge
    total_rows = merge_shards(shard_paths, output_dir, args.shard_size)

    # Write aggregated metadata
    info = {
        "description": (
            "Obfuscated AArch64 assembly dataset for neural deobfuscation. "
            "Generated from ExeBench using Tigress compiler-level obfuscation "
            "and aarch64-linux-gnu-gcc cross-compilation."
        ),
        "features": {
            "fname": "string",
            "func_def": "string",
            "technique": "string",
            "clean_asm": "string",
            "obfuscated_asm": "string",
            "clean_ir": "string",
            "obfuscated_c": "string",
            "tigress_seed": "int32",
            "exebench_split": "string",
        },
        "total_rows": total_rows,
        "num_shards": len(list(output_dir.glob("shard-*.parquet"))),
        "aggregated_stats": agg_stats,
        "source": "jordiae/exebench",
        "architecture": "AArch64",
    }
    info_path = output_dir / "dataset_info.json"
    info_path.write_text(json.dumps(info, indent=2))

    # Print summary
    print("\n" + "=" * 60)
    print("Merge Summary")
    print("=" * 60)
    print(f"  Input shards:           {len(shard_paths):>8}")
    print(f"  Chunks merged:          {agg_stats['num_chunks']:>8}")
    print(f"  Total rows merged:      {total_rows:>8}")
    print(f"  Output shards:          {info['num_shards']:>8}")
    print(f"  Output directory:       {output_dir}")
    print()
    print("  Aggregated stats:")
    print(f"    Rows seen:            {agg_stats['total_rows_seen']:>8}")
    print(f"    Samples generated:    {agg_stats['samples_generated']:>8}")
    print(f"    Tigress failures:     {agg_stats['tigress_failures']:>8}")
    print(f"    GCC failures:         {agg_stats['gcc_failures']:>8}")
    print()
    print("  Per-transform:")
    for tech, count in agg_stats["samples_per_transform"].items():
        print(f"    {tech:>30s}: {count:>8}")

    # Upload
    if args.upload_to:
        upload_to_hub(output_dir, args.upload_to)


if __name__ == "__main__":
    main()
