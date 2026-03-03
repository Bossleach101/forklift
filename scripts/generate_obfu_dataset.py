#!/usr/bin/env python3
"""
Generate an obfuscated AArch64 assembly dataset from ExeBench.

Uses Tigress compiler-level obfuscation (CFF, EncodeArithmetic, etc.)
applied to ExeBench C source functions, cross-compiled to AArch64
with ``aarch64-linux-gnu-gcc``, and paired with clean LLVM IR.

Examples
--------
    # Generate 1000 samples with all transforms (quick test)
    python scripts/generate_obfu_dataset.py \\
        --output-dir ./obfu_dataset \\
        --max-samples 1000

    # Full generation with CFF only
    python scripts/generate_obfu_dataset.py \\
        --output-dir ./obfu_dataset_cff \\
        --transforms Flatten

    # Use a specific ExeBench split
    python scripts/generate_obfu_dataset.py \\
        --output-dir ./obfu_dataset_real \\
        --split train_real_compilable \\
        --max-samples 5000
"""

import argparse
import logging
import sys

from neurel_deob.dataset.generator import DatasetGenerator, GeneratorConfig


def main():
    parser = argparse.ArgumentParser(
        description="Generate obfuscated AArch64 assembly dataset from ExeBench",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Output
    parser.add_argument(
        "--output-dir", "-o",
        default="./obfu_dataset_output",
        help="Output directory for Parquet shards (default: ./obfu_dataset_output)",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=5000,
        help="Number of samples per Parquet shard (default: 5000)",
    )

    # ExeBench source
    parser.add_argument(
        "--split",
        default="train_synth_compilable",
        help="ExeBench split to use (default: train_synth_compilable)",
    )

    # Sampling
    parser.add_argument(
        "--max-samples", "-n",
        type=int,
        default=None,
        help="Maximum number of samples to generate (default: all)",
    )
    parser.add_argument(
        "--max-failures",
        type=int,
        default=50000,
        help="Stop after this many consecutive failures (default: 50000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    # Chunking (for SLURM job arrays)
    parser.add_argument(
        "--skip-rows",
        type=int,
        default=0,
        help="Skip this many ExeBench rows at the start (default: 0)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Process at most this many ExeBench rows (default: all)",
    )
    parser.add_argument(
        "--chunk-id",
        type=int,
        default=None,
        help="Chunk identifier for this job (used in shard filenames)",
    )

    # Parallelism
    parser.add_argument(
        "--num-workers", "-j",
        type=int,
        default=1,
        help="Number of parallel Tigress/GCC workers (default: 1)",
    )
    parser.add_argument(
        "--worker-queue-size",
        type=int,
        default=200,
        help="Max pending futures before blocking (default: 200)",
    )

    # Transforms
    parser.add_argument(
        "--transforms", "-t",
        nargs="+",
        default=["Flatten", "EncodeArithmetic", "Flatten+EncodeArithmetic"],
        help=(
            "Tigress transforms to apply. "
            "Available: Flatten, EncodeArithmetic, Flatten+EncodeArithmetic "
            "(default: all three)"
        ),
    )

    # Filtering
    parser.add_argument(
        "--max-func-len",
        type=int,
        default=8000,
        help="Max C source length in chars (default: 8000)",
    )
    parser.add_argument(
        "--max-asm-lines",
        type=int,
        default=2000,
        help="Max obfuscated assembly lines (default: 2000)",
    )

    # Tigress
    parser.add_argument(
        "--tigress-timeout",
        type=int,
        default=120,
        help="Timeout per Tigress call in seconds (default: 120)",
    )
    parser.add_argument(
        "--tigress-home",
        default=None,
        help="TIGRESS_HOME path (auto-detected if not set)",
    )

    # Logging
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Write logs to file in addition to stderr",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    handlers = [logging.StreamHandler(sys.stderr)]
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file))

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=handlers,
    )

    # Build config
    config = GeneratorConfig(
        output_dir=args.output_dir,
        shard_size=args.shard_size,
        split=args.split,
        max_samples=args.max_samples,
        max_failures=args.max_failures,
        seed=args.seed,
        skip_rows=args.skip_rows,
        max_rows=args.max_rows,
        chunk_id=args.chunk_id,
        num_workers=args.num_workers,
        worker_queue_size=args.worker_queue_size,
        transforms=args.transforms,
        max_func_def_len=args.max_func_len,
        max_asm_lines=args.max_asm_lines,
        tigress_timeout=args.tigress_timeout,
        tigress_home=args.tigress_home,
    )

    # Run generation
    gen = DatasetGenerator(config)
    stats = gen.run()

    # Print summary
    print("\n" + "=" * 60)
    print("Dataset Generation Summary")
    print("=" * 60)
    if args.chunk_id is not None:
        print(f"  Chunk ID:                    {args.chunk_id:>8}")
        print(f"  Skip rows:                   {args.skip_rows:>8}")
        if args.max_rows:
            print(f"  Max rows:                    {args.max_rows:>8}")
    print(f"  Num workers:                 {args.num_workers:>8}")
    print(f"  Total ExeBench rows seen:    {stats['total_rows_seen']:>8}")
    print(f"  Rows skipped (no data):      {stats['rows_skipped_no_data']:>8}")
    print(f"  Rows skipped (too long):     {stats['rows_skipped_too_long']:>8}")
    print(f"  Rows skipped (too short):    {stats['rows_skipped_too_short']:>8}")
    print(f"  Tigress failures:            {stats['tigress_failures']:>8}")
    print(f"  GCC failures:                {stats['gcc_failures']:>8}")
    print(f"  Samples generated:           {stats['samples_generated']:>8}")
    print()
    print("  Per-transform breakdown:")
    for tech, count in stats["samples_per_transform"].items():
        print(f"    {tech:>30s}: {count:>8}")
    print("=" * 60)
    print(f"  Output: {args.output_dir}")
    print()


if __name__ == "__main__":
    main()
