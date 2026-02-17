"""
ExeBench → Obfuscated AArch64 dataset generator.

Iterates over ExeBench functions, applies Tigress obfuscation at the
C source level, cross-compiles to AArch64 assembly, and pairs the
obfuscated assembly with clean LLVM IR from ExeBench.

Produces a HuggingFace-compatible dataset in Parquet format.

Supports multiprocessing for parallel Tigress/GCC invocations and
chunk-based splitting (``skip_rows`` / ``max_rows``) for SLURM job
array parallelism.

Usage
-----
    from neurel_deob.dataset.generator import DatasetGenerator, GeneratorConfig

    gen = DatasetGenerator(GeneratorConfig(
        output_dir="./obfu_dataset",
        max_samples=10000,
        num_workers=8,
        transforms=["Flatten", "EncodeArithmetic", "Flatten+EncodeArithmetic"],
    ))
    gen.run()
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed, Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

from datasets import load_dataset

from neurel_deob.dataset.tigress import (
    TigressObfuscator,
    TigressResult,
    TigressTransform,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

@dataclass
class GeneratorConfig:
    """Configuration for dataset generation."""

    # ── Output ────────────────────────────────────────────────────────
    output_dir: str = "./obfu_dataset_output"
    shard_size: int = 5000  # rows per Parquet shard

    # ── ExeBench source ───────────────────────────────────────────────
    hf_dataset: str = "jordiae/exebench"
    revision: str = "clang"
    split: str = "train_synth_compilable"

    # ── Sampling ──────────────────────────────────────────────────────
    max_samples: Optional[int] = None  # None = process all
    max_failures: int = 50_000  # stop after this many consecutive fails
    seed: int = 42

    # ── Chunking (for SLURM job arrays) ───────────────────────────────
    skip_rows: int = 0           # skip this many rows at the start
    max_rows: Optional[int] = None  # process at most this many rows (None = all)
    chunk_id: Optional[int] = None  # identifier for this chunk (used in shard names)

    # ── Parallelism ───────────────────────────────────────────────────
    num_workers: int = 1  # number of parallel Tigress/GCC workers
    worker_queue_size: int = 200  # max pending futures before blocking

    # ── Tigress transforms ────────────────────────────────────────────
    transforms: list[str] = field(
        default_factory=lambda: [
            "Flatten",
            "EncodeArithmetic",
            "Flatten+EncodeArithmetic",
        ]
    )

    # ── Filtering ─────────────────────────────────────────────────────
    max_func_def_len: int = 8000  # skip very long C sources
    min_func_def_len: int = 20   # skip trivially short functions
    max_asm_lines: int = 2000    # skip if obfuscated asm is too long
    max_ir_len: int = 8000       # skip if clean IR is too long

    # ── Tigress settings ──────────────────────────────────────────────
    tigress_timeout: int = 120   # seconds per Tigress call
    tigress_home: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────
# Dataset schema
# ──────────────────────────────────────────────────────────────────────

DATASET_COLUMNS = [
    "fname",            # function name
    "func_def",         # original C source
    "technique",        # obfuscation technique name
    "clean_asm",        # clean AArch64 assembly (from ExeBench)
    "obfuscated_asm",   # obfuscated AArch64 assembly
    "clean_ir",         # clean LLVM IR (from ExeBench)
    "obfuscated_c",     # Tigress-obfuscated C source
    "tigress_seed",     # Tigress random seed used
    "exebench_split",   # ExeBench split name
]


# ──────────────────────────────────────────────────────────────────────
# Worker function (runs in subprocess)
# ──────────────────────────────────────────────────────────────────────

def _worker_obfuscate(
    func_def: str,
    fname: str,
    transform_name: str,
    seed: int,
    synth_deps: str,
    tigress_home: Optional[str],
    tigress_timeout: int,
) -> dict:
    """
    Top-level function for ProcessPoolExecutor workers.

    Creates a TigressObfuscator per call (lightweight — it just stores
    config; the real work is forking Tigress/GCC subprocesses).  This
    avoids pickling issues with the class.

    Returns a dict with result fields or ``{"success": False, ...}``.
    """
    try:
        tigress = TigressObfuscator(
            tigress_home=tigress_home,
            timeout=tigress_timeout,
        )
        transform = TigressTransform(transform_name)
        result = tigress.obfuscate(
            func_def=func_def,
            fname=fname,
            transform=transform,
            seed=seed,
            synth_deps=synth_deps,
        )
        return {
            "success": result.success,
            "obfuscated_c": result.obfuscated_c,
            "obfuscated_asm": result.obfuscated_asm,
            "error": result.error,
            "transform_name": transform_name,
        }
    except Exception as e:
        return {
            "success": False,
            "obfuscated_c": None,
            "obfuscated_asm": None,
            "error": str(e),
            "transform_name": transform_name,
        }


# ──────────────────────────────────────────────────────────────────────
# Generator
# ──────────────────────────────────────────────────────────────────────

class DatasetGenerator:
    """
    Generates an obfuscation dataset from ExeBench.

    For each ExeBench function that compiles successfully, applies each
    configured Tigress transform and pairs the resulting obfuscated
    AArch64 assembly with the original clean LLVM IR.

    Supports multiprocessing: the main thread reads ExeBench rows and
    dispatches (func_def, fname, transform, seed) work items to a
    ``ProcessPoolExecutor``.  Each worker creates its own Tigress
    subprocess, so there are no shared-state issues.

    Supports chunked generation via ``skip_rows`` / ``max_rows`` for
    SLURM job array parallelism across nodes.

    The output is a directory of Parquet shards plus a ``dataset_info.json``
    metadata file, ready for upload to the HuggingFace Hub.
    """

    def __init__(self, config: Optional[GeneratorConfig] = None):
        self.config = config or GeneratorConfig()
        self.rng = random.Random(self.config.seed)

        # Parse transform names into enum values
        self.transforms = self._parse_transforms(self.config.transforms)

        # Create output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # For single-worker mode, create a shared Tigress instance
        # (avoids creating one per call)
        if self.config.num_workers <= 1:
            self.tigress = TigressObfuscator(
                tigress_home=self.config.tigress_home,
                timeout=self.config.tigress_timeout,
            )
        else:
            self.tigress = None  # workers create their own

        # Statistics
        self.stats = {
            "total_rows_seen": 0,
            "rows_skipped_no_data": 0,
            "rows_skipped_too_long": 0,
            "rows_skipped_too_short": 0,
            "rows_skipped_chunk": 0,  # rows skipped by skip_rows
            "tigress_failures": 0,
            "gcc_failures": 0,
            "samples_generated": 0,
            "samples_per_transform": {t.value: 0 for t in self.transforms},
            "chunk_id": self.config.chunk_id,
            "skip_rows": self.config.skip_rows,
            "max_rows": self.config.max_rows,
            "num_workers": self.config.num_workers,
        }

    # ------------------------------------------------------------------
    # Transform parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_transforms(names: list[str]) -> list[TigressTransform]:
        """Parse transform names into TigressTransform enum values."""
        mapping = {t.value: t for t in TigressTransform}
        result = []
        for name in names:
            if name not in mapping:
                raise ValueError(
                    f"Unknown transform '{name}'. Available: {list(mapping.keys())}"
                )
            result.append(mapping[name])
        return result

    # ------------------------------------------------------------------
    # ExeBench streaming (with chunking support)
    # ------------------------------------------------------------------

    def _iter_exebench(self) -> Iterator[dict]:
        """
        Yield ExeBench rows with chunking support.

        If ``skip_rows > 0``, skips that many rows at the start.
        If ``max_rows`` is set, yields at most that many rows.
        """
        ds = load_dataset(
            self.config.hf_dataset,
            split=self.config.split,
            revision=self.config.revision,
            subsets=[self.config.split],
            streaming=True,
        )

        rows_yielded = 0
        for i, row in enumerate(ds):
            # Skip rows for chunking
            if i < self.config.skip_rows:
                continue
            # Stop after max_rows
            if self.config.max_rows is not None and rows_yielded >= self.config.max_rows:
                break
            rows_yielded += 1
            yield row

        self.stats["rows_skipped_chunk"] = min(i + 1, self.config.skip_rows) if self.config.skip_rows > 0 else 0

    # ------------------------------------------------------------------
    # Row extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_asm_code(row: dict, target_name: str) -> Optional[str]:
        """Extract assembly/IR code for a specific target from an ExeBench row."""
        targets = row.get("asm", {}).get("target", [])
        codes = row.get("asm", {}).get("code", [])
        if target_name in targets:
            idx = targets.index(target_name)
            code = codes[idx] if idx < len(codes) else None
            return code if code else None
        return None

    @staticmethod
    def _get_fname(row: dict) -> Optional[str]:
        """Extract function name from an ExeBench row."""
        fname = row.get("fname")
        if fname and isinstance(fname, str) and fname.strip():
            return fname.strip()
        return None

    @staticmethod
    def _get_func_def(row: dict) -> Optional[str]:
        """Extract C function definition from an ExeBench row."""
        func_def = row.get("func_def")
        if func_def and isinstance(func_def, str) and func_def.strip():
            return func_def.strip()
        return None

    # ------------------------------------------------------------------
    # Shard naming
    # ------------------------------------------------------------------

    def _shard_name(self, shard_idx: int) -> str:
        """Generate shard filename, incorporating chunk_id if set."""
        if self.config.chunk_id is not None:
            return f"shard-chunk{self.config.chunk_id:04d}-{shard_idx:05d}.parquet"
        return f"shard-{shard_idx:05d}.parquet"

    # ------------------------------------------------------------------
    # Core generation loop — single worker
    # ------------------------------------------------------------------

    def _run_single(self) -> dict:
        """Single-threaded generation (num_workers=1)."""
        logger.info("Running single-threaded generation")

        samples: list[dict] = []
        shard_idx = 0
        consecutive_fails = 0
        start_time = time.time()

        for row in self._iter_exebench():
            self.stats["total_rows_seen"] += 1

            # Check stopping conditions
            if (
                self.config.max_samples
                and self.stats["samples_generated"] >= self.config.max_samples
            ):
                logger.info("Reached max_samples=%d", self.config.max_samples)
                break

            if consecutive_fails >= self.config.max_failures:
                logger.warning(
                    "Stopping after %d consecutive failures",
                    consecutive_fails,
                )
                break

            # Extract fields
            fname = self._get_fname(row)
            func_def = self._get_func_def(row)
            clean_asm = self._get_asm_code(row, "angha_gcc_arm_O0")
            clean_ir = self._get_asm_code(row, "angha_clang_ir_O0")
            synth_deps = row.get("synth_deps", "") or ""

            if not all([fname, func_def, clean_asm, clean_ir]):
                self.stats["rows_skipped_no_data"] += 1
                consecutive_fails += 1
                continue

            # Length filters
            if len(func_def) > self.config.max_func_def_len:
                self.stats["rows_skipped_too_long"] += 1
                consecutive_fails += 1
                continue
            if len(func_def) < self.config.min_func_def_len:
                self.stats["rows_skipped_too_short"] += 1
                consecutive_fails += 1
                continue
            if len(clean_ir) > self.config.max_ir_len:
                self.stats["rows_skipped_too_long"] += 1
                consecutive_fails += 1
                continue

            # Apply each transform
            row_produced_sample = False
            for transform in self.transforms:
                tigress_seed = self.rng.randint(0, 2**31)

                result = self.tigress.obfuscate(
                    func_def=func_def,
                    fname=fname,
                    transform=transform,
                    seed=tigress_seed,
                    synth_deps=synth_deps,
                )

                if not result.success:
                    if "cross-compilation" in (result.error or ""):
                        self.stats["gcc_failures"] += 1
                    else:
                        self.stats["tigress_failures"] += 1
                    logger.debug(
                        "Failed %s on %s: %s",
                        transform.value,
                        fname,
                        result.error,
                    )
                    continue

                # Validate obfuscated asm length
                asm_lines = result.obfuscated_asm.count("\n") + 1
                if asm_lines > self.config.max_asm_lines:
                    self.stats["rows_skipped_too_long"] += 1
                    continue

                sample = {
                    "fname": fname,
                    "func_def": func_def,
                    "technique": transform.value,
                    "clean_asm": clean_asm,
                    "obfuscated_asm": result.obfuscated_asm,
                    "clean_ir": clean_ir,
                    "obfuscated_c": result.obfuscated_c or "",
                    "tigress_seed": tigress_seed,
                    "exebench_split": self.config.split,
                }
                samples.append(sample)
                self.stats["samples_generated"] += 1
                self.stats["samples_per_transform"][transform.value] += 1
                row_produced_sample = True

                # Check if we should flush a shard
                if len(samples) >= self.config.shard_size:
                    self._write_shard(samples, shard_idx)
                    shard_idx += 1
                    samples = []

                # Log progress
                if self.stats["samples_generated"] % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = self.stats["samples_generated"] / elapsed
                    logger.info(
                        "Generated %d samples (%.1f/s) | seen %d rows | "
                        "tigress_fail=%d gcc_fail=%d",
                        self.stats["samples_generated"],
                        rate,
                        self.stats["total_rows_seen"],
                        self.stats["tigress_failures"],
                        self.stats["gcc_failures"],
                    )

            if row_produced_sample:
                consecutive_fails = 0
            else:
                consecutive_fails += 1

        # Write remaining samples
        if samples:
            self._write_shard(samples, shard_idx)

        return self.stats

    # ------------------------------------------------------------------
    # Core generation loop — multiprocessing
    # ------------------------------------------------------------------

    def _run_parallel(self) -> dict:
        """Parallel generation using ProcessPoolExecutor."""
        num_workers = self.config.num_workers
        logger.info("Running parallel generation with %d workers", num_workers)

        samples: list[dict] = []
        shard_idx = 0
        consecutive_fails = 0
        start_time = time.time()

        # We collect futures in batches to avoid unbounded memory growth.
        # Each future carries metadata (fname, func_def, clean_asm, etc.)
        # that we need to build the final sample.

        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            # Track pending futures with their metadata
            pending: dict[Future, dict] = {}

            def _drain_completed(block: bool = False):
                """Collect completed futures and accumulate samples."""
                nonlocal shard_idx, consecutive_fails, samples

                done_futures = []
                for fut in list(pending):
                    if fut.done():
                        done_futures.append(fut)

                if block and not done_futures and pending:
                    # Wait for at least one to finish
                    done_iter = as_completed(list(pending.keys()))
                    try:
                        fut = next(done_iter)
                        done_futures.append(fut)
                    except StopIteration:
                        pass

                for fut in done_futures:
                    meta = pending.pop(fut)
                    try:
                        worker_result = fut.result(timeout=0)
                    except Exception as e:
                        self.stats["tigress_failures"] += 1
                        logger.debug("Worker exception: %s", e)
                        continue

                    if not worker_result["success"]:
                        err = worker_result.get("error", "")
                        if "cross-compilation" in err:
                            self.stats["gcc_failures"] += 1
                        else:
                            self.stats["tigress_failures"] += 1
                        logger.debug(
                            "Failed %s on %s: %s",
                            worker_result["transform_name"],
                            meta["fname"],
                            err,
                        )
                        continue

                    obfuscated_asm = worker_result["obfuscated_asm"]
                    if not obfuscated_asm:
                        self.stats["tigress_failures"] += 1
                        continue

                    # Validate obfuscated asm length
                    asm_lines = obfuscated_asm.count("\n") + 1
                    if asm_lines > self.config.max_asm_lines:
                        self.stats["rows_skipped_too_long"] += 1
                        continue

                    sample = {
                        "fname": meta["fname"],
                        "func_def": meta["func_def"],
                        "technique": worker_result["transform_name"],
                        "clean_asm": meta["clean_asm"],
                        "obfuscated_asm": obfuscated_asm,
                        "clean_ir": meta["clean_ir"],
                        "obfuscated_c": worker_result["obfuscated_c"] or "",
                        "tigress_seed": meta["tigress_seed"],
                        "exebench_split": self.config.split,
                    }
                    samples.append(sample)
                    self.stats["samples_generated"] += 1
                    transform_name = worker_result["transform_name"]
                    if transform_name in self.stats["samples_per_transform"]:
                        self.stats["samples_per_transform"][transform_name] += 1
                    meta["_produced"] = True

                    # Check if we should flush a shard
                    if len(samples) >= self.config.shard_size:
                        self._write_shard(samples, shard_idx)
                        shard_idx += 1
                        samples = []

                    # Log progress
                    if self.stats["samples_generated"] % 100 == 0:
                        elapsed = time.time() - start_time
                        rate = self.stats["samples_generated"] / elapsed
                        logger.info(
                            "Generated %d samples (%.1f/s) | seen %d rows | "
                            "pending=%d | tigress_fail=%d gcc_fail=%d",
                            self.stats["samples_generated"],
                            rate,
                            self.stats["total_rows_seen"],
                            len(pending),
                            self.stats["tigress_failures"],
                            self.stats["gcc_failures"],
                        )

            for row in self._iter_exebench():
                self.stats["total_rows_seen"] += 1

                # Check max_samples
                if (
                    self.config.max_samples
                    and self.stats["samples_generated"] >= self.config.max_samples
                ):
                    logger.info("Reached max_samples=%d", self.config.max_samples)
                    break

                if consecutive_fails >= self.config.max_failures:
                    logger.warning(
                        "Stopping after %d consecutive failures",
                        consecutive_fails,
                    )
                    break

                # Extract fields
                fname = self._get_fname(row)
                func_def = self._get_func_def(row)
                clean_asm = self._get_asm_code(row, "angha_gcc_arm_O0")
                clean_ir = self._get_asm_code(row, "angha_clang_ir_O0")
                synth_deps = row.get("synth_deps", "") or ""

                if not all([fname, func_def, clean_asm, clean_ir]):
                    self.stats["rows_skipped_no_data"] += 1
                    consecutive_fails += 1
                    continue

                # Length filters
                if len(func_def) > self.config.max_func_def_len:
                    self.stats["rows_skipped_too_long"] += 1
                    consecutive_fails += 1
                    continue
                if len(func_def) < self.config.min_func_def_len:
                    self.stats["rows_skipped_too_short"] += 1
                    consecutive_fails += 1
                    continue
                if len(clean_ir) > self.config.max_ir_len:
                    self.stats["rows_skipped_too_long"] += 1
                    consecutive_fails += 1
                    continue

                # Submit transforms to the pool
                row_submitted = False
                for transform in self.transforms:
                    tigress_seed = self.rng.randint(0, 2**31)

                    meta = {
                        "fname": fname,
                        "func_def": func_def,
                        "clean_asm": clean_asm,
                        "clean_ir": clean_ir,
                        "tigress_seed": tigress_seed,
                        "_produced": False,
                    }

                    fut = pool.submit(
                        _worker_obfuscate,
                        func_def=func_def,
                        fname=fname,
                        transform_name=transform.value,
                        seed=tigress_seed,
                        synth_deps=synth_deps,
                        tigress_home=self.config.tigress_home,
                        tigress_timeout=self.config.tigress_timeout,
                    )
                    pending[fut] = meta
                    row_submitted = True

                # Drain completed futures if queue is getting large
                if len(pending) >= self.config.worker_queue_size:
                    _drain_completed(block=True)

                # Periodically drain without blocking
                if self.stats["total_rows_seen"] % 10 == 0:
                    _drain_completed(block=False)

                # For consecutive_fails tracking in parallel mode,
                # we approximate: reset on submission, increment if
                # the row had no usable data (already handled above).
                if row_submitted:
                    consecutive_fails = 0

            # Drain all remaining futures
            logger.info("Draining %d remaining futures...", len(pending))
            while pending:
                _drain_completed(block=True)

        # Write remaining samples
        if samples:
            self._write_shard(samples, shard_idx)

        return self.stats

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """
        Run the full dataset generation pipeline.

        Returns
        -------
        dict
            Generation statistics.
        """
        logger.info("Starting dataset generation")
        logger.info("Config: %s", self.config)
        logger.info("Transforms: %s", [t.value for t in self.transforms])
        if self.config.skip_rows > 0:
            logger.info("Skipping first %d rows", self.config.skip_rows)
        if self.config.max_rows is not None:
            logger.info("Processing at most %d rows", self.config.max_rows)
        if self.config.chunk_id is not None:
            logger.info("Chunk ID: %d", self.config.chunk_id)

        start_time = time.time()

        if self.config.num_workers > 1:
            self._run_parallel()
        else:
            self._run_single()

        # Write metadata
        self._write_metadata()

        elapsed = time.time() - start_time
        logger.info(
            "Dataset generation complete: %d samples in %.1fs",
            self.stats["samples_generated"],
            elapsed,
        )
        logger.info("Stats: %s", json.dumps(self.stats, indent=2))

        return self.stats

    # ------------------------------------------------------------------
    # Parquet I/O
    # ------------------------------------------------------------------

    def _write_shard(self, samples: list[dict], shard_idx: int):
        """Write a list of samples to a Parquet shard."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        shard_path = self.output_dir / self._shard_name(shard_idx)

        # Build column arrays
        table = pa.table({
            col: pa.array([s[col] for s in samples])
            for col in DATASET_COLUMNS
        })

        pq.write_table(table, shard_path, compression="snappy")
        logger.info(
            "Wrote shard %d: %d samples → %s (%.1f MB)",
            shard_idx,
            len(samples),
            shard_path,
            shard_path.stat().st_size / 1e6,
        )

    def _write_metadata(self):
        """Write dataset_info.json metadata."""
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
            "techniques": [t.value for t in self.transforms],
            "source": "jordiae/exebench",
            "architecture": "AArch64",
            "generation_stats": self.stats,
            "config": {
                "max_func_def_len": self.config.max_func_def_len,
                "min_func_def_len": self.config.min_func_def_len,
                "max_asm_lines": self.config.max_asm_lines,
                "max_ir_len": self.config.max_ir_len,
                "seed": self.config.seed,
                "split": self.config.split,
            },
        }
        info_path = self.output_dir / "dataset_info.json"
        info_path.write_text(json.dumps(info, indent=2))
        logger.info("Wrote dataset info → %s", info_path)
