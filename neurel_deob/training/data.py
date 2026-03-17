"""
ExeBench streaming data loader for Forklift fine-tuning.

Streams ExeBench rows from the HuggingFace Hub and yields tokenized
(source, target) pairs suitable for seq2seq training.

The heavy lifting of pair routing and tokenization is delegated to the
existing ``forklift.par_data.DP`` class, so the token format is
guaranteed to be identical to the original Forklift training regime.

Usage
-----
    from neurel_deob.training.data import ExeBenchDataset, collate_fn

    ds = ExeBenchDataset(
        tokenizer=tok,
        split='train_synth_compilable',
        pair='arm_ir-ir',
        max_source_len=1024,
        max_target_len=1024,
    )
    loader = DataLoader(ds, batch_size=8, collate_fn=collate_fn)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple

import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset

from forklift.par_data import DP
from forklift.utils import normalize_structs
from neurel_deob.obfuscation.pipeline import ObfuscationConfig, ObfuscationPipeline

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# IR target cleaning
# ──────────────────────────────────────────────────────────────────────

def strip_ir_noise(ir_text: str) -> str:
    """Remove noise from LLVM IR targets that causes repetitive generation.

    Option 2: Aggressive Function Extraction.
    Instead of carefully filtering out lines, this function targets the 
    core mathematical/logic functions and strips away all ExeBench globals,
    dummy strings (`@.str`), and struct definitions unless strictly necessary.

    1. We extract only `define ... { ... }` blocks.
    2. We throw away ExeBench's internal dummy/wrapper functions if possible.
    """
    if not ir_text:
        return ir_text

    kept: list[str] = []
    
    # We want to extract ONLY the function bodies and their immediate types.
    # ExeBench often defines globals like `@.str = external hidden unnamed_addr...`
    # We will ignore those to prevent the model from memorizing them.
    
    in_function = False
    for line in ir_text.split("\n"):
        stripped = line.strip()
        
        # Start of a function
        if stripped.startswith("define "):
            in_function = True
            kept.append(line)
            continue
            
        # End of a function
        if in_function and stripped == "}":
            kept.append(line)
            in_function = False
            continue
            
        # Ensure we are only grabbing code inside a function body OR vital type definitions
        if in_function:
            kept.append(line)
        elif stripped.startswith("%struct.") or stripped.startswith("%union."):
            # We keep struct/union types as they are required for syntax validation 
            # if they are passed as arguments to the function.
            kept.append(line)

    # Remove trailing blank lines
    while kept and not kept[-1].strip():
        kept.pop()

    return "\n".join(kept)


@dataclass
class DataConfig:
    """Configuration for ExeBench data loading."""

    hf_dataset: str = "jordiae/exebench"
    revision: str = "clang"
    split: str = "train_synth_compilable"
    pair: str = "arm_ir-ir"
    asm_key: str = "angha"
    max_source_len: int = 1024
    max_target_len: int = 1024
    normalize_ir_structs: bool = True
    streaming: bool = True
    strip_ir_declares: bool = True  # Remove declare/attributes/metadata from targets


class ExeBenchDataset(IterableDataset):
    """
    Streaming PyTorch IterableDataset over ExeBench arm_ir-ir pairs.

    Each yielded sample is a dict with:
        ``input_ids``  – tokenized source (encoder input)
        ``labels``     – tokenized target (decoder labels)

    Rows where the required ASM target is missing are silently skipped.
    Rows exceeding *max_source_len* / *max_target_len* are also skipped
    to keep GPU memory predictable during training.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        config: Optional[DataConfig] = None,
        *,
        split: Optional[str] = None,
        pair: Optional[str] = None,
    ):
        super().__init__()
        self.config = config or DataConfig()
        # Allow overrides via kwargs
        if split is not None:
            self.config.split = split
        if pair is not None:
            self.config.pair = pair

        self.dp = DP(tokenizer=tokenizer)
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.get_vocab()["<pad>"]

        # Pre-resolve the asm keys so we can fast-skip rows missing them.
        _dp = DP()
        src_key, tgt_key, _, _ = _dp.get_par_data(
            row=None, pair=self.config.pair, asm_key=self.config.asm_key, fPIC=False
        )
        self.source_key = src_key  # e.g. angha_gcc_arm_O0
        self.target_key = tgt_key  # e.g. angha_clang_ir_O0
        logger.info(
            "Resolved pair %s → source=%s  target=%s",
            self.config.pair,
            self.source_key,
            self.target_key,
        )

    # ------------------------------------------------------------------
    # Streaming iterator
    # ------------------------------------------------------------------

    def _load_hf_stream(self):
        """Return a HuggingFace streaming iterator for the configured split."""
        # ExeBench uses a custom loader script that requires `subsets=[split]`.
        # datasets==2.15 doesn't support `trust_remote_code`, so we omit it.
        return load_dataset(
            self.config.hf_dataset,
            split=self.config.split,
            revision=self.config.revision,
            subsets=[self.config.split],
            streaming=self.config.streaming,
        )

    def __iter__(self) -> Iterator[dict]:
        hf_stream = self._load_hf_stream()
        skipped = 0
        yielded = 0

        for row in hf_stream:
            result = self._process_row(row)
            if result is None:
                skipped += 1
                if skipped % 5000 == 0:
                    logger.debug(
                        "Skipped %d rows so far (yielded %d)", skipped, yielded
                    )
                continue
            yielded += 1
            if yielded % 10000 == 0:
                logger.info("Yielded %d samples (skipped %d)", yielded, skipped)
            yield result

        logger.info(
            "Finished streaming %s: yielded=%d  skipped=%d",
            self.config.split,
            yielded,
            skipped,
        )

    # ------------------------------------------------------------------
    # Row processing
    # ------------------------------------------------------------------

    def _process_row(self, row: dict) -> Optional[dict]:
        """
        Convert a single ExeBench row into tokenized tensors.

        Returns ``None`` if the row should be skipped (missing data or
        sequence too long).
        """
        # Fast-path check: are the required asm targets present?
        targets = row.get("asm", {}).get("target", [])
        if self.source_key not in targets or self.target_key not in targets:
            return None

        # Also skip rows where the code is empty / None
        codes = row["asm"]["code"]
        src_idx = targets.index(self.source_key)
        tgt_idx = targets.index(self.target_key)
        source_text = codes[src_idx]
        target_text = codes[tgt_idx]
        if not source_text or not target_text:
            return None

        # ── Clean IR target before tokenization ──────────────────────
        if self.config.strip_ir_declares and "ir" in self.config.pair:
            target_text = strip_ir_noise(target_text)
            if not target_text.strip():
                return None
            # We need to inject the cleaned target back into the row so
            # that get_par_data picks it up.  Make a shallow copy to
            # avoid mutating the original streaming row.
            codes = list(codes)  # copy
            codes[tgt_idx] = target_text
            row = dict(row)
            row["asm"] = dict(row["asm"])
            row["asm"]["code"] = codes

        # Delegate to DP.get_par_data for proper tokenization
        try:
            _src, _tgt, tok_src, tok_tgt = self.dp.get_par_data(
                row,
                pair=self.config.pair,
                asm_key=self.config.asm_key,
                fPIC=False,
                tokenize_ids=True,
                do_normalize_ir_structs=self.config.normalize_ir_structs,
            )
        except (ValueError, KeyError, IndexError) as exc:
            logger.debug("Skipping row due to error: %s", exc)
            return None

        if tok_src is None or tok_tgt is None:
            return None

        # Length filter
        if len(tok_src) > self.config.max_source_len:
            return None
        if len(tok_tgt) > self.config.max_target_len:
            return None

        return {
            "input_ids": torch.tensor(tok_src, dtype=torch.long),
            "labels": torch.tensor(tok_tgt, dtype=torch.long),
        }


# ------------------------------------------------------------------
# Collation — pads variable-length sequences into a batch
# ------------------------------------------------------------------


class ObfuscatedExeBenchDataset(ExeBenchDataset):
    """
    Streaming dataset that applies assembly-level obfuscation to the
    *source* (AArch64 assembly) while keeping the *target* (clean LLVM IR)
    unchanged.

    This produces ``(obfuscated_arm_asm, clean_ir)`` training pairs for
    neural deobfuscation training.  The obfuscation is applied on-the-fly
    so every epoch sees different obfuscation patterns (data augmentation).

    Parameters
    ----------
    tokenizer : Tokenizer
        The Forklift tokenizer.
    config : DataConfig
        Data loading configuration (inherits all ExeBenchDataset settings).
    obfu_config : ObfuscationConfig
        Which obfuscation techniques to use and at what intensity.
    seed : int
        Base seed for reproducible obfuscation.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        config: Optional[DataConfig] = None,
        obfu_config: Optional[ObfuscationConfig] = None,
        seed: int = 42,
        *,
        split: Optional[str] = None,
        pair: Optional[str] = None,
    ):
        super().__init__(tokenizer, config, split=split, pair=pair)
        self.obfu_config = obfu_config or ObfuscationConfig()
        self.obfu_pipeline = ObfuscationPipeline(self.obfu_config, seed=seed)

    def _process_row(self, row: dict) -> Optional[dict]:
        """
        Process a row with obfuscation applied to the source assembly.

        1. Extract clean source asm and target IR
        2. Apply obfuscation transforms to the source asm
        3. Inject the obfuscated source back into the row
        4. Delegate to parent for tokenization
        """
        # Fast-path check: are the required asm targets present?
        targets = row.get("asm", {}).get("target", [])
        if self.source_key not in targets or self.target_key not in targets:
            return None

        codes = row["asm"]["code"]
        src_idx = targets.index(self.source_key)
        tgt_idx = targets.index(self.target_key)
        source_text = codes[src_idx]
        target_text = codes[tgt_idx]
        if not source_text or not target_text:
            return None

        # ── Apply obfuscation to source assembly ────────────────────
        try:
            obfuscated_source = self.obfu_pipeline(source_text)
        except Exception as exc:
            logger.debug("Obfuscation failed, skipping row: %s", exc)
            return None

        # ── Inject obfuscated source back into the row ──────────────
        codes = list(codes)  # copy
        codes[src_idx] = obfuscated_source
        row = dict(row)
        row["asm"] = dict(row["asm"])
        row["asm"]["code"] = codes

        # ── Clean IR target before tokenization ─────────────────────
        if self.config.strip_ir_declares and "ir" in self.config.pair:
            target_text = strip_ir_noise(target_text)
            if not target_text.strip():
                return None
            codes = list(row["asm"]["code"])
            codes[tgt_idx] = target_text
            row["asm"]["code"] = codes

        # ── Delegate to DP for tokenization ─────────────────────────
        try:
            _src, _tgt, tok_src, tok_tgt = self.dp.get_par_data(
                row,
                pair=self.config.pair,
                asm_key=self.config.asm_key,
                fPIC=False,
                tokenize_ids=True,
                do_normalize_ir_structs=self.config.normalize_ir_structs,
            )
        except (ValueError, KeyError, IndexError) as exc:
            logger.debug("Skipping row due to error: %s", exc)
            return None

        if tok_src is None or tok_tgt is None:
            return None

        # Length filter (obfuscated source is longer than clean)
        if len(tok_src) > self.config.max_source_len:
            return None
        if len(tok_tgt) > self.config.max_target_len:
            return None

        return {
            "input_ids": torch.tensor(tok_src, dtype=torch.long),
            "labels": torch.tensor(tok_tgt, dtype=torch.long),
        }


# ------------------------------------------------------------------
# Tigress-obfuscated dataset (leachl/obfuscated-exebench)
# ------------------------------------------------------------------


class TigressObfuDataset(IterableDataset):
    """
    Streaming dataset for pre-generated Tigress-obfuscated samples.

    Reads from ``leachl/obfuscated-exebench`` (flat schema) which has
    columns like ``obfuscated_asm``, ``clean_ir``, ``technique``, etc.

    Each yielded sample is a dict with:
        ``input_ids``  – tokenized obfuscated ARM assembly (encoder)
        ``labels``     – tokenized clean LLVM IR (decoder labels)

    Uses ``dp.tokenize()`` directly with the raw text, matching the
    tokenization format used by ``evaluate_obfu.py``.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        config: Optional[DataConfig] = None,
        *,
        split: Optional[str] = None,
        pair: Optional[str] = None,
    ):
        super().__init__()
        self.config = config or DataConfig()
        if split is not None:
            self.config.split = split
        if pair is not None:
            self.config.pair = pair

        self.dp = DP(tokenizer=tokenizer)
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.get_vocab()["<pad>"]

        logger.info(
            "TigressObfuDataset: dataset=%s  split=%s  pair=%s",
            self.config.hf_dataset,
            self.config.split,
            self.config.pair,
        )

    def _load_hf_stream(self):
        return load_dataset(
            self.config.hf_dataset,
            split=self.config.split,
            streaming=self.config.streaming,
        )

    def __iter__(self) -> Iterator[dict]:
        hf_stream = self._load_hf_stream()
        skipped = 0
        yielded = 0

        for row in hf_stream:
            result = self._process_row(row)
            if result is None:
                skipped += 1
                if skipped % 5000 == 0:
                    logger.debug(
                        "Skipped %d rows so far (yielded %d)", skipped, yielded
                    )
                continue
            yielded += 1
            if yielded % 10000 == 0:
                logger.info("Yielded %d samples (skipped %d)", yielded, skipped)
            yield result

        logger.info(
            "Finished streaming %s: yielded=%d  skipped=%d",
            self.config.split, yielded, skipped,
        )

    def _process_row(self, row: dict) -> Optional[dict]:
        """
        Convert a single obfuscated-exebench row into tokenized tensors.

        Flat schema columns:
            obfuscated_asm, clean_ir, technique, fname, ...
        """
        source_text = row.get("obfuscated_asm", "")
        target_text = row.get("clean_ir", "")

        if not source_text or not target_text:
            return None

        # ── Clean IR target (same preprocessing as training/eval) ────
        if self.config.strip_ir_declares:
            target_text = strip_ir_noise(target_text)
            if not target_text.strip():
                return None

        if self.config.normalize_ir_structs:
            target_text = normalize_structs(target_text)

        # ── Tokenize using dp.tokenize (raw text → token IDs) ───────
        try:
            tok_src, tok_tgt = self.dp.tokenize(
                source=source_text,
                target=target_text,
                pair=self.config.pair,
                ids=True,
            )
        except (ValueError, KeyError, IndexError) as exc:
            logger.debug("Skipping row due to tokenize error: %s", exc)
            return None

        if tok_src is None or tok_tgt is None:
            return None

        # Length filter
        if len(tok_src) > self.config.max_source_len:
            return None
        if len(tok_tgt) > self.config.max_target_len:
            return None

        return {
            "input_ids": torch.tensor(tok_src, dtype=torch.long),
            "labels": torch.tensor(tok_tgt, dtype=torch.long),
        }


def collate_fn(
    batch: List[dict],
    pad_id: int = 1,
    label_pad_id: int = -100,
) -> dict:
    """
    Collate a list of samples into a padded batch dict.

    ``input_ids``     – padded with *pad_id*
    ``attention_mask`` – 1 where real token, 0 where padded
    ``labels``        – padded with *label_pad_id* (``-100`` so CrossEntropy
                        ignores padding positions)
    ``decoder_input_ids`` – right-shifted labels (standard BART teacher forcing)
    """
    input_ids = [s["input_ids"] for s in batch]
    labels = [s["labels"] for s in batch]

    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=label_pad_id)

    attention_mask = (input_ids_padded != pad_id).long()

    # BART decoder_input_ids = shift labels right, prepend <s> (id=0 for BART),
    # and use pad where label == -100
    bos_id = 0  # BART convention
    decoder_input_ids = labels_padded.new_full(labels_padded.shape, pad_id)
    decoder_input_ids[:, 0] = bos_id
    decoder_input_ids[:, 1:] = labels_padded[:, :-1].clone()
    # Replace -100 with pad_id in decoder_input_ids (can't embed -100)
    decoder_input_ids[decoder_input_ids == label_pad_id] = pad_id

    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask,
        "decoder_input_ids": decoder_input_ids,
        "labels": labels_padded,
    }
