#!/usr/bin/env python3
"""
Evaluate a Forklift ARM→IR model on ExeBench test splits.

Streams ExeBench test data, generates predictions with beam search,
and computes BLEU, NED, and exact-match metrics.

Usage
-----
    # Evaluate fine-tuned v2 model on test_synth (default)
    python scripts/evaluate_exebench.py

    # Evaluate baseline model
    python scripts/evaluate_exebench.py \
        --model jordiae/clang_opt3_ir_optz-ir_optz-2024-01-15-0959-e1bf-bc2b \
        --pair opt3_ir_optz-ir_optz

    # Evaluate on test_real split
    python scripts/evaluate_exebench.py --split test_real --asm-key real

    # Quick smoke test (20 samples)
    python scripts/evaluate_exebench.py --max-samples 20
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer
from torch.nn.utils.rnn import pad_sequence
from transformers import BartForConditionalGeneration

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forklift.par_data import DP
from forklift.utils import normalize_structs, truncate_ir_output
from neurel_deob.training.data import strip_ir_noise

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ======================================================================
# Metrics
# ======================================================================

def compute_bleu(predictions: list[str], references: list[str]) -> float:
    """Corpus-level BLEU via sacrebleu."""
    import sacrebleu
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    return bleu.score


def compute_ned(predictions: list[str], references: list[str]) -> float:
    """Mean normalised edit distance (0 = identical, 1 = completely different)."""
    import editdistance
    total = 0.0
    count = 0
    for pred, ref in zip(predictions, references):
        if not ref:
            continue
        d = editdistance.eval(pred, ref)
        total += d / max(len(pred), len(ref), 1)
        count += 1
    return total / max(count, 1)


def compute_exact_match(predictions: list[str], references: list[str]) -> float:
    """Fraction of predictions that exactly match the reference."""
    if not predictions:
        return 0.0
    matches = sum(1 for p, r in zip(predictions, references)
                  if p.strip() == r.strip())
    return matches / len(predictions) * 100.0


# ======================================================================
# Data loading
# ======================================================================

def load_test_stream(split: str, hf_dataset: str, revision: str):
    """Load an ExeBench test split as a streaming iterator."""
    from datasets import load_dataset
    return load_dataset(
        hf_dataset,
        split=split,
        revision=revision,
        subsets=[split],
        streaming=True,
    )


def get_asm_code(row: dict, target_name: str) -> str | None:
    """Extract assembly/IR code for a specific target from an ExeBench row."""
    targets = row.get("asm", {}).get("target", [])
    codes = row.get("asm", {}).get("code", [])
    if target_name in targets:
        idx = targets.index(target_name)
        code = codes[idx] if idx < len(codes) else None
        return code if code else None
    return None


# ======================================================================
# Evaluation loop
# ======================================================================

def _flush_batch(
    batch_srcs: list[list[int]],
    batch_refs_tok: list[list[int]],
    batch_fnames: list[str],
    model,
    dp: DP,
    pad_id: int,
    device: torch.device,
    args,
    all_preds: list[str],
    all_refs: list[str],
    all_fnames: list[str],
):
    """Generate predictions for an accumulated batch."""
    if not batch_srcs:
        return

    # Pad source sequences
    tensors = [torch.tensor(s) for s in batch_srcs]
    input_ids = pad_sequence(tensors, batch_first=True, padding_value=pad_id).to(device)
    attention_mask = (input_ids != pad_id).long()

    with torch.no_grad():
        generated = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.beam,
            early_stopping=True,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )

    for gen, ref_tok, fname in zip(generated, batch_refs_tok, batch_fnames):
        pred_text = dp.detokenize(gen.cpu().tolist())
        ref_text = dp.detokenize(ref_tok)
        pred_text = truncate_ir_output(pred_text)
        all_preds.append(pred_text)
        all_refs.append(ref_text)
        all_fnames.append(fname)


def run_evaluation(args) -> dict:
    """Main evaluation loop with batched generation."""

    device = torch.device(args.device if args.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Device: %s", device)

    # ── Load model & tokenizer ────────────────────────────────────────
    logger.info("Loading model: %s", args.model)
    try:
        tok = Tokenizer.from_file(os.path.join(args.model, "tokenizer.json"))
    except Exception:
        from huggingface_hub import HfFileSystem
        fs = HfFileSystem()
        tok = Tokenizer.from_str(
            fs.open(os.path.join(args.model, "tokenizer.json"), "r").read()
        )

    model = BartForConditionalGeneration.from_pretrained(args.model).eval().to(device)
    dp = DP(tokenizer=tok)
    pad_id = tok.get_vocab()["<pad>"]
    logger.info("Model loaded (%d params)", sum(p.numel() for p in model.parameters()))

    # ── Resolve pair keys ─────────────────────────────────────────────
    _dp_resolver = DP()
    src_key, tgt_key, _, _ = _dp_resolver.get_par_data(
        row=None, pair=args.pair, asm_key=args.asm_key, fPIC=False
    )
    logger.info("Pair %s → source=%s  target=%s", args.pair, src_key, tgt_key)

    # ── Stream test data ──────────────────────────────────────────────
    logger.info("Loading split: %s", args.split)
    test_stream = load_test_stream(args.split, args.hf_dataset, args.revision)

    all_preds: list[str] = []
    all_refs: list[str] = []
    all_fnames: list[str] = []
    skipped = 0
    total_seen = 0
    start_time = time.time()

    batch_size = args.batch_size
    batch_srcs: list[list[int]] = []
    batch_refs_tok: list[list[int]] = []
    batch_fnames: list[str] = []

    for row in test_stream:
        total_seen += 1

        if args.max_samples and len(all_preds) + len(batch_srcs) >= args.max_samples:
            break

        # Check required targets exist
        source_text = get_asm_code(row, src_key)
        target_text = get_asm_code(row, tgt_key)
        if not source_text or not target_text:
            skipped += 1
            continue

        # Clean IR reference (same preprocessing as training)
        if args.strip_ir and "ir" in args.pair:
            target_text = strip_ir_noise(target_text)
            if not target_text.strip():
                skipped += 1
                continue
            # Inject cleaned target back for tokenisation
            codes = list(row["asm"]["code"])
            tgt_idx = row["asm"]["target"].index(tgt_key)
            codes[tgt_idx] = target_text
            row = dict(row)
            row["asm"] = dict(row["asm"])
            row["asm"]["code"] = codes

        # Tokenize
        try:
            _src, _tgt, tok_src, tok_tgt = dp.get_par_data(
                row, pair=args.pair, asm_key=args.asm_key,
                fPIC=False, tokenize_ids=True,
                do_normalize_ir_structs=args.normalize_structs,
            )
        except (ValueError, KeyError, IndexError):
            skipped += 1
            continue

        if tok_src is None or tok_tgt is None:
            skipped += 1
            continue

        # Length filter (same as training)
        if len(tok_src) > args.max_source_len:
            skipped += 1
            continue

        # Accumulate into batch
        batch_srcs.append(tok_src)
        batch_refs_tok.append(tok_tgt)
        batch_fnames.append(row.get("fname", "?"))

        # Flush batch when full
        if len(batch_srcs) >= batch_size:
            _flush_batch(
                batch_srcs, batch_refs_tok, batch_fnames,
                model, dp, pad_id, device, args,
                all_preds, all_refs, all_fnames,
            )
            batch_srcs, batch_refs_tok, batch_fnames = [], [], []

        if len(all_preds) % 50 == 0 and len(all_preds) > 0:
            elapsed = time.time() - start_time
            rate = len(all_preds) / elapsed
            logger.info(
                "Evaluated %d samples (%.1f/s) | skipped %d | seen %d rows",
                len(all_preds), rate, skipped, total_seen,
            )

    # Flush remaining samples
    _flush_batch(
        batch_srcs, batch_refs_tok, batch_fnames,
        model, dp, pad_id, device, args,
        all_preds, all_refs, all_fnames,
    )

    elapsed = time.time() - start_time

    # ── Compute metrics ───────────────────────────────────────────────
    bleu = compute_bleu(all_preds, all_refs)
    ned = compute_ned(all_preds, all_refs)
    exact_match = compute_exact_match(all_preds, all_refs)

    results = {
        "model": args.model,
        "pair": args.pair,
        "split": args.split,
        "asm_key": args.asm_key,
        "num_evaluated": len(all_preds),
        "num_skipped": skipped,
        "total_rows_seen": total_seen,
        "bleu": round(bleu, 2),
        "ned": round(ned, 4),
        "exact_match_pct": round(exact_match, 2),
        "beam": args.beam,
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "max_new_tokens": args.max_new_tokens,
        "elapsed_seconds": round(elapsed, 1),
        "samples_per_second": round(len(all_preds) / elapsed, 2) if elapsed > 0 else 0,
    }

    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    for k, v in results.items():
        logger.info("  %-25s %s", k, v)
    logger.info("=" * 60)

    # ── Save results ──────────────────────────────────────────────────
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2))
        logger.info("Results saved to %s", output_path)

    if args.save_predictions:
        pred_path = Path(args.save_predictions)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pred_path, "w") as f:
            for fname, pred, ref in zip(all_fnames, all_preds, all_refs):
                json.dump({"fname": fname, "prediction": pred, "reference": ref}, f)
                f.write("\n")
        logger.info("Predictions saved to %s", pred_path)

    return results


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Forklift model on ExeBench test split"
    )

    # Model
    parser.add_argument(
        "--model", default="leachl/forklift-arm-ir-ir",
        help="HF model path or local checkpoint dir",
    )
    parser.add_argument(
        "--pair", default="arm_ir-ir",
        help="Forklift pair string (default: arm_ir-ir)",
    )

    # Data
    parser.add_argument(
        "--split", default="test_synth",
        help="ExeBench split to evaluate on (default: test_synth)",
    )
    parser.add_argument(
        "--hf-dataset", default="jordiae/exebench",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--revision", default="clang",
        help="HF dataset revision",
    )
    parser.add_argument(
        "--asm-key", default="angha",
        help="ASM key: angha (synth) or real",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Max samples to evaluate (default: all)",
    )
    parser.add_argument(
        "--max-source-len", type=int, default=1024,
        help="Max source sequence length",
    )

    # Generation
    parser.add_argument("--beam", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=6)
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Number of samples to generate in parallel (default: 8)",
    )

    # Preprocessing
    parser.add_argument(
        "--strip-ir", action="store_true", default=True,
        help="Strip declare/attributes/metadata from IR reference",
    )
    parser.add_argument("--no-strip-ir", dest="strip_ir", action="store_false")
    parser.add_argument(
        "--normalize-structs", action="store_true", default=True,
        help="Normalize struct names in IR",
    )
    parser.add_argument("--no-normalize-structs", dest="normalize_structs",
                        action="store_false")

    # Output
    parser.add_argument(
        "--output", default=None,
        help="Path to save results JSON",
    )
    parser.add_argument(
        "--save-predictions", default=None,
        help="Path to save per-sample predictions (JSONL)",
    )
    parser.add_argument(
        "--device", default="auto",
        help="Device: auto, cpu, cuda, cuda:0",
    )

    args = parser.parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
