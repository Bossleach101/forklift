#!/usr/bin/env python3
"""
Evaluate a Forklift model on the obfuscated-exebench dataset.

Task: obfuscated AArch64 assembly → clean LLVM IR
Dataset: leachl/obfuscated-exebench (test split)

Uses the same tokenization as the arm_ir-ir pair so the fine-tuned
model sees input in the same format it was trained on.

Usage
-----
    # Default: evaluate leachl/forklift-arm-ir-ir on obfuscated test set
    python scripts/evaluate_obfu.py

    # Evaluate baseline
    python scripts/evaluate_obfu.py \\
        --model jordiae/clang_opt3_ir_optz-ir_optz-2024-01-15-0959-e1bf-bc2b \\
        --pair opt3_ir_optz-ir_optz

    # Quick smoke test
    python scripts/evaluate_obfu.py --max-samples 10

    # Filter by technique
    python scripts/evaluate_obfu.py --technique Flatten
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
from forklift.ir_checker import check_ir, CompilabilityStats
from neurel_deob.training.data import strip_ir_noise

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ======================================================================
# Metrics  (identical to evaluate_exebench.py)
# ======================================================================

def compute_bleu(predictions: list[str], references: list[str]) -> float:
    import sacrebleu
    return sacrebleu.corpus_bleu(predictions, [references]).score


def compute_ned(predictions: list[str], references: list[str]) -> float:
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
    if not predictions:
        return 0.0
    matches = sum(1 for p, r in zip(predictions, references)
                  if p.strip() == r.strip())
    return matches / len(predictions) * 100.0


# ======================================================================
# Batched generation
# ======================================================================

def _flush_batch(
    batch_srcs: list[list[int]],
    batch_refs_tok: list[list[int]],
    batch_fnames: list[str],
    batch_techniques: list[str],
    model,
    dp: DP,
    pad_id: int,
    device: torch.device,
    args,
    all_preds: list[str],
    all_refs: list[str],
    all_fnames: list[str],
    all_techniques: list[str],
    all_metadata: list[dict]=None,
):
    if not batch_srcs:
        return

    tensors = [torch.tensor(s) for s in batch_srcs]
    input_ids = pad_sequence(tensors, batch_first=True, padding_value=pad_id).to(device)
    attention_mask = (input_ids != pad_id).long()

    with torch.no_grad():
        gen_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "num_beams": args.beam,
            "early_stopping": True,
        }
        if args.repetition_penalty != 1.0:
            gen_kwargs["repetition_penalty"] = args.repetition_penalty
        else:
             # Explicitly set to 1.0 to override model config
            gen_kwargs["repetition_penalty"] = 1.0

        if args.no_repeat_ngram_size != 0:
            gen_kwargs["no_repeat_ngram_size"] = args.no_repeat_ngram_size
        else:
            # Explicitly set to 0 to override model config
            gen_kwargs["no_repeat_ngram_size"] = 0
            
        generated = model.generate(
            input_ids,
            attention_mask=attention_mask,
            **gen_kwargs
        )

    for i, (gen, ref_tok, fname, tech) in enumerate(zip(generated, batch_refs_tok, batch_fnames, batch_techniques)):
        pred_text = dp.detokenize(gen.cpu().tolist())
        ref_text = dp.detokenize(ref_tok)
        pred_text = truncate_ir_output(pred_text)
        all_preds.append(pred_text)
        all_refs.append(ref_text)
        all_fnames.append(fname)
        all_techniques.append(tech)
        if all_metadata is not None:
            # We assume chunk length matches
            pass # Appended in caller

def _flush_buffer_sorted(
    buffer: list[tuple],
    batch_size: int,
    model, dp, pad_id, device, args,
    all_preds, all_refs, all_fnames, all_techniques, all_metadata=None,
):
    if not buffer:
        return
    buffer.sort(key=lambda t: len(t[0]))
    for i in range(0, len(buffer), batch_size):
        chunk = buffer[i : i + batch_size]
        _flush_batch(
            [c[0] for c in chunk],
            [c[1] for c in chunk],
            [c[2] for c in chunk],
            [c[3] for c in chunk],
            model, dp, pad_id, device, args,
            all_preds, all_refs, all_fnames, all_techniques, all_metadata
        )
        if all_metadata is not None:
            for c in chunk:
                all_metadata.append(c[4])


# ======================================================================
# Main evaluation
# ======================================================================

def run_evaluation(args) -> dict:
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
    logger.info("Pair: %s", args.pair)

    # ── Load obfuscated dataset ───────────────────────────────────────
    from datasets import load_dataset
    logger.info("Loading %s split=%s", args.dataset, args.split)
    ds = load_dataset(args.dataset, split=args.split, streaming=True)

    all_preds: list[str] = []
    all_refs: list[str] = []
    all_fnames: list[str] = []
    all_techniques: list[str] = []
    all_metadata: list[dict] = []
    skipped = 0
    total_seen = 0
    start_time = time.time()

    batch_size = args.batch_size
    BUFFER_MULT = 4
    buffer_cap = batch_size * BUFFER_MULT
    buffer: list[tuple[list[int], list[int], str, str, dict]] = []

    for row in ds:
        total_seen += 1

        if args.max_samples and len(all_preds) + len(buffer) >= args.max_samples:
            break

        # Optional technique filter
        if args.technique and row["technique"] != args.technique:
            continue

        # ── Source: obfuscated asm, Target: clean IR ──────────────────
        source_text = row.get("obfuscated_asm", "")
        target_text = row.get("clean_ir", "")

        if not source_text or not target_text:
            skipped += 1
            continue

        # Preprocess IR reference (same as training)
        if args.strip_ir:
            target_text = strip_ir_noise(target_text)
            if not target_text.strip():
                skipped += 1
                continue

        if args.normalize_structs:
            target_text = normalize_structs(target_text)

        # ── Tokenize using the same pair format ───────────────────────
        try:
            tok_src, tok_tgt = dp.tokenize(
                source=source_text,
                target=target_text,
                pair=args.pair,
                ids=True,
            )
        except (ValueError, KeyError, IndexError) as e:
            skipped += 1
            continue

        if tok_src is None or tok_tgt is None:
            skipped += 1
            continue

        if len(tok_src) > args.max_source_len:
            skipped += 1
            continue

        meta = {}
        if args.check_compilability or getattr(args, "check_functional", False):
            meta = {
                "synth_deps": row.get("c_deps", ""),
                "func_head_types": row.get("func_head_types", ""),
                "synth_exe_wrapper": row.get("c_wrapper", ""),
                "synth_io_pairs": row.get("io_pairs", []),
            }

        buffer.append((tok_src, tok_tgt, row.get("fname", "?"), row["technique"], meta))

        if len(buffer) >= buffer_cap:
            _flush_buffer_sorted(
                buffer, batch_size,
                model, dp, pad_id, device, args,
                all_preds, all_refs, all_fnames, all_techniques, all_metadata,
            )
            buffer = []

            elapsed = time.time() - start_time
            rate = len(all_preds) / elapsed if elapsed > 0 else 0
            logger.info(
                "Evaluated %d samples (%.2f/s) | skipped %d | seen %d rows",
                len(all_preds), rate, skipped, total_seen,
            )

    # Flush remaining
    _flush_buffer_sorted(
        buffer, batch_size,
        model, dp, pad_id, device, args,
        all_preds, all_refs, all_fnames, all_techniques, all_metadata,
    )

    elapsed = time.time() - start_time

    # ── Compute metrics ───────────────────────────────────────────────
    bleu = compute_bleu(all_preds, all_refs)
    ned = compute_ned(all_preds, all_refs)
    exact_match = compute_exact_match(all_preds, all_refs)

    # Per-technique breakdown
    from collections import defaultdict
    tech_preds: dict[str, list[str]] = defaultdict(list)
    tech_refs: dict[str, list[str]] = defaultdict(list)
    for p, r, t in zip(all_preds, all_refs, all_techniques):
        tech_preds[t].append(p)
        tech_refs[t].append(r)

    per_technique = {}
    for t in sorted(tech_preds.keys()):
        tp, tr = tech_preds[t], tech_refs[t]
        per_technique[t] = {
            "count": len(tp),
            "bleu": round(compute_bleu(tp, tr), 2),
            "ned": round(compute_ned(tp, tr), 4),
            "exact_match_pct": round(compute_exact_match(tp, tr), 2),
        }

    results = {
        "model": args.model,
        "dataset": args.dataset,
        "pair": args.pair,
        "split": args.split,
        "technique_filter": args.technique or "all",
        "num_evaluated": len(all_preds),
        "num_skipped": skipped,
        "total_rows_seen": total_seen,
        "bleu": round(bleu, 2),
        "ned": round(ned, 4),
        "exact_match_pct": round(exact_match, 2),
        "per_technique": per_technique,
        "beam": args.beam,
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "elapsed_seconds": round(elapsed, 1),
        "samples_per_second": round(len(all_preds) / elapsed, 2) if elapsed > 0 else 0,
    }

    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    for k, v in results.items():
        if k == "per_technique":
            logger.info("  per_technique:")
            for t, metrics in v.items():
                logger.info("    %-30s  %s", t, metrics)
        else:
            logger.info("  %-25s %s", k, v)
    logger.info("=" * 60)

    # ── Compilability / Functional checks ─────────────────────────────
    if args.check_compilability or getattr(args, "check_functional", False):
        check_level = 3 if getattr(args, "check_functional", False) else 2
        logger.info(
            "Running Level %d checks on %d predictions...", check_level, len(all_preds),
        )
        comp_stats = CompilabilityStats()
        from collections import defaultdict as _ddict
        tech_comp_stats: dict[str, CompilabilityStats] = _ddict(CompilabilityStats)
        for idx, (pred, fname, tech, meta) in enumerate(
            zip(all_preds, all_fnames, all_techniques, all_metadata)
        ):
            cr = check_ir(
                pred, 
                level=check_level,
                c_deps=meta.get("synth_deps"),
                func_c_signature=(meta.get("func_head_types") or "").replace("extern", ""),
                cpp_wrapper=meta.get("synth_exe_wrapper"),
                io_pairs=meta.get("synth_io_pairs"),
                max_io_tests=getattr(args, "max_io_tests", 5),
            )
            comp_stats.update(cr, fname=fname)
            tech_comp_stats[tech].update(cr, fname=fname)
            if (idx + 1) % 50 == 0:
                if check_level == 3:
                     logger.info(
                        "  Checked %d / %d  (syntax=%d  compile=%d  func=%d/%d)",
                        idx + 1, len(all_preds),
                        comp_stats.syntax_valid, comp_stats.compiles,
                        comp_stats.functional_pass, comp_stats.functional_tested,
                     )
                else:
                    logger.info(
                        "  Checked %d / %d  (syntax=%d  compile=%d)",
                        idx + 1, len(all_preds),
                        comp_stats.syntax_valid, comp_stats.compiles,
                    )
        comp_stats.log_summary()
        results["compilability"] = comp_stats.to_dict()
        # Per-technique compilability
        per_tech_comp = {}
        for t in sorted(tech_comp_stats.keys()):
            per_tech_comp[t] = tech_comp_stats[t].to_dict()
        results["compilability_per_technique"] = per_tech_comp

    # ── Save ──────────────────────────────────────────────────────────
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2))
        logger.info("Results saved to %s", output_path)

    if args.save_predictions:
        pred_path = Path(args.save_predictions)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pred_path, "w") as f:
            for fname, tech, pred, ref in zip(all_fnames, all_techniques, all_preds, all_refs):
                json.dump({
                    "fname": fname,
                    "technique": tech,
                    "prediction": pred,
                    "reference": ref,
                }, f)
                f.write("\n")
        logger.info("Predictions saved to %s", pred_path)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Forklift model on obfuscated-exebench"
    )
    # Dataset
    parser.add_argument(
        "--dataset", default="leachl/obfuscated-exebench",
        help="HF dataset repo (default: leachl/obfuscated-exebench)",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--technique", default=None,
        help="Filter to a single technique (e.g. Flatten, EncodeArithmetic, Flatten+EncodeArithmetic)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit number of evaluated samples",
    )
    parser.add_argument(
        "--max-source-len", type=int, default=2048,
        help="Skip samples with source tokens > this (default: 2048)",
    )

    # Model
    parser.add_argument(
        "--model", default="leachl/forklift-arm-ir-ir",
        help="HF model ID or local path",
    )
    parser.add_argument(
        "--pair", default="arm_ir-ir",
        help="Forklift pair key for tokenization (default: arm_ir-ir)",
    )

    # Generation
    parser.add_argument("--beam", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)

    # Preprocessing
    parser.add_argument("--strip-ir", action="store_true", default=True)
    parser.add_argument("--no-strip-ir", dest="strip_ir", action="store_false")
    parser.add_argument("--normalize-structs", action="store_true", default=True)
    parser.add_argument("--no-normalize-structs", dest="normalize_structs", action="store_false")

    # Output
    parser.add_argument("--output", default=None)
    parser.add_argument("--save-predictions", default=None)
    parser.add_argument("--device", default="auto")

    # Compilability
    parser.add_argument(
        "--check-compilability", action="store_true", default=False,
        help="Run Level 1 (llvm-as) and Level 2 (clang -c) checks on predictions",
    )
    parser.add_argument(
        "--check-functional", action="store_true", default=False,
        help="Run Level 3 functional tests on predictions (implies compilability)",
    )
    parser.add_argument(
        "--max-io-tests", type=int, default=5,
        help="Max IO test cases per sample for functional testing (default: 5)",
    )

    args = parser.parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
