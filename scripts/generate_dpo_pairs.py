#!/usr/bin/env python3
"""
Generate DPO preference pairs using compiler feedback.

For each ExeBench sample, generates N candidate IR translations,
scores them with check_ir(), and pairs the highest-scoring with the
lowest-scoring to create (prompt, chosen, rejected) triples for DPO
training.

Scoring hierarchy (higher = better):
    Level 3: functional IO pass   → score 4
    Level 3: links but IO fails   → score 3
    Level 2: compiles (clang -c)  → score 2
    Level 1: syntax valid         → score 1
    Nothing                       → score 0

When no candidate passes even Level 1, we fall back to using the
ground-truth IR as "chosen" and the worst candidate as "rejected".
This teaches the model "anything is better than garbled output".

Output: A HuggingFace Dataset (saved to disk or pushed to the Hub)
with columns: prompt, chosen, rejected, score_chosen, score_rejected,
fname, technique (if applicable).

Usage
-----
    # Generate from the v2 model on ExeBench test_synth
    python scripts/generate_dpo_pairs.py \
        --model leachl/forklift-arm-ir-ir \
        --num-candidates 5 \
        --max-samples 1000 \
        --output dpo_pairs/

    # Use ground-truth as chosen (always correct), model outputs as rejected
    python scripts/generate_dpo_pairs.py \
        --model leachl/forklift-arm-ir-ir \
        --num-candidates 1 \
        --gt-as-chosen \
        --max-samples 5000 \
        --output dpo_pairs_gt/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset, load_dataset
from tokenizers import Tokenizer
from transformers import BartForConditionalGeneration

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forklift.par_data import DP
from forklift.utils import normalize_structs, truncate_ir_output
from forklift.ir_checker import validate_ir_syntax, compile_ir_to_object, check_ir
from neurel_deob.training.data import strip_ir_noise

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ======================================================================
# Scoring
# ======================================================================

def score_ir(ir_text: str, *, level: int = 2) -> int:
    """Score an IR candidate.  Higher = better.

    Returns
    -------
    0 : fails syntax check
    1 : passes syntax (llvm-as) but fails compilation
    2 : passes compilation (clang -c)
    3 : links with wrapper (if metadata provided — not used here)
    4 : passes functional IO tests (if metadata provided — not used here)
    """
    ok, _ = validate_ir_syntax(ir_text, auto_declare=True)
    if not ok:
        return 0
    if level < 2:
        return 1
    ok, _ = compile_ir_to_object(ir_text, auto_declare=True)
    if not ok:
        return 1
    return 2


# ======================================================================
# Candidate generation
# ======================================================================

def generate_candidates(
    model: BartForConditionalGeneration,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    num_candidates: int,
    max_new_tokens: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> list[torch.Tensor]:
    """Generate N candidate outputs via diverse beam search."""
    # Ensure num_beams is divisible by num_beam_groups for diverse beam search
    num_beam_groups = min(num_candidates, 5)
    # E.g. if num_candidates=3 and groups=3, then beams=3 doesn't work if we want some extra beams
    # Make num_beams a strict multiple of num_beam_groups that is >= num_candidates
    multiplier = (max(num_candidates, 5) + num_beam_groups - 1) // num_beam_groups
    num_beams = multiplier * num_beam_groups
    
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        num_return_sequences=num_candidates,
        num_beam_groups=num_beam_groups,
        diversity_penalty=1.0,
        early_stopping=True,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        trust_remote_code=True,
        custom_generate='transformers-community/group-beam-search',
    )
    return list(outputs)


# ======================================================================
# ExeBench helpers (copied from evaluate_exebench.py for independence)
# ======================================================================

def load_test_stream(split: str, dataset: str, revision: str):
    return load_dataset(
        dataset,
        split=split,
        revision=revision,
        subsets=[split],
        streaming=True,
    )


def get_asm_code(row: dict, key: str) -> Optional[str]:
    targets = row.get("asm", {}).get("target", [])
    if key not in targets:
        return None
    idx = targets.index(key)
    return row["asm"]["code"][idx]


# ======================================================================
# Main
# ======================================================================

def run(args):
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

    # ── Stream data ───────────────────────────────────────────────────
    logger.info("Loading split: %s", args.split)
    test_stream = load_test_stream(args.split, args.hf_dataset, args.revision)

    pairs = []   # collected preference pairs
    skipped = 0
    total_seen = 0
    n_gt_fallback = 0
    start_time = time.time()

    score_hist = {0: 0, 1: 0, 2: 0}  # score distribution of best candidates

    for row in test_stream:
        total_seen += 1

        if args.max_samples and len(pairs) >= args.max_samples:
            break

        # Get source / target text
        source_text = get_asm_code(row, src_key)
        target_text = get_asm_code(row, tgt_key)
        if not source_text or not target_text:
            skipped += 1
            continue

        # Clean IR reference (same as training)
        if args.strip_ir and "ir" in args.pair:
            target_text = strip_ir_noise(target_text)
            if not target_text.strip():
                skipped += 1
                continue
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
        if len(tok_src) > args.max_source_len:
            skipped += 1
            continue

        # ── Generate candidates ──────────────────────────────────────
        input_ids = torch.tensor([tok_src], dtype=torch.long, device=device)
        attention_mask = (input_ids != pad_id).long()

        with torch.no_grad():
            candidates = generate_candidates(
                model, input_ids, attention_mask,
                num_candidates=args.num_candidates,
                max_new_tokens=args.max_new_tokens,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
            )

        # Decode and score each candidate
        scored = []
        for cand_ids in candidates:
            cand_text = dp.detokenize(cand_ids.cpu().tolist())
            cand_text = truncate_ir_output(cand_text)
            sc = score_ir(cand_text, level=args.check_level)
            scored.append((sc, cand_text))

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_text = scored[0]
        worst_score, worst_text = scored[-1]

        score_hist[best_score] = score_hist.get(best_score, 0) + 1

        # ── Build preference pair ────────────────────────────────────
        # Decode source text (prompt) from tokens
        prompt_text = dp.detokenize(tok_src)

        if args.gt_as_chosen:
            # Always use ground-truth as chosen
            # Decode target from the token IDs
            ref_text = dp.detokenize(tok_tgt)
            chosen = ref_text
            rejected = worst_text
            sc_chosen = 99   # ground-truth is definitionally the best
            sc_rejected = worst_score
        elif best_score > worst_score:
            # Normal case: best candidate > worst candidate
            chosen = best_text
            rejected = worst_text
            sc_chosen = best_score
            sc_rejected = worst_score
        elif best_score == 0:
            # All candidates equally bad — use ground truth as chosen
            ref_text = dp.detokenize(tok_tgt)
            chosen = ref_text
            rejected = worst_text
            sc_chosen = 99
            sc_rejected = 0
            n_gt_fallback += 1
        else:
            # All candidates have same (non-zero) score — skip,
            # no signal for DPO
            skipped += 1
            continue

        pairs.append({
            "prompt": prompt_text,
            "chosen": chosen,
            "rejected": rejected,
            "score_chosen": sc_chosen,
            "score_rejected": sc_rejected,
            "fname": row.get("fname", "?"),
        })

        if len(pairs) % 50 == 0:
            elapsed = time.time() - start_time
            rate = len(pairs) / elapsed if elapsed > 0 else 0
            logger.info(
                "Generated %d pairs (%.1f/s) | skipped %d | gt_fallback %d | "
                "best_score dist: %s",
                len(pairs), rate, skipped, n_gt_fallback,
                {k: v for k, v in sorted(score_hist.items())},
            )

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("DONE: %d preference pairs in %.1fs", len(pairs), elapsed)
    logger.info("  skipped: %d   gt_fallback: %d", skipped, n_gt_fallback)
    logger.info("  best_score distribution: %s", score_hist)
    logger.info("=" * 60)

    if not pairs:
        logger.error("No pairs generated — nothing to save.")
        return

    # ── Save as HuggingFace Dataset ───────────────────────────────────
    ds = Dataset.from_list(pairs)
    out_path = Path(args.output)

    if args.push_to_hub:
        ds.push_to_hub(args.push_to_hub, split="train")
        logger.info("Pushed to Hub: %s", args.push_to_hub)
    else:
        out_path.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(out_path))
        logger.info("Saved dataset to: %s", out_path)

    # Also save a human-readable sample
    sample_path = out_path / "samples.json"
    with open(sample_path, "w") as f:
        json.dump(pairs[:10], f, indent=2)
    logger.info("Saved %d sample pairs to: %s", min(10, len(pairs)), sample_path)


def main():
    parser = argparse.ArgumentParser(
        description="Generate DPO preference pairs with compiler feedback",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    parser.add_argument("--model", default="leachl/forklift-arm-ir-ir",
                        help="Model to generate candidates from")
    parser.add_argument("--pair", default="arm_ir-ir")
    parser.add_argument("--asm-key", default="angha")

    # Data
    parser.add_argument("--hf-dataset", default="jordiae/exebench")
    parser.add_argument("--revision", default="clang")
    parser.add_argument("--split", default="train_synth_compilable",
                        help="ExeBench split to generate from")
    parser.add_argument("--max-samples", type=int, default=5000,
                        help="Max number of preference pairs to generate")
    parser.add_argument("--max-source-len", type=int, default=1024)

    # Generation
    parser.add_argument("--num-candidates", type=int, default=5,
                        help="Number of candidate outputs per sample")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=6)
    parser.add_argument("--check-level", type=int, default=2,
                        help="Max check level (1=syntax, 2=compile)")

    # Output strategy
    parser.add_argument("--gt-as-chosen", action="store_true",
                        help="Always use ground-truth IR as the chosen response")

    # Preprocessing
    parser.add_argument("--strip-ir", action="store_true", default=True)
    parser.add_argument("--no-strip-ir", dest="strip_ir", action="store_false")
    parser.add_argument("--normalize-structs", action="store_true", default=True)
    parser.add_argument("--no-normalize-structs", dest="normalize_structs",
                        action="store_false")

    # Output
    parser.add_argument("--output", default="dpo_pairs/",
                        help="Directory to save dataset")
    parser.add_argument("--push-to-hub", default=None,
                        help="Push dataset to HF Hub repo (e.g. leachl/forklift-dpo-pairs)")

    # Hardware
    parser.add_argument("--device", default="auto")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
