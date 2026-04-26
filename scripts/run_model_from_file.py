#!/usr/bin/env python3
"""Run a Forklift seq2seq model on a single source function from a text file.

The input file should contain one source-language function definition, for
example ARM assembly when using the default ``arm_ir-ir`` pair.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from forklift.ir_checker import _inject_missing_declares

# Add project root to path so the script works when launched from `scripts/`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_tokenizer(model_path: str):
    from tokenizers import Tokenizer

    tokenizer_path = Path(model_path) / "tokenizer.json"
    if tokenizer_path.is_file():
        return Tokenizer.from_file(str(tokenizer_path))

    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    return Tokenizer.from_str(fs.open(str(tokenizer_path), "r").read())


def build_source_ids(
    dp,
    source_text: str,
    pair: str,
    *,
    normalize_structs_flag: bool = False,
    opt3_legacy: bool = False,
) -> list[int]:
    from forklift.utils import normalize_structs

    if normalize_structs_flag:
        source_text = normalize_structs(source_text)

    lang_source = dp.get_lang_from_pair(pair, "source")
    start_tok, end_tok, modifiers = dp.lang_special_token(
        lang_source, opt3_legacy=opt3_legacy
    )

    if modifiers:
        modifiers_start = " ".join(start for start, _ in modifiers)
        modifiers_end = " ".join(end for _, end in reversed(modifiers))
        wrapped = f"{start_tok} {modifiers_start} {source_text} {modifiers_end} {end_tok}"
    else:
        wrapped = f"{start_tok} {source_text} {end_tok}"

    normalized = dp.tokenizer.normalizer.normalize_str(wrapped)
    return dp.tokenizer.encode(normalized).ids


def load_model(model_path: str, device):
    from transformers import BartForConditionalGeneration

    return BartForConditionalGeneration.from_pretrained(model_path).eval().to(device)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a Forklift model on a single input file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_file", help="Text file containing the source function definition")
    parser.add_argument(
        "--model",
        default="checkpoints/arm_ir_ir_v4/step_80000",
        help="Model checkpoint directory or HF repo ID",
    )
    parser.add_argument(
        "--pair",
        default="arm_ir-ir",
        help="Forklift pair key that matches the input language",
    )
    parser.add_argument("--beam", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument(
        "--max-source-len",
        type=int,
        default=2048,
        help="Maximum source token length before the script errors",
    )
    parser.add_argument(
        "--truncate-input",
        action="store_true",
        help="Truncate the source to --max-source-len tokens instead of failing",
    )
    parser.add_argument(
        "--normalize-structs",
        action="store_true",
        help="Normalize struct names before tokenizing",
    )
    parser.add_argument(
        "--legacy-opt3-ir",
        action="store_true",
        help="Use the legacy opt3/IR token layout for older checkpoints",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional file to write the generated IR to",
    )
    args = parser.parse_args()

    import torch

    input_path = Path(args.input_file)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    from forklift.par_data import DP
    from forklift.utils import truncate_ir_output

    tokenizer = load_tokenizer(args.model)
    dp = DP(tokenizer=tokenizer)
    model = load_model(args.model, device)

    source_text = input_path.read_text().strip()
    if not source_text:
        raise ValueError(f"Input file is empty: {input_path}")

    input_ids = build_source_ids(
        dp,
        source_text,
        args.pair,
        normalize_structs_flag=args.normalize_structs,
        opt3_legacy=args.legacy_opt3_ir,
    )

    if len(input_ids) > args.max_source_len:
        if args.truncate_input:
            input_ids = input_ids[: args.max_source_len]
        else:
            raise ValueError(
                f"Input token length {len(input_ids)} exceeds --max-source-len {args.max_source_len}. "
                "Pass --truncate-input if you want to force truncation."
            )

    pad_id = tokenizer.get_vocab()["<pad>"]
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention_mask = (input_tensor != pad_id).long()

    with torch.no_grad():
        generated = model.generate(
            input_ids=input_tensor,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.beam,
            early_stopping=True,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )

    output_text = dp.detokenize(generated[0].detach().cpu().tolist())
    output_text = truncate_ir_output(output_text).strip()
    output_text = _inject_missing_declares(output_text)

    if args.output:
        Path(args.output).write_text(output_text + "\n")

    print(output_text)


if __name__ == "__main__":
    main()