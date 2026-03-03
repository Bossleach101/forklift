"""
Fine-tune the Forklift BART model on ExeBench arm_ir-ir pairs.

This is the main training entry-point.  It can be run directly::

    python -m neurel_deob.training.finetune                    # defaults
    python -m neurel_deob.training.finetune --max_steps 500    # quick test
    python -m neurel_deob.training.finetune --resume_from checkpoints/arm_ir_ir/step_4000

Key design decisions
--------------------
* **Streaming data** – ExeBench is too large to download in full on
  most machines.  We stream from the HF Hub and iterate once per
  "epoch".  Because the stream is effectively infinite for practical
  purposes, we train for a fixed number of *steps* rather than epochs.

* **Teacher forcing** – Standard seq2seq: the model receives the
  right-shifted ground-truth target as ``decoder_input_ids``.  Loss is
  cross-entropy on the ``labels``.

* **Mixed precision** – fp16/bf16 via ``torch.cuda.amp`` for ~2× speed
  on modern GPUs.  Falls back gracefully on CPU.

* **Periodic validation** – Every ``eval_steps`` we stream a fixed
  number of validation samples, compute loss + BLEU + NED, and decide
  whether to keep the checkpoint.

* **Pair routing** – Handled entirely by ``forklift.par_data.DP``, so
  the token format is *identical* to the original Forklift training.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from tokenizers import Tokenizer
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BartForConditionalGeneration, get_scheduler

from forklift.par_data import DP

from .config import TrainConfig
from .data import DataConfig, ExeBenchDataset, ObfuscatedExeBenchDataset, collate_fn
from neurel_deob.obfuscation.pipeline import ObfuscationConfig

logger = logging.getLogger(__name__)


# ======================================================================
# Helpers
# ======================================================================

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ======================================================================
# Metrics (lightweight — no external deps beyond sacrebleu/editdistance)
# ======================================================================

def compute_bleu(predictions: list[str], references: list[str]) -> float:
    """Corpus-level BLEU via sacrebleu."""
    try:
        import sacrebleu
        bleu = sacrebleu.corpus_bleu(predictions, [references])
        return bleu.score
    except ImportError:
        return 0.0


def compute_ned(predictions: list[str], references: list[str]) -> float:
    """Mean normalised edit distance (0 = identical, 1 = completely different)."""
    try:
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
    except ImportError:
        return 0.0


# ======================================================================
# Evaluation
# ======================================================================

@torch.no_grad()
def evaluate(
    model: BartForConditionalGeneration,
    tokenizer: Tokenizer,
    dp: DP,
    config: TrainConfig,
    device: torch.device,
) -> dict:
    """
    Run evaluation on ``valid_synth`` split.

    Returns a dict with ``val_loss``, ``bleu``, ``ned``, and
    ``num_samples``.
    """
    model.eval()

    data_cfg = DataConfig(
        hf_dataset=config.hf_dataset,
        revision=config.hf_revision,
        split=config.valid_split,
        pair=config.pair,
        asm_key=config.asm_key,
        max_source_len=config.max_source_len,
        max_target_len=config.max_target_len,
        normalize_ir_structs=config.normalize_ir_structs,
        strip_ir_declares=config.strip_ir_declares,
        streaming=config.streaming,
    )

    val_ds = ExeBenchDataset(tokenizer, data_cfg)
    pad_id = tokenizer.get_vocab()["<pad>"]
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        collate_fn=lambda b: collate_fn(b, pad_id=pad_id),
        num_workers=0,
    )

    total_loss = 0.0
    n_samples = 0
    all_preds: list[str] = []
    all_refs: list[str] = []

    for batch in val_loader:
        if n_samples >= config.eval_samples:
            break

        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            decoder_input_ids=batch["decoder_input_ids"],
            labels=batch["labels"],
        )
        total_loss += outputs.loss.item() * batch["input_ids"].size(0)
        n_samples += batch["input_ids"].size(0)

        # Generate predictions for BLEU/NED (beam search, slower)
        generated = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_new_tokens=config.eval_max_new_tokens,
            num_beams=config.eval_beam,
            early_stopping=True,
            repetition_penalty=config.repetition_penalty,
            no_repeat_ngram_size=config.no_repeat_ngram_size,
        )
        for gen, lab in zip(generated, batch["labels"]):
            pred_text = dp.detokenize(gen.cpu().tolist())
            # Reconstruct reference from labels (replace -100 → pad so decode works)
            lab_ids = lab.clone()
            lab_ids[lab_ids == -100] = pad_id
            ref_text = dp.detokenize(lab_ids.cpu().tolist())
            all_preds.append(pred_text)
            all_refs.append(ref_text)

    avg_loss = total_loss / max(n_samples, 1)
    bleu = compute_bleu(all_preds, all_refs)
    ned = compute_ned(all_preds, all_refs)

    model.train()

    return {
        "val_loss": avg_loss,
        "bleu": bleu,
        "ned": ned,
        "num_samples": n_samples,
    }


# ======================================================================
# Checkpoint management
# ======================================================================

def save_checkpoint(
    model: BartForConditionalGeneration,
    tokenizer: Tokenizer,
    optimizer: AdamW,
    scaler: Optional[GradScaler],
    step: int,
    config: TrainConfig,
    metrics: dict,
):
    """Save model + optimizer + metadata to ``checkpoint_dir/step_{step}``."""
    ckpt_dir = Path(config.checkpoint_dir) / f"step_{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save HuggingFace model (can be loaded with from_pretrained later)
    model.save_pretrained(str(ckpt_dir))
    tokenizer.save(str(ckpt_dir / "tokenizer.json"))

    # Save optimizer & scaler state
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler else None,
            "step": step,
            "config": vars(config),
            "metrics": metrics,
        },
        str(ckpt_dir / "training_state.pt"),
    )

    # Write a human-readable metrics file
    with open(ckpt_dir / "metrics.json", "w") as f:
        json.dump({"step": step, **metrics}, f, indent=2)

    logger.info("Saved checkpoint at step %d → %s", step, ckpt_dir)

    # Prune old checkpoints
    _prune_checkpoints(config)


def _prune_checkpoints(config: TrainConfig):
    """Keep only the ``save_total_limit`` most recent checkpoints."""
    ckpt_root = Path(config.checkpoint_dir)
    if not ckpt_root.exists():
        return
    dirs = sorted(ckpt_root.glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
    while len(dirs) > config.save_total_limit:
        old = dirs.pop(0)
        logger.info("Pruning old checkpoint %s", old)
        import shutil
        shutil.rmtree(old)


# ======================================================================
# Main training loop
# ======================================================================

def train(config: TrainConfig):
    """Execute the full fine-tuning run."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    set_seed(config.seed)
    device = resolve_device(config.device)
    logger.info("Device: %s", device)

    # ── Load tokenizer ───────────────────────────────────────────────
    tok_path = os.path.join(config.model_path, "tokenizer.json")
    if os.path.exists(tok_path):
        tokenizer = Tokenizer.from_file(tok_path)
    else:
        from huggingface_hub import HfFileSystem
        fs = HfFileSystem()
        tokenizer = Tokenizer.from_str(
            fs.open(os.path.join(config.model_path, "tokenizer.json"), "r").read()
        )

    dp = DP(tokenizer=tokenizer)
    pad_id = tokenizer.get_vocab()["<pad>"]

    # ── Load model ───────────────────────────────────────────────────
    # When resuming, load model weights from the checkpoint directory
    # (which contains model.safetensors from save_pretrained()), not
    # from the original base model.
    model_load_path = config.model_path
    if config.resume_from:
        ckpt = Path(config.resume_from)
        if (ckpt / "model.safetensors").exists() or (ckpt / "pytorch_model.bin").exists():
            model_load_path = str(ckpt)
            logger.info("Will load model weights from checkpoint: %s", model_load_path)
        else:
            logger.warning(
                "Checkpoint %s has no model weights — falling back to %s",
                ckpt, config.model_path,
            )

    logger.info("Loading model from %s", model_load_path)
    model = BartForConditionalGeneration.from_pretrained(model_load_path)
    model.to(device)
    model.train()
    logger.info("Model parameters: %s (%.1fM trainable)", 
                count_parameters(model),
                count_parameters(model) / 1e6)

    # ── Optimizer ────────────────────────────────────────────────────
    no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight"}
    param_groups = [
        {
            "params": [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        param_groups,
        lr=config.lr,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_eps,
    )

    # ── LR scheduler ────────────────────────────────────────────────
    scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.max_steps,
    )

    # ── Mixed precision ─────────────────────────────────────────────
    use_amp = (config.fp16 or config.bf16) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if config.bf16 else torch.float16
    scaler = GradScaler("cuda", enabled=use_amp and not config.bf16)  # no scaler for bf16

    # ── Resume from checkpoint ───────────────────────────────────────
    global_step = 0
    if config.resume_from:
        ckpt = Path(config.resume_from)
        logger.info("Resuming from %s", ckpt)
        state = torch.load(ckpt / "training_state.pt", map_location=device)
        optimizer.load_state_dict(state["optimizer"])
        if scaler and state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        global_step = state["step"]
        # Advance LR scheduler to the correct position
        for _ in range(global_step):
            scheduler.step()
        logger.info("Resumed at step %d (lr=%.2e)", global_step, scheduler.get_last_lr()[0])

    # ── Data ─────────────────────────────────────────────────────────
    data_cfg = DataConfig(
        hf_dataset=config.hf_dataset,
        revision=config.hf_revision,
        split=config.train_split,
        pair=config.pair,
        asm_key=config.asm_key,
        max_source_len=config.max_source_len,
        max_target_len=config.max_target_len,
        normalize_ir_structs=config.normalize_ir_structs,
        strip_ir_declares=config.strip_ir_declares,
        streaming=config.streaming,
    )

    if config.obfuscate:
        obfu_cfg = ObfuscationConfig(
            techniques=config.obfu_techniques,
            intensity_range=(config.obfu_intensity_min, config.obfu_intensity_max),
            min_transforms=config.obfu_min_transforms,
            max_transforms=config.obfu_max_transforms,
        )
        train_ds = ObfuscatedExeBenchDataset(
            tokenizer, data_cfg, obfu_config=obfu_cfg, seed=config.seed,
        )
        logger.info(
            "Obfuscation ENABLED: techniques=%s  intensity=(%.2f, %.2f)  transforms=(%d, %d)",
            config.obfu_techniques, config.obfu_intensity_min, config.obfu_intensity_max,
            config.obfu_min_transforms, config.obfu_max_transforms,
        )
    else:
        train_ds = ExeBenchDataset(tokenizer, data_cfg)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        collate_fn=lambda b: collate_fn(b, pad_id=pad_id),
        num_workers=config.num_workers,
    )

    # ── TensorBoard ─────────────────────────────────────────────────
    tb_writer = None
    if config.use_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_writer = SummaryWriter(log_dir=config.tensorboard_dir)
            logger.info("TensorBoard logging to %s", config.tensorboard_dir)
        except ImportError:
            logger.warning("tensorboard not installed — skipping TB logging")

    # ── Training ─────────────────────────────────────────────────────
    logger.info("Starting training: max_steps=%d  eff_batch=%d  lr=%.2e",
                config.max_steps, config.effective_batch_size, config.lr)

    model.zero_grad()
    running_loss = 0.0
    accum_count = 0
    epoch = 0
    t_start = time.time()

    while global_step < config.max_steps:
        epoch += 1
        logger.info("Starting epoch %d (streaming) at step %d", epoch, global_step)

        for batch in train_loader:
            if global_step >= config.max_steps:
                break

            batch = {k: v.to(device) for k, v in batch.items()}

            with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    decoder_input_ids=batch["decoder_input_ids"],
                    labels=batch["labels"],
                )
                loss = outputs.loss / config.gradient_accumulation_steps

            if use_amp and not config.bf16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += loss.item()
            accum_count += 1

            # Gradient accumulation step
            if accum_count % config.gradient_accumulation_steps == 0:
                if use_amp and not config.bf16:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()

                scheduler.step()
                model.zero_grad()
                global_step += 1

                # ── Logging ──────────────────────────────────────────
                if global_step % config.log_steps == 0:
                    avg_loss = running_loss / config.gradient_accumulation_steps
                    lr_now = scheduler.get_last_lr()[0]
                    elapsed = time.time() - t_start
                    steps_per_sec = global_step / elapsed if elapsed > 0 else 0
                    logger.info(
                        "step=%d  loss=%.4f  lr=%.2e  steps/s=%.2f",
                        global_step, avg_loss, lr_now, steps_per_sec,
                    )
                    if tb_writer:
                        tb_writer.add_scalar("train/loss", avg_loss, global_step)
                        tb_writer.add_scalar("train/lr", lr_now, global_step)
                    running_loss = 0.0

                # ── Evaluation ───────────────────────────────────────
                if global_step % config.eval_steps == 0:
                    logger.info("Running evaluation at step %d …", global_step)
                    metrics = evaluate(model, tokenizer, dp, config, device)
                    logger.info(
                        "EVAL step=%d  val_loss=%.4f  BLEU=%.2f  NED=%.4f  (n=%d)",
                        global_step,
                        metrics["val_loss"],
                        metrics["bleu"],
                        metrics["ned"],
                        metrics["num_samples"],
                    )
                    if tb_writer:
                        tb_writer.add_scalar("eval/loss", metrics["val_loss"], global_step)
                        tb_writer.add_scalar("eval/bleu", metrics["bleu"], global_step)
                        tb_writer.add_scalar("eval/ned", metrics["ned"], global_step)
                    model.train()

                # ── Save checkpoint ──────────────────────────────────
                if global_step % config.save_steps == 0:
                    metrics = {"step": global_step}
                    save_checkpoint(
                        model, tokenizer, optimizer, scaler,
                        global_step, config, metrics,
                    )

    # ── Final save ───────────────────────────────────────────────────
    logger.info("Training complete at step %d", global_step)
    save_checkpoint(
        model, tokenizer, optimizer, scaler,
        global_step, config, {"step": global_step, "final": True},
    )

    if tb_writer:
        tb_writer.close()

    return model


# ======================================================================
# CLI
# ======================================================================

def parse_args() -> TrainConfig:
    """Build a TrainConfig from CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Forklift on ExeBench arm_ir-ir pairs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    cfg = TrainConfig()

    for name, default in vars(cfg).items():
        if name.startswith("_"):
            continue
        arg_type = type(default) if default is not None else str
        if arg_type is bool:
            # For booleans, create --flag / --no-flag pairs so both
            # directions work regardless of the default value.
            group = parser.add_mutually_exclusive_group()
            group.add_argument(
                f"--{name}",
                dest=name,
                action="store_true",
                default=default,
            )
            group.add_argument(
                f"--no_{name}",
                dest=name,
                action="store_false",
            )
        elif isinstance(default, list):
            # List fields: accept multiple values via --name val1 val2
            elem_type = type(default[0]) if default else str
            parser.add_argument(
                f"--{name}", type=elem_type, nargs="+", default=default
            )
        else:
            parser.add_argument(f"--{name}", type=arg_type, default=default)

    args = parser.parse_args()

    # Build config from parsed args
    config = TrainConfig(**{k: v for k, v in vars(args).items() if hasattr(cfg, k)})
    return config


def main():
    config = parse_args()
    train(config)


if __name__ == "__main__":
    main()
