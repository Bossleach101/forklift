"""
DPO (Direct Preference Optimization) trainer for seq2seq BART models.

TRL's DPOTrainer only supports causal (decoder-only) LMs.  This module
implements DPO for encoder-decoder models like BART, using compiler
feedback as the preference signal.

The DPO loss is:

    L = -E[ log σ( β · (log π_θ(y⁺|x)/π_ref(y⁺|x)
                       - log π_θ(y⁻|x)/π_ref(y⁻|x)) ) ]

where:
    x   = prompt (ARM assembly)
    y⁺  = chosen (better IR)
    y⁻  = rejected (worse IR)
    π_θ = policy model being trained
    π_ref = frozen reference model (initial policy)
    β   = temperature controlling preference strength

Usage
-----
    python -m neurel_deob.training.dpo_finetune \\
        --model_path leachl/forklift-arm-ir-ir \\
        --dpo_dataset dpo_pairs/ \\
        --max_steps 5000 \\
        --lr 5e-7 \\
        --beta 0.1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset, load_from_disk, Dataset
from tokenizers import Tokenizer
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, IterableDataset
from torch.nn.utils.rnn import pad_sequence
from transformers import BartForConditionalGeneration, get_scheduler

from forklift.par_data import DP
from forklift.ir_checker import validate_ir_syntax

logger = logging.getLogger(__name__)


# ======================================================================
# Configuration
# ======================================================================

@dataclass
class DPOConfig:
    """Configuration for DPO training of seq2seq models."""

    # ── Model ────────────────────────────────────────────────────────
    model_path: str = "leachl/forklift-arm-ir-ir"
    pair: str = "arm_ir-ir"

    # ── DPO dataset ──────────────────────────────────────────────────
    dpo_dataset: str = "dpo_pairs/"       # path to saved dataset or HF repo
    dpo_hub: bool = False                  # True if dpo_dataset is a HF Hub repo
    max_source_len: int = 1024
    max_target_len: int = 1024

    # ── DPO hyperparameters ──────────────────────────────────────────
    beta: float = 0.1                      # KL penalty strength
    label_smoothing: float = 0.0           # for robust DPO
    loss_type: str = "sigmoid"             # "sigmoid" or "ipo"

    # ── Optimiser ────────────────────────────────────────────────────
    lr: float = 5e-7                       # DPO uses much lower LR
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    max_grad_norm: float = 1.0

    # ── Schedule ─────────────────────────────────────────────────────
    warmup_steps: int = 100
    lr_scheduler: str = "cosine"

    # ── Training loop ────────────────────────────────────────────────
    max_steps: int = 5000
    batch_size: int = 2                    # per-device (DPO doubles memory)
    gradient_accumulation_steps: int = 8
    fp16: bool = True
    bf16: bool = False

    # ── Evaluation & checkpointing ───────────────────────────────────
    eval_steps: int = 500
    eval_samples: int = 200
    save_steps: int = 500
    save_total_limit: int = 3
    log_steps: int = 25
    checkpoint_dir: str = "checkpoints/dpo"
    resume_from: Optional[str] = None

    # ── Inference at eval time ───────────────────────────────────────
    eval_beam: int = 3
    eval_max_new_tokens: int = 2048
    repetition_penalty: float = 1.2
    no_repeat_ngram_size: int = 6

    # ── Hardware ─────────────────────────────────────────────────────
    device: str = "auto"
    num_workers: int = 0
    seed: int = 42

    # ── Logging ──────────────────────────────────────────────────────
    use_tensorboard: bool = True
    tensorboard_dir: str = "runs/dpo"

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps


# ======================================================================
# Dataset
# ======================================================================

class DPODataset(torch.utils.data.Dataset):
    """
    Map-style dataset for DPO preference pairs.

    Each item returns:
        prompt_ids     : tokenized source (ARM assembly)
        chosen_ids     : tokenized chosen IR
        rejected_ids   : tokenized rejected IR
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        config: DPOConfig,
        split: str = "train",
    ):
        self.config = config
        self.dp = DP(tokenizer=tokenizer)
        self.tokenizer = tokenizer

        # Load dataset
        if config.dpo_hub:
            self.data = load_dataset(config.dpo_dataset, split=split)
        else:
            self.data = load_from_disk(config.dpo_dataset)
            if isinstance(self.data, dict):
                self.data = self.data[split]

        logger.info("DPODataset: loaded %d samples from %s",
                     len(self.data), config.dpo_dataset)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Optional[dict]:
        row = self.data[idx]

        prompt = row["prompt"]
        chosen = row["chosen"]
        rejected = row["rejected"]

        # Tokenize with the Forklift tokenizer
        try:
            prompt_ids = self._tokenize(prompt)
            chosen_ids = self._tokenize(chosen)
            rejected_ids = self._tokenize(rejected)
        except Exception:
            return None

        if (prompt_ids is None or chosen_ids is None or
                rejected_ids is None):
            return None

        # Length filter
        if len(prompt_ids) > self.config.max_source_len:
            return None
        if (len(chosen_ids) > self.config.max_target_len or
                len(rejected_ids) > self.config.max_target_len):
            return None

        return {
            "prompt_ids": torch.tensor(prompt_ids, dtype=torch.long),
            "chosen_ids": torch.tensor(chosen_ids, dtype=torch.long),
            "rejected_ids": torch.tensor(rejected_ids, dtype=torch.long),
        }

    def _tokenize(self, text: str) -> Optional[list[int]]:
        """Tokenize raw text to IDs using the Forklift tokenizer."""
        enc = self.tokenizer.encode(text)
        if enc is None:
            return None
        return enc.ids


def dpo_collate_fn(
    batch: list[Optional[dict]],
    pad_id: int = 1,
    label_pad_id: int = -100,
) -> Optional[dict]:
    """Collate DPO samples into padded batches.

    Returns a dict with:
        prompt_ids, prompt_mask   – encoder inputs
        chosen_ids, chosen_mask   – decoder labels for chosen
        rejected_ids, rejected_mask – decoder labels for rejected
        chosen_decoder_input_ids  – teacher-forced decoder input (chosen)
        rejected_decoder_input_ids – teacher-forced decoder input (rejected)
    """
    # Filter None samples
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    prompt_ids = [b["prompt_ids"] for b in batch]
    chosen_ids = [b["chosen_ids"] for b in batch]
    rejected_ids = [b["rejected_ids"] for b in batch]

    prompt_padded = pad_sequence(prompt_ids, batch_first=True, padding_value=pad_id)
    chosen_padded = pad_sequence(chosen_ids, batch_first=True, padding_value=label_pad_id)
    rejected_padded = pad_sequence(rejected_ids, batch_first=True, padding_value=label_pad_id)

    prompt_mask = (prompt_padded != pad_id).long()

    # BART decoder_input_ids = right-shift labels, prepend <s>
    bos_id = 0
    chosen_dec = chosen_padded.new_full(chosen_padded.shape, pad_id)
    chosen_dec[:, 0] = bos_id
    chosen_dec[:, 1:] = chosen_padded[:, :-1].clone()
    chosen_dec[chosen_dec == label_pad_id] = pad_id

    rejected_dec = rejected_padded.new_full(rejected_padded.shape, pad_id)
    rejected_dec[:, 0] = bos_id
    rejected_dec[:, 1:] = rejected_padded[:, :-1].clone()
    rejected_dec[rejected_dec == label_pad_id] = pad_id

    return {
        "prompt_ids": prompt_padded,
        "prompt_mask": prompt_mask,
        "chosen_labels": chosen_padded,
        "rejected_labels": rejected_padded,
        "chosen_decoder_input_ids": chosen_dec,
        "rejected_decoder_input_ids": rejected_dec,
    }


# ======================================================================
# DPO loss
# ======================================================================

def compute_seq2seq_logps(
    model: BartForConditionalGeneration,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    decoder_input_ids: torch.Tensor,
    labels: torch.Tensor,
    label_pad_id: int = -100,
) -> torch.Tensor:
    """Compute per-sequence average log-probabilities for a seq2seq model.

    Returns a tensor of shape (batch_size,) with the average log-prob
    over non-padding tokens in each sequence.
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
    )
    logits = outputs.logits  # (B, T, V)

    # Per-token log-probs
    log_probs = F.log_softmax(logits, dim=-1)

    # Gather log-probs for the actual target tokens
    # labels has -100 for padding — mask those out
    mask = (labels != label_pad_id).float()
    # Clamp labels to valid range (replace -100 with 0 for gathering)
    safe_labels = labels.clamp(min=0)
    token_log_probs = log_probs.gather(2, safe_labels.unsqueeze(-1)).squeeze(-1)
    token_log_probs = token_log_probs * mask

    # Average over non-padding tokens per sequence
    seq_log_probs = token_log_probs.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)

    return seq_log_probs


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
    loss_type: str = "sigmoid",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute DPO loss.

    Returns (loss, chosen_reward, rejected_reward).
    """
    # Log-ratio differences
    chosen_logratios = policy_chosen_logps - ref_chosen_logps
    rejected_logratios = policy_rejected_logps - ref_rejected_logps
    logits = beta * (chosen_logratios - rejected_logratios)

    if loss_type == "sigmoid":
        loss = -F.logsigmoid(logits * (1 - 2 * label_smoothing))
    elif loss_type == "ipo":
        loss = (logits - 1 / (2 * beta)) ** 2
    else:
        raise ValueError(f"Unknown DPO loss type: {loss_type}")

    chosen_rewards = beta * chosen_logratios.detach()
    rejected_rewards = beta * rejected_logratios.detach()

    return loss.mean(), chosen_rewards.mean(), rejected_rewards.mean()


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


def save_checkpoint(
    model: BartForConditionalGeneration,
    tokenizer: Tokenizer,
    optimizer: AdamW,
    step: int,
    config: DPOConfig,
    metrics: dict,
):
    ckpt_dir = Path(config.checkpoint_dir) / f"step_{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(ckpt_dir))
    tokenizer.save(str(ckpt_dir / "tokenizer.json"))

    torch.save(
        {"optimizer": optimizer.state_dict(), "step": step,
         "config": {k: v for k, v in vars(config).items()
                    if not k.startswith("_")},
         "metrics": metrics},
        str(ckpt_dir / "training_state.pt"),
    )
    with open(ckpt_dir / "metrics.json", "w") as f:
        json.dump({"step": step, **metrics}, f, indent=2)

    logger.info("Saved checkpoint at step %d → %s", step, ckpt_dir)

    # Prune old checkpoints
    parent = Path(config.checkpoint_dir)
    ckpts = sorted(parent.glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
    while len(ckpts) > config.save_total_limit:
        old = ckpts.pop(0)
        import shutil
        shutil.rmtree(old)
        logger.info("Pruned old checkpoint: %s", old)


# ======================================================================
# Evaluation
# ======================================================================

@torch.no_grad()
def evaluate_dpo(
    policy: BartForConditionalGeneration,
    ref_model: BartForConditionalGeneration,
    val_loader: DataLoader,
    config: DPOConfig,
    device: torch.device,
    dp: DP,
    pad_id: int,
) -> dict:
    """Evaluate DPO model: compute reward accuracy & syntax validity."""
    policy.eval()

    total_reward_acc = 0
    n_samples = 0

    for batch in val_loader:
        if n_samples >= config.eval_samples:
            break
        if batch is None:
            continue

        batch = {k: v.to(device) for k, v in batch.items()}

        # Policy log-probs
        policy_chosen_logps = compute_seq2seq_logps(
            policy, batch["prompt_ids"], batch["prompt_mask"],
            batch["chosen_decoder_input_ids"], batch["chosen_labels"],
        )
        policy_rejected_logps = compute_seq2seq_logps(
            policy, batch["prompt_ids"], batch["prompt_mask"],
            batch["rejected_decoder_input_ids"], batch["rejected_labels"],
        )

        # Reference log-probs
        ref_chosen_logps = compute_seq2seq_logps(
            ref_model, batch["prompt_ids"], batch["prompt_mask"],
            batch["chosen_decoder_input_ids"], batch["chosen_labels"],
        )
        ref_rejected_logps = compute_seq2seq_logps(
            ref_model, batch["prompt_ids"], batch["prompt_mask"],
            batch["rejected_decoder_input_ids"], batch["rejected_labels"],
        )

        # Reward accuracy: does the policy prefer chosen over rejected?
        chosen_rewards = config.beta * (policy_chosen_logps - ref_chosen_logps)
        rejected_rewards = config.beta * (policy_rejected_logps - ref_rejected_logps)
        total_reward_acc += (chosen_rewards > rejected_rewards).float().sum().item()
        n_samples += batch["prompt_ids"].size(0)

    # Quick syntax validity check on a few generations
    syntax_ok = 0
    syntax_total = 0
    n_gen = min(50, config.eval_samples)

    # Use the first batch for generation
    for batch in val_loader:
        if batch is None:
            continue
        batch = {k: v.to(device) for k, v in batch.items()}
        gen_ids = policy.generate(
            input_ids=batch["prompt_ids"],
            attention_mask=batch["prompt_mask"],
            max_new_tokens=config.eval_max_new_tokens,
            num_beams=config.eval_beam,
            early_stopping=True,
            repetition_penalty=config.repetition_penalty,
            no_repeat_ngram_size=config.no_repeat_ngram_size,
        )
        for g in gen_ids:
            text = dp.detokenize(g.cpu().tolist())
            ok, _ = validate_ir_syntax(text, auto_declare=True)
            if ok:
                syntax_ok += 1
            syntax_total += 1
            if syntax_total >= n_gen:
                break
        if syntax_total >= n_gen:
            break

    policy.train()

    reward_acc = 100.0 * total_reward_acc / max(n_samples, 1)
    syntax_pct = 100.0 * syntax_ok / max(syntax_total, 1)

    return {
        "reward_accuracy": round(reward_acc, 2),
        "syntax_valid_pct": round(syntax_pct, 2),
        "num_samples": n_samples,
    }


# ======================================================================
# Main training loop
# ======================================================================

def train(config: DPOConfig):
    """Run DPO fine-tuning."""
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

    # ── Load policy model ────────────────────────────────────────────
    model_load_path = config.model_path
    if config.resume_from:
        ckpt = Path(config.resume_from)
        if (ckpt / "model.safetensors").exists() or (ckpt / "pytorch_model.bin").exists():
            model_load_path = str(ckpt)
            logger.info("Resuming policy from checkpoint: %s", model_load_path)

    logger.info("Loading policy model from %s", model_load_path)
    policy = BartForConditionalGeneration.from_pretrained(model_load_path)
    policy.to(device)
    policy.train()

    # ── Load reference model (frozen copy of initial policy) ─────────
    logger.info("Loading reference model from %s", config.model_path)
    ref_model = BartForConditionalGeneration.from_pretrained(config.model_path)
    ref_model.to(device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    logger.info("Policy parameters: %d (%.1fM trainable)", n_params, n_params / 1e6)

    # ── Load DPO dataset ─────────────────────────────────────────────
    train_ds = DPODataset(tokenizer, config, split="train")
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda b: dpo_collate_fn(b, pad_id=pad_id),
        num_workers=config.num_workers,
        drop_last=True,
    )

    # ── Optimizer ────────────────────────────────────────────────────
    no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight"}
    param_groups = [
        {"params": [p for n, p in policy.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         "weight_decay": config.weight_decay},
        {"params": [p for n, p in policy.named_parameters()
                    if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(param_groups, lr=config.lr,
                      betas=(config.adam_beta1, config.adam_beta2),
                      eps=config.adam_eps)

    scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.max_steps,
    )

    # ── Mixed precision ──────────────────────────────────────────────
    use_amp = (config.fp16 or config.bf16) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if config.bf16 else torch.float16
    scaler = GradScaler("cuda", enabled=use_amp and not config.bf16)

    # ── Resume ───────────────────────────────────────────────────────
    global_step = 0
    if config.resume_from:
        ckpt = Path(config.resume_from)
        state_file = ckpt / "training_state.pt"
        if state_file.exists():
            state = torch.load(state_file, map_location=device)
            optimizer.load_state_dict(state["optimizer"])
            global_step = state.get("step", 0)
            logger.info("Resumed from step %d", global_step)

    # ── TensorBoard ──────────────────────────────────────────────────
    tb_writer = None
    if config.use_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_writer = SummaryWriter(log_dir=config.tensorboard_dir)
            logger.info("TensorBoard logging to %s", config.tensorboard_dir)
        except ImportError:
            logger.warning("tensorboard not installed")

    # ── Training ─────────────────────────────────────────────────────
    logger.info(
        "Starting DPO training: max_steps=%d  eff_batch=%d  lr=%.2e  β=%.2f",
        config.max_steps, config.effective_batch_size, config.lr, config.beta,
    )

    policy.zero_grad()
    running_loss = 0.0
    running_chosen_reward = 0.0
    running_rejected_reward = 0.0
    accum_count = 0
    epoch = 0
    t_start = time.time()

    while global_step < config.max_steps:
        epoch += 1
        logger.info("Starting epoch %d at step %d", epoch, global_step)

        for batch in train_loader:
            if global_step >= config.max_steps:
                break
            if batch is None:
                continue

            batch = {k: v.to(device) for k, v in batch.items()}

            with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                # ── Policy log-probs ─────────────────────────────────
                policy_chosen_logps = compute_seq2seq_logps(
                    policy, batch["prompt_ids"], batch["prompt_mask"],
                    batch["chosen_decoder_input_ids"], batch["chosen_labels"],
                )
                policy_rejected_logps = compute_seq2seq_logps(
                    policy, batch["prompt_ids"], batch["prompt_mask"],
                    batch["rejected_decoder_input_ids"], batch["rejected_labels"],
                )

                # ── Reference log-probs (frozen, no grad) ────────────
                with torch.no_grad():
                    ref_chosen_logps = compute_seq2seq_logps(
                        ref_model, batch["prompt_ids"], batch["prompt_mask"],
                        batch["chosen_decoder_input_ids"], batch["chosen_labels"],
                    )
                    ref_rejected_logps = compute_seq2seq_logps(
                        ref_model, batch["prompt_ids"], batch["prompt_mask"],
                        batch["rejected_decoder_input_ids"], batch["rejected_labels"],
                    )

                # ── DPO loss ─────────────────────────────────────────
                loss, chosen_reward, rejected_reward = dpo_loss(
                    policy_chosen_logps, policy_rejected_logps,
                    ref_chosen_logps, ref_rejected_logps,
                    beta=config.beta,
                    label_smoothing=config.label_smoothing,
                    loss_type=config.loss_type,
                )
                loss = loss / config.gradient_accumulation_steps

            if use_amp and not config.bf16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += loss.item()
            running_chosen_reward += chosen_reward.item()
            running_rejected_reward += rejected_reward.item()
            accum_count += 1

            # Gradient accumulation step
            if accum_count % config.gradient_accumulation_steps == 0:
                if use_amp and not config.bf16:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
                    optimizer.step()

                scheduler.step()
                policy.zero_grad()
                global_step += 1

                # ── Logging ──────────────────────────────────────────
                if global_step % config.log_steps == 0:
                    n_accum = config.gradient_accumulation_steps
                    avg_loss = running_loss / n_accum
                    avg_chosen = running_chosen_reward / n_accum
                    avg_rejected = running_rejected_reward / n_accum
                    margin = avg_chosen - avg_rejected
                    lr_now = scheduler.get_last_lr()[0]
                    elapsed = time.time() - t_start
                    steps_s = global_step / elapsed if elapsed > 0 else 0

                    logger.info(
                        "step=%d  loss=%.4f  margin=%.4f  "
                        "rew_c=%.3f  rew_r=%.3f  lr=%.2e  s/s=%.2f",
                        global_step, avg_loss, margin,
                        avg_chosen, avg_rejected, lr_now, steps_s,
                    )
                    if tb_writer:
                        tb_writer.add_scalar("dpo/loss", avg_loss, global_step)
                        tb_writer.add_scalar("dpo/reward_chosen", avg_chosen, global_step)
                        tb_writer.add_scalar("dpo/reward_rejected", avg_rejected, global_step)
                        tb_writer.add_scalar("dpo/reward_margin", margin, global_step)
                        tb_writer.add_scalar("dpo/lr", lr_now, global_step)

                    running_loss = 0.0
                    running_chosen_reward = 0.0
                    running_rejected_reward = 0.0

                # ── Evaluation ───────────────────────────────────────
                if global_step % config.eval_steps == 0:
                    logger.info("Running evaluation at step %d …", global_step)
                    metrics = evaluate_dpo(
                        policy, ref_model, train_loader,
                        config, device, dp, pad_id,
                    )
                    logger.info(
                        "EVAL step=%d  reward_acc=%.1f%%  syntax_valid=%.1f%%",
                        global_step,
                        metrics["reward_accuracy"],
                        metrics["syntax_valid_pct"],
                    )
                    if tb_writer:
                        tb_writer.add_scalar(
                            "eval/reward_accuracy",
                            metrics["reward_accuracy"], global_step,
                        )
                        tb_writer.add_scalar(
                            "eval/syntax_valid_pct",
                            metrics["syntax_valid_pct"], global_step,
                        )
                    policy.train()

                # ── Save checkpoint ──────────────────────────────────
                if global_step % config.save_steps == 0:
                    save_checkpoint(
                        policy, tokenizer, optimizer,
                        global_step, config,
                        {"step": global_step},
                    )

    # ── Final save ───────────────────────────────────────────────────
    logger.info("DPO training complete at step %d", global_step)
    save_checkpoint(
        policy, tokenizer, optimizer,
        global_step, config,
        {"step": global_step, "final": True},
    )

    if tb_writer:
        tb_writer.close()

    return policy


# ======================================================================
# CLI
# ======================================================================

def parse_args() -> DPOConfig:
    parser = argparse.ArgumentParser(
        description="DPO fine-tuning for Forklift seq2seq model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    cfg = DPOConfig()

    for name, default in vars(cfg).items():
        if name.startswith("_"):
            continue
        arg_type = type(default) if default is not None else str
        if arg_type is bool:
            group = parser.add_mutually_exclusive_group()
            group.add_argument(f"--{name}", dest=name, action="store_true", default=default)
            group.add_argument(f"--no_{name}", dest=name, action="store_false")
        else:
            parser.add_argument(f"--{name}", type=arg_type, default=default)

    args = parser.parse_args()
    return DPOConfig(**{k: v for k, v in vars(args).items() if hasattr(cfg, k)})


def main():
    config = parse_args()
    train(config)


if __name__ == "__main__":
    main()
