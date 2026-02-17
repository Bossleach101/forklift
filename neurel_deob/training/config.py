"""
Fine-tuning configuration for AArch64 → LLVM IR.

All hyperparameters live here so they can be easily tweaked from the
CLI, a YAML file, or a SLURM launcher script.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TrainConfig:
    """Full training configuration."""

    # ── Model ────────────────────────────────────────────────────────
    model_path: str = "jordiae/clang_opt3_ir_optz-ir_optz-2024-01-15-0959-e1bf-bc2b"
    pair: str = "arm_ir-ir"

    # ── Data ─────────────────────────────────────────────────────────
    train_split: str = "train_synth_compilable"
    valid_split: str = "valid_synth"
    hf_dataset: str = "jordiae/exebench"
    hf_revision: str = "clang"
    asm_key: str = "angha"
    max_source_len: int = 1024
    max_target_len: int = 1024
    normalize_ir_structs: bool = True
    strip_ir_declares: bool = True   # Remove declare/attributes/metadata from IR targets
    streaming: bool = True

    # ── Optimiser ────────────────────────────────────────────────────
    lr: float = 5e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    max_grad_norm: float = 1.0

    # ── Schedule ─────────────────────────────────────────────────────
    warmup_steps: int = 500
    lr_scheduler: str = "linear"  # "linear", "cosine", "constant"

    # ── Training loop ────────────────────────────────────────────────
    max_steps: int = 100_000
    batch_size: int = 4
    gradient_accumulation_steps: int = 8  # effective batch = 4 * 8 = 32
    fp16: bool = True                     # mixed precision (A100/V100)
    bf16: bool = False                    # bfloat16 (A100)

    # ── Evaluation & checkpointing ───────────────────────────────────
    eval_steps: int = 2000
    eval_samples: int = 500              # how many validation samples per eval
    save_steps: int = 2000
    save_total_limit: int = 5            # keep only N best checkpoints
    log_steps: int = 50
    checkpoint_dir: str = "checkpoints/arm_ir_ir"
    resume_from: Optional[str] = None    # path to a checkpoint dir

    # ── Inference at eval time ───────────────────────────────────────
    eval_beam: int = 5
    eval_max_new_tokens: int = 2048
    repetition_penalty: float = 1.2      # Penalise repeated tokens during generation
    no_repeat_ngram_size: int = 6        # Forbid repeating any 6-gram

    # ── Hardware ─────────────────────────────────────────────────────
    device: str = "auto"                 # "auto", "cpu", "cuda", "cuda:0"
    num_workers: int = 0                 # DataLoader workers (0 for streaming)
    seed: int = 42

    # ── Logging ──────────────────────────────────────────────────────
    project_name: str = "forklift-arm-ir"
    run_name: Optional[str] = None
    use_tensorboard: bool = True
    tensorboard_dir: str = "runs/arm_ir_ir"

    # ── Obfuscation (Phase 4: deobfuscation training) ────────────────
    obfuscate: bool = False              # Apply asm-level obfuscation to source
    obfu_techniques: List[str] = field(
        default_factory=lambda: ["dead_code", "insn_sub", "opaque_pred", "junk_comp"]
    )
    obfu_intensity_min: float = 0.1      # Min per-transform intensity
    obfu_intensity_max: float = 0.4      # Max per-transform intensity
    obfu_min_transforms: int = 1         # Min transforms per sample
    obfu_max_transforms: int = 4         # Max transforms per sample

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps
