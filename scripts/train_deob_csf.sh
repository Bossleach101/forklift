#!/bin/bash
#SBATCH --job-name=forklift-deob
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH --cpus-per-task=12
#SBATCH --mem=125G
#SBATCH -t 4-0
#SBATCH --output=logs/train_deob_%j.out
#SBATCH --error=logs/train_deob_%j.err

# ──────────────────────────────────────────────────────────────
# SLURM launcher for Forklift deobfuscation fine-tuning
#
# Phase 4: Neural deobfuscation via lifting
#
# Fine-tunes the AArch64 → LLVM IR model on (obfuscated_asm → clean_ir)
# pairs.  Obfuscation is applied on-the-fly to clean ExeBench assembly
# using 4 synthetic transforms:
#   - dead_code:    dead instruction insertion (nops, identity ops)
#   - insn_sub:     instruction substitution (add→sub neg, mov→orr)
#   - opaque_pred:  opaque predicates (always-true/false branches)
#   - junk_comp:    junk computation sequences
#
# Each epoch sees different obfuscation patterns (data augmentation).
#
# The model starts from the v2 arm-ir-ir checkpoint (clean lifting),
# then learns to "see through" obfuscation while lifting.
#
# Usage:
#   sbatch scripts/train_deob_csf.sh
#   sbatch scripts/train_deob_csf.sh --max_steps 50000
#   sbatch scripts/train_deob_csf.sh --obfu_techniques dead_code insn_sub
#
# All extra arguments are forwarded to finetune.py.
# ──────────────────────────────────────────────────────────────

set -euo pipefail

echo "================================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date:      $(date)"
echo "================================================"

# ── Environment ──────────────────────────────────────────────
PROJECT_DIR="$HOME/scratch/forklift"
VENV_DIR="${PROJECT_DIR}/env"

cd "$PROJECT_DIR"
source "${VENV_DIR}/bin/activate"

mkdir -p logs

# ── Model to fine-tune from ─────────────────────────────────
# Start from the v2 arm-ir-ir checkpoint (clean lifting model).
# Change this to the best v2 checkpoint path once training completes.
MODEL_PATH="${MODEL_PATH:-leachl/forklift-arm-ir-ir}"

# ── Launch deobfuscation training ────────────────────────────
python -m neurel_deob.training.finetune \
    --model_path "$MODEL_PATH" \
    --pair "arm_ir-ir" \
    --max_steps 100000 \
    --batch_size 8 \
    --gradient_accumulation_steps 4 \
    --lr 2e-5 \
    --lr_scheduler cosine \
    --warmup_steps 500 \
    --fp16 \
    --strip_ir_declares \
    --obfuscate \
    --obfu_techniques dead_code insn_sub opaque_pred junk_comp \
    --obfu_intensity_min 0.1 \
    --obfu_intensity_max 0.4 \
    --obfu_min_transforms 1 \
    --obfu_max_transforms 4 \
    --eval_steps 2000 \
    --save_steps 2000 \
    --max_source_len 1024 \
    --max_target_len 1024 \
    --checkpoint_dir "checkpoints/arm_deob" \
    --tensorboard_dir "runs/arm_deob" \
    --project_name "forklift-deob" \
    "$@"

echo "Deobfuscation training finished at $(date)"
