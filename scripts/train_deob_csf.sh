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
# Fine-tunes the AArch64 → LLVM IR model on (obfuscated_asm → clean_ir)
# pairs from the Tigress-obfuscated ExeBench dataset.
#
# Dataset: leachl/obfuscated-exebench
#   - train  (980K samples: Flatten, EncodeArithmetic, Flatten+EncodeArithmetic)
#   - test_flatten, test_encode_arithmetic, test_combined  (~3.9K each)
#
# The model starts from the v4 arm-ir-ir checkpoint (clean lifting),
# then learns to "see through" Tigress obfuscation while lifting.
#
# Validation is run on test_combined by default (change with --valid_split).
#
# Usage:
#   sbatch scripts/train_deob_csf.sh
#   sbatch scripts/train_deob_csf.sh --max_steps 50000
#   sbatch scripts/train_deob_csf.sh --valid_split test_flatten
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

# Check for LLVM in local tools folder first
LLVM_BIN="$PROJECT_DIR/tools/clang+llvm-14.0.0-x86_64-linux-gnu-ubuntu-18.04/bin"
if [ -d "$LLVM_BIN" ]; then
    echo "Using local LLVM installation: $LLVM_BIN"
    export PATH="$LLVM_BIN:$PATH"
fi

export PATH="$HOME/scratch/forklift/tools/clang+llvm-14.0.0-x86_64-linux-gnu-ubuntu-18.04/bin:$PATH"

mkdir -p logs

# ── Model to fine-tune from ─────────────────────────────────
# Start from the v4 arm-ir-ir checkpoint (clean lifting model).
MODEL_PATH="${MODEL_PATH:-leachl/forklift-arm-ir-ir_v4}"

# ── Launch deobfuscation training ────────────────────────────
python -m neurel_deob.training.finetune \
    --model_path "$MODEL_PATH" \
    --pair "arm_ir-ir" \
    --deob_dataset "leachl/obfuscated-exebench" \
    --train_split "train" \
    --valid_split "test_combined" \
    --max_steps 100000 \
    --batch_size 8 \
    --gradient_accumulation_steps 4 \
    --lr 2e-5 \
    --lr_scheduler cosine \
    --warmup_steps 500 \
    --fp16 \
    --strip_ir_declares \
    --eval_steps 2000 \
    --save_steps 2000 \
    --eval_samples 500 \
    --max_source_len 1024 \
    --max_target_len 1024 \
    --checkpoint_dir "checkpoints/arm_deob" \
    --tensorboard_dir "runs/arm_deob" \
    --project_name "forklift-deob" \
    "$@"

echo "Deobfuscation training finished at $(date)"
