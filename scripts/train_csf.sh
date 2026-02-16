#!/bin/bash
#SBATCH --job-name=forklift-arm-ir
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH --cpus-per-task=12
#SBATCH --mem=125G
#SBATCH -t 2-0
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# ──────────────────────────────────────────────────────────────
# SLURM launcher for Forklift AArch64 → LLVM IR fine-tuning
#
# Usage:
#   sbatch scripts/train_csf.sh                    # default config
#   sbatch scripts/train_csf.sh --max_steps 50000  # override
#
# All extra arguments after the script are forwarded to finetune.py.
# ──────────────────────────────────────────────────────────────

set -euo pipefail

echo "================================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date:      $(date)"
echo "================================================"

# ── Environment ──────────────────────────────────────────────
# Adjust this path to wherever your venv lives on CSF
PROJECT_DIR="$HOME/scratch/forklift"
VENV_DIR="${PROJECT_DIR}/env"

cd "$PROJECT_DIR"
source "${VENV_DIR}/bin/activate"

# Create log directory
mkdir -p logs

# ── Launch training ──────────────────────────────────────────
python -m neurel_deob.training.finetune \
    --model_path "jordiae/clang_opt3_ir_optz-ir_optz-2024-01-15-0959-e1bf-bc2b" \
    --pair "arm_ir-ir" \
    --max_steps 100000 \
    --batch_size 8 \
    --gradient_accumulation_steps 4 \
    --lr 5e-5 \
    --warmup_steps 500 \
    --fp16 \
    --eval_steps 2000 \
    --save_steps 2000 \
    --max_source_len 1024 \
    --max_target_len 1024 \
    --checkpoint_dir "checkpoints/arm_ir_ir" \
    --tensorboard_dir "runs/arm_ir_ir" \
    "$@"

echo "Training finished at $(date)"
