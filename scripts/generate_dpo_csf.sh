#!/bin/bash
#SBATCH --job-name=forklift-dpo-gen
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 2-0
#SBATCH --output=logs/dpo_gen_%j.out
#SBATCH --error=logs/dpo_gen_%j.err

# ──────────────────────────────────────────────────────────────
# SLURM launcher: Generate DPO preference pairs with compiler
# feedback from the Forklift model.
#
# Stage 1 of the DPO pipeline:
#   1. generate_dpo_pairs.py  →  preference dataset
#   2. dpo_finetune.py        →  DPO-trained model
#
# Usage:
#   sbatch scripts/generate_dpo_csf.sh
#   sbatch scripts/generate_dpo_csf.sh --max-samples 10000
# ──────────────────────────────────────────────────────────────

set -euo pipefail

echo "Starting DPO pair generation at $(date)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Host: $(hostname)"

# ── Environment ──────────────────────────────────────────────
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs

# Activate virtual environment if present
if [ -f "env/bin/activate" ]; then
    source env/bin/activate
fi

export PATH="$HOME/scratch/forklift/tools/clang+llvm-14.0.0-x86_64-linux-gnu-ubuntu-18.04/bin:$PATH"

# ── Model to generate from ──────────────────────────────────
MODEL_PATH="${MODEL_PATH:checkpoints/arm_ir_ir_v4/step_80000}"

# ── Generate DPO pairs ──────────────────────────────────────
# Strategy: use ground-truth as chosen, model output as rejected.
# This is the safest approach when the model produces 0% valid IR.
python scripts/generate_dpo_pairs.py \
    --model "$MODEL_PATH" \
    --split "train_synth_compilable" \
    --num-candidates 3 \
    --gt-as-chosen \
    --max-samples 5000 \
    --max-new-tokens 2048 \
    --repetition-penalty 1.0 \
    --no-repeat-ngram-size 0 \
    --output "dpo_pairs/" \
    "$@"

echo "DPO pair generation finished at $(date)"
