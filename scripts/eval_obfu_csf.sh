#!/bin/bash
#SBATCH --job-name=forklift-obfu-eval
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 0-24:00:00
#SBATCH --output=logs/obfu_eval_%j.out
#SBATCH --error=logs/obfu_eval_%j.err

# ──────────────────────────────────────────────────────────────
# Evaluate Forklift model on obfuscated-exebench test set
#
# Task: obfuscated AArch64 assembly → clean LLVM IR
#
# Usage:
#   sbatch scripts/eval_obfu_csf.sh
#   sbatch scripts/eval_obfu_csf.sh --technique Flatten
#   sbatch scripts/eval_obfu_csf.sh --max-samples 500
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
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

# Check for LLVM in local tools folder first
LLVM_BIN="$PROJECT_DIR/tools/clang+llvm-14.0.0-x86_64-linux-gnu-ubuntu-18.04/bin"
if [ -d "$LLVM_BIN" ]; then
    echo "Using local LLVM installation: $LLVM_BIN"
    export PATH="$LLVM_BIN:$PATH"
fi

export PATH="$HOME/scratch/forklift/tools/clang+llvm-14.0.0-x86_64-linux-gnu-ubuntu-18.04/bin:$PATH"

mkdir -p logs results

# ── Default config ───────────────────────────────────────────
MODEL="${MODEL:-checkpoints/arm_ir_ir_v4/step_50000}"
PAIR="${PAIR:-arm_ir-ir}"
SPLIT="${SPLIT:-test}"
BATCH_SIZE="${BATCH_SIZE:-8}"

# ── Run evaluation ───────────────────────────────────────────
echo "Evaluating model: $MODEL"
echo "Pair: $PAIR  Split: $SPLIT  Batch size: $BATCH_SIZE"
echo ""

python scripts/evaluate_obfu.py \
    --model "$MODEL" \
    --pair "$PAIR" \
    --split "$SPLIT" \
    --beam 3 \
    --repetition-penalty 1.0 \
    --no-repeat-ngram-size 0 \
    --max-new-tokens 2048 \
    --batch-size "$BATCH_SIZE" \
    --strip-ir \
    --check-functional \
    --normalize-structs \
    --output "results/obfu_eval_${SPLIT}_$(echo $MODEL | tr '/' '_').json" \
    --save-predictions "results/obfu_preds_${SPLIT}_$(echo $MODEL | tr '/' '_').jsonl" \
    --device auto \
    "$@"

echo ""
echo "================================================"
echo "Evaluation complete at $(date)"
echo "================================================"
