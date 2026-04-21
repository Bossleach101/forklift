#!/bin/bash
#SBATCH --job-name=forklift-eval
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 0-24:00:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

# ──────────────────────────────────────────────────────────────
# SLURM launcher for Forklift evaluation on ExeBench test split
#
# Evaluates the fine-tuned ARM→IR v4 model on test_synth
# (5,000 samples) and optionally the baseline model.
#
# Usage:
#   sbatch scripts/eval_csf.sh                           # v4 model only
#   sbatch scripts/eval_csf.sh --split test_real          # override split
#   sbatch scripts/eval_csf.sh --model jordiae/clang_...  # evaluate baseline
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
MODEL="${MODEL:-leachl/forklift-arm-ir-ir_v4}"
PAIR="${PAIR:-arm_ir-ir}"
SPLIT="${SPLIT:-test_synth}"
ASM_KEY="${ASM_KEY:-angha}"

BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_SAMPLES="${MAX_SAMPLES:-5000}"

# ── Run evaluation ───────────────────────────────────────────
echo "Evaluating model: $MODEL"
echo "Pair: $PAIR  Split: $SPLIT  ASM key: $ASM_KEY  Batch size: $BATCH_SIZE"
echo ""

python scripts/evaluate_exebench.py \
    --model "$MODEL" \
    --pair "$PAIR" \
    --split "$SPLIT" \
    --asm-key "$ASM_KEY" \
    --beam 3 \
    --repetition-penalty 1.0 \
    --no-repeat-ngram-size 0 \
    --max-new-tokens 2048 \
    --batch-size "$BATCH_SIZE" \
    --max-samples "$MAX_SAMPLES" \
    --strip-ir \
    --normalize-structs \
    --check-functional \
    --output "results/eval_${SPLIT}_$(echo $MODEL | tr '/' '_').json" \
    --save-predictions "results/preds_${SPLIT}_$(echo $MODEL | tr '/' '_').jsonl" \
    --device auto \
    "$@"

echo ""
echo "================================================"
echo "Evaluation complete at $(date)"
echo "================================================"
