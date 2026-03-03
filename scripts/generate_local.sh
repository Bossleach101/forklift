#!/bin/bash
# ──────────────────────────────────────────────────────────────
# Local dataset generation runner
#
# Runs Tigress obfuscation locally using multiprocessing.
# Designed to run in tmux/screen for long-running generation.
#
# Usage:
#   # Start in tmux:
#   tmux new -s obfu
#   bash scripts/generate_local.sh
#
#   # Detach: Ctrl-B then D
#   # Reattach later: tmux attach -t obfu
#
#   # Or with nohup:
#   nohup bash scripts/generate_local.sh > logs/local_gen.out 2>&1 &
#
# Configuration:
#   Edit the variables below to control generation parameters.
# ──────────────────────────────────────────────────────────────

set -euo pipefail

# ── Configuration ────────────────────────────────────────────
PROJECT_DIR="/home/leachl/Project/Forklift/forklift"
OUTPUT_BASE="${PROJECT_DIR}/obfu_dataset"
NUM_WORKERS=16          # Use most of 20 cores, leave some for system
MAX_SAMPLES=""          # Empty = process all rows (set e.g. "500000" to cap)

# ── Environment ──────────────────────────────────────────────
cd "$PROJECT_DIR"
source env/bin/activate
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

mkdir -p logs

echo "================================================"
echo "Local Obfuscation Dataset Generation"
echo "================================================"
echo "Date:       $(date)"
echo "Workers:    $NUM_WORKERS"
echo "Output:     $OUTPUT_BASE"
echo "Max:        ${MAX_SAMPLES:-unlimited}"
echo "================================================"

# ── Helper ───────────────────────────────────────────────────
run_split() {
    local SPLIT="$1"
    local SPLIT_OUTPUT="${OUTPUT_BASE}/${SPLIT}"

    echo ""
    echo "──────────────────────────────────────────────"
    echo "Starting split: $SPLIT"
    echo "Output:         $SPLIT_OUTPUT"
    echo "Time:           $(date)"
    echo "──────────────────────────────────────────────"

    local ARGS=(
        --output-dir "$SPLIT_OUTPUT"
        --split "$SPLIT"
        --num-workers "$NUM_WORKERS"
        --worker-queue-size $(( NUM_WORKERS * 20 ))
        --shard-size 10000
        --max-failures 200000
        --tigress-timeout 120
        --log-file "logs/local_gen_${SPLIT}.log"
    )

    if [ -n "$MAX_SAMPLES" ]; then
        ARGS+=(--max-samples "$MAX_SAMPLES")
    fi

    python scripts/generate_obfu_dataset.py "${ARGS[@]}"

    echo ""
    echo "$SPLIT finished at $(date)"
    local SHARD_COUNT
    SHARD_COUNT=$(find "$SPLIT_OUTPUT" -name "*.parquet" 2>/dev/null | wc -l)
    echo "Shards written: $SHARD_COUNT"
}

# ── Run both splits ──────────────────────────────────────────
run_split "train_synth_compilable"
run_split "train_real_compilable"

echo ""
echo "================================================"
echo "All generation complete at $(date)"
echo "================================================"
echo ""
echo "Next steps:"
echo "  python scripts/merge_dataset.py \\"
echo "    -i ${OUTPUT_BASE}/train_synth_compilable \\"
echo "       ${OUTPUT_BASE}/train_real_compilable \\"
echo "    -o ${OUTPUT_BASE}_merged \\"
echo "    --upload-to leachl/obfuscated-exebench"
