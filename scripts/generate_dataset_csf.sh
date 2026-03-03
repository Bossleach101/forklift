#!/bin/bash
#SBATCH --job-name=obfu-dataset
#SBATCH -p multicore
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH -t 2-0
#SBATCH --array=0-99
#SBATCH --output=logs/obfu_dataset_%A_%a.out
#SBATCH --error=logs/obfu_dataset_%A_%a.err

# ──────────────────────────────────────────────────────────────
# SLURM job array for large-scale obfuscation dataset generation
#
# Splits ExeBench into chunks across array tasks.  Each task
# processes a slice of the dataset using multiprocessing within
# the node (--cpus-per-task workers).
#
# Two modes:
#   SPLIT=train_synth_compilable  → ~5.9M rows, 100 chunks of 59K each
#   SPLIT=train_real_compilable   → ~1.4M rows, 100 chunks of 14K each
#
# Usage:
#   # Generate synth split (default)
#   sbatch scripts/generate_dataset_csf.sh
#
#   # Generate real split
#   SPLIT=train_real_compilable sbatch scripts/generate_dataset_csf.sh
#
#   # Override chunk size or array size
#   sbatch --array=0-49 scripts/generate_dataset_csf.sh
#
# Output:
#   Each array task writes Parquet shards to:
#     $OUTPUT_BASE/<split>/chunk_<TASK_ID>/
#   After all tasks complete, run scripts/merge_dataset.py to
#   combine shards and upload to HuggingFace Hub.
# ──────────────────────────────────────────────────────────────

set -euo pipefail

# ── Configuration ────────────────────────────────────────────
PROJECT_DIR="${PROJECT_DIR:-$HOME/scratch/forklift}"
VENV_DIR="${PROJECT_DIR}/env"
SPLIT="${SPLIT:-train_synth_compilable}"
OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_DIR}/obfu_dataset}"

# Chunk sizing: total rows / number of array tasks
# These are approximate — we round up to ensure full coverage.
# synth_compilable: ~5,900,000 rows
# real_compilable:  ~1,400,000 rows
if [ "$SPLIT" = "train_synth_compilable" ]; then
    TOTAL_ROWS=5900000
elif [ "$SPLIT" = "train_real_compilable" ]; then
    TOTAL_ROWS=1400000
else
    echo "ERROR: Unknown split '$SPLIT'"
    exit 1
fi

# Compute chunk boundaries for this task
NUM_TASKS=${SLURM_ARRAY_TASK_COUNT:-100}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
CHUNK_SIZE=$(( (TOTAL_ROWS + NUM_TASKS - 1) / NUM_TASKS ))
SKIP_ROWS=$(( TASK_ID * CHUNK_SIZE ))

# Workers = number of CPUs allocated to this task
NUM_WORKERS=${SLURM_CPUS_PER_TASK:-12}

# Output directory for this chunk
CHUNK_OUTPUT="${OUTPUT_BASE}/${SPLIT}/chunk_${TASK_ID}"

echo "================================================"
echo "Job Array ID:  $SLURM_ARRAY_JOB_ID"
echo "Task ID:       $TASK_ID / $NUM_TASKS"
echo "Node:          $(hostname)"
echo "CPUs:          $NUM_WORKERS"
echo "Split:         $SPLIT"
echo "Skip rows:     $SKIP_ROWS"
echo "Chunk size:    $CHUNK_SIZE rows"
echo "Output:        $CHUNK_OUTPUT"
echo "Date:          $(date)"
echo "================================================"

# ── Environment ──────────────────────────────────────────────
cd "$PROJECT_DIR"
source "${VENV_DIR}/bin/activate"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

# Tigress
export TIGRESS_HOME="${TIGRESS_HOME:-$HOME/scratch/tigress-4.0.11}"

# Create directories
mkdir -p logs "$CHUNK_OUTPUT"

# ── Run generation ───────────────────────────────────────────
python scripts/generate_obfu_dataset.py \
    --output-dir "$CHUNK_OUTPUT" \
    --split "$SPLIT" \
    --skip-rows "$SKIP_ROWS" \
    --max-rows "$CHUNK_SIZE" \
    --chunk-id "$TASK_ID" \
    --num-workers "$NUM_WORKERS" \
    --worker-queue-size $(( NUM_WORKERS * 20 )) \
    --shard-size 5000 \
    --max-failures 100000 \
    --tigress-timeout 120 \
    --log-file "logs/obfu_dataset_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.log" \
    "$@"

echo ""
echo "Task $TASK_ID finished at $(date)"
echo "Output: $CHUNK_OUTPUT"
ls -lh "$CHUNK_OUTPUT"/*.parquet 2>/dev/null | wc -l | xargs -I{} echo "Shards written: {}"
