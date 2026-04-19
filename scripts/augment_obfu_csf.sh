#!/bin/bash
#SBATCH --job-name=augment-obfu
#SBATCH -p multicore
#SBATCH -n=8
#SBATCH --mem=32G
#SBATCH -t 0-04:00:00
#SBATCH --output=logs/augment_%j.out
#SBATCH --error=logs/augment_%j.err



# ──────────────────────────────────────────────────────────────
# SLURM launcher for checking and augmenting the obfuscated dataset
# with missing ExeBench metadata columns.
#
# Usage:
#   sbatch scripts/augment_obfu_csf.sh
# ──────────────────────────────────────────────────────────────

set -euo pipefail

echo "================================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Date:      $(date)"
echo "================================================"

# ── Environment ──────────────────────────────────────────────
PROJECT_DIR="$HOME/scratch/forklift"
VENV_DIR="${PROJECT_DIR}/env"

cd "$PROJECT_DIR"
source "${VENV_DIR}/bin/activate"

mkdir -p logs

# ── Metadata Augmentation ────────────────────────────────────
# This will stream metadata from jordiae/exebench and merge it
# into leachl/obfuscated-exebench.
# CAUTION: This pushes changes to Hugging Face Hub directly.

python scripts/augment_obfu_dataset.py \
    --dataset leachl/obfuscated-exebench \
    --source-dataset jordiae/exebench \
    --push-to-hub

echo "Augmentation complete at $(date)"
