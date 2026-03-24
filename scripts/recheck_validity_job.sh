#!/bin/bash
#SBATCH --job-name=val-recheck
#SBATCH -p multicore
#SBATCH --mem=16G
#SBATCH --array=0-99
#SBATCH -t 0-02:00:00
#SBATCH --output=logs/recheck_%j.out
#SBATCH --error=logs/recheck_%j.err

# Setup environment (same as eval_csf.sh)
source env/bin/activate
export PATH=$PWD/tools/clang+llvm-14.0.0-x86_64-linux-gnu-ubuntu-18.04/bin:$PATH
export PYTHONPATH=$PWD

# Input predictions file - change if needed or pass via env var: PREDS=foo sbatch ...
PREDS="${PREDS:-results/preds_test_synth_leachl_forklift-arm-ir-ir_v4.jsonl}"
OUTPUT="${OUTPUT:-results/recheck_test_synth_leachl_forklift-arm-ir-ir_v4.json}"

echo "Re-checking validity for $PREDS"
python3 scripts/recheck_validity.py "$PREDS" --workers 16 --output "$OUTPUT"
