#!/usr/bin/env bash
# Evaluate and visualize all 1A–1D experiments.
# Step 1: Reconstruct tracking for the 6 runs missing mode_tracking.jsonl
# Step 2: Run evaluation/figures for E1–E10, E12 (everything that uses 1A–1D data)
#
# Usage: nohup bash scripts/eval_1a_to_1d.sh > logs/eval_1a_to_1d.log 2>&1 &

set -euo pipefail
CONDA_PATH="/home/sky/miniconda3/envs/adjoint_samplers/bin/python"
export CUDA_VISIBLE_DEVICES=1

echo "=========================================="
echo "EVAL 1A–1D — $(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST')"
echo "=========================================="

# ── Step 1: Reconstruct missing tracking ──────────────────────────────
MISSING_DIRS=(
    results/b4_asbs
    results/b8_asbs
)

echo ""
echo "[Step 1/2] Reconstructing tracking for B4 and B8 (6 runs × 151 checkpoints)..."
echo "Started: $(date -u -d '+9 hours' '+%H:%M:%S KST')"

for dir in "${MISSING_DIRS[@]}"; do
    echo ""
    echo "  >> Reconstructing $dir ..."
    $CONDA_PATH scripts/reconstruct_tracking.py \
        --results-dir "$dir" \
        --n-samples 10000 \
        --device cuda
    echo "  >> Done: $dir at $(date -u -d '+9 hours' '+%H:%M:%S KST')"
done

echo ""
echo "[Step 1/2] Reconstruction complete at $(date -u -d '+9 hours' '+%H:%M:%S KST')"

# ── Step 2: Generate figures and tables ───────────────────────────────
echo ""
echo "=========================================="
echo "[Step 2/2] Generating figures and tables..."
echo "Started: $(date -u -d '+9 hours' '+%H:%M:%S KST')"
echo "Running: E1 E2 E3 E4 E5 E6 E7 E8 E9 E10 E12"
echo "=========================================="

$CONDA_PATH evaluation/run_all.py --only e1 e2 e3 e4 e5 e6 e7 e8 e9 e10 e12

echo ""
echo "=========================================="
echo "ALL DONE — $(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST')"
echo "Figures: figures/"
echo "Tables:  tables/"
echo "=========================================="
