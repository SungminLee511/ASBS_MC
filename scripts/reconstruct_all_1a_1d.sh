#!/usr/bin/env bash
# Full reconstruction of mode_tracking.jsonl for ALL 1A–1D runs.
# Then re-run E8, E10, tables + all other eval figures.
#
# Estimated: ~164 runs × ~3 min = ~8h
#
# Usage: nohup bash scripts/reconstruct_all_1a_1d.sh > logs/reconstruct_all.log 2>&1 &

set -euo pipefail
CONDA_PATH="/home/sky/miniconda3/envs/adjoint_samplers/bin/python"
export CUDA_VISIBLE_DEVICES=1

TOTAL=164
DONE=0
START=$(date +%s)

log_progress() {
    local now=$(date +%s)
    local elapsed=$((now - START))
    local rate=0
    local eta="?"
    if [ "$DONE" -gt 0 ]; then
        rate=$((elapsed / DONE))
        local remaining=$(( (TOTAL - DONE) * rate ))
        local finish=$((now + remaining))
        eta=$(date -u -d "@$((finish + 32400))" '+%H:%M KST' 2>/dev/null || echo "~${remaining}s")
    fi
    echo "[${DONE}/${TOTAL}] $(date -u -d '+9 hours' '+%H:%M:%S KST') | ${rate}s/run | ETA: ${eta}"
}

echo "=========================================="
echo "RECONSTRUCT ALL 1A–1D — $(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST')"
echo "Total runs: $TOTAL"
echo "=========================================="

# ── 1A: Base benchmarks ──
echo ""
echo "=== 1A: Base benchmarks (24 runs) ==="
for b in b1_asbs b2_asbs b3_asbs b4_asbs b5_asbs b6_asbs b7_asbs b8_asbs; do
    $CONDA_PATH scripts/reconstruct_tracking.py \
        --results-dir results/$b \
        --n-samples 10000 \
        --device cuda \
        --overwrite
    DONE=$((DONE + 3))
    log_progress
done

# ── 1B: E7 multi-seed ──
echo ""
echo "=== 1B: E7 multi-seed (50 runs) ==="
$CONDA_PATH scripts/reconstruct_tracking.py \
    --results-dir results/e7_b1_sym \
    --n-samples 10000 \
    --device cuda \
    --overwrite
DONE=$((DONE + 30))
log_progress

$CONDA_PATH scripts/reconstruct_tracking.py \
    --results-dir results/e7_b7 \
    --n-samples 10000 \
    --device cuda \
    --overwrite
DONE=$((DONE + 20))
log_progress

# ── 1C: Sep sweep B1 ──
echo ""
echo "=== 1C: Sep sweep B1 (60 runs) ==="
for d in 2 3 4 5 6 7 8 9 10 12 15 20; do
    $CONDA_PATH scripts/reconstruct_tracking.py \
        --results-dir results/sep_sweep_b1/d_$d \
        --n-samples 10000 \
        --device cuda \
        --overwrite
    DONE=$((DONE + 5))
    log_progress
done

# ── 1D: Sep sweep B5 ──
echo ""
echo "=== 1D: Sep sweep B5 (30 runs) ==="
for cs in 2 3 4 5 7 10; do
    $CONDA_PATH scripts/reconstruct_tracking.py \
        --results-dir results/sep_sweep_b5/cs_$cs \
        --n-samples 10000 \
        --device cuda \
        --overwrite
    DONE=$((DONE + 5))
    log_progress
done

# ── Re-run ALL evaluation ──
echo ""
echo "=========================================="
echo "RECONSTRUCTION DONE — $(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST')"
echo "Now running evaluation..."
echo "=========================================="

$CONDA_PATH evaluation/run_all.py --only e1 e2 e3 e4 e5 e6 e7 e8 e9 e10 e12
$CONDA_PATH evaluation/run_all.py --tables-only

echo ""
echo "=========================================="
echo "ALL DONE — $(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST')"
echo "Figures: figures/"
echo "Tables:  tables/"
echo "=========================================="
