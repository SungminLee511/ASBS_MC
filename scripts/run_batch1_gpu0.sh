#!/usr/bin/env bash
# Batch 1 — GPU 0 (10 runs)
# Prerequisites: B7 baselines (3), B1_sym baselines (3), C2 phase1 mode1 (3), C2p1 mode2 seed0 (1)
set -euo pipefail
cd /home/sky/SML/ASBS_MC

PYTHON="/home/sky/miniconda3/envs/adjoint_samplers/bin/python"
export CUDA_VISIBLE_DEVICES=0
COMMON="eval_freq=99999 save_freq=10 use_wandb=false"
R="results/v3"
SEEDS="0 1 2"
DONE=0; TOTAL=10; START=$(date +%s)

log() {
    local now=$(date +%s); local el=$((now - START))
    echo "[${DONE}/${TOTAL}] $(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST') | elapsed=${el}s | $1"
}

run() {
    local out="$1"; shift
    log "START $out"
    $PYTHON train.py $COMMON "$@" hydra.run.dir="$out"
    DONE=$((DONE + 1))
    log "DONE  $out"
}

echo "=== BATCH 1 GPU 0: $(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST') ==="

# --- B7 baselines (3 runs × 2000 ep) — unlocks B2, C1, D1, F1, F2 ---
for S in $SEEDS; do
    run "${R}/baselines/b7/seed_${S}" experiment=b7_asbs seed=$S num_epochs=2000
done

# --- B1 symmetric baselines (3 runs × 2000 ep) — unlocks A1, A2, A3 ---
for S in $SEEDS; do
    run "${R}/baselines/b1_sym/seed_${S}" experiment=b1_asbs seed=$S num_epochs=2000 w1=0.5
done

# --- C2 phase1 mode1 (3 runs × 1000 ep) ---
for S in $SEEDS; do
    run "${R}/family_c2/phase1_mode1/seed_${S}" \
        experiment=k_mode_gmm_asbs seed=$S num_epochs=1000 \
        "centers=[[1.0,0.0]]" "weights=[1.0]" gmm_sigma=0.22
done

# --- C2 phase1 mode2 seed 0 (1 run × 1000 ep) ---
run "${R}/family_c2/phase1_mode2/seed_0" \
    experiment=k_mode_gmm_asbs seed=0 num_epochs=1000 \
    "centers=[[-1.0,0.0]]" "weights=[1.0]" gmm_sigma=0.22

echo "=== BATCH 1 GPU 0 COMPLETE: $(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST') ==="
echo "Total: $DONE runs"
