#!/usr/bin/env bash
# Batch 1 — GPU 1 (10 runs)
# C2 phase1 mode2 seeds1,2 (2), C2p1 mode3 (3), E1 phase1 S12 (3), E1p1 S13 seeds0,1 (2)
set -euo pipefail
cd /home/sky/SML/ASBS_MC

PYTHON="/home/sky/miniconda3/envs/adjoint_samplers/bin/python"
export CUDA_VISIBLE_DEVICES=1
COMMON="eval_freq=99999 save_freq=10 use_wandb=false"
R="results/v3"
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

echo "=== BATCH 1 GPU 1: $(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST') ==="

# --- C2 phase1 mode2 seeds 1,2 (2 runs × 1000 ep) ---
for S in 1 2; do
    run "${R}/family_c2/phase1_mode2/seed_${S}" \
        experiment=k_mode_gmm_asbs seed=$S num_epochs=1000 \
        "centers=[[-1.0,0.0]]" "weights=[1.0]" gmm_sigma=0.22
done

# --- C2 phase1 mode3 (3 runs × 1000 ep) ---
for S in 0 1 2; do
    run "${R}/family_c2/phase1_mode3/seed_${S}" \
        experiment=k_mode_gmm_asbs seed=$S num_epochs=1000 \
        "centers=[[0.3,1.5]]" "weights=[1.0]" gmm_sigma=0.22
done

# --- E1 phase1 S12: modes 1+2 (3 runs × 1000 ep) ---
for S in 0 1 2; do
    run "${R}/family_e1/phase1_S12/seed_${S}" \
        experiment=k_mode_gmm_asbs seed=$S num_epochs=1000 \
        "centers=[[1.0,0.0],[-1.0,0.0]]" "weights=[0.5,0.5]" gmm_sigma=0.22
done

# --- E1 phase1 S13: modes 1+3, seeds 0,1 (2 runs × 1000 ep) ---
for S in 0 1; do
    run "${R}/family_e1/phase1_S13/seed_${S}" \
        experiment=k_mode_gmm_asbs seed=$S num_epochs=1000 \
        "centers=[[1.0,0.0],[0.3,1.5]]" "weights=[0.5,0.5]" gmm_sigma=0.22
done

echo "=== BATCH 1 GPU 1 COMPLETE: $(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST') ==="
echo "Total: $DONE runs"
