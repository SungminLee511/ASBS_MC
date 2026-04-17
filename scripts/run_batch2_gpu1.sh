#!/usr/bin/env bash
# Batch 2 — GPU 1 (10 runs)
# B1 injection rho={0.05, 0.1} × M=50 × 3 seeds (6 runs, 3000 ep each)
# E1 phase1 S23 × 3 seeds (3 runs, 1000 ep)
# C2 phase2 mode1 seed 0 (1 run, 2000 ep — starts phase2)
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

echo "=== BATCH 2 GPU 1: $(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST') ==="

# --- B1 injection: rho=0.05, M=50 (3 runs × 3000 ep) ---
for S in 0 1 2; do
    run "${R}/family_b1/rho_0.05_M50/seed_${S}" \
        experiment=b7_asbs seed=$S num_epochs=3000 \
        +v3_injection_start_epoch=2000 \
        +v3_injection_duration=50 \
        +v3_injection_fraction=0.05 \
        "+v3_injection_mode_center=[-1.0,0.0]" \
        +v3_injection_mode_sigma=0.22
done

# --- B1 injection: rho=0.1, M=50 (3 runs × 3000 ep) ---
for S in 0 1 2; do
    run "${R}/family_b1/rho_0.1_M50/seed_${S}" \
        experiment=b7_asbs seed=$S num_epochs=3000 \
        +v3_injection_start_epoch=2000 \
        +v3_injection_duration=50 \
        +v3_injection_fraction=0.1 \
        "+v3_injection_mode_center=[-1.0,0.0]" \
        +v3_injection_mode_sigma=0.22
done

# --- E1 phase1 S23: modes 2+3 (3 runs × 1000 ep — completes all E1 phase1) ---
for S in 0 1 2; do
    run "${R}/family_e1/phase1_S23/seed_${S}" \
        experiment=k_mode_gmm_asbs seed=$S num_epochs=1000 \
        "centers=[[-1.0,0.0],[0.3,1.5]]" "weights=[0.5,0.5]" gmm_sigma=0.22
done

# --- C2 phase2 mode1 seed 0 (1 run × 2000 ep — start phase2 chain) ---
CKPT="${R}/family_c2/phase1_mode1/seed_0/checkpoints/checkpoint_latest.pt"
run "${R}/family_c2/phase2_mode1/seed_0" \
    experiment=b7_asbs seed=0 num_epochs=2000 \
    checkpoint="${CKPT}"

echo "=== BATCH 2 GPU 1 COMPLETE: $(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST') ==="
echo "Total: $DONE runs"
