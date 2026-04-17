#!/usr/bin/env bash
# Batch 2 — GPU 0 (10 runs)
# B1 injection rho sweep: rho={0.001, 0.01} × M=50 × 3 seeds (6 runs, 3000 ep each)
# A1: 2-mode→3-mode switch × 3 seeds (2000 ep, from B1_sym checkpoint)
# E1 phase1 S13 seed 2 (1 run, 1000 ep — completes S13)
set -euo pipefail
cd /home/sky/SML/ASBS_MC

PYTHON="/home/sky/miniconda3/envs/adjoint_samplers/bin/python"
export CUDA_VISIBLE_DEVICES=0
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

echo "=== BATCH 2 GPU 0: $(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST') ==="

# --- B1 injection: rho=0.001, M=50 (3 runs × 3000 ep) --- MOST IMPORTANT
for S in 0 1 2; do
    run "${R}/family_b1/rho_0.001_M50/seed_${S}" \
        experiment=b7_asbs seed=$S num_epochs=3000 \
        +v3_injection_start_epoch=2000 \
        +v3_injection_duration=50 \
        +v3_injection_fraction=0.001 \
        "+v3_injection_mode_center=[-1.0,0.0]" \
        +v3_injection_mode_sigma=0.22
done

# --- B1 injection: rho=0.01, M=50 (3 runs × 3000 ep) ---
for S in 0 1 2; do
    run "${R}/family_b1/rho_0.01_M50/seed_${S}" \
        experiment=b7_asbs seed=$S num_epochs=3000 \
        +v3_injection_start_epoch=2000 \
        +v3_injection_duration=50 \
        +v3_injection_fraction=0.01 \
        "+v3_injection_mode_center=[-1.0,0.0]" \
        +v3_injection_mode_sigma=0.22
done

# --- A1: 2-mode → 3-mode switch (3 runs × 2000 ep from B1_sym ckpt) ---
for S in 0 1 2; do
    CKPT="${R}/baselines/b1_sym/seed_${S}/checkpoints/checkpoint_latest.pt"
    run "${R}/family_a/a1_3mode/seed_${S}" \
        experiment=k_mode_gmm_asbs seed=$S num_epochs=2000 \
        "centers=[[-4.0,0.0],[4.0,0.0],[0.0,4.0]]" \
        "weights=[0.333,0.333,0.334]" \
        gmm_sigma=1.0 \
        checkpoint="${CKPT}"
done

# --- E1 phase1 S13 seed 2 (1 run × 1000 ep — completes S13 set) ---
run "${R}/family_e1/phase1_S13/seed_2" \
    experiment=k_mode_gmm_asbs seed=2 num_epochs=1000 \
    "centers=[[1.0,0.0],[0.3,1.5]]" "weights=[0.5,0.5]" gmm_sigma=0.22

echo "=== BATCH 2 GPU 0 COMPLETE: $(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST') ==="
echo "Total: $DONE runs"
