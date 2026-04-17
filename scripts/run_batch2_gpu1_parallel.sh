#!/usr/bin/env bash
# Batch 2 — GPU 1 (10 runs IN PARALLEL)
# B1 injection rho={0.05, 0.1} × M=50 × 3 seeds (6 runs, 3000 ep each)
# E1 phase1 S23 × 3 seeds (3 runs, 1000 ep)
# C2 phase2 mode1 seed 0 (1 run, 2000 ep — starts phase2)
set -euo pipefail
cd /home/sky/SML/ASBS_MC

PYTHON="/home/sky/miniconda3/envs/adjoint_samplers/bin/python"
export CUDA_VISIBLE_DEVICES=1
COMMON="eval_freq=99999 save_freq=10 use_wandb=false"
R="results/v3"

echo "=== BATCH 2 GPU 1: $(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST') ==="
echo "Launching 10 experiments in parallel..."

PIDS=()

# --- B1 injection: rho=0.05, M=50 (3 runs × 3000 ep) ---
for S in 0 1 2; do
    mkdir -p "${R}/family_b1/rho_0.05_M50/seed_${S}"
    $PYTHON train.py $COMMON experiment=b7_asbs seed=$S num_epochs=3000 \
        +v3_injection_start_epoch=2000 \
        +v3_injection_duration=50 \
        +v3_injection_fraction=0.05 \
        "+v3_injection_mode_center=[-1.0,0.0]" \
        +v3_injection_mode_sigma=0.22 \
        hydra.run.dir="${R}/family_b1/rho_0.05_M50/seed_${S}" &
    PIDS+=($!)
    echo "  [PID $!] family_b1/rho_0.05_M50/seed_${S} (3000 ep)"
done

# --- B1 injection: rho=0.1, M=50 (3 runs × 3000 ep) ---
for S in 0 1 2; do
    mkdir -p "${R}/family_b1/rho_0.1_M50/seed_${S}"
    $PYTHON train.py $COMMON experiment=b7_asbs seed=$S num_epochs=3000 \
        +v3_injection_start_epoch=2000 \
        +v3_injection_duration=50 \
        +v3_injection_fraction=0.1 \
        "+v3_injection_mode_center=[-1.0,0.0]" \
        +v3_injection_mode_sigma=0.22 \
        hydra.run.dir="${R}/family_b1/rho_0.1_M50/seed_${S}" &
    PIDS+=($!)
    echo "  [PID $!] family_b1/rho_0.1_M50/seed_${S} (3000 ep)"
done

# --- E1 phase1 S23: modes 2+3 (3 runs × 1000 ep) ---
for S in 0 1 2; do
    mkdir -p "${R}/family_e1/phase1_S23/seed_${S}"
    $PYTHON train.py $COMMON experiment=k_mode_gmm_asbs seed=$S num_epochs=1000 \
        "centers=[[-1.0,0.0],[0.3,1.5]]" "weights=[0.5,0.5]" gmm_sigma=0.22 \
        hydra.run.dir="${R}/family_e1/phase1_S23/seed_${S}" &
    PIDS+=($!)
    echo "  [PID $!] family_e1/phase1_S23/seed_${S} (1000 ep)"
done

# --- C2 phase2 mode1 seed 0 (1 run × 2000 ep) ---
CKPT="${R}/family_c2/phase1_mode1/seed_0/checkpoints/checkpoint_latest.pt"
mkdir -p "${R}/family_c2/phase2_mode1/seed_0"
$PYTHON train.py $COMMON experiment=b7_asbs seed=0 num_epochs=2000 \
    checkpoint="${CKPT}" \
    hydra.run.dir="${R}/family_c2/phase2_mode1/seed_0" &
PIDS+=($!)
echo "  [PID $!] family_c2/phase2_mode1/seed_0 (2000 ep, from phase1 ckpt)"

echo ""
echo "All 10 launched. PIDs: ${PIDS[*]}"
echo "Waiting for all to finish..."

FAILED=0
for PID in "${PIDS[@]}"; do
    if ! wait "$PID"; then
        echo "  [FAILED] PID $PID exited with error"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "=== BATCH 2 GPU 1 COMPLETE: $(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST') ==="
echo "Total: ${#PIDS[@]} runs, $FAILED failed"
