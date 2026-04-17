#!/usr/bin/env bash
# Batch 1 — GPU 1 (10 runs IN PARALLEL)
# C2p1 mode2 seeds1,2 (2), C2p1 mode3 (3), E1p1 S12 (3), E1p1 S13 seeds0,1 (2)
set -euo pipefail
cd /home/sky/SML/ASBS_MC

PYTHON="/home/sky/miniconda3/envs/adjoint_samplers/bin/python"
export CUDA_VISIBLE_DEVICES=1
COMMON="eval_freq=99999 save_freq=10 use_wandb=false"
R="results/v3"

echo "=== BATCH 1 GPU 1: $(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST') ==="
echo "Launching 10 experiments in parallel..."

PIDS=()

# --- C2 phase1 mode2 seeds 1,2 (2 runs x 1000 ep) ---
for S in 1 2; do
    mkdir -p "${R}/family_c2/phase1_mode2/seed_${S}"
    $PYTHON train.py $COMMON experiment=k_mode_gmm_asbs seed=$S num_epochs=1000 \
        "centers=[[-1.0,0.0]]" "weights=[1.0]" gmm_sigma=0.22 \
        hydra.run.dir="${R}/family_c2/phase1_mode2/seed_${S}" &
    PIDS+=($!)
    echo "  [PID $!] family_c2/phase1_mode2/seed_${S} (1000 ep)"
done

# --- C2 phase1 mode3 (3 runs x 1000 ep) ---
for S in 0 1 2; do
    mkdir -p "${R}/family_c2/phase1_mode3/seed_${S}"
    $PYTHON train.py $COMMON experiment=k_mode_gmm_asbs seed=$S num_epochs=1000 \
        "centers=[[0.3,1.5]]" "weights=[1.0]" gmm_sigma=0.22 \
        hydra.run.dir="${R}/family_c2/phase1_mode3/seed_${S}" &
    PIDS+=($!)
    echo "  [PID $!] family_c2/phase1_mode3/seed_${S} (1000 ep)"
done

# --- E1 phase1 S12: modes 1+2 (3 runs x 1000 ep) ---
for S in 0 1 2; do
    mkdir -p "${R}/family_e1/phase1_S12/seed_${S}"
    $PYTHON train.py $COMMON experiment=k_mode_gmm_asbs seed=$S num_epochs=1000 \
        "centers=[[1.0,0.0],[-1.0,0.0]]" "weights=[0.5,0.5]" gmm_sigma=0.22 \
        hydra.run.dir="${R}/family_e1/phase1_S12/seed_${S}" &
    PIDS+=($!)
    echo "  [PID $!] family_e1/phase1_S12/seed_${S} (1000 ep)"
done

# --- E1 phase1 S13: modes 1+3, seeds 0,1 (2 runs x 1000 ep) ---
for S in 0 1; do
    mkdir -p "${R}/family_e1/phase1_S13/seed_${S}"
    $PYTHON train.py $COMMON experiment=k_mode_gmm_asbs seed=$S num_epochs=1000 \
        "centers=[[1.0,0.0],[0.3,1.5]]" "weights=[0.5,0.5]" gmm_sigma=0.22 \
        hydra.run.dir="${R}/family_e1/phase1_S13/seed_${S}" &
    PIDS+=($!)
    echo "  [PID $!] family_e1/phase1_S13/seed_${S} (1000 ep)"
done

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
echo "=== BATCH 1 GPU 1 COMPLETE: $(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST') ==="
echo "Total: ${#PIDS[@]} runs, $FAILED failed"
