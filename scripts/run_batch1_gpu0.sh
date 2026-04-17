#!/usr/bin/env bash
# Batch 1 — GPU 0 (10 runs IN PARALLEL)
# B7 baselines (3), B1_sym baselines (3), C2 phase1 mode1 (3), C2p1 mode2 seed0 (1)
set -euo pipefail
cd /home/sky/SML/ASBS_MC

PYTHON="/home/sky/miniconda3/envs/adjoint_samplers/bin/python"
export CUDA_VISIBLE_DEVICES=0
COMMON="eval_freq=99999 save_freq=10 use_wandb=false"
R="results/v3"

echo "=== BATCH 1 GPU 0: $(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST') ==="
echo "Launching 10 experiments in parallel..."

PIDS=()

# --- B7 baselines (3 runs x 2000 ep) ---
for S in 0 1 2; do
    mkdir -p "${R}/baselines/b7/seed_${S}"
    $PYTHON train.py $COMMON experiment=b7_asbs seed=$S num_epochs=2000 \
        hydra.run.dir="${R}/baselines/b7/seed_${S}" &
    PIDS+=($!)
    echo "  [PID $!] baselines/b7/seed_${S} (2000 ep)"
done

# --- B1 symmetric baselines (3 runs x 2000 ep) ---
for S in 0 1 2; do
    mkdir -p "${R}/baselines/b1_sym/seed_${S}"
    $PYTHON train.py $COMMON experiment=b1_asbs seed=$S num_epochs=2000 w1=0.5 \
        hydra.run.dir="${R}/baselines/b1_sym/seed_${S}" &
    PIDS+=($!)
    echo "  [PID $!] baselines/b1_sym/seed_${S} (2000 ep)"
done

# --- C2 phase1 mode1 (3 runs x 1000 ep) ---
for S in 0 1 2; do
    mkdir -p "${R}/family_c2/phase1_mode1/seed_${S}"
    $PYTHON train.py $COMMON experiment=k_mode_gmm_asbs seed=$S num_epochs=1000 \
        "centers=[[1.0,0.0]]" "weights=[1.0]" gmm_sigma=0.22 \
        hydra.run.dir="${R}/family_c2/phase1_mode1/seed_${S}" &
    PIDS+=($!)
    echo "  [PID $!] family_c2/phase1_mode1/seed_${S} (1000 ep)"
done

# --- C2 phase1 mode2 seed 0 (1 run x 1000 ep) ---
mkdir -p "${R}/family_c2/phase1_mode2/seed_0"
$PYTHON train.py $COMMON experiment=k_mode_gmm_asbs seed=0 num_epochs=1000 \
    "centers=[[-1.0,0.0]]" "weights=[1.0]" gmm_sigma=0.22 \
    hydra.run.dir="${R}/family_c2/phase1_mode2/seed_0" &
PIDS+=($!)
echo "  [PID $!] family_c2/phase1_mode2/seed_0 (1000 ep)"

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
echo "=== BATCH 1 GPU 0 COMPLETE: $(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST') ==="
echo "Total: ${#PIDS[@]} runs, $FAILED failed"
