#!/usr/bin/env bash
# ============================================================================
# Batch 2: 20 runs (10 per GPU, all concurrent)
#
# GPU 0 (10 runs):
#   B1 rho=0.001 M50 × seeds 0,1,2   (3 runs, 3000 ep)
#   B1 rho=0.01  M50 × seeds 0,1,2   (3 runs, 3000 ep)
#   A1 2→3 mode switch × seeds 0,1,2 (3 runs, 2000 ep, from B1_sym ckpt)
#   E1-ph1 S13 seed 2                (1 run, 1000 ep)
#
# GPU 1 (10 runs):
#   B1 rho=0.05 M50 × seeds 0,1,2    (3 runs, 3000 ep)
#   B1 rho=0.1  M50 × seeds 0,1,2    (3 runs, 3000 ep)
#   E1-ph1 S23 × seeds 0,1,2         (3 runs, 1000 ep)
#   C2-ph1 mode1 seed 0              (1 run, 1000 ep)
# ============================================================================
set -uo pipefail

cd /home/sky/SML/ASBS_MC

PYTHON="/home/sky/miniconda3/envs/adjoint_samplers/bin/python"
COMMON="eval_freq=99999 save_freq=10 use_wandb=false"
R="results/v3"
LOGDIR="logs/batch2"
mkdir -p "$LOGDIR"

PIDS=()
FAILED=0

run_one() {
    local gpu="$1" outdir="$2" log="$3"; shift 3
    echo "[$(date -u -d '+9 hours' '+%H:%M:%S KST')] GPU${gpu} START: $outdir"
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON train.py $COMMON "$@" hydra.run.dir="$outdir" \
        > "$log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "[$(date -u -d '+9 hours' '+%H:%M:%S KST')] GPU${gpu} DONE:  $outdir"
    else
        echo "[$(date -u -d '+9 hours' '+%H:%M:%S KST')] GPU${gpu} FAIL(rc=$rc): $outdir"
    fi
    return $rc
}

echo "=========================================="
echo "BATCH 2: 20 runs (10 per GPU)"
echo "Started: $(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST')"
echo "=========================================="

# ── GPU 0: B1 rho={0.001,0.01} + A1 + E1-ph1-S13 ──

# B1 injection: rho=0.001, M=50 (3 runs × 3000 ep)
for S in 0 1 2; do
    run_one 0 "${R}/family_b1/rho_0.001_M50/seed_${S}" \
        "${LOGDIR}/gpu0_b1_rho0.001_s${S}.log" \
        experiment=b7_asbs seed=$S num_epochs=3000 \
        +v3_injection_start_epoch=2000 \
        +v3_injection_duration=50 \
        +v3_injection_fraction=0.001 \
        "+v3_injection_mode_center=[-1.0,0.0]" \
        +v3_injection_mode_sigma=0.22 &
    PIDS+=($!)
done

# B1 injection: rho=0.01, M=50 (3 runs × 3000 ep)
for S in 0 1 2; do
    run_one 0 "${R}/family_b1/rho_0.01_M50/seed_${S}" \
        "${LOGDIR}/gpu0_b1_rho0.01_s${S}.log" \
        experiment=b7_asbs seed=$S num_epochs=3000 \
        +v3_injection_start_epoch=2000 \
        +v3_injection_duration=50 \
        +v3_injection_fraction=0.01 \
        "+v3_injection_mode_center=[-1.0,0.0]" \
        +v3_injection_mode_sigma=0.22 &
    PIDS+=($!)
done

# A1: 2-mode → 3-mode switch (3 runs × 2000 ep, from B1_sym checkpoint)
# NOTE: Must use absolute path — Hydra changes CWD to hydra.run.dir
for S in 0 1 2; do
    CKPT="$(pwd)/${R}/baselines/b1_sym/seed_${S}/checkpoints/checkpoint_latest.pt"
    run_one 0 "${R}/family_a/a1_3mode/seed_${S}" \
        "${LOGDIR}/gpu0_a1_s${S}.log" \
        experiment=k_mode_gmm_asbs seed=$S num_epochs=2000 \
        "centers=[[-4.0,0.0],[4.0,0.0],[0.0,4.0]]" \
        "weights=[0.333,0.333,0.334]" \
        gmm_sigma=1.0 \
        checkpoint="${CKPT}" &
    PIDS+=($!)
done

# E1 phase1 S13 seed 2 (1 run × 1000 ep — completes S13 set)
run_one 0 "${R}/family_e1/phase1_S13/seed_2" \
    "${LOGDIR}/gpu0_e1ph1_S13_s2.log" \
    experiment=k_mode_gmm_asbs seed=2 num_epochs=1000 \
    "centers=[[1.0,0.0],[0.3,1.5]]" "weights=[0.5,0.5]" gmm_sigma=0.22 &
PIDS+=($!)

# ── GPU 1: B1 rho={0.05,0.1} + E1-ph1-S23 + C2-ph1-mode1 ──

# B1 injection: rho=0.05, M=50 (3 runs × 3000 ep)
for S in 0 1 2; do
    run_one 1 "${R}/family_b1/rho_0.05_M50/seed_${S}" \
        "${LOGDIR}/gpu1_b1_rho0.05_s${S}.log" \
        experiment=b7_asbs seed=$S num_epochs=3000 \
        +v3_injection_start_epoch=2000 \
        +v3_injection_duration=50 \
        +v3_injection_fraction=0.05 \
        "+v3_injection_mode_center=[-1.0,0.0]" \
        +v3_injection_mode_sigma=0.22 &
    PIDS+=($!)
done

# B1 injection: rho=0.1, M=50 (3 runs × 3000 ep)
for S in 0 1 2; do
    run_one 1 "${R}/family_b1/rho_0.1_M50/seed_${S}" \
        "${LOGDIR}/gpu1_b1_rho0.1_s${S}.log" \
        experiment=b7_asbs seed=$S num_epochs=3000 \
        +v3_injection_start_epoch=2000 \
        +v3_injection_duration=50 \
        +v3_injection_fraction=0.1 \
        "+v3_injection_mode_center=[-1.0,0.0]" \
        +v3_injection_mode_sigma=0.22 &
    PIDS+=($!)
done

# E1 phase1 S23: modes {2,3} (3 runs × 1000 ep)
for S in 0 1 2; do
    run_one 1 "${R}/family_e1/phase1_S23/seed_${S}" \
        "${LOGDIR}/gpu1_e1ph1_S23_s${S}.log" \
        experiment=k_mode_gmm_asbs seed=$S num_epochs=1000 \
        "centers=[[-1.0,0.0],[0.3,1.5]]" "weights=[0.5,0.5]" gmm_sigma=0.22 &
    PIDS+=($!)
done

# C2 phase1 mode1 seed 0 (1 run × 1000 ep — starts building C2-ph1 checkpoints)
run_one 1 "${R}/family_c2/phase1_mode1/seed_0" \
    "${LOGDIR}/gpu1_c2ph1_m1_s0.log" \
    experiment=k_mode_gmm_asbs seed=0 num_epochs=1000 \
    "centers=[[1.0,0.0]]" "weights=[1.0]" gmm_sigma=0.22 &
PIDS+=($!)

echo ""
echo "Launched ${#PIDS[@]} jobs: PIDs = ${PIDS[*]}"
echo "Logs in: $LOGDIR"
echo ""

# Wait for all
for pid in "${PIDS[@]}"; do
    wait $pid || FAILED=$((FAILED + 1))
done

echo ""
echo "=========================================="
echo "BATCH 2 COMPLETE"
echo "Finished: $(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST')"
echo "Failed: $FAILED / ${#PIDS[@]}"
echo "=========================================="
