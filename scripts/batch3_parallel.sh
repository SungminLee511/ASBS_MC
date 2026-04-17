#!/usr/bin/env bash
# ============================================================================
# Batch 3: 20 runs (10 per GPU, all concurrent)
#
# Backfills for incomplete Phase 1 experiments + B1-ctrl start
#
# GPU 0 (10 runs):
#   B1 rho=0.001 M50 × seeds 3,4       (2 runs, 3000 ep)
#   B1 rho=0.01  M50 × seeds 3,4       (2 runs, 3000 ep)
#   A1 2→3 mode switch × seeds 3,4     (2 runs, 2000 ep, from B1_sym ckpt)
#   E1-ph1 S13 × seeds 0,1,3,4         (4 runs, 1000 ep)
#
# GPU 1 (10 runs):
#   B1 rho=0.05 M50 × seeds 3,4        (2 runs, 3000 ep)
#   B1 rho=0.1  M50 × seeds 3,4        (2 runs, 3000 ep)
#   B1-ctrl (no injection) × seeds 0,1,2 (3 runs, 3000 ep)
#   E1-ph1 S12 × seeds 3,4             (2 runs, 1000 ep)
#   E1-ph1 S23 × seed 4                (1 run, 1000 ep)
# ============================================================================
set -uo pipefail

cd /home/sky/SML/ASBS_MC

PYTHON="/home/sky/miniconda3/envs/adjoint_samplers/bin/python"
COMMON="eval_freq=99999 save_freq=10 use_wandb=false"
R="results/v3"
LOGDIR="logs/batch3"
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
echo "BATCH 3: 20 runs (10 per GPU)"
echo "Started: $(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST')"
echo "=========================================="

# ── GPU 0: B1 rho={0.001,0.01} backfills + A1 backfills + E1-ph1-S13 ──

# B1 injection: rho=0.001, M=50 — seeds 3,4 (backfill)
for S in 3 4; do
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

# B1 injection: rho=0.01, M=50 — seeds 3,4 (backfill)
for S in 3 4; do
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

# A1: 2-mode → 3-mode switch — seeds 3,4 (backfill, from B1_sym checkpoint)
for S in 3 4; do
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

# E1 phase1 S13 — seeds 0,1,3,4 (backfill; only s2 done so far)
for S in 0 1 3 4; do
    run_one 0 "${R}/family_e1/phase1_S13/seed_${S}" \
        "${LOGDIR}/gpu0_e1ph1_S13_s${S}.log" \
        experiment=k_mode_gmm_asbs seed=$S num_epochs=1000 \
        "centers=[[1.0,0.0],[0.3,1.5]]" "weights=[0.5,0.5]" gmm_sigma=0.22 &
    PIDS+=($!)
done

# ── GPU 1: B1 rho={0.05,0.1} backfills + B1-ctrl + E1-ph1-S12/S23 ──

# B1 injection: rho=0.05, M=50 — seeds 3,4 (backfill)
for S in 3 4; do
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

# B1 injection: rho=0.1, M=50 — seeds 3,4 (backfill)
for S in 3 4; do
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

# B1-ctrl: B7 extended baseline, no injection (new — seeds 0,1,2)
for S in 0 1 2; do
    run_one 1 "${R}/family_b1/baseline/seed_${S}" \
        "${LOGDIR}/gpu1_b1ctrl_s${S}.log" \
        experiment=b7_asbs seed=$S num_epochs=3000 &
    PIDS+=($!)
done

# E1 phase1 S12 — seeds 3,4 (backfill)
for S in 3 4; do
    run_one 1 "${R}/family_e1/phase1_S12/seed_${S}" \
        "${LOGDIR}/gpu1_e1ph1_S12_s${S}.log" \
        experiment=k_mode_gmm_asbs seed=$S num_epochs=1000 \
        "centers=[[1.0,0.0],[-1.0,0.0]]" "weights=[0.5,0.5]" gmm_sigma=0.22 &
    PIDS+=($!)
done

# E1 phase1 S23 — seed 4 (backfill; s0,1,3 done, s2 diverged)
run_one 1 "${R}/family_e1/phase1_S23/seed_4" \
    "${LOGDIR}/gpu1_e1ph1_S23_s4.log" \
    experiment=k_mode_gmm_asbs seed=4 num_epochs=1000 \
    "centers=[[-1.0,0.0],[0.3,1.5]]" "weights=[0.5,0.5]" gmm_sigma=0.22 &
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
echo "BATCH 3 COMPLETE"
echo "Finished: $(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST')"
echo "Failed: $FAILED / ${#PIDS[@]}"
echo "=========================================="
