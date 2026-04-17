#!/usr/bin/env bash
# ============================================================================
# Batch 5: 20 runs (10 per GPU) — Fam 0.5 kappa3 sweep continued
#
# GPU 0 (10 runs):
#   kappa3=4  × seeds 2,3,4        (3 runs, 2000 ep)
#   kappa3=16 × seeds 1,2,3,4      (4 runs, 2000 ep)
#   kappa3=18 × seeds 0,1,2        (3 runs, 2000 ep)
#
# GPU 1 (10 runs):
#   kappa3=18 × seeds 3,4          (2 runs, 2000 ep)
#   kappa3=20 × seeds 0,1,2,3,4    (5 runs, 2000 ep)
#   kappa3=24 × seeds 0,1,2        (3 runs, 2000 ep)
#
# After this batch, remaining: k24 s3-4, k32 s0-4 = 7 runs
# ============================================================================
set -uo pipefail

cd /home/sky/SML/ASBS_MC

PYTHON="/home/sky/miniconda3/envs/adjoint_samplers/bin/python"
COMMON="eval_freq=99999 save_freq=10 use_wandb=false"
R="results/v3"
LOGDIR="logs/batch5"
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
echo "BATCH 5: 20 runs (10 per GPU) — Fam 0.5 kappa3 sweep"
echo "Started: $(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST')"
echo "=========================================="

# ── GPU 0: k4 s2-4, k16 s1-4, k18 s0-2 ──

for S in 2 3 4; do
    run_one 0 "${R}/family_05/kappa3_4/seed_${S}" \
        "${LOGDIR}/gpu0_k4_s${S}.log" \
        experiment=b7_asbs seed=$S num_epochs=2000 kappa3=4 &
    PIDS+=($!)
done

for S in 1 2 3 4; do
    run_one 0 "${R}/family_05/kappa3_16/seed_${S}" \
        "${LOGDIR}/gpu0_k16_s${S}.log" \
        experiment=b7_asbs seed=$S num_epochs=2000 kappa3=16 &
    PIDS+=($!)
done

for S in 0 1 2; do
    run_one 0 "${R}/family_05/kappa3_18/seed_${S}" \
        "${LOGDIR}/gpu0_k18_s${S}.log" \
        experiment=b7_asbs seed=$S num_epochs=2000 kappa3=18 &
    PIDS+=($!)
done

# ── GPU 1: k18 s3-4, k20 s0-4, k24 s0-2 ──

for S in 3 4; do
    run_one 1 "${R}/family_05/kappa3_18/seed_${S}" \
        "${LOGDIR}/gpu1_k18_s${S}.log" \
        experiment=b7_asbs seed=$S num_epochs=2000 kappa3=18 &
    PIDS+=($!)
done

for S in 0 1 2 3 4; do
    run_one 1 "${R}/family_05/kappa3_20/seed_${S}" \
        "${LOGDIR}/gpu1_k20_s${S}.log" \
        experiment=b7_asbs seed=$S num_epochs=2000 kappa3=20 &
    PIDS+=($!)
done

for S in 0 1 2; do
    run_one 1 "${R}/family_05/kappa3_24/seed_${S}" \
        "${LOGDIR}/gpu1_k24_s${S}.log" \
        experiment=b7_asbs seed=$S num_epochs=2000 kappa3=24 &
    PIDS+=($!)
done

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
echo "BATCH 5 COMPLETE"
echo "Finished: $(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST')"
echo "Failed: $FAILED / ${#PIDS[@]}"
echo "=========================================="
