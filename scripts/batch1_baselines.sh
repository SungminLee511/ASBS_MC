#!/usr/bin/env bash
# ============================================================================
# Batch 1: All 20 baselines — 10 per GPU, all concurrent
# ============================================================================
set -uo pipefail

cd /home/sky/SML/ASBS_MC

PYTHON="/home/sky/miniconda3/envs/adjoint_samplers/bin/python"
COMMON="eval_freq=99999 save_freq=10 use_wandb=false"
RESULTS="results/v3"
LOGDIR="logs/batch1"
SEEDS="0 1 2 3 4"

mkdir -p "$LOGDIR"

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

PIDS=()

echo "=========================================="
echo "BATCH 1: 20 Baselines (10 per GPU)"
echo "Started: $(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST')"
echo "=========================================="

# ── GPU 0: BL-B1a (5 seeds) + BL-B5 (5 seeds) ──
for SEED in $SEEDS; do
    run_one 0 "${RESULTS}/baselines/b1_asym/seed_${SEED}" \
        "${LOGDIR}/gpu0_b1a_s${SEED}.log" \
        experiment=b1_asbs seed=$SEED num_epochs=2000 &
    PIDS+=($!)
done

for SEED in $SEEDS; do
    run_one 0 "${RESULTS}/baselines/b5/seed_${SEED}" \
        "${LOGDIR}/gpu0_b5_s${SEED}.log" \
        experiment=b5_asbs seed=$SEED num_epochs=2000 &
    PIDS+=($!)
done

# ── GPU 1: BL-B7 (5 seeds) + BL-B1s (5 seeds) ──
for SEED in $SEEDS; do
    run_one 1 "${RESULTS}/baselines/b7/seed_${SEED}" \
        "${LOGDIR}/gpu1_b7_s${SEED}.log" \
        experiment=b7_asbs seed=$SEED num_epochs=2000 &
    PIDS+=($!)
done

for SEED in $SEEDS; do
    run_one 1 "${RESULTS}/baselines/b1_sym/seed_${SEED}" \
        "${LOGDIR}/gpu1_b1s_s${SEED}.log" \
        experiment=b1_asbs seed=$SEED num_epochs=2000 w1=0.5 &
    PIDS+=($!)
done

echo ""
echo "Launched ${#PIDS[@]} jobs: PIDs = ${PIDS[*]}"
echo "Logs in: $LOGDIR"
echo ""

# Wait for all
FAILED=0
for pid in "${PIDS[@]}"; do
    wait $pid || FAILED=$((FAILED + 1))
done

echo ""
echo "=========================================="
echo "BATCH 1 COMPLETE"
echo "Finished: $(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST')"
echo "Failed: $FAILED / ${#PIDS[@]}"
echo "=========================================="
