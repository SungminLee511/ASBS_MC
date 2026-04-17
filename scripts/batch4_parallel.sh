#!/usr/bin/env bash
# ============================================================================
# Batch 4: 11 runs (5 on GPU0, 6 on GPU1) — fills to 10 per GPU
#
# Family 0.5: Metastable survival sweep (kappa_3)
#
# GPU 0 (5 runs):
#   kappa3=8 × seeds 0-4          (5 runs, 2000 ep)
#
# GPU 1 (6 runs):
#   kappa3=12 × seeds 0-4         (5 runs, 2000 ep)
#   kappa3=16 × seed 0            (1 run, 2000 ep)
# ============================================================================
set -uo pipefail

cd /home/sky/SML/ASBS_MC

PYTHON="/home/sky/miniconda3/envs/adjoint_samplers/bin/python"
COMMON="eval_freq=99999 save_freq=10 use_wandb=false"
R="results/v3"
LOGDIR="logs/batch4"
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
echo "BATCH 4: 11 runs (5 GPU0 + 6 GPU1) — Fam 0.5 kappa3 sweep"
echo "Started: $(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST')"
echo "=========================================="

# ── GPU 0: kappa3=8 × seeds 0-4 ──
for S in 0 1 2 3 4; do
    run_one 0 "${R}/family_05/kappa3_8/seed_${S}" \
        "${LOGDIR}/gpu0_k8_s${S}.log" \
        experiment=b7_asbs seed=$S num_epochs=2000 kappa3=8 &
    PIDS+=($!)
done

# ── GPU 1: kappa3=12 × seeds 0-4 ──
for S in 0 1 2 3 4; do
    run_one 1 "${R}/family_05/kappa3_12/seed_${S}" \
        "${LOGDIR}/gpu1_k12_s${S}.log" \
        experiment=b7_asbs seed=$S num_epochs=2000 kappa3=12 &
    PIDS+=($!)
done

# ── GPU 1: kappa3=16 × seed 0 ──
run_one 1 "${R}/family_05/kappa3_16/seed_0" \
    "${LOGDIR}/gpu1_k16_s0.log" \
    experiment=b7_asbs seed=0 num_epochs=2000 kappa3=16 &
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
echo "BATCH 4 COMPLETE"
echo "Finished: $(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST')"
echo "Failed: $FAILED / ${#PIDS[@]}"
echo "=========================================="
