#!/usr/bin/env bash
# 1F: κ₃ sweep — 45 runs queued on GPU 1, 10 at a time (last batch 5)
#
# Batches:
#   1: κ₃=4,6    × 5 seeds (10 runs)
#   2: κ₃=8,10   × 5 seeds (10 runs)
#   3: κ₃=12,14  × 5 seeds (10 runs)
#   4: κ₃=16,18  × 5 seeds (10 runs)
#   5: κ₃=20     × 5 seeds (5 runs)
#
# Usage: nohup bash scripts/run_1f_queued.sh > logs/1f_queued.log 2>&1 &

set -euo pipefail
CONDA_PATH="/home/sky/miniconda3/envs/adjoint_samplers/bin/python"
GPU=0
START=$(date +%s)

echo "=========================================="
echo "1F: κ₃ SWEEP — $(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST')"
echo "45 runs on GPU $GPU, queued in 5 batches"
echo "=========================================="

launch_batch() {
    local batch_num=$1
    shift
    local kappas=("$@")
    local count=0

    echo ""
    echo "--- Batch $batch_num: κ₃=${kappas[*]} — $(date -u -d '+9 hours' '+%H:%M:%S KST') ---"

    for k in "${kappas[@]}"; do
        for s in 0 1 2 3 4; do
            CUDA_VISIBLE_DEVICES=$GPU $CONDA_PATH train.py \
                experiment=b7_asbs kappa3=$k seed=$s \
                save_freq=10 eval_freq=99999 num_epochs=1500 \
                hydra.run.dir=results/e13_sweep/kappa3_${k}/seed_$s \
                > logs/k3_${k}_s${s}.log 2>&1 &
            count=$((count + 1))
        done
    done

    echo "  Launched $count runs. Waiting for completion..."

    # Wait for all background jobs
    wait

    echo "  Batch $batch_num DONE — $(date -u -d '+9 hours' '+%H:%M:%S KST')"

    # Print final losses
    for k in "${kappas[@]}"; do
        for s in 0 1 2 3 4; do
            local last=$(tail -1 logs/k3_${k}_s${s}.log 2>/dev/null)
            local ep=$(echo "$last" | grep -oP 'ep=\K[0-9]+')
            local loss=$(echo "$last" | grep -oP 'loss=\K[0-9.]+')
            printf "    k3=%-3s s=%s  ep=%-4s  loss=%s\n" "$k" "$s" "$ep" "$loss"
        done
    done
}

launch_batch 1  4 6
launch_batch 2  8 10
launch_batch 3  12 14
launch_batch 4  16 18
launch_batch 5  20

END=$(date +%s)
ELAPSED=$(( (END - START) / 60 ))

echo ""
echo "=========================================="
echo "1F COMPLETE — $(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST')"
echo "Total time: ${ELAPSED} min"
echo "=========================================="
