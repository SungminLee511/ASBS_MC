#!/usr/bin/env bash
# v2 A2.3: Asymmetry-Resolved Phase Diagram
# B1 d=6 (mu1=[-3,0], mu2=[3,0]), w1 sweep, 1000 epochs, 10 seeds each
#
# 20 runs per batch (10 per GPU), queued sequentially
# w1 values: 0.51, 0.52, 0.55, 0.60, 0.65, 0.70, 0.80, 0.90
# (w1=0.50 reuses e7_b1_sym)
#
# Batches:
#   1: w1=0.51 (GPU0) + w1=0.52 (GPU1)  — 20 runs
#   2: w1=0.55 (GPU0) + w1=0.60 (GPU1)  — 20 runs
#   3: w1=0.65 (GPU0) + w1=0.70 (GPU1)  — 20 runs
#   4: w1=0.80 (GPU0) + w1=0.90 (GPU1)  — 20 runs
#
# Usage: nohup bash scripts/run_v2_a23_queued.sh > logs/v2_a23/queued.log 2>&1 &

set -euo pipefail
cd /home/sky/SML/ASBS_MC
CONDA_PATH="/home/sky/miniconda3/envs/adjoint_samplers/bin/python"
LOGDIR="logs/v2_a23"
START=$(date +%s)

echo "=========================================="
echo "v2 A2.3: ASYMMETRY PHASE DIAGRAM"
echo "Started: $(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST')"
echo "B1 d=6, w1 sweep, 1000 epochs, 10 seeds"
echo "20 runs/batch (10 per GPU)"
echo "=========================================="

launch_batch() {
    local batch_num=$1
    local w1_gpu0=$2
    local w1_gpu1=$3
    local batch_start=$(date +%s)

    echo ""
    echo "=== Batch $batch_num: w1=${w1_gpu0} (GPU0) + w1=${w1_gpu1} (GPU1) ==="
    echo "  Started: $(date -u -d '+9 hours' '+%H:%M:%S KST')"

    # GPU 0: first w1 value
    for s in $(seq 0 9); do
        CUDA_VISIBLE_DEVICES=0 $CONDA_PATH train.py \
            experiment=b1_asbs \
            w1=$w1_gpu0 \
            "energy.mu1=[-3.0, 0.0]" \
            "energy.mu2=[3.0, 0.0]" \
            seed=$s \
            save_freq=10 eval_freq=99999 num_epochs=1000 \
            hydra.run.dir=results/v2_a23/w1_${w1_gpu0}/seed_$s \
            > ${LOGDIR}/w1_${w1_gpu0}_s${s}.log 2>&1 &
    done

    # GPU 1: second w1 value
    for s in $(seq 0 9); do
        CUDA_VISIBLE_DEVICES=1 $CONDA_PATH train.py \
            experiment=b1_asbs \
            w1=$w1_gpu1 \
            "energy.mu1=[-3.0, 0.0]" \
            "energy.mu2=[3.0, 0.0]" \
            seed=$s \
            save_freq=10 eval_freq=99999 num_epochs=1000 \
            hydra.run.dir=results/v2_a23/w1_${w1_gpu1}/seed_$s \
            > ${LOGDIR}/w1_${w1_gpu1}_s${s}.log 2>&1 &
    done

    echo "  Launched 20 runs. Waiting..."
    wait

    local batch_end=$(date +%s)
    local batch_elapsed=$(( (batch_end - batch_start) / 60 ))
    echo "  Batch $batch_num DONE — $(date -u -d '+9 hours' '+%H:%M:%S KST') (${batch_elapsed} min)"

    # Print final status
    for w1 in $w1_gpu0 $w1_gpu1; do
        echo "  --- w1=${w1} ---"
        for s in $(seq 0 9); do
            local logf="${LOGDIR}/w1_${w1}_s${s}.log"
            local last=$(tail -1 "$logf" 2>/dev/null || echo "NO LOG")
            local ep=$(echo "$last" | grep -oP 'ep=\K[0-9]+' || echo "?")
            local loss=$(echo "$last" | grep -oP 'loss=\K[0-9.]+' || echo "?")
            printf "    s=%s  ep=%-4s  loss=%s\n" "$s" "$ep" "$loss"
        done
    done
}

launch_batch 1  0.51 0.52
launch_batch 2  0.55 0.60
launch_batch 3  0.65 0.70
launch_batch 4  0.80 0.90

END=$(date +%s)
ELAPSED=$(( (END - START) / 60 ))

echo ""
echo "=========================================="
echo "v2 A2.3 COMPLETE — $(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST')"
echo "Total time: ${ELAPSED} min"
echo "=========================================="
