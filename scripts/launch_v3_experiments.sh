#!/usr/bin/env bash
# ============================================================================
# v3 Experiment Launcher: Verifying Stability of Collapsed States
# ============================================================================
#
# Runs ALL v3 training experiments with:
#   - eval_freq=99999 (no in-loop evaluation)
#   - save_freq=10 (checkpoint every 10 epochs)
#
# After all training is done, use reconstruct_tracking.py + analysis scripts.
#
# Usage:
#   # Run a specific batch:
#   bash scripts/launch_v3_experiments.sh baselines
#   bash scripts/launch_v3_experiments.sh family_a
#   bash scripts/launch_v3_experiments.sh family_b1
#   bash scripts/launch_v3_experiments.sh family_b2
#   bash scripts/launch_v3_experiments.sh family_c1
#   bash scripts/launch_v3_experiments.sh family_c2
#   bash scripts/launch_v3_experiments.sh family_e1
#
#   # Run everything:
#   nohup bash scripts/launch_v3_experiments.sh all > logs/v3_all.log 2>&1 &
#
# ============================================================================

set -euo pipefail

PYTHON="/home/sky/miniconda3/envs/adjoint_samplers/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Common flags: no eval during training, save every 10 epochs, no wandb
COMMON="eval_freq=99999 save_freq=10 use_wandb=false"

SEEDS="0 1 2 3 4"
RESULTS="results/v3"

START=$(date +%s)
DONE=0
TOTAL=0

log_progress() {
    local now=$(date +%s)
    local elapsed=$((now - START))
    local rate=0
    if [ "$DONE" -gt 0 ]; then
        rate=$((elapsed / DONE))
        local remaining=$(( (TOTAL - DONE) * rate ))
        echo "[${DONE}/${TOTAL}] $(date -u -d '+9 hours' '+%H:%M:%S KST') | ${rate}s/run | ~$((remaining/60))m left"
    else
        echo "[${DONE}/${TOTAL}] $(date -u -d '+9 hours' '+%H:%M:%S KST') | starting..."
    fi
}

run_train() {
    # Usage: run_train <output_dir> <extra_args...>
    local outdir="$1"; shift
    echo "  -> $outdir"
    $PYTHON train.py $COMMON "$@" hydra.run.dir="$outdir"
    DONE=$((DONE + 1))
    log_progress
}

# ============================================================================
# BASELINES: Pretrain B1, B5, B7 to convergence (needed for families B, C, E)
# ============================================================================
run_baselines() {
    echo ""
    echo "=========================================="
    echo "BASELINES: B1 (asym), B5, B7 — 5 seeds each"
    echo "=========================================="
    TOTAL=$((TOTAL + 15))

    for SEED in $SEEDS; do
        # B1 asymmetric (80/20) — standard config
        run_train "${RESULTS}/baselines/b1_asym/seed_${SEED}" \
            experiment=b1_asbs seed=$SEED num_epochs=2000

        # B5 (heterogeneous covariance, center_scale=5)
        run_train "${RESULTS}/baselines/b5/seed_${SEED}" \
            experiment=b5_asbs seed=$SEED num_epochs=2000

        # B7 (three-well, standard params)
        run_train "${RESULTS}/baselines/b7/seed_${SEED}" \
            experiment=b7_asbs seed=$SEED num_epochs=2000
    done

    # B1 symmetric — needed for Family 0 reinterpretation
    TOTAL=$((TOTAL + 5))
    for SEED in $SEEDS; do
        run_train "${RESULTS}/baselines/b1_sym/seed_${SEED}" \
            experiment=b1_asbs seed=$SEED num_epochs=2000 w1=0.5
    done
}

# ============================================================================
# FAMILY A: Sequential Mode Addition (A1, A2, A3)
# ============================================================================
run_family_a() {
    echo ""
    echo "=========================================="
    echo "FAMILY A: Sequential Mode Addition"
    echo "=========================================="

    # ── A1: Pretrain on 2-mode, then switch to 3-mode ──
    # Phase 1: 2-mode balanced training (use B1 symmetric)
    # Phase 2: Resume with 3-mode target (different energy config)
    #
    # Phase 1 is the baselines/b1_sym runs.
    # Phase 2: resume from phase 1 checkpoint with k_mode_gmm 3-mode target.
    TOTAL=$((TOTAL + 5))
    echo "--- A1: 2-mode -> 3-mode switch ---"
    for SEED in $SEEDS; do
        PHASE1_CKPT="${RESULTS}/baselines/b1_sym/seed_${SEED}/checkpoints/checkpoint_latest.pt"
        run_train "${RESULTS}/family_a/a1_3mode/seed_${SEED}" \
            experiment=k_mode_gmm_asbs seed=$SEED num_epochs=2000 \
            "centers=[[-4.0,0.0],[4.0,0.0],[0.0,4.0]]" \
            "weights=[0.333,0.333,0.334]" \
            gmm_sigma=1.0 \
            checkpoint="${PHASE1_CKPT}"
    done

    # ── A2: Injection distance sweep ──
    TOTAL=$((TOTAL + 25))
    echo "--- A2: Injection distance sweep ---"
    for D_INJECT in 2 4 6 8 12; do
        for SEED in $SEEDS; do
            PHASE1_CKPT="${RESULTS}/baselines/b1_sym/seed_${SEED}/checkpoints/checkpoint_latest.pt"
            run_train "${RESULTS}/family_a/a2_dist_${D_INJECT}/seed_${SEED}" \
                experiment=k_mode_gmm_asbs seed=$SEED num_epochs=2000 \
                "centers=[[-4.0,0.0],[4.0,0.0],[0.0,${D_INJECT}.0]]" \
                "weights=[0.333,0.333,0.334]" \
                gmm_sigma=1.0 \
                checkpoint="${PHASE1_CKPT}"
        done
    done

    # ── A3: Energy depth variation (vary sigma of injected mode) ──
    # Deeper = smaller sigma (tighter mode)
    TOTAL=$((TOTAL + 20))
    echo "--- A3: Energy depth variation ---"
    for SIGMA in 0.5 0.8 1.5 2.0; do
        for SEED in $SEEDS; do
            PHASE1_CKPT="${RESULTS}/baselines/b1_sym/seed_${SEED}/checkpoints/checkpoint_latest.pt"
            run_train "${RESULTS}/family_a/a3_sigma_${SIGMA}/seed_${SEED}" \
                experiment=k_mode_gmm_asbs seed=$SEED num_epochs=2000 \
                "centers=[[-4.0,0.0],[4.0,0.0],[0.0,4.0]]" \
                "weights=[0.333,0.333,0.334]" \
                gmm_sigma=${SIGMA} \
                checkpoint="${PHASE1_CKPT}"
        done
    done
}

# ============================================================================
# FAMILY B1: Dead Mode Revival via Data Injection (MOST IMPORTANT)
# ============================================================================
run_family_b1() {
    echo ""
    echo "=========================================="
    echo "FAMILY B1: Dead Mode Revival (most important)"
    echo "=========================================="

    # Sweep injection fraction rho and duration M
    # Uses B7 baseline (where mode 2 dies naturally)
    # Dead mode center for B7: mode 3 at (0.3, 1.5) or mode 2 at (-1, 0)
    # Based on v1: mode 2 (at -1,0) dies. We inject samples there.

    echo "--- B1: rho sweep (fixed M=50) ---"
    TOTAL=$((TOTAL + 20))
    for RHO in 0.001 0.01 0.05 0.1; do
        for SEED in $SEEDS; do
            run_train "${RESULTS}/family_b1/rho_${RHO}_M50/seed_${SEED}" \
                experiment=b7_asbs seed=$SEED num_epochs=3000 \
                +v3_injection_start_epoch=2000 \
                +v3_injection_duration=50 \
                +v3_injection_fraction=${RHO} \
                "+v3_injection_mode_center=[-1.0,0.0]" \
                +v3_injection_mode_sigma=0.22
        done
    done

    echo "--- B1: M sweep (fixed rho=0.05) ---"
    TOTAL=$((TOTAL + 15))
    for M in 10 50 200; do
        for SEED in $SEEDS; do
            run_train "${RESULTS}/family_b1/rho_0.05_M${M}/seed_${SEED}" \
                experiment=b7_asbs seed=$SEED num_epochs=$((2000 + M + 1000)) \
                +v3_injection_start_epoch=2000 \
                +v3_injection_duration=${M} \
                +v3_injection_fraction=0.05 \
                "+v3_injection_mode_center=[-1.0,0.0]" \
                +v3_injection_mode_sigma=0.22
        done
    done

    # Baseline: B7 with no injection, extended to 3000 epochs
    echo "--- B1: baseline (no injection, extended) ---"
    TOTAL=$((TOTAL + 5))
    for SEED in $SEEDS; do
        run_train "${RESULTS}/family_b1/baseline/seed_${SEED}" \
            experiment=b7_asbs seed=$SEED num_epochs=3000
    done
}

# ============================================================================
# FAMILY B2: Controller-Level Perturbation
# ============================================================================
run_family_b2() {
    echo ""
    echo "=========================================="
    echo "FAMILY B2: Controller-Level Perturbation"
    echo "=========================================="

    # Take converged B7 baselines, perturb weights, resume training
    TOTAL=$((TOTAL + 20))
    for SIGMA in 0.001 0.01 0.1 1.0; do
        for SEED in $SEEDS; do
            SRC_CKPT="${RESULTS}/baselines/b7/seed_${SEED}/checkpoints/checkpoint_1990.pt"
            PERTURBED_DIR="${RESULTS}/family_b2/sigma_${SIGMA}/seed_${SEED}"
            PERTURBED_CKPT="${PERTURBED_DIR}/checkpoints/checkpoint_perturbed.pt"

            # Perturb checkpoint
            mkdir -p "$(dirname "$PERTURBED_CKPT")"
            $PYTHON scripts/perturb_checkpoint.py \
                --ckpt "$SRC_CKPT" \
                --sigma "$SIGMA" \
                --output "$PERTURBED_CKPT" \
                --seed "$SEED"

            # Resume training from perturbed checkpoint for 1000 more epochs
            run_train "${PERTURBED_DIR}" \
                experiment=b7_asbs seed=$SEED num_epochs=3000 \
                checkpoint="${PERTURBED_CKPT}"
        done
    done
}

# ============================================================================
# FAMILY C1: Initialization-to-Collapse Distance Sweep
# ============================================================================
run_family_c1() {
    echo ""
    echo "=========================================="
    echo "FAMILY C1: Init Distance Sweep"
    echo "=========================================="

    # Take converged B7, perturb at various magnitudes, resume
    TOTAL=$((TOTAL + 20))
    for D_INIT in 0.01 0.1 1.0 10.0; do
        for SEED in $SEEDS; do
            SRC_CKPT="${RESULTS}/baselines/b7/seed_${SEED}/checkpoints/checkpoint_1990.pt"
            PERTURBED_DIR="${RESULTS}/family_c1/dinit_${D_INIT}/seed_${SEED}"
            PERTURBED_CKPT="${PERTURBED_DIR}/checkpoints/checkpoint_perturbed.pt"

            mkdir -p "$(dirname "$PERTURBED_CKPT")"
            $PYTHON scripts/perturb_checkpoint.py \
                --ckpt "$SRC_CKPT" \
                --sigma "$D_INIT" \
                --output "$PERTURBED_CKPT" \
                --seed "$SEED"

            run_train "${PERTURBED_DIR}" \
                experiment=b7_asbs seed=$SEED num_epochs=3000 \
                checkpoint="${PERTURBED_CKPT}"
        done
    done
}

# ============================================================================
# FAMILY C2: Adversarial Collapsed States
# ============================================================================
run_family_c2() {
    echo ""
    echo "=========================================="
    echo "FAMILY C2: Adversarial Collapsed States"
    echo "=========================================="

    # Phase 1: Pretrain on single-mode targets (one mode of B7 each)
    # We use KModeGaussianMixture with K=1 centered at each B7 mode

    B7_MODE_1="[1.0,0.0]"      # B7 mode 1
    B7_MODE_2="[-1.0,0.0]"     # B7 mode 2
    B7_MODE_3="[0.3,1.5]"      # B7 mode 3

    echo "--- C2 Phase 1: Single-mode pretraining ---"
    TOTAL=$((TOTAL + 15))
    for MODE_IDX in 1 2 3; do
        case $MODE_IDX in
            1) CENTER="$B7_MODE_1" ;;
            2) CENTER="$B7_MODE_2" ;;
            3) CENTER="$B7_MODE_3" ;;
        esac
        for SEED in $SEEDS; do
            run_train "${RESULTS}/family_c2/phase1_mode${MODE_IDX}/seed_${SEED}" \
                experiment=k_mode_gmm_asbs seed=$SEED num_epochs=1000 \
                "centers=[${CENTER}]" \
                "weights=[1.0]" \
                gmm_sigma=0.22
        done
    done

    # Phase 2: Resume on true B7 target
    echo "--- C2 Phase 2: Switch to true B7 ---"
    TOTAL=$((TOTAL + 15))
    for MODE_IDX in 1 2 3; do
        for SEED in $SEEDS; do
            PHASE1_CKPT="${RESULTS}/family_c2/phase1_mode${MODE_IDX}/seed_${SEED}/checkpoints/checkpoint_latest.pt"
            run_train "${RESULTS}/family_c2/phase2_mode${MODE_IDX}/seed_${SEED}" \
                experiment=b7_asbs seed=$SEED num_epochs=2000 \
                checkpoint="${PHASE1_CKPT}"
        done
    done
}

# ============================================================================
# FAMILY E1: Attractor Enumeration on B7
# ============================================================================
run_family_e1() {
    echo ""
    echo "=========================================="
    echo "FAMILY E1: Attractor Enumeration"
    echo "=========================================="

    # For B7 (3 modes), enumerate all proper subsets S:
    # S={1}, S={2}, S={3}, S={1,2}, S={1,3}, S={2,3}
    # Pretrain on p_S, then switch to full B7

    B7_MODE_1="[1.0,0.0]"
    B7_MODE_2="[-1.0,0.0]"
    B7_MODE_3="[0.3,1.5]"

    # Single-mode subsets (reuse C2 phase 1 if available)
    # Two-mode subsets
    echo "--- E1: Two-mode subset pretraining ---"
    TOTAL=$((TOTAL + 15))

    # S={1,2}: modes at (1,0) and (-1,0)
    for SEED in $SEEDS; do
        run_train "${RESULTS}/family_e1/phase1_S12/seed_${SEED}" \
            experiment=k_mode_gmm_asbs seed=$SEED num_epochs=1000 \
            "centers=[[1.0,0.0],[-1.0,0.0]]" \
            "weights=[0.5,0.5]" \
            gmm_sigma=0.22
    done

    # S={1,3}: modes at (1,0) and (0.3,1.5)
    for SEED in $SEEDS; do
        run_train "${RESULTS}/family_e1/phase1_S13/seed_${SEED}" \
            experiment=k_mode_gmm_asbs seed=$SEED num_epochs=1000 \
            "centers=[[1.0,0.0],[0.3,1.5]]" \
            "weights=[0.5,0.5]" \
            gmm_sigma=0.22
    done

    # S={2,3}: modes at (-1,0) and (0.3,1.5)
    for SEED in $SEEDS; do
        run_train "${RESULTS}/family_e1/phase1_S23/seed_${SEED}" \
            experiment=k_mode_gmm_asbs seed=$SEED num_epochs=1000 \
            "centers=[[-1.0,0.0],[0.3,1.5]]" \
            "weights=[0.5,0.5]" \
            gmm_sigma=0.22
    done

    # Phase 2: Resume all on true B7
    echo "--- E1 Phase 2: Switch to true B7 ---"
    TOTAL=$((TOTAL + 30))
    for SUBSET in S12 S13 S23; do
        for SEED in $SEEDS; do
            PHASE1_CKPT="${RESULTS}/family_e1/phase1_${SUBSET}/seed_${SEED}/checkpoints/checkpoint_latest.pt"
            run_train "${RESULTS}/family_e1/phase2_${SUBSET}/seed_${SEED}" \
                experiment=b7_asbs seed=$SEED num_epochs=2000 \
                checkpoint="${PHASE1_CKPT}"
        done
    done

    # Single-mode subsets phase 2 (reuse C2 phase1 checkpoints)
    TOTAL=$((TOTAL + 15))
    for MODE_IDX in 1 2 3; do
        for SEED in $SEEDS; do
            PHASE1_CKPT="${RESULTS}/family_c2/phase1_mode${MODE_IDX}/seed_${SEED}/checkpoints/checkpoint_latest.pt"
            run_train "${RESULTS}/family_e1/phase2_S${MODE_IDX}/seed_${SEED}" \
                experiment=b7_asbs seed=$SEED num_epochs=2000 \
                checkpoint="${PHASE1_CKPT}"
        done
    done
}

# ============================================================================
# DISPATCHER
# ============================================================================
BATCH="${1:-all}"

echo "=========================================="
echo "v3 EXPERIMENT LAUNCHER"
echo "Batch: $BATCH"
echo "GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Started: $(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST')"
echo "=========================================="

mkdir -p logs

case "$BATCH" in
    baselines)   run_baselines ;;
    family_a)    run_family_a ;;
    family_b1)   run_family_b1 ;;
    family_b2)   run_family_b2 ;;
    family_c1)   run_family_c1 ;;
    family_c2)   run_family_c2 ;;
    family_e1)   run_family_e1 ;;
    all)
        run_baselines
        run_family_b1
        run_family_a
        run_family_b2
        run_family_c1
        run_family_c2
        run_family_e1
        ;;
    *)
        echo "Unknown batch: $BATCH"
        echo "Available: baselines, family_a, family_b1, family_b2, family_c1, family_c2, family_e1, all"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "BATCH '$BATCH' COMPLETE"
echo "Finished: $(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST')"
echo "Total runs: $DONE"
echo ""
echo "Next steps:"
echo "  1. Reconstruct tracking: python scripts/reconstruct_tracking.py --results-dir ${RESULTS} --recursive --n-samples 10000"
echo "  2. Run analysis scripts (to be implemented)"
echo "=========================================="
