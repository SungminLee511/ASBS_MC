# Experiment & Evaluation v2 Checklist

Ref: `experiment_evaluation_guide_v2.md`

---

## Part 0: New Scripts & Infrastructure Required

Before running v2 experiments, the following scripts/modifications must be written.
Existing v1 scripts (e1–e13) and evaluation pipeline remain intact.

### 0A. Training Hooks & Modifications

| # | What | Why | Files to Modify/Create | Difficulty | Status |
|---|------|-----|------------------------|-----------|--------|
| 0A-1 | **Controller bias injection hook** | A1.1 needs to perturb controller output at epoch 0 to shift initial alpha | `adjoint_samplers/train_loop.py` or new wrapper | Medium | todo |
| 0A-2 | **Frozen-asymmetric model variant** | A1.2 adds a fixed random bias vector to controller output every step | New model wrapper or config in `adjoint_samplers/components/model.py` | Medium | todo |
| 0A-3 | **Mode-aware split controller** | A1.2 needs two independent sub-networks, one per side of x-axis | New model class in `adjoint_samplers/components/model.py` + config | Medium | todo |
| 0A-4 | **Biased initialization utility** | A1.3 adds constant epsilon to final-layer weights after standard init | Small utility function, called from train.py | Easy | todo |
| 0A-5 | **Two-stage training (pretrain + finetune)** | A1.3 pretrains on 80/20, then fine-tunes on 50/50 | Resume-from-checkpoint logic (may already exist via Hydra) | Medium | todo |
| 0A-6 | **Over-trained AM mode** | A4.1 runs 10x more SGD steps per AM epoch | Config override `inner_steps=10x` or new training flag | Easy | todo |
| 0A-7 | **Kernel-based AM regression** | A4.1 replaces neural net with kernel regression on spatial grid | New matcher in `adjoint_samplers/components/matcher.py` | Hard | todo |
| 0A-8 | **Parameter trajectory saver** | A4.2 saves controller theta every N epochs for PCA | Hook in `train_loop.py` to dump `state_dict()` periodically | Easy | todo |

### 0B. New Experiment Runner Scripts (in `scripts/`)

These follow the existing sweep pattern (e10, e11, e13).

| # | Script Name | Experiment | What it does | Status |
|---|-------------|-----------|--------------|--------|
| 0B-1 | `v2_a11_bias_injection.py` | A1.1 | Sweep initial alpha_1 in {0.50..0.70}, 10 seeds, B1 sym d=6 | todo |
| 0B-2 | `v2_a12_architecture.py` | A1.2 | Run symmetric / frozen-asymmetric / mode-aware variants, 10 seeds | todo |
| 0B-3 | `v2_a13_init.py` | A1.3 | Sweep biased init epsilon + pretrained-asymmetric, 10 seeds | todo |
| 0B-4 | `v2_a23_asymmetry_phase.py` | A2.3 | Sweep w1 in {0.50..0.90}, B1 d=6, 10 seeds each | todo |
| 0B-5 | `v2_a32_capacity.py` | A3.2 | Sweep hidden_dim in {32..1024}, B1 sym + B5 cs=4, 5 seeds | todo |
| 0B-6 | `v2_a33_batchlr.py` | A3.3 | Sweep batch_size x lr grid, B1 sym, 5 seeds each | todo |
| 0B-7 | `v2_a41_exact_am.py` | A4.1 | Run standard / over-trained / kernel AM, B1 sym + B1 55/45 | todo |
| 0B-8 | `v2_a43_seed_variance.py` | A4.3 | B1 sym d in {3,6,10}, 50 seeds each | todo |

### 0C. New Analysis / Post-Hoc Scripts (in `scripts/`)

These load existing or new checkpoints and produce data — no training.

| # | Script Name | Experiment | What it does | Status |
|---|-------------|-----------|--------------|--------|
| 0C-1 | `v2_a21_growth_rate.py` | A2.1 | Load B5/B7 tracking, fit log-linear collapse rate, compare to J_max | todo |
| 0C-2 | `v2_a22_jacobian_multi.py` | A2.2 | Perturbation-based multi-epoch Jacobian on B1 w=(0.6,0.4) | todo |
| 0C-3 | `v2_a31_residual_decomp.py` | A3.1 | Decompose AM regression residual by mode (B1 sym + asym) | todo |
| 0C-4 | `v2_a42_trajectory_pca.py` | A4.2 | PCA on saved theta^(n), effective dim, symmetry of PCs | todo |
| 0C-5 | `v2_a51_full_jacobian.py` | A5.1 | Full J^eta eigenvalue spectrum from existing B5/B7/MB checkpoints | todo |
| 0C-6 | `v2_a52_eta_field.py` | A5.2 | Evaluate eta_k(x,t) on dense grid, produce heatmap animation | todo |
| 0C-7 | `v2_a53_adjoint_drift.py` | A5.3 | Estimate conditional bar{Y}_k drift across epochs | todo |
| 0C-8 | `v2_a54_symmetry_diag.py` | A5.4 | Measure symmetry residual |u(x,t) + u(Rx,t)| on B1/B5 | todo |

### 0D. New Evaluation Figure Scripts (in `evaluation/`)

| # | Script Name | What it produces | Status |
|---|-------------|-----------------|--------|
| 0D-1 | `fig_a11_bias_injection.py` | alpha_1 vs epoch for each initial condition; growth rate plot | todo |
| 0D-2 | `fig_a12_architecture.py` | Collapse rate bar chart by architecture variant | todo |
| 0D-3 | `fig_a13_init.py` | alpha_1 trajectory by init scheme; collapse rate vs epsilon | todo |
| 0D-4 | `fig_a21_growth_rate.py` | Empirical slope vs log(lambda_max) scatter | todo |
| 0D-5 | `fig_a22_jacobian_multi.py` | Predicted vs actual delta_alpha for n=1..10 | todo |
| 0D-6 | `fig_a23_asymmetry_phase.py` | Collapse severity vs |w1-0.5| (log-log) | todo |
| 0D-7 | `fig_a31_residual.py` | R1/R2 ratio vs epoch (sym vs asym) | todo |
| 0D-8 | `fig_a32_capacity.py` | Collapse rate vs network width (B1 sym + B5) | todo |
| 0D-9 | `fig_a33_batchlr.py` | Collapse rate heatmap (batch_size x lr) | todo |
| 0D-10 | `fig_a41_exact_am.py` | KL comparison bar chart: standard / over-trained / kernel | todo |
| 0D-11 | `fig_a42_pca.py` | PCA explained variance; PC symmetry scores | todo |
| 0D-12 | `fig_a43_seed_variance.py` | alpha_1 histograms (50 seeds) per d | todo |
| 0D-13 | `fig_a51_spectrum.py` | J^eta eigenvalue spectrum vs epoch | todo |
| 0D-14 | `fig_a52_eta_field.py` | eta_k heatmap grid at multiple epochs | todo |
| 0D-15 | `fig_a53_adjoint_drift.py` | |bar{Y}_k drift| vs epoch per mode | todo |
| 0D-16 | `fig_a54_symmetry.py` | Symmetry residual maps for B1/B5 | todo |

---

## Part 1: Training Runs

### Tier 1 (Critical — run first)

#### 1A. A1.1: Controller Bias Injection (B1 symmetric, d=6)

**Prereqs:** 0A-1, 0B-1
**Config:** `b1_asbs`, `w1=0.5`, `d=6`, 1000 epochs, 10 seeds per initial alpha

| # | Initial alpha_1 | Output Dir | Seeds | Status |
|---|-----------------|-----------|-------|--------|
| 1 | 0.50 (control) | `results/v2_a11/alpha_0.50/seed_{s}` | 0–9 | todo |
| 2 | 0.51 | `results/v2_a11/alpha_0.51/seed_{s}` | 0–9 | todo |
| 3 | 0.52 | `results/v2_a11/alpha_0.52/seed_{s}` | 0–9 | todo |
| 4 | 0.55 | `results/v2_a11/alpha_0.55/seed_{s}` | 0–9 | todo |
| 5 | 0.60 | `results/v2_a11/alpha_0.60/seed_{s}` | 0–9 | todo |
| 6 | 0.70 | `results/v2_a11/alpha_0.70/seed_{s}` | 0–9 | todo |

**Subtotal: 60 runs**

#### 1B. A2.3: Asymmetry-Resolved Phase Diagram (B1, d=6)

**Prereqs:** 0B-4
**Config:** `b1_asbs`, `d=6`, 10 seeds per w1

| # | w1 | Output Dir | Seeds | Status |
|---|-----|-----------|-------|--------|
| 7 | 0.50 | `results/v2_a23/w1_0.50/seed_{s}` | 0–9 | todo (reuse e7_b1_sym seeds 0–9) |
| 8 | 0.51 | `results/v2_a23/w1_0.51/seed_{s}` | 0–9 | running (batch 1, GPU0) |
| 9 | 0.52 | `results/v2_a23/w1_0.52/seed_{s}` | 0–9 | running (batch 1, GPU1) |
| 10 | 0.55 | `results/v2_a23/w1_0.55/seed_{s}` | 0–9 | queued (batch 2) |
| 11 | 0.60 | `results/v2_a23/w1_0.60/seed_{s}` | 0–9 | queued (batch 2) |
| 12 | 0.65 | `results/v2_a23/w1_0.65/seed_{s}` | 0–9 | queued (batch 3) |
| 13 | 0.70 | `results/v2_a23/w1_0.70/seed_{s}` | 0–9 | queued (batch 3) |
| 14 | 0.80 | `results/v2_a23/w1_0.80/seed_{s}` | 0–9 | queued (batch 4) |
| 15 | 0.90 | `results/v2_a23/w1_0.90/seed_{s}` | 0–9 | queued (batch 4) |

**Subtotal: 90 runs (some reusable from v1)**

#### 1C. A3.2: Network Capacity Ablation (B1 sym + B5 cs=4)

**Prereqs:** 0B-5
**Config:** Fixed depth=3, 5 seeds each

| # | Benchmark | hidden_dim | Output Dir | Seeds | Status |
|---|-----------|-----------|-----------|-------|--------|
| 16 | B1 sym | 32 | `results/v2_a32/b1_sym/hd_32/seed_{s}` | 0–4 | todo |
| 17 | B1 sym | 64 | `results/v2_a32/b1_sym/hd_64/seed_{s}` | 0–4 | todo |
| 18 | B1 sym | 128 | `results/v2_a32/b1_sym/hd_128/seed_{s}` | 0–4 | todo |
| 19 | B1 sym | 256 | `results/v2_a32/b1_sym/hd_256/seed_{s}` | 0–4 | todo |
| 20 | B1 sym | 512 | `results/v2_a32/b1_sym/hd_512/seed_{s}` | 0–4 | todo |
| 21 | B1 sym | 1024 | `results/v2_a32/b1_sym/hd_1024/seed_{s}` | 0–4 | todo |
| 22 | B5 cs=4 | 32 | `results/v2_a32/b5_cs4/hd_32/seed_{s}` | 0–4 | todo |
| 23 | B5 cs=4 | 64 | `results/v2_a32/b5_cs4/hd_64/seed_{s}` | 0–4 | todo |
| 24 | B5 cs=4 | 128 | `results/v2_a32/b5_cs4/hd_128/seed_{s}` | 0–4 | todo |
| 25 | B5 cs=4 | 256 | `results/v2_a32/b5_cs4/hd_256/seed_{s}` | 0–4 | todo |
| 26 | B5 cs=4 | 512 | `results/v2_a32/b5_cs4/hd_512/seed_{s}` | 0–4 | todo |
| 27 | B5 cs=4 | 1024 | `results/v2_a32/b5_cs4/hd_1024/seed_{s}` | 0–4 | todo |

**Subtotal: 60 runs (12 configs x 5 seeds)**

#### 1D. A5.4: Symmetry Diagnostic (no new training)

**Prereqs:** 0C-8
**Uses:** Existing B1 sym checkpoints (e7_b1_sym) + B5 checkpoints (sep_sweep_b5)
**Status:** todo (analysis only)

---

### Tier 2 (Mechanism identification)

#### 1E. A4.1: Exact vs Approximate AM (B1 sym + B1 55/45)

**Prereqs:** 0A-6, 0A-7, 0B-7

| # | Benchmark | AM Variant | Output Dir | Seeds | Status |
|---|-----------|-----------|-----------|-------|--------|
| 28 | B1 sym | Standard | `results/v2_a41/b1_sym/standard/seed_{s}` | 0–4 | todo (reuse) |
| 29 | B1 sym | Over-trained (10x steps) | `results/v2_a41/b1_sym/overtrained/seed_{s}` | 0–4 | todo |
| 30 | B1 sym | Kernel AM | `results/v2_a41/b1_sym/kernel/seed_{s}` | 0–4 | todo |
| 31 | B1 55/45 | Standard | `results/v2_a41/b1_55/standard/seed_{s}` | 0–4 | todo |
| 32 | B1 55/45 | Over-trained (10x steps) | `results/v2_a41/b1_55/overtrained/seed_{s}` | 0–4 | todo |
| 33 | B1 55/45 | Kernel AM | `results/v2_a41/b1_55/kernel/seed_{s}` | 0–4 | todo |

**Subtotal: 30 runs (6 configs x 5 seeds)**

#### 1F. A1.3: Initialization Symmetry Breaking (B1 sym, d=6)

**Prereqs:** 0A-4, 0A-5, 0B-3

| # | Init Variant | Param | Output Dir | Seeds | Status |
|---|-------------|-------|-----------|-------|--------|
| 34 | Standard (control) | — | `results/v2_a13/standard/seed_{s}` | 0–9 | todo (reuse) |
| 35 | Biased | eps=0.01 | `results/v2_a13/biased_0.01/seed_{s}` | 0–9 | todo |
| 36 | Biased | eps=0.05 | `results/v2_a13/biased_0.05/seed_{s}` | 0–9 | todo |
| 37 | Biased | eps=0.1 | `results/v2_a13/biased_0.1/seed_{s}` | 0–9 | todo |
| 38 | Biased | eps=0.3 | `results/v2_a13/biased_0.3/seed_{s}` | 0–9 | todo |
| 39 | Pretrained 80/20 | 50 ep pretrain | `results/v2_a13/pretrained/seed_{s}` | 0–9 | todo |

**Subtotal: 60 runs (6 variants x 10 seeds)**

#### 1G. A3.3: Batch Size & LR Ablation (B1 sym)

**Prereqs:** 0B-6

| # | batch_size | lr | Output Dir | Seeds | Status |
|---|-----------|-----|-----------|-------|--------|
| 40–59 | {32, 64, 128, 512, 2048} x {1e-5, 1e-4, 1e-3, 1e-2} | — | `results/v2_a33/bs_{bs}_lr_{lr}/seed_{s}` | 0–4 | todo |

**Subtotal: 100 runs (20 configs x 5 seeds)**

#### 1H. A2.1: Growth Rate Matching (no new training)

**Prereqs:** 0C-1
**Uses:** Existing B5 sep_sweep (cs=4,5,7) + B7 e13_sweep (kappa3=20) tracking data
**Status:** todo (analysis only)

---

### Tier 3 (Deeper characterization)

#### 1I. A1.2: Architecture Symmetry Breaking (B1 sym, d=6)

**Prereqs:** 0A-2, 0A-3, 0B-2

| # | Variant | Bias Scale | Output Dir | Seeds | Status |
|---|---------|-----------|-----------|-------|--------|
| 60 | Symmetric (control) | — | `results/v2_a12/symmetric/seed_{s}` | 0–9 | todo (reuse) |
| 61 | Frozen-asymmetric | 0.01 | `results/v2_a12/frozen_0.01/seed_{s}` | 0–9 | todo |
| 62 | Frozen-asymmetric | 0.05 | `results/v2_a12/frozen_0.05/seed_{s}` | 0–9 | todo |
| 63 | Frozen-asymmetric | 0.1 | `results/v2_a12/frozen_0.1/seed_{s}` | 0–9 | todo |
| 64 | Frozen-asymmetric | 0.5 | `results/v2_a12/frozen_0.5/seed_{s}` | 0–9 | todo |
| 65 | Mode-aware | — | `results/v2_a12/mode_aware/seed_{s}` | 0–9 | todo |

**Subtotal: 60 runs (6 variants x 10 seeds)**

#### 1J. A4.2: Parameter Trajectory PCA (B1 sym + B1 asym)

**Prereqs:** 0A-8, 0C-4
**Config:** Save theta every 10 epochs, 5 seeds each

| # | Benchmark | Output Dir | Seeds | Status |
|---|-----------|-----------|-------|--------|
| 66 | B1 sym | `results/v2_a42/b1_sym/seed_{s}` | 0–4 | todo |
| 67 | B1 80/20 | `results/v2_a42/b1_asym/seed_{s}` | 0–4 | todo |

**Subtotal: 10 runs**

#### 1K. A4.3: Seed Variance Analysis (B1 sym, 50 seeds)

**Prereqs:** 0B-8

| # | d | Output Dir | Seeds | Status |
|---|---|-----------|-------|--------|
| 68 | 3 | `results/v2_a43/d_3/seed_{s}` | 0–49 | todo |
| 69 | 6 | `results/v2_a43/d_6/seed_{s}` | 0–49 | todo |
| 70 | 10 | `results/v2_a43/d_10/seed_{s}` | 0–49 | todo |

**Subtotal: 150 runs (3 separations x 50 seeds)**

#### 1L. A2.2: Multi-Epoch Jacobian Consistency (B1 w=0.6/0.4)

**Prereqs:** 0C-2
**Config:** Train to approximate fixed point, then apply perturbation and track

| # | Benchmark | Output Dir | Seeds | Status |
|---|-----------|-----------|-------|--------|
| 71 | B1 w=(0.6, 0.4) | `results/v2_a22/seed_{s}` | 0–4 | todo |

**Subtotal: 5 runs + perturbation analysis**

---

### Tier N/A: Post-Hoc Analysis on Existing v1 Runs (no new training)

#### 1M. A5.1–A5.3: Detailed Evaluations

| # | Experiment | Prereqs | Uses | Status |
|---|-----------|---------|------|--------|
| 72 | A5.1: Full Jacobian spectrum | 0C-5 | B5, B7, MB existing checkpoints | todo |
| 73 | A5.2: eta_k field evolution | 0C-6 | B5, B7, MB existing checkpoints | todo |
| 74 | A5.3: Adjoint target drift | 0C-7 | B5, B7 existing checkpoints | todo |

---

## Part 2: Post-Hoc Reconstruction & Evaluation

After all training runs complete:

| Step | Command | Status |
|------|---------|--------|
| Reconstruct tracking for new runs | `python scripts/reconstruct_tracking.py --results-dir results --recursive --n-samples 10000` | todo |
| Run all v2 figure generators | `python evaluation/run_all.py` (after adding v2 generators) | todo |
| Compile RESULT.md | Manual / scripted compilation of all v2 findings | todo |
| Generate RESULT.pdf | `python scripts/md_to_pdf.py` | todo |

---

## Summary

| Part | Runs | Status |
|------|------|--------|
| **v1 (preserved, not re-run)** | **239** | **done** |
| Tier 1: A1.1 bias injection | 60 | todo |
| Tier 1: A2.3 asymmetry phase diagram | 90 | todo |
| Tier 1: A3.2 network capacity ablation | 60 | todo |
| Tier 1: A5.4 symmetry diagnostic | 0 (analysis) | todo |
| Tier 2: A4.1 exact vs approx AM | 30 | todo |
| Tier 2: A1.3 init symmetry breaking | 60 | todo |
| Tier 2: A3.3 batch/LR ablation | 100 | todo |
| Tier 2: A2.1 growth rate matching | 0 (analysis) | todo |
| Tier 3: A1.2 architecture ablation | 60 | todo |
| Tier 3: A4.2 parameter trajectory PCA | 10 | todo |
| Tier 3: A4.3 seed variance | 150 | todo |
| Tier 3: A2.2 multi-epoch Jacobian | 5 | todo |
| Tier N/A: A5.1–A5.3 existing eval | 0 (analysis) | todo |
| **Total new training runs** | **625** | |
| **Total new analysis-only tasks** | **5** | |
| Infrastructure scripts to write (Part 0) | 8 + 8 + 8 + 16 = **40 items** | |
