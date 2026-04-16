# Experiment Checklist

**Training:** `save_freq=10 eval_freq=99999 num_epochs=1500`
**Post-hoc:** `python scripts/reconstruct_tracking.py --results-dir results --recursive --n-samples 10000`
**Figures/tables:** `python evaluation/run_all.py`

---

## Part 1: Training Runs

Training runs produce checkpoints. All evaluation/visualization (E1–E13) is done **post-hoc** from these checkpoints — no separate "experiment runs" needed.

### 1A. Base Benchmark Training (3 seeds each)

Needed by: E1 (all), E2 (B3), E3 (B1,B5,B7), E4 (B1,B3,B7), E5 (all), E6 (B1,B3,B7), E9 (B1,B5,B6)

| # | Benchmark | Config | Params | Output Dir | Seeds | State |
|---|---|---|---|---|---|---|
| 1 | B1: 2-mode Gaussian (80/20) | `b1_asbs` | default | `results/b1_asbs/seed_{s}` | 0,1,2 | done |
| 2 | B2: Müller-Brown | `b2_asbs` | `beta=0.01` | `results/b2_asbs/seed_{s}` | 0,1,2 | done |
| 3 | B3: Warped Double Well | `b3_asbs` | default (γ=5) | `results/b3_asbs/seed_{s}` | 0,1,2 | done |
| 4 | B4: Neal's Funnel | `b4_asbs` | default | `results/b4_asbs/seed_{s}` | 0,1,2 | done |
| 5 | B5: Het. Covariance Mixture | `b5_asbs` | default (cs=5) | `results/b5_asbs/seed_{s}` | 0,1,2 | done |
| 6 | B6: 25-mode Power-Law Grid | `b6_asbs` | default | `results/b6_asbs/seed_{s}` | 0,1,2 | done |
| 7 | B7: Three-Well Metastable | `b7_asbs` | default (κ₃=8) | `results/b7_asbs/seed_{s}` | 0,1,2 | done |
| 8 | B8: LJ3 | `b8_asbs` | default | `results/b8_asbs/seed_{s}` | 0,1,2 | done |

**Subtotal: 24 runs**

### 1B. E7 Multi-Seed (pitchfork + ternary)

Extra seeds beyond the base 3. Needed by: E7

| # | Benchmark | Config | Params | Output Dir | Seeds | State |
|---|---|---|---|---|---|---|
| 9 | B1 symmetric | `b1_asbs` | `w1=0.5` | `results/e7_b1_sym/seed_{s}` | 0–29 | done |
| 10 | B7 | `b7_asbs` | default | `results/e7_b7/seed_{s}` | 0–19 | done |

**Subtotal: 50 runs**

### 1C. E8 + E10 Separation Sweep — B1 Symmetric

Shared between E8 (Jacobian threshold) and E10 (separation sweep). All use `w1=0.5`.

| # | d | mu1 | mu2 | Output Dir | Seeds | State |
|---|---|---|---|---|---|---|
| 11 | 2 | [-1, 0] | [1, 0] | `results/sep_sweep_b1/d_2/seed_{s}` | 0–4 | done |
| 12 | 3 | [-1.5, 0] | [1.5, 0] | `results/sep_sweep_b1/d_3/seed_{s}` | 0–4 | done |
| 13 | 4 | [-2, 0] | [2, 0] | `results/sep_sweep_b1/d_4/seed_{s}` | 0–4 | done |
| 14 | 5 | [-2.5, 0] | [2.5, 0] | `results/sep_sweep_b1/d_5/seed_{s}` | 0–4 | done |
| 15 | 6 | [-3, 0] | [3, 0] | `results/sep_sweep_b1/d_6/seed_{s}` | 0–4 | done |
| 16 | 7 | [-3.5, 0] | [3.5, 0] | `results/sep_sweep_b1/d_7/seed_{s}` | 0–4 | done |
| 17 | 8 | [-4, 0] | [4, 0] | `results/sep_sweep_b1/d_8/seed_{s}` | 0–4 | done |
| 18 | 9 | [-4.5, 0] | [4.5, 0] | `results/sep_sweep_b1/d_9/seed_{s}` | 0–4 | done |
| 19 | 10 | [-5, 0] | [5, 0] | `results/sep_sweep_b1/d_10/seed_{s}` | 0–4 | running |
| 20 | 12 | [-6, 0] | [6, 0] | `results/sep_sweep_b1/d_12/seed_{s}` | 0–4 | running |
| 21 | 15 | [-7.5, 0] | [7.5, 0] | `results/sep_sweep_b1/d_15/seed_{s}` | 0–4 | running |
| 22 | 20 | [-10, 0] | [10, 0] | `results/sep_sweep_b1/d_20/seed_{s}` | 0–4 | running |

Command template: `experiment=b1_asbs w1=0.5 energy.mu1=[{mu1}] energy.mu2=[{mu2}] hydra.run.dir=results/sep_sweep_b1/d_{d}/seed_{s}`

**Subtotal: 60 runs (12 separations × 5 seeds)**

### 1D. E10 Separation Sweep — B5 Heterogeneous

| # | center_scale | source scale | Output Dir | Seeds | State |
|---|---|---|---|---|---|
| 23 | 2 | 3 | `results/sep_sweep_b5/cs_2/seed_{s}` | 0–4 | done |
| 24 | 3 | 4 | `results/sep_sweep_b5/cs_3/seed_{s}` | 0–4 | done |
| 25 | 4 | 5 | `results/sep_sweep_b5/cs_4/seed_{s}` | 0–4 | done |
| 26 | 5 | 6 | `results/sep_sweep_b5/cs_5/seed_{s}` | 0–4 | done |
| 27 | 7 | 8 | `results/sep_sweep_b5/cs_7/seed_{s}` | 0–4 | done |
| 28 | 10 | 11 | `results/sep_sweep_b5/cs_10/seed_{s}` | 0–4 | done |

Command template: `experiment=b5_asbs center_scale={cs} scale={cs+1} hydra.run.dir=results/sep_sweep_b5/cs_{cs}/seed_{s}`

**Subtotal: 30 runs (6 scales × 5 seeds)**

### 1E. E11 Müller-Brown Temperature Sweep

| # | β | Output Dir | Seeds | State |
|---|---|---|---|---|
| 29 | 0.005 | `results/e11_sweep/beta_0.005/seed_{s}` | 0–4 | done |
| 30 | 0.01 | `results/e11_sweep/beta_0.01/seed_{s}` | 0–4 | done |
| 31 | 0.02 | `results/e11_sweep/beta_0.02/seed_{s}` | 0–4 | running |
| 32 | 0.05 | `results/e11_sweep/beta_0.05/seed_{s}` | 0–4 | running |
| 33 | 0.1 | `results/e11_sweep/beta_0.1/seed_{s}` | 0–4 | running |
| 34 | 0.2 | `results/e11_sweep/beta_0.2/seed_{s}` | 0–4 | running |

Command template: `experiment=b2_asbs beta={b} hydra.run.dir=results/e11_sweep/beta_{b}/seed_{s}`

**Subtotal: 30 runs (6 β × 5 seeds)**

### 1F. E13 Metastable Survival Sweep

| # | κ₃ | Output Dir | Seeds | State |
|---|---|---|---|---|
| 35 | 4 | `results/e13_sweep/kappa3_4/seed_{s}` | 0–4 | running |
| 36 | 6 | `results/e13_sweep/kappa3_6/seed_{s}` | 0–4 | running |
| 37 | 8 | `results/e13_sweep/kappa3_8/seed_{s}` | 0–4 | queued |
| 38 | 10 | `results/e13_sweep/kappa3_10/seed_{s}` | 0–4 | queued |
| 39 | 12 | `results/e13_sweep/kappa3_12/seed_{s}` | 0–4 | queued |
| 40 | 14 | `results/e13_sweep/kappa3_14/seed_{s}` | 0–4 | queued |
| 41 | 16 | `results/e13_sweep/kappa3_16/seed_{s}` | 0–4 | queued |
| 42 | 18 | `results/e13_sweep/kappa3_18/seed_{s}` | 0–4 | queued |
| 43 | 20 | `results/e13_sweep/kappa3_20/seed_{s}` | 0–4 | queued |

Command template: `experiment=b7_asbs kappa3={k} hydra.run.dir=results/e13_sweep/kappa3_{k}/seed_{s}`

**Subtotal: 45 runs (9 κ₃ × 5 seeds)**

---

## Part 2: Post-Hoc Evaluation (no training needed)

All of these read checkpoints from Part 1 runs. Run after training + reconstruction.

| Eval | What it produces | Reads from | Script |
|------|-----------------|------------|--------|
| E1: Mode tracking | α_k vs epoch curves, B6 heatmap | All base runs (1A) | `evaluation/fig_e1_mode_tracking.py` |
| E2: η_k heatmaps | Learned vs oracle regression weights | B3, B7 checkpoints | `evaluation/fig_e2_heatmaps.py` |
| E3: Loss decomposition | V_intra + V_inter stacked area | B1, B5, B7 checkpoints | `evaluation/fig_e3_loss_decomp.py` |
| E4: Controller field | Quiver plots + deficiency | B1, B3, B7 checkpoints | `evaluation/fig_e4_controller.py` |
| E5: Terminal density | KDE contours vs target, B6 scatter | All base checkpoints | `evaluation/fig_e5_density.py` |
| E6: Trajectories | Spaghetti plots at multiple epochs | B1, B3, B7 checkpoints | `evaluation/fig_e6_trajectories.py` |
| E7: Phase portrait | Pitchfork (B1) + ternary (B7) | E7 multi-seed runs (1B) | `evaluation/fig_e7_phase_portrait.py` |
| E8: Jacobian threshold | J₁₁ vs ρ_sep + collapse probability | Sep sweep (1C) | `evaluation/fig_e8_jacobian.py` |
| E9: KL decomposition | KL(α‖w), TV, dead modes vs epoch | B1, B5, B6 tracking | `evaluation/fig_e9_kl.py` |
| E10: Separation sweep | Collapse prob + time-to-collapse | Sep sweep B1 (1C) + B5 (1D) | `evaluation/fig_e10_separation.py` |
| E11: Temperature sweep | Per-β panels + summary | MB sweep (1E) | `evaluation/fig_e11_temperature.py` |
| E12: Death cascade | Survival curve, death order, grid snapshots | B6 tracking | `evaluation/fig_e12_cascade.py` |
| E13: Metastable survival | Survival prob + α₃/w₃ vs depth ratio | κ₃ sweep (1F) | `evaluation/fig_e13_metastable.py` |
| Tables | LaTeX summary tables | All tracking files | `evaluation/tables.py` |

**Command:** `python evaluation/run_all.py` (runs all of the above)

---

## Summary

| Part | Runs | Status |
|------|------|--------|
| 1A: Base benchmarks (8 × 3 seeds) | 24 | done |
| 1B: E7 multi-seed (30 + 20) | 50 | done |
| 1C: Separation sweep B1 (12 × 5 seeds) | 60 | done |
| 1D: Separation sweep B5 (6 × 5 seeds) | 30 | done |
| 1E: MB temperature sweep (6 × 5 seeds) | 30 | done |
| 1F: Metastable κ₃ sweep (9 × 5 seeds) | 45 | running (queued, 10/batch on GPU 1) |
| **Total training runs** | **239** | |
| Post-hoc reconstruction | 1 command | pending |
| Evaluation + figures | 1 command | pending |

Estimated GPU-hours: 239 runs × ~17h each (1500 ep × 40s/ep) ≈ **4,063 GPU-h**
With 2× A100 (2 parallel): ~2,032h wall-clock ≈ **85 days on this server**.
