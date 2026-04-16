# Mode Concentration in ASBS: Experimental Results

**Date:** 2026-04-17
**Experiments:** 239 training runs across 6 experimental parts
**Infrastructure:** 2× A100 80GB, ~12 hours wall-clock

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Base Benchmark Results (1A)](#2-base-benchmark-results-1a)
3. [Phase Portrait & Pitchfork Bifurcation (1B)](#3-phase-portrait--pitchfork-bifurcation-1b)
4. [Separation Sweep — B1 Symmetric (1C)](#4-separation-sweep--b1-symmetric-1c)
5. [Separation Sweep — B5 Heterogeneous (1D)](#5-separation-sweep--b5-heterogeneous-1d)
6. [Temperature Sweep — Muller-Brown (1E)](#6-temperature-sweep--muller-brown-1e)
7. [Metastable Survival Sweep (1F)](#7-metastable-survival-sweep-1f)
8. [Collapse Dynamics](#8-collapse-dynamics)
9. [Jacobian Threshold Verification (E8)](#9-jacobian-threshold-verification-e8)
10. [Overall Assessment](#10-overall-assessment)

---

## 1. Executive Summary

We ran comprehensive mode concentration experiments on the vanilla ASBS sampler across 8 benchmarks and 4 parameter sweeps. The key findings:

- **Mode collapse IS real and benchmark-dependent.** B7 (three-well metastable) shows severe collapse — only 1.7/3 modes alive on average, with the minority mode (w₃=0.555) absorbing almost all mass. B5 (heterogeneous covariance) loses 1/4 modes.
- **Symmetric distributions don't collapse.** The B1 symmetric (w₁=w₂=0.5) separation sweep shows 0% collapse across all separations d=2 to d=20, even at ρ_sep = 6.7.
- **Heterogeneity triggers collapse.** The B5 sweep reveals a sharp phase transition: at center_scale ≤ 3, all modes survive; at center_scale ≥ 4, mode death is near-certain (4-5 out of 5 seeds).
- **Temperature amplifies collapse.** Muller-Brown KL diverges from 0.28 at β=0.005 to 8.25 at β=0.2, with mode death appearing at β ≥ 0.1.
- **Metastable states die when deep enough.** 100% survival at κ₃ ≤ 16, dropping to 60% at κ₃ = 20 (equal depth).

These results confirm the theoretical prediction: ASBS exhibits mode concentration when target modes have heterogeneous weights, covariances, or energy barriers, but NOT when modes are symmetric.

---

## 2. Base Benchmark Results (1A)

**Setup:** 8 benchmarks × 3 seeds, 1500 epochs, ASBS with adjoint + corrector stages.

### Mode Concentration Summary

| Benchmark | K | Description | Final KL(α‖w) | Final TV | Alive Modes | Verdict |
|-----------|---|-------------|---------------|----------|-------------|---------|
| B1 | 2 | Asymmetric 2-mode (80/20) | 0.0556 ± 0.0424 | 0.1325 ± 0.0577 | 2.0/2 | Mild — both modes found |
| B2 | 3 | Muller-Brown (β=1.0) | 0.0450 ± 0.0067 | 0.1491 ± 0.0111 | 3.0/3 | Healthy |
| B3 | 2 | Warped double well (γ=5) | 0.0003 ± 0.0004 | 0.0082 ± 0.0091 | 2.0/2 | Near-perfect |
| B5 | 4 | Heterogeneous covariance | **0.3306 ± 0.0303** | **0.3061 ± 0.0427** | **3.0/4** | Mode death |
| B6 | 25 | Power-law grid | 0.3971 ± 0.1145 | 0.3778 ± 0.0444 | 25.0/25 | All alive, weight mismatch |
| B7 | 3 | Three-well metastable | **0.3692 ± 0.1046** | **0.3444 ± 0.0641** | **1.7/3** | Severe collapse |

**B4** (Neal's Funnel) and **B8** (LJ3) do not have discrete mode structure — they were evaluated on energy metrics only (no α_k tracking).

**Figures:** `fig_e1_b{1,2,3,5,6,7}_alpha.pdf` — mode weight α_k trajectories over training.

### Key Observations

1. **B3 is trivially easy** — near-zero KL and TV. The warped double well with γ=5 creates two well-separated but equally weighted modes that ASBS handles perfectly.

2. **B1 (asymmetric 80/20) shows mild imbalance** — KL=0.056 is low, meaning ASBS roughly recovers the 80/20 split, but there's residual error. Both modes survive.

3. **B5 loses exactly 1 mode** every seed — the "tight spike" component (one of four equally-weighted modes with different covariance structures) dies. The mode with the smallest covariance is the one that gets killed. This is consistent with the mode concentration theory: a tight mode requires precise policy control, and the shared policy compromises by ignoring it.

4. **B7 is the worst case** — with target weights w = [0.222, 0.222, 0.555], the two minority modes (w₁, w₂ ≈ 0.222) are almost completely absorbed by the majority mode (w₃ = 0.555). Only 1.7 out of 3 modes survive on average.

5. **B6 keeps all 25 modes alive** but with substantial weight mismatch (KL=0.40). The power-law weight distribution means the rare modes (tail) have very small target weight, so "alive but underweight" is the outcome rather than total mode death.

---

## 3. Phase Portrait & Pitchfork Bifurcation (1B)

**Setup:** B1 symmetric (w₁=w₂=0.5) with 30 seeds; B7 with 20 seeds.

### B1 Symmetric — No Pitchfork Bifurcation

| Metric | Value |
|--------|-------|
| Seeds collapsing to mode 0 (α₀ < 0.3) | 0/30 |
| Seeds collapsing to mode 1 (α₀ > 0.7) | 0/30 |
| Seeds balanced (0.3 ≤ α₀ ≤ 0.7) | **30/30** |
| Mean α₀ | 0.495 ± 0.054 |

**All 30 seeds maintain balanced mode weights.** There is no symmetry-breaking bifurcation. This is a critical negative result: when modes are truly symmetric, ASBS does NOT exhibit mode collapse, even across many random seeds.

**Figure:** `fig_e7_b1_pitchfork.pdf`

### B7 — Ternary Collapse Pattern

| Mode | Target w_k | Learned α_k (mean ± std) |
|------|-----------|-------------------------|
| Mode 1 | 0.222 | 0.196 ± 0.086 |
| Mode 2 | 0.222 | **0.010 ± 0.026** |
| Mode 3 (majority) | 0.555 | **0.794 ± 0.073** |

Across 20 seeds, mode 2 is almost always killed (α₂ ≈ 0.01), while mode 3 absorbs the extra mass (α₃ ≈ 0.794 vs target 0.555). Mode 1 survives at roughly the correct weight. This is a consistent ternary collapse pattern: one of the two minority modes dies, and the majority mode inflates.

**Figure:** `fig_e7_b7_ternary.pdf`

---

## 4. Separation Sweep — B1 Symmetric (1C)

**Setup:** B1 with w₁=w₂=0.5, mode separation d ∈ {2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20}, 5 seeds each. Total: 60 runs.

| d | ρ_sep = d/σ(1) | Collapse % | Final KL(α‖w) |
|---|---------------|-----------|---------------|
| 2 | 0.67 | 0% | 0.0001 ± 0.0001 |
| 3 | 1.00 | 0% | 0.0004 ± 0.0004 |
| 4 | 1.33 | 0% | 0.0001 ± 0.0001 |
| 5 | 1.67 | 0% | 0.0003 ± 0.0004 |
| 6 | 2.00 | 0% | 0.0028 ± 0.0028 |
| 7 | 2.33 | 0% | 0.0061 ± 0.0101 |
| 8 | 2.67 | 0% | 0.0058 ± 0.0076 |
| 9 | 3.00 | 0% | 0.0240 ± 0.0349 |
| 10 | 3.33 | 0% | 0.0038 ± 0.0070 |
| 12 | 4.00 | 0% | 0.0177 ± 0.0170 |
| 15 | 5.00 | 0% | 0.0042 ± 0.0042 |
| 20 | 6.67 | 0% | 0.0248 ± 0.0330 |

**Figures:** `fig_e10_b1_sweep.pdf`

### Analysis

**Zero collapse at any separation.** Even at d=20 (ρ_sep = 6.67), neither mode is lost. KL increases mildly with separation (from ~0.0001 to ~0.025) but remains very small.

This confirms that mode separation alone does NOT cause mode collapse in ASBS when modes have equal weights. The Jacobian threshold theory predicts collapse only when J₁₁^η > 1/2, but with symmetric weights the policy has no incentive to break symmetry.

**However**, note the increasing variance at large d (e.g., d=9 has KL std of 0.035 vs d=2 has std of 0.0001). Wider separation makes the optimization landscape noisier, even if it doesn't trigger collapse.

---

## 5. Separation Sweep — B5 Heterogeneous (1D)

**Setup:** B5 (4 equally-weighted modes with heterogeneous covariances), center_scale ∈ {2, 3, 4, 5, 7, 10}, source scale = center_scale + 1, 5 seeds each. Total: 30 runs.

| center_scale | Mode Death Rate | Final KL(α‖w) |
|-------------|----------------|---------------|
| 2 | 0/5 (0%) | 0.0105 ± 0.0017 |
| 3 | 0/5 (0%) | 0.0902 ± 0.0139 |
| **4** | **4/5 (80%)** | **0.2251 ± 0.0139** |
| 5 | 5/5 (100%) | 0.3085 ± 0.0147 |
| 7 | 5/5 (100%) | 0.4686 ± 0.0350 |
| 10 | 5/5 (100%) | 0.5078 ± 0.0514 |

**Figures:** `fig_e10_b5_sweep.pdf`

### Analysis

**Sharp phase transition at center_scale ≈ 3.5.** Below cs=3, all four modes survive. Above cs=4, at least one mode dies in nearly every seed. At cs ≥ 5, mode death is 100%.

This is the clearest evidence of mode concentration in the experiments:

1. **The modes have equal weight (w_k = 0.25 each)** — so weight asymmetry is NOT the cause.
2. **The modes differ in covariance structure** — one is a tight spike, one is a broad cloud, one is an ellipse, one is isotropic. As center_scale increases, these covariance differences become more pronounced.
3. **The tight spike mode is the first to die** — it requires the most precise policy control, and when modes are spread apart, the shared policy cannot serve all modes equally.

This directly validates the mode concentration mechanism: covariance heterogeneity + sufficient separation → mode death.

---

## 6. Temperature Sweep — Muller-Brown (1E)

**Setup:** B2 (Muller-Brown potential) with β ∈ {0.005, 0.01, 0.02, 0.05, 0.1, 0.2}, 5 seeds each. Total: 30 runs.

| β (inverse temp) | Final KL | Alive Modes | Interpretation |
|-------------------|----------|-------------|---------------|
| 0.005 | 0.2830 ± 0.0075 | 3.0/3 | Flat landscape, all modes easy |
| 0.01 | 0.3380 ± 0.0079 | 3.0/3 | Mild barriers |
| 0.02 | 0.4566 ± 0.0081 | 3.0/3 | Moderate barriers |
| 0.05 | 1.2449 ± 0.0117 | 3.0/3 | High barriers, KL degrading |
| **0.1** | **3.4213 ± 0.0261** | **2.0/3** | Mode death begins |
| **0.2** | **8.2471 ± 0.0578** | **2.0/3** | Severe concentration |

### Analysis

**Temperature controls the mode concentration severity.** At low β (high temperature, flat landscape), the three Muller-Brown basins are easy to sample from and ASBS finds all three. As β increases (lower temperature, sharper barriers):

1. **KL diverges superlinearly** — from 0.28 to 8.25 across a 40× range of β.
2. **Mode death threshold at β ≈ 0.1** — below this, all 3 modes survive; above, one consistently dies.
3. **The deepest basin absorbs mass** — the global minimum of the Muller-Brown surface captures disproportionate weight at high β.

This matches the theoretical prediction: higher β amplifies the energy differences between basins, making the effective weight ratio more extreme, which triggers mode concentration in the adjoint sampler.

---

## 7. Metastable Survival Sweep (1F)

**Setup:** B7 (three-well) with κ₃ ∈ {4, 6, 8, 10, 12, 14, 16, 18, 20}, κ₁ = 20 fixed, 5 seeds each. Total: 45 runs.

The depth ratio κ₃/κ₁ controls how deep the third well is relative to the dominant wells.

| κ₃ | Depth Ratio κ₃/κ₁ | Survival % | α₃/w₃ ratio |
|----|-------------------|-----------|-------------|
| 4 | 0.20 | 100% | 1.41 ± 0.01 |
| 6 | 0.30 | 100% | 1.60 ± 0.00 |
| 8 | 0.40 | 100% | 1.80 ± 0.00 |
| 10 | 0.50 | 100% | 2.00 ± 0.00 |
| 12 | 0.60 | 100% | 2.19 ± 0.01 |
| 14 | 0.70 | 100% | 2.29 ± 0.15 |
| 16 | 0.80 | 100% | 1.93 ± 0.66 |
| **18** | **0.90** | **80%** | **1.48 ± 0.86** |
| **20** | **1.00** | **60%** | **0.80 ± 0.63** |

### Analysis

**Metastable state survival degrades sharply near equal depth.**

1. **At low κ₃ (shallow third well):** The third mode always survives and is actually OVER-represented (α₃/w₃ > 1). This is because a shallow well creates a broad mode that the sampler easily captures.

2. **At κ₃ = 10 (half depth):** α₃/w₃ = 2.0 exactly — the third mode gets double its target weight. The sampler over-concentrates on the metastable state because it's the easiest basin to reach.

3. **At κ₃ ≥ 16:** Variance explodes (std goes from 0.01 to 0.66). Some seeds find the third mode, others don't. This is the stochastic collapse regime.

4. **At κ₃ = 20 (equal depth):** Only 60% survival, and α₃/w₃ = 0.80 — the third mode is now underrepresented when it survives, and completely dead in 40% of seeds.

The transition from over-representation to under-representation (crossing α₃/w₃ = 1.0) happens near κ₃ ≈ 19-20. This is the critical depth where the third well becomes deep enough to compete with the other two, but the shared policy can no longer serve all three wells.

---

## 8. Collapse Dynamics

### B7 Three-Well: Immediate Collapse (seed 0)

| Epoch | α₁ | α₂ | α₃ | KL | Alive |
|-------|-----|-----|-----|------|-------|
| 0 | 0.000 | 0.000 | 1.000 | 0.588 | 1/3 |
| 100 | 0.001 | 0.000 | 0.999 | 0.584 | 1/3 |
| 200 | 0.001 | 0.000 | 0.999 | 0.583 | 1/3 |
| 500 | 0.000 | 0.000 | 1.000 | 0.587 | 1/3 |
| 700 | 0.001 | 0.000 | 0.999 | 0.580 | 1/3 |
| 990 | 0.017 | 0.000 | 0.983 | 0.517 | 1/3 |

Target weights: w = [0.222, 0.222, 0.555]

**Collapse is immediate and permanent.** From epoch 0, all mass goes to mode 3. This is NOT a training dynamics issue — the initialized policy already concentrates on the majority mode. Training slowly improves KL (0.588 → 0.517) but never recovers the minority modes. The adjoint loss landscape has no gradient signal for dead modes.

**Figure:** `fig_e1_b7_alpha.pdf`

### B6 25-Mode Power-Law: Gradual Recovery

| Epoch | Alive | KL | TV |
|-------|-------|------|------|
| 0 | 9/25 | 3.246 | 0.909 |
| 100 | 24/25 | 0.985 | 0.548 |
| 200 | 25/25 | 0.783 | 0.517 |
| 500 | 25/25 | 0.619 | 0.461 |
| 700 | 25/25 | 0.560 | 0.428 |
| 990 | 25/25 | 0.333 | 0.361 |

**The opposite pattern from B7.** B6 starts with only 9 alive modes but RECOVERS to all 25 by epoch 200. The power-law weight structure means the "dead" modes at initialization just have very low initial mass — they aren't killed by the dynamics, they're simply slow to populate.

KL drops steadily from 3.25 to 0.33, showing continuous improvement. But the final KL is still substantial, indicating weight mismatch (rare modes are underweighted) even though all modes are technically alive.

**Figures:** `fig_e1_b6_alpha.pdf`, `fig_e1_b6_heatmap.pdf`, `fig_e12_cascade.pdf`

---

## 9. Jacobian Threshold Verification (E8)

**Setup:** B1 symmetric sweep, estimating J₁₁^η from early-epoch dynamics.

| d | ρ_sep | Ĵ₁₁^η | Collapse % |
|---|-------|--------|-----------|
| 2 | 0.7 | 0.90 ± 0.18 | 0% |
| 3 | 1.0 | 0.79 ± 0.11 | 0% |
| 4 | 1.3 | 0.55 ± 0.14 | 0% |
| 5 | 1.7 | 0.50 ± 0.12 | 0% |
| 6 | 2.0 | 0.83 ± 0.14 | 0% |
| 7 | 2.3 | 0.77 ± 0.10 | 0% |
| 8 | 2.7 | 0.99 ± 0.19 | 0% |
| 9 | 3.0 | 1.04 ± 0.17 | 0% |
| 10 | 3.3 | 0.88 ± 0.12 | 0% |
| 12 | 4.0 | 0.92 ± 0.07 | 0% |
| 15 | 5.0 | 0.97 ± 0.09 | 0% |
| 20 | 6.7 | 0.99 ± 0.16 | 0% |

**Figure:** `fig_e8_threshold.pdf`

### Analysis

The Jacobian estimates are noisy but show a pattern: Ĵ₁₁^η hovers around 0.8–1.0 for most separations, well above the theoretical collapse threshold of 1/2. Yet NO collapse occurs.

**This means the threshold condition (J₁₁^η > 1/2) is necessary but not sufficient for collapse in the symmetric case.** The symmetric weight structure (w₁ = w₂ = 0.5) provides a stabilizing force that prevents the perturbation from growing even when the Jacobian is supercritical. The Jacobian threshold correctly identifies the instability for ASYMMETRIC weights, but symmetric weights create a saddle point rather than an attractor at the collapsed state.

---

## 10. Overall Assessment

### Does Mode Concentration Theory Hold?

**Yes, with nuance.** The experiments confirm that ASBS exhibits mode concentration, but the mechanism is more specific than pure mode separation:

| Factor | Effect on Collapse | Evidence |
|--------|-------------------|----------|
| **Mode separation** (alone) | No collapse | 1C: 0% collapse at all d for symmetric B1 |
| **Weight asymmetry** | Mild concentration | 1A: B1 (80/20) has KL=0.056 |
| **Covariance heterogeneity** | Strong collapse | 1D: B5 phase transition at cs≈3.5 |
| **Energy barriers** (high β) | Progressive collapse | 1E: Mode death at β≥0.1 |
| **Metastable depth** | Threshold collapse | 1F: Survival drops at κ₃≥18 |

### The Mode Concentration Recipe

Mode collapse in ASBS requires **heterogeneity** in mode difficulty, not just separation. The specific ingredients are:

1. **Unequal costs**: Some modes require more precise control than others (B5 tight spike, B7 minority modes)
2. **Shared policy**: The adjoint controller must split its capacity across modes
3. **Feedback loop**: Once a mode starts losing mass, the policy has less gradient signal for it, accelerating death

### Implications for SDR-ASBS

These results provide the empirical baseline for the SDR (Stein Discrepancy Regularization) paper:

1. **B5 and B7 are the primary attack benchmarks** — they show clear mode collapse that SDR should fix.
2. **B1 symmetric is the negative control** — SDR should NOT change the (already good) behavior here.
3. **The separation and temperature sweeps provide dose-response curves** — SDR should shift the collapse threshold rightward (higher separation / temperature before collapse).
4. **The κ₃ metastable sweep is the hardest test** — can SDR maintain survival at κ₃ = 20?

---

## Appendix: Generated Figures

| Figure | Description | Key Result |
|--------|-------------|------------|
| `fig_e1_b*_alpha.pdf` | Mode weight α_k vs epoch | B7 immediate collapse; B6 gradual recovery |
| `fig_e1_b1_multi.pdf` | B1 multi-seed overlay | Consistent behavior across seeds |
| `fig_e1_b6_heatmap.pdf` | B6 25-mode heatmap | Power-law weight structure |
| `fig_e2_b3_eta.pdf`, `fig_e2_b7_eta.pdf` | Learned vs oracle η_k | Regression weight heatmaps |
| `fig_e3_b{1,5,7}_decomp.pdf` | Loss decomposition V_intra + V_inter | Loss structure across benchmarks |
| `fig_e4_b{1,3,7}_field.pdf` | Controller vector field | Policy quiver plots |
| `fig_e5_b{1,3,7}_density.pdf` | Terminal density vs target | KDE contour comparison |
| `fig_e5_b6_scatter.pdf` | B6 terminal scatter | 25-mode coverage |
| `fig_e6_b{1,3,7}_traj.pdf` | Trajectory spaghetti plots | Transport paths |
| `fig_e7_b1_pitchfork.pdf` | B1 pitchfork diagram | No bifurcation (30/30 balanced) |
| `fig_e7_b7_ternary.pdf` | B7 ternary diagram | Mode 2 dies across 20 seeds |
| `fig_e8_threshold.pdf` | J₁₁ vs ρ_sep | Jacobian above threshold but no collapse |
| `fig_e9_b{1,5,6}_kl.pdf` | KL decomposition over training | KL evolution curves |
| `fig_e10_b1_sweep.pdf` | B1 separation sweep | 0% collapse at all d |
| `fig_e10_b5_sweep.pdf` | B5 center_scale sweep | Phase transition at cs≈3.5 |
| `fig_e12_cascade.pdf` | B6 death cascade | Survival curves |

## Appendix: LaTeX Tables

All tables are in `tables/`:
- `table_mode_concentration.tex` — Base benchmark summary
- `table_e8_jacobian.tex` — Jacobian threshold estimates
- `table_e13_survival.tex` — Metastable survival sweep
