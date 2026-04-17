# Experiment Checklist: Verifying Stability of Collapsed States (v3)

Reference: `Experimental Guide v3: Verifying Stability of Collapsed States.md`

---

## Tier 0 — v1 Reinterpretation (No new training; analysis only)

- [ ] **0.1: v1 Reachability as Basin Evidence**
  - Re-analyze B5 (5/5 seeds collapse at center_scale >= 5), B7 (20/20 seeds same ternary pattern), Muller-Brown (beta >= 0.1 mode death)
  - Reframe as "attractor-driven collapses" under new stability theory

- [ ] **0.2: v1 Symmetric B1 as Balanced-State Stability**
  - Re-analyze B1 symmetric (30/30 seeds at alpha_1 = 0.5, 0% collapse across d in {2,...,20})
  - Show balanced state is S={1,...,K} case of collapsed controller family — resolves v1 tension

- [ ] **0.3: v1 Seed Consistency as Basin Size Evidence**
  - Re-analyze B7 seed consistency (20 seeds all mode-3-dominant)
  - Characterize basin sizes from seed convergence patterns

- [ ] **0.4: v1 Separation/Temperature Sweeps as Basin Phase Transitions**
  - Re-analyze B5 transition at center_scale ~ 3.5, Muller-Brown at beta ~ 0.1
  - Reframe as "phase transition in basin geometry"

- [ ] **0.5: v1 Metastable Survival Sweep as Basin Threshold**
  - Re-analyze 1F survival curve (100% -> 60% as kappa_3 rises)
  - Interpret transition as basin size shift between balanced and collapsed attractors

---

## Tier 1 — Direct Stability Tests (Highest priority new experiments)

- [ ] **B1: Dead Mode Revival via Data Injection** ⭐ MOST IMPORTANT
  - Pretrain ASBS on B7 to convergence (mode 2 dead)
  - Inject fraction rho in {0.001, 0.01, 0.05, 0.1} of dead-mode samples for M in {10, 50, 200} epochs
  - Continue normal training for 1000 epochs after injection
  - Measure alpha_2 decay rate lambda after injection ends
  - Compare lambda to theoretical J^{S^c S^c} from Proposition D2'

- [ ] **A1: Sequential Mode Addition on 2D Gaussian Mixture**
  - Phase 1: Pretrain on 2-mode GMM (w1=w2=0.5, centers +/-4, sigma=1) to convergence
  - Phase 2: Switch target to 3-mode (add mode at (0,4), w_k=1/3 each), train 2000 more epochs
  - Measure alpha_k trajectories, time to discover mode 3, final KL
  - Baseline: train from scratch on 3-mode target

- [ ] **C2: Start from Adversarial Collapsed States**
  - Pretrain B7 on modified target with w1=1 (single-mode)
  - Switch to true B7 target, continue training
  - Measure whether alpha_1 stays dominant or transitions to alpha_3-dominant
  - Tests plurality of stable attractors

---

## Tier 2 — Quantitative Jacobian Verification

- [ ] **D1: Block-by-Block Jacobian Estimation**
  - At B7 collapsed state, apply perturbation delta_alpha = epsilon * e_j for each mode j
  - epsilon in {0.001, 0.005, 0.01, 0.02}
  - Extract J^{SS}, J^{SS^c}, J^{S^c S^c} blocks
  - Compare to Proposition D2' formulas (within factor 2 = success)

- [ ] **D2: Spectral Radius Measurement**
  - From D1's Jacobian, compute eigenvalues
  - Verify |lambda_max| < 1

- [ ] **D3: Multi-Epoch Perturbation Decay**
  - Inject known perturbation, track ||delta_alpha^(n)|| over many epochs
  - Fit exponential decay rate r
  - Compare r to |lambda_max| from D2

- [ ] **B2: Controller-Level Perturbation**
  - Pretrain B7 to convergence
  - Add Gaussian noise sigma_pert in {0.001, 0.01, 0.1, 1.0} to all controller weights
  - Train 1000 more epochs
  - Measure recovery time; check if same or different collapsed state

- [ ] **F1: Direct Measurement of Dead Mode Adjoints**
  - At B7 collapsed state, compute Y_bar_2 for trajectories reaching M_2
  - Monte Carlo with 10^6 samples
  - Measure value and variance across seeds

- [ ] **F2: Sensitivity of Dead Mode Adjoints**
  - Perturb controller slightly, measure dY_bar_2/d_alpha_j for each j
  - Check if bounded (BRA holds) or diverges as epsilon -> 0

---

## Tier 3 — Scope and Basin Mapping

- [ ] **A2: Mode Addition with Varying Injection Distance**
  - Same as A1 but sweep d_inject in {2, 4, 6, 8, 12}
  - Map basin of attraction in injection-distance space

- [ ] **A3: Mode Addition with Energy Depth Variation**
  - Inject new mode with kappa_3 in {4, 8, 16, 32}, distance fixed
  - Test depth vs. rejection robustness

- [ ] **C1: Initialization-to-Collapse Distance Sweep**
  - Initialize theta_0 = theta* + d_init * xi, d_init in {0.01, 0.1, 1.0, 10.0}
  - Train 1000 epochs, measure convergence rate and fraction returning to same state

- [ ] **E1: Attractor Enumeration on Small Benchmarks**
  - B1 asymmetric (80/20) or 3-mode benchmark
  - For each subset S: pretrain on p_S, switch to full p, check stability
  - Measure fraction of seeds remaining in S-state per subset

- [ ] **E2: Transition Threshold Between Collapsed States**
  - Start at "only mode 1 alive" for B7
  - Inject mode 2 revival at increasing magnitudes
  - Find transition threshold — maps separatrix between basins

- [ ] **B3: Bias Injection with Continuous Training**
  - Pretrain to convergence, inject small asymmetric drift (delta=0.01) during rollouts
  - Measure how much asymmetry accumulates before being pushed back

---

## Supplementary Analyses on v1 Data

- [ ] **G1: Empirical Contraction Factor from v1 Trajectories**
  - Extract late-epoch decay rate from v1 B5, B7, Muller-Brown training curves
  - Compare to theoretical |lambda_max(J^{(S)})| from Proposition D2'

- [ ] **G2: Jacobian Estimation from Small Fluctuations**
  - Compute autocorrelation of alpha^(n) in converged v1 runs
  - Extract autoregression coefficients as Jacobian eigenvalue estimates
  - Verify modulus < 1

---

## Summary Table

| Family | # Experiments | Status |
|--------|--------------|--------|
| 0 (v1 reinterpretation) | 5 | Not started |
| A (mode injection) | 3 | Not started |
| B (direct perturbation) | 3 | Not started |
| C (collapsed init) | 2 | Not started |
| D (Jacobian verification) | 3 | Not started |
| E (multiple attractors) | 2 | Not started |
| F (BRA verification) | 2 | Not started |
| G (v1 supplementary) | 2 | Not started |
| **Total** | **22** | |
