# Experiment Checklist: Verifying Stability of Collapsed States (v3)

Reference: `Experimental Guide v3: Verifying Stability of Collapsed States.md`
Seeds: 3 per condition (seeds 0, 1, 2). Up to 1000 epochs per run (prereqs up to 2000).

---

## Training Run Inventory

### Batch 1 — Baselines & Phase-1 Pretrains (20 runs, launched 2026-04-17)

| # | GPU | Experiment | Seeds | Epochs | Status |
|---|-----|-----------|-------|--------|--------|
| 1-3 | 0 | B7 baseline | 0,1,2 | 2000 | 🔄 Running |
| 4-6 | 0 | B1_sym baseline | 0,1,2 | 2000 | 🔄 Queued |
| 7-9 | 0 | C2 phase1 mode1 | 0,1,2 | 1000 | 🔄 Queued |
| 10 | 0 | C2 phase1 mode2 | 0 | 1000 | 🔄 Queued |
| 11-12 | 1 | C2 phase1 mode2 | 1,2 | 1000 | 🔄 Running |
| 13-15 | 1 | C2 phase1 mode3 | 0,1,2 | 1000 | 🔄 Queued |
| 16-18 | 1 | E1 phase1 S12 | 0,1,2 | 1000 | 🔄 Queued |
| 19-20 | 1 | E1 phase1 S13 | 0,1 | 1000 | 🔄 Queued |

**Unlocks:** B7 baselines → B2, C1, D1, F1, F2. B1_sym → A1, A2, A3. C2p1 → C2p2, E1p2-single.

### Batch 2 — B1 Injection (⭐ most important) + A1 + remaining prereqs (20 runs, auto-queued)

| # | GPU | Experiment | Seeds | Epochs | Status |
|---|-----|-----------|-------|--------|--------|
| 1-3 | 0 | B1 inject rho=0.001/M=50 | 0,1,2 | 3000 | ⏳ After batch 1 |
| 4-6 | 0 | B1 inject rho=0.01/M=50 | 0,1,2 | 3000 | ⏳ After batch 1 |
| 7-9 | 0 | A1: 2→3 mode switch | 0,1,2 | 2000 | ⏳ After batch 1 |
| 10 | 0 | E1 phase1 S13 | 2 | 1000 | ⏳ After batch 1 |
| 11-13 | 1 | B1 inject rho=0.05/M=50 | 0,1,2 | 3000 | ⏳ After batch 1 |
| 14-16 | 1 | B1 inject rho=0.1/M=50 | 0,1,2 | 3000 | ⏳ After batch 1 |
| 17-19 | 1 | E1 phase1 S23 | 0,1,2 | 1000 | ⏳ After batch 1 |
| 20 | 1 | C2 phase2 mode1 | 0 | 2000 | ⏳ After batch 1 |

### Future Batches (not yet launched)

**Batch 3 — Phase-2 experiments + B1 M-sweep + B2:**
- C2 phase2: 3 modes × 3 seeds = 9 runs (minus the 1 in batch 2) = 8
- E1 phase2: 6 subsets × 3 seeds = 18 (single-mode reuses C2p1 ckpts)
- B1 inject M-sweep: M={10,200} × rho=0.05 × 3 seeds = 6
- B2: sigma={0.001,0.01,0.1,1.0} × 3 seeds = 12

**Batch 4 — Tier 3 sweeps:**
- A2: d_inject={2,4,6,8,12} × 3 seeds = 15
- A3: sigma={0.5,0.8,1.5,2.0} × 3 seeds = 12
- C1: d_init={0.01,0.1,1.0,10.0} × 3 seeds = 12
- B1 baseline (no inject, extended): 3 seeds = 3

---

## Experiment Families — Status

### Tier 0 — v1 Reinterpretation (No new training; analysis only)

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

### Tier 1 — Direct Stability Tests (Highest priority new experiments)

- [ ] **B1: Dead Mode Revival via Data Injection** ⭐ MOST IMPORTANT
  - rho sweep (M=50 fixed): rho={0.001, 0.01, 0.05, 0.1} × 3 seeds = **12 runs** → Batch 2 🔄
  - M sweep (rho=0.05 fixed): M={10, 200} × 3 seeds = **6 runs** → Batch 3
  - (M=50/rho=0.05 covered in rho sweep above)
  - Baseline (no inject, 3000 ep): 3 seeds = **3 runs** → Batch 4
  - **Total: 21 training runs**

- [ ] **A1: Sequential Mode Addition on 2D Gaussian Mixture**
  - Phase 1 = B1_sym baselines (batch 1) ✓
  - Phase 2: 3 seeds = **3 runs** → Batch 2 🔄
  - **Total: 3 new runs** (phase 1 shared)

- [ ] **C2: Start from Adversarial Collapsed States**
  - Phase 1: 3 modes × 3 seeds = **9 runs** → Batch 1 🔄
  - Phase 2: 3 modes × 3 seeds = **9 runs** → Batch 2 (1 run) + Batch 3 (8 runs)
  - **Total: 18 training runs**

---

### Tier 2 — Quantitative Jacobian Verification

- [ ] **D1: Block-by-Block Jacobian Estimation**
  - Requires B7 baselines (batch 1)
  - K modes × 4 epsilon values × 3 seeds = **~36 micro-runs** (short, 1 epoch each)

- [ ] **D2: Spectral Radius Measurement**
  - Derived from D1 data — no separate training

- [ ] **D3: Multi-Epoch Perturbation Decay**
  - 3 seeds × 1 perturbation = **3 runs** → Batch 3

- [ ] **B2: Controller-Level Perturbation**
  - sigma={0.001, 0.01, 0.1, 1.0} × 3 seeds = **12 runs** → Batch 3
  - Requires B7 baselines (batch 1)

- [ ] **F1: Direct Measurement of Dead Mode Adjoints**
  - MC estimation at B7 checkpoint — **3 analysis runs** (not training)

- [ ] **F2: Sensitivity of Dead Mode Adjoints**
  - Perturbation sensitivity — **~9 analysis runs**

---

### Tier 3 — Scope and Basin Mapping

- [ ] **A2: Mode Addition with Varying Injection Distance**
  - d_inject={2, 4, 6, 8, 12} × 3 seeds = **15 runs** → Batch 4
  - Requires B1_sym baselines (batch 1)

- [ ] **A3: Mode Addition with Energy Depth Variation**
  - sigma={0.5, 0.8, 1.5, 2.0} × 3 seeds = **12 runs** → Batch 4
  - Requires B1_sym baselines (batch 1)

- [ ] **C1: Initialization-to-Collapse Distance Sweep**
  - d_init={0.01, 0.1, 1.0, 10.0} × 3 seeds = **12 runs** → Batch 4
  - Requires B7 baselines (batch 1)

- [ ] **E1: Attractor Enumeration on Small Benchmarks**
  - Phase 1: 6 subsets × 3 seeds = **18 runs** (9 single-mode = C2p1, 9 two-mode)
    - S12 × 3 → Batch 1 🔄
    - S13 × 3 → Batch 1 (2) + Batch 2 (1) 🔄
    - S23 × 3 → Batch 2 🔄
  - Phase 2: 6 subsets × 3 seeds = **18 runs** → Batch 3
  - **Total: 36 training runs** (9 shared with C2p1)

- [ ] **E2: Transition Threshold Between Collapsed States**
  - Magnitude sweep — **~15 runs** → Batch 5+

- [ ] **B3: Bias Injection with Continuous Training**
  - 3 seeds = **3 runs** → Batch 5+

---

### Supplementary Analyses on v1 Data

- [ ] **G1: Empirical Contraction Factor from v1 Trajectories**
  - Analysis only — no training

- [ ] **G2: Jacobian Estimation from Small Fluctuations**
  - Analysis only — no training

---

## Run Count Summary

| Batch | Runs | Epochs (approx) | Content | Status |
|-------|------|-----------------|---------|--------|
| 1 | 20 | ~26,000 | Baselines + C2p1 + E1p1 (partial) | 🔄 Running |
| 2 | 20 | ~48,000 | B1 inject rho-sweep + A1 + E1p1 S23 + C2p2 start | ⏳ Auto-queued |
| 3 | ~44 | ~70,000 | C2p2 + E1p2 + B1 M-sweep + B2 | Not launched |
| 4 | ~42 | ~60,000 | A2 + A3 + C1 + B1 baseline | Not launched |
| 5+ | ~20 | ~30,000 | E2 + B3 + D1/D3 micro-runs | Not launched |
| **Total** | **~146** | **~234,000** | | |

---

## Logs & Monitoring

```bash
# Watch live progress
tail -f logs/gpu0_chain.log
tail -f logs/gpu1_chain.log

# Check GPU utilization
nvidia-smi

# Count completed runs
find results/v3 -name "checkpoint_latest.pt" | wc -l

# Post-hoc reconstruction (after training)
python scripts/reconstruct_tracking.py --results-dir results/v3 --recursive --n-samples 10000
```
