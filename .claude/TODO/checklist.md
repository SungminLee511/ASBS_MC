# v3 Experiment Checklist

All evaluation is **post-hoc** via `reconstruct_tracking.py`. Training runs use `eval_freq=99999 save_freq=10`.

**Legend:** Code = launcher/script exists | Train = GPU training done | Recon = post-hoc `mode_tracking.jsonl` reconstructed | Analysis = final plots/tables/numbers extracted

---

## Master Table

| Tier | Family | ID | Description | Benchmark | Sweep Params | Seeds | Epochs | Depends On | Code | Train | Recon | Analysis |
|------|--------|----|-------------|-----------|-------------|-------|--------|------------|------|-------|-------|----------|
| 0 | Baseline | BL-B1a | B1 asymmetric (80/20) pretrain | B1 | w1=0.8 | 5 | 2000 | — | YES | [x] | [ ] | [ ] |
| 0 | Baseline | BL-B5 | B5 heterogeneous covariance pretrain | B5 | center_scale=5 | 5 | 2000 | — | YES | [x] | [ ] | [ ] |
| 0 | Baseline | BL-B7 | B7 three-well pretrain | B7 | — | 5 | 2000 | — | YES | [x] | [ ] | [ ] |
| 0 | Baseline | BL-B1s | B1 symmetric (50/50) pretrain | B1 | w1=0.5 | 5 | 2000 | — | YES | [x] | [ ] | [ ] |
| 0 | Fam 0 | 0.1 | v1 reachability as basin evidence (B5, B7 collapse from random init) | BL-B5, BL-B7 | — | — | — | BL-B5, BL-B7 | N/A | N/A | N/A | [ ] |
| 0 | Fam 0 | 0.2 | B1 symmetric as balanced-state stability (0% collapse) | BL-B1s | — | — | — | BL-B1s | N/A | N/A | N/A | [ ] |
| 0 | Fam 0 | 0.3 | Seed consistency as basin size evidence (B7 pattern) | BL-B7 | — | — | — | BL-B7 | N/A | N/A | N/A | [ ] |
| 0 | Fam 0 | 0.4 | Separation/temperature sweeps as basin phase transitions | BL-B5 | — | — | — | BL-B5 | N/A | N/A | N/A | [ ] |
| 0 | Fam 0 | 0.5 | Metastable survival sweep as basin threshold | B7 | kappa3={4,8,12,16,18,20,24,32} | 5 | 2000 | — | YES | [ ] | [ ] | [ ] |
| 1 | Fam B1 | B1-rho-0.001 | Dead mode revival: rho=0.001, M=50 | B7 | rho=0.001, M=50 | 5 | 3000 | — | YES | [~] 3/5 seeds running (batch2) | [ ] | [ ] |
| 1 | Fam B1 | B1-rho-0.01 | Dead mode revival: rho=0.01, M=50 | B7 | rho=0.01, M=50 | 5 | 3000 | — | YES | [~] 3/5 seeds running (batch2) | [ ] | [ ] |
| 1 | Fam B1 | B1-rho-0.05 | Dead mode revival: rho=0.05, M=50 | B7 | rho=0.05, M=50 | 5 | 3000 | — | YES | [~] 3/5 seeds running (batch2) | [ ] | [ ] |
| 1 | Fam B1 | B1-rho-0.1 | Dead mode revival: rho=0.1, M=50 | B7 | rho=0.1, M=50 | 5 | 3000 | — | YES | [~] 3/5 seeds running (batch2) | [ ] | [ ] |
| 1 | Fam B1 | B1-M10 | Dead mode revival: rho=0.05, M=10 | B7 | rho=0.05, M=10 | 5 | 3010 | — | YES | [ ] | [ ] | [ ] |
| 1 | Fam B1 | B1-M50 | Dead mode revival: rho=0.05, M=50 (same as B1-rho-0.05) | B7 | rho=0.05, M=50 | 5 | 3050 | — | YES | [ ] | [ ] | [ ] |
| 1 | Fam B1 | B1-M200 | Dead mode revival: rho=0.05, M=200 | B7 | rho=0.05, M=200 | 5 | 3200 | — | YES | [ ] | [ ] | [ ] |
| 1 | Fam B1 | B1-ctrl | B7 extended baseline (no injection, 3000 ep) | B7 | — | 5 | 3000 | — | YES | [ ] | [ ] | [ ] |
| 1 | Fam A | A1 | 2-mode pretrain -> 3-mode switch | k_mode_gmm | 3rd mode at (0,4) | 5 | 2000 | BL-B1s | YES | [~] 3/5 seeds running (batch2) | [ ] | [ ] |
| 1 | Fam C2 | C2-ph1-m1 | Phase 1: pretrain single-mode at B7 mode 1 (1,0) | k_mode_gmm | K=1, center=(1,0) | 5 | 1000 | — | YES | [~] 1/5 seeds running (batch2) | [ ] | [ ] |
| 1 | Fam C2 | C2-ph1-m2 | Phase 1: pretrain single-mode at B7 mode 2 (-1,0) | k_mode_gmm | K=1, center=(-1,0) | 5 | 1000 | — | YES | [ ] | [ ] | [ ] |
| 1 | Fam C2 | C2-ph1-m3 | Phase 1: pretrain single-mode at B7 mode 3 (0.3,1.5) | k_mode_gmm | K=1, center=(0.3,1.5) | 5 | 1000 | — | YES | [ ] | [ ] | [ ] |
| 1 | Fam C2 | C2-ph2-m1 | Phase 2: resume mode-1 ckpt on true B7 | B7 | — | 5 | 2000 | C2-ph1-m1 | YES | [ ] | [ ] | [ ] |
| 1 | Fam C2 | C2-ph2-m2 | Phase 2: resume mode-2 ckpt on true B7 | B7 | — | 5 | 2000 | C2-ph1-m2 | YES | [ ] | [ ] | [ ] |
| 1 | Fam C2 | C2-ph2-m3 | Phase 2: resume mode-3 ckpt on true B7 | B7 | — | 5 | 2000 | C2-ph1-m3 | YES | [ ] | [ ] | [ ] |
| 2 | Fam D | D1 | Block-by-block Jacobian estimation at B7 collapsed state | B7 | eps={0.001,0.005,0.01,0.02} per mode | — | 1 per probe | BL-B7 | NO | [ ] | N/A | [ ] |
| 2 | Fam D | D2 | Spectral radius from D1 Jacobian | — | — | — | — | D1 | NO | N/A | N/A | [ ] |
| 2 | Fam D | D3 | Multi-epoch perturbation decay fit | — | — | — | — | B1-* or B2-* | NO | N/A | N/A | [ ] |
| 2 | Fam B2 | B2-sig-0.001 | Controller weight perturbation: sigma=0.001 | B7 | sigma=0.001 | 5 | 3000 | BL-B7 | YES | [ ] | [ ] | [ ] |
| 2 | Fam B2 | B2-sig-0.01 | Controller weight perturbation: sigma=0.01 | B7 | sigma=0.01 | 5 | 3000 | BL-B7 | YES | [ ] | [ ] | [ ] |
| 2 | Fam B2 | B2-sig-0.1 | Controller weight perturbation: sigma=0.1 | B7 | sigma=0.1 | 5 | 3000 | BL-B7 | YES | [ ] | [ ] | [ ] |
| 2 | Fam B2 | B2-sig-1.0 | Controller weight perturbation: sigma=1.0 | B7 | sigma=1.0 | 5 | 3000 | BL-B7 | YES | [ ] | [ ] | [ ] |
| 2 | Fam F | F1 | Dead mode adjoint measurement (Monte Carlo) | B7 | 10^6 samples | — | — | BL-B7 | NO | [ ] | N/A | [ ] |
| 2 | Fam F | F2 | Adjoint sensitivity dY_k/dalpha_j | B7 | — | — | — | F1 | NO | [ ] | N/A | [ ] |
| 3 | Fam A | A2-d2 | Injection distance sweep: d_inject=2 | k_mode_gmm | 3rd mode at (0,2) | 5 | 2000 | BL-B1s | YES | [ ] | [ ] | [ ] |
| 3 | Fam A | A2-d4 | Injection distance sweep: d_inject=4 | k_mode_gmm | 3rd mode at (0,4) | 5 | 2000 | BL-B1s | YES | [ ] | [ ] | [ ] |
| 3 | Fam A | A2-d6 | Injection distance sweep: d_inject=6 | k_mode_gmm | 3rd mode at (0,6) | 5 | 2000 | BL-B1s | YES | [ ] | [ ] | [ ] |
| 3 | Fam A | A2-d8 | Injection distance sweep: d_inject=8 | k_mode_gmm | 3rd mode at (0,8) | 5 | 2000 | BL-B1s | YES | [ ] | [ ] | [ ] |
| 3 | Fam A | A2-d12 | Injection distance sweep: d_inject=12 | k_mode_gmm | 3rd mode at (0,12) | 5 | 2000 | BL-B1s | YES | [ ] | [ ] | [ ] |
| 3 | Fam A | A3-sig0.5 | Energy depth: tight injected mode (sigma=0.5) | k_mode_gmm | gmm_sigma=0.5 | 5 | 2000 | BL-B1s | YES | [ ] | [ ] | [ ] |
| 3 | Fam A | A3-sig0.8 | Energy depth: sigma=0.8 | k_mode_gmm | gmm_sigma=0.8 | 5 | 2000 | BL-B1s | YES | [ ] | [ ] | [ ] |
| 3 | Fam A | A3-sig1.5 | Energy depth: sigma=1.5 | k_mode_gmm | gmm_sigma=1.5 | 5 | 2000 | BL-B1s | YES | [ ] | [ ] | [ ] |
| 3 | Fam A | A3-sig2.0 | Energy depth: wide injected mode (sigma=2.0) | k_mode_gmm | gmm_sigma=2.0 | 5 | 2000 | BL-B1s | YES | [ ] | [ ] | [ ] |
| 3 | Fam C1 | C1-d0.01 | Init distance sweep: d_init=0.01 | B7 | sigma=0.01 | 5 | 3000 | BL-B7 | YES | [ ] | [ ] | [ ] |
| 3 | Fam C1 | C1-d0.1 | Init distance sweep: d_init=0.1 | B7 | sigma=0.1 | 5 | 3000 | BL-B7 | YES | [ ] | [ ] | [ ] |
| 3 | Fam C1 | C1-d1.0 | Init distance sweep: d_init=1.0 | B7 | sigma=1.0 | 5 | 3000 | BL-B7 | YES | [ ] | [ ] | [ ] |
| 3 | Fam C1 | C1-d10.0 | Init distance sweep: d_init=10.0 | B7 | sigma=10.0 | 5 | 3000 | BL-B7 | YES | [ ] | [ ] | [ ] |
| 3 | Fam E1 | E1-ph1-S12 | Phase 1: pretrain on modes {1,2} | k_mode_gmm | centers=(1,0),(-1,0) | 5 | 1000 | — | YES | [ ] | [ ] | [ ] |
| 3 | Fam E1 | E1-ph1-S13 | Phase 1: pretrain on modes {1,3} | k_mode_gmm | centers=(1,0),(0.3,1.5) | 5 | 1000 | — | YES | [~] 1/5 seeds running (batch2) | [ ] | [ ] |
| 3 | Fam E1 | E1-ph1-S23 | Phase 1: pretrain on modes {2,3} | k_mode_gmm | centers=(-1,0),(0.3,1.5) | 5 | 1000 | — | YES | [~] 3/5 seeds running (batch2) | [ ] | [ ] |
| 3 | Fam E1 | E1-ph2-S12 | Phase 2: resume S={1,2} on true B7 | B7 | — | 5 | 2000 | E1-ph1-S12 | YES | [ ] | [ ] | [ ] |
| 3 | Fam E1 | E1-ph2-S13 | Phase 2: resume S={1,3} on true B7 | B7 | — | 5 | 2000 | E1-ph1-S13 | YES | [ ] | [ ] | [ ] |
| 3 | Fam E1 | E1-ph2-S23 | Phase 2: resume S={2,3} on true B7 | B7 | — | 5 | 2000 | E1-ph1-S23 | YES | [ ] | [ ] | [ ] |
| 3 | Fam E1 | E1-ph2-S1 | Phase 2: resume S={1} on true B7 | B7 | — | 5 | 2000 | C2-ph1-m1 | YES | [ ] | [ ] | [ ] |
| 3 | Fam E1 | E1-ph2-S2 | Phase 2: resume S={2} on true B7 | B7 | — | 5 | 2000 | C2-ph1-m2 | YES | [ ] | [ ] | [ ] |
| 3 | Fam E1 | E1-ph2-S3 | Phase 2: resume S={3} on true B7 | B7 | — | 5 | 2000 | C2-ph1-m3 | YES | [ ] | [ ] | [ ] |
| 3 | Fam E | E2 | Transition threshold between collapsed states (injection magnitude sweep) | B7 | — | — | — | E1 | NO | [ ] | [ ] | [ ] |
| 3 | Fam B | B3 | Bias injection with continuous training | B7 | delta=0.01 | — | — | BL-B7 | PARTIAL | [ ] | [ ] | [ ] |
| — | Fam G | G1 | Empirical contraction factor from baseline trajectories | — | — | — | — | BL-B5, BL-B7 | NO | N/A | N/A | [ ] |
| — | Fam G | G2 | Jacobian estimation from small fluctuations (autocorrelation) | — | — | — | — | BL-B7 | NO | N/A | N/A | [ ] |

---

## Run Counts

| Category | Configs | x Seeds | = Total Runs |
|----------|---------|---------|--------------|
| Baselines (BL-*) | 4 | 5 | 20 |
| Family 0.5 (kappa_3 sweep) | 8 | 5 | 40 |
| Family B1 (rho sweep + M sweep + ctrl) | 8 | 5 | 40 |
| Family A1 | 1 | 5 | 5 |
| Family C2 (phase 1 + phase 2) | 6 | 5 | 30 |
| Family B2 | 4 | 5 | 20 |
| Family A2 (distance sweep) | 5 | 5 | 25 |
| Family A3 (depth sweep) | 4 | 5 | 20 |
| Family C1 (init distance) | 4 | 5 | 20 |
| Family E1 (phase 1 two-mode + phase 2 all) | 9 | 5 | 45 |
| **TOTAL training runs** | | | **265** |

---

## Dependency Graph (execution order)

```
Phase 1 — Independent pretraining (can all run in parallel):
  BL-B1a, BL-B5, BL-B7, BL-B1s        (20 runs)
  Family 0.5 kappa_3 sweep              (40 runs — B7 with varied kappa3)
  B1-rho-*, B1-M*, B1-ctrl              (40 runs — B7 trains from scratch)
  C2-ph1-m1, C2-ph1-m2, C2-ph1-m3      (15 runs)
  E1-ph1-S12, E1-ph1-S13, E1-ph1-S23   (15 runs)

Phase 2 — Needs Phase 1 checkpoints:
  A1, A2-*, A3-*                        (50 runs — needs BL-B1s)
  B2-sig-*                              (20 runs — needs BL-B7)
  C1-d*                                 (20 runs — needs BL-B7)
  C2-ph2-m1, C2-ph2-m2, C2-ph2-m3      (15 runs — needs C2-ph1-*)
  E1-ph2-*                              (30 runs — needs E1-ph1-* and C2-ph1-*)

Phase 3 — Post-hoc reconstruction:
  python scripts/reconstruct_tracking.py --results-dir results/v3 --recursive --n-samples 10000

Phase 4 — Analysis (scripts needed):
  Family 0 re-analysis                  (from BL-* mode_tracking.jsonl)
  Family D Jacobian estimation          (SCRIPT NEEDED: estimate_jacobian.py)
  Family D3 decay fitting               (SCRIPT NEEDED: fit_decay.py)
  Family F BRA verification             (SCRIPT NEEDED: measure_dead_adjoints.py)
  Family G supplementary                (SCRIPT NEEDED: v1_contraction.py, autocorrelation.py)
```

---

## Missing Code

All scripts and launcher entries are now implemented. ✅

| Item | Status | Location |
|------|--------|----------|
| D1 `estimate_jacobian.py` | ✅ Done | `scripts/estimate_jacobian.py` |
| D2 spectral analysis | ✅ Done | Included in `scripts/estimate_jacobian.py` |
| D3 `fit_decay.py` | ✅ Done | `scripts/fit_decay.py` |
| F1 `measure_dead_adjoints.py` | ✅ Done | `scripts/measure_dead_adjoints.py` |
| F2 adjoint sensitivity | ✅ Done | `scripts/measure_dead_adjoints.py` (via `--perturb-sigmas`) |
| E2 transition threshold | ✅ Done | `launch_v3_experiments.sh` → `run_family_e2()` |
| B3 launcher entry | ✅ Done | `launch_v3_experiments.sh` → `run_family_b3()` |
| G1 `v1_contraction.py` | ✅ Done | `scripts/v1_contraction.py` |
| G2 `autocorrelation.py` | ✅ Done | `scripts/autocorrelation.py` |
| 0.5 metastable survival | ✅ Done | `launch_v3_experiments.sh` → `run_family_05()` (kappa3 sweep: 4,8,12,16,18,20,24,32) |

---

## Batch Execution Log

| Batch | Runs | Server | Started (KST) | Finished (KST) | Status | Notes |
|-------|------|--------|---------------|-----------------|--------|-------|
| 1 | 20 | ASBS_MC | 2026-04-17 18:07 | 2026-04-17 19:27 | ✅ Done | BL-B1a(5), BL-B5(5), BL-B7(5), BL-B1s(5). b5-s3 exploded, b1s-s3 spiked, b7-s4 NaN at end. |
| 2 | 20 | ASBS_MC | 2026-04-17 19:37 | — | 🏃 Running | B1-rho{0.001,0.01,0.05,0.1}×3seeds, A1×3seeds, E1-ph1-S13-s2, E1-ph1-S23×3seeds, C2-ph1-m1-s0. A1 initially failed (relative ckpt path), relaunched with abs path. |
