# ASBS_MC — Mode Concentration in Adjoint Schrödinger Bridge Samplers

## Project Identity
- **Name**: ASBS_MC (Adjoint SBS — Mode Concentration)
- **Purpose**: Prove that ASBS (Meta's Adjoint Schrödinger Bridge Sampler) suffers from structural mode concentration, and characterize when/how it happens
- **Origin**: Forked from [facebookresearch/adjoint_samplers](https://github.com/facebookresearch/adjoint_samplers) — the codebase is **unmodified Meta code** plus our `.claude/` research docs
- **Conda env**: `Sampling_env`

## Research Focus

The AM (Adjoint Matching) regression loss used by ASBS weights training data by the **current policy's mode weights** (`α_k`), not the **target's mode weights** (`w_k`). This creates a self-reinforcing feedback loop:
- Overrepresented modes get more regression weight → better controllers → more samples → even more weight
- Underrepresented modes starve → worse controllers → fewer samples → extinction

### Theory (in `.claude/TODO/`)
| File | Contents |
|------|----------|
| `Static_mode_concentration.md` | Formal proof chain: AM loss decomposition, KL lower bound `KL(Q_θ\|p) ≥ KL(α\|w)` |
| `Dynamic_mode_concentration.md` | Stability analysis: Jacobian of mode weight dynamics, instability conditions |
| `Experiments_for_mode_concentration.md` | 13 experiments (E1–E13) across 8 benchmarks to validate theory |

## File Tree

```
ASBS_MC/
├── train.py                              # Hydra entry point
├── environment.yml                       # Conda env spec
├── adjoint_samplers/
│   ├── train_loop.py                     # Main training loop
│   ├── components/
│   │   ├── matcher.py                    # AdjointVE/VPMatcher (core AM regression)
│   │   ├── model.py                      # FourierMLP, EGNN architectures
│   │   ├── sde.py                        # ControlledSDE, sdeint
│   │   ├── buffer.py                     # BatchBuffer for trajectory storage
│   │   ├── evaluator.py                  # SyntheticEnergyEvaluator
│   │   ├── state_cost.py                 # ZeroGradStateCost
│   │   └── term_cost.py                  # Terminal cost (score/corrector)
│   ├── energies/
│   │   ├── base_energy.py                # Abstract BaseEnergy
│   │   ├── double_well_energy.py         # DW4 (8D, 2 modes)
│   │   ├── lennard_jones_energy.py       # LJ3/13/55 (molecular)
│   │   ├── dist_energy.py                # Distribution-based energy
│   │   └── synthetic_energies.py         # B1–B7 mode concentration benchmarks (NEW)
│   └── utils/
│       ├── train_utils.py                # Timestep helpers
│       ├── eval_utils.py                 # Interatomic distance metrics
│       ├── graph_utils.py                # Graph/COM-free coordinates
│       ├── dist_utils.py                 # Distributed training
│       └── distributed_mode.py           # Multi-GPU setup
├── configs/                              # Hydra config hierarchy
│   ├── train.yaml                        # Top-level
│   ├── experiment/                       # (original) demo_asbs, dw4_{as,asbs}, lj{13,55}_{as,asbs}
│   │   ├── b1_asbs.yaml                  # B1: Asymmetric 2-mode Gaussian (NEW)
│   │   ├── b2_asbs.yaml                  # B2: Müller-Brown potential (NEW)
│   │   ├── b3_asbs.yaml                  # B3: Warped double well (NEW)
│   │   ├── b4_asbs.yaml                  # B4: Neal's Funnel (NEW)
│   │   ├── b5_asbs.yaml                  # B5: Heterogeneous covariance mixture (NEW)
│   │   ├── b6_asbs.yaml                  # B6: 25-mode power-law grid (NEW)
│   │   ├── b7_asbs.yaml                  # B7: Three-well metastable (NEW)
│   │   └── b8_asbs.yaml                  # B8: LJ3 (NEW)
│   ├── problem/                          # (original) demo, dw4, lj13, lj55
│   │   ├── b1_two_mode.yaml              # (NEW)
│   │   ├── b2_muller_brown.yaml          # (NEW)
│   │   ├── b3_warped_dw.yaml             # (NEW)
│   │   ├── b4_neals_funnel.yaml          # (NEW)
│   │   ├── b5_het_cov.yaml               # (NEW)
│   │   ├── b6_power_law_grid.yaml        # (NEW)
│   │   ├── b7_three_well.yaml            # (NEW)
│   │   └── b8_lj3.yaml                   # (NEW)
│   ├── matcher/                          # adjoint_ve, adjoint_vp, corrector
│   ├── model/                            # fouriermlp, egnn
│   ├── sde/                              # ve, vp, brownian_motion, graph_ve, graph_vp
│   ├── source/                           # gauss, harmonic, delta, meanfree
│   ├── state_cost/                       # zero
│   ├── term_cost/                        # score/corrector (+ graph variants)
│   └── lancher/                          # demo_slurm
├── scripts/
│   ├── mc_utils.py                       # Shared utilities + checkpoint loading (NEW)
│   ├── e1_mode_tracking.py               # E1: α_k tracking + death cascade heatmap (NEW)
│   ├── e3_loss_decomposition.py          # E3: V_intra + V_inter decomposition (NEW)
│   ├── e4_controller_field.py            # E4: learned vs oracle controller fields (NEW)
│   ├── e6_trajectory_viz.py              # E6: particle trajectory spaghetti plots (NEW)
│   ├── e7_phase_portrait.py              # E7: pitchfork + ternary simplex (NEW)
│   ├── e8_jacobian_threshold.py          # E8: J₁₁ threshold verification (NEW)
│   ├── e9_kl_decomposition.py            # E9: KL decomposition tracking (NEW)
│   ├── e10_separation_sweep.py           # E10: separation sweep (B1 + B5) (NEW)
│   ├── e11_muller_brown_sweep.py         # E11: temperature sweep (NEW)
│   ├── e13_metastable_survival.py        # E13: metastable survival (NEW)
│   ├── e2_regression_heatmaps.py         # E2: η_k learned vs oracle heatmaps (NEW)
│   ├── e5_terminal_density.py            # E5: KDE contours vs target density (NEW)
│   ├── e12_death_cascade.py              # E12: full death cascade analysis (NEW)
│   ├── e1_lj3_permutations.py            # E1/B8: LJ3 permutation counting (NEW)
│   ├── reconstruct_tracking.py           # Post-hoc: rebuild mode_tracking.jsonl from ckpts (NEW)
│   ├── demo.sh, dw4.sh, lj13.sh, lj55.sh, download.sh  # (original)
├── evaluation/                           # Post-hoc figure/table generation
│   ├── run_all.py                        # Master script: generates all figures + tables
│   ├── config.py                         # Centralized paths, styles, experiment defs
│   ├── tables.py                         # LaTeX summary tables
│   ├── fig_e1_mode_tracking.py           # α_k curves + heatmap
│   ├── fig_e2_heatmaps.py               # η_k learned vs oracle
│   ├── fig_e3_loss_decomp.py             # V_intra + V_inter stacked area
│   ├── fig_e4_controller.py              # Controller field quiver + deficiency
│   ├── fig_e5_density.py                 # KDE contours vs target
│   ├── fig_e6_trajectories.py            # Spaghetti trajectory plots
│   ├── fig_e7_phase_portrait.py          # Pitchfork + ternary simplex
│   ├── fig_e8_jacobian.py                # J₁₁ threshold + collapse prob
│   ├── fig_e9_kl.py                      # KL/TV decomposition
│   ├── fig_e10_separation.py             # Separation sweep (B1 + B5)
│   ├── fig_e11_temperature.py            # Müller-Brown β sweep
│   ├── fig_e12_cascade.py                # Death cascade (B6)
│   └── fig_e13_metastable.py             # Metastable survival (B7)
├── data/                                 # Reference splits (DW4, LJ13/55)
└── assets/demo.png
```

## Key Classes

| Class | File | Role |
|-------|------|------|
| `AdjointVEMatcher` | `matcher.py` | Core VE adjoint matching — **this is where mode concentration originates** |
| `AdjointVPMatcher` | `matcher.py` | VP variant of adjoint matching |
| `ControlledSDE` | `sde.py` | Forward SDE with learned drift controller |
| `FourierMLP` | `model.py` | Controller network (Fourier features + MLP) |
| `EGNN` | `model.py` | Equivariant GNN for molecular systems |
| `BatchBuffer` | `buffer.py` | Stores trajectory data for AM regression |
| `SyntheticEnergyEvaluator` | `evaluator.py` | W2 distance evaluation |

## Benchmark Energies (`synthetic_energies.py`)

All benchmarks inherit from `BaseEnergy`. Interface: `eval(x) → E(x)`, `grad_E(x) → ∇E(x)`, `score(x) → -∇E(x)`.

| Class | Benchmark | Dim | Modes | Key Parameters |
|-------|-----------|-----|-------|----------------|
| `AsymmetricTwoModeGaussian` | B1 | 2 | 2 | `w1` (weight split), `sigma`, `mu1/mu2` (centers) |
| `MullerBrownEnergy` | B2 | 2 | 3 | `beta` (inverse temperature) |
| `WarpedDoubleWellEnergy` | B3 | 2 | 2 | `gamma` (banana curvature) |
| `NealsFunnelEnergy` | B4 | 2 | continuous | (no tunable params) |
| `HeterogeneousCovarianceMixture` | B5 | 2 | 4 | `center_scale` |
| `PowerLawGridMixture` | B6 | 2 | 25 | `spacing`, `sigma`, `alpha` (Zipf exponent) |
| `ThreeWellMetastableEnergy` | B7 | 2 | 3 | `kappa1` (deep wells), `kappa3` (metastable) |
| `LennardJonesEnergy` (existing) | B8 | 6 | 6 (permutational) | `n_particles=3` |

Properties: `mode_centers` (tensor of centers), `mode_weights` (target weights) — available on B1, B2, B3, B5, B6, B7.

**Note on B2 (Müller-Brown):** Raw energy values are in the hundreds, so the interesting β range is {0.005–0.2}, not {1–10}. At β≥0.5, mode 0 dominates completely.

## Evaluation & Mode Tracking

**Evaluator:** `Synthetic2DEvaluator` (new file `synthetic_2d_evaluator.py`) — generates reference samples via importance sampling at init, then computes energy W2, marginal W2, sliced W2, mode weights, KL, TV, alive modes, and density comparison figures.

**During training:** `train.py` logs mode weights and runs the evaluator at `eval_freq` intervals. Set `eval_freq=99999` to skip eval during training for maximum speed.

**Post-hoc reconstruction:** `scripts/reconstruct_tracking.py` rebuilds `mode_tracking.jsonl` + `eval_metrics.jsonl` from saved checkpoints. Use after training with eval disabled:
```bash
conda run -n $ENV python scripts/reconstruct_tracking.py --results-dir results --recursive --n-samples 10000
```

Output files per run:
- `mode_tracking.jsonl` — one JSON line per epoch: `epoch, alpha, target_w, kl, tv, alive_modes`
- `eval_metrics.jsonl` — one JSON line per epoch: `epoch, mean_energy, std_energy, median_energy`

## Experiment Scripts (Tier 1)

All scripts live in `scripts/` and have `launch` (run training) and `plot` (analyze/visualize) subcommands.

| Script | Experiment | What it does | Key Benchmarks |
|--------|------------|-------------|----------------|
| `mc_utils.py` | — | Shared utilities: mode assignment, metrics, tracking I/O, checkpoint loading | — |
| **Tier 1** | | | |
| `e1_mode_tracking.py` | E1 | α_k vs. epoch plots, death cascade heatmap | All (B6 for heatmap) |
| `e7_phase_portrait.py` | E7 | Pitchfork bifurcation (B1), ternary simplex flow (B7) | B1, B7 |
| `e8_jacobian_threshold.py` | E8 | J₁₁^η threshold estimation, collapse probability vs. ρ_sep | Symmetric B1 sweep |
| `e11_muller_brown_sweep.py` | E11 | Temperature sweep on Müller-Brown | B2 |
| `e13_metastable_survival.py` | E13 | Metastable state survival vs. well depth | B7 |
| **Tier 2** | | | |
| `e6_trajectory_viz.py` | E6 | Particle trajectory spaghetti plots at multiple epochs | B1, B3, B7 |
| `e4_controller_field.py` | E4 | Learned vs. oracle controller quiver plots + deficiency | B1, B3, B7 |
| `e3_loss_decomposition.py` | E3 | V_intra + V_inter stacked area (Theorem 1) from checkpoints | B1, B2, B5, B7 |
| `e10_separation_sweep.py` | E10 | Collapse probability + time-to-collapse vs. separation | B1, B5 |
| `e9_kl_decomposition.py` | E9 | KL(α\|\|w) + full KL proxy + dead mode tracking | B1, B2, B5, B6 |
| **Tier 3** | | | |
| `e2_regression_heatmaps.py` | E2 | η_k heatmaps: learned vs oracle vs difference | B1, B3, B7 |
| `e5_terminal_density.py` | E5 | KDE contours vs target density at multiple epochs | All (B6 scatter mode) |
| `e1_lj3_permutations.py` | E1/B8 | LJ3 permutational basin discovery counting | B8 |
| `e12_death_cascade.py` | E12 | Survival curve, death order, grid snapshots, redistribution | B6 |

### Example: Running an experiment end-to-end

```bash
ENV=adjoint_samplers

# E7: Pitchfork on B1 (30 seeds)
conda run -n $ENV python scripts/e7_phase_portrait.py launch --benchmark b1 --seeds 30 --num-epochs 1000
conda run -n $ENV python scripts/e7_phase_portrait.py plot --benchmark b1

# E13: Metastable survival sweep
conda run -n $ENV python scripts/e13_metastable_survival.py run --kappa3-values 4 8 12 16 20 --n-seeds 10
conda run -n $ENV python scripts/e13_metastable_survival.py plot
```

## Running

```bash
# Conda env
ENV=adjoint_samplers

# Original benchmarks
conda run -n $ENV python train.py experiment=demo_asbs
conda run -n $ENV python train.py experiment=dw4_asbs
conda run -n $ENV python train.py experiment=lj13_asbs

# Mode concentration benchmarks (B1–B8)
conda run -n $ENV python train.py experiment=b1_asbs                      # 2-mode Gaussian (80/20)
conda run -n $ENV python train.py experiment=b1_asbs w1=0.9               # 2-mode Gaussian (90/10)
conda run -n $ENV python train.py experiment=b2_asbs beta=2.0             # Müller-Brown at β=2
conda run -n $ENV python train.py experiment=b3_asbs gamma=10.0           # Warped DW, narrow bananas
conda run -n $ENV python train.py experiment=b4_asbs                      # Neal's Funnel
conda run -n $ENV python train.py experiment=b5_asbs                      # Het. covariance mixture
conda run -n $ENV python train.py experiment=b6_asbs                      # 25-mode power-law grid
conda run -n $ENV python train.py experiment=b7_asbs kappa3=4.0           # Three-well, shallow metastable
conda run -n $ENV python train.py experiment=b8_asbs                      # LJ3 (permutation symmetry)
```

## Evaluation (Post-Experiment)

```bash
ENV=adjoint_samplers

# Generate ALL figures + tables (tracking-based figures are fast, GPU figs need checkpoints)
conda run -n $ENV python evaluation/run_all.py

# Skip GPU-heavy figures (E2/E3/E4/E5/E6) — only reads mode_tracking.jsonl
conda run -n $ENV python evaluation/run_all.py --no-gpu

# Only specific experiments
conda run -n $ENV python evaluation/run_all.py --only e7 e1 e8

# Only LaTeX tables
conda run -n $ENV python evaluation/run_all.py --tables-only
```

Output: `figures/*.pdf` and `tables/*.tex`

## Gotchas
1. **Minimal modifications to Meta's code** — prefer new files. Only touch `train.py`/`train_loop.py` when experiment hooks are strictly necessary. Current modifications to `train.py`: null-guard for evaluator, mode tracking hook.
2. **`results/` and `figures/` are gitignored** — training outputs and plots stay local only.
3. **`data/` contains reference test splits** — needed for evaluation, do not delete.
4. **Hydra configs override via CLI** — e.g. `python train.py experiment=b1_asbs w1=0.9 seed=42`
5. **B1–B7 use FourierMLP + VESDE** (flat 2D), **B8 uses FourierMLP + VESDE** (LJ3 too small for EGNN/harmonic source).
6. **Mode tracking only fires for energies with `mode_centers` AND `mode_weights`** — B4 (Neal's Funnel) has no discrete modes, so mode tracking logs energy stats only.
7. **All B1–B7 now have `Synthetic2DEvaluator`** — generates 50k reference samples at init via grid importance sampling. B4 has no mode metrics but still gets energy W2 and density figures.
7. **Experiment scripts have `launch` and `plot` subcommands** — `launch` spawns training processes, `plot` reads `mode_tracking.jsonl` files. Override `hydra.run.dir` to control output location.
8. **Müller-Brown β range** — interesting regime is β∈{0.005–0.2}. At β≥0.5, one mode dominates the entire mass.
