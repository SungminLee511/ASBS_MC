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

## File Tree (Meta's Original Code)

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
│   │   ├── lennard_jones_energy.py       # LJ13/55 (molecular)
│   │   └── dist_energy.py                # Distribution-based energy
│   └── utils/
│       ├── train_utils.py                # Timestep helpers
│       ├── eval_utils.py                 # Interatomic distance metrics
│       ├── graph_utils.py                # Graph/COM-free coordinates
│       ├── dist_utils.py                 # Distributed training
│       └── distributed_mode.py           # Multi-GPU setup
├── configs/                              # Hydra config hierarchy
│   ├── train.yaml                        # Top-level
│   ├── experiment/                       # demo_asbs, dw4_{as,asbs}, lj{13,55}_{as,asbs}
│   ├── problem/                          # demo, dw4, lj13, lj55
│   ├── matcher/                          # adjoint_ve, adjoint_vp, corrector
│   ├── model/                            # fouriermlp, egnn
│   ├── sde/                              # ve, vp, brownian_motion, graph_ve, graph_vp
│   ├── source/                           # gauss, harmonic, delta, meanfree
│   ├── state_cost/                       # zero
│   ├── term_cost/                        # score/corrector (+ graph variants)
│   └── lancher/                          # demo_slurm
├── data/                                 # Reference splits (DW4, LJ13/38/55, MW5, MW32, ALDP)
├── evaluation/                           # (empty — for experiment outputs)
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

## Running

```bash
# Demo (2D)
conda run -n Sampling_env python train.py experiment=demo_asbs

# DW4
conda run -n Sampling_env python train.py experiment=dw4_asbs

# LJ13
conda run -n Sampling_env python train.py experiment=lj13_asbs
```

## Gotchas
1. **Do NOT modify Meta's code** — this repo's value is showing the problem exists in the *original* algorithm. All analysis should be external scripts or new files.
2. **`results/` is gitignored** — training outputs stay local only.
3. **`data/` contains reference test splits** — needed for evaluation, do not delete.
4. **Hydra configs override via CLI** — e.g. `python train.py experiment=dw4_asbs matcher.num_steps=100`
