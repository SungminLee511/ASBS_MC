# Experiment Checklist

All training runs use `save_freq=10 eval_freq=99999 num_epochs=2000`.
After training: `python scripts/reconstruct_tracking.py --results-dir results --recursive --n-samples 10000`
Then: `python evaluation/run_all.py`

---

## Single-Benchmark Runs (3 seeds each)

These are the base training runs needed by all experiments. Each produces checkpoints + logs.

| # | Experiment Config | Seeds | Command Template | State |
|---|---|---|---|---|
| 1 | `b1_asbs` (80/20 split) | 0,1,2 | `experiment=b1_asbs` | pending |
| 2 | `b2_asbs` (β=0.01) | 0,1,2 | `experiment=b2_asbs beta=0.01` | pending |
| 3 | `b3_asbs` (γ=5) | 0,1,2 | `experiment=b3_asbs` | pending |
| 4 | `b4_asbs` | 0,1,2 | `experiment=b4_asbs` | pending |
| 5 | `b5_asbs` | 0,1,2 | `experiment=b5_asbs` | pending |
| 6 | `b6_asbs` | 0,1,2 | `experiment=b6_asbs` | pending |
| 7 | `b7_asbs` (κ3=8) | 0,1,2 | `experiment=b7_asbs` | pending |
| 8 | `b8_asbs` (LJ3) | 0,1,2 | `experiment=b8_asbs` | pending |

---

## Sweep Experiments

### E7: Phase Portrait (pitchfork + ternary)

| # | Description | Seeds | Command | State |
|---|---|---|---|---|
| 9 | B1 symmetric (w1=0.5, d=8) | 0–29 | `experiment=b1_asbs w1=0.5` | pending |
| 10 | B7 (κ1=20, κ3=8) | 0–19 | `experiment=b7_asbs` | pending |

### E8: Jacobian Threshold (symmetric 2-mode, separation sweep)

| # | Description | d | Seeds | Command | State |
|---|---|---|---|---|---|
| 11 | d=3 | 3 | 0–9 | `experiment=b1_asbs w1=0.5 energy.mu1=[-1.5,0] energy.mu2=[1.5,0]` | pending |
| 12 | d=5 | 5 | 0–9 | `experiment=b1_asbs w1=0.5 energy.mu1=[-2.5,0] energy.mu2=[2.5,0]` | pending |
| 13 | d=7 | 7 | 0–9 | `experiment=b1_asbs w1=0.5 energy.mu1=[-3.5,0] energy.mu2=[3.5,0]` | pending |
| 14 | d=9 | 9 | 0–9 | `experiment=b1_asbs w1=0.5 energy.mu1=[-4.5,0] energy.mu2=[4.5,0]` | pending |
| 15 | d=12 | 12 | 0–9 | `experiment=b1_asbs w1=0.5 energy.mu1=[-6,0] energy.mu2=[6,0]` | pending |
| 16 | d=15 | 15 | 0–9 | `experiment=b1_asbs w1=0.5 energy.mu1=[-7.5,0] energy.mu2=[7.5,0]` | pending |
| 17 | d=20 | 20 | 0–9 | `experiment=b1_asbs w1=0.5 energy.mu1=[-10,0] energy.mu2=[10,0]` | pending |

### E11: Müller-Brown Temperature Sweep

| # | Description | β | Seeds | Command | State |
|---|---|---|---|---|---|
| 18 | β=0.005 | 0.005 | 0–4 | `experiment=b2_asbs beta=0.005` | pending |
| 19 | β=0.01 | 0.01 | 0–4 | `experiment=b2_asbs beta=0.01` | pending |
| 20 | β=0.02 | 0.02 | 0–4 | `experiment=b2_asbs beta=0.02` | pending |
| 21 | β=0.05 | 0.05 | 0–4 | `experiment=b2_asbs beta=0.05` | pending |
| 22 | β=0.1 | 0.1 | 0–4 | `experiment=b2_asbs beta=0.1` | pending |
| 23 | β=0.2 | 0.2 | 0–4 | `experiment=b2_asbs beta=0.2` | pending |

### E13: Metastable Survival (κ₃ sweep)

| # | Description | κ₃ | Seeds | Command | State |
|---|---|---|---|---|---|
| 24 | κ₃=4 | 4 | 0–19 | `experiment=b7_asbs kappa3=4` | pending |
| 25 | κ₃=6 | 6 | 0–19 | `experiment=b7_asbs kappa3=6` | pending |
| 26 | κ₃=8 | 8 | 0–19 | `experiment=b7_asbs kappa3=8` | pending |
| 27 | κ₃=10 | 10 | 0–19 | `experiment=b7_asbs kappa3=10` | pending |
| 28 | κ₃=12 | 12 | 0–19 | `experiment=b7_asbs kappa3=12` | pending |
| 29 | κ₃=14 | 14 | 0–19 | `experiment=b7_asbs kappa3=14` | pending |
| 30 | κ₃=16 | 16 | 0–19 | `experiment=b7_asbs kappa3=16` | pending |
| 31 | κ₃=18 | 18 | 0–19 | `experiment=b7_asbs kappa3=18` | pending |
| 32 | κ₃=20 | 20 | 0–19 | `experiment=b7_asbs kappa3=20` | pending |

### E10: Separation Sweep — B1 (symmetric)

| # | Description | d | Seeds | Command | State |
|---|---|---|---|---|---|
| 33 | d=2 | 2 | 0–9 | `experiment=b1_asbs w1=0.5 energy.mu1=[-1,0] energy.mu2=[1,0]` | pending |
| 34 | d=3 | 3 | 0–9 | (same as E8 #11) | pending |
| 35 | d=4 | 4 | 0–9 | `experiment=b1_asbs w1=0.5 energy.mu1=[-2,0] energy.mu2=[2,0]` | pending |
| 36 | d=5 | 5 | 0–9 | (same as E8 #12) | pending |
| 37 | d=6 | 6 | 0–9 | `experiment=b1_asbs w1=0.5 energy.mu1=[-3,0] energy.mu2=[3,0]` | pending |
| 38 | d=8 | 8 | 0–9 | `experiment=b1_asbs w1=0.5 energy.mu1=[-4,0] energy.mu2=[4,0]` | pending |
| 39 | d=10 | 10 | 0–9 | `experiment=b1_asbs w1=0.5 energy.mu1=[-5,0] energy.mu2=[5,0]` | pending |

**Note:** E8 and E10 share runs for d={3,5,7,9,12,15,20}. Only d={2,4,6,8,10} are E10-only.

### E10: Separation Sweep — B5 (heterogeneous)

| # | Description | center_scale | Seeds | Command | State |
|---|---|---|---|---|---|
| 40 | cs=2 | 2 | 0–9 | `experiment=b5_asbs center_scale=2 scale=3` | pending |
| 41 | cs=3 | 3 | 0–9 | `experiment=b5_asbs center_scale=3 scale=4` | pending |
| 42 | cs=4 | 4 | 0–9 | `experiment=b5_asbs center_scale=4 scale=5` | pending |
| 43 | cs=5 | 5 | 0–9 | (same as base #5) | pending |
| 44 | cs=7 | 7 | 0–9 | `experiment=b5_asbs center_scale=7 scale=8` | pending |
| 45 | cs=10 | 10 | 0–9 | `experiment=b5_asbs center_scale=10 scale=11` | pending |

---

## Total Run Count

| Category | Unique Runs | GPU-hours estimate (2D ≈ 40s/ep × 2000 ep ≈ 22h each) |
|---|---|---|
| Base benchmarks (8 configs × 3 seeds) | 24 | ~528 GPU-h |
| E7 pitchfork (30 seeds B1 + 20 seeds B7) | 50 | ~1100 GPU-h |
| E8 Jacobian sweep (7 separations × 10 seeds) | 70 | ~1540 GPU-h |
| E11 MB sweep (6 β × 5 seeds) | 30 | ~660 GPU-h |
| E13 metastable sweep (9 κ₃ × 20 seeds) | 180 | ~3960 GPU-h |
| E10 B1 sweep (5 new d × 10 seeds) | 50 | ~1100 GPU-h |
| E10 B5 sweep (5 new cs × 10 seeds) | 50 | ~1100 GPU-h |
| **Total (deduped)** | **~420** | **~9240 GPU-h** |

**Note:** These estimates assume 2D benchmarks. B8 (LJ3, 6D) may be slower. The E8/E10 overlap saves ~50 runs. With 2× A100 running 2 jobs in parallel, wall-clock ≈ 4620 hours ≈ 193 days. **This needs parallelization across multiple servers.**

---

## Post-Training Steps

| Step | Command | State |
|---|---|---|
| Reconstruct tracking (all runs) | `python scripts/reconstruct_tracking.py --results-dir results --recursive --n-samples 10000` | pending |
| Generate all figures | `python evaluation/run_all.py` | pending |
| Generate tables | `python evaluation/run_all.py --tables-only` | pending |

---

## Dedup Notes

The following runs can be shared across experiments:
- E7/B7 seeds 0–2 overlap with base benchmark #7
- E8 d={3,5,7,9,12,15,20} overlap with E10/B1 for those separations
- E13 κ₃=8 seeds 0–2 overlap with base benchmark #7
- E10/B5 cs=5 seeds 0–2 overlap with base benchmark #5

Plan sweep output directories carefully to avoid redundant runs:
- E8 + E10/B1 → `results/e8_sweep/d_{d}/seed_{s}/` (shared)
- E13 → `results/e13_sweep/kappa3_{k}/seed_{s}/`
- E11 → `results/e11_sweep/beta_{b}/seed_{s}/`
- E10/B5 → `results/e10_b5_sweep/cs_{c}/seed_{s}/`
