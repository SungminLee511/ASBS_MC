"""
Centralized configuration for evaluation and figure generation.

All paths, plot styles, experiment definitions, and shared constants.
"""

from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# Directories
# ──────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
TABLES_DIR = ROOT / "tables"

# ──────────────────────────────────────────────────────────────────────
# Plot style
# ──────────────────────────────────────────────────────────────────────

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = {
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
}

def apply_style():
    plt.rcParams.update(STYLE)

# Color palettes
MODE_COLORS_2 = ["#1f77b4", "#ff7f0e"]
MODE_COLORS_3 = ["#1f77b4", "#ff7f0e", "#2ca02c"]
MODE_COLORS_4 = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
MODE_COLORS_25 = [plt.cm.tab20(i / 24) for i in range(25)]

# ──────────────────────────────────────────────────────────────────────
# Experiment definitions
# ──────────────────────────────────────────────────────────────────────

# E7: Phase portrait
E7_B1_SEEDS = 30
E7_B7_SEEDS = 20
E7_NUM_EPOCHS = 1000

# E1: Mode tracking
E1_BENCHMARKS = ["b1", "b2", "b3", "b5", "b6", "b7"]

# E8: Jacobian threshold
E8_SEPARATIONS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]
E8_N_SEEDS = 5
E8_SIGMA_SDE = 3.0  # σ_max from b1_asbs

# E11: Müller-Brown sweep
E11_BETAS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
E11_N_SEEDS = 5

# E13: Metastable survival
E13_KAPPA3_VALUES = [4, 6, 8, 10, 12, 14, 16, 18, 20]
E13_KAPPA1 = 20.0
E13_N_SEEDS = 20

# E10: Separation sweep
E10_B1_SEPARATIONS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]
E10_B5_CENTER_SCALES = [2, 3, 4, 5, 7, 10]
E10_N_SEEDS = 5

# Checkpoints to visualize for post-hoc scripts (E3, E4, E5, E6)
VIZ_EPOCHS = [50, 200, 500, 1000]
VIZ_T_VALUES = [0.3, 0.5, 0.7]

# ──────────────────────────────────────────────────────────────────────
# Result path helpers
# ──────────────────────────────────────────────────────────────────────

def tracking_path(exp_name, seed):
    return RESULTS_DIR / exp_name / f"seed_{seed}" / "mode_tracking.jsonl"

def checkpoint_path(exp_name, seed, epoch):
    return RESULTS_DIR / exp_name / f"seed_{seed}" / "checkpoints" / f"checkpoint_{epoch}.pt"

def latest_checkpoint(exp_name, seed):
    return RESULTS_DIR / exp_name / f"seed_{seed}" / "checkpoints" / "checkpoint_latest.pt"

def sweep_tracking(sweep_name, param_name, param_val, seed):
    return RESULTS_DIR / sweep_name / f"{param_name}_{param_val}" / f"seed_{seed}" / "mode_tracking.jsonl"

def ensure_dirs():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
