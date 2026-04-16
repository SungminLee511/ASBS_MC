"""
E11: Müller-Brown Temperature Sweep

Figures:
  - fig_e11_mb_panels.pdf: per-β α_k evolution panels
  - fig_e11_mb_summary.pdf: final α_k vs β summary
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from mc_utils import load_tracking
from config import (
    RESULTS_DIR, FIGURES_DIR, apply_style,
    E11_BETAS, E11_N_SEEDS, MODE_COLORS_3,
)


def generate_all():
    apply_style()
    print("E11: Müller-Brown Temperature Sweep")
    base = RESULTS_DIR / "e11_sweep"

    betas = E11_BETAS
    n_seeds = E11_N_SEEDS
    mode_labels = ["Deep well 0", "Deep well 1", "Shallow well 2"]

    # ── Per-β panels ──
    ncols = min(3, len(betas))
    nrows = (len(betas) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows),
                              sharex=True, sharey=True,
                              gridspec_kw={"hspace": 0.3, "wspace": 0.15})
    if nrows == 1:
        axes = axes.reshape(1, -1)

    summary = []

    for idx, beta in enumerate(betas):
        r, c = idx // ncols, idx % ncols
        ax = axes[r, c]
        final_alphas = []

        for seed in range(n_seeds):
            records = load_tracking(base / f"beta_{beta}" / f"seed_{seed}" / "mode_tracking.jsonl")
            if not records:
                continue
            epochs = [rec["epoch"] for rec in records]
            alphas = np.array([rec["alpha"] for rec in records])
            K = alphas.shape[1]
            for k in range(min(K, 3)):
                ax.plot(epochs, alphas[:, k], color=MODE_COLORS_3[k],
                        alpha=0.35, linewidth=0.7)
            final_alphas.append(records[-1]["alpha"][:3])

        ax.set_title(f"$\\beta$ = {beta}", fontsize=11)
        ax.set_ylim(-0.05, 1.05)
        if c == 0:
            ax.set_ylabel("$\\alpha_k$")
        if r == nrows - 1:
            ax.set_xlabel("Epoch")

        if final_alphas:
            fa = np.array(final_alphas)
            summary.append({"beta": beta, "mean": fa.mean(0), "std": fa.std(0)})

    for idx in range(len(betas), nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle("Müller-Brown Temperature Sweep", fontsize=14, y=1.01)
    out = FIGURES_DIR / "fig_e11_mb_panels.pdf"
    fig.savefig(out)
    print(f"  Saved: {out}")
    plt.close(fig)

    # ── Summary ──
    if not summary:
        print("  [skip] No summary data")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    beta_arr = [s["beta"] for s in summary]
    for k in range(3):
        means = [s["mean"][k] for s in summary]
        stds = [s["std"][k] for s in summary]
        ax.errorbar(beta_arr, means, yerr=stds, fmt="o-",
                     color=MODE_COLORS_3[k], capsize=3, label=mode_labels[k])
    ax.set_xlabel("$\\beta$ (inverse temperature)")
    ax.set_ylabel("Final $\\alpha_k$")
    ax.set_title("Final Mode Weights vs. Temperature")
    ax.set_xscale("log")
    ax.legend()

    out = FIGURES_DIR / "fig_e11_mb_summary.pdf"
    fig.savefig(out)
    print(f"  Saved: {out}")
    plt.close(fig)
