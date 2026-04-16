"""
E13: Metastable State Survival

Figures:
  - fig_e13_survival.pdf: survival probability + α₃/w₃ vs depth ratio
  - fig_e13_evolution.pdf: per-κ₃ mode weight evolution panels
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from mc_utils import load_tracking
from config import (
    RESULTS_DIR, FIGURES_DIR, apply_style,
    E13_KAPPA3_VALUES, E13_KAPPA1, E13_N_SEEDS, MODE_COLORS_3,
)


def _analyze(kappa3_values, n_seeds, survival_threshold=0.5):
    base = RESULTS_DIR / "e13_sweep"
    results = []
    for k3 in kappa3_values:
        surv, ratios, total = 0, [], 0
        for seed in range(n_seeds):
            records = load_tracking(base / f"kappa3_{k3}" / f"seed_{seed}" / "mode_tracking.jsonl")
            if not records:
                continue
            total += 1
            last = records[-1]
            a_meta = last["alpha"][2]
            w_meta = last["target_w"][2]
            r = a_meta / max(w_meta, 1e-10)
            ratios.append(r)
            if r > survival_threshold:
                surv += 1
        if total > 0:
            results.append({
                "kappa3": k3,
                "ratio": k3 / E13_KAPPA1,
                "surv_prob": surv / total,
                "mean_ratio": np.mean(ratios),
                "std_ratio": np.std(ratios),
            })
    return results


def fig_survival():
    apply_style()
    results = _analyze(E13_KAPPA3_VALUES, E13_N_SEEDS)
    if not results:
        print("  [skip] No E13 data")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True,
                                     gridspec_kw={"hspace": 0.1})
    ratios = [r["ratio"] for r in results]

    ax1.plot(ratios, [r["surv_prob"] for r in results], "o-", color="navy",
             linewidth=2, markersize=8)
    ax1.fill_between(ratios, [r["surv_prob"] for r in results], alpha=0.1, color="navy")
    ax1.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    ax1.set_ylabel("Survival probability")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_title("Metastable State Survival vs. Well Depth")

    ax2.errorbar(ratios,
                 [r["mean_ratio"] for r in results],
                 yerr=[r["std_ratio"] for r in results],
                 fmt="s-", color="darkred", capsize=3)
    ax2.axhline(y=1.0, color="green", linestyle="--", label="Healthy ($\\alpha_3/w_3=1$)")
    ax2.axhline(y=0.5, color="orange", linestyle="--", label="Threshold (0.5)")
    ax2.set_xlabel("Depth ratio $\\kappa_3 / \\kappa_1$")
    ax2.set_ylabel("$\\alpha_3 / w_3$")
    ax2.legend(fontsize=9)

    # Secondary x-axis: absolute κ₃
    ax_top = ax1.twiny()
    ax_top.set_xlim(ax1.get_xlim())
    ax_top.set_xticks(ratios)
    ax_top.set_xticklabels([f"{r['kappa3']:.0f}" for r in results], fontsize=8)
    ax_top.set_xlabel("$\\kappa_3$", fontsize=10)

    out = FIGURES_DIR / "fig_e13_survival.pdf"
    fig.savefig(out)
    print(f"  Saved: {out}")
    plt.close(fig)


def fig_evolution():
    apply_style()
    base = RESULTS_DIR / "e13_sweep"

    selected = E13_KAPPA3_VALUES[::2] if len(E13_KAPPA3_VALUES) > 6 else E13_KAPPA3_VALUES
    ncols = min(3, len(selected))
    nrows = (len(selected) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows),
                              sharex=True, sharey=True,
                              gridspec_kw={"hspace": 0.3, "wspace": 0.15})
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)

    for idx, k3 in enumerate(selected):
        r, c = idx // ncols, idx % ncols
        ax = axes[r, c]
        for seed in range(E13_N_SEEDS):
            records = load_tracking(base / f"kappa3_{k3}" / f"seed_{seed}" / "mode_tracking.jsonl")
            if not records:
                continue
            epochs = [rec["epoch"] for rec in records]
            alphas = np.array([rec["alpha"] for rec in records])
            for k in range(min(alphas.shape[1], 3)):
                ax.plot(epochs, alphas[:, k], color=MODE_COLORS_3[k],
                        alpha=0.25, linewidth=0.5)
        ax.set_title(f"$\\kappa_3$={k3:.0f} (ratio={k3/E13_KAPPA1:.2f})", fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        if c == 0:
            ax.set_ylabel("$\\alpha_k$")
        if r == nrows - 1:
            ax.set_xlabel("Epoch")

    for idx in range(len(selected), nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    # Legend
    for k, lbl in enumerate(["Deep 0", "Deep 1", "Metastable"]):
        axes[0, 0].plot([], [], color=MODE_COLORS_3[k], linewidth=2, label=lbl)
    axes[0, 0].legend(fontsize=8)

    fig.suptitle("Mode Weight Evolution vs. Metastable Well Depth", fontsize=13, y=1.01)
    out = FIGURES_DIR / "fig_e13_evolution.pdf"
    fig.savefig(out)
    print(f"  Saved: {out}")
    plt.close(fig)


def generate_all():
    print("E13: Metastable Survival")
    fig_survival()
    fig_evolution()
