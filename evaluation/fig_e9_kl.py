"""
E9: KL Decomposition Tracking

Figures:
  - fig_e9_{benchmark}_kl.pdf: KL(α||w) + TV + alive modes vs epoch
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from mc_utils import load_tracking
from config import RESULTS_DIR, FIGURES_DIR, apply_style


def fig_kl(benchmark, seed=0):
    apply_style()
    records = load_tracking(RESULTS_DIR / f"{benchmark}_asbs" / f"seed_{seed}" / "mode_tracking.jsonl")
    if not records:
        print(f"  [skip] No data for {benchmark}")
        return

    epochs = np.array([r["epoch"] for r in records])
    kl = np.array([r["kl"] for r in records])
    tv = np.array([r["tv"] for r in records])
    alive = np.array([r["alive_modes"] for r in records])
    K = len(records[0]["alpha"])

    nrows = 3 if K > 10 else 2
    fig, axes = plt.subplots(nrows, 1, figsize=(10, 3.5 * nrows),
                              sharex=True, gridspec_kw={"hspace": 0.1})

    axes[0].plot(epochs, kl, "o-", color="darkred", markersize=3, label="KL($\\alpha$ || $w$)")
    axes[0].fill_between(epochs, kl, alpha=0.15, color="darkred")
    axes[0].set_ylabel("KL divergence")
    axes[0].set_title(f"KL Decomposition — {benchmark.upper()} (K={K})")
    axes[0].legend()

    axes[1].plot(epochs, tv, "D-", color="darkorange", markersize=3, label="TV($\\alpha$, $w$)")
    ax_r = axes[1].twinx()
    ax_r.plot(epochs, alive, "x--", color="forestgreen", markersize=4, alpha=0.7, label=f"Alive (/{K})")
    ax_r.set_ylabel("Alive modes", color="forestgreen")
    ax_r.set_ylim(0, K + 1)
    axes[1].set_ylabel("TV distance")
    lines = axes[1].get_legend_handles_labels()
    lines2 = ax_r.get_legend_handles_labels()
    axes[1].legend(lines[0] + lines2[0], lines[1] + lines2[1], fontsize=9)

    if nrows == 3:
        dead = K - alive
        axes[2].plot(epochs, dead, "o-", color="darkred", markersize=3)
        axes[2].fill_between(epochs, dead, alpha=0.2, color="darkred")
        axes[2].set_ylabel(f"Dead modes (/{K})")
        axes[2].set_ylim(-0.5, K + 0.5)

    axes[-1].set_xlabel("Epoch")

    out = FIGURES_DIR / f"fig_e9_{benchmark}_kl.pdf"
    fig.savefig(out)
    print(f"  Saved: {out}")
    plt.close(fig)


def generate_all():
    print("E9: KL Decomposition")
    for bm in ["b1", "b5", "b6"]:
        fig_kl(bm)
