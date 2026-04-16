"""
E1: Mode Weight Tracking

Figures:
  - fig_e1_{benchmark}_alpha.pdf: α_k vs epoch for each benchmark (single seed, seed 0)
  - fig_e1_b6_heatmap.pdf: death cascade heatmap for B6
  - fig_e1_{benchmark}_multi.pdf: multi-seed overlay
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from mc_utils import load_tracking
from config import RESULTS_DIR, FIGURES_DIR, apply_style, E1_BENCHMARKS


def fig_alpha_single(benchmark, seed=0):
    """α_k vs epoch for a single run."""
    apply_style()
    tracking = RESULTS_DIR / f"{benchmark}_asbs" / f"seed_{seed}" / "mode_tracking.jsonl"
    records = load_tracking(tracking)
    if not records:
        print(f"  [skip] No data: {tracking}")
        return

    epochs = np.array([r["epoch"] for r in records])
    alphas = np.array([r["alpha"] for r in records])
    target_w = np.array(records[0]["target_w"])
    K = alphas.shape[1]
    kl_vals = np.array([r["kl"] for r in records])
    alive_vals = np.array([r["alive_modes"] for r in records])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7),
                                     height_ratios=[3, 1], sharex=True,
                                     gridspec_kw={"hspace": 0.08})

    cmap = plt.cm.tab20 if K > 10 else plt.cm.tab10
    sort_idx = np.argsort(-target_w)
    for rank, k in enumerate(sort_idx):
        c = cmap(rank / max(K - 1, 1))
        label = f"Mode {k} ($w$={target_w[k]:.3f})" if K <= 10 else None
        ax1.plot(epochs, alphas[:, k], color=c, linewidth=1.0, label=label)
        ax1.axhline(y=target_w[k], color=c, linestyle="--", alpha=0.3, linewidth=0.6)

    ax1.set_ylabel("$\\alpha_k$")
    ax1.set_title(f"Mode Weight Evolution — {benchmark.upper()} (K={K})")
    if K <= 10:
        ax1.legend(fontsize=8, ncol=2)

    ax2.plot(epochs, kl_vals, color="darkred", label="KL($\\alpha$ || $w$)")
    ax2_r = ax2.twinx()
    ax2_r.plot(epochs, alive_vals, color="forestgreen", linestyle="--",
               label=f"Alive (/{K})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("KL", color="darkred")
    ax2_r.set_ylabel("Alive", color="forestgreen")

    out = FIGURES_DIR / f"fig_e1_{benchmark}_alpha.pdf"
    fig.savefig(out)
    print(f"  Saved: {out}")
    plt.close(fig)


def fig_b6_heatmap(seed=0):
    """Death cascade heatmap for B6 (25 modes)."""
    apply_style()
    tracking = RESULTS_DIR / "b6_asbs" / f"seed_{seed}" / "mode_tracking.jsonl"
    records = load_tracking(tracking)
    if not records:
        print(f"  [skip] No B6 data")
        return

    epochs = np.array([r["epoch"] for r in records])
    alphas = np.array([r["alpha"] for r in records])
    target_w = np.array(records[0]["target_w"])
    K = alphas.shape[1]

    sort_idx = np.argsort(target_w)  # smallest first
    ratios = alphas[:, sort_idx] / np.maximum(target_w[sort_idx], 1e-10)
    ratios = np.clip(ratios, 0, 3)

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    im = ax.imshow(ratios.T, aspect="auto", origin="lower",
                    extent=[epochs[0], epochs[-1], 0, K],
                    cmap="RdYlGn", vmin=0, vmax=2, interpolation="nearest")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mode (sorted by $w_k$ $\\uparrow$)")
    ax.set_title(f"Death Cascade Heatmap — B6 (K={K})")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("$\\alpha_k / w_k$")

    out = FIGURES_DIR / "fig_e1_b6_heatmap.pdf"
    fig.savefig(out)
    print(f"  Saved: {out}")
    plt.close(fig)


def fig_multi_seed(benchmark, n_seeds=10):
    """Multi-seed overlay."""
    apply_style()
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    target_w = None
    K = None
    n_plotted = 0

    for seed in range(n_seeds):
        records = load_tracking(
            RESULTS_DIR / f"{benchmark}_asbs" / f"seed_{seed}" / "mode_tracking.jsonl"
        )
        if not records:
            continue
        epochs = [r["epoch"] for r in records]
        alphas = np.array([r["alpha"] for r in records])
        if target_w is None:
            target_w = np.array(records[0]["target_w"])
            K = len(target_w)
        cmap = plt.cm.tab10
        for k in range(K):
            ax.plot(epochs, alphas[:, k], color=cmap(k / max(K - 1, 1)),
                    alpha=0.25, linewidth=0.6)
        n_plotted += 1

    if target_w is not None:
        cmap = plt.cm.tab10
        for k in range(K):
            ax.axhline(y=target_w[k], color=cmap(k / max(K - 1, 1)),
                        linestyle="--", linewidth=1.5, alpha=0.8,
                        label=f"$w_{k}$={target_w[k]:.3f}")
        if K <= 6:
            ax.legend(fontsize=9)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("$\\alpha_k$")
    ax.set_title(f"Multi-Seed Mode Weights — {benchmark.upper()} ({n_plotted} seeds)")

    out = FIGURES_DIR / f"fig_e1_{benchmark}_multi.pdf"
    fig.savefig(out)
    print(f"  Saved: {out}")
    plt.close(fig)


def generate_all():
    print("E1: Mode Weight Tracking")
    for bm in E1_BENCHMARKS:
        fig_alpha_single(bm)
    fig_b6_heatmap()
    for bm in ["b1", "b7"]:
        fig_multi_seed(bm)
