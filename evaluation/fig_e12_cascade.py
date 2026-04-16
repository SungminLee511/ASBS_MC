"""
E12: Mode Death Cascade on B6

Figures:
  - fig_e12_cascade.pdf: survival curve + death order + heatmap + grid snapshots
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from mc_utils import load_tracking
from config import RESULTS_DIR, FIGURES_DIR, apply_style


def _find_deaths(records, threshold=0.01, consecutive=5):
    tw = np.array(records[0]["target_w"])
    K = len(tw)
    deaths = {}
    streak = np.zeros(K, dtype=int)
    for r in records:
        a = np.array(r["alpha"])
        for k in range(K):
            if k in deaths:
                continue
            if tw[k] > 1e-10 and a[k] < threshold * tw[k]:
                streak[k] += 1
                if streak[k] >= consecutive:
                    deaths[k] = r["epoch"]
            else:
                streak[k] = 0
    return deaths


def generate_all(seed=0):
    apply_style()
    print("E12: Death Cascade")

    records = load_tracking(RESULTS_DIR / "b6_asbs" / f"seed_{seed}" / "mode_tracking.jsonl")
    if not records:
        print("  [skip] No B6 data")
        return

    tw = np.array(records[0]["target_w"])
    K = len(tw)
    epochs = np.array([r["epoch"] for r in records])
    alphas = np.array([r["alpha"] for r in records])
    deaths = _find_deaths(records)

    # Reconstruct grid centers (5×5, spacing=6)
    grid = np.arange(5) - 2.0
    gx, gy = np.meshgrid(grid, grid, indexing="ij")
    centers = np.stack([gx.ravel(), gy.ravel()], axis=-1) * 6.0

    snapshot_epochs = [int(epochs[-1] * f) for f in [0.05, 0.2, 0.5, 0.8, 1.0]]

    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(3, 5, hspace=0.4, wspace=0.35)

    # Row 0: Survival curve + Death order scatter + summary stats
    ax_surv = fig.add_subplot(gs[0, :2])
    alive_c = []
    for r in records:
        a = np.array(r["alpha"])
        alive_c.append(sum(1 for k in range(K) if a[k] > 0.01 * tw[k]))
    ax_surv.step(epochs, alive_c, where="post", color="navy", linewidth=2)
    ax_surv.fill_between(epochs, alive_c, step="post", alpha=0.15, color="navy")
    ax_surv.axhline(y=K, color="gray", linestyle="--", alpha=0.5)
    ax_surv.set_ylabel("Surviving modes")
    ax_surv.set_xlabel("Epoch")
    ax_surv.set_title("Survival Curve")
    ax_surv.set_ylim(0, K + 1)

    ax_death = fig.add_subplot(gs[0, 2:4])
    if deaths:
        ks = sorted(deaths.keys(), key=lambda x: deaths[x])
        ws = [tw[k] for k in ks]
        eps = [deaths[k] for k in ks]
        ax_death.scatter(ws, eps, c="darkred", s=40, zorder=3)
        if len(ws) > 2:
            corr = np.corrcoef(ws, eps)[0, 1]
            ax_death.text(0.95, 0.95, f"r = {corr:.2f}", transform=ax_death.transAxes,
                          ha="right", va="top", fontsize=10,
                          bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax_death.set_xlabel("Target weight $w_k$")
    ax_death.set_ylabel("Death epoch")
    ax_death.set_title("Death Order vs. Target Weight")

    # Stats text
    ax_stats = fig.add_subplot(gs[0, 4])
    ax_stats.axis("off")
    stats_text = (
        f"K = {K}\n"
        f"Died: {len(deaths)}/{K}\n"
        f"Survived: {K - len(deaths)}\n"
        f"Final KL = {records[-1]['kl']:.4f}\n"
        f"Final TV = {records[-1]['tv']:.4f}"
    )
    ax_stats.text(0.1, 0.5, stats_text, transform=ax_stats.transAxes,
                  fontsize=11, verticalalignment="center", family="monospace")

    # Row 1: Grid snapshots
    norm = TwoSlopeNorm(vmin=0, vcenter=1.0, vmax=2.0)
    for i, sep in enumerate(snapshot_epochs):
        ax = fig.add_subplot(gs[1, i])
        closest = min(records, key=lambda r: abs(r["epoch"] - sep))
        a = np.array(closest["alpha"])
        ratios = np.clip(a / np.maximum(tw, 1e-10), 0, 2)
        colors = plt.cm.RdYlGn(norm(ratios))
        for k in range(K):
            ax.scatter(centers[k, 0], centers[k, 1], c=[colors[k]], s=100,
                       edgecolors="black", linewidths=0.5, zorder=3)
        ax.set_title(f"Epoch {closest['epoch']}", fontsize=10)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)

    # Colorbar for snapshots
    sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.38, 0.015, 0.22])
    fig.colorbar(sm, cax=cbar_ax, label="$\\alpha_k/w_k$")

    # Row 2: Full heatmap
    ax_hm = fig.add_subplot(gs[2, :])
    sort_idx = np.argsort(tw)
    ratios_all = alphas[:, sort_idx] / np.maximum(tw[sort_idx], 1e-10)
    ratios_all = np.clip(ratios_all, 0, 3)
    im = ax_hm.imshow(ratios_all.T, aspect="auto", origin="lower",
                       extent=[epochs[0], epochs[-1], 0, K],
                       cmap="RdYlGn", vmin=0, vmax=2, interpolation="nearest")
    ax_hm.set_xlabel("Epoch")
    ax_hm.set_ylabel("Mode (sorted by $w_k$ $\\uparrow$)")
    ax_hm.set_title("Death Cascade Heatmap")
    fig.colorbar(im, ax=ax_hm, fraction=0.02, pad=0.01, label="$\\alpha_k/w_k$")

    fig.suptitle(f"Mode Death Cascade — B6 (K={K})", fontsize=14, y=1.01)
    out = FIGURES_DIR / "fig_e12_cascade.pdf"
    fig.savefig(out)
    print(f"  Saved: {out}")
    plt.close(fig)

    # Save death data
    data = {"deaths": {str(k): v for k, v in deaths.items()}, "K": K, "target_w": tw.tolist()}
    with open(FIGURES_DIR / "fig_e12_deaths.json", "w") as f:
        json.dump(data, f, indent=2)
