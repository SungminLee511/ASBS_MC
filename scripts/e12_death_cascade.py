#!/usr/bin/env python
"""
E12: Mode Death Cascade on 25-Mode Grid (B6)

Comprehensive analysis of mode death dynamics:
  - Survival curve: surviving modes vs. epoch
  - Death order vs. target weight: scatter (theory predicts negative correlation)
  - 5×5 grid snapshots colored by α_k/w_k at selected epochs
  - Redistribution analysis: where does mass go when a mode dies

Usage:
    python scripts/e12_death_cascade.py \\
        --tracking results/b6_asbs/seed_0/mode_tracking.jsonl \\
        --output figures/e12_b6_cascade.pdf
"""

import argparse
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mc_utils import load_tracking


def find_death_epochs(records, death_threshold=0.01, consecutive=5):
    """Find the epoch at which each mode "dies".

    A mode is dead when α_k < death_threshold * w_k for `consecutive`
    consecutive tracking points.

    Returns:
        death_epochs: dict {mode_k: epoch} (None if never dies)
    """
    if not records:
        return {}

    target_w = np.array(records[0]["target_w"])
    K = len(target_w)

    death_epochs = {}
    streak = np.zeros(K, dtype=int)

    for r in records:
        alpha = np.array(r["alpha"])
        for k in range(K):
            if k in death_epochs:
                continue
            if target_w[k] > 1e-10 and alpha[k] < death_threshold * target_w[k]:
                streak[k] += 1
                if streak[k] >= consecutive:
                    death_epochs[k] = r["epoch"]
            else:
                streak[k] = 0

    return death_epochs


def plot_survival_curve(records, death_epochs, ax, K):
    """Plot number of surviving modes vs. epoch."""
    epochs = [r["epoch"] for r in records]
    alive_counts = []

    for r in records:
        alpha = np.array(r["alpha"])
        target_w = np.array(r["target_w"])
        alive = sum(1 for k in range(K) if alpha[k] > 0.01 * target_w[k])
        alive_counts.append(alive)

    ax.step(epochs, alive_counts, where="post", color="navy", linewidth=2)
    ax.fill_between(epochs, alive_counts, step="post", alpha=0.15, color="navy")
    ax.set_ylabel("Surviving modes", fontsize=12)
    ax.set_ylim(0, K + 1)
    ax.axhline(y=K, color="gray", linestyle="--", alpha=0.5, label=f"Total modes ({K})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title("Survival Curve", fontsize=12)


def plot_death_order(death_epochs, target_w, ax):
    """Scatter: death epoch vs. target weight w_k."""
    ks = sorted(death_epochs.keys())
    if not ks:
        ax.text(0.5, 0.5, "No modes died", transform=ax.transAxes,
                ha="center", fontsize=12)
        return

    w_vals = [target_w[k] for k in ks]
    ep_vals = [death_epochs[k] for k in ks]

    ax.scatter(w_vals, ep_vals, c="darkred", s=50, zorder=3)

    # Fit trend line
    if len(w_vals) > 2:
        z = np.polyfit(np.log(w_vals), ep_vals, 1)
        w_fit = np.linspace(min(w_vals), max(w_vals), 50)
        ep_fit = z[0] * np.log(w_fit) + z[1]
        ax.plot(w_fit, ep_fit, "--", color="gray", alpha=0.6)

        # Correlation
        corr = np.corrcoef(w_vals, ep_vals)[0, 1]
        ax.text(0.95, 0.95, f"r = {corr:.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=10,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("Target weight $w_k$", fontsize=11)
    ax.set_ylabel("Death epoch", fontsize=11)
    ax.set_title("Death Order vs. Target Weight", fontsize=12)
    ax.grid(True, alpha=0.3)


def plot_grid_snapshots(records, target_w, mode_centers_2d, snapshot_epochs, axes_row):
    """Plot 5×5 grid colored by α_k/w_k at selected epochs."""
    if mode_centers_2d is None:
        return

    K = len(target_w)

    for ax, target_ep in zip(axes_row, snapshot_epochs):
        # Find closest record
        closest_r = min(records, key=lambda r: abs(r["epoch"] - target_ep))
        alpha = np.array(closest_r["alpha"])
        actual_ep = closest_r["epoch"]

        ratios = np.zeros(K)
        for k in range(K):
            if target_w[k] > 1e-10:
                ratios[k] = alpha[k] / target_w[k]

        # Color: green=healthy (ratio≈1), yellow=suppressed, red=dead
        norm = TwoSlopeNorm(vmin=0, vcenter=1.0, vmax=2.0)
        colors = plt.cm.RdYlGn(norm(np.clip(ratios, 0, 2)))

        for k in range(K):
            ax.scatter(mode_centers_2d[k, 0], mode_centers_2d[k, 1],
                       c=[colors[k]], s=120, edgecolors="black", linewidths=0.5,
                       zorder=3)

        ax.set_title(f"Epoch {actual_ep}", fontsize=10)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=8)


def plot_redistribution(records, death_epochs, target_w, mode_centers_2d, ax):
    """Analyze where mass goes when modes die.

    For each dead mode k, find the epoch just before and after death,
    compute Δα_l for surviving modes l.
    """
    K = len(target_w)
    if not death_epochs or mode_centers_2d is None:
        ax.text(0.5, 0.5, "No redistribution data", transform=ax.transAxes,
                ha="center", fontsize=12)
        return

    # For each dead mode, find nearest surviving mode that gained mass
    nearest_gains = []  # (distance, gain fraction) pairs

    epoch_list = [r["epoch"] for r in records]
    alpha_list = [np.array(r["alpha"]) for r in records]

    for dead_k, death_ep in sorted(death_epochs.items(), key=lambda x: x[1]):
        # Find record index at death
        idx = min(range(len(epoch_list)), key=lambda i: abs(epoch_list[i] - death_ep))
        if idx < 1:
            continue

        alpha_before = alpha_list[max(0, idx - 1)]
        alpha_after = alpha_list[min(len(alpha_list) - 1, idx + 1)]
        delta_alpha = alpha_after - alpha_before

        # For each surviving mode, compute gain and distance to dead mode
        for l in range(K):
            if l == dead_k or l in death_epochs:
                continue
            if delta_alpha[l] > 0.001:
                dist = np.linalg.norm(mode_centers_2d[dead_k] - mode_centers_2d[l])
                nearest_gains.append((dist, delta_alpha[l]))

    if nearest_gains:
        dists, gains = zip(*nearest_gains)
        ax.scatter(dists, gains, c="steelblue", s=30, alpha=0.6)
        ax.set_xlabel("Distance to dead mode", fontsize=11)
        ax.set_ylabel("$\\Delta\\alpha_l$ (gain)", fontsize=11)
        ax.set_title("Mass Redistribution (local?)", fontsize=12)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                ha="center", fontsize=12)


def plot_full_cascade(
    tracking_path: str,
    output_path: str = "figures/e12_cascade.pdf",
    snapshot_epochs: list = None,
    title: str = None,
):
    """Full death cascade analysis."""
    records = load_tracking(tracking_path)
    if not records:
        print(f"No records in {tracking_path}")
        return

    target_w = np.array(records[0]["target_w"])
    K = len(target_w)

    if snapshot_epochs is None:
        max_ep = records[-1]["epoch"]
        snapshot_epochs = [int(max_ep * f) for f in [0.05, 0.2, 0.5, 0.8, 1.0]]

    death_epochs = find_death_epochs(records)
    print(f"K={K} modes, {len(death_epochs)} died")
    for k in sorted(death_epochs, key=lambda x: death_epochs[x]):
        print(f"  Mode {k} (w={target_w[k]:.4f}): died at epoch {death_epochs[k]}")

    # Try to reconstruct mode centers from energy
    # For B6, centers are on a 5×5 grid
    mode_centers_2d = None
    if K == 25:
        grid = np.arange(5) - 2.0  # centered
        gx, gy = np.meshgrid(grid, grid, indexing="ij")
        mode_centers_2d = np.stack([gx.ravel(), gy.ravel()], axis=-1) * 6.0  # default spacing

    n_snapshots = len(snapshot_epochs)

    fig = plt.figure(figsize=(max(14, 3 * n_snapshots), 14))
    gs = fig.add_gridspec(3, max(n_snapshots, 3), hspace=0.4, wspace=0.35)

    # Row 1: Survival curve + Death order + Redistribution
    ax_surv = fig.add_subplot(gs[0, 0])
    plot_survival_curve(records, death_epochs, ax_surv, K)

    ax_death = fig.add_subplot(gs[0, 1])
    plot_death_order(death_epochs, target_w, ax_death)

    ax_redist = fig.add_subplot(gs[0, 2])
    plot_redistribution(records, death_epochs, target_w, mode_centers_2d, ax_redist)

    # Row 2: Grid snapshots
    if mode_centers_2d is not None:
        snap_axes = [fig.add_subplot(gs[1, i]) for i in range(n_snapshots)]
        plot_grid_snapshots(records, target_w, mode_centers_2d, snapshot_epochs, snap_axes)

        # Add colorbar for the grid snapshots
        sm = plt.cm.ScalarMappable(cmap="RdYlGn",
                                    norm=TwoSlopeNorm(vmin=0, vcenter=1.0, vmax=2.0))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=snap_axes, fraction=0.02, pad=0.02)
        cbar.set_label("$\\alpha_k / w_k$", fontsize=10)

    # Row 3: α_k/w_k heatmap (modes × epochs)
    ax_hm = fig.add_subplot(gs[2, :])
    epochs = np.array([r["epoch"] for r in records])
    alphas = np.array([r["alpha"] for r in records])
    sort_idx = np.argsort(target_w)  # smallest first

    ratios = np.zeros_like(alphas)
    for k in range(K):
        ratios[:, k] = alphas[:, k] / max(target_w[k], 1e-10)
    ratios = np.clip(ratios[:, sort_idx], 0, 3)

    im = ax_hm.imshow(ratios.T, aspect="auto", origin="lower",
                       extent=[epochs[0], epochs[-1], 0, K],
                       cmap="RdYlGn", vmin=0, vmax=2,
                       interpolation="nearest")
    ax_hm.set_xlabel("Epoch", fontsize=12)
    ax_hm.set_ylabel("Mode (sorted by $w_k$ ↑)", fontsize=12)
    ax_hm.set_title("Death Cascade Heatmap", fontsize=12)
    fig.colorbar(im, ax=ax_hm, fraction=0.02, pad=0.01, label="$\\alpha_k/w_k$")

    if title:
        fig.suptitle(title, fontsize=14, y=1.01)
    else:
        fig.suptitle(f"E12: Mode Death Cascade (K={K})", fontsize=14, y=1.01)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"Saved: {output_path}")
    plt.close(fig)

    # Save death epoch data
    data_path = output_path.replace(".pdf", "_deaths.json")
    death_data = {
        "death_epochs": {str(k): ep for k, ep in death_epochs.items()},
        "target_weights": target_w.tolist(),
        "K": K,
    }
    with open(data_path, "w") as f:
        json.dump(death_data, f, indent=2)
    print(f"Saved: {data_path}")


def main():
    parser = argparse.ArgumentParser(description="E12: Mode Death Cascade")
    parser.add_argument("--tracking", required=True)
    parser.add_argument("--output", default="figures/e12_cascade.pdf")
    parser.add_argument("--snapshot-epochs", type=int, nargs="+", default=None)
    parser.add_argument("--title", default=None)

    args = parser.parse_args()
    plot_full_cascade(args.tracking, args.output, args.snapshot_epochs, args.title)


if __name__ == "__main__":
    main()
