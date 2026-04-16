#!/usr/bin/env python
"""
E1: Mode Weight Tracking

Tracks α_k vs. epoch for all benchmarks. Key visualization for B6:
death cascade heatmap (25 modes × epochs, colored by α_k / w_k).

Usage:
    # Launch a training run (mode_tracking.jsonl is auto-generated)
    python train.py experiment=b6_asbs num_epochs=2000 \\
        hydra.run.dir=results/b6_asbs/seed_0

    # Plot mode weight evolution
    python scripts/e1_mode_tracking.py plot \\
        --tracking results/b6_asbs/seed_0/mode_tracking.jsonl \\
        --output figures/e1_b6_alpha.pdf

    # Plot death cascade heatmap (B6 only)
    python scripts/e1_mode_tracking.py heatmap \\
        --tracking results/b6_asbs/seed_0/mode_tracking.jsonl \\
        --output figures/e1_b6_heatmap.pdf

    # Plot multi-seed overlay
    python scripts/e1_mode_tracking.py multi \\
        --results-dir results/b1_asbs \\
        --output figures/e1_b1_multi.pdf
"""

import argparse
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mc_utils import load_tracking


def plot_alpha_vs_epoch(
    tracking_path: str,
    output_path: str = "figures/e1_alpha.pdf",
    title: str = None,
):
    """Plot α_k vs. epoch for all modes, with target weights as dashed lines."""
    records = load_tracking(tracking_path)
    if not records:
        print(f"No records found in {tracking_path}")
        return

    epochs = np.array([r["epoch"] for r in records])
    alphas = np.array([r["alpha"] for r in records])  # (T, K)
    target_w = np.array(records[0]["target_w"])  # (K,)
    K = alphas.shape[1]

    # Sort modes by target weight (largest first)
    sort_idx = np.argsort(-target_w)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[3, 1],
                                     sharex=True, gridspec_kw={"hspace": 0.08})
    cmap = plt.cm.tab20 if K > 10 else plt.cm.tab10
    colors = [cmap(i / max(K - 1, 1)) for i in range(K)]

    for rank, k in enumerate(sort_idx):
        c = colors[rank]
        ax1.plot(epochs, alphas[:, k], color=c, linewidth=1.2,
                 label=f"Mode {k} ($w$={target_w[k]:.3f})" if K <= 10 else None)
        ax1.axhline(y=target_w[k], color=c, linestyle="--", alpha=0.4, linewidth=0.8)

    ax1.set_ylabel("$\\alpha_k$ (mode weight)", fontsize=12)
    if title:
        ax1.set_title(title, fontsize=14)
    else:
        ax1.set_title(f"E1: Mode Weight Evolution (K={K})", fontsize=14)
    if K <= 10:
        ax1.legend(fontsize=8, ncol=2, loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Bottom panel: KL and TV
    kl_vals = np.array([r["kl"] for r in records])
    tv_vals = np.array([r["tv"] for r in records])
    alive_vals = np.array([r["alive_modes"] for r in records])

    ax2.plot(epochs, kl_vals, color="darkred", linewidth=1.5, label="KL($\\alpha$ || $w$)")
    ax2_tv = ax2.twinx()
    ax2_tv.plot(epochs, alive_vals, color="forestgreen", linewidth=1.5,
                linestyle="--", label=f"Alive modes (/{K})")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("KL divergence", fontsize=12, color="darkred")
    ax2_tv.set_ylabel("Alive modes", fontsize=12, color="forestgreen")
    ax2.grid(True, alpha=0.3)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_tv.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_death_cascade_heatmap(
    tracking_path: str,
    output_path: str = "figures/e1_heatmap.pdf",
    title: str = None,
):
    """Plot death cascade heatmap: modes (sorted by w_k) × epochs, colored by α_k / w_k."""
    records = load_tracking(tracking_path)
    if not records:
        print(f"No records found in {tracking_path}")
        return

    epochs = np.array([r["epoch"] for r in records])
    alphas = np.array([r["alpha"] for r in records])  # (T, K)
    target_w = np.array(records[0]["target_w"])  # (K,)
    K = alphas.shape[1]

    # Sort modes by target weight (smallest first for death wave visualization)
    sort_idx = np.argsort(target_w)

    # Compute α_k / w_k ratio (clamp to avoid division by zero)
    ratios = alphas[:, sort_idx] / np.maximum(target_w[sort_idx], 1e-10)  # (T, K)
    ratios = np.clip(ratios, 0, 5)  # clip for visualization

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    im = ax.imshow(
        ratios.T,
        aspect="auto",
        origin="lower",
        extent=[epochs[0], epochs[-1], 0, K],
        cmap="RdYlGn",
        vmin=0,
        vmax=2,
        interpolation="nearest",
    )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Mode (sorted by target weight ↑)", fontsize=12)
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f"E1: Death Cascade Heatmap (K={K})", fontsize=14)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("$\\alpha_k / w_k$ ratio", fontsize=11)
    cbar.set_ticks([0, 0.5, 1.0, 1.5, 2.0])

    # Add "dead" annotation
    ax.text(epochs[-1] * 0.02, K * 0.05, "← smallest $w_k$", fontsize=9, color="white")
    ax.text(epochs[-1] * 0.02, K * 0.92, "← largest $w_k$", fontsize=9, color="black")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_multi_seed(
    results_dir: str,
    output_path: str = "figures/e1_multi.pdf",
    title: str = None,
):
    """Plot α_k vs. epoch overlaid across multiple seeds."""
    results_dir = Path(results_dir)
    seed_dirs = sorted(results_dir.glob("seed_*"))
    if not seed_dirs:
        print(f"No seed directories found in {results_dir}")
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    target_w = None
    K = None

    for seed_dir in seed_dirs:
        records = load_tracking(seed_dir / "mode_tracking.jsonl")
        if not records:
            continue

        epochs = np.array([r["epoch"] for r in records])
        alphas = np.array([r["alpha"] for r in records])
        if target_w is None:
            target_w = np.array(records[0]["target_w"])
            K = len(target_w)

        cmap = plt.cm.tab10
        for k in range(K):
            ax.plot(epochs, alphas[:, k], color=cmap(k / max(K - 1, 1)),
                    alpha=0.3, linewidth=0.7)

    if target_w is not None:
        cmap = plt.cm.tab10
        for k in range(K):
            ax.axhline(y=target_w[k], color=cmap(k / max(K - 1, 1)),
                        linestyle="--", linewidth=1.5, alpha=0.8,
                        label=f"$w_{k}$ = {target_w[k]:.3f}")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("$\\alpha_k$", fontsize=12)
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f"E1: Multi-Seed Mode Weights ({len(seed_dirs)} seeds, K={K})", fontsize=14)
    if K and K <= 6:
        ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"Saved: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="E1: Mode Weight Tracking")
    subparsers = parser.add_subparsers(dest="command")

    # Single run α_k plot
    plot_parser = subparsers.add_parser("plot", help="Plot α_k vs. epoch (single run)")
    plot_parser.add_argument("--tracking", required=True, help="Path to mode_tracking.jsonl")
    plot_parser.add_argument("--output", default="figures/e1_alpha.pdf")
    plot_parser.add_argument("--title", default=None)

    # Death cascade heatmap
    hm_parser = subparsers.add_parser("heatmap", help="Plot death cascade heatmap")
    hm_parser.add_argument("--tracking", required=True, help="Path to mode_tracking.jsonl")
    hm_parser.add_argument("--output", default="figures/e1_heatmap.pdf")
    hm_parser.add_argument("--title", default=None)

    # Multi-seed overlay
    multi_parser = subparsers.add_parser("multi", help="Plot multi-seed overlay")
    multi_parser.add_argument("--results-dir", required=True)
    multi_parser.add_argument("--output", default="figures/e1_multi.pdf")
    multi_parser.add_argument("--title", default=None)

    args = parser.parse_args()

    if args.command == "plot":
        plot_alpha_vs_epoch(args.tracking, args.output, args.title)
    elif args.command == "heatmap":
        plot_death_cascade_heatmap(args.tracking, args.output, args.title)
    elif args.command == "multi":
        plot_multi_seed(args.results_dir, args.output, args.title)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
