#!/usr/bin/env python
"""
E1 on B8: LJ3 Permutation Counting

Track how many of the 6 permutational basins the sampler discovers
across training epochs. Uses the sorting-based permutation assignment.

Usage:
    # Analyze a completed B8 run
    python scripts/e1_lj3_permutations.py \\
        --run-dir results/b8_asbs/seed_0 \\
        --output figures/e1_b8_permutations.pdf

    # Multi-seed analysis
    python scripts/e1_lj3_permutations.py \\
        --results-dir results/b8_asbs \\
        --output figures/e1_b8_multi.pdf
"""

import argparse
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mc_utils import (
    load_model_from_checkpoint,
    generate_samples,
    assign_lj3_permutations,
    compute_mode_weights,
    find_checkpoints,
)


def analyze_checkpoints(run_dir, n_samples=5000, device="cuda"):
    """Count discovered permutational basins at each checkpoint."""
    ckpts = find_checkpoints(run_dir)
    if not ckpts:
        print(f"No checkpoints in {run_dir}")
        return []

    results = []
    for ckpt in ckpts:
        ep = int(ckpt.stem.split("_")[-1])
        print(f"  Epoch {ep}...")

        try:
            model = load_model_from_checkpoint(str(ckpt), device=device)
            samples = generate_samples(model, n_samples=n_samples, device=device)

            assignments = assign_lj3_permutations(samples)
            alpha = compute_mode_weights(assignments, 6)

            # Count discovered basins (α_k > 1%)
            discovered = int((alpha > 0.01).sum().item())

            results.append({
                "epoch": ep,
                "alpha": alpha.tolist(),
                "discovered": discovered,
            })
        except Exception as e:
            print(f"    Error: {e}")

    return results


def plot_single_run(run_dir, output_path="figures/e1_b8_permutations.pdf",
                    n_samples=5000, device="cuda"):
    """Plot permutation basin discovery vs. epoch for a single run."""
    results = analyze_checkpoints(run_dir, n_samples, device)
    if not results:
        return

    epochs = [r["epoch"] for r in results]
    discovered = [r["discovered"] for r in results]
    alphas = np.array([r["alpha"] for r in results])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                     gridspec_kw={"hspace": 0.1})

    # Top: basins discovered
    ax1.step(epochs, discovered, where="post", color="navy", linewidth=2)
    ax1.fill_between(epochs, discovered, step="post", alpha=0.15, color="navy")
    ax1.axhline(y=6, color="red", linestyle="--", linewidth=1, label="All 6 permutations")
    ax1.set_ylabel("Basins discovered", fontsize=12)
    ax1.set_ylim(0, 7)
    ax1.set_title("E1/B8: LJ3 Permutational Basin Discovery", fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Bottom: α_k for each permutation
    colors = plt.cm.tab10(np.linspace(0, 0.6, 6))
    for k in range(6):
        ax2.plot(epochs, alphas[:, k], color=colors[k], linewidth=1,
                 alpha=0.7, label=f"Perm {k}")
    ax2.axhline(y=1/6, color="gray", linestyle="--", alpha=0.5,
                label="Target $w_k = 1/6$")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("$\\alpha_k$", fontsize=12)
    ax2.legend(fontsize=8, ncol=4, loc="upper right")
    ax2.grid(True, alpha=0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_multi_seed(results_dir, output_path="figures/e1_b8_multi.pdf",
                    n_samples=5000, device="cuda"):
    """Plot permutation discovery across multiple seeds."""
    results_dir = Path(results_dir)
    seed_dirs = sorted(results_dir.glob("seed_*"))
    if not seed_dirs:
        print(f"No seed dirs in {results_dir}")
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    for seed_dir in seed_dirs:
        results = analyze_checkpoints(str(seed_dir), n_samples, device)
        if not results:
            continue
        epochs = [r["epoch"] for r in results]
        discovered = [r["discovered"] for r in results]
        ax.plot(epochs, discovered, alpha=0.4, linewidth=1, color="steelblue")

    ax.axhline(y=6, color="red", linestyle="--", linewidth=1.5, label="All 6 permutations")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Basins discovered", fontsize=12)
    ax.set_title(f"E1/B8: LJ3 Basin Discovery ({len(seed_dirs)} seeds)", fontsize=14)
    ax.set_ylim(0, 7)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"Saved: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="E1/B8: LJ3 Permutation Counting")
    parser.add_argument("--run-dir", default=None, help="Single run directory")
    parser.add_argument("--results-dir", default=None, help="Multi-seed parent directory")
    parser.add_argument("--output", default="figures/e1_b8_permutations.pdf")
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    if args.run_dir:
        plot_single_run(args.run_dir, args.output, args.n_samples, args.device)
    elif args.results_dir:
        plot_multi_seed(args.results_dir, args.output, args.n_samples, args.device)
    else:
        parser.error("Provide --run-dir or --results-dir")


if __name__ == "__main__":
    main()
