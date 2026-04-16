#!/usr/bin/env python
"""
E11: Temperature Sweep on Müller-Brown

Train ASBS on B2 (Müller-Brown) at varying inverse temperatures
β ∈ {0.25, 0.5, 1.0, 2.0, 5.0, 10.0} and track mode weight evolution.

At low β: modes overlap, no collapse.
At high β: modes sharpen, shallowest well collapses.

Usage:
    # Run sweep
    python scripts/e11_muller_brown_sweep.py run \\
        --betas 0.25 0.5 1.0 2.0 5.0 10.0 \\
        --n-seeds 5 --num-epochs 1000

    # Plot results
    python scripts/e11_muller_brown_sweep.py plot \\
        --results-dir results/e11_sweep \\
        --output figures/e11_mb_sweep.pdf
"""

import argparse
import subprocess
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mc_utils import load_tracking


def launch_sweep(betas, n_seeds, num_epochs, extra_args):
    """Launch training runs for each β × seed."""
    results_base = Path("results/e11_sweep")

    procs = []
    for beta in betas:
        for seed in range(n_seeds):
            run_dir = results_base / f"beta_{beta}" / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                "python", "train.py",
                "experiment=b2_asbs",
                f"seed={seed}",
                f"num_epochs={num_epochs}",
                f"beta={beta}",
                f"hydra.run.dir={run_dir}",
            ] + extra_args

            log_file = open(run_dir / "stdout.log", "w")
            proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
            procs.append((beta, seed, proc, log_file))

    print(f"Launched {len(procs)} runs across {len(betas)} β values × {n_seeds} seeds")
    print("Waiting for completion...")

    for beta, seed, proc, log_file in procs:
        proc.wait()
        log_file.close()
        if proc.returncode != 0:
            print(f"  [β={beta}, seed={seed}] FAILED")

    print("All runs complete.")


def plot_sweep(
    results_dir: str,
    betas: list,
    n_seeds: int,
    output_path: str = "figures/e11_mb_sweep.pdf",
):
    """Plot mode weight evolution per β, and summary panel."""
    results_dir = Path(results_dir)

    # Determine layout
    n_betas = len(betas)
    ncols = min(3, n_betas)
    nrows = (n_betas + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows + 1, ncols, figsize=(5 * ncols, 4 * (nrows + 1)),
                              gridspec_kw={"hspace": 0.35, "wspace": 0.3})
    if nrows + 1 == 1:
        axes = axes.reshape(1, -1)

    # Mode colors
    mode_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # 3 modes for Müller-Brown
    mode_labels = ["Mode 0 (deep)", "Mode 1 (deep)", "Mode 2 (shallow)"]

    # Per-β panels: α_k vs. epoch
    summary_data = []

    for idx, beta in enumerate(betas):
        row, col = idx // ncols, idx % ncols
        ax = axes[row, col]

        final_alphas_all_seeds = []
        for seed in range(n_seeds):
            tracking_path = results_dir / f"beta_{beta}" / f"seed_{seed}" / "mode_tracking.jsonl"
            records = load_tracking(tracking_path)
            if not records:
                continue

            epochs = np.array([r["epoch"] for r in records])
            alphas = np.array([r["alpha"] for r in records])
            K = alphas.shape[1]

            for k in range(min(K, 3)):
                ax.plot(epochs, alphas[:, k], color=mode_colors[k],
                        alpha=0.4, linewidth=0.8)

            if records:
                final_alphas_all_seeds.append(records[-1]["alpha"])

        # Target weights as dashed
        if final_alphas_all_seeds:
            sample_record = None
            for seed in range(n_seeds):
                recs = load_tracking(results_dir / f"beta_{beta}" / f"seed_{seed}" / "mode_tracking.jsonl")
                if recs:
                    sample_record = recs[0]
                    break
            if sample_record:
                target_w = sample_record["target_w"]
                for k in range(min(len(target_w), 3)):
                    ax.axhline(y=target_w[k], color=mode_colors[k],
                               linestyle="--", alpha=0.6, linewidth=1)

        ax.set_title(f"$\\beta$ = {beta}", fontsize=12)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel("$\\alpha_k$", fontsize=11)
        if row == nrows - 1 or idx >= n_betas - ncols:
            ax.set_xlabel("Epoch", fontsize=11)

        # Summary stats
        if final_alphas_all_seeds:
            fa = np.array(final_alphas_all_seeds)
            summary_data.append({
                "beta": beta,
                "mean_alpha": fa.mean(axis=0).tolist(),
                "std_alpha": fa.std(axis=0).tolist(),
            })

    # Hide unused subplot cells
    for idx in range(n_betas, nrows * ncols):
        row, col = idx // ncols, idx % ncols
        axes[row, col].set_visible(False)

    # Summary panel: final α_k vs. β
    ax_summary = axes[nrows, 0] if ncols > 0 else axes[nrows]
    # Span full width
    for c in range(1, ncols):
        axes[nrows, c].set_visible(False)
    ax_summary = fig.add_subplot(nrows + 1, 1, nrows + 1)

    if summary_data:
        betas_arr = [s["beta"] for s in summary_data]
        for k in range(3):
            means = [s["mean_alpha"][k] if k < len(s["mean_alpha"]) else 0
                     for s in summary_data]
            stds = [s["std_alpha"][k] if k < len(s["std_alpha"]) else 0
                    for s in summary_data]
            ax_summary.errorbar(betas_arr, means, yerr=stds,
                                fmt="o-", color=mode_colors[k],
                                linewidth=1.5, markersize=5, capsize=3,
                                label=mode_labels[k])

    ax_summary.set_xlabel("Inverse temperature $\\beta$", fontsize=12)
    ax_summary.set_ylabel("Final $\\alpha_k$", fontsize=12)
    ax_summary.set_title("Summary: Final Mode Weights vs. Temperature", fontsize=13)
    ax_summary.set_xscale("log")
    ax_summary.legend(fontsize=9)
    ax_summary.grid(True, alpha=0.3)

    # Hide the original bottom-row axes since we replaced them
    for c in range(ncols):
        axes[nrows, c].set_visible(False)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"Saved: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="E11: Müller-Brown Temperature Sweep")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run sweep")
    run_parser.add_argument("--betas", type=float, nargs="+",
                            default=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
    run_parser.add_argument("--n-seeds", type=int, default=5)
    run_parser.add_argument("--num-epochs", type=int, default=1000)
    run_parser.add_argument("extra_args", nargs="*", default=[])

    plot_parser = subparsers.add_parser("plot", help="Plot results")
    plot_parser.add_argument("--results-dir", default="results/e11_sweep")
    plot_parser.add_argument("--betas", type=float, nargs="+",
                             default=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
    plot_parser.add_argument("--n-seeds", type=int, default=5)
    plot_parser.add_argument("--output", default="figures/e11_mb_sweep.pdf")

    args = parser.parse_args()

    if args.command == "run":
        launch_sweep(args.betas, args.n_seeds, args.num_epochs, args.extra_args)
    elif args.command == "plot":
        plot_sweep(args.results_dir, args.betas, args.n_seeds, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
