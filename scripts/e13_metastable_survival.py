#!/usr/bin/env python
"""
E13: Metastable State Survival (Critical Experiment)

Sweep the depth ratio of the metastable well in B7 (three-well):
  kappa3 ∈ {4, 6, 8, 10, 12, 14, 16, 18, 20}
(kappa1 = 20 is fixed for the two deep wells)

For each depth, run multiple seeds and measure:
  - Metastable mode weight α_3 at convergence
  - Whether metastable state survives (α_3 > 0.5 * w_3)
  - Survival probability across seeds

Usage:
    # Run sweep
    python scripts/e13_metastable_survival.py run \\
        --kappa3-values 4 6 8 10 12 14 16 18 20 \\
        --n-seeds 20 --num-epochs 1000

    # Plot phase diagram
    python scripts/e13_metastable_survival.py plot \\
        --results-dir results/e13_sweep \\
        --output figures/e13_metastable.pdf
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


def launch_sweep(kappa3_values, n_seeds, num_epochs, extra_args):
    """Launch training runs for each κ3 × seed."""
    results_base = Path("results/e13_sweep")

    procs = []
    for kappa3 in kappa3_values:
        for seed in range(n_seeds):
            run_dir = results_base / f"kappa3_{kappa3}" / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                "python", "train.py",
                "experiment=b7_asbs",
                f"seed={seed}",
                f"num_epochs={num_epochs}",
                f"kappa3={kappa3}",
                f"hydra.run.dir={run_dir}",
            ] + extra_args

            log_file = open(run_dir / "stdout.log", "w")
            proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
            procs.append((kappa3, seed, proc, log_file))

    print(f"Launched {len(procs)} runs across {len(kappa3_values)} κ3 values × {n_seeds} seeds")
    print("Waiting for completion...")

    for kappa3, seed, proc, log_file in procs:
        proc.wait()
        log_file.close()
        if proc.returncode != 0:
            print(f"  [κ3={kappa3}, seed={seed}] FAILED")

    print("All runs complete.")


def analyze_survival(results_dir, kappa3_values, n_seeds, survival_threshold=0.5):
    """Analyze metastable state survival for each κ3."""
    results_dir = Path(results_dir)
    results = []

    for kappa3 in kappa3_values:
        survival_count = 0
        alpha3_list = []
        ratio_list = []
        total_runs = 0

        for seed in range(n_seeds):
            tracking_path = results_dir / f"kappa3_{kappa3}" / f"seed_{seed}" / "mode_tracking.jsonl"
            records = load_tracking(tracking_path)
            if not records:
                continue

            total_runs += 1
            last = records[-1]

            # Mode 2 (index 2) is the metastable state at (0.3, 1.5)
            alpha_meta = last["alpha"][2]
            w_meta = last["target_w"][2]
            alpha3_list.append(alpha_meta)

            ratio = alpha_meta / max(w_meta, 1e-10)
            ratio_list.append(ratio)

            if ratio > survival_threshold:
                survival_count += 1

        if total_runs > 0:
            entry = {
                "kappa3": kappa3,
                "survival_prob": survival_count / total_runs,
                "mean_alpha3": np.mean(alpha3_list),
                "std_alpha3": np.std(alpha3_list),
                "mean_ratio": np.mean(ratio_list),
                "std_ratio": np.std(ratio_list),
                "n_runs": total_runs,
            }
            results.append(entry)
            print(f"  κ3={kappa3:5.1f}: survival={survival_count}/{total_runs} "
                  f"({100*survival_count/total_runs:.0f}%), "
                  f"α₃/w₃={np.mean(ratio_list):.3f}±{np.std(ratio_list):.3f}")

    return results


def plot_results(
    results_dir: str,
    kappa3_values: list,
    n_seeds: int,
    output_path: str = "figures/e13_metastable.pdf",
):
    """Plot survival probability and α₃/w₃ vs. depth ratio."""
    print("Analyzing metastable state survival...")
    results = analyze_survival(results_dir, kappa3_values, n_seeds)

    if not results:
        print("No results to plot.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True,
                                     gridspec_kw={"hspace": 0.1})

    kappas = [r["kappa3"] for r in results]
    depth_ratios = [k3 / 20.0 for k3 in kappas]  # κ3/κ1

    # Top: Survival probability
    survival_probs = [r["survival_prob"] for r in results]
    ax1.plot(depth_ratios, survival_probs, "o-", color="navy",
             linewidth=2, markersize=8)
    ax1.fill_between(depth_ratios, survival_probs, alpha=0.1, color="navy")
    ax1.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    ax1.set_ylabel("Survival probability", fontsize=13)
    ax1.set_title("E13: Metastable State Survival vs. Well Depth", fontsize=14)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)

    # Annotations
    ax1.text(0.05, 0.95, "Metastable\nstate dies",
             transform=ax1.transAxes, fontsize=10, va="top",
             color="darkred", style="italic")
    ax1.text(0.9, 0.95, "Metastable\nstate survives",
             transform=ax1.transAxes, fontsize=10, va="top", ha="right",
             color="darkgreen", style="italic")

    # Bottom: α₃/w₃ ratio
    mean_ratios = [r["mean_ratio"] for r in results]
    std_ratios = [r["std_ratio"] for r in results]
    ax2.errorbar(depth_ratios, mean_ratios, yerr=std_ratios,
                 fmt="s-", color="darkred", linewidth=1.5, markersize=6, capsize=3)
    ax2.axhline(y=1.0, color="green", linestyle="--", linewidth=1,
                label="$\\alpha_3 / w_3 = 1$ (healthy)")
    ax2.axhline(y=0.5, color="orange", linestyle="--", linewidth=1,
                label="Survival threshold (0.5)")
    ax2.set_xlabel("Depth ratio $\\kappa_3 / \\kappa_1$", fontsize=13)
    ax2.set_ylabel("$\\alpha_3 / w_3$", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.1, max(max(mean_ratios) + 0.5, 2.0))

    # Add secondary x-axis for absolute κ3
    ax_top = ax1.twiny()
    ax_top.set_xlim(ax1.get_xlim())
    tick_locs = depth_ratios
    ax_top.set_xticks(tick_locs)
    ax_top.set_xticklabels([f"{k3:.0f}" for k3 in kappas], fontsize=9)
    ax_top.set_xlabel("$\\kappa_3$", fontsize=11)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"Saved: {output_path}")
    plt.close(fig)

    # Also plot per-κ3 mode weight evolution
    plot_per_kappa_evolution(results_dir, kappa3_values, n_seeds,
                             output_path.replace(".pdf", "_evolution.pdf"))


def plot_per_kappa_evolution(results_dir, kappa3_values, n_seeds, output_path):
    """Plot mode weight evolution panels for selected κ3 values."""
    results_dir = Path(results_dir)

    # Select a subset of κ3 values to show
    if len(kappa3_values) > 6:
        selected = kappa3_values[::2]  # every other
    else:
        selected = kappa3_values

    ncols = min(3, len(selected))
    nrows = (len(selected) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows),
                              sharex=True, sharey=True,
                              gridspec_kw={"hspace": 0.3, "wspace": 0.15})
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)

    mode_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    mode_labels = ["Deep well 1", "Deep well 2", "Metastable"]

    for idx, kappa3 in enumerate(selected):
        row, col = idx // ncols, idx % ncols
        ax = axes[row, col]

        for seed in range(n_seeds):
            tracking_path = results_dir / f"kappa3_{kappa3}" / f"seed_{seed}" / "mode_tracking.jsonl"
            records = load_tracking(tracking_path)
            if not records:
                continue

            epochs = np.array([r["epoch"] for r in records])
            alphas = np.array([r["alpha"] for r in records])

            for k in range(min(alphas.shape[1], 3)):
                ax.plot(epochs, alphas[:, k], color=mode_colors[k],
                        alpha=0.3, linewidth=0.6)

        ax.set_title(f"$\\kappa_3$ = {kappa3:.0f}  (ratio={kappa3/20:.2f})", fontsize=11)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel("$\\alpha_k$", fontsize=10)
        if row == nrows - 1:
            ax.set_xlabel("Epoch", fontsize=10)

    # Add legend to first panel
    for k in range(3):
        axes[0, 0].plot([], [], color=mode_colors[k], linewidth=2, label=mode_labels[k])
    axes[0, 0].legend(fontsize=8, loc="upper right")

    # Hide unused
    for idx in range(len(selected), nrows * ncols):
        row, col = idx // ncols, idx % ncols
        axes[row, col].set_visible(False)

    fig.suptitle("E13: Mode Weight Evolution vs. Metastable Well Depth", fontsize=13, y=1.02)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"Saved: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="E13: Metastable State Survival")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run sweep")
    run_parser.add_argument("--kappa3-values", type=float, nargs="+",
                            default=[4, 6, 8, 10, 12, 14, 16, 18, 20])
    run_parser.add_argument("--n-seeds", type=int, default=20)
    run_parser.add_argument("--num-epochs", type=int, default=1000)
    run_parser.add_argument("extra_args", nargs="*", default=[])

    plot_parser = subparsers.add_parser("plot", help="Plot results")
    plot_parser.add_argument("--results-dir", default="results/e13_sweep")
    plot_parser.add_argument("--kappa3-values", type=float, nargs="+",
                             default=[4, 6, 8, 10, 12, 14, 16, 18, 20])
    plot_parser.add_argument("--n-seeds", type=int, default=20)
    plot_parser.add_argument("--output", default="figures/e13_metastable.pdf")

    args = parser.parse_args()

    if args.command == "run":
        launch_sweep(args.kappa3_values, args.n_seeds, args.num_epochs, args.extra_args)
    elif args.command == "plot":
        plot_results(args.results_dir, args.kappa3_values, args.n_seeds, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
