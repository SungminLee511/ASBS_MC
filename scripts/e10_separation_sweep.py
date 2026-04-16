#!/usr/bin/env python
"""
E10: Separation Sweep

Sweep mode separation for:
  - Symmetric 2-mode Gaussian (B1): d ∈ {2,3,4,5,6,8,10,12,15,20}
  - Heterogeneous 4-mode (B5): center_scale ∈ {2,3,4,5,7,10}

Measures: collapse indicator, time to collapse, final KL, which mode dies.

Usage:
    # Run B1 sweep
    python scripts/e10_separation_sweep.py run --benchmark b1 \\
        --separations 2 3 4 5 6 8 10 12 15 20 --n-seeds 10

    # Run B5 sweep
    python scripts/e10_separation_sweep.py run --benchmark b5 \\
        --center-scales 2 3 4 5 7 10 --n-seeds 10

    # Plot
    python scripts/e10_separation_sweep.py plot --benchmark b1 \\
        --results-dir results/e10_b1_sweep
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


def launch_b1_sweep(separations, n_seeds, num_epochs, extra_args):
    """Sweep mode separation d for symmetric 2-mode Gaussian."""
    results_base = Path("results/e10_b1_sweep")
    procs = []

    for d in separations:
        for seed in range(n_seeds):
            run_dir = results_base / f"d_{d}" / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)

            mu1_x = -d / 2
            mu2_x = d / 2
            cmd = [
                "python", "train.py",
                "experiment=b1_asbs",
                f"seed={seed}",
                f"num_epochs={num_epochs}",
                "w1=0.5",
                f"energy.mu1=[{mu1_x}, 0.0]",
                f"energy.mu2=[{mu2_x}, 0.0]",
                f"hydra.run.dir={run_dir}",
            ] + extra_args

            log_file = open(run_dir / "stdout.log", "w")
            proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
            procs.append((d, seed, proc, log_file))

    print(f"Launched {len(procs)} B1 runs")
    for d, seed, proc, lf in procs:
        proc.wait()
        lf.close()
    print("Done.")


def launch_b5_sweep(center_scales, n_seeds, num_epochs, extra_args):
    """Sweep center_scale for heterogeneous 4-mode mixture."""
    results_base = Path("results/e10_b5_sweep")
    procs = []

    for cs in center_scales:
        for seed in range(n_seeds):
            run_dir = results_base / f"cs_{cs}" / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                "python", "train.py",
                "experiment=b5_asbs",
                f"seed={seed}",
                f"num_epochs={num_epochs}",
                f"center_scale={cs}",
                f"scale={cs + 1}",
                f"hydra.run.dir={run_dir}",
            ] + extra_args

            log_file = open(run_dir / "stdout.log", "w")
            proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
            procs.append((cs, seed, proc, log_file))

    print(f"Launched {len(procs)} B5 runs")
    for cs, seed, proc, lf in procs:
        proc.wait()
        lf.close()
    print("Done.")


def analyze_sweep(results_dir, param_values, param_name, n_seeds,
                  collapse_rel_threshold=0.1):
    """Analyze sweep results: collapse probability, time to collapse, final KL."""
    results_dir = Path(results_dir)
    results = []

    for pval in param_values:
        collapse_count = 0
        collapse_times = []
        final_kl_list = []
        first_dead_mode = []
        total_runs = 0

        for seed in range(n_seeds):
            tracking_path = results_dir / f"{param_name}_{pval}" / f"seed_{seed}" / "mode_tracking.jsonl"
            records = load_tracking(tracking_path)
            if not records:
                continue

            total_runs += 1
            target_w = np.array(records[0]["target_w"])
            K = len(target_w)

            # Check collapse: any mode drops below threshold * w_k
            collapsed = False
            collapse_epoch = None
            dead_mode = None
            for r in records:
                alpha = np.array(r["alpha"])
                for k in range(K):
                    if target_w[k] > 0.01 and alpha[k] < collapse_rel_threshold * target_w[k]:
                        if not collapsed:
                            collapsed = True
                            collapse_epoch = r["epoch"]
                            dead_mode = k
                        break

            if collapsed:
                collapse_count += 1
                collapse_times.append(collapse_epoch)
                first_dead_mode.append(dead_mode)

            final_kl_list.append(records[-1]["kl"])

        if total_runs > 0:
            entry = {
                "param": pval,
                "collapse_prob": collapse_count / total_runs,
                "mean_time_to_collapse": np.mean(collapse_times) if collapse_times else None,
                "mean_kl": np.mean(final_kl_list),
                "std_kl": np.std(final_kl_list),
                "first_dead_mode_counts": {},
                "n_runs": total_runs,
            }
            # Count which mode dies first
            for m in first_dead_mode:
                entry["first_dead_mode_counts"][str(m)] = \
                    entry["first_dead_mode_counts"].get(str(m), 0) + 1

            results.append(entry)
            print(f"  {param_name}={pval}: collapse={collapse_count}/{total_runs}, "
                  f"KL={np.mean(final_kl_list):.4f}")

    return results


def plot_b1_sweep(results_dir, separations, n_seeds,
                  output_path="figures/e10_b1_sweep.pdf"):
    """Plot collapse probability, time to collapse, final KL vs. separation."""
    results = analyze_sweep(results_dir, separations, "d", n_seeds)
    if not results:
        print("No results to plot.")
        return

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10), sharex=True,
                                          gridspec_kw={"hspace": 0.12})
    params = [r["param"] for r in results]

    # Panel 1: Collapse probability
    ax1.plot(params, [r["collapse_prob"] for r in results],
             "o-", color="navy", linewidth=2, markersize=7)
    ax1.set_ylabel("Collapse probability", fontsize=12)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_title("E10: Separation Sweep — B1 (Symmetric 2-Mode)", fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Time to collapse
    times = []
    time_params = []
    for r in results:
        if r["mean_time_to_collapse"] is not None:
            times.append(r["mean_time_to_collapse"])
            time_params.append(r["param"])
    if times:
        ax2.plot(time_params, times, "s-", color="darkred", linewidth=1.5, markersize=6)
    ax2.set_ylabel("Mean time to collapse (epoch)", fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Final KL
    ax3.errorbar(params,
                 [r["mean_kl"] for r in results],
                 yerr=[r["std_kl"] for r in results],
                 fmt="D-", color="forestgreen", linewidth=1.5, markersize=5, capsize=3)
    ax3.set_xlabel("Mode separation $d$", fontsize=12)
    ax3.set_ylabel("Final KL($\\alpha$ || $w$)", fontsize=12)
    ax3.grid(True, alpha=0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_b5_sweep(results_dir, center_scales, n_seeds,
                  output_path="figures/e10_b5_sweep.pdf"):
    """Plot B5 sweep with which-mode-dies-first analysis."""
    results = analyze_sweep(results_dir, center_scales, "cs", n_seeds)
    if not results:
        print("No results to plot.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True,
                                     gridspec_kw={"hspace": 0.15})
    params = [r["param"] for r in results]

    ax1.plot(params, [r["collapse_prob"] for r in results],
             "o-", color="navy", linewidth=2, markersize=7)
    ax1.set_ylabel("Collapse probability", fontsize=12)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_title("E10: Separation Sweep — B5 (Heterogeneous 4-Mode)", fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Which mode dies first — stacked bar
    mode_names = ["Tight spike", "Broad cloud", "Ellipse", "Unit Gaussian"]
    mode_colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

    x_pos = np.arange(len(params))
    bottom = np.zeros(len(params))
    for k in range(4):
        counts = []
        for r in results:
            counts.append(r["first_dead_mode_counts"].get(str(k), 0))
        totals = [r["n_runs"] for r in results]
        fracs = [c / max(t, 1) for c, t in zip(counts, totals)]
        ax2.bar(x_pos, fracs, bottom=bottom, color=mode_colors[k],
                label=mode_names[k], alpha=0.8)
        bottom += np.array(fracs)

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([str(p) for p in params])
    ax2.set_xlabel("Center scale", fontsize=12)
    ax2.set_ylabel("Fraction (first mode to die)", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"Saved: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="E10: Separation Sweep")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run sweep")
    run_parser.add_argument("--benchmark", required=True, choices=["b1", "b5"])
    run_parser.add_argument("--separations", type=float, nargs="+",
                            default=[2, 3, 4, 5, 6, 8, 10, 12, 15, 20])
    run_parser.add_argument("--center-scales", type=float, nargs="+",
                            default=[2, 3, 4, 5, 7, 10])
    run_parser.add_argument("--n-seeds", type=int, default=10)
    run_parser.add_argument("--num-epochs", type=int, default=500)
    run_parser.add_argument("extra_args", nargs="*", default=[])

    plot_parser = subparsers.add_parser("plot", help="Plot results")
    plot_parser.add_argument("--benchmark", required=True, choices=["b1", "b5"])
    plot_parser.add_argument("--results-dir", default=None)
    plot_parser.add_argument("--separations", type=float, nargs="+",
                             default=[2, 3, 4, 5, 6, 8, 10, 12, 15, 20])
    plot_parser.add_argument("--center-scales", type=float, nargs="+",
                             default=[2, 3, 4, 5, 7, 10])
    plot_parser.add_argument("--n-seeds", type=int, default=10)
    plot_parser.add_argument("--output", default=None)

    args = parser.parse_args()

    if args.command == "run":
        if args.benchmark == "b1":
            launch_b1_sweep(args.separations, args.n_seeds,
                           args.num_epochs, args.extra_args)
        else:
            launch_b5_sweep(args.center_scales, args.n_seeds,
                           args.num_epochs, args.extra_args)

    elif args.command == "plot":
        if args.benchmark == "b1":
            rd = args.results_dir or "results/e10_b1_sweep"
            out = args.output or "figures/e10_b1_sweep.pdf"
            plot_b1_sweep(rd, args.separations, args.n_seeds, out)
        else:
            rd = args.results_dir or "results/e10_b5_sweep"
            out = args.output or "figures/e10_b5_sweep.pdf"
            plot_b5_sweep(rd, args.center_scales, args.n_seeds, out)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
