#!/usr/bin/env python
"""
E8: J₁₁^η Estimation and Threshold Verification

For symmetric 2-mode Gaussian (w1=w2=0.5), sweep mode separation d and:
  1. Train ASBS to near-balance
  2. Perturb α_1 by ε (via reweighting replay buffer)
  3. Run one AM epoch, measure new α_1'
  4. Estimate J₁₁^η ≈ (α_1' - 0.5) / ε

Also runs full training for each d and records collapse probability.

Usage:
    # Run the full sweep
    python scripts/e8_jacobian_threshold.py run \\
        --separations 3 5 7 9 12 15 20 \\
        --n-seeds 10 --num-epochs 500

    # Plot J₁₁ vs. ρ_sep and collapse probability
    python scripts/e8_jacobian_threshold.py plot \\
        --results-dir results/e8_sweep \\
        --output figures/e8_threshold.pdf
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


def launch_sweep(separations, n_seeds, num_epochs, extra_args):
    """Launch training runs for each separation × seed."""
    results_base = Path("results/e8_sweep")

    procs = []
    for d in separations:
        for seed in range(n_seeds):
            run_dir = results_base / f"d_{d}" / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)

            # Symmetric 2-mode: w1=0.5, centers at (-d/2, 0) and (d/2, 0)
            mu1_x = -d / 2
            mu2_x = d / 2

            cmd = [
                "python", "train.py",
                "experiment=b1_asbs",
                f"seed={seed}",
                f"num_epochs={num_epochs}",
                "w1=0.5",  # symmetric
                f"energy.mu1=[{mu1_x}, 0.0]",
                f"energy.mu2=[{mu2_x}, 0.0]",
                f"hydra.run.dir={run_dir}",
            ] + extra_args

            log_file = open(run_dir / "stdout.log", "w")
            proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
            procs.append((d, seed, proc, log_file))

    print(f"Launched {len(procs)} runs across {len(separations)} separations × {n_seeds} seeds")
    print("Waiting for completion...")

    for d, seed, proc, log_file in procs:
        proc.wait()
        log_file.close()
        status = "OK" if proc.returncode == 0 else f"FAILED"
        if proc.returncode != 0:
            print(f"  [d={d}, seed={seed}] {status}")

    print("All runs complete.")


def analyze_collapse(results_dir, separations, n_seeds, collapse_threshold=0.3):
    """Analyze collapse probability for each separation."""
    results_dir = Path(results_dir)
    results = []

    for d in separations:
        collapse_count = 0
        final_kl_list = []
        total_runs = 0

        for seed in range(n_seeds):
            tracking_path = results_dir / f"d_{d}" / f"seed_{seed}" / "mode_tracking.jsonl"
            records = load_tracking(tracking_path)
            if not records:
                continue

            total_runs += 1
            last = records[-1]
            alpha1 = last["alpha"][0]

            # Collapse: |α_1 - 0.5| > threshold
            if abs(alpha1 - 0.5) > collapse_threshold:
                collapse_count += 1

            final_kl_list.append(last["kl"])

        if total_runs > 0:
            results.append({
                "d": d,
                "collapse_prob": collapse_count / total_runs,
                "mean_kl": np.mean(final_kl_list),
                "std_kl": np.std(final_kl_list),
                "n_runs": total_runs,
            })
            print(f"  d={d:5.1f}: collapse={collapse_count}/{total_runs} "
                  f"({100*collapse_count/total_runs:.0f}%), "
                  f"KL={np.mean(final_kl_list):.4f}±{np.std(final_kl_list):.4f}")

    return results


def estimate_jacobian(results_dir, separations, n_seeds):
    """Estimate J₁₁^η from the mode weight dynamics.

    For each run at near-balance, compute the effective amplification factor:
    J₁₁ ≈ Δα_1' / Δα_1 between consecutive tracking points.
    """
    results_dir = Path(results_dir)
    j11_estimates = {}

    for d in separations:
        j11_list = []
        for seed in range(n_seeds):
            tracking_path = results_dir / f"d_{d}" / f"seed_{seed}" / "mode_tracking.jsonl"
            records = load_tracking(tracking_path)
            if len(records) < 3:
                continue

            # Use early epochs where α ≈ w for the best estimate
            for i in range(1, min(len(records) - 1, 5)):
                a_prev = records[i]["alpha"][0] - 0.5
                a_next = records[i + 1]["alpha"][0] - 0.5

                if abs(a_prev) > 0.01:  # need some signal
                    j11_est = a_next / a_prev
                    if 0 < j11_est < 5:  # sanity bound
                        j11_list.append(j11_est)

        if j11_list:
            j11_estimates[d] = {
                "mean": np.mean(j11_list),
                "std": np.std(j11_list) / np.sqrt(len(j11_list)),
                "n": len(j11_list),
            }

    return j11_estimates


def plot_results(
    results_dir: str,
    separations: list,
    n_seeds: int,
    output_path: str = "figures/e8_threshold.pdf",
    sigma_sde: float = 3.0,  # σ_max from b1_asbs config
):
    """Plot J₁₁ vs. ρ_sep and collapse probability."""
    results_dir = Path(results_dir)

    print("Analyzing collapse...")
    collapse_results = analyze_collapse(results_dir, separations, n_seeds)

    print("Estimating J₁₁...")
    j11_estimates = estimate_jacobian(results_dir, separations, n_seeds)

    if not collapse_results:
        print("No results to plot.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True,
                                     gridspec_kw={"hspace": 0.1})

    # ρ_sep = d / σ(1)
    ds = np.array([r["d"] for r in collapse_results])
    rho_seps = ds / sigma_sde

    # Top panel: J₁₁^η estimate
    j11_ds = sorted(j11_estimates.keys())
    j11_rhos = np.array(j11_ds) / sigma_sde
    j11_means = [j11_estimates[d]["mean"] for d in j11_ds]
    j11_errs = [j11_estimates[d]["std"] for d in j11_ds]

    ax1.errorbar(j11_rhos, j11_means, yerr=j11_errs,
                 fmt="o-", color="navy", linewidth=1.5, markersize=6,
                 capsize=3, label="$\\hat{J}_{11}^\\eta$")
    ax1.axhline(y=0.5, color="red", linestyle="--", linewidth=1.5,
                label="Instability threshold (1/2)")
    ax1.set_ylabel("$\\hat{J}_{11}^\\eta$", fontsize=13)
    ax1.set_title("E8: Jacobian Threshold Verification", fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Bottom panel: Collapse probability
    collapse_probs = [r["collapse_prob"] for r in collapse_results]
    ax2.plot(rho_seps, collapse_probs, "s-", color="darkred",
             linewidth=1.5, markersize=7, label="Collapse probability")
    ax2.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    ax2.set_xlabel("$\\rho_{\\mathrm{sep}}$ = $d / \\sigma(1)$", fontsize=13)
    ax2.set_ylabel("Collapse probability", fontsize=13)
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"Saved: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="E8: J₁₁ Threshold Verification")
    subparsers = parser.add_subparsers(dest="command")

    # Run subcommand
    run_parser = subparsers.add_parser("run", help="Run the sweep")
    run_parser.add_argument("--separations", type=float, nargs="+",
                            default=[3, 5, 7, 9, 12, 15, 20])
    run_parser.add_argument("--n-seeds", type=int, default=10)
    run_parser.add_argument("--num-epochs", type=int, default=500)
    run_parser.add_argument("extra_args", nargs="*", default=[])

    # Plot subcommand
    plot_parser = subparsers.add_parser("plot", help="Plot results")
    plot_parser.add_argument("--results-dir", default="results/e8_sweep")
    plot_parser.add_argument("--separations", type=float, nargs="+",
                             default=[3, 5, 7, 9, 12, 15, 20])
    plot_parser.add_argument("--n-seeds", type=int, default=10)
    plot_parser.add_argument("--output", default="figures/e8_threshold.pdf")

    args = parser.parse_args()

    if args.command == "run":
        launch_sweep(args.separations, args.n_seeds, args.num_epochs, args.extra_args)
    elif args.command == "plot":
        plot_results(args.results_dir, args.separations, args.n_seeds, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
