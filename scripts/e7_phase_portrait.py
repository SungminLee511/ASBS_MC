#!/usr/bin/env python
"""
E7: Phase Portrait of Mode Weights

- B1 (K=2): Pitchfork bifurcation — 30 seeds of α_1 vs epoch
- B7 (K=3): Ternary simplex flow — trajectories on the 2-simplex

Usage:
    # Step 1: Launch training runs (or use the launcher below)
    python scripts/e7_phase_portrait.py launch --benchmark b1 --seeds 30
    python scripts/e7_phase_portrait.py launch --benchmark b7 --seeds 20

    # Step 2: Plot results after training completes
    python scripts/e7_phase_portrait.py plot --benchmark b1 --results-dir results
    python scripts/e7_phase_portrait.py plot --benchmark b7 --results-dir results
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


def launch_runs(benchmark: str, n_seeds: int, num_epochs: int, extra_args: list):
    """Launch training runs for multiple seeds."""
    if benchmark == "b1":
        experiment = "b1_asbs"
    elif benchmark == "b7":
        experiment = "b7_asbs"
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    procs = []
    for seed in range(n_seeds):
        cmd = [
            "python", "train.py",
            f"experiment={experiment}",
            f"seed={seed}",
            f"num_epochs={num_epochs}",
            f"hydra.run.dir=results/{experiment}/seed_{seed}",
        ] + extra_args

        print(f"[seed={seed}] Launching: {' '.join(cmd)}")
        log_path = Path(f"results/{experiment}/seed_{seed}")
        log_path.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path / "stdout.log", "w")

        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        procs.append((seed, proc, log_file))

    print(f"\nLaunched {len(procs)} runs. Waiting for completion...")
    for seed, proc, log_file in procs:
        proc.wait()
        log_file.close()
        status = "OK" if proc.returncode == 0 else f"FAILED (rc={proc.returncode})"
        print(f"  [seed={seed}] {status}")


def plot_b1_pitchfork(results_dir: str, output_path: str = "figures/e7_b1_pitchfork.pdf"):
    """Plot pitchfork bifurcation diagram for B1 (K=2)."""
    results_dir = Path(results_dir) / "b1_asbs"
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    seed_dirs = sorted(results_dir.glob("seed_*"))
    if not seed_dirs:
        print("No seed directories found.")
        return

    target_w1 = None
    for seed_dir in seed_dirs:
        tracking = load_tracking(seed_dir / "mode_tracking.jsonl")
        if not tracking:
            continue

        epochs = [r["epoch"] for r in tracking]
        alpha1 = [r["alpha"][0] for r in tracking]
        if target_w1 is None:
            target_w1 = tracking[0]["target_w"][0]

        ax.plot(epochs, alpha1, alpha=0.4, linewidth=0.8, color="steelblue")

    if target_w1 is not None:
        ax.axhline(y=target_w1, color="red", linestyle="--", linewidth=1.5,
                    label=f"Target $w_1$ = {target_w1:.2f}")

    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("$\\alpha_1$ (mode 1 weight)", fontsize=13)
    ax.set_title(f"E7: Pitchfork Bifurcation — B1 ({len(seed_dirs)} seeds)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_b7_ternary(results_dir: str, output_path: str = "figures/e7_b7_ternary.pdf"):
    """Plot ternary simplex flow for B7 (K=3)."""
    results_dir = Path(results_dir) / "b7_asbs"
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    # Draw simplex triangle
    # Corners: mode 1 at top, mode 2 at bottom-left, mode 3 at bottom-right
    # Barycentric → Cartesian: (α1, α2, α3) → (x, y)
    def bary_to_cart(a1, a2, a3):
        x = 0.5 * (2 * a3 + a1)
        y = (np.sqrt(3) / 2) * a1
        return x, y

    # Triangle edges
    corners = np.array([
        bary_to_cart(1, 0, 0),
        bary_to_cart(0, 1, 0),
        bary_to_cart(0, 0, 1),
    ])
    triangle = plt.Polygon(corners, fill=False, edgecolor="black", linewidth=1.5)
    ax.add_patch(triangle)

    # Labels
    ax.text(corners[0][0], corners[0][1] + 0.03, "Mode 1\n(deep)", ha="center", fontsize=10)
    ax.text(corners[1][0] - 0.03, corners[1][1] - 0.03, "Mode 2\n(deep)", ha="center", fontsize=10)
    ax.text(corners[2][0] + 0.03, corners[2][1] - 0.03, "Mode 3\n(meta)", ha="center", fontsize=10)

    seed_dirs = sorted(results_dir.glob("seed_*"))
    target_w = None

    cmap = plt.cm.viridis
    for seed_dir in seed_dirs:
        tracking = load_tracking(seed_dir / "mode_tracking.jsonl")
        if not tracking:
            continue

        alphas = np.array([r["alpha"] for r in tracking])
        if target_w is None:
            target_w = np.array(tracking[0]["target_w"])

        xs, ys = [], []
        for a in alphas:
            x, y = bary_to_cart(a[0], a[1], a[2])
            xs.append(x)
            ys.append(y)

        # Color by epoch
        epochs = np.array([r["epoch"] for r in tracking])
        if len(xs) > 1:
            ax.scatter(xs, ys, c=epochs, cmap=cmap, s=8, alpha=0.6, zorder=2)
            ax.plot(xs, ys, alpha=0.3, linewidth=0.5, color="gray", zorder=1)

    # Mark target point
    if target_w is not None:
        tx, ty = bary_to_cart(target_w[0], target_w[1], target_w[2])
        ax.scatter([tx], [ty], color="red", s=100, marker="*", zorder=5,
                   label=f"Target $w$ = ({target_w[0]:.2f}, {target_w[1]:.2f}, {target_w[2]:.2f})")
        ax.legend(fontsize=10, loc="upper right")

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.15, 1.0)
    ax.set_aspect("equal")
    ax.set_title(f"E7: Simplex Flow — B7 ({len(seed_dirs)} seeds)", fontsize=14)
    ax.axis("off")

    # Add colorbar for epoch
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Epoch", fontsize=11)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"Saved: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="E7: Phase Portrait of Mode Weights")
    subparsers = parser.add_subparsers(dest="command")

    # Launch subcommand
    launch_parser = subparsers.add_parser("launch", help="Launch training runs")
    launch_parser.add_argument("--benchmark", required=True, choices=["b1", "b7"])
    launch_parser.add_argument("--seeds", type=int, default=30)
    launch_parser.add_argument("--num-epochs", type=int, default=1000)
    launch_parser.add_argument("extra_args", nargs="*", default=[])

    # Plot subcommand
    plot_parser = subparsers.add_parser("plot", help="Plot results")
    plot_parser.add_argument("--benchmark", required=True, choices=["b1", "b7"])
    plot_parser.add_argument("--results-dir", default="results")
    plot_parser.add_argument("--output", default=None)

    args = parser.parse_args()

    if args.command == "launch":
        launch_runs(args.benchmark, args.seeds, args.num_epochs, args.extra_args)

    elif args.command == "plot":
        if args.benchmark == "b1":
            out = args.output or "figures/e7_b1_pitchfork.pdf"
            plot_b1_pitchfork(args.results_dir, out)
        elif args.benchmark == "b7":
            out = args.output or "figures/e7_b7_ternary.pdf"
            plot_b7_ternary(args.results_dir, out)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
