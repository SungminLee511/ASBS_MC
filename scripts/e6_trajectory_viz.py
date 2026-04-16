#!/usr/bin/env python
"""
E6: Particle Trajectory Visualization

Forward-simulate N=200 particles from saved checkpoints, save full
trajectories, color by terminal mode assignment. Produces spaghetti
plots at early/mid/late training epochs.

Usage:
    python scripts/e6_trajectory_viz.py \\
        --run-dir results/b7_asbs/seed_0 \\
        --epochs 20 200 1000 \\
        --output figures/e6_b7_trajectories.pdf
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
    generate_trajectories,
    assign_modes_nearest,
    find_checkpoints,
)


def plot_trajectories(
    run_dir: str,
    epochs: list,
    n_particles: int = 200,
    output_path: str = "figures/e6_trajectories.pdf",
    device: str = "cuda",
):
    """Plot spaghetti trajectories at specified epochs."""
    run_dir = Path(run_dir)
    ckpt_dir = run_dir / "checkpoints"

    # Find available checkpoints closest to requested epochs
    available_ckpts = {}
    for ckpt_path in find_checkpoints(run_dir):
        ep = int(ckpt_path.stem.split("_")[-1])
        available_ckpts[ep] = ckpt_path

    if not available_ckpts:
        print(f"No checkpoints found in {ckpt_dir}")
        return

    # Find closest checkpoint for each requested epoch
    matched = []
    avail_epochs = sorted(available_ckpts.keys())
    for target_ep in epochs:
        closest = min(avail_epochs, key=lambda e: abs(e - target_ep))
        matched.append((target_ep, closest, available_ckpts[closest]))

    ncols = len(matched)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5),
                              squeeze=False)

    for idx, (target_ep, actual_ep, ckpt_path) in enumerate(matched):
        ax = axes[0, idx]
        print(f"Loading checkpoint epoch={actual_ep} (requested {target_ep})...")

        model = load_model_from_checkpoint(ckpt_path, device=device)
        energy = model["energy"]

        # Generate trajectories
        states, timesteps = generate_trajectories(model, n_particles=n_particles, device=device)

        # Terminal samples and mode assignment
        x1 = states[-1]  # (N, D)
        if hasattr(energy, "mode_centers"):
            mode_centers = energy.mode_centers.to(device)
            assignments = assign_modes_nearest(x1, mode_centers).cpu().numpy()
            K = mode_centers.shape[0]
        else:
            assignments = np.zeros(n_particles, dtype=int)
            K = 1

        # Colors per mode
        cmap = plt.cm.tab10
        colors = [cmap(k / max(K - 1, 1)) for k in range(K)]

        # Subsample timesteps for plotting (every nth step)
        n_steps = len(states)
        step_indices = np.linspace(0, n_steps - 1, min(50, n_steps), dtype=int)

        # Plot trajectories
        for i in range(n_particles):
            traj = torch.stack([states[s][i].cpu() for s in step_indices]).numpy()
            c = colors[assignments[i]]
            ax.plot(traj[:, 0], traj[:, 1], color=c, alpha=0.15, linewidth=0.4)

        # Mark terminal positions
        x1_np = x1.cpu().numpy()
        for k in range(K):
            mask = assignments == k
            ax.scatter(x1_np[mask, 0], x1_np[mask, 1],
                       c=[colors[k]], s=8, alpha=0.5, zorder=3,
                       label=f"Mode {k}" if idx == 0 else None)

        # Mark mode centers
        if hasattr(energy, "mode_centers"):
            centers = mode_centers.cpu().numpy()
            ax.scatter(centers[:, 0], centers[:, 1],
                       c="black", s=100, marker="*", zorder=5)

        ax.set_title(f"Epoch {actual_ep}", fontsize=12)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)

    if ncols > 0:
        axes[0, 0].legend(fontsize=8, loc="upper left")

    fig.suptitle("E6: Particle Trajectories (colored by terminal mode)", fontsize=14, y=1.02)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"Saved: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="E6: Particle Trajectory Visualization")
    parser.add_argument("--run-dir", required=True, help="Path to training run directory")
    parser.add_argument("--epochs", type=int, nargs="+", default=[20, 200, 1000],
                        help="Epochs to visualize (will use closest checkpoint)")
    parser.add_argument("--n-particles", type=int, default=200)
    parser.add_argument("--output", default="figures/e6_trajectories.pdf")
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()
    plot_trajectories(args.run_dir, args.epochs, args.n_particles,
                      args.output, args.device)


if __name__ == "__main__":
    main()
