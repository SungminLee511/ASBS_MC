#!/usr/bin/env python
"""
E5: Terminal Density Visualization

Forward-simulate N=50000 particles at various checkpoints, compute
KDE of terminal positions, and compare to target density contours.

Usage:
    python scripts/e5_terminal_density.py \\
        --run-dir results/b2_asbs/seed_0 \\
        --epochs 50 200 500 1000 \\
        --output figures/e5_b2_density.pdf

    # B6 scatter mode (dot size ∝ α_k/w_k)
    python scripts/e5_terminal_density.py \\
        --run-dir results/b6_asbs/seed_0 \\
        --epochs 50 200 500 1000 \\
        --scatter-mode \\
        --output figures/e5_b6_scatter.pdf
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
    assign_modes_nearest,
    compute_mode_weights,
    find_checkpoints,
)


def compute_target_density_grid(energy, x_range, y_range, grid_size=200, device="cpu"):
    """Compute target density p(x) ∝ exp(-E(x)) on a 2D grid."""
    gx = torch.linspace(x_range[0], x_range[1], grid_size, device=device)
    gy = torch.linspace(y_range[0], y_range[1], grid_size, device=device)
    grid_x, grid_y = torch.meshgrid(gx, gy, indexing="ij")
    grid = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)

    with torch.no_grad():
        E = energy.eval(grid)
        log_p = -E
        log_p = log_p - log_p.max()
        p = torch.exp(log_p).reshape(grid_size, grid_size)

    return gx.cpu().numpy(), gy.cpu().numpy(), p.cpu().numpy()


def plot_density_panels(
    run_dir: str,
    epochs: list,
    n_samples: int = 20000,
    scatter_mode: bool = False,
    output_path: str = "figures/e5_density.pdf",
    device: str = "cuda",
):
    """Plot terminal density panels at specified epochs."""
    run_dir = Path(run_dir)
    available_ckpts = {}
    for ckpt in find_checkpoints(run_dir):
        ep = int(ckpt.stem.split("_")[-1])
        available_ckpts[ep] = ckpt

    if not available_ckpts:
        print(f"No checkpoints in {run_dir}")
        return

    avail_epochs = sorted(available_ckpts.keys())
    matched = []
    for target_ep in epochs:
        closest = min(avail_epochs, key=lambda e: abs(e - target_ep))
        matched.append((target_ep, closest, available_ckpts[closest]))

    ncols = len(matched)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5), squeeze=False)

    # Load first checkpoint to get energy for target density
    first_model = load_model_from_checkpoint(str(matched[0][2]), device=device)
    energy = first_model["energy"]

    # Determine plot range
    if hasattr(energy, "mode_centers"):
        centers = energy.mode_centers.cpu().numpy()
        margin = 4.0
        x_range = (centers[:, 0].min() - margin, centers[:, 0].max() + margin)
        y_range = (centers[:, 1].min() - margin, centers[:, 1].max() + margin)
    else:
        x_range = y_range = (-6, 6)

    # Target density
    gx, gy, target_p = compute_target_density_grid(
        energy, x_range, y_range, grid_size=200, device=device
    )

    for idx, (target_ep, actual_ep, ckpt_path) in enumerate(matched):
        ax = axes[0, idx]
        print(f"Epoch {actual_ep}: generating {n_samples} samples...")

        model = load_model_from_checkpoint(str(ckpt_path), device=device)
        samples = generate_samples(model, n_samples=n_samples, device=device)
        samples_np = samples.cpu().numpy()

        if scatter_mode and hasattr(energy, "mode_centers") and hasattr(energy, "mode_weights"):
            # B6-style: scatter with dot size ∝ α_k/w_k
            mode_centers = energy.mode_centers.to(device)
            mode_weights = energy.mode_weights.to(device)
            K = mode_centers.shape[0]

            assignments = assign_modes_nearest(samples, mode_centers)
            alpha = compute_mode_weights(assignments, K)
            ratios = alpha / mode_weights.cpu()

            centers_np = mode_centers.cpu().numpy()
            cmap = plt.cm.RdYlGn
            for k in range(K):
                r = ratios[k].item()
                size = max(10, min(300, r * 100))
                color = cmap(min(r, 2.0) / 2.0)
                ax.scatter(centers_np[k, 0], centers_np[k, 1],
                           s=size, c=[color], edgecolors="black", linewidths=0.5,
                           zorder=3)

            # Background: target density contours
            ax.contour(gx, gy, target_p.T, levels=10,
                       colors="gray", alpha=0.4, linewidths=0.5)
        else:
            # KDE density contours
            from scipy.stats import gaussian_kde
            try:
                kde = gaussian_kde(samples_np.T)
                gx_dense = np.linspace(x_range[0], x_range[1], 100)
                gy_dense = np.linspace(y_range[0], y_range[1], 100)
                GX, GY = np.meshgrid(gx_dense, gy_dense)
                positions = np.vstack([GX.ravel(), GY.ravel()])
                Z = kde(positions).reshape(GX.shape)

                ax.contourf(gx_dense, gy_dense, Z, levels=20,
                            cmap="Blues", alpha=0.6)
            except Exception:
                ax.scatter(samples_np[:, 0], samples_np[:, 1],
                           s=1, alpha=0.1, color="steelblue")

            # Target contours
            ax.contour(gx, gy, target_p.T, levels=10,
                       colors="black", linewidths=0.8, alpha=0.7)

        # Mode centers
        if hasattr(energy, "mode_centers"):
            centers_np = energy.mode_centers.cpu().numpy()
            if not scatter_mode:
                ax.scatter(centers_np[:, 0], centers_np[:, 1],
                           c="red", s=60, marker="*", zorder=5)

        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_title(f"Epoch {actual_ep}", fontsize=12)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)

    mode_str = "scatter" if scatter_mode else "KDE"
    fig.suptitle(f"E5: Terminal Density ({mode_str}, N={n_samples})",
                 fontsize=14, y=1.02)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"Saved: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="E5: Terminal Density Visualization")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--epochs", type=int, nargs="+", default=[50, 200, 500, 1000])
    parser.add_argument("--n-samples", type=int, default=20000)
    parser.add_argument("--scatter-mode", action="store_true",
                        help="Use scatter mode (B6-style, dot size ∝ α/w)")
    parser.add_argument("--output", default="figures/e5_density.pdf")
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()
    plot_density_panels(args.run_dir, args.epochs, args.n_samples,
                        args.scatter_mode, args.output, args.device)


if __name__ == "__main__":
    main()
