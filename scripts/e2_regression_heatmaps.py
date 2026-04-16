#!/usr/bin/env python
"""
E2: Regression Weight Heatmaps

Estimate η_k(x, t_fix) on a dense 2D grid by forward-simulating
particles from each grid point and counting which mode they reach.
Compare learned η_k vs oracle η_k*, show difference map.

Usage:
    python scripts/e2_regression_heatmaps.py \\
        --checkpoint results/b3_asbs/seed_0/checkpoints/checkpoint_500.pt \\
        --t-values 0.3 0.5 0.7 \\
        --output figures/e2_b3_heatmaps.pdf
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
    assign_modes_nearest,
)


def estimate_eta_on_grid(
    model_dict,
    t_val: float,
    grid_points: torch.Tensor,
    n_sims_per_point: int = 200,
    device: str = "cuda",
):
    """Estimate η_k(x, t) on a grid by forward-simulating from each point.

    For each grid point x, simulate n_sims particles starting at X_t = x,
    evolve forward to X_1, and count which mode each reaches.

    Returns:
        eta: (G, K) tensor of η_k estimates on the grid
    """
    import adjoint_samplers.utils.train_utils as train_utils
    from adjoint_samplers.components.sde import sdeint

    sde = model_dict["sde"]
    energy = model_dict["energy"]
    cfg = model_dict["cfg"]

    mode_centers = energy.mode_centers.to(device)
    K = mode_centers.shape[0]
    G = grid_points.shape[0]

    # Get timesteps from t_val to t1
    full_timesteps = train_utils.get_timesteps(**cfg.timesteps).to(device)
    t_idx = (full_timesteps - t_val).abs().argmin().item()
    timesteps_from_t = full_timesteps[t_idx:]

    if len(timesteps_from_t) < 2:
        timesteps_from_t = full_timesteps[-2:]

    eta = torch.zeros(G, K, device=device)

    # Process in chunks to manage memory
    chunk_size = 32
    for g_start in range(0, G, chunk_size):
        g_end = min(g_start + chunk_size, G)
        n_pts = g_end - g_start

        # Replicate each grid point n_sims_per_point times
        x_start = grid_points[g_start:g_end]  # (C, D)
        x_start_rep = x_start.unsqueeze(1).expand(-1, n_sims_per_point, -1)
        x_start_rep = x_start_rep.reshape(-1, x_start.shape[-1])  # (C*S, D)

        with torch.no_grad():
            _, x1 = sdeint(sde, x_start_rep, timesteps_from_t, only_boundary=True)

        # Assign to modes
        assignments = assign_modes_nearest(x1, mode_centers)  # (C*S,)
        assignments = assignments.reshape(n_pts, n_sims_per_point)  # (C, S)

        # Count per mode
        for k in range(K):
            eta[g_start:g_end, k] = (assignments == k).float().mean(dim=1)

    return eta


def compute_oracle_eta(mode_centers, mode_weights, grid_points, device="cuda"):
    """Compute oracle η_k*(x, t) = w_k * q_t^(k)(x) / Σ_l w_l * q_t^(l)(x).

    Approximated using Gaussian kernels centered at mode centers with
    a bandwidth estimated from mode separation.
    """
    K = mode_centers.shape[0]
    G = grid_points.shape[0]

    # Bandwidth: half the minimum inter-mode distance
    dists = torch.cdist(mode_centers.unsqueeze(0), mode_centers.unsqueeze(0)).squeeze(0)
    dists.fill_diagonal_(float("inf"))
    bandwidth = dists.min().item() / 2.0
    bandwidth = max(bandwidth, 0.5)

    # q_t^(k)(x) ∝ exp(-|x - μ_k|² / (2σ²))
    diff = grid_points.unsqueeze(1) - mode_centers.unsqueeze(0)  # (G, K, D)
    sq_dist = (diff**2).sum(dim=-1)  # (G, K)
    log_q = -sq_dist / (2 * bandwidth**2)

    # η_k* = w_k * q_k / Σ w_l * q_l
    log_weighted = log_q + mode_weights.log().unsqueeze(0)  # (G, K)
    eta_oracle = torch.softmax(log_weighted, dim=-1)  # (G, K)

    return eta_oracle


def plot_heatmaps(
    checkpoint_path: str,
    t_values: list,
    grid_size: int = 40,
    n_sims: int = 200,
    output_path: str = "figures/e2_heatmaps.pdf",
    device: str = "cuda",
):
    """Plot regression weight heatmaps: learned η, oracle η*, difference."""
    model = load_model_from_checkpoint(checkpoint_path, device=device)
    energy = model["energy"]
    epoch = model["epoch"]

    if not hasattr(energy, "mode_centers"):
        print("Energy has no mode_centers — cannot compute η.")
        return

    mode_centers = energy.mode_centers.to(device)
    mode_weights = energy.mode_weights.to(device)
    K = mode_centers.shape[0]

    # Grid range from mode centers
    centers_np = mode_centers.cpu().numpy()
    margin = 3.0
    x_range = (centers_np[:, 0].min() - margin, centers_np[:, 0].max() + margin)
    y_range = (centers_np[:, 1].min() - margin, centers_np[:, 1].max() + margin)

    gx = torch.linspace(x_range[0], x_range[1], grid_size, device=device)
    gy = torch.linspace(y_range[0], y_range[1], grid_size, device=device)
    grid_x, grid_y = torch.meshgrid(gx, gy, indexing="ij")
    grid_points = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)

    # For each mode k, we show: learned η_k, oracle η_k*, difference
    # Layout: rows = t values, columns = [η_0 learned | η_0 oracle | diff_0 | η_1 learned | ...]
    # Simplified: for first mode only if K=2, or dominant mode
    # Better layout: rows = t values, 3 columns = learned η_k*, oracle η_k*, diff for mode 0

    # Pick the minority mode (most interesting for mode concentration)
    w_np = mode_weights.cpu().numpy()
    minority_k = int(np.argmin(w_np))

    nrows = len(t_values)
    fig, axes = plt.subplots(nrows, 3, figsize=(14, 4 * nrows),
                              gridspec_kw={"wspace": 0.25, "hspace": 0.35})
    if nrows == 1:
        axes = axes.reshape(1, -1)

    gx_np = gx.cpu().numpy()
    gy_np = gy.cpu().numpy()
    G = grid_size

    for row, t_val in enumerate(t_values):
        print(f"Estimating η at t={t_val} (grid={grid_size}², sims={n_sims}/point)...")

        # Learned η
        eta_learned = estimate_eta_on_grid(
            model, t_val, grid_points, n_sims_per_point=n_sims, device=device
        )
        eta_k_learned = eta_learned[:, minority_k].cpu().numpy().reshape(G, G)

        # Oracle η
        eta_oracle = compute_oracle_eta(mode_centers, mode_weights, grid_points, device)
        eta_k_oracle = eta_oracle[:, minority_k].cpu().numpy().reshape(G, G)

        # Difference
        diff = eta_k_learned - eta_k_oracle

        # Plot learned
        ax = axes[row, 0]
        im = ax.pcolormesh(gx_np, gy_np, eta_k_learned.T, cmap="YlOrRd",
                           vmin=0, vmax=1)
        ax.scatter(centers_np[:, 0], centers_np[:, 1], c="black", s=60, marker="*", zorder=5)
        ax.set_title(f"Learned $\\eta_{minority_k}$ (t={t_val})", fontsize=11)
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Plot oracle
        ax = axes[row, 1]
        im = ax.pcolormesh(gx_np, gy_np, eta_k_oracle.T, cmap="YlOrRd",
                           vmin=0, vmax=1)
        ax.scatter(centers_np[:, 0], centers_np[:, 1], c="black", s=60, marker="*", zorder=5)
        ax.set_title(f"Oracle $\\eta_{minority_k}^*$ (t={t_val})", fontsize=11)
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Plot difference
        ax = axes[row, 2]
        vmax_diff = max(abs(diff.min()), abs(diff.max()), 0.1)
        im = ax.pcolormesh(gx_np, gy_np, diff.T, cmap="RdBu_r",
                           vmin=-vmax_diff, vmax=vmax_diff)
        ax.scatter(centers_np[:, 0], centers_np[:, 1], c="black", s=60, marker="*", zorder=5)
        ax.set_title(f"$\\eta_{minority_k} - \\eta_{minority_k}^*$ (t={t_val})", fontsize=11)
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="$\\Delta\\eta$")

    fig.suptitle(
        f"E2: Regression Weight Heatmaps — Mode {minority_k} "
        f"($w_{minority_k}$={w_np[minority_k]:.3f}) @ epoch {epoch}",
        fontsize=13, y=1.02,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"Saved: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="E2: Regression Weight Heatmaps")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--t-values", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    parser.add_argument("--grid-size", type=int, default=40)
    parser.add_argument("--n-sims", type=int, default=200,
                        help="Simulations per grid point")
    parser.add_argument("--output", default="figures/e2_heatmaps.pdf")
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()
    plot_heatmaps(args.checkpoint, args.t_values, args.grid_size,
                  args.n_sims, args.output, args.device)


if __name__ == "__main__":
    main()
