#!/usr/bin/env python
"""
E4: Controller Field Comparison

At a fixed t, evaluate u_θ(x, t) on a 2D grid. Compare learned vs.
oracle controller (importance-reweighted AM regression). Plot quiver
fields and deficiency magnitude.

Usage:
    python scripts/e4_controller_field.py \\
        --checkpoint results/b3_asbs/seed_0/checkpoints/checkpoint_500.pt \\
        --t-values 0.3 0.5 0.7 \\
        --output figures/e4_b3_controller.pdf
"""

import argparse
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mc_utils import (
    load_model_from_checkpoint,
    generate_samples,
    assign_modes_nearest,
)


def evaluate_controller_on_grid(
    controller, ref_sde, grid_points, t_val, device="cuda"
):
    """Evaluate u_θ(x, t) on a grid.

    Args:
        controller: the learned controller network
        ref_sde: reference SDE (for g(t))
        grid_points: (G, 2) tensor of grid positions
        t_val: scalar time value

    Returns:
        u_field: (G, 2) tensor of controller outputs
    """
    G = grid_points.shape[0]
    t = torch.full((G, 1), t_val, device=device)

    with torch.no_grad():
        u = controller(t, grid_points)

    return u


def compute_oracle_field(
    controller, ref_sde, energy, source, cfg, grid_points, t_val,
    n_samples=10000, device="cuda",
):
    """Estimate the oracle controller field by importance-reweighted regression.

    The oracle controller is what AM regression would produce if the policy
    had perfect mode weights α = w. We approximate this by:
    1. Generate samples with current policy
    2. Compute importance weights w_k / α_k for each sample
    3. For each grid point, compute weighted conditional expectation

    For simplicity, we use a kernel-smoothed estimate:
    u_oracle(x, t) ≈ Σ_i w̃_i * K(x, x_t^i) * target_i / Σ_i w̃_i * K(x, x_t^i)
    """
    import adjoint_samplers.utils.train_utils as train_utils
    from adjoint_samplers.components.sde import ControlledSDE, sdeint

    sde = ControlledSDE(ref_sde, controller).to(device)

    # Generate trajectories
    x0 = source.sample([n_samples,]).to(device)
    timesteps = train_utils.get_timesteps(**cfg.timesteps).to(x0)
    with torch.no_grad():
        states = sdeint(sde, x0, timesteps, only_boundary=False)

    x1 = states[-1]

    # Find x_t at the closest timestep
    t_idx = (timesteps - t_val).abs().argmin().item()
    x_t = states[t_idx]  # (N, D)

    # Compute adjoint target: -∇E(x1) (simplified terminal cost)
    with torch.enable_grad():
        x1_req = x1.clone().detach().requires_grad_(True)
        E = energy.eval(x1_req)
        grad_E = torch.autograd.grad(E.sum(), x1_req)[0]
    target = -grad_E.detach()  # (N, D)

    # Compute importance weights if mode info available
    if hasattr(energy, "mode_centers") and hasattr(energy, "mode_weights"):
        mode_centers = energy.mode_centers.to(device)
        mode_weights = energy.mode_weights.to(device)
        K = mode_centers.shape[0]

        assignments = assign_modes_nearest(x1, mode_centers)
        alpha = torch.zeros(K, device=device)
        for k in range(K):
            alpha[k] = (assignments == k).float().mean()
        alpha = alpha.clamp(min=1e-6)

        # Importance weight per sample: w_k / α_k
        iw = torch.ones(n_samples, device=device)
        for k in range(K):
            mask = assignments == k
            iw[mask] = mode_weights[k] / alpha[k]
        iw = iw / iw.mean()  # normalize
    else:
        iw = torch.ones(n_samples, device=device)

    # Kernel-smoothed oracle estimate on grid
    # Use bandwidth based on median distance
    G = grid_points.shape[0]
    bandwidth = 0.5  # fixed for 2D visualization

    oracle = torch.zeros(G, 2, device=device)
    for start in range(0, G, 256):
        end = min(start + 256, G)
        chunk = grid_points[start:end]  # (C, 2)
        dists = torch.cdist(chunk, x_t)  # (C, N)
        K_vals = torch.exp(-dists**2 / (2 * bandwidth**2))  # (C, N)

        # Weighted kernel regression
        weighted_K = K_vals * iw.unsqueeze(0)  # (C, N)
        numerator = torch.bmm(
            weighted_K.unsqueeze(1),  # (C, 1, N)
            target.unsqueeze(0).expand(end - start, -1, -1),  # (C, N, D)
        ).squeeze(1)  # (C, D)
        denominator = weighted_K.sum(dim=-1, keepdim=True).clamp(min=1e-10)  # (C, 1)
        oracle[start:end] = numerator / denominator

    # Scale by -g(t)^2 to match controller convention
    g_t = ref_sde.diff(torch.tensor([t_val], device=device))
    oracle = -g_t**2 * oracle

    return oracle


def plot_controller_fields(
    checkpoint_path: str,
    t_values: list,
    grid_range: tuple = (-5, 5),
    grid_size: int = 25,
    output_path: str = "figures/e4_controller.pdf",
    device: str = "cuda",
):
    """Plot learned vs. oracle controller fields and deficiency."""
    model = load_model_from_checkpoint(checkpoint_path, device=device)
    controller = model["controller"]
    ref_sde = model["ref_sde"]
    energy = model["energy"]
    source = model["source"]
    cfg = model["cfg"]
    epoch = model["epoch"]

    # Determine grid range from mode centers if available
    if hasattr(energy, "mode_centers"):
        centers = energy.mode_centers.cpu().numpy()
        margin = 3.0
        x_range = (centers[:, 0].min() - margin, centers[:, 0].max() + margin)
        y_range = (centers[:, 1].min() - margin, centers[:, 1].max() + margin)
    else:
        x_range = y_range = grid_range

    gx = torch.linspace(x_range[0], x_range[1], grid_size, device=device)
    gy = torch.linspace(y_range[0], y_range[1], grid_size, device=device)
    grid_x, grid_y = torch.meshgrid(gx, gy, indexing="ij")
    grid_points = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)

    nrows = len(t_values)
    fig, axes = plt.subplots(nrows, 3, figsize=(15, 4.5 * nrows),
                              gridspec_kw={"wspace": 0.25, "hspace": 0.35})
    if nrows == 1:
        axes = axes.reshape(1, -1)

    for row, t_val in enumerate(t_values):
        print(f"Computing fields for t={t_val}...")

        # Learned controller
        u_learned = evaluate_controller_on_grid(
            controller, ref_sde, grid_points, t_val, device
        ).cpu().numpy()

        # Oracle controller
        u_oracle = compute_oracle_field(
            controller, ref_sde, energy, source, cfg,
            grid_points, t_val, n_samples=5000, device=device
        ).cpu().numpy()

        # Deficiency
        delta_u = u_learned - u_oracle

        gx_np = gx.cpu().numpy()
        gy_np = gy.cpu().numpy()
        G = grid_size

        # Energy contours
        with torch.no_grad():
            E = energy.eval(grid_points).cpu().numpy().reshape(G, G)
        E_plot = np.clip(E, E.min(), np.percentile(E, 95))

        titles = [f"Learned $u_\\theta$ (t={t_val})",
                  f"Oracle $u^*$ (t={t_val})",
                  f"Deficiency $\\Delta u$ (t={t_val})"]
        fields = [u_learned, u_oracle, delta_u]

        for col, (field, title) in enumerate(zip(fields, titles)):
            ax = axes[row, col]

            # Energy contours
            ax.contour(gx_np, gy_np, E_plot.T, levels=15,
                       colors="gray", alpha=0.3, linewidths=0.5)

            ux = field[:, 0].reshape(G, G)
            uy = field[:, 1].reshape(G, G)

            if col < 2:
                # Quiver plot for learned and oracle
                ax.quiver(gx_np, gy_np, ux.T, uy.T,
                          alpha=0.7, scale=None, width=0.004)
            else:
                # Deficiency: color by magnitude, quiver for direction
                mag = np.sqrt(ux**2 + uy**2)
                im = ax.pcolormesh(gx_np, gy_np, mag.T,
                                    cmap="Reds", alpha=0.6)
                ax.quiver(gx_np, gy_np, ux.T, uy.T,
                          alpha=0.5, scale=None, width=0.004)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                             label="$|\\Delta u|$")

            # Mark mode centers
            if hasattr(energy, "mode_centers"):
                centers = energy.mode_centers.cpu().numpy()
                ax.scatter(centers[:, 0], centers[:, 1],
                           c="red", s=80, marker="*", zorder=5)

            ax.set_title(title, fontsize=11)
            ax.set_aspect("equal")

    fig.suptitle(f"E4: Controller Field Comparison (epoch {epoch})", fontsize=14, y=1.01)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"Saved: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="E4: Controller Field Comparison")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--t-values", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    parser.add_argument("--grid-size", type=int, default=25)
    parser.add_argument("--output", default="figures/e4_controller.pdf")
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()
    plot_controller_fields(args.checkpoint, args.t_values,
                           grid_size=args.grid_size,
                           output_path=args.output, device=args.device)


if __name__ == "__main__":
    main()
