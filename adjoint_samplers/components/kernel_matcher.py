"""
Kernel-based AM regression matcher for A4.1 (Exact vs Approximate AM).

Replaces the neural network with a kernel regression estimator on a spatial
grid.  For 2D problems this can be made arbitrarily close to "exact AM"
at the cost of compute.

Usage:
    Instantiate KernelAMMatcher and use it in place of AdjointVEMatcher.
    The matcher maintains a grid-based adjoint estimate instead of a neural net.
"""

import torch
import numpy as np
from adjoint_samplers.components.sde import BaseSDE, sdeint


class KernelAMController(torch.nn.Module):
    """Non-parametric controller based on kernel regression on a grid.

    Stores a grid of (t, x) -> adjoint values and interpolates at query points
    using a Gaussian kernel.

    Args:
        dim: Spatial dimension.
        grid_range: (min, max) for each spatial dimension.
        grid_size: Number of grid points per spatial dimension.
        n_time_bins: Number of time bins.
        bandwidth: Kernel bandwidth (sigma for Gaussian kernel).
    """

    def __init__(
        self,
        dim: int = 2,
        grid_range: tuple = (-8.0, 8.0),
        grid_size: int = 50,
        n_time_bins: int = 20,
        bandwidth: float = 0.5,
    ):
        super().__init__()
        self.dim = dim
        self.grid_size = grid_size
        self.n_time_bins = n_time_bins
        self.bandwidth = bandwidth

        # Build spatial grid
        lin = torch.linspace(grid_range[0], grid_range[1], grid_size)
        if dim == 2:
            gx, gy = torch.meshgrid(lin, lin, indexing="ij")
            grid_points = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)
        else:
            grid_points = lin.unsqueeze(-1)  # 1D fallback

        self.register_buffer("grid_points", grid_points)  # (G, D)

        # Time bin edges
        t_edges = torch.linspace(0.0, 1.0, n_time_bins + 1)
        self.register_buffer("t_edges", t_edges)

        # Adjoint values on grid: (n_time_bins, G, D)
        G = grid_points.shape[0]
        self.register_buffer(
            "grid_adjoints",
            torch.zeros(n_time_bins, G, dim),
        )
        # Counts for running average
        self.register_buffer(
            "grid_counts",
            torch.zeros(n_time_bins, G),
        )

    def _time_bin(self, t: torch.Tensor) -> torch.Tensor:
        """Map time values to bin indices."""
        t_flat = t.view(-1)
        bins = torch.bucketize(t_flat, self.t_edges[1:-1])
        return bins.clamp(0, self.n_time_bins - 1)

    def update(self, t: torch.Tensor, x: torch.Tensor, adjoint: torch.Tensor):
        """Update grid adjoint estimates with new data.

        Args:
            t: (B, 1) time values
            x: (B, D) spatial positions
            adjoint: (B, D) adjoint values
        """
        B = x.shape[0]
        t_bins = self._time_bin(t)  # (B,)

        # For each sample, find nearest grid point and update
        dists = torch.cdist(x, self.grid_points)  # (B, G)
        nearest = dists.argmin(dim=1)  # (B,)

        for i in range(B):
            tb = t_bins[i]
            gi = nearest[i]
            n = self.grid_counts[tb, gi]
            # Running average update
            self.grid_adjoints[tb, gi] = (
                n * self.grid_adjoints[tb, gi] + adjoint[i]
            ) / (n + 1)
            self.grid_counts[tb, gi] = n + 1

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Kernel-regression interpolation at query (t, x).

        Args:
            t: scalar or (B, 1) time
            x: (B, D) spatial positions

        Returns:
            u: (B, D) controller output (negative adjoint)
        """
        B, D = x.shape
        t_bins = self._time_bin(t.view(-1, 1))  # (B,)

        # Compute Gaussian kernel weights: K(x, x_grid)
        sq_dists = torch.cdist(x, self.grid_points).pow(2)  # (B, G)
        weights = torch.exp(-sq_dists / (2 * self.bandwidth ** 2))  # (B, G)

        # Weighted average of grid adjoint values
        out = torch.zeros(B, D, device=x.device)
        for i in range(B):
            tb = t_bins[i]
            w = weights[i]  # (G,)
            # Only use grid points that have been updated
            mask = self.grid_counts[tb] > 0
            if mask.any():
                w_masked = w[mask]
                w_sum = w_masked.sum() + 1e-10
                adj = self.grid_adjoints[tb][mask]  # (G', D)
                out[i] = (w_masked.unsqueeze(-1) * adj).sum(0) / w_sum

        return -out  # Controller output is negative adjoint
