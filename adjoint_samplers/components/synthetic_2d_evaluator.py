"""
Evaluator for 2D synthetic benchmarks (B1–B7).

Computes:
  - Mode weights α_k, KL(α||w), TV(α,w), alive mode count
  - Energy W2 against reference samples
  - 1D marginal W2 for each coordinate
  - 2D density comparison figure (generated KDE vs target contours)
"""

import numpy as np
from typing import Dict
from pathlib import Path

import torch
import ot as pot

from adjoint_samplers.energies.base_energy import BaseEnergy
from adjoint_samplers.utils.eval_utils import get_fig_axes, fig2img


class Synthetic2DEvaluator:
    """Evaluator for 2D synthetic energy benchmarks.

    Generates reference samples at init via grid-based importance sampling,
    then computes distributional metrics and density figures at each call.

    Args:
        energy: a BaseEnergy with `mode_centers` and `mode_weights` properties
        n_ref_samples: number of reference samples to generate
        grid_size: grid resolution for reference sample generation
        grid_range: (low, high) for the sampling grid (auto-detected from mode centers if None)
    """

    def __init__(
        self,
        energy: BaseEnergy,
        n_ref_samples: int = 50000,
        grid_size: int = 500,
        grid_range: float = None,
        device: str = "cpu",
    ):
        self.energy = energy
        self.dim = energy.dim
        # Use the energy's device if available (energy is on cuda during training)
        self.device = getattr(energy, "device", device)

        # Detect grid range from mode centers
        if grid_range is not None:
            self.x_range = (-grid_range, grid_range)
            self.y_range = (-grid_range, grid_range)
        elif hasattr(energy, "mode_centers"):
            centers = energy.mode_centers
            margin = 5.0
            self.x_range = (centers[:, 0].min().item() - margin,
                            centers[:, 0].max().item() + margin)
            self.y_range = (centers[:, 1].min().item() - margin,
                            centers[:, 1].max().item() + margin)
        else:
            self.x_range = (-10, 10)
            self.y_range = (-10, 10)

        # Mode info
        self.has_modes = hasattr(energy, "mode_centers") and hasattr(energy, "mode_weights")
        if self.has_modes:
            self.mode_centers = energy.mode_centers
            self.mode_weights = energy.mode_weights
            self.K = self.mode_centers.shape[0]

        # Generate reference samples via grid importance sampling
        print(f"  Generating {n_ref_samples} reference samples via importance sampling...")
        self.ref_samples = self._generate_reference_samples(
            n_ref_samples, grid_size, self.device
        )
        print(f"  Reference samples: shape={self.ref_samples.shape}")

        # Rolling density figure
        self.fig, axes = get_fig_axes(ncol=6, nrow=8, ax_length_in=3.5)
        self.axes = axes.reshape(-1)
        self.subplot_idx = 0

        # Plot target density on first subplot
        self._plot_density(self.ref_samples[:5000].cpu().numpy(), "Target (ref)")

    def _generate_reference_samples(self, n_samples, grid_size, device):
        """Generate reference samples via grid-based importance sampling."""
        gx = torch.linspace(self.x_range[0], self.x_range[1], grid_size, device=device)
        gy = torch.linspace(self.y_range[0], self.y_range[1], grid_size, device=device)
        GX, GY = torch.meshgrid(gx, gy, indexing="ij")
        grid = torch.stack([GX.reshape(-1), GY.reshape(-1)], dim=-1)

        with torch.no_grad():
            E = self.energy.eval(grid)
            log_p = -E
            log_p = log_p - log_p.max()
            probs = torch.exp(log_p)
            probs = probs / probs.sum()

        # Sample indices proportional to probability
        indices = torch.multinomial(probs, n_samples, replacement=True)
        samples = grid[indices]

        # Add small noise to avoid grid artifacts
        cell_size = (self.x_range[1] - self.x_range[0]) / grid_size
        samples = samples + cell_size * 0.3 * torch.randn_like(samples)

        return samples.cpu()

    def _plot_density(self, samples_np, title):
        """Plot 2D density on the rolling figure."""
        if self.subplot_idx >= len(self.axes):
            return

        ax = self.axes[self.subplot_idx]
        self.subplot_idx += 1

        ax.hist2d(samples_np[:, 0], samples_np[:, 1], bins=80,
                  range=[[self.x_range[0], self.x_range[1]],
                         [self.y_range[0], self.y_range[1]]],
                  cmap="viridis", density=True)

        if self.has_modes:
            centers = self.mode_centers.cpu().numpy()
            ax.scatter(centers[:, 0], centers[:, 1],
                       c="red", s=40, marker="*", zorder=5)

        ax.set_xlim(self.x_range)
        ax.set_ylim(self.y_range)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=8)
        ax.tick_params(labelsize=6)

    def __call__(self, samples: torch.Tensor) -> Dict:
        B, D = samples.shape
        assert D == self.dim

        # Move samples to energy's device for eval, keep cpu copy for metrics
        energy_device = self.device
        samples_dev = samples.to(energy_device)
        samples_cpu = samples.cpu()
        result = {}

        # ── Energy W2 ──
        idxs = torch.randperm(len(self.ref_samples))[:B]
        ref = self.ref_samples[idxs].to(energy_device)

        with torch.no_grad():
            gen_E = self.energy.eval(samples_dev).cpu().numpy()
            ref_E = self.energy.eval(ref).cpu().numpy()

        result["energy_w2"] = float(pot.emd2_1d(ref_E, gen_E) ** 0.5)

        ref_cpu = ref.cpu()

        # ── Marginal W2 (per coordinate) ──
        for d in range(D):
            w2_d = float(pot.emd2_1d(
                ref_cpu[:, d].numpy(), samples_cpu[:, d].numpy()
            ) ** 0.5)
            result[f"marginal_w2_x{d}"] = w2_d
        result["marginal_w2_mean"] = float(np.mean(
            [result[f"marginal_w2_x{d}"] for d in range(D)]
        ))

        # ── 2D sample W2 (sliced Wasserstein as proxy for full W2) ──
        n_proj = 50
        sw2 = 0.0
        for _ in range(n_proj):
            theta = torch.randn(D)
            theta = theta / theta.norm()
            proj_gen = (samples_cpu @ theta).numpy()
            proj_ref = (ref_cpu @ theta).numpy()
            sw2 += pot.emd2_1d(proj_ref, proj_gen)
        result["sliced_w2"] = float((sw2 / n_proj) ** 0.5)

        # ── Mode metrics ──
        if self.has_modes:
            mode_centers = self.mode_centers.to(samples_cpu.device)
            mode_weights = self.mode_weights.to(samples_cpu.device)
            K = self.K

            dists = torch.cdist(
                samples_cpu.unsqueeze(0), mode_centers.unsqueeze(0)
            ).squeeze(0)
            assignments = dists.argmin(dim=-1)

            alpha = torch.zeros(K, dtype=torch.float64)
            for k in range(K):
                alpha[k] = (assignments == k).sum().item()
            alpha = (alpha / alpha.sum()).float()

            # KL(α || w)
            a_safe = alpha.clamp(min=1e-10)
            w_safe = mode_weights.clamp(min=1e-10)
            kl_val = (a_safe * (a_safe.log() - w_safe.log())).sum().item()
            tv_val = 0.5 * (alpha - mode_weights).abs().sum().item()
            alive = int((alpha > 0.1 * mode_weights).sum().item())

            result["kl_mode"] = kl_val
            result["tv_mode"] = tv_val
            result["alive_modes"] = alive
            result["total_modes"] = K

            for k in range(K):
                result[f"alpha_{k}"] = alpha[k].item()
                result[f"target_w_{k}"] = mode_weights[k].item()

        # ── Energy statistics ──
        result["mean_energy"] = float(gen_E.mean())
        result["std_energy"] = float(gen_E.std())
        result["median_energy"] = float(np.median(gen_E))

        # ── Density figure ──
        self._plot_density(
            samples_cpu[:5000].numpy(),
            f"Eval #{self.subplot_idx}"
        )
        self.fig.canvas.draw()
        result["density_img"] = fig2img(self.fig)

        return result
