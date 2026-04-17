"""
v2 Model Variants for symmetry-breaking experiments.

Provides:
  - FrozenBiasFourierMLP: Standard FourierMLP + a fixed random bias (A1.2)
  - ModeAwareFourierMLP: Two independent sub-networks split by x-axis (A1.2)
"""

from __future__ import annotations
from typing import Callable

import torch
import torch.nn as nn

from adjoint_samplers.components.model import FourierMLP, Model


class FrozenBiasFourierMLP(FourierMLP):
    """FourierMLP with a frozen (non-trainable) random bias added to the output.

    Used by A1.2 (architecture symmetry breaking) to test whether a fixed
    asymmetric bias in the controller output breaks shielding.

    Args:
        bias_scale: Scale of the frozen random bias vector (drawn once at init).
        bias_seed: RNG seed for reproducibility of the bias vector.
        **kwargs: All FourierMLP arguments.
    """

    def __init__(
        self,
        bias_scale: float = 0.1,
        bias_seed: int = 42,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.bias_scale = bias_scale
        # Draw a fixed random bias vector of shape (dim,)
        rng = torch.Generator().manual_seed(bias_seed)
        frozen_bias = torch.randn(self.dim, generator=rng) * bias_scale
        # Register as buffer (moves with device, saved in state_dict, but NOT a parameter)
        self.register_buffer("frozen_bias", frozen_bias)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(t, x)
        return out + self.frozen_bias.unsqueeze(0)


class ModeAwareFourierMLP(Model):
    """Two independent FourierMLPs, one for each side of the x-axis.

    For 2D problems with modes on opposite sides of the x-axis (e.g., B1),
    this removes parameter sharing across the symmetry axis. Samples with
    x[0] < 0 are processed by network A, x[0] >= 0 by network B.

    Used by A1.2 (architecture symmetry breaking).

    Args:
        split_dim: Which dimension to split on (default 0 = x-axis).
        split_value: Threshold value for splitting (default 0.0).
        **kwargs: FourierMLP arguments passed to both sub-networks.
    """

    def __init__(
        self,
        split_dim: int = 0,
        split_value: float = 0.0,
        dim: int = 2,
        activation: Callable = None,
        num_layers: int = 4,
        channels: int = 64,
        last_bias_init: Callable | None = None,
        last_weight_init: Callable | None = None,
        **kwargs,
    ):
        super().__init__(dim=dim, **kwargs)
        self.split_dim = split_dim
        self.split_value = split_value

        mlp_kwargs = dict(
            dim=dim,
            activation=activation or nn.GELU(),
            num_layers=num_layers,
            channels=channels,
            last_bias_init=last_bias_init,
            last_weight_init=last_weight_init,
        )
        self.net_left = FourierMLP(**mlp_kwargs)
        self.net_right = FourierMLP(**mlp_kwargs)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        mask_left = x[:, self.split_dim] < self.split_value
        mask_right = ~mask_left

        out = torch.zeros_like(x)
        if mask_left.any():
            out[mask_left] = self.net_left(t, x[mask_left])
        if mask_right.any():
            out[mask_right] = self.net_right(t, x[mask_right])
        return out
