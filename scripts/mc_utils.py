#!/usr/bin/env python
"""
Utility library for post-hoc reconstruction from checkpoints.

Provides functions to:
  - Load a full model (energy + controller + SDE) from a saved checkpoint
  - Generate samples via forward SDE rollout
  - Compute mode assignment, weights, KL, TV, alive count

Used by reconstruct_tracking.py and other post-hoc analysis scripts.
"""

import sys
from pathlib import Path

import torch
import numpy as np

# Ensure project root is importable
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import hydra
from omegaconf import OmegaConf

from adjoint_samplers.components.sde import ControlledSDE, sdeint
import adjoint_samplers.utils.train_utils as train_utils


def find_checkpoints(run_dir) -> list:
    """Find all epoch checkpoints in a run directory, sorted by epoch.

    Looks for checkpoints/<checkpoint_NNN.pt> files, excludes checkpoint_latest.pt.

    Returns:
        Sorted list of Path objects.
    """
    run_dir = Path(run_dir)
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return []

    ckpts = []
    for p in ckpt_dir.glob("checkpoint_*.pt"):
        name = p.stem  # e.g. "checkpoint_100"
        if "latest" in name:
            continue
        try:
            epoch = int(name.split("_")[-1])
            ckpts.append((epoch, p))
        except ValueError:
            continue

    ckpts.sort(key=lambda x: x[0])
    return [p for _, p in ckpts]


def load_model_from_checkpoint(ckpt_path: str, device: str = "cuda") -> dict:
    """Rebuild energy + controller + SDE from a saved checkpoint.

    The checkpoint contains the full Hydra config (cfg) and the controller
    state dict. We re-instantiate all components via hydra.utils.instantiate
    and load the trained weights.

    Args:
        ckpt_path: Path to checkpoint_NNN.pt file.
        device: Device to load model onto.

    Returns:
        Dict with keys: energy, sde, controller, ref_sde, cfg, source,
                        corrector (if ASBS), epoch.
    """
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = checkpoint["cfg"]

    # Ensure cfg is an OmegaConf DictConfig (it should be, but guard)
    if not OmegaConf.is_config(cfg):
        cfg = OmegaConf.create(cfg)

    # ── Instantiate energy ──
    energy = hydra.utils.instantiate(cfg.energy, device=device)

    # ── Instantiate source ──
    source = hydra.utils.instantiate(cfg.source, device=device)

    # ── Instantiate SDE ──
    ref_sde = hydra.utils.instantiate(cfg.ref_sde).to(device)
    controller = hydra.utils.instantiate(cfg.controller).to(device)
    controller.load_state_dict(checkpoint["controller"])
    controller.eval()
    sde = ControlledSDE(ref_sde, controller).to(device)

    result = {
        "energy": energy,
        "sde": sde,
        "controller": controller,
        "ref_sde": ref_sde,
        "source": source,
        "cfg": cfg,
        "epoch": checkpoint.get("epoch", -1),
    }

    # ── Optional: corrector (ASBS) ──
    if "corrector" in cfg and "corrector" in checkpoint:
        corrector = hydra.utils.instantiate(cfg.corrector).to(device)
        corrector.load_state_dict(checkpoint["corrector"])
        corrector.eval()
        result["corrector"] = corrector

    return result


def generate_samples(
    model_dict: dict,
    n_samples: int,
    device: str = "cuda",
    batch_size: int = 2000,
) -> torch.Tensor:
    """Generate terminal samples via forward SDE rollout.

    Args:
        model_dict: Output of load_model_from_checkpoint().
        n_samples: Total number of samples to generate.
        device: Device for computation.
        batch_size: Samples per batch (to avoid OOM).

    Returns:
        Tensor of shape (n_samples, D) on the given device.
    """
    sde = model_dict["sde"]
    source = model_dict["source"]
    cfg = model_dict["cfg"]

    sde.eval()

    x1_list = []
    n_generated = 0

    while n_generated < n_samples:
        B = min(batch_size, n_samples - n_generated)
        x0 = source.sample([B]).to(device)
        timesteps = train_utils.get_timesteps(**cfg.timesteps).to(device)

        with torch.no_grad():
            _, x1 = sdeint(sde, x0, timesteps, only_boundary=True)

        x1_list.append(x1)
        n_generated += B

    return torch.cat(x1_list, dim=0)[:n_samples]


def assign_modes_nearest(
    samples: torch.Tensor,
    mode_centers: torch.Tensor,
) -> torch.Tensor:
    """Assign each sample to the nearest mode center (Voronoi).

    Args:
        samples: (N, D) tensor of generated samples.
        mode_centers: (K, D) tensor of mode centers.

    Returns:
        (N,) integer tensor of mode assignments.
    """
    # Move to same device
    samples_dev = samples.to(mode_centers.device)
    dists = torch.cdist(
        samples_dev.unsqueeze(0),
        mode_centers.unsqueeze(0),
    ).squeeze(0)  # (N, K)
    return dists.argmin(dim=-1)  # (N,)


def compute_mode_weights(assignments: torch.Tensor, K: int) -> torch.Tensor:
    """Compute empirical mode weights alpha_k from assignments.

    Args:
        assignments: (N,) integer tensor of mode indices.
        K: Number of modes.

    Returns:
        (K,) float tensor with alpha_k = count_k / N.
    """
    alpha = torch.zeros(K, dtype=torch.float64)
    for k in range(K):
        alpha[k] = (assignments == k).sum().item()
    total = alpha.sum()
    if total > 0:
        alpha = alpha / total
    return alpha.float()


def kl_mode_weights(alpha: torch.Tensor, w: torch.Tensor) -> float:
    """Compute KL(alpha || w) between empirical and target mode weights.

    Matches the formula in train.py mode tracking.
    """
    a_safe = alpha.clamp(min=1e-10)
    w_safe = w.clamp(min=1e-10)
    return (a_safe * (a_safe.log() - w_safe.log())).sum().item()


def tv_mode_weights(alpha: torch.Tensor, w: torch.Tensor) -> float:
    """Compute total variation distance TV(alpha, w)."""
    return 0.5 * (alpha - w).abs().sum().item()


def count_alive_modes(alpha: torch.Tensor, w: torch.Tensor) -> int:
    """Count modes with alpha_k > 0.1 * w_k (alive threshold).

    Matches the threshold used in train.py mode tracking.
    """
    return int((alpha > 0.1 * w).sum().item())
