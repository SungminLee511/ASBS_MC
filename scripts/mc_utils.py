"""
Mode Concentration utilities — shared across all experiment scripts.

Provides:
  - Mode assignment (nearest center or gradient-flow basin)
  - Mode weight computation (α_k)
  - Distributional metrics (KL, TV on mode weights)
  - Tracking file I/O
"""

import json
import numpy as np
import torch
from pathlib import Path
from typing import Optional, List, Dict


# ──────────────────────────────────────────────────────────────────────
# Mode assignment
# ──────────────────────────────────────────────────────────────────────

def assign_modes_nearest(
    samples: torch.Tensor,
    mode_centers: torch.Tensor,
) -> torch.Tensor:
    """Assign each sample to the nearest mode center.

    Args:
        samples: (N, D) tensor of terminal positions
        mode_centers: (K, D) tensor of mode centers

    Returns:
        assignments: (N,) LongTensor of mode indices in [0, K)
    """
    # (N, 1, D) - (1, K, D) → (N, K, D) → (N, K)
    dists = torch.cdist(samples.unsqueeze(0), mode_centers.unsqueeze(0)).squeeze(0)
    return dists.argmin(dim=-1)


def assign_modes_gradient_flow(
    samples: torch.Tensor,
    energy_fn,
    mode_centers: torch.Tensor,
    step_size: float = 0.01,
    max_steps: int = 1000,
    tol: float = 1e-6,
) -> torch.Tensor:
    """Assign each sample to its gradient-flow basin.

    Runs gradient descent x ← x - lr * ∇E(x) until convergence,
    then assigns to the nearest mode center.

    Args:
        samples: (N, D) terminal samples
        energy_fn: callable with .grad_E(x) method
        mode_centers: (K, D) mode centers
        step_size: gradient descent step size
        max_steps: maximum number of GD steps
        tol: convergence tolerance (max gradient norm)

    Returns:
        assignments: (N,) LongTensor
    """
    x = samples.clone().detach()
    for _ in range(max_steps):
        grad = energy_fn.grad_E(x)
        x = x - step_size * grad
        if grad.norm(dim=-1).max().item() < tol:
            break
    return assign_modes_nearest(x, mode_centers)


# ──────────────────────────────────────────────────────────────────────
# Mode weights and metrics
# ──────────────────────────────────────────────────────────────────────

def compute_mode_weights(
    assignments: torch.Tensor,
    K: int,
) -> torch.Tensor:
    """Compute empirical mode weights α_k = count_k / N.

    Args:
        assignments: (N,) LongTensor of mode indices in [0, K)
        K: number of modes

    Returns:
        alpha: (K,) tensor of mode weights summing to 1
    """
    counts = torch.zeros(K, dtype=torch.float64)
    for k in range(K):
        counts[k] = (assignments == k).sum().item()
    return (counts / counts.sum()).float()


def kl_mode_weights(alpha: torch.Tensor, w: torch.Tensor, eps: float = 1e-10) -> float:
    """KL(α || w) = Σ_k α_k log(α_k / w_k).

    Args:
        alpha: (K,) empirical mode weights
        w: (K,) target mode weights

    Returns:
        KL divergence (scalar)
    """
    alpha_safe = alpha.clamp(min=eps)
    w_safe = w.clamp(min=eps)
    return (alpha_safe * (alpha_safe.log() - w_safe.log())).sum().item()


def tv_mode_weights(alpha: torch.Tensor, w: torch.Tensor) -> float:
    """TV(α, w) = (1/2) Σ_k |α_k - w_k|.

    Args:
        alpha: (K,) empirical mode weights
        w: (K,) target mode weights

    Returns:
        Total variation distance (scalar)
    """
    return 0.5 * (alpha - w).abs().sum().item()


def count_alive_modes(
    alpha: torch.Tensor,
    w: torch.Tensor,
    threshold: float = 0.1,
) -> int:
    """Count modes with α_k > threshold * w_k.

    Args:
        alpha: (K,) empirical mode weights
        w: (K,) target mode weights
        threshold: relative threshold (e.g., 0.1 means 10% of target)

    Returns:
        Number of alive modes
    """
    return int((alpha > threshold * w).sum().item())


# ──────────────────────────────────────────────────────────────────────
# Tracking file I/O
# ──────────────────────────────────────────────────────────────────────

def load_tracking(path: str | Path) -> List[Dict]:
    """Load a mode tracking JSON file.

    Each line is a JSON object with keys:
        epoch, stage, loss, alpha_k (list), kl, tv, alive_modes

    Returns:
        List of dicts, one per logged epoch.
    """
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_tracking_record(path: str | Path, record: Dict):
    """Append a single tracking record (JSON line) to the file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


# ──────────────────────────────────────────────────────────────────────
# Helpers for finding run directories
# ──────────────────────────────────────────────────────────────────────

def find_tracking_files(results_dir: str | Path, pattern: str = "mode_tracking.jsonl") -> List[Path]:
    """Recursively find all mode tracking files under a results directory."""
    results_dir = Path(results_dir)
    return sorted(results_dir.rglob(pattern))


def get_run_dir(base_dir: str | Path, exp_name: str, seed: int) -> Path:
    """Construct the Hydra output directory for a given experiment+seed.

    Hydra default output pattern: outputs/<date>/<time>/
    But we override with results/<exp_name>/seed_<seed>/
    """
    return Path(base_dir) / "results" / exp_name / f"seed_{seed}"


# ──────────────────────────────────────────────────────────────────────
# Checkpoint loading and model reconstruction
# ──────────────────────────────────────────────────────────────────────

def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: str = "cuda",
):
    """Load a trained model from a checkpoint file.

    Returns:
        dict with keys: sde, controller, energy, source, cfg, epoch
    """
    import hydra
    from omegaconf import OmegaConf
    from adjoint_samplers.components.sde import ControlledSDE

    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint["cfg"]

    # Reconstruct components
    energy = hydra.utils.instantiate(cfg.energy, device=device)
    source = hydra.utils.instantiate(cfg.source, device=device)
    ref_sde = hydra.utils.instantiate(cfg.ref_sde).to(device)
    controller = hydra.utils.instantiate(cfg.controller).to(device)
    controller.load_state_dict(checkpoint["controller"])
    controller.eval()

    sde = ControlledSDE(ref_sde, controller).to(device)

    return {
        "sde": sde,
        "controller": controller,
        "ref_sde": ref_sde,
        "energy": energy,
        "source": source,
        "cfg": cfg,
        "epoch": checkpoint["epoch"],
    }


def generate_samples(model_dict, n_samples=10000, batch_size=2000, device="cuda"):
    """Generate terminal samples from a loaded model.

    Returns:
        x1: (N, D) tensor of terminal samples
    """
    import adjoint_samplers.utils.train_utils as train_utils
    from adjoint_samplers.components.sde import sdeint

    sde = model_dict["sde"]
    source = model_dict["source"]
    cfg = model_dict["cfg"]

    x1_list = []
    n_gen = 0
    while n_gen < n_samples:
        B = min(batch_size, n_samples - n_gen)
        x0 = source.sample([B,]).to(device)
        timesteps = train_utils.get_timesteps(**cfg.timesteps).to(x0)
        with torch.no_grad():
            _, x1 = sdeint(sde, x0, timesteps, only_boundary=True)
        x1_list.append(x1)
        n_gen += B

    return torch.cat(x1_list, dim=0)


def generate_trajectories(model_dict, n_particles=200, device="cuda"):
    """Generate full trajectories from a loaded model.

    Returns:
        trajectories: list of (N, D) tensors, one per timestep
    """
    import adjoint_samplers.utils.train_utils as train_utils
    from adjoint_samplers.components.sde import sdeint

    sde = model_dict["sde"]
    source = model_dict["source"]
    cfg = model_dict["cfg"]

    x0 = source.sample([n_particles,]).to(device)
    timesteps = train_utils.get_timesteps(**cfg.timesteps).to(x0)
    with torch.no_grad():
        states = sdeint(sde, x0, timesteps, only_boundary=False)

    return states, timesteps


def assign_lj3_permutations(
    samples: torch.Tensor,
    n_particles: int = 3,
    spatial_dim: int = 2,
) -> torch.Tensor:
    """Assign LJ3 samples to one of 6 permutational basins.

    For 3 particles in 2D, the 6 permutations of S_3 produce 6 distinct
    but energetically equivalent configurations. We assign each sample
    to the permutation that best matches a canonical ordering
    (sorted by the first coordinate of each particle).

    Returns:
        assignments: (N,) LongTensor with values in [0, 5]
    """
    import itertools
    N = samples.shape[0]
    x = samples.reshape(N, n_particles, spatial_dim)

    # All 6 permutations of 3 particles
    perms = list(itertools.permutations(range(n_particles)))

    # Canonical form: sort particles by x-coordinate, then y
    # For each sample, find which permutation maps it closest to canonical
    # Actually: just identify which permutation of particle indices gives
    # the sorted order — this directly labels the basin.

    assignments = torch.zeros(N, dtype=torch.long, device=samples.device)
    for i in range(N):
        # Sort particle indices by (x, y)
        coords = x[i]  # (3, 2)
        # Sort by first coord, break ties by second
        sort_key = coords[:, 0] * 1000 + coords[:, 1]
        sorted_indices = sort_key.argsort().tolist()
        # Find which permutation this corresponds to
        for p_idx, perm in enumerate(perms):
            if list(perm) == sorted_indices:
                assignments[i] = p_idx
                break

    return assignments


def find_checkpoints(run_dir: str | Path) -> List[Path]:
    """Find all checkpoint files in a run directory, sorted by epoch."""
    run_dir = Path(run_dir)
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return []
    ckpts = sorted(ckpt_dir.glob("checkpoint_*.pt"))
    # Filter out 'latest'
    ckpts = [c for c in ckpts if "latest" not in c.name]
    return ckpts
