#!/usr/bin/env python
"""
v3 Family F1 (Dead Mode Adjoint Measurement) & F2 (Adjoint Sensitivity).

F1 — Dead Mode Conditional Adjoint:
    At a pretrained collapsed state (e.g. B7), the dead mode k has alpha_k ~ 0.
    For trajectories that happen to land in the dead mode's Voronoi cell, compute
    the conditional adjoint:
        Y_bar_k = E[ -grad Phi_0(X_1) | X_1 in M_k ]
    using Monte Carlo over a large number of forward SDE samples.

F2 — Adjoint Sensitivity (BRA test):
    Perturb the controller weights by N(0, sigma^2 I) for several sigma values.
    The perturbation changes WHERE samples land (different x1 positions), but
    grad_term_cost itself depends on x1 position and corrector/energy — not
    directly on the controller. Measure how Y_bar_k changes and compute
    ||Delta Y_bar_k|| / sigma to test the Bridge Regularity Assumption (BRA).

Usage:
    python scripts/measure_dead_adjoints.py \
        --ckpt results/v3/baselines/b7/seed_0/checkpoints/checkpoint_1990.pt \
        --n-samples 1000000 \
        --batch-size 5000 \
        --perturb-sigmas 0.001,0.01,0.1 \
        --output results/v3/family_f/seed_0/dead_adjoint_results.json
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch
import numpy as np

# ── Project root on sys.path ──
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import hydra
from omegaconf import OmegaConf

from adjoint_samplers.components.sde import ControlledSDE, sdeint
import adjoint_samplers.utils.train_utils as train_utils

from mc_utils import (
    load_model_from_checkpoint,
    generate_samples,
    assign_modes_nearest,
    compute_mode_weights,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_grad_term_cost(cfg, energy, corrector, ref_sde, source):
    """Instantiate grad_term_cost from Hydra config, matching train.py logic."""
    return hydra.utils.instantiate(
        cfg.term_cost,
        corrector=corrector,
        energy=energy,
        ref_sde=ref_sde,
        source=source,
    )


def compute_adjoints_batched(grad_term_cost, x1_all, batch_size, device):
    """Compute adjoint1 = grad_term_cost(x1) in batches.

    Args:
        grad_term_cost: Callable that maps x1 -> adjoint vector.
        x1_all: (N, D) tensor of terminal samples.
        batch_size: Process this many samples at a time.
        device: Torch device.

    Returns:
        (N, D) tensor of adjoint vectors.
    """
    adj_list = []
    N = x1_all.shape[0]
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        x1_batch = x1_all[start:end].to(device)
        x1_batch.requires_grad_(True)
        adj = grad_term_cost(x1_batch)
        adj_list.append(adj.detach())
    return torch.cat(adj_list, dim=0)


def per_mode_adjoint_stats(adjoints, assignments, K):
    """Compute per-mode adjoint statistics.

    Args:
        adjoints: (N, D) tensor.
        assignments: (N,) integer tensor of mode indices.
        K: Number of modes.

    Returns:
        Dict mapping mode index -> stats dict.
    """
    results = {}
    for k in range(K):
        mask = (assignments == k)
        count = int(mask.sum().item())
        if count == 0:
            D = adjoints.shape[1]
            results[f"mode_{k}"] = {
                "count": 0,
                "adjoint_mean": [0.0] * D,
                "adjoint_std": [0.0] * D,
                "adjoint_norm": 0.0,
            }
            continue

        adj_k = adjoints[mask].float()  # (count, D)
        mean_k = adj_k.mean(dim=0)  # (D,)
        std_k = adj_k.std(dim=0) if count > 1 else torch.zeros_like(mean_k)
        norm_k = mean_k.norm().item()

        results[f"mode_{k}"] = {
            "count": count,
            "adjoint_mean": mean_k.tolist(),
            "adjoint_std": std_k.tolist(),
            "adjoint_norm": norm_k,
        }
    return results


def perturb_controller_weights(controller, sigma):
    """Return a deep copy of the controller with N(0, sigma^2 I) noise added.

    Only floating-point parameters are perturbed.
    """
    perturbed = copy.deepcopy(controller)
    with torch.no_grad():
        for param in perturbed.parameters():
            if param.is_floating_point():
                param.add_(sigma * torch.randn_like(param))
    perturbed.eval()
    return perturbed


def rebuild_sde_and_grad_term_cost(cfg, ref_sde, perturbed_controller, corrector,
                                    energy, source, device):
    """Build a new ControlledSDE and grad_term_cost with a perturbed controller.

    The corrector and energy are reused as-is (only the controller changed).
    """
    sde_new = ControlledSDE(ref_sde, perturbed_controller).to(device)
    sde_new.eval()
    grad_tc = build_grad_term_cost(cfg, energy, corrector, ref_sde, source)
    return sde_new, grad_tc


def generate_samples_with_sde(sde, source, cfg, n_samples, batch_size, device):
    """Forward SDE rollout using an arbitrary sde (not from model_dict).

    Mirrors mc_utils.generate_samples but accepts an explicit sde object.
    """
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


# ---------------------------------------------------------------------------
# F1: Baseline dead-mode adjoint measurement
# ---------------------------------------------------------------------------

def run_f1(model_dict, n_samples, batch_size, device):
    """Run Family F1: conditional adjoint measurement per mode.

    Returns:
        Dict with per-mode adjoint stats.
    """
    print(f"\n{'='*60}")
    print("F1: Dead Mode Adjoint Measurement")
    print(f"{'='*60}")

    cfg = model_dict["cfg"]
    energy = model_dict["energy"]
    corrector = model_dict.get("corrector", None)
    ref_sde = model_dict["ref_sde"]
    source = model_dict["source"]

    mode_centers = energy.mode_centers  # (K, D)
    mode_weights = energy.mode_weights  # (K,)
    K = mode_centers.shape[0]

    print(f"Target has {K} modes, target weights: {mode_weights.tolist()}")

    # 1. Build grad_term_cost
    grad_tc = build_grad_term_cost(cfg, energy, corrector, ref_sde, source)

    # 2. Generate samples
    print(f"Generating {n_samples} samples (batch_size={batch_size})...")
    t0 = time.time()
    x1_all = generate_samples(model_dict, n_samples, device=device,
                              batch_size=batch_size)
    print(f"  Generated in {time.time() - t0:.1f}s, shape={x1_all.shape}")

    # 3. Compute adjoints
    print("Computing adjoint vectors...")
    t0 = time.time()
    adj_all = compute_adjoints_batched(grad_tc, x1_all, batch_size, device)
    print(f"  Computed in {time.time() - t0:.1f}s, shape={adj_all.shape}")

    # 4. Assign modes
    assignments = assign_modes_nearest(x1_all, mode_centers.to(device))
    alpha = compute_mode_weights(assignments, K)
    print(f"Empirical mode weights: {alpha.tolist()}")

    # 5. Per-mode stats
    stats = per_mode_adjoint_stats(adj_all, assignments, K)

    for k in range(K):
        mk = stats[f"mode_{k}"]
        print(f"  Mode {k}: count={mk['count']:>8d}, "
              f"||Y_bar||={mk['adjoint_norm']:.6f}, "
              f"alpha={alpha[k]:.6f}, "
              f"target_w={mode_weights[k]:.4f}")

    return stats


# ---------------------------------------------------------------------------
# F2: Adjoint sensitivity under controller perturbation
# ---------------------------------------------------------------------------

def run_f2(model_dict, baseline_stats, n_samples, batch_size, sigmas, device):
    """Run Family F2: adjoint sensitivity measurement.

    Args:
        model_dict: Loaded model dict.
        baseline_stats: F1 baseline results (from run_f1).
        n_samples: Number of samples per perturbation level.
        batch_size: Batch size for generation/adjoint computation.
        sigmas: List of perturbation magnitudes.
        device: Torch device string.

    Returns:
        Dict mapping sigma -> per-mode sensitivity stats.
    """
    print(f"\n{'='*60}")
    print("F2: Adjoint Sensitivity (BRA Test)")
    print(f"{'='*60}")

    cfg = model_dict["cfg"]
    energy = model_dict["energy"]
    corrector = model_dict.get("corrector", None)
    ref_sde = model_dict["ref_sde"]
    source = model_dict["source"]
    controller = model_dict["controller"]

    mode_centers = energy.mode_centers.to(device)
    K = mode_centers.shape[0]

    # Baseline adjoint means (as tensors for delta computation)
    baseline_means = {}
    for k in range(K):
        mk = baseline_stats[f"mode_{k}"]
        baseline_means[k] = torch.tensor(mk["adjoint_mean"], device=device)

    sensitivity_results = {}

    for sigma in sigmas:
        print(f"\n--- sigma = {sigma} ---")

        # 1. Perturb controller
        perturbed_ctrl = perturb_controller_weights(controller, sigma)
        sde_pert, grad_tc_pert = rebuild_sde_and_grad_term_cost(
            cfg, ref_sde, perturbed_ctrl, corrector, energy, source, device
        )

        # 2. Generate samples with perturbed SDE
        print(f"  Generating {n_samples} samples with perturbed controller...")
        t0 = time.time()
        x1_pert = generate_samples_with_sde(
            sde_pert, source, cfg, n_samples, batch_size, device
        )
        print(f"  Generated in {time.time() - t0:.1f}s")

        # 3. Compute adjoints at perturbed terminal positions
        print("  Computing perturbed adjoints...")
        t0 = time.time()
        adj_pert = compute_adjoints_batched(grad_tc_pert, x1_pert, batch_size, device)
        print(f"  Computed in {time.time() - t0:.1f}s")

        # 4. Assign modes and compute per-mode stats
        assignments_pert = assign_modes_nearest(x1_pert, mode_centers)
        pert_stats = per_mode_adjoint_stats(adj_pert, assignments_pert, K)

        # 5. Compute sensitivity: ||Delta Y_bar_k|| / sigma
        sigma_result = {}
        for k in range(K):
            mp = pert_stats[f"mode_{k}"]
            pert_mean = torch.tensor(mp["adjoint_mean"], device=device)
            delta = pert_mean - baseline_means[k]
            delta_norm = delta.norm().item()
            sensitivity = delta_norm / sigma if sigma > 0 else float("inf")

            sigma_result[f"mode_{k}"] = {
                "count_perturbed": mp["count"],
                "adjoint_mean_perturbed": mp["adjoint_mean"],
                "adjoint_norm_perturbed": mp["adjoint_norm"],
                "delta_adjoint": delta.tolist(),
                "delta_adjoint_norm": delta_norm,
                "sensitivity": sensitivity,
            }
            print(f"  Mode {k}: count={mp['count']:>8d}, "
                  f"||Delta Y||={delta_norm:.6f}, "
                  f"sensitivity={sensitivity:.6f}")

        sensitivity_results[f"sigma_{sigma}"] = sigma_result

        # Free GPU memory from perturbed model
        del sde_pert, grad_tc_pert, perturbed_ctrl, x1_pert, adj_pert
        torch.cuda.empty_cache()

    return sensitivity_results


def assess_bra(sensitivity_results, K):
    """Assess BRA verdict: 'bounded' if sensitivity stays O(1), else 'potentially_diverging'.

    Heuristic: if for any mode, sensitivity grows super-linearly across sigma levels
    (i.e. sensitivity at smallest sigma >> sensitivity at largest sigma by a large factor),
    flag as potentially diverging.

    More precisely: if max sensitivity across all (mode, sigma) pairs exceeds 1000,
    or if sensitivity increases by >10x as sigma decreases by 10x, flag divergence.
    """
    sigmas_sorted = sorted(sensitivity_results.keys(),
                           key=lambda s: float(s.split("_")[1]))

    max_sensitivity = 0.0
    for sigma_key in sigmas_sorted:
        for k in range(K):
            s = sensitivity_results[sigma_key][f"mode_{k}"]["sensitivity"]
            max_sensitivity = max(max_sensitivity, s)

    # Check for divergence pattern: sensitivity increasing as sigma -> 0
    if len(sigmas_sorted) >= 2:
        smallest_sigma_key = sigmas_sorted[0]
        largest_sigma_key = sigmas_sorted[-1]
        for k in range(K):
            s_small = sensitivity_results[smallest_sigma_key][f"mode_{k}"]["sensitivity"]
            s_large = sensitivity_results[largest_sigma_key][f"mode_{k}"]["sensitivity"]
            if s_large > 0 and s_small / s_large > 10:
                return "potentially_diverging"

    if max_sensitivity > 1000:
        return "potentially_diverging"

    return "bounded"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="v3 Family F1/F2: Dead mode adjoint measurement & sensitivity"
    )
    parser.add_argument("--ckpt", required=True,
                        help="Path to converged checkpoint (e.g. B7)")
    parser.add_argument("--n-samples", type=int, default=1_000_000,
                        help="Number of MC samples for adjoint estimation")
    parser.add_argument("--batch-size", type=int, default=5000,
                        help="Batch size for sample generation and adjoint computation")
    parser.add_argument("--perturb-sigmas", type=str, default="0.001,0.01,0.1",
                        help="Comma-separated perturbation sigmas for F2")
    parser.add_argument("--output", type=str, required=True,
                        help="Path for output JSON")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Torch device (default: cuda)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    # Parse sigmas
    sigmas = [float(s.strip()) for s in args.perturb_sigmas.split(",")]

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    # ── Load model ──
    print(f"Loading checkpoint: {args.ckpt}")
    model_dict = load_model_from_checkpoint(args.ckpt, device=device)
    print(f"  Epoch: {model_dict['epoch']}")
    has_corrector = "corrector" in model_dict
    print(f"  Corrector: {'yes (ASBS)' if has_corrector else 'no (AS)'}")

    energy = model_dict["energy"]
    K = energy.mode_centers.shape[0]
    print(f"  Modes: K={K}, dim={energy.mode_centers.shape[1]}")

    # ── F1: Baseline ──
    f1_results = run_f1(model_dict, args.n_samples, args.batch_size, device)

    # ── F2: Sensitivity ──
    f2_results = run_f2(model_dict, f1_results, args.n_samples, args.batch_size,
                        sigmas, device)

    # ── BRA verdict ──
    verdict = assess_bra(f2_results, K)
    print(f"\n{'='*60}")
    print(f"BRA Verdict: {verdict}")
    print(f"{'='*60}")

    # ── Assemble and save output ──
    output = {
        "checkpoint": str(args.ckpt),
        "epoch": model_dict["epoch"],
        "n_samples": args.n_samples,
        "seed": args.seed,
        "has_corrector": has_corrector,
        "K": K,
        "f1_baseline": f1_results,
        "f2_sensitivity": f2_results,
        "bra_verdict": verdict,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
