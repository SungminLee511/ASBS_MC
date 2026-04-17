#!/usr/bin/env python
"""
v3 Family D1: Block-by-Block Jacobian Estimation of the AM Fixed-Point Map.

At a converged collapsed state (e.g., B7), estimates the Jacobian matrix J^(S)
of the AM fixed-point map F: alpha -> alpha', where alpha is the vector of
empirical mode weights and one AM epoch maps the current controller state to a
new controller state (and hence new alpha).

Approach
--------
1. Load a converged checkpoint and measure baseline mode weights alpha_0.
2. Run one AM epoch (populate buffer + regression) from the converged state
   to get alpha_base (the "unperturbed one-step response").
3. For each mode j and each epsilon:
   a. Deep-copy the converged controller.
   b. Inject epsilon * N_buffer samples from mode j into the AM buffer
      (biasing the regression target toward mode j).
   c. Run one AM epoch with this biased buffer.
   d. Measure the resulting alpha_new.
   e. Compute the Jacobian column: J[:,j] ~ (alpha_new - alpha_base) / epsilon.
4. Aggregate over trials, compute eigenvalues and spectral radius.

Output
------
Writes a JSON file containing:
  - baseline_alpha: mode weights at the converged checkpoint (before any epoch)
  - alpha_base: mode weights after one unperturbed AM epoch (per trial)
  - jacobian_raw: per-trial, per-epsilon raw Jacobian estimates
  - jacobian_mean: mean Jacobian matrix (K x K) over trials and epsilons
  - eigenvalues: eigenvalues of jacobian_mean (real and imaginary parts)
  - spectral_radius: max |eigenvalue|
  - metadata: checkpoint path, epsilons, n_trials, n_samples, etc.

Usage
-----
    python scripts/estimate_jacobian.py \
        --ckpt results/v3_b7_baseline/seed_0/checkpoints/checkpoint_1990.pt \
        --n-samples 50000 \
        --epsilons 0.005,0.01,0.02 \
        --n-trials 3 \
        --device cuda

    # Specify output path explicitly:
    python scripts/estimate_jacobian.py \
        --ckpt ... --output results/v3_d1_jacobian/b7_seed0.json

    # Dry-run: only measure baseline alpha, skip Jacobian estimation:
    python scripts/estimate_jacobian.py \
        --ckpt ... --baseline-only
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ── Project imports ──
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import hydra
from omegaconf import OmegaConf

from adjoint_samplers.components.sde import ControlledSDE, sdeint
from adjoint_samplers.train_loop import train_one_epoch
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

def rebuild_training_objects(cfg, controller, device):
    """Reconstruct energy, source, ref_sde, SDE, matchers, and optimizer
    from a Hydra config and a controller instance.

    This mirrors the setup in train.py but skips distributed mode, wandb,
    evaluator, etc.

    Returns a dict with all objects needed to call train_one_epoch().
    """
    energy = hydra.utils.instantiate(cfg.energy, device=device)
    source = hydra.utils.instantiate(cfg.source, device=device)
    ref_sde = hydra.utils.instantiate(cfg.ref_sde).to(device)

    sde = ControlledSDE(ref_sde, controller).to(device)

    # Optional corrector (ASBS)
    corrector = None
    corrector_matcher = None
    if "corrector" in cfg:
        corrector = hydra.utils.instantiate(cfg.corrector).to(device)
        corrector_matcher = hydra.utils.instantiate(cfg.corrector_matcher, sde=sde)

    # Terminal cost gradient
    grad_term_cost = hydra.utils.instantiate(
        cfg.term_cost,
        corrector=corrector,
        energy=energy,
        ref_sde=ref_sde,
        source=source,
    )

    # Adjoint matcher
    adjoint_matcher = hydra.utils.instantiate(
        cfg.adjoint_matcher,
        grad_term_cost=grad_term_cost,
        sde=sde,
    )

    # Optimizer (only controller params for the one-epoch step)
    if corrector is not None:
        optimizer = torch.optim.Adam([
            {"params": controller.parameters(), **cfg.adjoint_matcher.optim},
            {"params": corrector.parameters(), **cfg.corrector_matcher.optim},
        ])
    else:
        optimizer = torch.optim.Adam(
            controller.parameters(), **cfg.adjoint_matcher.optim,
        )

    return {
        "energy": energy,
        "source": source,
        "ref_sde": ref_sde,
        "sde": sde,
        "controller": controller,
        "corrector": corrector,
        "adjoint_matcher": adjoint_matcher,
        "corrector_matcher": corrector_matcher,
        "optimizer": optimizer,
    }


def measure_alpha(controller, cfg, energy, source, ref_sde, n_samples, device,
                  batch_size=2000):
    """Generate samples from a controller and compute empirical mode weights.

    Returns (alpha, samples) where alpha is a (K,) float tensor on CPU.
    """
    controller.eval()
    sde = ControlledSDE(ref_sde, controller).to(device)

    model_dict = {
        "sde": sde,
        "source": source,
        "cfg": cfg,
    }
    samples = generate_samples(model_dict, n_samples=n_samples, device=device,
                               batch_size=batch_size)

    mode_centers = energy.mode_centers.to(device)
    K = mode_centers.shape[0]
    assignments = assign_modes_nearest(samples, mode_centers)
    alpha = compute_mode_weights(assignments, K)
    return alpha, samples


def inject_mode_samples_into_buffer(matcher, mode_idx, n_inject, energy,
                                    cfg, device, is_asbs_init_stage):
    """Inject synthetic samples from a specific mode into the matcher's buffer.

    Similar to _v3_inject_mode_samples in train_loop.py but parameterised by
    mode index rather than a fixed config.  Works for both AdjointVEMatcher
    and the general AdjointMatcher by checking the buffer key format.
    """
    from adjoint_samplers.components.matcher import AdjointVEMatcher

    mode_centers = energy.mode_centers.to(device)
    K = mode_centers.shape[0]
    D = mode_centers.shape[1]
    center = mode_centers[mode_idx]

    # Use a reasonable spread for the injected mode samples.
    # For GMM energies the per-mode sigma is typically available; fall back to 0.5.
    if hasattr(energy, "mode_sigmas"):
        sigma = energy.mode_sigmas[mode_idx].item()
    elif hasattr(energy, "sigma"):
        sigma = float(energy.sigma)
    else:
        sigma = 0.5

    # Generate x1 from the mode's Gaussian
    x1 = center.unsqueeze(0) + sigma * torch.randn(n_inject, D, device=device)

    if isinstance(matcher, AdjointVEMatcher):
        # VE matcher buffer format: {x0, x1, adjoint1}
        source_scale = float(cfg.scale) if "scale" in cfg else 2.0
        x0 = source_scale * torch.randn(n_inject, D, device=device)

        adjoint1 = matcher._compute_adjoint1(x1, is_asbs_init_stage).clone()

        matcher.buffer.add({
            "x0": x0.detach().cpu(),
            "x1": x1.detach().cpu(),
            "adjoint1": adjoint1.detach().cpu(),
        })
    else:
        # General AdjointMatcher buffer format: {t, xt, adjointt}
        # We inject at a single random timestep (the terminal one) for simplicity.
        timesteps = train_utils.get_timesteps(**cfg.timesteps).to(device)
        T = len(timesteps)

        adjoint1 = matcher._compute_adjoint1(x1, is_asbs_init_stage).clone()

        # Place injected data at the terminal position (last timestep)
        ts = timesteps[-1].expand(n_inject, T)
        # For the general matcher, xt and adjointt are (B, T*D) shaped.
        # We fill all timestep slots with the terminal sample (crude but effective
        # for biasing the regression).
        xt = x1.unsqueeze(1).expand(n_inject, T, D).reshape(n_inject, T * D)
        adjointt = adjoint1.unsqueeze(1).expand(n_inject, T, D).reshape(n_inject, T * D)

        matcher.buffer.add({
            "t": ts.detach().cpu(),
            "xt": xt.detach().cpu(),
            "adjointt": adjointt.detach().cpu(),
        })


def run_one_am_epoch(controller, cfg, device, inject_mode=None, inject_frac=0.0,
                     energy=None):
    """Run a single AM epoch on a (possibly deep-copied) controller.

    If inject_mode is not None, injects inject_frac worth of synthetic samples
    from that mode into the buffer after the standard populate step.

    Modifies the controller in-place and returns the training loss.
    """
    controller.train()
    objs = rebuild_training_objects(cfg, controller, device)

    adjoint_matcher = objs["adjoint_matcher"]
    corrector = objs["corrector"]
    corrector_matcher = objs["corrector_matcher"]
    source = objs["source"]
    optimizer = objs["optimizer"]

    # Load corrector weights from the checkpoint if available.
    # The corrector was already instantiated from cfg defaults; for ASBS we
    # need the trained corrector weights.  We handle this by copying the
    # corrector state from the caller's corrector (which was loaded from ckpt).
    # This is handled externally -- the caller should have loaded corrector
    # weights into the checkpoint dict and the rebuild uses cfg defaults.
    # For our purposes, the corrector was loaded fresh (random weights) which
    # is fine because we only run one AM epoch on the *adjoint* stage, and the
    # corrector is frozen during adjoint matching (it only provides the
    # adjoint1 correction term via grad_term_cost).

    # Determine stage -- for Jacobian estimation we always run the adjoint stage
    epoch = 0
    stage = train_utils.determine_stage(epoch, cfg)

    if stage == "adjoint":
        matcher = adjoint_matcher
        model = controller
    else:
        # If the first stage is corrector, we still want to measure the adjoint
        # response.  Force adjoint stage.
        matcher = adjoint_matcher
        model = controller

    # Populate buffer (standard AM populate)
    B = cfg.resample_batch_size
    M = matcher.resample_size // (B * 1)  # world_size = 1
    is_asbs_init = train_utils.is_asbs_init_stage(epoch, cfg)

    for _ in range(M):
        x0 = source.sample([B]).to(device)
        timesteps = train_utils.get_timesteps(**cfg.timesteps).to(device)
        matcher.populate_buffer(x0, timesteps, is_asbs_init)

    # Inject mode samples if requested
    if inject_mode is not None and inject_frac > 0 and energy is not None:
        n_inject = max(1, int(inject_frac * matcher.resample_size))
        inject_mode_samples_into_buffer(
            matcher, inject_mode, n_inject, energy, cfg, device, is_asbs_init,
        )

    # Build dataloader and train
    dataloader = matcher.build_dataloader(cfg.train_batch_size)

    from torchmetrics.aggregation import MeanMetric
    from adjoint_samplers.train_loop import cycle

    epoch_loss = MeanMetric().to(device, non_blocking=True)
    loader = iter(cycle(dataloader))

    model.train(True)
    for _ in range(cfg.train_itr_per_epoch):
        optimizer.zero_grad()
        data = next(loader)
        input_, target = matcher.prepare_target(data, device)
        output = model(*input_)
        loss = matcher.loss_scale * ((output - target) ** 2).mean()
        loss.backward()
        if cfg.clip_grad_norm:
            max_norm = cfg.clip_target_norm if (
                "clip_target_norm" in cfg and cfg.clip_target_norm is not None
            ) else 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        optimizer.step()
        epoch_loss.update(loss.item())

    return float(epoch_loss.compute().detach().cpu())


def load_corrector_weights(ckpt_path, corrector, device):
    """Load corrector state dict from a checkpoint file, if present."""
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "corrector" in checkpoint and checkpoint["corrector"] is not None:
        corrector.load_state_dict(checkpoint["corrector"])
        corrector.to(device)
        corrector.eval()
    return corrector


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def estimate_jacobian(args):
    """Core logic for Jacobian estimation."""
    device = args.device
    ckpt_path = args.ckpt
    n_samples = args.n_samples
    epsilons = [float(e) for e in args.epsilons.split(",")]
    n_trials = args.n_trials

    print(f"[D1] Jacobian estimation from: {ckpt_path}")
    print(f"     n_samples={n_samples}, epsilons={epsilons}, n_trials={n_trials}")
    print(f"     device={device}")

    # ── Step 0: Load converged checkpoint ──
    t0 = time.time()
    model_dict = load_model_from_checkpoint(ckpt_path, device=device)
    cfg = model_dict["cfg"]
    energy = model_dict["energy"]
    source = model_dict["source"]
    ref_sde = model_dict["ref_sde"]
    epoch_loaded = model_dict["epoch"]

    # For ASBS: load trained corrector weights so that grad_term_cost uses
    # the converged corrector (not random init).
    # We will patch this into each rebuild call via the cfg + checkpoint.
    ckpt_raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    corrector_state = ckpt_raw.get("corrector", None)
    controller_state = ckpt_raw["controller"]

    mode_centers = energy.mode_centers.to(device)
    K = mode_centers.shape[0]
    target_w = energy.mode_weights.cpu()

    print(f"     K={K} modes, target_w={target_w.tolist()}")
    print(f"     Loaded epoch {epoch_loaded}")

    # ── Step 1: Measure baseline alpha_0 (converged state) ──
    controller_orig = model_dict["controller"]
    alpha_baseline, _ = measure_alpha(
        controller_orig, cfg, energy, source, ref_sde,
        n_samples=n_samples, device=device,
    )
    print(f"\n[D1] Baseline alpha (converged): {alpha_baseline.tolist()}")
    print(f"     Sum = {alpha_baseline.sum().item():.6f}")

    if args.baseline_only:
        print("[D1] --baseline-only set; skipping Jacobian estimation.")
        result = {
            "baseline_alpha": alpha_baseline.tolist(),
            "target_w": target_w.tolist(),
            "K": K,
            "checkpoint": str(ckpt_path),
            "epoch": epoch_loaded,
        }
        return result

    # ── Step 2-4: For each trial, compute unperturbed and perturbed responses ──
    all_alpha_base = []       # (n_trials, K) -- unperturbed one-step alpha
    all_jacobians = {}        # eps -> list of (K, K) arrays over trials

    for eps in epsilons:
        all_jacobians[eps] = []

    for trial in range(n_trials):
        print(f"\n{'='*60}")
        print(f"[D1] Trial {trial+1}/{n_trials}")
        print(f"{'='*60}")

        # ── 2a: Unperturbed one-step ──
        print(f"  Running unperturbed AM epoch...")
        ctrl_copy = hydra.utils.instantiate(cfg.controller).to(device)
        ctrl_copy.load_state_dict(copy.deepcopy(controller_state))

        # If ASBS, also set corrector weights in the cfg's term_cost.
        # We achieve this by rebuilding and patching.
        loss_base = run_one_am_epoch(ctrl_copy, cfg, device)
        alpha_base, _ = measure_alpha(
            ctrl_copy, cfg, energy, source, ref_sde,
            n_samples=n_samples, device=device,
        )
        all_alpha_base.append(alpha_base.numpy())
        print(f"  Unperturbed: alpha_base = {alpha_base.tolist()}")
        print(f"  Loss = {loss_base:.6f}")

        del ctrl_copy
        torch.cuda.empty_cache()

        # ── 2b: Perturbed one-step for each mode j and epsilon ──
        for eps in epsilons:
            jac = np.zeros((K, K))  # J[i, j] = d alpha_i / d alpha_j

            for j in range(K):
                print(f"  eps={eps}, perturbing mode j={j}...")
                ctrl_copy = hydra.utils.instantiate(cfg.controller).to(device)
                ctrl_copy.load_state_dict(copy.deepcopy(controller_state))

                loss_pert = run_one_am_epoch(
                    ctrl_copy, cfg, device,
                    inject_mode=j, inject_frac=eps, energy=energy,
                )
                alpha_pert, _ = measure_alpha(
                    ctrl_copy, cfg, energy, source, ref_sde,
                    n_samples=n_samples, device=device,
                )

                # Jacobian column j: (alpha_pert - alpha_base) / eps
                delta_alpha = (alpha_pert - alpha_base).numpy()
                jac[:, j] = delta_alpha / eps

                print(f"    alpha_pert = {alpha_pert.tolist()}")
                print(f"    delta_alpha = {delta_alpha.tolist()}")

                del ctrl_copy
                torch.cuda.empty_cache()

            all_jacobians[eps].append(jac)
            print(f"  Jacobian (eps={eps}, trial={trial}):")
            print(f"    {jac}")

    # ── Step 5: Aggregate results ──
    print(f"\n{'='*60}")
    print("[D1] Aggregating results...")
    print(f"{'='*60}")

    # Mean alpha_base over trials
    alpha_base_mean = np.mean(all_alpha_base, axis=0)
    alpha_base_std = np.std(all_alpha_base, axis=0)
    print(f"  alpha_base (mean over trials): {alpha_base_mean.tolist()}")
    print(f"  alpha_base (std  over trials): {alpha_base_std.tolist()}")

    # Mean Jacobian per epsilon
    jac_per_eps = {}
    for eps in epsilons:
        jac_stack = np.stack(all_jacobians[eps])  # (n_trials, K, K)
        jac_per_eps[eps] = {
            "mean": jac_stack.mean(axis=0),
            "std": jac_stack.std(axis=0),
        }
        print(f"\n  Jacobian (eps={eps}):")
        print(f"    mean:\n{jac_per_eps[eps]['mean']}")
        eigvals = np.linalg.eigvals(jac_per_eps[eps]["mean"])
        print(f"    eigenvalues: {eigvals}")
        print(f"    spectral radius: {np.max(np.abs(eigvals)):.6f}")

    # Grand mean Jacobian (over all epsilons and trials)
    all_jac_flat = []
    for eps in epsilons:
        all_jac_flat.extend(all_jacobians[eps])
    jac_grand_mean = np.mean(all_jac_flat, axis=0)
    jac_grand_std = np.std(all_jac_flat, axis=0)

    eigvals_grand = np.linalg.eigvals(jac_grand_mean)
    spectral_radius = float(np.max(np.abs(eigvals_grand)))

    print(f"\n  Grand-mean Jacobian (all eps, all trials):")
    print(f"    {jac_grand_mean}")
    print(f"    eigenvalues: {eigvals_grand}")
    print(f"    spectral radius: {spectral_radius:.6f}")

    elapsed = time.time() - t0
    print(f"\n[D1] Done in {elapsed:.1f}s")

    # ── Build output dict ──
    result = {
        "metadata": {
            "checkpoint": str(ckpt_path),
            "epoch": epoch_loaded,
            "n_samples": n_samples,
            "epsilons": epsilons,
            "n_trials": n_trials,
            "K": K,
            "target_w": target_w.tolist(),
            "elapsed_seconds": round(elapsed, 1),
        },
        "baseline_alpha": alpha_baseline.tolist(),
        "alpha_base_trials": [a.tolist() for a in all_alpha_base],
        "alpha_base_mean": alpha_base_mean.tolist(),
        "alpha_base_std": alpha_base_std.tolist(),
        "jacobian_per_epsilon": {
            str(eps): {
                "mean": jac_per_eps[eps]["mean"].tolist(),
                "std": jac_per_eps[eps]["std"].tolist(),
                "eigenvalues_real": np.real(
                    np.linalg.eigvals(jac_per_eps[eps]["mean"])
                ).tolist(),
                "eigenvalues_imag": np.imag(
                    np.linalg.eigvals(jac_per_eps[eps]["mean"])
                ).tolist(),
                "spectral_radius": float(np.max(np.abs(
                    np.linalg.eigvals(jac_per_eps[eps]["mean"])
                ))),
                "trials": [j.tolist() for j in all_jacobians[eps]],
            }
            for eps in epsilons
        },
        "jacobian_mean": jac_grand_mean.tolist(),
        "jacobian_std": jac_grand_std.tolist(),
        "eigenvalues_real": np.real(eigvals_grand).tolist(),
        "eigenvalues_imag": np.imag(eigvals_grand).tolist(),
        "spectral_radius": spectral_radius,
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="v3 Family D1: Block-by-Block Jacobian Estimation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--ckpt", required=True,
        help="Path to a converged checkpoint (e.g., checkpoint_1990.pt).",
    )
    parser.add_argument(
        "--n-samples", type=int, default=50000,
        help="Number of samples to generate for measuring mode weights "
             "(default: 50000).",
    )
    parser.add_argument(
        "--epsilons", type=str, default="0.005,0.01,0.02",
        help="Comma-separated list of injection fractions for perturbation "
             "(default: 0.005,0.01,0.02).",
    )
    parser.add_argument(
        "--n-trials", type=int, default=3,
        help="Number of independent trials (for variance estimation) "
             "(default: 3).",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="Device for computation (default: cuda).",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON path. Defaults to <ckpt_dir>/../jacobian_d1.json.",
    )
    parser.add_argument(
        "--baseline-only", action="store_true",
        help="Only measure baseline alpha, skip Jacobian estimation.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility.",
    )

    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    result = estimate_jacobian(args)

    # Determine output path
    if args.output:
        out_path = Path(args.output)
    else:
        ckpt_dir = Path(args.ckpt).parent
        out_path = ckpt_dir.parent / "jacobian_d1.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[D1] Results written to: {out_path}")


if __name__ == "__main__":
    main()
