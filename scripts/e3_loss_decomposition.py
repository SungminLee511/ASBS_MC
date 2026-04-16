#!/usr/bin/env python
"""
E3: AM Loss Decomposition

Decomposes the AM residual into intra-mode and inter-mode variance
(Theorem 1) at each checkpoint. Produces stacked area chart showing
that inter-mode fraction drops as modes collapse.

Usage:
    python scripts/e3_loss_decomposition.py \\
        --run-dir results/b5_asbs/seed_0 \\
        --output figures/e3_b5_decomposition.pdf
"""

import argparse
import sys
import json
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
    find_checkpoints,
)


def compute_loss_decomposition(
    checkpoint_path: str,
    n_samples: int = 5000,
    device: str = "cuda",
):
    """Compute V_intra and V_inter from Theorem 1 at a checkpoint.

    V_inter = Σ_k η_k |Y̅_k - Y̅|²
    V_intra = Σ_k η_k Var[Y₁ | M=k]

    where Y₁ = -∇Φ₀(X₁) is the adjoint terminal value.
    """
    import adjoint_samplers.utils.train_utils as train_utils
    from adjoint_samplers.components.sde import sdeint

    model = load_model_from_checkpoint(checkpoint_path, device=device)
    sde = model["sde"]
    energy = model["energy"]
    source = model["source"]
    cfg = model["cfg"]
    epoch = model["epoch"]

    if not (hasattr(energy, "mode_centers") and hasattr(energy, "mode_weights")):
        return None

    mode_centers = energy.mode_centers.to(device)
    K = mode_centers.shape[0]

    # Generate terminal samples
    x0 = source.sample([n_samples,]).to(device)
    timesteps = train_utils.get_timesteps(**cfg.timesteps).to(x0)
    with torch.no_grad():
        _, x1 = sdeint(sde, x0, timesteps, only_boundary=True)

    # Compute adjoint terminal value Y₁ = -∇E(x₁)  (simplified)
    with torch.enable_grad():
        x1_req = x1.clone().detach().requires_grad_(True)
        E = energy.eval(x1_req)
        grad_E = torch.autograd.grad(E.sum(), x1_req)[0]
    Y1 = -grad_E.detach()  # (N, D) — the adjoint terminal value

    # Assign modes
    assignments = assign_modes_nearest(x1, mode_centers)

    # Compute η_k (empirical regression coefficients)
    eta = torch.zeros(K, device=device)
    for k in range(K):
        eta[k] = (assignments == k).float().mean()

    # Y̅_k = E[Y₁ | M=k]  (conditional mean per mode)
    Y_bar_k = torch.zeros(K, Y1.shape[1], device=device)
    for k in range(K):
        mask = assignments == k
        if mask.sum() > 0:
            Y_bar_k[k] = Y1[mask].mean(dim=0)

    # Y̅ = Σ_k η_k Y̅_k  (overall mean)
    Y_bar = (eta.unsqueeze(1) * Y_bar_k).sum(dim=0)

    # V_inter = Σ_k η_k |Y̅_k - Y̅|²
    V_inter = 0.0
    for k in range(K):
        if eta[k] > 0:
            V_inter += eta[k] * ((Y_bar_k[k] - Y_bar) ** 2).sum()
    V_inter = V_inter.item()

    # V_intra = Σ_k η_k Var[Y₁ | M=k]
    V_intra = 0.0
    for k in range(K):
        mask = assignments == k
        if mask.sum() > 1:
            var_k = ((Y1[mask] - Y_bar_k[k]) ** 2).sum(dim=-1).mean()
            V_intra += eta[k].item() * var_k.item()

    return {
        "epoch": epoch,
        "V_inter": V_inter,
        "V_intra": V_intra,
        "V_total": V_inter + V_intra,
        "inter_fraction": V_inter / max(V_inter + V_intra, 1e-10),
        "eta": eta.cpu().tolist(),
    }


def analyze_run(run_dir: str, n_samples: int = 5000, device: str = "cuda"):
    """Compute loss decomposition at all checkpoints in a run."""
    ckpts = find_checkpoints(run_dir)
    if not ckpts:
        print(f"No checkpoints found in {run_dir}")
        return []

    results = []
    for ckpt in ckpts:
        print(f"  Processing {ckpt.name}...")
        try:
            r = compute_loss_decomposition(str(ckpt), n_samples, device)
            if r is not None:
                results.append(r)
        except Exception as e:
            print(f"    Error: {e}")
    return results


def plot_decomposition(
    run_dir: str,
    output_path: str = "figures/e3_decomposition.pdf",
    n_samples: int = 5000,
    device: str = "cuda",
    title: str = None,
):
    """Plot stacked area chart of V_intra + V_inter vs. epoch."""
    results = analyze_run(run_dir, n_samples, device)
    if not results:
        return

    epochs = [r["epoch"] for r in results]
    v_intra = [r["V_intra"] for r in results]
    v_inter = [r["V_inter"] for r in results]
    inter_frac = [r["inter_fraction"] for r in results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7),
                                     sharex=True,
                                     gridspec_kw={"hspace": 0.08, "height_ratios": [2, 1]})

    # Top: Stacked area
    ax1.fill_between(epochs, 0, v_intra, alpha=0.6, color="steelblue",
                      label="$V_{\\mathrm{intra}}$ (within-mode)")
    ax1.fill_between(epochs, v_intra, np.array(v_intra) + np.array(v_inter),
                      alpha=0.6, color="indianred",
                      label="$V_{\\mathrm{inter}}$ (between-mode)")
    ax1.set_ylabel("Irreducible AM variance", fontsize=12)
    if title:
        ax1.set_title(title, fontsize=14)
    else:
        ax1.set_title("E3: AM Loss Decomposition (Theorem 1)", fontsize=14)
    ax1.legend(fontsize=10, loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Bottom: Inter-mode fraction
    ax2.plot(epochs, inter_frac, "o-", color="darkred", linewidth=1.5, markersize=4)
    ax2.fill_between(epochs, inter_frac, alpha=0.2, color="indianred")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Inter-mode fraction", fontsize=12)
    ax2.set_ylim(-0.05, 1.05)
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax2.grid(True, alpha=0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"Saved: {output_path}")
    plt.close(fig)

    # Save raw data
    data_path = output_path.replace(".pdf", ".json")
    with open(data_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved data: {data_path}")


def main():
    parser = argparse.ArgumentParser(description="E3: AM Loss Decomposition")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", default="figures/e3_decomposition.pdf")
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--title", default=None)

    args = parser.parse_args()
    plot_decomposition(args.run_dir, args.output, args.n_samples,
                       args.device, args.title)


if __name__ == "__main__":
    main()
