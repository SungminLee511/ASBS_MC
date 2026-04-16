"""
E3: AM Loss Decomposition

Figures:
  - fig_e3_{benchmark}_decomp.pdf: V_intra + V_inter stacked area
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from mc_utils import (
    load_model_from_checkpoint,
    assign_modes_nearest,
    find_checkpoints,
)
import adjoint_samplers.utils.train_utils as train_utils
from adjoint_samplers.components.sde import sdeint
from config import RESULTS_DIR, FIGURES_DIR, apply_style


def _decompose_at_checkpoint(ckpt_path, n_samples=5000, device="cuda"):
    model = load_model_from_checkpoint(ckpt_path, device=device)
    sde, energy, source, cfg = model["sde"], model["energy"], model["source"], model["cfg"]

    if not (hasattr(energy, "mode_centers") and hasattr(energy, "mode_weights")):
        return None

    mc = energy.mode_centers.to(device)
    K = mc.shape[0]

    x0 = source.sample([n_samples,]).to(device)
    ts = train_utils.get_timesteps(**cfg.timesteps).to(x0)
    with torch.no_grad():
        _, x1 = sdeint(sde, x0, ts, only_boundary=True)

    with torch.enable_grad():
        x1r = x1.clone().detach().requires_grad_(True)
        E = energy.eval(x1r)
        gE = torch.autograd.grad(E.sum(), x1r)[0]
    Y1 = -gE.detach()

    asgn = assign_modes_nearest(x1, mc)
    eta = torch.zeros(K, device=device)
    Y_bar_k = torch.zeros(K, Y1.shape[1], device=device)
    for k in range(K):
        m = asgn == k
        eta[k] = m.float().mean()
        if m.sum() > 0:
            Y_bar_k[k] = Y1[m].mean(0)

    Y_bar = (eta.unsqueeze(1) * Y_bar_k).sum(0)

    V_inter = sum(eta[k] * ((Y_bar_k[k] - Y_bar)**2).sum() for k in range(K) if eta[k] > 0).item()
    V_intra = 0.0
    for k in range(K):
        m = asgn == k
        if m.sum() > 1:
            V_intra += eta[k].item() * ((Y1[m] - Y_bar_k[k])**2).sum(-1).mean().item()

    return {"epoch": model["epoch"], "V_inter": V_inter, "V_intra": V_intra}


def fig_decomposition(benchmark, seed=0, n_samples=5000, device="cuda"):
    apply_style()
    ckpts = find_checkpoints(RESULTS_DIR / f"{benchmark}_asbs" / f"seed_{seed}")
    if not ckpts:
        print(f"  [skip] No checkpoints for {benchmark}")
        return

    results = []
    for c in ckpts:
        try:
            r = _decompose_at_checkpoint(str(c), n_samples, device)
            if r:
                results.append(r)
        except Exception as e:
            print(f"    Error at {c.name}: {e}")

    if not results:
        return

    epochs = [r["epoch"] for r in results]
    vi = np.array([r["V_intra"] for r in results])
    ve = np.array([r["V_inter"] for r in results])
    frac = ve / np.maximum(vi + ve, 1e-10)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                     height_ratios=[2, 1], gridspec_kw={"hspace": 0.08})

    ax1.fill_between(epochs, 0, vi, alpha=0.6, color="steelblue", label="$V_{\\mathrm{intra}}$")
    ax1.fill_between(epochs, vi, vi + ve, alpha=0.6, color="indianred", label="$V_{\\mathrm{inter}}$")
    ax1.set_ylabel("Irreducible AM variance")
    ax1.set_title(f"AM Loss Decomposition — {benchmark.upper()}")
    ax1.legend()

    ax2.plot(epochs, frac, "o-", color="darkred", markersize=4)
    ax2.fill_between(epochs, frac, alpha=0.2, color="indianred")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Inter-mode fraction")
    ax2.set_ylim(-0.05, 1.05)

    out = FIGURES_DIR / f"fig_e3_{benchmark}_decomp.pdf"
    fig.savefig(out)
    print(f"  Saved: {out}")
    plt.close(fig)


def generate_all(device="cuda"):
    print("E3: Loss Decomposition")
    for bm in ["b1", "b5", "b7"]:
        fig_decomposition(bm, device=device)
