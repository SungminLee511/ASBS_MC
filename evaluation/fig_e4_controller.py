"""
E4: Controller Field Comparison

Figures:
  - fig_e4_{benchmark}_field.pdf: learned vs oracle controller + deficiency
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from mc_utils import (
    load_model_from_checkpoint,
    generate_samples,
    assign_modes_nearest,
    find_checkpoints,
)
import adjoint_samplers.utils.train_utils as train_utils
from adjoint_samplers.components.sde import ControlledSDE, sdeint
from config import RESULTS_DIR, FIGURES_DIR, apply_style, VIZ_T_VALUES


def _eval_controller_grid(controller, grid_points, t_val, device):
    G = grid_points.shape[0]
    t = torch.full((G, 1), t_val, device=device)
    with torch.no_grad():
        return controller(t, grid_points)


def _oracle_kernel_regression(controller, ref_sde, energy, source, cfg,
                               grid_points, t_val, n_samples=5000, device="cuda"):
    """Importance-reweighted kernel regression for oracle controller."""
    sde = ControlledSDE(ref_sde, controller).to(device)
    x0 = source.sample([n_samples,]).to(device)
    timesteps = train_utils.get_timesteps(**cfg.timesteps).to(x0)

    with torch.no_grad():
        states = sdeint(sde, x0, timesteps, only_boundary=False)
    x1 = states[-1]
    t_idx = (timesteps - t_val).abs().argmin().item()
    x_t = states[t_idx]

    with torch.enable_grad():
        x1r = x1.clone().detach().requires_grad_(True)
        E = energy.eval(x1r)
        grad_E = torch.autograd.grad(E.sum(), x1r)[0]
    target = -grad_E.detach()

    # Importance weights
    iw = torch.ones(n_samples, device=device)
    if hasattr(energy, "mode_centers") and hasattr(energy, "mode_weights"):
        mc = energy.mode_centers.to(device)
        mw = energy.mode_weights.to(device)
        K = mc.shape[0]
        asgn = assign_modes_nearest(x1, mc)
        alpha = torch.zeros(K, device=device)
        for k in range(K):
            alpha[k] = (asgn == k).float().mean()
        alpha = alpha.clamp(min=1e-6)
        for k in range(K):
            iw[asgn == k] = mw[k] / alpha[k]
        iw = iw / iw.mean()

    G = grid_points.shape[0]
    bw = 0.5
    oracle = torch.zeros(G, 2, device=device)
    for s in range(0, G, 256):
        e = min(s + 256, G)
        d = torch.cdist(grid_points[s:e], x_t)
        Kv = torch.exp(-d**2 / (2 * bw**2))
        wK = Kv * iw.unsqueeze(0)
        num = (wK.unsqueeze(-1) * target.unsqueeze(0)).sum(dim=1)
        den = wK.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        oracle[s:e] = num / den

    g_t = ref_sde.diff(torch.tensor([t_val], device=device))
    return -g_t**2 * oracle


def fig_controller(benchmark, seed=0, t_val=0.5, grid_size=25, device="cuda"):
    apply_style()
    run_dir = RESULTS_DIR / f"{benchmark}_asbs" / f"seed_{seed}"
    ckpts = find_checkpoints(run_dir)
    if not ckpts:
        print(f"  [skip] No checkpoints for {benchmark}")
        return

    # Use the last checkpoint
    ckpt = ckpts[-1]
    model = load_model_from_checkpoint(str(ckpt), device=device)
    energy = model["energy"]
    epoch = model["epoch"]

    if hasattr(energy, "mode_centers"):
        c = energy.mode_centers.cpu().numpy()
        margin = 3.0
        xr = (c[:, 0].min() - margin, c[:, 0].max() + margin)
        yr = (c[:, 1].min() - margin, c[:, 1].max() + margin)
    else:
        xr = yr = (-5, 5)

    gx = torch.linspace(xr[0], xr[1], grid_size, device=device)
    gy = torch.linspace(yr[0], yr[1], grid_size, device=device)
    GX, GY = torch.meshgrid(gx, gy, indexing="ij")
    grid = torch.stack([GX.reshape(-1), GY.reshape(-1)], dim=-1)

    u_learned = _eval_controller_grid(model["controller"], grid, t_val, device).cpu().numpy()
    u_oracle = _oracle_kernel_regression(
        model["controller"], model["ref_sde"], energy, model["source"],
        model["cfg"], grid, t_val, device=device
    ).cpu().numpy()
    delta = u_learned - u_oracle

    gx_np, gy_np = gx.cpu().numpy(), gy.cpu().numpy()
    G = grid_size

    with torch.no_grad():
        E = energy.eval(grid).cpu().numpy().reshape(G, G)
    E_plot = np.clip(E, E.min(), np.percentile(E, 95))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    titles = ["Learned $u_\\theta$", "Oracle $u^*$", "Deficiency $\\Delta u$"]

    for col, (field, title) in enumerate(zip([u_learned, u_oracle, delta], titles)):
        ax = axes[col]
        ax.contour(gx_np, gy_np, E_plot.T, levels=12, colors="gray", alpha=0.3, linewidths=0.5)
        ux = field[:, 0].reshape(G, G)
        uy = field[:, 1].reshape(G, G)

        if col < 2:
            ax.quiver(gx_np, gy_np, ux.T, uy.T, alpha=0.7, width=0.004)
        else:
            mag = np.sqrt(ux**2 + uy**2)
            im = ax.pcolormesh(gx_np, gy_np, mag.T, cmap="Reds", alpha=0.6)
            ax.quiver(gx_np, gy_np, ux.T, uy.T, alpha=0.5, width=0.004)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="$|\\Delta u|$")

        if hasattr(energy, "mode_centers"):
            cnp = energy.mode_centers.cpu().numpy()
            ax.scatter(cnp[:, 0], cnp[:, 1], c="red", s=80, marker="*", zorder=5)
        ax.set_title(title)
        ax.set_aspect("equal")

    fig.suptitle(f"Controller Field — {benchmark.upper()} (epoch {epoch}, t={t_val})",
                 fontsize=13, y=1.02)
    out = FIGURES_DIR / f"fig_e4_{benchmark}_field.pdf"
    fig.savefig(out)
    print(f"  Saved: {out}")
    plt.close(fig)


def generate_all(device="cuda"):
    print("E4: Controller Field")
    for bm in ["b1", "b3", "b7"]:
        fig_controller(bm, device=device)
