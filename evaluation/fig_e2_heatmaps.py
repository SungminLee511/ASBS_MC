"""
E2: Regression Weight Heatmaps

Figures:
  - fig_e2_{benchmark}_eta.pdf: learned η_k vs oracle η_k* vs difference
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
from adjoint_samplers.components.sde import ControlledSDE, sdeint
from config import RESULTS_DIR, FIGURES_DIR, apply_style, VIZ_T_VALUES


def _estimate_eta_grid(model_dict, t_val, grid_points, n_sims=100, device="cuda"):
    sde = model_dict["sde"]
    energy = model_dict["energy"]
    cfg = model_dict["cfg"]
    mc = energy.mode_centers.to(device)
    K = mc.shape[0]
    G = grid_points.shape[0]

    ts = train_utils.get_timesteps(**cfg.timesteps).to(device)
    t_idx = (ts - t_val).abs().argmin().item()
    ts_from_t = ts[t_idx:]
    if len(ts_from_t) < 2:
        ts_from_t = ts[-2:]

    eta = torch.zeros(G, K, device=device)
    chunk = 32
    for s in range(0, G, chunk):
        e = min(s + chunk, G)
        n_pts = e - s
        x_rep = grid_points[s:e].unsqueeze(1).expand(-1, n_sims, -1).reshape(-1, 2)
        with torch.no_grad():
            _, x1 = sdeint(sde, x_rep, ts_from_t, only_boundary=True)
        asgn = assign_modes_nearest(x1, mc).reshape(n_pts, n_sims)
        for k in range(K):
            eta[s:e, k] = (asgn == k).float().mean(dim=1)
    return eta


def _oracle_eta(mc, mw, grid_points):
    K = mc.shape[0]
    dists = torch.cdist(mc.unsqueeze(0), mc.unsqueeze(0)).squeeze(0)
    dists.fill_diagonal_(float("inf"))
    bw = max(dists.min().item() / 2, 0.5)
    diff = grid_points.unsqueeze(1) - mc.unsqueeze(0)
    sq = (diff**2).sum(-1)
    log_q = -sq / (2 * bw**2)
    log_w = log_q + mw.log().unsqueeze(0)
    return torch.softmax(log_w, dim=-1)


def fig_heatmap(benchmark, seed=0, t_val=0.5, grid_size=35, n_sims=100, device="cuda"):
    apply_style()
    run_dir = RESULTS_DIR / f"{benchmark}_asbs" / f"seed_{seed}"
    ckpts = find_checkpoints(run_dir)
    if not ckpts:
        print(f"  [skip] No checkpoints for {benchmark}")
        return

    model = load_model_from_checkpoint(str(ckpts[-1]), device=device)
    energy = model["energy"]
    epoch = model["epoch"]

    if not (hasattr(energy, "mode_centers") and hasattr(energy, "mode_weights")):
        print(f"  [skip] {benchmark} has no mode_centers/weights")
        return

    mc = energy.mode_centers.to(device)
    mw = energy.mode_weights.to(device)
    K = mc.shape[0]
    c_np = mc.cpu().numpy()
    margin = 3.0
    xr = (c_np[:, 0].min() - margin, c_np[:, 0].max() + margin)
    yr = (c_np[:, 1].min() - margin, c_np[:, 1].max() + margin)

    gx = torch.linspace(xr[0], xr[1], grid_size, device=device)
    gy = torch.linspace(yr[0], yr[1], grid_size, device=device)
    GX, GY = torch.meshgrid(gx, gy, indexing="ij")
    grid = torch.stack([GX.reshape(-1), GY.reshape(-1)], dim=-1)

    # Pick minority mode
    minority_k = int(mw.cpu().argmin().item())

    print(f"  Estimating η on {grid_size}² grid for {benchmark} t={t_val}...")
    eta_learned = _estimate_eta_grid(model, t_val, grid, n_sims, device)
    eta_oracle = _oracle_eta(mc, mw, grid)

    el = eta_learned[:, minority_k].cpu().numpy().reshape(grid_size, grid_size)
    eo = eta_oracle[:, minority_k].cpu().numpy().reshape(grid_size, grid_size)
    diff = el - eo

    gx_np, gy_np = gx.cpu().numpy(), gy.cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), gridspec_kw={"wspace": 0.25})
    for ax, data, title, cmap_name, vrange in [
        (axes[0], el, f"Learned $\\eta_{minority_k}$", "YlOrRd", (0, 1)),
        (axes[1], eo, f"Oracle $\\eta_{minority_k}^*$", "YlOrRd", (0, 1)),
        (axes[2], diff, f"$\\Delta\\eta_{minority_k}$", "RdBu_r", None),
    ]:
        if vrange:
            im = ax.pcolormesh(gx_np, gy_np, data.T, cmap=cmap_name, vmin=vrange[0], vmax=vrange[1])
        else:
            vm = max(abs(diff.min()), abs(diff.max()), 0.1)
            im = ax.pcolormesh(gx_np, gy_np, data.T, cmap=cmap_name, vmin=-vm, vmax=vm)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.scatter(c_np[:, 0], c_np[:, 1], c="black", s=60, marker="*", zorder=5)
        ax.set_title(f"{title} (t={t_val})", fontsize=10)
        ax.set_aspect("equal")

    fig.suptitle(f"Regression Weight Heatmap — {benchmark.upper()} (epoch {epoch})", fontsize=13, y=1.02)
    out = FIGURES_DIR / f"fig_e2_{benchmark}_eta.pdf"
    fig.savefig(out)
    print(f"  Saved: {out}")
    plt.close(fig)


def generate_all(device="cuda"):
    print("E2: Regression Weight Heatmaps")
    for bm in ["b3", "b7"]:
        fig_heatmap(bm, device=device)
