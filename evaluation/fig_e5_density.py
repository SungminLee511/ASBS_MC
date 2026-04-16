"""
E5: Terminal Density Visualization

Figures:
  - fig_e5_{benchmark}_density.pdf: KDE contours vs target at multiple epochs
  - fig_e5_b6_scatter.pdf: B6 scatter mode (dot size ∝ α/w)
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
    compute_mode_weights,
    find_checkpoints,
)
from config import RESULTS_DIR, FIGURES_DIR, apply_style, VIZ_EPOCHS


def _target_density_grid(energy, xr, yr, gs=200, device="cpu"):
    gx = torch.linspace(xr[0], xr[1], gs, device=device)
    gy = torch.linspace(yr[0], yr[1], gs, device=device)
    GX, GY = torch.meshgrid(gx, gy, indexing="ij")
    grid = torch.stack([GX.reshape(-1), GY.reshape(-1)], dim=-1)
    with torch.no_grad():
        E = energy.eval(grid)
        lp = -E; lp = lp - lp.max()
        p = torch.exp(lp).reshape(gs, gs)
    return gx.cpu().numpy(), gy.cpu().numpy(), p.cpu().numpy()


def fig_density(benchmark, seed=0, epochs=None, n_samples=20000, scatter=False, device="cuda"):
    apply_style()
    run_dir = RESULTS_DIR / f"{benchmark}_asbs" / f"seed_{seed}"
    ckpts = {int(c.stem.split("_")[-1]): c for c in find_checkpoints(run_dir)}
    if not ckpts:
        print(f"  [skip] No checkpoints for {benchmark}")
        return

    epochs = epochs or VIZ_EPOCHS
    avail = sorted(ckpts.keys())
    matched = [(ep, min(avail, key=lambda e: abs(e - ep)), ckpts[min(avail, key=lambda e: abs(e - ep))]) for ep in epochs]

    ncols = len(matched)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5), squeeze=False)

    # Target density
    first_model = load_model_from_checkpoint(str(matched[0][2]), device=device)
    energy = first_model["energy"]
    if hasattr(energy, "mode_centers"):
        c = energy.mode_centers.cpu().numpy()
        margin = 4.0
        xr = (c[:, 0].min() - margin, c[:, 0].max() + margin)
        yr = (c[:, 1].min() - margin, c[:, 1].max() + margin)
    else:
        xr = yr = (-6, 6)

    gx, gy, tp = _target_density_grid(energy, xr, yr, device=device)

    for idx, (tep, aep, cp) in enumerate(matched):
        ax = axes[0, idx]
        model = load_model_from_checkpoint(str(cp), device=device)
        samples = generate_samples(model, n_samples=n_samples, device=device)
        s_np = samples.cpu().numpy()

        if scatter and hasattr(energy, "mode_centers") and hasattr(energy, "mode_weights"):
            mc = energy.mode_centers.to(device)
            mw = energy.mode_weights.to(device)
            K = mc.shape[0]
            asgn = assign_modes_nearest(samples, mc)
            alpha = compute_mode_weights(asgn, K)
            ratios = alpha / mw.cpu()
            c_np = mc.cpu().numpy()
            cmap = plt.cm.RdYlGn
            for k in range(K):
                r = ratios[k].item()
                ax.scatter(c_np[k, 0], c_np[k, 1], s=max(10, min(300, r * 100)),
                           c=[cmap(min(r, 2) / 2)], edgecolors="black", linewidths=0.5, zorder=3)
            ax.contour(gx, gy, tp.T, levels=10, colors="gray", alpha=0.4, linewidths=0.5)
        else:
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(s_np.T)
                gxd = np.linspace(xr[0], xr[1], 80)
                gyd = np.linspace(yr[0], yr[1], 80)
                GX, GY = np.meshgrid(gxd, gyd)
                Z = kde(np.vstack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
                ax.contourf(gxd, gyd, Z, levels=20, cmap="Blues", alpha=0.6)
            except Exception:
                ax.scatter(s_np[:, 0], s_np[:, 1], s=1, alpha=0.05, color="steelblue")
            ax.contour(gx, gy, tp.T, levels=10, colors="black", linewidths=0.8, alpha=0.7)

            if hasattr(energy, "mode_centers"):
                c_np = energy.mode_centers.cpu().numpy()
                ax.scatter(c_np[:, 0], c_np[:, 1], c="red", s=60, marker="*", zorder=5)

        ax.set_xlim(xr); ax.set_ylim(yr)
        ax.set_title(f"Epoch {aep}")
        ax.set_aspect("equal")

    suffix = "scatter" if scatter else "density"
    fig.suptitle(f"Terminal Density — {benchmark.upper()} (N={n_samples})", fontsize=14, y=1.02)
    out = FIGURES_DIR / f"fig_e5_{benchmark}_{suffix}.pdf"
    fig.savefig(out)
    print(f"  Saved: {out}")
    plt.close(fig)


def generate_all(device="cuda"):
    print("E5: Terminal Density")
    for bm in ["b1", "b3", "b7"]:
        fig_density(bm, device=device)
    fig_density("b6", scatter=True, device=device)
