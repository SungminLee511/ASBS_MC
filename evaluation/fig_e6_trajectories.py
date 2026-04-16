"""
E6: Particle Trajectory Visualization

Figures:
  - fig_e6_{benchmark}_traj.pdf: spaghetti plots at early/mid/late epochs
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from mc_utils import (
    load_model_from_checkpoint,
    generate_trajectories,
    assign_modes_nearest,
    find_checkpoints,
)
from config import RESULTS_DIR, FIGURES_DIR, apply_style, VIZ_EPOCHS


def fig_trajectories(benchmark, seed=0, epochs=None, n_particles=200, device="cuda"):
    apply_style()
    run_dir = RESULTS_DIR / f"{benchmark}_asbs" / f"seed_{seed}"
    ckpts = {int(c.stem.split("_")[-1]): c for c in find_checkpoints(run_dir)}
    if not ckpts:
        print(f"  [skip] No checkpoints for {benchmark}")
        return

    epochs = epochs or VIZ_EPOCHS
    avail = sorted(ckpts.keys())
    matched = [(ep, min(avail, key=lambda e: abs(e - ep))) for ep in epochs]
    matched = [(t, a, ckpts[a]) for t, a in matched]

    ncols = len(matched)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5), squeeze=False)

    for idx, (target_ep, actual_ep, ckpt_path) in enumerate(matched):
        ax = axes[0, idx]
        model = load_model_from_checkpoint(str(ckpt_path), device=device)
        energy = model["energy"]

        states, timesteps = generate_trajectories(model, n_particles=n_particles, device=device)
        x1 = states[-1]

        if hasattr(energy, "mode_centers"):
            mode_centers = energy.mode_centers.to(device)
            assignments = assign_modes_nearest(x1, mode_centers).cpu().numpy()
            K = mode_centers.shape[0]
        else:
            assignments = np.zeros(n_particles, dtype=int)
            K = 1

        cmap = plt.cm.tab10
        colors = [cmap(k / max(K - 1, 1)) for k in range(K)]

        n_steps = len(states)
        step_idx = np.linspace(0, n_steps - 1, min(50, n_steps), dtype=int)

        for i in range(n_particles):
            traj = torch.stack([states[s][i].cpu() for s in step_idx]).numpy()
            ax.plot(traj[:, 0], traj[:, 1], color=colors[assignments[i]],
                    alpha=0.12, linewidth=0.35)

        x1_np = x1.cpu().numpy()
        for k in range(K):
            mask = assignments == k
            ax.scatter(x1_np[mask, 0], x1_np[mask, 1], c=[colors[k]],
                       s=6, alpha=0.4, zorder=3)

        if hasattr(energy, "mode_centers"):
            c_np = mode_centers.cpu().numpy()
            ax.scatter(c_np[:, 0], c_np[:, 1], c="black", s=80, marker="*", zorder=5)

        ax.set_title(f"Epoch {actual_ep}")
        ax.set_aspect("equal")

    fig.suptitle(f"Particle Trajectories — {benchmark.upper()}", fontsize=14, y=1.02)
    out = FIGURES_DIR / f"fig_e6_{benchmark}_traj.pdf"
    fig.savefig(out)
    print(f"  Saved: {out}")
    plt.close(fig)


def generate_all(device="cuda"):
    print("E6: Trajectory Visualization")
    for bm in ["b1", "b3", "b7"]:
        fig_trajectories(bm, device=device)
