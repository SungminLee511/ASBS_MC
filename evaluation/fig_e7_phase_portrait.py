"""
E7: Phase Portrait of Mode Weights

Figures:
  - fig_e7_b1_pitchfork.pdf: 30 seeds of α_1 vs epoch (B1 symmetric)
  - fig_e7_b7_ternary.pdf: trajectories on 2-simplex (B7)
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from mc_utils import load_tracking
from config import (
    RESULTS_DIR, FIGURES_DIR, apply_style,
    E7_B1_SEEDS, E7_B7_SEEDS, MODE_COLORS_3,
)


def bary_to_cart(a1, a2, a3):
    """Barycentric → Cartesian for equilateral triangle."""
    x = 0.5 * (2 * a3 + a1)
    y = (np.sqrt(3) / 2) * a1
    return x, y


def fig_b1_pitchfork():
    apply_style()
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    target_w1 = None
    n_plotted = 0
    for seed in range(E7_B1_SEEDS):
        records = load_tracking(RESULTS_DIR / "b1_asbs" / f"seed_{seed}" / "mode_tracking.jsonl")
        if not records:
            continue
        epochs = [r["epoch"] for r in records]
        alpha1 = [r["alpha"][0] for r in records]
        if target_w1 is None:
            target_w1 = records[0]["target_w"][0]
        ax.plot(epochs, alpha1, alpha=0.35, linewidth=0.8, color="steelblue")
        n_plotted += 1

    if target_w1 is not None:
        ax.axhline(y=target_w1, color="red", linestyle="--", linewidth=2,
                    label=f"Target $w_1$ = {target_w1:.2f}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("$\\alpha_1$ (mode 1 weight)")
    ax.set_title(f"Pitchfork Bifurcation — B1 Symmetric 2-Mode ({n_plotted} seeds)")
    ax.legend(fontsize=11)
    ax.set_ylim(-0.05, 1.05)

    out = FIGURES_DIR / "fig_e7_b1_pitchfork.pdf"
    fig.savefig(out)
    print(f"  Saved: {out}")
    plt.close(fig)


def fig_b7_ternary():
    apply_style()
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    # Triangle
    corners = np.array([bary_to_cart(1, 0, 0), bary_to_cart(0, 1, 0), bary_to_cart(0, 0, 1)])
    triangle = plt.Polygon(corners, fill=False, edgecolor="black", linewidth=1.5)
    ax.add_patch(triangle)
    labels = ["Mode 0\n(deep)", "Mode 1\n(deep)", "Mode 2\n(metastable)"]
    offsets = [(0, 0.04), (-0.06, -0.05), (0.06, -0.05)]
    for c, lbl, off in zip(corners, labels, offsets):
        ax.text(c[0] + off[0], c[1] + off[1], lbl, ha="center", fontsize=9)

    target_w = None
    cmap = plt.cm.viridis
    n_plotted = 0

    for seed in range(E7_B7_SEEDS):
        records = load_tracking(RESULTS_DIR / "b7_asbs" / f"seed_{seed}" / "mode_tracking.jsonl")
        if not records:
            continue
        alphas = np.array([r["alpha"] for r in records])
        epochs = np.array([r["epoch"] for r in records])
        if target_w is None:
            target_w = np.array(records[0]["target_w"])

        xs = [bary_to_cart(a[0], a[1], a[2])[0] for a in alphas]
        ys = [bary_to_cart(a[0], a[1], a[2])[1] for a in alphas]

        ax.scatter(xs, ys, c=epochs, cmap=cmap, s=6, alpha=0.5, zorder=2)
        ax.plot(xs, ys, alpha=0.2, linewidth=0.4, color="gray", zorder=1)
        n_plotted += 1

    if target_w is not None:
        tx, ty = bary_to_cart(*target_w)
        ax.scatter([tx], [ty], color="red", s=150, marker="*", zorder=5,
                   label=f"Target $w$")

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.15, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Simplex Flow — B7 Three-Well ({n_plotted} seeds)")
    ax.legend(fontsize=10, loc="upper right")

    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Epoch")

    out = FIGURES_DIR / "fig_e7_b7_ternary.pdf"
    fig.savefig(out)
    print(f"  Saved: {out}")
    plt.close(fig)


def generate_all():
    print("E7: Phase Portrait")
    fig_b1_pitchfork()
    fig_b7_ternary()
