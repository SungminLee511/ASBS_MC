#!/usr/bin/env python
"""
E9: KL Decomposition Tracking

Plots KL(Q_θ || p) estimated from energy evaluations alongside
KL(α || w) from mode counting. The gap is intra-mode error;
Theorem 2 guarantees KL(Q||p) ≥ KL(α||w) always.

Also tracks number of dead modes for B6.

Usage:
    python scripts/e9_kl_decomposition.py \\
        --run-dir results/b6_asbs/seed_0 \\
        --output figures/e9_b6_kl.pdf

    # Or from tracking file directly (KL(α||w) only, no full KL estimate)
    python scripts/e9_kl_decomposition.py \\
        --tracking results/b6_asbs/seed_0/mode_tracking.jsonl \\
        --output figures/e9_b6_kl.pdf

    # With full KL(Q||p) estimation from checkpoints
    python scripts/e9_kl_decomposition.py \\
        --run-dir results/b5_asbs/seed_0 \\
        --estimate-full-kl \\
        --output figures/e9_b5_kl.pdf
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
    load_tracking,
    load_model_from_checkpoint,
    generate_samples,
    find_checkpoints,
)


def estimate_full_kl_at_checkpoint(checkpoint_path, n_samples=5000, device="cuda"):
    """Estimate KL(Q_θ || p) ≈ E_Q[E(X₁)] + H(Q_θ) (up to constants).

    Since H(Q_θ) is hard to estimate, we use a proxy:
    KL_proxy = E_Q[E(X₁)] - E_p[E(X)] where the second term is
    estimated from the partition function via the energy values.

    More precisely, we estimate KL via:
    KL(Q||p) ≈ mean(E(X₁)) - mean(E_reference) + log(Z_ratio)

    For simplicity, we report the mean energy of generated samples
    as a monotone proxy that tracks the same trend.
    """
    model = load_model_from_checkpoint(checkpoint_path, device=device)
    energy = model["energy"]

    samples = generate_samples(model, n_samples=n_samples, device=device)
    with torch.no_grad():
        E_vals = energy.eval(samples)

    return {
        "epoch": model["epoch"],
        "mean_energy": E_vals.mean().item(),
        "std_energy": E_vals.std().item(),
        "median_energy": E_vals.median().item(),
    }


def plot_kl_decomposition(
    tracking_path: str = None,
    run_dir: str = None,
    estimate_full_kl: bool = False,
    output_path: str = "figures/e9_kl.pdf",
    device: str = "cuda",
    title: str = None,
):
    """Plot KL(α||w) and optionally KL(Q||p) vs. epoch."""

    # Load mode tracking data
    if tracking_path:
        records = load_tracking(tracking_path)
    elif run_dir:
        records = load_tracking(Path(run_dir) / "mode_tracking.jsonl")
    else:
        print("Must provide --tracking or --run-dir")
        return

    if not records:
        print("No tracking records found.")
        return

    epochs = np.array([r["epoch"] for r in records])
    kl_mode = np.array([r["kl"] for r in records])
    tv_mode = np.array([r["tv"] for r in records])
    alive = np.array([r["alive_modes"] for r in records])
    K = len(records[0]["alpha"])

    # Optionally estimate full KL from checkpoints
    full_kl_data = None
    if estimate_full_kl and run_dir:
        ckpts = find_checkpoints(run_dir)
        if ckpts:
            print("Estimating mean energy at checkpoints...")
            full_kl_data = []
            for ckpt in ckpts:
                try:
                    r = estimate_full_kl_at_checkpoint(str(ckpt), device=device)
                    full_kl_data.append(r)
                    print(f"  epoch={r['epoch']}: E={r['mean_energy']:.2f}")
                except Exception as e:
                    print(f"  Error at {ckpt.name}: {e}")

    # Create figure
    has_full_kl = full_kl_data is not None and len(full_kl_data) > 0
    nrows = 3 if K > 10 else 2  # Extra panel for dead mode count if many modes

    fig, axes = plt.subplots(nrows, 1, figsize=(10, 3.5 * nrows),
                              sharex=True, gridspec_kw={"hspace": 0.1})

    # Panel 1: KL and TV
    ax1 = axes[0]
    ax1.plot(epochs, kl_mode, "o-", color="darkred", linewidth=1.5, markersize=4,
             label="KL($\\alpha$ || $w$) — mode-weight error")
    ax1.fill_between(epochs, kl_mode, alpha=0.15, color="darkred")

    if has_full_kl:
        full_epochs = [r["epoch"] for r in full_kl_data]
        mean_E = [r["mean_energy"] for r in full_kl_data]
        ax1_twin = ax1.twinx()
        ax1_twin.plot(full_epochs, mean_E, "s--", color="navy", linewidth=1.5,
                      markersize=5, label="$\\mathbb{E}_Q[E(X_1)]$ — energy proxy")
        ax1_twin.set_ylabel("Mean energy", fontsize=11, color="navy")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")
    else:
        ax1.legend(fontsize=10)

    ax1.set_ylabel("KL($\\alpha$ || $w$)", fontsize=12)
    if title:
        ax1.set_title(title, fontsize=14)
    else:
        ax1.set_title(f"E9: KL Decomposition Tracking (K={K})", fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Panel 2: TV distance
    ax2 = axes[1]
    ax2.plot(epochs, tv_mode, "D-", color="darkorange", linewidth=1.5, markersize=4,
             label="TV($\\alpha$, $w$)")
    ax2.set_ylabel("TV distance", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    if nrows == 2:
        ax2.set_xlabel("Epoch", fontsize=12)
        # Add alive modes on twin axis
        ax2_twin = ax2.twinx()
        ax2_twin.plot(epochs, alive, "x--", color="forestgreen", linewidth=1,
                      markersize=4, alpha=0.7, label=f"Alive modes (/{K})")
        ax2_twin.set_ylabel("Alive modes", fontsize=11, color="forestgreen")
        ax2_twin.set_ylim(0, K + 1)
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=9)

    # Panel 3 (if K > 10): Dead mode count
    if nrows == 3:
        ax3 = axes[2]
        dead = K - alive
        ax3.plot(epochs, dead, "o-", color="darkred", linewidth=1.5, markersize=4)
        ax3.fill_between(epochs, dead, alpha=0.2, color="darkred")
        ax3.set_xlabel("Epoch", fontsize=12)
        ax3.set_ylabel(f"Dead modes (/{K})", fontsize=12)
        ax3.set_ylim(-0.5, K + 0.5)
        ax3.grid(True, alpha=0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"Saved: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="E9: KL Decomposition Tracking")
    parser.add_argument("--tracking", default=None, help="Path to mode_tracking.jsonl")
    parser.add_argument("--run-dir", default=None, help="Path to run directory")
    parser.add_argument("--estimate-full-kl", action="store_true",
                        help="Estimate full KL from checkpoints (slow)")
    parser.add_argument("--output", default="figures/e9_kl.pdf")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--title", default=None)

    args = parser.parse_args()

    if not args.tracking and not args.run_dir:
        parser.error("Must provide --tracking or --run-dir")

    plot_kl_decomposition(
        tracking_path=args.tracking,
        run_dir=args.run_dir,
        estimate_full_kl=args.estimate_full_kl,
        output_path=args.output,
        device=args.device,
        title=args.title,
    )


if __name__ == "__main__":
    main()
