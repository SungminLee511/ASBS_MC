"""
E8: J₁₁^η Threshold Estimation

Figures:
  - fig_e8_threshold.pdf: J₁₁ vs ρ_sep + collapse probability
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from mc_utils import load_tracking
from config import (
    RESULTS_DIR, FIGURES_DIR, apply_style,
    E8_SEPARATIONS, E8_N_SEEDS, E8_SIGMA_SDE,
)


def _analyze(separations, n_seeds, collapse_threshold=0.3):
    base = RESULTS_DIR / "e8_sweep"
    collapse = {}
    j11_est = {}

    for d in separations:
        c_count, total, kl_list = 0, 0, []
        j11_list = []

        for seed in range(n_seeds):
            records = load_tracking(base / f"d_{d}" / f"seed_{seed}" / "mode_tracking.jsonl")
            if not records:
                continue
            total += 1
            last = records[-1]
            if abs(last["alpha"][0] - 0.5) > collapse_threshold:
                c_count += 1
            kl_list.append(last["kl"])

            # J₁₁ estimate from early dynamics
            for i in range(1, min(len(records) - 1, 5)):
                a_prev = records[i]["alpha"][0] - 0.5
                a_next = records[i + 1]["alpha"][0] - 0.5
                if abs(a_prev) > 0.01:
                    ratio = a_next / a_prev
                    if 0 < ratio < 5:
                        j11_list.append(ratio)

        if total > 0:
            collapse[d] = c_count / total
        if j11_list:
            j11_est[d] = (np.mean(j11_list), np.std(j11_list) / np.sqrt(len(j11_list)))

    return collapse, j11_est


def generate_all():
    apply_style()
    print("E8: Jacobian Threshold")

    collapse, j11_est = _analyze(E8_SEPARATIONS, E8_N_SEEDS)
    if not collapse and not j11_est:
        print("  [skip] No E8 data")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True,
                                     gridspec_kw={"hspace": 0.1})

    # Top: J₁₁
    if j11_est:
        ds = sorted(j11_est.keys())
        rhos = np.array(ds) / E8_SIGMA_SDE
        means = [j11_est[d][0] for d in ds]
        errs = [j11_est[d][1] for d in ds]
        ax1.errorbar(rhos, means, yerr=errs, fmt="o-", color="navy",
                      capsize=3, label="$\\hat{J}_{11}^\\eta$")
    ax1.axhline(y=0.5, color="red", linestyle="--", linewidth=1.5,
                label="Threshold ($1/2$)")
    ax1.set_ylabel("$\\hat{J}_{11}^\\eta$")
    ax1.set_title("Jacobian Threshold Verification")
    ax1.legend()

    # Bottom: collapse probability
    if collapse:
        ds = sorted(collapse.keys())
        rhos = np.array(ds) / E8_SIGMA_SDE
        probs = [collapse[d] for d in ds]
        ax2.plot(rhos, probs, "s-", color="darkred", linewidth=2, markersize=7)
    ax2.set_xlabel("$\\rho_{\\mathrm{sep}} = d / \\sigma(1)$")
    ax2.set_ylabel("Collapse probability")
    ax2.set_ylim(-0.05, 1.05)

    out = FIGURES_DIR / "fig_e8_threshold.pdf"
    fig.savefig(out)
    print(f"  Saved: {out}")
    plt.close(fig)
