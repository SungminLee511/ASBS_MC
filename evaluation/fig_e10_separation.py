"""
E10: Separation Sweep

Figures:
  - fig_e10_b1_sweep.pdf: collapse probability + time-to-collapse + KL vs d
  - fig_e10_b5_sweep.pdf: collapse + which-mode-dies-first
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from mc_utils import load_tracking
from config import (
    RESULTS_DIR, FIGURES_DIR, apply_style,
    E10_B1_SEPARATIONS, E10_B5_CENTER_SCALES, E10_N_SEEDS, MODE_COLORS_4,
)


def _analyze_sweep(base_dir, param_values, param_prefix, n_seeds, threshold=0.1):
    results = []
    for pv in param_values:
        c_count, total, kl_list, collapse_times, dead_modes = 0, 0, [], [], []
        for seed in range(n_seeds):
            records = load_tracking(base_dir / f"{param_prefix}_{pv}" / f"seed_{seed}" / "mode_tracking.jsonl")
            if not records:
                continue
            total += 1
            tw = np.array(records[0]["target_w"])
            K = len(tw)
            collapsed, c_ep, dm = False, None, None
            for r in records:
                a = np.array(r["alpha"])
                for k in range(K):
                    if tw[k] > 0.01 and a[k] < threshold * tw[k]:
                        if not collapsed:
                            collapsed, c_ep, dm = True, r["epoch"], k
                        break
            if collapsed:
                c_count += 1
                collapse_times.append(c_ep)
                dead_modes.append(dm)
            kl_list.append(records[-1]["kl"])

        if total > 0:
            dm_counts = {}
            for m in dead_modes:
                dm_counts[m] = dm_counts.get(m, 0) + 1
            results.append({
                "param": pv, "collapse_prob": c_count / total,
                "mean_time": np.mean(collapse_times) if collapse_times else None,
                "mean_kl": np.mean(kl_list), "std_kl": np.std(kl_list),
                "dead_mode_counts": dm_counts, "n_runs": total,
            })
    return results


def fig_b1():
    apply_style()
    results = _analyze_sweep(RESULTS_DIR / "sep_sweep_b1", E10_B1_SEPARATIONS, "d", E10_N_SEEDS)
    if not results:
        print("  [skip] No E10/B1 data")
        return

    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(8, 10), sharex=True, gridspec_kw={"hspace": 0.12})
    ps = [r["param"] for r in results]

    a1.plot(ps, [r["collapse_prob"] for r in results], "o-", color="navy", linewidth=2, markersize=7)
    a1.set_ylabel("Collapse probability")
    a1.set_ylim(-0.05, 1.05)
    a1.set_title("Separation Sweep — B1 (Symmetric 2-Mode)")

    tp, tv = [], []
    for r in results:
        if r["mean_time"] is not None:
            tp.append(r["param"]); tv.append(r["mean_time"])
    if tp:
        a2.plot(tp, tv, "s-", color="darkred", linewidth=1.5, markersize=6)
    a2.set_ylabel("Mean time to collapse")

    a3.errorbar(ps, [r["mean_kl"] for r in results], yerr=[r["std_kl"] for r in results],
                fmt="D-", color="forestgreen", capsize=3)
    a3.set_xlabel("Mode separation $d$")
    a3.set_ylabel("Final KL($\\alpha$ || $w$)")

    out = FIGURES_DIR / "fig_e10_b1_sweep.pdf"
    fig.savefig(out)
    print(f"  Saved: {out}")
    plt.close(fig)


def fig_b5():
    apply_style()
    results = _analyze_sweep(RESULTS_DIR / "sep_sweep_b5", E10_B5_CENTER_SCALES, "cs", E10_N_SEEDS)
    if not results:
        print("  [skip] No E10/B5 data")
        return

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True, gridspec_kw={"hspace": 0.15})
    ps = [r["param"] for r in results]

    a1.plot(ps, [r["collapse_prob"] for r in results], "o-", color="navy", linewidth=2, markersize=7)
    a1.set_ylabel("Collapse probability")
    a1.set_ylim(-0.05, 1.05)
    a1.set_title("Separation Sweep — B5 (Heterogeneous 4-Mode)")

    mode_names = ["Tight spike", "Broad cloud", "Ellipse", "Unit"]
    x_pos = np.arange(len(ps))
    bottom = np.zeros(len(ps))
    for k in range(4):
        fracs = [r["dead_mode_counts"].get(k, 0) / max(r["n_runs"], 1) for r in results]
        a2.bar(x_pos, fracs, bottom=bottom, color=MODE_COLORS_4[k], label=mode_names[k], alpha=0.8)
        bottom += np.array(fracs)
    a2.set_xticks(x_pos)
    a2.set_xticklabels([str(p) for p in ps])
    a2.set_xlabel("Center scale")
    a2.set_ylabel("First mode to die")
    a2.legend(fontsize=9)

    out = FIGURES_DIR / "fig_e10_b5_sweep.pdf"
    fig.savefig(out)
    print(f"  Saved: {out}")
    plt.close(fig)


def generate_all():
    print("E10: Separation Sweep")
    fig_b1()
    fig_b5()
