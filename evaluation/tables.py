"""
Summary tables in LaTeX format.

Tables:
  - table_mode_concentration.tex: final α_k, KL, TV, alive modes per benchmark
  - table_e8_jacobian.tex: J₁₁ estimates and collapse probability per separation
  - table_e13_survival.tex: metastable survival per κ₃
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from mc_utils import load_tracking
from config import (
    RESULTS_DIR, TABLES_DIR,
    E8_SEPARATIONS, E8_N_SEEDS, E8_SIGMA_SDE,
    E13_KAPPA3_VALUES, E13_KAPPA1, E13_N_SEEDS,
    E1_BENCHMARKS, ensure_dirs,
)


def _latex_header(caption, label, columns):
    n = len(columns)
    col_spec = "l" + "c" * (n - 1)
    header = " & ".join(columns)
    return (
        f"\\begin{{table}}[h]\n\\centering\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        f"\\begin{{tabular}}{{{col_spec}}}\n\\toprule\n"
        f"{header} \\\\\n\\midrule\n"
    )


def _latex_footer():
    return "\\bottomrule\n\\end{tabular}\n\\end{table}\n"


def table_mode_concentration(seed=0):
    """Table: final mode concentration metrics per benchmark."""
    lines = [_latex_header(
        "Mode concentration summary across benchmarks",
        "tab:mc_summary",
        ["Benchmark", "K", "Final KL", "Final TV", "Alive", "Epochs"],
    )]

    for bm in E1_BENCHMARKS:
        records = load_tracking(RESULTS_DIR / f"{bm}_asbs" / f"seed_{seed}" / "mode_tracking.jsonl")
        if not records:
            continue
        last = records[-1]
        K = len(last["alpha"])
        lines.append(
            f"{bm.upper()} & {K} & {last['kl']:.4f} & {last['tv']:.4f} & "
            f"{last['alive_modes']}/{K} & {last['epoch']} \\\\\n"
        )

    lines.append(_latex_footer())
    out = TABLES_DIR / "table_mode_concentration.tex"
    with open(out, "w") as f:
        f.writelines(lines)
    print(f"  Saved: {out}")


def table_e8_jacobian():
    """Table: J₁₁ and collapse probability per separation."""
    base = RESULTS_DIR / "e8_sweep"
    lines = [_latex_header(
        "$J_{11}^\\eta$ threshold verification",
        "tab:e8_jacobian",
        ["$d$", "$\\rho_{\\mathrm{sep}}$", "$\\hat{J}_{11}^\\eta$", "Collapse \\%"],
    )]

    for d in E8_SEPARATIONS:
        c_count, total = 0, 0
        j11_list = []
        for seed in range(E8_N_SEEDS):
            records = load_tracking(base / f"d_{d}" / f"seed_{seed}" / "mode_tracking.jsonl")
            if not records:
                continue
            total += 1
            if abs(records[-1]["alpha"][0] - 0.5) > 0.3:
                c_count += 1
            for i in range(1, min(len(records) - 1, 5)):
                ap = records[i]["alpha"][0] - 0.5
                an = records[i + 1]["alpha"][0] - 0.5
                if abs(ap) > 0.01:
                    r = an / ap
                    if 0 < r < 5:
                        j11_list.append(r)

        if total == 0:
            continue
        rho = d / E8_SIGMA_SDE
        j11_str = f"{np.mean(j11_list):.2f} $\\pm$ {np.std(j11_list)/np.sqrt(len(j11_list)):.2f}" if j11_list else "—"
        cp = f"{100 * c_count / total:.0f}\\%"
        lines.append(f"{d} & {rho:.1f} & {j11_str} & {cp} \\\\\n")

    lines.append(_latex_footer())
    out = TABLES_DIR / "table_e8_jacobian.tex"
    with open(out, "w") as f:
        f.writelines(lines)
    print(f"  Saved: {out}")


def table_e13_survival():
    """Table: metastable survival per κ₃."""
    base = RESULTS_DIR / "e13_sweep"
    lines = [_latex_header(
        "Metastable state survival vs.\\ well depth",
        "tab:e13_survival",
        ["$\\kappa_3$", "Depth ratio", "Survival \\%", "$\\alpha_3/w_3$"],
    )]

    for k3 in E13_KAPPA3_VALUES:
        surv, ratios, total = 0, [], 0
        for seed in range(E13_N_SEEDS):
            records = load_tracking(base / f"kappa3_{k3}" / f"seed_{seed}" / "mode_tracking.jsonl")
            if not records:
                continue
            total += 1
            last = records[-1]
            a3 = last["alpha"][2]
            w3 = last["target_w"][2]
            r = a3 / max(w3, 1e-10)
            ratios.append(r)
            if r > 0.5:
                surv += 1

        if total == 0:
            continue
        ratio_str = f"{np.mean(ratios):.2f} $\\pm$ {np.std(ratios):.2f}"
        sp = f"{100 * surv / total:.0f}\\%"
        lines.append(f"{k3:.0f} & {k3/E13_KAPPA1:.2f} & {sp} & {ratio_str} \\\\\n")

    lines.append(_latex_footer())
    out = TABLES_DIR / "table_e13_survival.tex"
    with open(out, "w") as f:
        f.writelines(lines)
    print(f"  Saved: {out}")


def generate_all():
    ensure_dirs()
    print("Tables:")
    table_mode_concentration()
    table_e8_jacobian()
    table_e13_survival()
