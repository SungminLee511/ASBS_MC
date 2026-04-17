#!/usr/bin/env python
"""
v3 Family D3: Multi-Epoch Perturbation Decay Fitting.

Given mode_tracking.jsonl files from B1 (dead mode revival) or B2
(controller perturbation) experiments, fit exponential decay curves
to the perturbation response and extract the empirical contraction factor.

After perturbation ends at epoch T, the deviation from the final
(converged) mode weight vector decays approximately as:

    ||δα(n)|| = A * r^n        (overall, n = epochs since perturbation end)
    |δα_k(n)| = A_k * r_k^n   (per-mode)

where r < 1 indicates contraction (return to the fixed point).

Usage:
    # Single run (B1 — injection ended at epoch 2050)
    python scripts/fit_decay.py \\
        --run-dir results/v3/family_b1/rho_0.05_M50/seed_0 \\
        --injection-end-epoch 2050 \\
        --output results/v3/family_b1/rho_0.05_M50/seed_0/decay_fit.json

    # Batch mode: process all seeds under an experiment directory
    python scripts/fit_decay.py \\
        --results-dir results/v3/family_b1/rho_0.05_M50 \\
        --injection-end-epoch 2050

    # B2 run — perturbation is at epoch 0 (resumed from perturbed checkpoint)
    python scripts/fit_decay.py \\
        --run-dir results/v3/family_b2/sigma_0.01/seed_0 \\
        --perturbation-epoch 0
"""

import argparse
import json
import sys
import numpy as np
from pathlib import Path

# Ensure project root is on path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


# ──────────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────────

def load_tracking(run_dir: Path):
    """Load mode_tracking.jsonl and return list of records sorted by epoch."""
    tracking_path = run_dir / "mode_tracking.jsonl"
    if not tracking_path.exists():
        raise FileNotFoundError(f"mode_tracking.jsonl not found in {run_dir}")

    records = []
    with open(tracking_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    records.sort(key=lambda r: r["epoch"])
    return records


# ──────────────────────────────────────────────────────────────────────
# Fitting
# ──────────────────────────────────────────────────────────────────────

def fit_exponential_decay(n_arr, y_arr):
    """Fit y = A * r^n via log-linear regression.

    Parameters
    ----------
    n_arr : 1-D array of epoch offsets (0, 1, 2, ...)
    y_arr : 1-D array of positive values (||δα|| or |δα_k|)

    Returns
    -------
    A : amplitude
    r : contraction factor (< 1 means decay)
    r2 : R² goodness-of-fit
    """
    # Filter out zeros / negatives (can't take log)
    mask = y_arr > 0
    if mask.sum() < 2:
        return np.nan, 1.0, 0.0

    n_fit = n_arr[mask]
    log_y = np.log(y_arr[mask])

    # log(y) = log(A) + n * log(r)  →  linear fit
    coeffs = np.polyfit(n_fit, log_y, 1)  # [log(r), log(A)]
    log_r, log_A = coeffs
    A = np.exp(log_A)
    r = np.exp(log_r)

    # R² on the log scale
    log_y_pred = np.polyval(coeffs, n_fit)
    ss_res = np.sum((log_y - log_y_pred) ** 2)
    ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return float(A), float(r), float(r2)


def fit_decay_for_run(
    run_dir: str,
    injection_end_epoch: int = None,
    perturbation_epoch: int = None,
    output: str = None,
):
    """Fit perturbation decay for a single run directory.

    Parameters
    ----------
    run_dir : path to the run (contains mode_tracking.jsonl)
    injection_end_epoch : for B1 — epoch at which injection *ends*
        (post-injection training starts at this epoch).
    perturbation_epoch : for B2 — epoch at which perturbation was applied.
        Decay is measured starting from the epoch *after* this one.
    output : path for the output JSON; defaults to <run_dir>/decay_fit.json.
    """
    run_dir = Path(run_dir)
    records = load_tracking(run_dir)

    if not records:
        print(f"  [skip] No records in {run_dir}")
        return None

    # ── Determine the post-perturbation window ──
    if injection_end_epoch is not None:
        start_epoch = injection_end_epoch
    elif perturbation_epoch is not None:
        # Decay starts one epoch after the perturbation
        start_epoch = perturbation_epoch + 1
    else:
        # Heuristic: assume perturbation is at the first recorded epoch
        start_epoch = records[0]["epoch"]
        print(f"  [info] No perturbation epoch specified; using first epoch {start_epoch}")

    # Keep only records at or after start_epoch
    post = [r for r in records if r["epoch"] >= start_epoch]
    if len(post) < 3:
        print(f"  [skip] Only {len(post)} post-perturbation epochs in {run_dir}")
        return None

    epochs = np.array([r["epoch"] for r in post])
    alphas = np.array([r["alpha"] for r in post])  # shape (T, K)
    K = alphas.shape[1]

    # α_final = average of last 10% of post-perturbation trajectory (or last record)
    tail_len = max(1, len(post) // 10)
    alpha_final = alphas[-tail_len:].mean(axis=0)  # (K,)

    # δα(n) for each epoch
    delta_alpha = alphas - alpha_final[None, :]     # (T, K)
    delta_norm = np.linalg.norm(delta_alpha, axis=1)  # (T,)

    # Epoch offsets (n = 0, 1, 2, ... relative to start_epoch)
    n_arr = epochs - epochs[0]

    # ── Overall fit: ||δα(n)|| = A * r^n ──
    A_all, r_all, r2_all = fit_exponential_decay(n_arr, delta_norm)

    # ── Per-mode fit: |δα_k(n)| = A_k * r_k^n ──
    per_mode_A = []
    per_mode_r = []
    per_mode_r2 = []
    for k in range(K):
        abs_delta_k = np.abs(delta_alpha[:, k])
        # Skip modes that are always ~0 (dead throughout)
        if abs_delta_k.max() < 1e-8:
            per_mode_A.append(0.0)
            per_mode_r.append(1.0)
            per_mode_r2.append(0.0)
            continue
        Ak, rk, r2k = fit_exponential_decay(n_arr, abs_delta_k)
        per_mode_A.append(Ak)
        per_mode_r.append(rk)
        per_mode_r2.append(r2k)

    # ── Epochs to recovery: ||δα|| < 0.01 ──
    recovery_mask = delta_norm < 0.01
    if recovery_mask.any():
        first_recovery_idx = int(np.argmax(recovery_mask))
        epochs_to_recovery = int(n_arr[first_recovery_idx])
    else:
        epochs_to_recovery = None  # never recovered within the window

    # ── Assemble result ──
    result = {
        "run_dir": str(run_dir),
        "injection_end_epoch": injection_end_epoch,
        "perturbation_epoch": perturbation_epoch,
        "start_epoch": int(epochs[0]),
        "n_post_epochs": len(post),
        "contraction_factor_r": r_all,
        "amplitude_A": A_all,
        "fit_quality_r2": r2_all,
        "per_mode_rates": per_mode_r,
        "per_mode_amplitudes": per_mode_A,
        "per_mode_r2": per_mode_r2,
        "peak_perturbation": float(delta_norm[0]),
        "final_alpha": alpha_final.tolist(),
        "epochs_to_recovery": epochs_to_recovery,
    }

    # ── Write output ──
    if output is None:
        output = run_dir / "decay_fit.json"
    else:
        output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  r={r_all:.6f}  R²={r2_all:.4f}  "
          f"peak={delta_norm[0]:.4f}  recovery={epochs_to_recovery}  "
          f"→ {output}")

    return result


def fit_decay_batch(
    results_dir: str,
    injection_end_epoch: int = None,
    perturbation_epoch: int = None,
):
    """Process all seed_* subdirectories under results_dir."""
    results_dir = Path(results_dir)
    seed_dirs = sorted(results_dir.glob("seed_*"))

    if not seed_dirs:
        print(f"No seed_* directories found under {results_dir}")
        return

    print(f"Found {len(seed_dirs)} seed(s) to process")
    all_results = []

    for seed_dir in seed_dirs:
        print(f"Processing {seed_dir.name}...")
        res = fit_decay_for_run(
            run_dir=str(seed_dir),
            injection_end_epoch=injection_end_epoch,
            perturbation_epoch=perturbation_epoch,
        )
        if res is not None:
            all_results.append(res)

    # ── Summary across seeds ──
    if all_results:
        rs = [r["contraction_factor_r"] for r in all_results
              if not np.isnan(r["contraction_factor_r"])]
        if rs:
            print(f"\n{'='*60}")
            print(f"  Aggregate over {len(rs)} seed(s):")
            print(f"    r  = {np.mean(rs):.6f} ± {np.std(rs):.6f}")
            print(f"    R² = {np.mean([r['fit_quality_r2'] for r in all_results]):.4f}")
            print(f"{'='*60}")

        # Write aggregate summary
        summary = {
            "n_seeds": len(all_results),
            "contraction_factor_mean": float(np.mean(rs)),
            "contraction_factor_std": float(np.std(rs)),
            "per_seed": all_results,
        }
        summary_path = results_dir / "decay_fit_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Summary → {summary_path}")


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fit exponential decay to post-perturbation α trajectories"
    )
    parser.add_argument("--run-dir", default=None,
                        help="Single run directory (contains mode_tracking.jsonl)")
    parser.add_argument("--results-dir", default=None,
                        help="Parent directory with seed_* subdirs (batch mode)")
    parser.add_argument("--injection-end-epoch", type=int, default=None,
                        help="B1: epoch at which injection ends "
                             "(e.g. injection_start + injection_duration)")
    parser.add_argument("--perturbation-epoch", type=int, default=None,
                        help="B2: epoch at which perturbation was applied "
                             "(decay measured from next epoch)")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (single-run mode only; "
                             "defaults to <run_dir>/decay_fit.json)")

    args = parser.parse_args()

    if args.injection_end_epoch is not None and args.perturbation_epoch is not None:
        parser.error("Specify --injection-end-epoch OR --perturbation-epoch, not both")

    if args.run_dir:
        fit_decay_for_run(
            run_dir=args.run_dir,
            injection_end_epoch=args.injection_end_epoch,
            perturbation_epoch=args.perturbation_epoch,
            output=args.output,
        )
    elif args.results_dir:
        fit_decay_batch(
            results_dir=args.results_dir,
            injection_end_epoch=args.injection_end_epoch,
            perturbation_epoch=args.perturbation_epoch,
        )
    else:
        parser.error("Provide --run-dir or --results-dir")


if __name__ == "__main__":
    main()
