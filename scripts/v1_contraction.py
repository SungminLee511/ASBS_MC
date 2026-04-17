#!/usr/bin/env python
"""
Extract the empirical contraction factor from converged baseline trajectories.

From mode_tracking.jsonl, extract the late-epoch decay rate of
||α(n) - α(final)|| as the system settles into its collapsed attractor.
Fits an exponential decay model: ||δα(n)|| = A * r^n and extracts the
contraction factor r.

Usage:
    # Single run
    python scripts/v1_contraction.py \
        --run-dir results/v3/baselines/b7/seed_0

    # All seeds for an experiment
    python scripts/v1_contraction.py \
        --results-dir results/v3/baselines/b7

    # All experiments recursively
    python scripts/v1_contraction.py \
        --results-dir results/v3/baselines --recursive
"""

import argparse
import json
import sys
import numpy as np
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def load_tracking(run_dir: Path):
    """Load mode_tracking.jsonl and return list of records sorted by epoch."""
    tracking_path = run_dir / "mode_tracking.jsonl"
    if not tracking_path.exists():
        return None
    records = []
    with open(tracking_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    records.sort(key=lambda r: r["epoch"])
    return records


def fit_contraction(records, window_size=200):
    """Fit exponential decay to ||α(n) - α_final|| over the late portion.

    Returns a dict with contraction fit results, or None on failure.
    """
    if len(records) < 15:
        print("    [skip] Not enough data points ({})".format(len(records)))
        return None

    epochs = np.array([r["epoch"] for r in records])
    alphas = np.array([r["alpha"] for r in records])  # shape (T, K)

    # Identify converged α_final: average of last 10 data points
    n_final = min(10, len(records))
    alpha_final = alphas[-n_final:].mean(axis=0)  # shape (K,)

    # Compute ||δα(n)|| for each epoch
    delta = alphas - alpha_final[np.newaxis, :]  # (T, K)
    norms = np.linalg.norm(delta, axis=1)  # (T,)

    # Select late portion (last window_size epochs by epoch value, not count)
    max_epoch = epochs[-1]
    min_epoch_window = max_epoch - window_size
    mask = epochs >= min_epoch_window
    if mask.sum() < 5:
        # Fall back to last N data points
        n_pts = min(window_size, len(records))
        mask = np.zeros(len(records), dtype=bool)
        mask[-n_pts:] = True

    fit_epochs = epochs[mask]
    fit_norms = norms[mask]
    fit_delta = delta[mask]  # (n_fit, K)

    # Filter out zero/tiny norms (already converged exactly)
    valid = fit_norms > 1e-15
    if valid.sum() < 3:
        print("    [skip] α already constant in fit window")
        return {
            "final_alpha": alpha_final.tolist(),
            "contraction_factor_r": 0.0,
            "fit_r2": float("nan"),
            "per_mode_rates": [0.0] * alphas.shape[1],
            "fit_window_epochs": [int(fit_epochs[0]), int(fit_epochs[-1])],
            "n_points": int(mask.sum()),
        }

    # Fit log(||δα||) = log(A) + n * log(r) via linear regression
    log_norms = np.log(fit_norms[valid])
    fit_ep_valid = fit_epochs[valid].astype(float)

    # Use epoch indices relative to start for numerical stability
    ep_offset = fit_ep_valid[0]
    ep_rel = fit_ep_valid - ep_offset

    # Linear fit: log_norms = intercept + slope * ep_rel
    # slope = log(r), intercept = log(A) + ep_offset * log(r)
    X = np.column_stack([np.ones(len(ep_rel)), ep_rel])
    result = np.linalg.lstsq(X, log_norms, rcond=None)
    coeffs = result[0]
    intercept, slope = coeffs[0], coeffs[1]

    r_global = float(np.exp(slope))

    # R^2 for goodness of fit
    predicted = intercept + slope * ep_rel
    ss_res = np.sum((log_norms - predicted) ** 2)
    ss_tot = np.sum((log_norms - log_norms.mean()) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-30 else float("nan")

    # Per-mode decay rates
    K = alphas.shape[1]
    per_mode_rates = []
    for k in range(K):
        mode_delta = np.abs(fit_delta[:, k])
        mode_valid = mode_delta > 1e-15
        if mode_valid.sum() < 3:
            per_mode_rates.append(0.0)
            continue
        log_d = np.log(mode_delta[mode_valid])
        ep_k = fit_epochs[mask][mode_valid].astype(float)
        ep_k_rel = ep_k - ep_k[0]
        X_k = np.column_stack([np.ones(len(ep_k_rel)), ep_k_rel])
        res_k = np.linalg.lstsq(X_k, log_d, rcond=None)
        per_mode_rates.append(float(np.exp(res_k[0][1])))

    return {
        "final_alpha": alpha_final.tolist(),
        "contraction_factor_r": r_global,
        "fit_r2": r2,
        "per_mode_rates": per_mode_rates,
        "fit_window_epochs": [int(fit_epochs[0]), int(fit_epochs[-1])],
        "n_points": int(mask.sum()),
    }


def process_single_run(run_dir, window_size=200, overwrite=False):
    """Process a single run directory."""
    run_dir = Path(run_dir)
    output_path = run_dir / "contraction_fit.json"

    if output_path.exists() and not overwrite:
        print(f"  [skip] {output_path} already exists (use --overwrite)")
        return

    records = load_tracking(run_dir)
    if records is None:
        print(f"  [skip] No mode_tracking.jsonl in {run_dir}")
        return

    print(f"  Processing {run_dir}: {len(records)} data points")
    result = fit_contraction(records, window_size=window_size)
    if result is None:
        return

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"    r={result['contraction_factor_r']:.4f}  "
          f"R²={result['fit_r2']:.4f}  "
          f"window=[{result['fit_window_epochs'][0]}, {result['fit_window_epochs'][1]}]")
    print(f"    Written: {output_path}")


def process_dir(results_dir, recursive=False, **kwargs):
    """Process all seed directories under a results directory."""
    results_dir = Path(results_dir)

    if recursive:
        # Find all directories that contain mode_tracking.jsonl
        targets = sorted(set(
            p.parent for p in results_dir.rglob("mode_tracking.jsonl")
        ))
    else:
        targets = sorted(results_dir.glob("seed_*"))

    if not targets:
        print(f"No run directories found under {results_dir}")
        return

    print(f"Found {len(targets)} run(s) to process")
    for run_dir in targets:
        process_single_run(run_dir, **kwargs)


def main():
    parser = argparse.ArgumentParser(
        description="Extract empirical contraction factor from converged trajectories"
    )
    parser.add_argument("--run-dir", default=None,
                        help="Single run directory")
    parser.add_argument("--results-dir", default=None,
                        help="Parent directory with seed_* subdirs")
    parser.add_argument("--recursive", action="store_true",
                        help="Recursively find all runs under results-dir")
    parser.add_argument("--window-size", type=int, default=200,
                        help="Number of epochs in the late-portion fit window (default: 200)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing contraction_fit.json files")

    args = parser.parse_args()

    if args.run_dir:
        process_single_run(
            args.run_dir, window_size=args.window_size, overwrite=args.overwrite,
        )
    elif args.results_dir:
        process_dir(
            args.results_dir, recursive=args.recursive,
            window_size=args.window_size, overwrite=args.overwrite,
        )
    else:
        parser.error("Provide --run-dir or --results-dir")


if __name__ == "__main__":
    main()
