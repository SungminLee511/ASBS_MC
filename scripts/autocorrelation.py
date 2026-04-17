#!/usr/bin/env python
"""
Estimate Jacobian eigenvalues from small fluctuations in converged training runs.

In converged runs, α(n) oscillates slightly around the collapsed state.
Fits a VAR(1) model: δα(n+1) = A · δα(n) + noise, where the matrix A
approximates the Jacobian J^(S). Reports eigenvalues and spectral radius.

Usage:
    # Single run
    python scripts/autocorrelation.py \
        --run-dir results/v3/baselines/b7/seed_0 --window-size 200

    # All seeds for an experiment
    python scripts/autocorrelation.py \
        --results-dir results/v3/baselines/b7

    # All experiments recursively
    python scripts/autocorrelation.py \
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


def fit_var1(records, window_size=200):
    """Fit a VAR(1) model to the converged portion of α(n).

    Returns a dict with VAR(1) fit results, or None on failure.
    """
    if len(records) < 10:
        print("    [skip] Not enough data points ({})".format(len(records)))
        return None

    alphas = np.array([r["alpha"] for r in records])  # (T, K)
    T, K = alphas.shape

    # Select the converged regime: last window_size data points
    n_pts = min(window_size, T)
    alphas_window = alphas[-n_pts:]  # (n_pts, K)

    # Compute residuals: δα(n) = α(n) - mean(α)
    mean_alpha = alphas_window.mean(axis=0)  # (K,)
    residuals = alphas_window - mean_alpha[np.newaxis, :]  # (n_pts, K)

    # Check for degenerate case: all residuals near zero
    max_resid = np.max(np.abs(residuals))
    if max_resid < 1e-15:
        print("    [skip] α is constant in the window (no fluctuations)")
        return {
            "var1_matrix": np.zeros((K, K)).tolist(),
            "eigenvalues_real": [0.0] * K,
            "eigenvalues_imag": [0.0] * K,
            "spectral_radius": 0.0,
            "n_points": n_pts,
            "mean_alpha": mean_alpha.tolist(),
            "verdict": "degenerate (constant α, no fluctuations)",
        }

    # Build VAR(1) regression: δα(n+1) = A · δα(n) + noise
    # X = [δα(0), δα(1), ..., δα(n_pts-2)]^T  shape (n_pts-1, K)
    # Y = [δα(1), δα(2), ..., δα(n_pts-1)]^T  shape (n_pts-1, K)
    X = residuals[:-1]  # (n_pts-1, K)
    Y = residuals[1:]   # (n_pts-1, K)

    if X.shape[0] < K:
        print(f"    [skip] Not enough observations ({X.shape[0]}) "
              f"for {K} modes")
        return None

    # Solve Y = X @ A^T  =>  A^T = lstsq(X, Y)
    # A^T has shape (K, K), so A = (lstsq result)^T
    result = np.linalg.lstsq(X, Y, rcond=None)
    A_T = result[0]  # (K, K)
    A = A_T.T         # (K, K)

    # Compute eigenvalues
    eigenvalues = np.linalg.eig(A)[0]  # complex array of length K
    spectral_radius = float(np.max(np.abs(eigenvalues)))

    # Stability verdict
    if spectral_radius < 1.0:
        verdict = "stable (spectral_radius < 1)"
    elif abs(spectral_radius - 1.0) < 0.01:
        verdict = "marginal (spectral_radius ≈ 1)"
    else:
        verdict = "unstable (spectral_radius > 1)"

    return {
        "var1_matrix": A.tolist(),
        "eigenvalues_real": eigenvalues.real.tolist(),
        "eigenvalues_imag": eigenvalues.imag.tolist(),
        "spectral_radius": spectral_radius,
        "n_points": n_pts,
        "mean_alpha": mean_alpha.tolist(),
        "verdict": verdict,
    }


def process_single_run(run_dir, window_size=200, overwrite=False):
    """Process a single run directory."""
    run_dir = Path(run_dir)
    output_path = run_dir / "autocorrelation_fit.json"

    if output_path.exists() and not overwrite:
        print(f"  [skip] {output_path} already exists (use --overwrite)")
        return

    records = load_tracking(run_dir)
    if records is None:
        print(f"  [skip] No mode_tracking.jsonl in {run_dir}")
        return

    print(f"  Processing {run_dir}: {len(records)} data points")
    result = fit_var1(records, window_size=window_size)
    if result is None:
        return

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"    spectral_radius={result['spectral_radius']:.4f}  "
          f"verdict={result['verdict']}  "
          f"n_points={result['n_points']}")
    print(f"    Written: {output_path}")


def process_dir(results_dir, recursive=False, **kwargs):
    """Process all seed directories under a results directory."""
    results_dir = Path(results_dir)

    if recursive:
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
        description="Estimate Jacobian eigenvalues via VAR(1) on converged α fluctuations"
    )
    parser.add_argument("--run-dir", default=None,
                        help="Single run directory")
    parser.add_argument("--results-dir", default=None,
                        help="Parent directory with seed_* subdirs")
    parser.add_argument("--recursive", action="store_true",
                        help="Recursively find all runs under results-dir")
    parser.add_argument("--window-size", type=int, default=200,
                        help="Number of data points in the converged window (default: 200)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing autocorrelation_fit.json files")

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
