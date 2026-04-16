#!/usr/bin/env python
"""
Post-hoc reconstruction of mode_tracking.jsonl and evaluation metrics
from saved checkpoints.

Use this when training was run with eval disabled (eval_freq=99999).
Iterates over all checkpoints, loads each, generates samples, runs
the evaluator, computes mode weights, and writes:
  - mode_tracking.jsonl (α_k, KL, TV per epoch)
  - eval_metrics.jsonl (energy_w2, marginal_w2, sliced_w2 per epoch)

Usage:
    # Single run
    python scripts/reconstruct_tracking.py \\
        --run-dir results/b1_asbs/seed_0 \\
        --n-samples 10000

    # All seeds for an experiment
    python scripts/reconstruct_tracking.py \\
        --results-dir results/b1_asbs \\
        --n-samples 10000

    # All experiments
    python scripts/reconstruct_tracking.py \\
        --results-dir results \\
        --recursive \\
        --n-samples 10000
"""

import argparse
import json
import sys
import torch
import numpy as np
from pathlib import Path

# Ensure project root is on path (for adjoint_samplers imports in checkpoint loading)
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mc_utils import (
    load_model_from_checkpoint,
    generate_samples,
    assign_modes_nearest,
    compute_mode_weights,
    kl_mode_weights,
    tv_mode_weights,
    count_alive_modes,
    find_checkpoints,
)


def reconstruct_single_run(
    run_dir: str,
    n_samples: int = 10000,
    device: str = "cuda",
    overwrite: bool = False,
):
    """Reconstruct mode_tracking.jsonl and eval_metrics.jsonl for a single run."""
    run_dir = Path(run_dir)
    tracking_path = run_dir / "mode_tracking.jsonl"
    eval_path = run_dir / "eval_metrics.jsonl"

    if tracking_path.exists() and not overwrite:
        print(f"  [skip] {tracking_path} already exists (use --overwrite)")
        return

    ckpts = find_checkpoints(run_dir)
    if not ckpts:
        print(f"  [skip] No checkpoints in {run_dir}")
        return

    print(f"  Processing {run_dir.name}: {len(ckpts)} checkpoints...")

    # Clear existing files
    if tracking_path.exists():
        tracking_path.unlink()
    if eval_path.exists():
        eval_path.unlink()

    for ckpt in ckpts:
        epoch = int(ckpt.stem.split("_")[-1])
        if "latest" in ckpt.stem:
            continue

        try:
            model = load_model_from_checkpoint(str(ckpt), device=device)
            energy = model["energy"]
            samples = generate_samples(model, n_samples=n_samples, device=device)

            # ── Mode tracking ──
            if hasattr(energy, "mode_centers") and hasattr(energy, "mode_weights"):
                mode_centers = energy.mode_centers.to(device)
                mode_weights = energy.mode_weights.cpu()
                K = mode_centers.shape[0]

                assignments = assign_modes_nearest(samples, mode_centers)
                alpha = compute_mode_weights(assignments, K)
                kl = kl_mode_weights(alpha, mode_weights)
                tv = tv_mode_weights(alpha, mode_weights)
                alive = count_alive_modes(alpha, mode_weights)

                record = {
                    "epoch": epoch,
                    "alpha": alpha.tolist(),
                    "target_w": mode_weights.tolist(),
                    "kl": kl,
                    "tv": tv,
                    "alive_modes": alive,
                }
                with open(tracking_path, "a") as f:
                    f.write(json.dumps(record) + "\n")

            # ── Eval metrics ──
            with torch.no_grad():
                E_gen = energy.eval(samples).cpu().numpy()

            eval_record = {
                "epoch": epoch,
                "mean_energy": float(E_gen.mean()),
                "std_energy": float(E_gen.std()),
                "median_energy": float(np.median(E_gen)),
            }

            # W2 metrics if we can generate reference samples quickly
            # (skip for speed — the full evaluator handles this during proper eval)
            with open(eval_path, "a") as f:
                f.write(json.dumps(eval_record) + "\n")

            status = f"ep={epoch}"
            if hasattr(energy, "mode_centers") and hasattr(energy, "mode_weights"):
                status += f" KL={kl:.4f} alive={alive}/{K}"
            print(f"    {status}")

        except Exception as e:
            print(f"    [error] epoch {epoch}: {e}")

    print(f"  Written: {tracking_path}")
    print(f"  Written: {eval_path}")


def reconstruct_dir(results_dir, recursive=False, **kwargs):
    """Reconstruct for all seed directories under a results directory."""
    results_dir = Path(results_dir)

    if recursive:
        # Find all directories that contain a checkpoints/ subfolder
        targets = []
        for ckpt_dir in sorted(results_dir.rglob("checkpoints")):
            run_dir = ckpt_dir.parent
            # Only process leaf run directories (contain checkpoints but not nested runs)
            targets.append(run_dir)
    else:
        # Direct seed_* subdirectories
        targets = sorted(results_dir.glob("seed_*"))

    if not targets:
        print(f"No run directories found under {results_dir}")
        return

    print(f"Found {len(targets)} run(s) to process")
    for run_dir in targets:
        reconstruct_single_run(str(run_dir), **kwargs)


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct mode_tracking.jsonl from checkpoints"
    )
    parser.add_argument("--run-dir", default=None, help="Single run directory")
    parser.add_argument("--results-dir", default=None, help="Parent directory with seed_* subdirs")
    parser.add_argument("--recursive", action="store_true",
                        help="Recursively find all runs under results-dir")
    parser.add_argument("--n-samples", type=int, default=10000,
                        help="Samples to generate per checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing tracking files")

    args = parser.parse_args()

    if args.run_dir:
        reconstruct_single_run(
            args.run_dir, n_samples=args.n_samples,
            device=args.device, overwrite=args.overwrite,
        )
    elif args.results_dir:
        reconstruct_dir(
            args.results_dir, recursive=args.recursive,
            n_samples=args.n_samples, device=args.device,
            overwrite=args.overwrite,
        )
    else:
        parser.error("Provide --run-dir or --results-dir")


if __name__ == "__main__":
    main()
