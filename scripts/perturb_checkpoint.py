#!/usr/bin/env python
"""
Load a checkpoint, add N(0, sigma^2 I) noise to controller weights,
save as a new checkpoint.

Used for v3 Family B2 (Controller-Level Perturbation) and
Family C1 (Initialization-to-Collapse Distance Sweep).

Usage:
    python scripts/perturb_checkpoint.py \
        --ckpt results/v3_b7_baseline/seed_0/checkpoints/checkpoint_1990.pt \
        --sigma 0.01 \
        --output results/v3_b7_perturbed/sigma_0.01/seed_0/checkpoints/checkpoint_1990.pt

    # With fixed seed for reproducibility:
    python scripts/perturb_checkpoint.py \
        --ckpt ... --sigma 0.01 --output ... --seed 42
"""

import argparse
import torch
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Perturb controller weights in a checkpoint"
    )
    parser.add_argument("--ckpt", required=True, help="Path to source checkpoint")
    parser.add_argument("--sigma", type=float, required=True,
                        help="Std dev of Gaussian noise to add")
    parser.add_argument("--output", required=True, help="Path for perturbed checkpoint")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    n_params = 0
    n_perturbed = 0
    for key, param in checkpoint["controller"].items():
        n_params += 1
        if param.is_floating_point():
            checkpoint["controller"][key] = param + args.sigma * torch.randn_like(param)
            n_perturbed += 1

    # Also perturb corrector if present
    if "corrector" in checkpoint and checkpoint["corrector"] is not None:
        for key, param in checkpoint["corrector"].items():
            n_params += 1
            if param.is_floating_point():
                checkpoint["corrector"][key] = param + args.sigma * torch.randn_like(param)
                n_perturbed += 1

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f"Perturbed {n_perturbed}/{n_params} parameter tensors (sigma={args.sigma})")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
