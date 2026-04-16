#!/usr/bin/env python
"""
Master evaluation script — generates all figures and tables.

Assumes all experiments have been run and results (checkpoints +
mode_tracking.jsonl) exist under results/.

Usage:
    # Generate everything
    python evaluation/run_all.py

    # Only tracking-based figures (no GPU needed)
    python evaluation/run_all.py --no-gpu

    # Only specific experiments
    python evaluation/run_all.py --only e7 e1 e8

    # Only tables
    python evaluation/run_all.py --tables-only
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from config import ensure_dirs, FIGURES_DIR, TABLES_DIR


# ──────────────────────────────────────────────────────────────────────
# Registry of all figure generators
# ──────────────────────────────────────────────────────────────────────

# "tracking-only" = reads mode_tracking.jsonl only, no GPU needed
# "gpu" = needs to load checkpoints and run forward passes

GENERATORS = {
    # Tier 1
    "e7":  {"module": "fig_e7_phase_portrait",  "gpu": False},
    "e1":  {"module": "fig_e1_mode_tracking",   "gpu": False},
    "e8":  {"module": "fig_e8_jacobian",        "gpu": False},
    "e11": {"module": "fig_e11_temperature",    "gpu": False},
    "e13": {"module": "fig_e13_metastable",     "gpu": False},
    # Tier 2
    "e6":  {"module": "fig_e6_trajectories",    "gpu": True},
    "e4":  {"module": "fig_e4_controller",      "gpu": True},
    "e3":  {"module": "fig_e3_loss_decomp",     "gpu": True},
    "e10": {"module": "fig_e10_separation",     "gpu": False},
    "e9":  {"module": "fig_e9_kl",              "gpu": False},
    # Tier 3
    "e2":  {"module": "fig_e2_heatmaps",        "gpu": True},
    "e5":  {"module": "fig_e5_density",         "gpu": True},
    "e12": {"module": "fig_e12_cascade",        "gpu": False},
}


def run_generator(name, info, device="cuda"):
    """Import and run a figure generator module."""
    try:
        mod = __import__(info["module"])
        if info["gpu"] and hasattr(mod, "generate_all"):
            mod.generate_all(device=device)
        elif hasattr(mod, "generate_all"):
            mod.generate_all()
        else:
            print(f"  [warn] {info['module']} has no generate_all()")
    except Exception as e:
        print(f"  [ERROR] {name}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate all evaluation figures and tables")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Skip GPU-dependent figures (E2, E3, E4, E5, E6)")
    parser.add_argument("--only", nargs="+", default=None,
                        help="Only generate specific experiments (e.g., e7 e1 e8)")
    parser.add_argument("--tables-only", action="store_true",
                        help="Only generate tables")
    parser.add_argument("--device", default="cuda",
                        help="Device for GPU figures")

    args = parser.parse_args()
    ensure_dirs()

    print(f"Output directories:")
    print(f"  Figures: {FIGURES_DIR}")
    print(f"  Tables:  {TABLES_DIR}")
    print()

    t0 = time.time()

    if not args.tables_only:
        selected = args.only or list(GENERATORS.keys())
        for name in selected:
            if name not in GENERATORS:
                print(f"  [warn] Unknown experiment: {name}")
                continue
            info = GENERATORS[name]
            if args.no_gpu and info["gpu"]:
                print(f"  [skip] {name} (requires GPU)")
                continue
            run_generator(name, info, device=args.device)

    # Tables
    if args.tables_only or args.only is None:
        print()
        from tables import generate_all as gen_tables
        gen_tables()

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Figures: {FIGURES_DIR}")
    print(f"Tables:  {TABLES_DIR}")


if __name__ == "__main__":
    main()
