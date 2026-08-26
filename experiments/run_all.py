"""Run all experiments in order."""

from __future__ import annotations

import argparse
import runpy
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = [
    "01_relaxation.py",
    "02_shock.py",
    "03_parameter_sweep.py",
    "04_solver_benchmark.py",
    "05_synthetic_falsification.py",
    "06_moment_validation.py",
    "07_distribution_forecast.py",
    "08_directional_alignment.py",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", type=str, default=None, help="substring filter, e.g. 01 or forecast")
    args = p.parse_args()
    t0 = time.time()
    for name in SCRIPTS:
        if args.only and args.only not in name:
            continue
        print(f"\n===== {name} =====")
        runpy.run_path(str(ROOT / name), run_name="__main__")
    print(f"\nall done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
