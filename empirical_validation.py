"""Deprecated shim. Use `python experiments/05_synthetic_falsification.py`."""

from __future__ import annotations

import runpy
from pathlib import Path


def main():
    target = Path(__file__).resolve().parent / "experiments" / "05_synthetic_falsification.py"
    print("empirical_validation.py is a shim; running experiments/05_synthetic_falsification.py")
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
