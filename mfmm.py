"""Deprecated shim. Use `python experiments/run_all.py` or the wfmm package."""

from __future__ import annotations

import runpy
from pathlib import Path


def main():
    target = Path(__file__).resolve().parent / "experiments" / "run_all.py"
    print("mfmm.py is a shim; running experiments/run_all.py")
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
