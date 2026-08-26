"""Deprecated shim. Use `python experiments/08_directional_alignment.py`."""

from __future__ import annotations

import runpy
from pathlib import Path


def main():
    target = Path(__file__).resolve().parent / "experiments" / "08_directional_alignment.py"
    print("gradient_flow_alignment.py is a shim; running experiments/08_directional_alignment.py")
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
