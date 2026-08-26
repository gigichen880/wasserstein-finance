from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIGS_DIR = ROOT / "figs"
RESULTS_DIR = ROOT / "results"
DOCS_DIR = ROOT / "docs"


def ensure_output_dirs() -> None:
    FIGS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
