from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIGS_DIR = ROOT / "figs"
RESULTS_DIR = ROOT / "results"
DOCS_DIR = ROOT / "docs"
EMPIRICAL_DATA_DIR = ROOT / "data_by_stocks"
EMPIRICAL_CACHE_DIR = ROOT / "research" / "empirical" / "cache"
EMPIRICAL_RESULTS_DIR = RESULTS_DIR / "empirical"
EMPIRICAL_FIGS_DIR = FIGS_DIR / "empirical"


def ensure_output_dirs() -> None:
    FIGS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)


def ensure_empirical_dirs() -> None:
    ensure_output_dirs()
    EMPIRICAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    EMPIRICAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    EMPIRICAL_FIGS_DIR.mkdir(parents=True, exist_ok=True)
