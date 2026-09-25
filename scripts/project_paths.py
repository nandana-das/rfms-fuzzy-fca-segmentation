"""Project-relative paths shared by the analysis scripts.

All paths are derived from this file's location, so scripts work from any current
working directory when invoked as ``python scripts/stepN_*.py``.
"""
from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OLIST_RAW_DIR = DATA_DIR / "olist_raw"
RETAIL2_RAW_DIR = DATA_DIR / "raw_retail2"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# Existing scripts concatenate directory strings and filenames. Keep that
# interface while resolving the directory roots portably with pathlib.
OLIST_RAW = str(OLIST_RAW_DIR) + os.sep
RETAIL2_RAW = str(RETAIL2_RAW_DIR) + os.sep
PROCESSED = str(PROCESSED_DIR) + os.sep
RESULTS = str(RESULTS_DIR) + os.sep
FIGURES = str(FIGURES_DIR) + os.sep


def ensure_output_dirs():
    """Create output locations needed by the scripts, without touching data."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
