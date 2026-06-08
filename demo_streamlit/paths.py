from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

CHECKPOINT_PATH = PROJECT_ROOT / "experiments" / "checkpoint_full" / "convnext_tiny_cxrbert_baseline_full_checkpoint_bs96" / "best_model.pt"
CONFIG_PATH = PROJECT_ROOT / "experiments" / "checkpoint_full" / "convnext_tiny_cxrbert_baseline_full_checkpoint_bs96" / "config_snapshot.yaml"
PAIRS_CSV_PATH = PROJECT_ROOT / "data" / "pairs" / "cxr_consistency_pairs_hard.csv"
MODEL_SOURCE_PATH = SRC_DIR / "cxr_consistency" / "model.py"
DEFAULT_THRESHOLD = 0.34


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
