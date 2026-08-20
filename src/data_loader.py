from pathlib import Path

import pandas as pd

from src.constants import COLUMN_NAMES

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def load_raw(filename: str) -> pd.DataFrame:
    """Load an NSL-KDD file (KDDTrain+.txt / KDDTest+.txt) with the correct
    column schema applied (see src/constants.py for why this matters)."""
    path = RAW_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Expected NSL-KDD file in data/raw/."
        )
    df = pd.read_csv(path, header=None)
    if df.shape[1] != len(COLUMN_NAMES):
        raise ValueError(
            f"{path} has {df.shape[1]} columns, expected {len(COLUMN_NAMES)}."
        )
    df.columns = COLUMN_NAMES
    return df


def load_train() -> pd.DataFrame:
    return load_raw("KDDTrain+.txt")


def load_test() -> pd.DataFrame:
    return load_raw("KDDTest+.txt")
