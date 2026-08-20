import numpy as np
import pandas as pd

from src.constants import ATTACK_CATEGORY_MAP

UNSEEN_LABEL = "__unseen_at_train__"


def binary_labels(df: pd.DataFrame) -> pd.Series:
    return (df["attack_type"] != "normal").map({True: "attack", False: "normal"})


def category_labels(df: pd.DataFrame) -> pd.Series:
    unmapped = set(df["attack_type"].unique()) - set(ATTACK_CATEGORY_MAP)
    if unmapped:
        raise KeyError(f"Attack types missing from ATTACK_CATEGORY_MAP: {unmapped}")
    return df["attack_type"].map(ATTACK_CATEGORY_MAP)


def multiclass_labels(df: pd.DataFrame, known_classes) -> pd.Series:
    """Raw attack_type, with any class not present in `known_classes` (i.e.
    not seen during training) replaced by a sentinel. A closed-set classifier
    can never predict a class it wasn't trained on, so these rows are by
    construction unclassifiable by exact type -- see evaluate.py for how this
    is surfaced rather than silently averaged away."""
    known = set(known_classes)
    return df["attack_type"].where(df["attack_type"].isin(known), UNSEEN_LABEL)


TARGET_BUILDERS = {
    "binary": lambda train_df, other_df: (binary_labels(train_df), binary_labels(other_df)),
    "category5": lambda train_df, other_df: (category_labels(train_df), category_labels(other_df)),
    "multiclass": lambda train_df, other_df: (
        train_df["attack_type"],
        multiclass_labels(other_df, known_classes=train_df["attack_type"].unique()),
    ),
}
