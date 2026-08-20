from dataclasses import dataclass, field

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.constants import CATEGORICAL_COLUMNS, NON_FEATURE_COLUMNS


@dataclass
class Preprocessor:
    """Fits on the training split only; reused as-is on any other split so
    there is no leakage of test-set statistics into feature scaling/encoding.
    """

    encoders: dict = field(default_factory=dict)
    feature_min: pd.Series = None
    feature_max: pd.Series = None

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        for col in CATEGORICAL_COLUMNS:
            enc = LabelEncoder()
            enc.fit(df[col])
            self.encoders[col] = enc

        features = self._encode_categoricals(df).drop(columns=NON_FEATURE_COLUMNS)
        self.feature_min = features.min()
        self.feature_max = features.max()
        return self

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in CATEGORICAL_COLUMNS:
            enc = self.encoders[col]
            known = set(enc.classes_)
            # Any category unseen at fit time gets its own out-of-vocabulary
            # bucket rather than crashing LabelEncoder.transform().
            df[col] = df[col].where(df[col].isin(known), other=enc.classes_[0])
            df[col] = enc.transform(df[col])
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        features = self._encode_categoricals(df).drop(columns=NON_FEATURE_COLUMNS)
        feature_range = (self.feature_max - self.feature_min).replace(0, 1)
        # Not clipped to [0, 1]: values outside the training range (e.g. a
        # src_bytes flood larger than anything seen in training) stay
        # informative for distance-based models instead of being flattened.
        return (features - self.feature_min) / feature_range
