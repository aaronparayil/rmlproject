"""Numeric feature assembly and split-safe scaling.

Speaker credit history is produced by :mod:`src.features.credit_history`, which
recomputes it from training labels only. The raw ``*_c`` columns shipped with
LIAR are deliberately **not** used as model features -- see that module for the
measurements showing they encode the current statement's own label and the
labels of the held-out splits.
"""

from typing import List, Optional, Tuple

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .credit_history import (
    HISTORY_PROPORTION_COLS,
    LeakFreeCreditHistory,
)

#: Raw LIAR credit-history columns. Retained in the processed CSVs for the
#: leakage analysis, never fed to a model.
LEAKY_RAW_COUNT_COLS = [
    "barely_true_c", "false_c", "half_true_c", "mostly_true_c", "pants_fire_c",
]

#: Text-derived statistics computed per statement.
TEXT_STAT_COLS = [
    "statement_char_len", "statement_word_count", "avg_word_length",
]

#: Numeric columns that receive z-score / min-max variants.
DEFAULT_SCALE_COLUMNS = (
    HISTORY_PROPORTION_COLS
    + ["speaker_history_log_size", "credibility_score"]
    + TEXT_STAT_COLS
)


class FeatureEngineer:
    """Fits numeric scalers on the training split and applies them elsewhere."""

    def __init__(self, scale_columns: Optional[List[str]] = None) -> None:
        self.scale_columns = scale_columns or DEFAULT_SCALE_COLUMNS
        self.minmax_scaler = MinMaxScaler()
        self.standard_scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, train_df: pd.DataFrame) -> "FeatureEngineer":
        cols_to_scale = [c for c in self.scale_columns if c in train_df.columns]
        if cols_to_scale:
            self.minmax_scaler.fit(train_df[cols_to_scale])
            self.standard_scaler.fit(train_df[cols_to_scale])
            self.is_fitted = True
        return self

    def transform_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted before transforming scaling.")

        df_out = df.copy()
        cols_to_scale = [c for c in self.scale_columns if c in df_out.columns]

        if cols_to_scale:
            minmax_df = pd.DataFrame(
                self.minmax_scaler.transform(df_out[cols_to_scale]),
                columns=[f"{c}_minmax" for c in cols_to_scale],
                index=df_out.index,
            )
            std_df = pd.DataFrame(
                self.standard_scaler.transform(df_out[cols_to_scale]),
                columns=[f"{c}_zscore" for c in cols_to_scale],
                index=df_out.index,
            )
            df_out = pd.concat([df_out, minmax_df, std_df], axis=1)

        return df_out

    def fit_transform_split(
        self,
        train_df: pd.DataFrame,
        valid_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Fit scalers on ``train_df`` only, then transform every split."""
        self.fit(train_df)
        train_trans = self.transform_scaling(train_df)
        val_trans = self.transform_scaling(valid_df) if valid_df is not None else None
        test_trans = self.transform_scaling(test_df) if test_df is not None else None
        return train_trans, val_trans, test_trans


__all__ = [
    "FeatureEngineer",
    "LeakFreeCreditHistory",
    "LEAKY_RAW_COUNT_COLS",
    "TEXT_STAT_COLS",
    "DEFAULT_SCALE_COLUMNS",
]
