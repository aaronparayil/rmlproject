from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

DEFAULT_CREDIBILITY_WEIGHTS = {
    "barely_true_c": -1.0,
    "false_c": -2.0,
    "half_true_c": 0.5,
    "mostly_true_c": 1.0,
    "pants_fire_c": -3.0
}

COUNT_COLS = ["barely_true_c", "false_c", "half_true_c", "mostly_true_c", "pants_fire_c"]

class FeatureEngineer:
    def __init__(
        self,
        credibility_weights: Optional[Dict[str, float]] = None,
        scale_columns: Optional[List[str]] = None
    ):
        self.credibility_weights = credibility_weights or DEFAULT_CREDIBILITY_WEIGHTS
        self.scale_columns = scale_columns or [
            "barely_true_c", "false_c", "half_true_c", "mostly_true_c", "pants_fire_c",
            "statement_char_len", "statement_word_count", "avg_word_length",
            "total_history_count", "credibility_score"
        ]
        self.minmax_scaler = MinMaxScaler()
        self.standard_scaler = StandardScaler()
        self.is_fitted = False

    def compute_credit_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        for col in COUNT_COLS:
            if col in df_out.columns:
                df_out[col] = pd.to_numeric(df_out[col], errors="coerce").fillna(0)

        df_out["total_history_count"] = df_out[COUNT_COLS].sum(axis=1)

        df_out["credibility_score"] = sum(
            df_out[col] * weight for col, weight in self.credibility_weights.items() if col in df_out.columns
        )

        if "label" in df_out.columns:
            df_out["is_true"] = df_out["label"].isin(["mostly-true", "true"]).astype(int)

        return df_out

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
            minmax_vals = self.minmax_scaler.transform(df_out[cols_to_scale])
            minmax_cols = [f"{c}_minmax" for c in cols_to_scale]
            minmax_df = pd.DataFrame(minmax_vals, columns=minmax_cols, index=df_out.index)

            std_vals = self.standard_scaler.transform(df_out[cols_to_scale])
            std_cols = [f"{c}_zscore" for c in cols_to_scale]
            std_df = pd.DataFrame(std_vals, columns=std_cols, index=df_out.index)

            df_out = pd.concat([df_out, minmax_df, std_df], axis=1)

        return df_out

    def fit_transform_split(
        self,
        train_df: pd.DataFrame,
        valid_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        self.fit(train_df)
        train_trans = self.transform_scaling(train_df)
        val_trans = self.transform_scaling(valid_df) if valid_df is not None else None
        test_trans = self.transform_scaling(test_df) if test_df is not None else None
        return train_trans, val_trans, test_trans
