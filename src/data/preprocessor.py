"""LIAR preprocessing: cleaning, label encoding, and leakage-free feature assembly.

The split pipeline is deliberately asymmetric. Speaker credit history is fitted
on the training split alone and applied out-of-fold to training rows, so no row
contributes to its own features and no held-out label is ever observed. See
:mod:`src.features.credit_history` for the evidence motivating this.
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from ..features.credit_history import LeakFreeCreditHistory
from ..features.engineer import FeatureEngineer
from ..features.text_features import extract_text_statistics

#: Columns a row must have in order to be usable.
DEFAULT_CORE_COLS = ["label", "statement", "party"]

LABEL_TO_INT_6CLASS = {
    "pants-fire": 0,
    "false": 1,
    "barely-true": 2,
    "half-true": 3,
    "mostly-true": 4,
    "true": 5,
}

#: LIAR's binary collapse: the three more-truthful labels against the rest.
TRUE_LEANING_LABELS = ["half-true", "mostly-true", "true"]


class LIARPreprocessor:
    def __init__(
        self,
        core_cols: Optional[List[str]] = None,
        label_mapping: Optional[dict] = None,
        history_smoothing: float = 10.0,
        history_folds: int = 5,
        seed: int = 42,
    ) -> None:
        self.core_cols = core_cols or DEFAULT_CORE_COLS
        self.label_mapping = label_mapping or LABEL_TO_INT_6CLASS
        self.credit_history = LeakFreeCreditHistory(
            n_splits=history_folds, smoothing=history_smoothing, seed=seed
        )
        self.feature_engineer = FeatureEngineer()

    # ------------------------------------------------------------------
    # per-split cleaning
    # ------------------------------------------------------------------
    def clean_strings(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        for c in df_out.select_dtypes(include="object").columns:
            df_out[c] = df_out[c].astype(str).str.strip()
            df_out[c] = df_out[c].replace({"": np.nan, "nan": np.nan, "None": np.nan})
        return df_out

    def remove_duplicates(self, df: pd.DataFrame, subset_col: str = "id") -> pd.DataFrame:
        if subset_col in df.columns:
            return df.drop_duplicates(subset=[subset_col]).reset_index(drop=True)
        return df.drop_duplicates().reset_index(drop=True)

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        active_core = [c for c in self.core_cols if c in df_out.columns]
        df_out = df_out.dropna(subset=active_core).reset_index(drop=True)

        for cat_col in ["job_title", "state", "context", "subject", "speaker"]:
            if cat_col in df_out.columns:
                df_out[cat_col] = df_out[cat_col].fillna("Unknown")

        return df_out

    def encode_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        if "label" in df_out.columns:
            df_out["label_idx"] = df_out["label"].map(self.label_mapping).fillna(-1).astype(int)
            df_out["is_true"] = df_out["label"].isin(TRUE_LEANING_LABELS).astype(int)
            df_out = df_out[df_out["label_idx"] >= 0].reset_index(drop=True)
        return df_out

    def preprocess_single(self, df: pd.DataFrame) -> pd.DataFrame:
        """Split-local steps only: nothing here crosses split boundaries."""
        df_clean = self.clean_strings(df)
        df_dedup = self.remove_duplicates(df_clean)
        df_imputed = self.handle_missing_values(df_dedup)
        df_text = extract_text_statistics(df_imputed)
        return self.encode_labels(df_text)

    # ------------------------------------------------------------------
    # full split pipeline
    # ------------------------------------------------------------------
    def preprocess_pipeline(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_prep = self.preprocess_single(train_df)
        valid_prep = self.preprocess_single(valid_df)
        test_prep = self.preprocess_single(test_df)

        # Credit history: fitted on train labels only, out-of-fold for train rows.
        self.credit_history.fit(train_prep)
        train_prep = train_prep.join(self.credit_history.transform_train(train_prep))
        valid_prep = valid_prep.join(self.credit_history.transform(valid_prep))
        test_prep = test_prep.join(self.credit_history.transform(test_prep))

        # Scalers: fitted on train only.
        return self.feature_engineer.fit_transform_split(train_prep, valid_prep, test_prep)
