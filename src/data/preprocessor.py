from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from ..features.text_features import extract_text_statistics
from ..features.engineer import FeatureEngineer

DEFAULT_CORE_COLS = [
    "label", "statement", "party",
    "barely_true_c", "false_c", "half_true_c", "mostly_true_c", "pants_fire_c"
]

LABEL_TO_INT_6CLASS = {
    "pants-fire": 0,
    "false": 1,
    "barely-true": 2,
    "half-true": 3,
    "mostly-true": 4,
    "true": 5
}

class LIARPreprocessor:
    def __init__(
        self,
        core_cols: Optional[List[str]] = None,
        label_mapping: Optional[dict] = None
    ):
        self.core_cols = core_cols or DEFAULT_CORE_COLS
        self.label_mapping = label_mapping or LABEL_TO_INT_6CLASS
        self.feature_engineer = FeatureEngineer()

    def clean_strings(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        obj_cols = df_out.select_dtypes(include="object").columns.tolist()
        for c in obj_cols:
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

        for num_col in ["barely_true_c", "false_c", "half_true_c", "mostly_true_c", "pants_fire_c"]:
            if num_col in df_out.columns:
                df_out[num_col] = pd.to_numeric(df_out[num_col], errors="coerce").fillna(0)

        return df_out

    def encode_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        if "label" in df_out.columns:
            df_out["label_idx"] = df_out["label"].map(self.label_mapping).fillna(-1).astype(int)
            df_out["is_true"] = df_out["label"].isin(["mostly-true", "true"]).astype(int)
        return df_out

    def preprocess_single(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = self.clean_strings(df)
        df_dedup = self.remove_duplicates(df_clean)
        df_imputed = self.handle_missing_values(df_dedup)
        df_text = extract_text_statistics(df_imputed)
        df_feat = self.feature_engineer.compute_credit_features(df_text)
        df_encoded = self.encode_labels(df_feat)
        return df_encoded

    def preprocess_pipeline(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_prep = self.preprocess_single(train_df)
        valid_prep = self.preprocess_single(valid_df)
        test_prep = self.preprocess_single(test_df)

        train_scaled, valid_scaled, test_scaled = self.feature_engineer.fit_transform_split(
            train_prep, valid_prep, test_prep
        )
        return train_scaled, valid_scaled, test_scaled
