from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack, csr_matrix, issparse

class LIARFeaturePipeline:
    """Combines statement TF-IDF features with tabular numerical and categorical metadata."""

    def __init__(
        self,
        text_column: str = "statement",
        numeric_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None,
        tfidf_params: Optional[dict] = None,
        feature_mode: str = "hybrid"  # "hybrid", "text_only", or "metadata_only"
    ):
        self.text_column = text_column
        self.numeric_columns = numeric_columns or [
            "barely_true_c_zscore", "false_c_zscore", "half_true_c_zscore",
            "mostly_true_c_zscore", "pants_fire_c_zscore", "credibility_score_zscore",
            "total_history_count_zscore", "statement_char_len_zscore",
            "statement_word_count_zscore", "avg_word_length_zscore"
        ]
        self.categorical_columns = categorical_columns or ["party", "state"]
        self.feature_mode = feature_mode

        tfidf_args = tfidf_params or {
            "max_features": 5000,
            "ngram_range": (1, 2),
            "sublinear_tf": True,
            "min_df": 2,
            "max_df": 0.95,
            "stop_words": "english"
        }
        if "ngram_range" in tfidf_args and isinstance(tfidf_args["ngram_range"], list):
            tfidf_args["ngram_range"] = tuple(tfidf_args["ngram_range"])

        self.tfidf = TfidfVectorizer(**tfidf_args)
        self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> "LIARFeaturePipeline":
        """Fit feature extractors strictly on the training set."""
        if self.feature_mode in ["hybrid", "text_only"]:
            texts = df[self.text_column].fillna("").astype(str)
            self.tfidf.fit(texts)

        if self.feature_mode in ["hybrid", "metadata_only"]:
            cats = df[self.categorical_columns].fillna("Unknown").astype(str)
            self.ohe.fit(cats)

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> csr_matrix:
        """Transform dataframe into a unified feature matrix."""
        if not self.is_fitted:
            raise ValueError("LIARFeaturePipeline must be fitted before transforming data.")

        matrices = []

        if self.feature_mode in ["hybrid", "text_only"]:
            texts = df[self.text_column].fillna("").astype(str)
            text_feat = self.tfidf.transform(texts)
            matrices.append(text_feat)

        if self.feature_mode in ["hybrid", "metadata_only"]:
            # Numeric features
            active_num = [c for c in self.numeric_columns if c in df.columns]
            if active_num:
                num_vals = df[active_num].fillna(0).values
                num_sparse = csr_matrix(num_vals)
                matrices.append(num_sparse)

            # Categorical features
            cats = df[self.categorical_columns].fillna("Unknown").astype(str)
            cat_feat = self.ohe.transform(cats)
            matrices.append(cat_feat)

        if len(matrices) == 1:
            return matrices[0]
        return hstack(matrices, format="csr")

    def fit_transform(self, df: pd.DataFrame) -> csr_matrix:
        """Fit extractors and transform dataframe."""
        return self.fit(df).transform(df)
