from typing import List, Optional

import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

#: Leakage-free numeric features. The raw LIAR ``*_c`` counts are excluded by
#: design; see src/features/credit_history.py.
DEFAULT_NUMERIC_COLUMNS = [
    "hist_p_pants_fire_zscore", "hist_p_false_zscore",
    "hist_p_barely_true_zscore", "hist_p_half_true_zscore",
    "hist_p_mostly_true_zscore", "hist_p_true_zscore",
    "credibility_score_zscore", "speaker_history_log_size_zscore",
    "speaker_is_unseen",
    "statement_char_len_zscore", "statement_word_count_zscore",
    "avg_word_length_zscore",
]

#: Speaker identity is a legitimate predictor: it is observed at inference time
#: and its association with veracity is learned from training rows only.
DEFAULT_CATEGORICAL_COLUMNS = ["party", "state", "speaker"]


def _split_subjects(value: str) -> List[str]:
    """Split LIAR's comma-separated subject tags into individual topics."""
    return [tok.strip() for tok in str(value).split(",") if tok.strip()]


class LIARFeaturePipeline:
    """Combines statement TF-IDF features with tabular numerical and categorical metadata."""

    def __init__(
        self,
        text_column: str = "statement",
        numeric_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None,
        tfidf_params: Optional[dict] = None,
        feature_mode: str = "hybrid",  # "hybrid", "text_only", or "metadata_only"
        subject_column: Optional[str] = "subject",
        char_ngrams: bool = False,
        char_tfidf_params: Optional[dict] = None,
    ):
        self.text_column = text_column
        self.numeric_columns = numeric_columns or list(DEFAULT_NUMERIC_COLUMNS)
        self.categorical_columns = categorical_columns or list(DEFAULT_CATEGORICAL_COLUMNS)
        self.feature_mode = feature_mode
        self.subject_column = subject_column
        self.char_ngrams = char_ngrams

        tfidf_args = dict(tfidf_params or {
            "max_features": 5000,
            "ngram_range": (1, 2),
            "sublinear_tf": True,
            "min_df": 2,
            "max_df": 0.95,
            "stop_words": "english",
        })
        if isinstance(tfidf_args.get("ngram_range"), list):
            tfidf_args["ngram_range"] = tuple(tfidf_args["ngram_range"])
        self.tfidf = TfidfVectorizer(**tfidf_args)

        char_args = dict(char_tfidf_params or {
            "analyzer": "char_wb",
            "ngram_range": (3, 5),
            "max_features": 20000,
            "sublinear_tf": True,
            "min_df": 3,
        })
        if isinstance(char_args.get("ngram_range"), list):
            char_args["ngram_range"] = tuple(char_args["ngram_range"])
        self.char_tfidf = TfidfVectorizer(**char_args) if char_ngrams else None

        self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        # Subject is multi-label ("energy,history,jobs"), so it is tokenised on
        # commas rather than one-hot encoded over whole combinations.
        self.subject_vectorizer = TfidfVectorizer(
            tokenizer=_split_subjects, lowercase=True, min_df=2, token_pattern=None
        )
        self.is_fitted = False

    # ------------------------------------------------------------------
    @property
    def _uses_text(self) -> bool:
        return self.feature_mode in ("hybrid", "text_only")

    @property
    def _uses_metadata(self) -> bool:
        return self.feature_mode in ("hybrid", "metadata_only")

    def _texts(self, df: pd.DataFrame) -> pd.Series:
        return df[self.text_column].fillna("").astype(str)

    def _active_cats(self, df: pd.DataFrame) -> List[str]:
        return [c for c in self.categorical_columns if c in df.columns]

    def _cats(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[self._active_cats(df)].fillna("Unknown").astype(str)

    def _has_subject(self, df: pd.DataFrame) -> bool:
        return bool(self.subject_column) and self.subject_column in df.columns

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "LIARFeaturePipeline":
        """Fit feature extractors strictly on the training set."""
        if self._uses_text:
            self.tfidf.fit(self._texts(df))
            if self.char_tfidf is not None:
                self.char_tfidf.fit(self._texts(df))

        if self._uses_metadata:
            if self._active_cats(df):
                self.ohe.fit(self._cats(df))
            if self._has_subject(df):
                self.subject_vectorizer.fit(df[self.subject_column].fillna("").astype(str))

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> csr_matrix:
        """Transform dataframe into a unified feature matrix."""
        if not self.is_fitted:
            raise ValueError("LIARFeaturePipeline must be fitted before transforming data.")

        matrices = []

        if self._uses_text:
            matrices.append(self.tfidf.transform(self._texts(df)))
            if self.char_tfidf is not None:
                matrices.append(self.char_tfidf.transform(self._texts(df)))

        if self._uses_metadata:
            active_num = [c for c in self.numeric_columns if c in df.columns]
            if active_num:
                matrices.append(csr_matrix(df[active_num].fillna(0).values))

            if self._active_cats(df):
                matrices.append(self.ohe.transform(self._cats(df)))

            if self._has_subject(df):
                matrices.append(
                    self.subject_vectorizer.transform(df[self.subject_column].fillna("").astype(str))
                )

        if not matrices:
            raise ValueError(f"No features produced for feature_mode={self.feature_mode!r}.")
        if len(matrices) == 1:
            return matrices[0]
        return hstack(matrices, format="csr")

    def fit_transform(self, df: pd.DataFrame) -> csr_matrix:
        """Fit extractors and transform dataframe."""
        return self.fit(df).transform(df)
