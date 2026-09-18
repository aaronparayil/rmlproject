"""Leakage-free speaker credit-history features for the LIAR dataset.

The credit-history columns shipped with LIAR (``barely_true_c`` ... ``pants_fire_c``)
cannot be used as-is. Empirically they are:

1. constant per speaker (3314/3318 speakers), and
2. equal to that speaker's label histogram over the **entire corpus**, i.e. the
   train, validation *and* test splits combined (exact for 2846/2947 speakers).

Two distinct leaks follow. The current statement is counted in its own features
(1689/1701 single-statement speakers are a one-hot encoding of their own label),
and test-set labels are visible at training time through the shared aggregate.

The self-inclusion is documented: Wang (2017), who released LIAR, writes that
"since this vector also includes the count for the current statement, it is
important to subtract the current label from the credit history when using this
meta data vector in prediction experiments."

That prescribed correction does **not** repair the leak -- it widens it. Because
the per-speaker vector ``B`` is constant and therefore memorisable (3318
speakers share only 392 distinct vectors), ``B - e_y`` identifies ``y``
outright for 36.2% of test rows, and history-only 6-class accuracy rises from
0.3944 to 0.6594. Subtraction also leaves the second leak untouched, since the
counts still aggregate over the held-out splits.

This module instead recomputes the history from scratch under two rules:

* only **training** labels ever contribute to any feature, so validation and
  test labels are never observed; and
* training rows are encoded **out-of-fold** -- a row's own fold is excluded from
  its own history -- so no row contributes to its own features.

Counts are converted to prior-smoothed proportions, which keeps speakers with a
single statement from producing saturated, memorisable vectors.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

#: Canonical LIAR label order, least to most truthful.
LABEL_ORDER: List[str] = [
    "pants-fire", "false", "barely-true", "half-true", "mostly-true", "true",
]

#: Column-safe slug per label.
LABEL_SLUG: Dict[str, str] = {
    "pants-fire": "pants_fire",
    "false": "false",
    "barely-true": "barely_true",
    "half-true": "half_true",
    "mostly-true": "mostly_true",
    "true": "true",
}

#: Proportion feature emitted per label.
HISTORY_PROPORTION_COLS: List[str] = [f"hist_p_{LABEL_SLUG[l]}" for l in LABEL_ORDER]

#: Auxiliary features describing the reliability of the history estimate.
HISTORY_AUX_COLS: List[str] = [
    "speaker_history_size",
    "speaker_history_log_size",
    "speaker_is_unseen",
    "credibility_score",
]

HISTORY_FEATURE_COLS: List[str] = HISTORY_PROPORTION_COLS + HISTORY_AUX_COLS

#: Ordinal truthfulness weight per label, used to collapse the history into a
#: single expected-credibility scalar in [-3, 2].
DEFAULT_CREDIBILITY_WEIGHTS: Dict[str, float] = {
    "pants-fire": -3.0,
    "false": -2.0,
    "barely-true": -1.0,
    "half-true": 0.5,
    "mostly-true": 1.0,
    "true": 2.0,
}

#: Speaker values that carry no identity and must not be pooled into one bucket.
_ANONYMOUS_SPEAKERS = {"unknown", "nan", "none", ""}


class LeakFreeCreditHistory:
    """Out-of-fold speaker credit history computed from training labels only.

    Parameters
    ----------
    n_splits:
        Number of out-of-fold partitions used to encode the training split.
    smoothing:
        Strength of the pull toward the global training label prior. Larger
        values shrink sparse speakers harder toward the corpus average.
    seed:
        Seed for the stratified fold assignment.
    """

    def __init__(
        self,
        n_splits: int = 5,
        smoothing: float = 10.0,
        seed: int = 42,
        speaker_col: str = "speaker",
        label_col: str = "label",
        credibility_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.n_splits = n_splits
        self.smoothing = smoothing
        self.seed = seed
        self.speaker_col = speaker_col
        self.label_col = label_col
        self.credibility_weights = credibility_weights or DEFAULT_CREDIBILITY_WEIGHTS
        self.prior_: Optional[np.ndarray] = None
        self.speaker_counts_: Optional[pd.DataFrame] = None
        self.is_fitted = False

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _speaker_key(self, df: pd.DataFrame) -> pd.Series:
        """Normalise speaker ids, mapping anonymous placeholders to NaN."""
        keys = df[self.speaker_col].astype(str).str.strip().str.lower()
        return keys.mask(keys.isin(_ANONYMOUS_SPEAKERS))

    def _count_matrix(self, keys: pd.Series, labels: pd.Series) -> pd.DataFrame:
        """Per-speaker label histogram as a (speaker x label) frame."""
        valid = keys.notna() & labels.isin(LABEL_ORDER)
        counts = (
            pd.crosstab(keys[valid], labels[valid])
            .reindex(columns=LABEL_ORDER, fill_value=0)
            .astype(float)
        )
        return counts

    def _to_features(self, counts: np.ndarray, index: pd.Index) -> pd.DataFrame:
        """Convert raw history counts into smoothed proportions plus aggregates."""
        totals = counts.sum(axis=1)
        smoothed = counts + self.smoothing * self.prior_[None, :]
        proportions = smoothed / (totals[:, None] + self.smoothing)

        out = pd.DataFrame(proportions, columns=HISTORY_PROPORTION_COLS, index=index)
        weights = np.array([self.credibility_weights[l] for l in LABEL_ORDER])
        out["speaker_history_size"] = totals
        out["speaker_history_log_size"] = np.log1p(totals)
        out["speaker_is_unseen"] = (totals == 0).astype(float)
        out["credibility_score"] = proportions @ weights
        return out

    # ------------------------------------------------------------------
    # fit / transform
    # ------------------------------------------------------------------
    def fit(self, train_df: pd.DataFrame) -> "LeakFreeCreditHistory":
        """Learn the global prior and per-speaker histogram from the train split."""
        labels = train_df[self.label_col]
        counts = labels.value_counts().reindex(LABEL_ORDER, fill_value=0).astype(float)
        total = counts.sum()
        if total <= 0:
            raise ValueError("Training split contains no recognised LIAR labels.")
        self.prior_ = (counts / total).to_numpy()
        self.speaker_counts_ = self._count_matrix(self._speaker_key(train_df), labels)
        self.is_fitted = True
        return self

    def transform_train(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Encode the training split out-of-fold.

        A row's history is built from the training rows outside its own fold,
        so no statement contributes to its own features.
        """
        self._check_fitted()
        keys = self._speaker_key(train_df)
        labels = train_df[self.label_col]

        folds = np.full(len(train_df), -1, dtype=int)
        splitter = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.seed
        )
        for fold_id, (_, holdout_idx) in enumerate(
            splitter.split(np.zeros(len(train_df)), labels)
        ):
            folds[holdout_idx] = fold_id

        counts = np.zeros((len(train_df), len(LABEL_ORDER)), dtype=float)
        for fold_id in range(self.n_splits):
            in_fold = folds == fold_id
            # History from every training row outside this fold.
            fold_counts = self._count_matrix(keys[in_fold], labels[in_fold])
            oof = self.speaker_counts_.subtract(fold_counts, fill_value=0.0)
            oof = oof.reindex(columns=LABEL_ORDER, fill_value=0.0).clip(lower=0.0)
            counts[in_fold] = self._lookup(keys[in_fold], oof)

        return self._to_features(counts, train_df.index)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode a held-out split using the full training history."""
        self._check_fitted()
        counts = self._lookup(self._speaker_key(df), self.speaker_counts_)
        return self._to_features(counts, df.index)

    # ------------------------------------------------------------------
    def _lookup(self, keys: pd.Series, table: pd.DataFrame) -> np.ndarray:
        """Row-wise history lookup; unseen or anonymous speakers yield zeros."""
        aligned = table.reindex(keys.to_numpy(), fill_value=0.0)
        return np.nan_to_num(aligned.to_numpy(dtype=float), nan=0.0)

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise ValueError(
                "LeakFreeCreditHistory must be fitted on the training split first."
            )
