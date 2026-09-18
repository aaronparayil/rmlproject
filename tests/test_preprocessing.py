import numpy as np
import pandas as pd
import pytest

from src.data.loader import LIARDataLoader
from src.data.preprocessor import LIARPreprocessor
from src.features.credit_history import (
    HISTORY_PROPORTION_COLS,
    LeakFreeCreditHistory,
)
from src.features.engineer import FeatureEngineer
from src.features.text_features import extract_text_statistics


def test_text_statistics():
    df = pd.DataFrame({
        "statement": ["This is a test statement.", "Another one with more words here."],
    })
    out = extract_text_statistics(df)
    assert {"statement_char_len", "statement_word_count", "avg_word_length"} <= set(out.columns)
    assert out["statement_word_count"].iloc[0] == 5


def _toy_train():
    # speaker_a: 2 false + 1 true; speaker_b: 1 true only
    return pd.DataFrame({
        "statement": ["s1", "s2", "s3", "s4"],
        "speaker": ["speaker_a", "speaker_a", "speaker_a", "speaker_b"],
        "label": ["false", "false", "true", "true"],
    })


def test_credit_history_proportions_are_normalised():
    train = _toy_train()
    ch = LeakFreeCreditHistory(n_splits=2, smoothing=1.0).fit(train)
    feats = ch.transform(train)
    assert np.allclose(feats[HISTORY_PROPORTION_COLS].sum(axis=1), 1.0)
    assert (feats["speaker_history_size"] >= 0).all()


def test_credit_history_ignores_holdout_labels():
    """Features for unseen speakers must not depend on their own labels."""
    train = _toy_train()
    ch = LeakFreeCreditHistory(n_splits=2, smoothing=1.0).fit(train)

    test_a = pd.DataFrame({"statement": ["x"], "speaker": ["speaker_c"], "label": ["false"]})
    test_b = pd.DataFrame({"statement": ["x"], "speaker": ["speaker_c"], "label": ["true"]})

    fa = ch.transform(test_a)[HISTORY_PROPORTION_COLS].to_numpy()
    fb = ch.transform(test_b)[HISTORY_PROPORTION_COLS].to_numpy()
    # Same unseen speaker, different gold labels -> identical features.
    assert np.allclose(fa, fb)
    assert ch.transform(test_a)["speaker_is_unseen"].iloc[0] == 1.0


def test_out_of_fold_excludes_own_row():
    """A speaker appearing once in train must get an empty history."""
    train = pd.DataFrame({
        "statement": [f"s{i}" for i in range(10)],
        "speaker": [f"solo_{i}" for i in range(10)],
        "label": ["false", "true"] * 5,
    })
    ch = LeakFreeCreditHistory(n_splits=5, smoothing=1.0).fit(train)
    feats = ch.transform_train(train)
    assert (feats["speaker_history_size"] == 0).all()


def test_raw_leaky_counts_are_not_model_features():
    """The shipped LIAR count columns must never reach the feature config."""
    from src.features.feature_union import LIARFeaturePipeline

    numeric = LIARFeaturePipeline().numeric_columns
    for leaky in ["barely_true_c", "false_c", "half_true_c", "mostly_true_c", "pants_fire_c"]:
        assert not any(col.startswith(leaky) for col in numeric), (
            f"Leaky raw count column {leaky} is being used as a model feature."
        )


@pytest.mark.slow
def test_full_pipeline_on_real_data():
    loader = LIARDataLoader(data_dir="data/raw")
    splits = loader.load_all_splits()

    train, valid, test = LIARPreprocessor().preprocess_pipeline(
        splits["train"], splits["valid"], splits["test"]
    )

    for df in (train, valid, test):
        assert df["label_idx"].between(0, 5).all()
        assert df["is_true"].isin([0, 1]).all()
        assert df[HISTORY_PROPORTION_COLS].notna().all().all()

    assert "credibility_score_minmax" in train.columns
    assert "credibility_score_zscore" in train.columns
    assert train["credibility_score_minmax"].between(0.0, 1.0).all()
