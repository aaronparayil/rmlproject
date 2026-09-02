import pytest
import numpy as np
import pandas as pd
from src.features.feature_union import LIARFeaturePipeline
from src.models.classical import get_baseline_model, list_available_models
from src.utils.metrics import compute_metrics

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "statement": [
            "Says the tax rate is highest in history.",
            "Economy created 5 million jobs last quarter.",
            "Global warming is a hoax created by others.",
            "Health care premiums went down by 20 percent."
        ],
        "party": ["republican", "democrat", "republican", "democrat"],
        "state": ["Texas", "California", "Florida", "New York"],
        "barely_true_c_zscore": [-0.5, 1.2, 0.0, -0.2],
        "false_c_zscore": [1.5, -0.4, 2.0, -0.8],
        "half_true_c_zscore": [0.0, 0.5, -0.5, 1.0],
        "mostly_true_c_zscore": [-0.8, 1.5, -1.0, 0.8],
        "pants_fire_c_zscore": [2.0, -0.5, 1.0, -0.5],
        "credibility_score_zscore": [-1.2, 1.4, -0.9, 0.7],
        "total_history_count_zscore": [0.5, 1.0, 0.2, 0.4],
        "statement_char_len_zscore": [0.1, -0.2, 0.3, -0.1],
        "statement_word_count_zscore": [0.0, -0.1, 0.2, 0.0],
        "avg_word_length_zscore": [0.2, -0.3, 0.1, -0.2],
        "label_idx": [0, 5, 1, 4],
        "is_true": [0, 1, 0, 1]
    })

def test_feature_pipeline(sample_df):
    pipeline = LIARFeaturePipeline(feature_mode="hybrid")
    X = pipeline.fit_transform(sample_df)
    assert X.shape[0] == len(sample_df)
    assert X.shape[1] > 10

def test_models_fit_and_predict(sample_df):
    pipeline = LIARFeaturePipeline(feature_mode="hybrid")
    X = pipeline.fit_transform(sample_df)
    y = sample_df["is_true"].values

    for model_name in list_available_models():
        model = get_baseline_model(model_name)
        model.fit(X, y)
        preds = model.predict(X)
        probs = model.predict_proba(X)
        
        assert len(preds) == len(sample_df)
        if probs is not None:
            assert probs.shape[0] == len(sample_df)

        metrics = compute_metrics(y, preds, probs, is_binary=True)
        assert "accuracy" in metrics
        assert "f1_macro" in metrics
