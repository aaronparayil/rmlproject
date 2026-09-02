import pytest
import pandas as pd
import numpy as np
from src.data.loader import LIARDataLoader
from src.data.preprocessor import LIARPreprocessor
from src.features.engineer import FeatureEngineer
from src.features.text_features import extract_text_statistics

def test_feature_engineering_computation():
    sample_data = {
        "statement": ["This is a test statement.", "Another one with more words here."],
        "label": ["true", "false"],
        "barely_true_c": [1, 2],
        "false_c": [0, 5],
        "half_true_c": [2, 1],
        "mostly_true_c": [4, 0],
        "pants_fire_c": [0, 3]
    }
    df = pd.DataFrame(sample_data)
    df_text = extract_text_statistics(df)
    
    assert "statement_char_len" in df_text.columns
    assert "statement_word_count" in df_text.columns
    assert "avg_word_length" in df_text.columns
    assert df_text["statement_word_count"].iloc[0] == 5
    
    fe = FeatureEngineer()
    df_feat = fe.compute_credit_features(df_text)
    
    assert "total_history_count" in df_feat.columns
    assert "credibility_score" in df_feat.columns
    assert "is_true" in df_feat.columns
    assert df_feat["total_history_count"].iloc[0] == (1 + 0 + 2 + 4 + 0)
    assert df_feat["is_true"].iloc[0] == 1
    assert df_feat["is_true"].iloc[1] == 0

def test_leakage_free_pipeline():
    loader = LIARDataLoader(data_dir="data/raw")
    splits = loader.load_all_splits()
    
    preprocessor = LIARPreprocessor()
    train_proc, val_proc, test_proc = preprocessor.preprocess_pipeline(
        splits["train"], splits["valid"], splits["test"]
    )
    
    # Check that output has no nulls in critical columns
    assert train_proc["label_idx"].isna().sum() == 0
    assert val_proc["label_idx"].isna().sum() == 0
    assert test_proc["label_idx"].isna().sum() == 0
    
    # Check that scaled minmax columns are in valid ranges
    assert "credibility_score_minmax" in train_proc.columns
    assert "credibility_score_zscore" in train_proc.columns
    
    assert (train_proc["credibility_score_minmax"] >= 0.0).all()
    assert (train_proc["credibility_score_minmax"] <= 1.0).all()
