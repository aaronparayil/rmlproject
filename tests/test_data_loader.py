import pytest
from src.data.loader import LIARDataLoader

def test_loader_raw_splits():
    loader = LIARDataLoader(data_dir="data/raw")
    splits = loader.load_all_splits()
    
    assert "train" in splits
    assert "valid" in splits
    assert "test" in splits
    
    assert len(splits["train"]) > 10000
    assert len(splits["valid"]) > 1000
    assert len(splits["test"]) > 1000
    
    for name, df in splits.items():
        assert "statement" in df.columns
        assert "label" in df.columns
        assert "speaker" in df.columns
        assert "split" in df.columns
        assert (df["split"] == name).all()

def test_loader_combined():
    loader = LIARDataLoader(data_dir="data/raw")
    splits = loader.load_all_splits()
    combined = loader.load_combined()
    expected_len = sum(len(s) for s in splits.values())
    assert len(combined) == expected_len
