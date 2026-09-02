from typing import Dict, List, Optional
import os
import pandas as pd

DEFAULT_COLUMNS = [
    "id", "label", "statement", "subject", "speaker", "job_title",
    "state", "party", "barely_true_c", "false_c", "half_true_c",
    "mostly_true_c", "pants_fire_c", "context"
]

class LIARDataLoader:
    def __init__(
        self,
        data_dir: str = "data/raw",
        columns: Optional[List[str]] = None
    ):
        self.data_dir = data_dir
        self.columns = columns or DEFAULT_COLUMNS

    def load_split(self, split_name: str) -> pd.DataFrame:
        filename = split_name if split_name.endswith(".tsv") else f"{split_name}.tsv"
        filepath = os.path.join(self.data_dir, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"LIAR dataset split file not found at: {filepath}")

        df = pd.read_csv(filepath, sep="\t", header=None, names=self.columns, quoting=3)
        clean_split_tag = split_name.replace(".tsv", "")
        df["split"] = clean_split_tag
        return df

    def load_all_splits(self) -> Dict[str, pd.DataFrame]:
        return {
            "train": self.load_split("train"),
            "valid": self.load_split("valid"),
            "test": self.load_split("test")
        }

    def load_combined(self) -> pd.DataFrame:
        splits = self.load_all_splits()
        return pd.concat(list(splits.values()), ignore_index=True)
