from typing import List, Optional, Tuple
import pandas as pd
import numpy as np

class TabularDataset:
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "label_idx"
    ):
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.target_col = target_col

    def get_X_y(self) -> Tuple[np.ndarray, np.ndarray]:
        X = self.df[self.feature_cols].values
        y = self.df[self.target_col].values
        return X, y

    def get_statements(self) -> List[str]:
        return self.df["statement"].tolist()

    def __len__(self) -> int:
        return len(self.df)
