import os
import sys
import argparse

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from src.utils.logger import setup_logger
from src.data.loader import LIARDataLoader

def main():
    parser = argparse.ArgumentParser(description="Run quick exploratory data analysis.")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Path to raw data directory.")
    args = parser.parse_args()

    logger = setup_logger("EDA")
    logger.info(f"Loading raw dataset from {args.data_dir}")
    loader = LIARDataLoader(data_dir=args.data_dir)
    splits = loader.load_all_splits()

    combined = pd.concat(list(splits.values()), ignore_index=True)
    logger.info(f"Total rows across splits: {len(combined)}")
    logger.info("Class distribution:\n%s", str(combined['label'].value_counts()))
    logger.info("Top 5 speakers:\n%s", str(combined['speaker'].value_counts().head(5)))
    logger.info("Party distribution:\n%s", str(combined['party'].value_counts().head(5)))

if __name__ == "__main__":
    main()
