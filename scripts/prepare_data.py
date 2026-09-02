import os
import sys
import argparse

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.data.loader import LIARDataLoader
from src.data.preprocessor import LIARPreprocessor

def main():
    parser = argparse.ArgumentParser(description="Prepare and preprocess LIAR dataset splits.")
    parser.add_argument("--config", type=str, default="configs/data_config.yaml", help="Path to data config YAML.")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Output directory for processed splits.")
    args = parser.parse_args()

    logger = setup_logger("PrepareData")
    logger.info("Loading configuration from %s", args.config)
    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})

    raw_dir = data_cfg.get("raw_dir", "data/raw")
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Initializing DataLoader with raw directory: %s", raw_dir)
    loader = LIARDataLoader(data_dir=raw_dir)
    raw_splits = loader.load_all_splits()

    logger.info("Loaded raw splits - Train: %d, Valid: %d, Test: %d",
                len(raw_splits["train"]), len(raw_splits["valid"]), len(raw_splits["test"]))

    logger.info("Running preprocessing and leakage-free feature engineering...")
    preprocessor = LIARPreprocessor()
    train_proc, valid_proc, test_proc = preprocessor.preprocess_pipeline(
        raw_splits["train"],
        raw_splits["valid"],
        raw_splits["test"]
    )

    logger.info("Processed shapes - Train: %s, Valid: %s, Test: %s",
                train_proc.shape, valid_proc.shape, test_proc.shape)

    # Save individual splits
    train_path = os.path.join(output_dir, "train_processed.csv")
    valid_path = os.path.join(output_dir, "valid_processed.csv")
    test_path = os.path.join(output_dir, "test_processed.csv")

    train_proc.to_csv(train_path, index=False)
    valid_proc.to_csv(valid_path, index=False)
    test_proc.to_csv(test_path, index=False)

    # Save combined dataset
    combined_df = pd.concat([train_proc, valid_proc, test_proc], ignore_index=True)
    combined_path = os.path.join(output_dir, "liar_full_processed.csv")
    combined_df.to_csv(combined_path, index=False)

    logger.info("Saved processed splits to:")
    logger.info("  - %s", train_path)
    logger.info("  - %s", valid_path)
    logger.info("  - %s", test_path)
    logger.info("  - %s (combined: %s)", combined_path, combined_df.shape)
    logger.info("Data preparation completed successfully.")

if __name__ == "__main__":
    main()
