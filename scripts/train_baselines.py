import os
import sys
import json
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.metrics import compute_metrics, classification_report_dict
from src.features.feature_union import LIARFeaturePipeline
from src.models.classical import get_baseline_model, list_available_models

LABELS_6CLASS = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]
LABELS_BINARY = ["false_leaning", "true_leaning"]

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate baseline ML models on LIAR dataset.")
    parser.add_argument("--config", type=str, default="configs/baseline_config.yaml", help="Path to baseline config.")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Path to processed data directory.")
    parser.add_argument("--task", type=str, default="6class", choices=["6class", "binary"], help="Task type.")
    parser.add_argument("--feature-mode", type=str, default="hybrid", choices=["hybrid", "text_only", "metadata_only"], help="Feature mode.")
    parser.add_argument("--output-dir", type=str, default="experiments/results", help="Directory to save evaluation results.")
    args = parser.parse_args()

    logger = setup_logger("TrainBaselines")
    logger.info(f"Initializing Baseline Training Suite (Task: {args.task}, Feature Mode: {args.feature_mode})")

    os.makedirs(args.output_dir, exist_ok=True)
    cfg = load_config(args.config).get("baseline_experiment", {})
    seed = cfg.get("seed", 42)

    # 1. Load processed splits
    train_path = os.path.join(args.data_dir, "train_processed.csv")
    valid_path = os.path.join(args.data_dir, "valid_processed.csv")
    test_path = os.path.join(args.data_dir, "test_processed.csv")

    if not all(os.path.exists(p) for p in [train_path, valid_path, test_path]):
        raise FileNotFoundError("Processed split files not found in data/processed/. Please run scripts/prepare_data.py first.")

    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)
    test_df = pd.read_csv(test_path)

    logger.info(f"Loaded splits - Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")

    # 2. Extract Features
    pipeline = LIARFeaturePipeline(
        text_column=cfg.get("text_column", "statement"),
        numeric_columns=cfg.get("numeric_features"),
        categorical_columns=cfg.get("categorical_features"),
        tfidf_params=cfg.get("tfidf"),
        feature_mode=args.feature_mode
    )

    logger.info("Fitting feature pipeline on Train split...")
    pipeline.fit(train_df)
    X_train = pipeline.transform(train_df)
    X_valid = pipeline.transform(valid_df)
    X_test = pipeline.transform(test_df)

    logger.info(f"Feature Matrix Shapes - Train: {X_train.shape}, Valid: {X_valid.shape}, Test: {X_test.shape}")

    # 3. Targets
    target_col = "label_idx" if args.task == "6class" else "is_true"
    is_binary = (args.task == "binary")
    target_names = LABELS_BINARY if is_binary else LABELS_6CLASS

    y_train = train_df[target_col].values
    y_valid = valid_df[target_col].values
    y_test = test_df[target_col].values

    # 4. Train and Evaluate Each Model
    results = []
    models_to_run = list_available_models()
    model_configs = cfg.get("models", {})

    for model_name in models_to_run:
        logger.info(f"--> Training model: {model_name}")
        model_cfg = model_configs.get(model_name, {})
        model = get_baseline_model(model_name, config=model_cfg, seed=seed)

        # Fit
        model.fit(X_train, y_train)

        # Predict Valid
        val_pred = model.predict(X_valid)
        val_prob = model.predict_proba(X_valid)
        val_metrics = compute_metrics(y_valid, val_pred, val_prob, target_names=target_names, is_binary=is_binary)

        # Predict Test
        test_pred = model.predict(X_test)
        test_prob = model.predict_proba(X_test)
        test_metrics = compute_metrics(y_test, test_pred, test_prob, target_names=target_names, is_binary=is_binary)

        row = {
            "model": model.model_name,
            "task": args.task,
            "feature_mode": args.feature_mode,
            "val_acc": round(val_metrics["accuracy"], 4),
            "val_f1_macro": round(val_metrics["f1_macro"], 4),
            "val_f1_weighted": round(val_metrics["f1_weighted"], 4),
            "test_acc": round(test_metrics["accuracy"], 4),
            "test_f1_macro": round(test_metrics["f1_macro"], 4),
            "test_f1_weighted": round(test_metrics["f1_weighted"], 4)
        }
        if is_binary:
            row["val_f1_positive"] = round(val_metrics["f1_positive"], 4)
            row["test_f1_positive"] = round(test_metrics["f1_positive"], 4)
        results.append(row)
        logger.info(f"    [{model.model_name}] Val Acc: {row['val_acc']:.4f} | Val F1-Macro: {row['val_f1_macro']:.4f} | Test Acc: {row['test_acc']:.4f} | Test F1-Macro: {row['test_f1_macro']:.4f}")

    # 5. Output summary tables and JSON
    results_df = pd.DataFrame(results)
    csv_out = os.path.join(args.output_dir, f"baseline_{args.task}_{args.feature_mode}.csv")
    json_out = os.path.join(args.output_dir, f"baseline_{args.task}_{args.feature_mode}.json")

    results_df.to_csv(csv_out, index=False)
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved baseline results to {csv_out} and {json_out}")
    print("\n" + "=" * 80)
    print(f"BASELINE COMPARISON RESULTS (Task: {args.task.upper()}, Features: {args.feature_mode.upper()})")
    print("=" * 80)
    print(results_df.to_string(index=False))
    print("=" * 80)

if __name__ == "__main__":
    main()
