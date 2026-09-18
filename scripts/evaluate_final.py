"""Score the validation-selected configurations on the test split, once.

Reads ``experiments/results/tuning_<task>.json`` (produced by
``scripts/tune_baselines.py``) and evaluates each winning configuration on the
held-out test split. Models are refit on the training split only, so the
credit-history encoder -- which is fitted on train -- stays consistent with the
reported baselines.
"""

import argparse
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd

from src.features.feature_union import LIARFeaturePipeline
from src.models.classical import get_baseline_model
from src.utils.logger import setup_logger
from src.utils.metrics import compute_metrics
from scripts.tune_baselines import FEATURE_VARIANTS

LABELS_6CLASS = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]
LABELS_BINARY = ["false_leaning", "true_leaning"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--task", default="6class", choices=["6class", "binary"])
    parser.add_argument("--results-dir", default="experiments/results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger = setup_logger("EvaluateFinal")
    tuning_path = os.path.join(args.results_dir, f"tuning_{args.task}.json")
    if not os.path.exists(tuning_path):
        raise FileNotFoundError(
            f"{tuning_path} not found. Run: python scripts/tune_baselines.py --task {args.task}"
        )
    with open(tuning_path, "r", encoding="utf-8") as fh:
        tuning = json.load(fh)

    train_df = pd.read_csv(os.path.join(args.data_dir, "train_processed.csv"))
    valid_df = pd.read_csv(os.path.join(args.data_dir, "valid_processed.csv"))
    test_df = pd.read_csv(os.path.join(args.data_dir, "test_processed.csv"))

    target = "label_idx" if args.task == "6class" else "is_true"
    is_binary = args.task == "binary"
    names = LABELS_BINARY if is_binary else LABELS_6CLASS
    y_train, y_valid, y_test = (d[target].values for d in (train_df, valid_df, test_df))

    rows = []
    for cell in tuning["best_per_feature_mode_and_model"]:
        variant = FEATURE_VARIANTS[cell["variant"]]
        pipe = LIARFeaturePipeline(
            feature_mode=cell["feature_mode"],
            categorical_columns=variant["cats"],
            subject_column=variant["subject"],
            char_ngrams=cell["char_ngrams"],
        )
        pipe.fit(train_df)
        X_tr, X_va, X_te = (pipe.transform(d) for d in (train_df, valid_df, test_df))

        model = get_baseline_model(cell["model"], config=cell["params"], seed=args.seed)
        model.fit(X_tr, y_train)

        val_m = compute_metrics(y_valid, model.predict(X_va), None, names, is_binary)
        test_m = compute_metrics(y_test, model.predict(X_te), None, names, is_binary)

        rows.append({
            "task": args.task,
            "feature_mode": cell["feature_mode"],
            "model": cell["model"],
            "variant": cell["variant"],
            "char_ngrams": cell["char_ngrams"],
            "params": json.dumps(cell["params"]),
            "val_acc": round(val_m["accuracy"], 4),
            "test_acc": round(test_m["accuracy"], 4),
            "test_f1_macro": round(test_m["f1_macro"], 4),
            "test_f1_weighted": round(test_m["f1_weighted"], 4),
        })
        logger.info("%s / %s -> test_acc=%.4f", cell["feature_mode"], cell["model"], test_m["accuracy"])

    df = pd.DataFrame(rows).sort_values(["feature_mode", "test_acc"], ascending=[True, False])
    out_csv = os.path.join(args.results_dir, f"tuned_{args.task}_test.csv")
    df.to_csv(out_csv, index=False)

    print("\n" + "=" * 100)
    print(f"FINAL TEST RESULTS ({args.task}) - configurations selected on validation")
    print("=" * 100)
    print(df.to_string(index=False))
    print("=" * 100)
    logger.info("Wrote %s", out_csv)


if __name__ == "__main__":
    main()
