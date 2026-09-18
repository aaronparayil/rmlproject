"""Select feature sets and hyperparameters on the VALIDATION split.

The test split is never read here. Model selection is treated as a single
search over (feature variant x char n-grams x hyperparameters), scored by
validation accuracy; ``scripts/evaluate_final.py`` then scores the chosen
configuration on test exactly once.

Outputs ``experiments/results/tuning_<task>.json`` with the full search trace
and the winning configuration per (feature_mode, model).
"""

import argparse
import itertools
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.features.feature_union import LIARFeaturePipeline
from src.models.classical import get_baseline_model, list_available_models
from src.utils.logger import setup_logger

warnings.filterwarnings("ignore")

#: Metadata column subsets treated as a tunable choice.
FEATURE_VARIANTS = {
    "base":            {"cats": ["party", "state"], "subject": None},
    "speaker":         {"cats": ["party", "state", "speaker"], "subject": None},
    "speaker_subject": {"cats": ["party", "state", "speaker"], "subject": "subject"},
    "all_meta":        {"cats": ["party", "state", "speaker", "job_title", "context"],
                        "subject": "subject"},
}

#: Per-model hyperparameter grids.
PARAM_GRIDS = {
    "logistic_regression": {"C": [0.1, 0.5, 1.0, 3.0], "class_weight": [None, "balanced"]},
    "linear_svc":          {"C": [0.01, 0.05, 0.1, 0.5], "class_weight": [None, "balanced"]},
    "sgd_classifier":      {"alpha": [1e-5, 1e-4, 1e-3], "loss": ["modified_huber", "log_loss"]},
    "multinomial_nb":      {"alpha": [0.1, 0.5, 1.0, 2.0]},
    "random_forest":       {"n_estimators": [300], "max_depth": [None, 30],
                            "min_samples_leaf": [1, 3]},
    "extra_trees":         {"n_estimators": [300], "max_depth": [None, 30],
                            "min_samples_leaf": [1, 3]},
}


def grid_points(grid):
    keys = list(grid)
    for combo in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, combo))


def build_matrices(train_df, valid_df, feature_mode, variant, char_ngrams):
    cfg = FEATURE_VARIANTS[variant]
    pipe = LIARFeaturePipeline(
        feature_mode=feature_mode,
        categorical_columns=cfg["cats"],
        subject_column=cfg["subject"],
        char_ngrams=char_ngrams,
    )
    pipe.fit(train_df)
    return pipe.transform(train_df), pipe.transform(valid_df)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--task", default="6class", choices=["6class", "binary"])
    parser.add_argument("--output-dir", default="experiments/results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger = setup_logger("TuneBaselines")
    train_df = pd.read_csv(os.path.join(args.data_dir, "train_processed.csv"))
    valid_df = pd.read_csv(os.path.join(args.data_dir, "valid_processed.csv"))

    target = "label_idx" if args.task == "6class" else "is_true"
    y_train, y_valid = train_df[target].values, valid_df[target].values
    is_binary = args.task == "binary"

    trace, best_per_cell = [], {}

    for feature_mode in ["text_only", "metadata_only", "hybrid"]:
        # Metadata variants only matter when metadata is in play; char n-grams
        # only matter when statement text is in play.
        variants = ["base"] if feature_mode == "text_only" else list(FEATURE_VARIANTS)
        char_opts = [False] if feature_mode == "metadata_only" else [False, True]

        for variant, char_ngrams in itertools.product(variants, char_opts):
            X_tr, X_va = build_matrices(train_df, valid_df, feature_mode, variant, char_ngrams)
            logger.info(
                "[%s/%s] variant=%s char_ngrams=%s -> %d features",
                args.task, feature_mode, variant, char_ngrams, X_tr.shape[1],
            )

            for model_name in list_available_models():
                for params in grid_points(PARAM_GRIDS[model_name]):
                    model = get_baseline_model(model_name, config=params, seed=args.seed)
                    model.fit(X_tr, y_train)
                    pred = model.predict(X_va)
                    entry = {
                        "task": args.task,
                        "feature_mode": feature_mode,
                        "variant": variant,
                        "char_ngrams": char_ngrams,
                        "model": model_name,
                        "params": params,
                        "val_acc": round(float(accuracy_score(y_valid, pred)), 4),
                        "val_f1_macro": round(
                            float(f1_score(y_valid, pred, average="macro", zero_division=0)), 4
                        ),
                    }
                    trace.append(entry)
                    key = (feature_mode, model_name)
                    if key not in best_per_cell or entry["val_acc"] > best_per_cell[key]["val_acc"]:
                        best_per_cell[key] = entry

    os.makedirs(args.output_dir, exist_ok=True)
    best_list = sorted(best_per_cell.values(), key=lambda e: (e["feature_mode"], -e["val_acc"]))
    payload = {
        "task": args.task,
        "selection_split": "validation",
        "n_configurations_evaluated": len(trace),
        "best_per_feature_mode_and_model": best_list,
        "trace": trace,
    }
    out_path = os.path.join(args.output_dir, f"tuning_{args.task}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print("\n" + "=" * 92)
    print(f"VALIDATION-SELECTED CONFIGURATIONS ({args.task}) - {len(trace)} configs evaluated")
    print("=" * 92)
    for e in best_list:
        print(f"  {e['feature_mode']:14s} {e['model']:22s} val_acc={e['val_acc']*100:5.2f}  "
              f"variant={e['variant']:16s} char={str(e['char_ngrams']):5s} {e['params']}")
    print("=" * 92)
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
