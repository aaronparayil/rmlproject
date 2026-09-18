"""Quantify the target leakage in LIAR's shipped credit-history columns.

Produces the evidence behind the leakage-free redesign in
``src/features/credit_history.py``:

1. **Structure.** Are the ``*_c`` columns constant per speaker, and do they equal
   that speaker's label histogram over the whole corpus (train+valid+test)?
2. **Self-inclusion.** Do single-statement speakers carry a one-hot encoding of
   their own label?
3. **Effect size.** How much test accuracy does the leak manufacture, and what
   happens under the correction Wang (2017) prescribes, i.e. subtracting the
   current statement's own label from its speaker's history?

Run:  python scripts/leakage_analysis.py
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from src.data.loader import LIARDataLoader
from src.features.credit_history import HISTORY_FEATURE_COLS, LeakFreeCreditHistory
from src.utils.logger import setup_logger

COUNT_COLS = ["barely_true_c", "false_c", "half_true_c", "mostly_true_c", "pants_fire_c"]
LABEL_TO_COUNT = {
    "barely-true": "barely_true_c",
    "false": "false_c",
    "half-true": "half_true_c",
    "mostly-true": "mostly_true_c",
    "pants-fire": "pants_fire_c",
}
LABEL_TO_INT = {
    "pants-fire": 0, "false": 1, "barely-true": 2,
    "half-true": 3, "mostly-true": 4, "true": 5,
}


def _rf(seed: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=100, max_depth=25, n_jobs=-1, random_state=seed
    )


def _score(model, X_tr, y_tr, X_te, y_te) -> dict:
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    return {
        "test_acc": round(float(accuracy_score(y_te, pred)), 4),
        "test_f1_macro": round(float(f1_score(y_te, pred, average="macro")), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="experiments/results/leakage_analysis.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger = setup_logger("LeakageAnalysis")
    splits = LIARDataLoader(data_dir="data/raw").load_all_splits()
    train, valid, test = splits["train"], splits["valid"], splits["test"]
    full = pd.concat([train, valid, test], ignore_index=True)

    for df in (train, valid, test, full):
        for c in COUNT_COLS:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        df["y"] = df["label"].map(LABEL_TO_INT)

    report = {}

    # -- 1. structure ------------------------------------------------------
    varies = full.groupby("speaker")[COUNT_COLS].nunique().max(axis=1)
    hist = (
        full[full.label != "true"]
        .groupby(["speaker", "label"]).size().unstack(fill_value=0)
        .rename(columns=LABEL_TO_COUNT)
        .reindex(columns=COUNT_COLS, fill_value=0)
    )
    stated = full.groupby("speaker")[COUNT_COLS].first().reindex(hist.index)
    matches = (hist == stated).all(axis=1)
    report["structure"] = {
        "speakers": int(full.speaker.nunique()),
        "speakers_with_constant_counts": int((varies <= 1).sum()),
        "speakers_matching_full_corpus_histogram": int(matches.sum()),
        "speakers_compared": int(len(matches)),
    }

    # -- 2. self-inclusion -------------------------------------------------
    counts = full.speaker.value_counts()
    solo = full[full.speaker.isin(counts[counts == 1].index)]
    solo_non_true = solo[solo.label != "true"]
    own = np.array([r[LABEL_TO_COUNT[r.label]] for _, r in solo_non_true.iterrows()])
    totals = solo_non_true[COUNT_COLS].sum(axis=1).to_numpy()
    solo_true = solo[solo.label == "true"]
    report["self_inclusion"] = {
        "single_statement_speakers_non_true": int(len(solo_non_true)),
        "encoding_own_label_as_one_hot": int(((own == 1) & (totals == 1)).sum()),
        "single_statement_speakers_true": int(len(solo_true)),
        "true_speakers_with_zero_counts": int((solo_true[COUNT_COLS].sum(axis=1) == 0).sum()),
    }

    # -- 3. effect size ----------------------------------------------------
    def prescribed_repair(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for label, col in LABEL_TO_COUNT.items():
            mask = out.label == label
            out.loc[mask, col] = (out.loc[mask, col] - 1).clip(lower=0)
        return out

    variants = {}
    variants["leaky_raw_counts"] = _score(
        _rf(args.seed), train[COUNT_COLS], train.y, test[COUNT_COLS], test.y
    )
    tr_n, te_n = prescribed_repair(train), prescribed_repair(test)
    variants["prescribed_subtract_own_label"] = _score(
        _rf(args.seed), tr_n[COUNT_COLS], tr_n.y, te_n[COUNT_COLS], te_n.y
    )
    ch = LeakFreeCreditHistory(seed=args.seed).fit(train)
    variants["leak_free_oof_history"] = _score(
        _rf(args.seed),
        ch.transform_train(train)[HISTORY_FEATURE_COLS], train.y,
        ch.transform(test)[HISTORY_FEATURE_COLS], test.y,
    )
    variants["majority_class"] = {
        "test_acc": round(float(test.y.value_counts(normalize=True).max()), 4),
        "test_f1_macro": None,
    }

    # -- 4. why the prescribed repair fails -------------------------------------
    # The repaired vector b' = B - e_y sits one unit below its speaker's own
    # (constant, memorisable) vector B. Any label c whose re-addition lands on
    # an observed vector is a candidate; when exactly one does, the label is
    # recoverable outright.
    observed = np.array(sorted(set(map(tuple, train[COUNT_COLS].to_numpy(dtype=int)))))
    col_index = {c: COUNT_COLS.index(c) for c in COUNT_COLS}

    def candidate_count(row) -> int:
        b = row[COUNT_COLS].to_numpy(dtype=int).copy()
        if row.label in LABEL_TO_COUNT:
            k = col_index[LABEL_TO_COUNT[row.label]]
            b[k] = max(0, b[k] - 1)
        delta = observed - b
        total = delta.sum(axis=1)
        n = sum(bool(((delta[:, col_index[c]] == 1) & (total == 1)).any()) for c in COUNT_COLS)
        return n + int((delta == 0).all(axis=1).any())  # the extra one is 'true'

    sizes = test.apply(candidate_count, axis=1)
    report["prescribed_repair_decodability"] = {
        "distinct_count_vectors": int(len(observed)),
        "speakers": int(full.speaker.nunique()),
        "test_rows": int(len(test)),
        "label_recovered_exactly": int((sizes == 1).sum()),
        "share_recovered_exactly": round(float((sizes == 1).mean()), 4),
        "mean_candidate_set_size": round(float(sizes.mean()), 2),
    }
    report["history_only_rf_6class"] = variants

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    s, si = report["structure"], report["self_inclusion"]
    print("\n" + "=" * 72)
    print("LIAR CREDIT-HISTORY LEAKAGE ANALYSIS")
    print("=" * 72)
    print(f"Counts constant per speaker : {s['speakers_with_constant_counts']}/{s['speakers']}")
    print(f"Counts == full-corpus tally : {s['speakers_matching_full_corpus_histogram']}/{s['speakers_compared']}"
          "   (spans train+valid+test)")
    print(f"Single-statement speakers one-hot encoding their own label : "
          f"{si['encoding_own_label_as_one_hot']}/{si['single_statement_speakers_non_true']}")
    print(f"'true' single-statement speakers with all-zero counts      : "
          f"{si['true_speakers_with_zero_counts']}/{si['single_statement_speakers_true']}")
    dec = report["prescribed_repair_decodability"]
    print(f"Distinct count vectors across {dec['speakers']} speakers            : "
          f"{dec['distinct_count_vectors']}")
    print(f"Test rows whose label the prescribed repair recovers exactly: "
          f"{dec['label_recovered_exactly']}/{dec['test_rows']} "
          f"({dec['share_recovered_exactly']*100:.1f}%), mean candidate set "
          f"{dec['mean_candidate_set_size']}/6")
    print("-" * 72)
    print("History-only Random Forest, 6-class test accuracy:")
    for name, m in variants.items():
        f1 = "   --" if m["test_f1_macro"] is None else f"{m['test_f1_macro']:.4f}"
        print(f"  {name:28s} acc={m['test_acc']:.4f}  macro-F1={f1}")
    print("=" * 72)
    logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
