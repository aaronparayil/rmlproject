"""Consolidate baseline ablation runs into publication-ready tables.

Reads every ``baseline_<task>_<mode>.json`` in the results directory and emits
a tidy long-form CSV plus per-task Markdown tables suitable for a paper.
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd

MODE_LABEL = {
    "text_only": "Text only",
    "metadata_only": "Metadata only",
    "hybrid": "Hybrid",
}
MODE_ORDER = ["text_only", "metadata_only", "hybrid"]

# Reference points for contextualising the corrected numbers.
MAJORITY_BASELINE = {"6class": 0.2081, "binary": 0.5666}


def load_results(results_dir: str) -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "baseline_*.json"))):
        with open(path, "r", encoding="utf-8") as fh:
            rows.extend(json.load(fh))
    if not rows:
        raise FileNotFoundError(f"No baseline_*.json files found in {results_dir}")
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, task: str, metric: str) -> str:
    sub = df[df.task == task]
    pivot = sub.pivot_table(index="model", columns="feature_mode", values=metric)
    pivot = pivot.reindex(columns=[m for m in MODE_ORDER if m in pivot.columns])
    pivot = pivot.rename(columns=MODE_LABEL)
    pivot = pivot.sort_values(pivot.columns[-1], ascending=False)

    header = f"| Model | {' | '.join(pivot.columns)} |"
    sep = "|" + "---|" * (len(pivot.columns) + 1)
    lines = [header, sep]
    best = pivot.max().max()
    for model, row in pivot.iterrows():
        cells = []
        for v in row:
            cells.append(f"**{v:.4f}**" if v == best else f"{v:.4f}")
        lines.append(f"| {model} | {' | '.join(cells)} |")
    if task in MAJORITY_BASELINE:
        pad = " | ".join(["--"] * (len(pivot.columns) - 1))
        lines.append(
            f"| _Majority class_ | {MAJORITY_BASELINE[task]:.4f}"
            + (f" | {pad}" if pad else "")
            + " |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="experiments/results")
    parser.add_argument("--output-dir", default="experiments/results")
    args = parser.parse_args()

    df = load_results(args.results_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    tidy_path = os.path.join(args.output_dir, "all_baselines.csv")
    df.to_csv(tidy_path, index=False)

    sections = [
        "# Baseline Results (leakage-free)",
        "",
        "All speaker credit-history features are recomputed from **training labels",
        "only** and encoded out-of-fold; the raw LIAR `*_c` count columns are",
        "excluded. See `src/features/credit_history.py`.",
        "",
    ]
    for task in ["6class", "binary"]:
        if task not in set(df.task):
            continue
        for metric, name in [("test_acc", "Test Accuracy"), ("test_f1_macro", "Test Macro-F1")]:
            sections += [f"## {task} - {name}", "", markdown_table(df, task, metric), ""]

    # Validation-selected configurations, scored once on test.
    tuned_frames = []
    for task in ["6class", "binary"]:
        path = os.path.join(args.results_dir, f"tuned_{task}_test.csv")
        if os.path.exists(path):
            tuned_frames.append(pd.read_csv(path))

    if tuned_frames:
        tuned = pd.concat(tuned_frames, ignore_index=True)
        sections += [
            "## Tuned configurations (selected on validation, scored once on test)",
            "",
            "Feature variant, char n-grams and hyperparameters were searched over 476",
            "configurations per task, ranked by **validation** accuracy; the test split",
            "was scored exactly once afterwards. Validation accuracy is reported only to",
            "document the protocol -- it is optimistically biased by the size of the",
            "search and should not be quoted as a result.",
            "",
        ]
        for task in ["6class", "binary"]:
            sub = tuned[tuned.task == task]
            if sub.empty:
                continue
            best = sub.loc[sub.groupby("feature_mode")["test_acc"].idxmax()]
            best = best.set_index("feature_mode").reindex(
                [m for m in MODE_ORDER if m in set(best.feature_mode if "feature_mode" in best else best.index)]
            )
            sections += [f"### {task}", "", "| Feature mode | Best model | Variant | Val acc | Test acc | Test Macro-F1 |",
                         "|---|---|---|---|---|---|"]
            for mode, row in best.iterrows():
                sections.append(
                    f"| {MODE_LABEL.get(mode, mode)} | {row['model']} | {row['variant']} | "
                    f"{row['val_acc']:.4f} | {row['test_acc']:.4f} | {row['test_f1_macro']:.4f} |"
                )
            sections.append("")

    md_path = os.path.join(args.output_dir, "RESULTS.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(sections))

    print("\n".join(sections))
    print(f"\nWrote {tidy_path}\nWrote {md_path}")


if __name__ == "__main__":
    main()
