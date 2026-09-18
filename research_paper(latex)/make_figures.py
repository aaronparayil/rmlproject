"""Regenerate every figure used in the paper from the current repository state.

Run from the project root:  python "research_paper(latex)/make_figures.py"

All figures are written as vector PDFs into research_paper(latex)/figures/ and
are styled for greyscale print: hatching and value labels carry the information
that colour would otherwise carry.
"""

import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.metrics import confusion_matrix

from src.features.feature_union import LIARFeaturePipeline
from src.models.classical import get_baseline_model
from src.utils.config import load_config

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
ROOT = os.path.dirname(HERE)
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

COL = 3.35          # one-column width (inches)
WIDE = 6.9          # two-column width (inches)
LABELS = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]
RAW_COUNTS = ["barely_true_c", "false_c", "half_true_c", "mostly_true_c", "pants_fire_c"]
GREY = ["0.25", "0.45", "0.62", "0.78"]
HATCH = ["", "///", "...", "xxx"]


def save(fig, name):
    path = os.path.join(FIG, name)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


def load_raw():
    cols = ["id", "label", "statement", "subject", "speaker", "job_title", "state",
            "party", "barely_true_c", "false_c", "half_true_c", "mostly_true_c",
            "pants_fire_c", "context"]
    frames = []
    for split in ["train", "valid", "test"]:
        df = pd.read_csv(os.path.join(ROOT, "data", "raw", split + ".tsv"), sep="\t",
                         header=None, names=cols, quoting=3)
        df["split"] = split
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_processed():
    return tuple(
        pd.read_csv(os.path.join(ROOT, "data", "processed", s + "_processed.csv"))
        for s in ["train", "valid", "test"]
    )


# ----------------------------------------------------------------------
def fig_pipeline():
    """Block diagram of the end-to-end pipeline."""
    stages = [
        ("Raw LIAR splits", "train.tsv / valid.tsv / test.tsv (12,836 rows)"),
        ("Split-local cleaning", "whitespace and empty-string normalisation, dtype checks"),
        ("Deduplication", "on id only; repeated statement text retained"),
        ("Null handling", "drop core-null rows; impute metadata with 'Unknown'"),
        ("Text statistics", "char length, word count, mean word length"),
        ("Leak-free credit history", "train labels only, out-of-fold, prior-smoothed"),
        ("Split-safe scaling", "scalers fitted on train, applied to valid and test"),
        ("Feature union", "word TF-IDF + char TF-IDF + numeric + one-hot + subject"),
        ("Classifier bank", "LR, SVM, SGD, MNB, RF, ET; six-class and binary"),
        ("Protocol", "select on validation, score test once"),
    ]
    fig, ax = plt.subplots(figsize=(COL, 6.1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(stages))
    ax.axis("off")
    for i, (title, sub) in enumerate(stages):
        y = len(stages) - i - 1
        shaded = i in (5, 7, 9)
        box = FancyBboxPatch((0.03, y + 0.18), 0.94, 0.62,
                             boxstyle="round,pad=0.012,rounding_size=0.02",
                             linewidth=1.1 if shaded else 0.7,
                             edgecolor="black",
                             facecolor="0.88" if shaded else "white")
        ax.add_patch(box)
        ax.text(0.5, y + 0.60, title, ha="center", va="center", fontsize=7.8,
                fontweight="bold")
        ax.text(0.5, y + 0.36, sub, ha="center", va="center", fontsize=6.0)
        if i < len(stages) - 1:
            ax.add_patch(FancyArrowPatch((0.5, y + 0.18), (0.5, y - 0.02),
                                         arrowstyle="-|>", mutation_scale=7,
                                         linewidth=0.7, color="black"))
    save(fig, "fig_pipeline.pdf")


def fig_missing(raw):
    pct = (raw.isna().mean() * 100).sort_values(ascending=False)
    pct = pct[pct > 0]
    fig, ax = plt.subplots(figsize=(COL, 1.85))
    bars = ax.barh(range(len(pct)), pct.values, color="0.55", edgecolor="black",
                   linewidth=0.6, height=0.62)
    ax.set_yticks(range(len(pct)))
    ax.set_yticklabels(list(pct.index))
    ax.invert_yaxis()
    ax.set_xlabel("missing values (% of 12,836 rows)")
    ax.set_xlim(0, max(pct.values) * 1.3)
    for b, v in zip(bars, pct.values):
        ax.text(v + 0.6, b.get_y() + b.get_height() / 2, "%.2f%%" % v,
                va="center", fontsize=7)
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    save(fig, "fig_missing.pdf")


def fig_labels(tr, va, te):
    counts = {name: d.label.value_counts().reindex(LABELS).values
              for name, d in [("train", tr), ("valid", va), ("test", te)]}
    shares = {k: v / v.sum() * 100 for k, v in counts.items()}
    x = np.arange(len(LABELS))
    w = 0.27
    fig, ax = plt.subplots(figsize=(COL, 2.05))
    for i, (name, vals) in enumerate(shares.items()):
        ax.bar(x + (i - 1) * w, vals, w, label=name, color=GREY[i],
               edgecolor="black", linewidth=0.5, hatch=HATCH[i])
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, rotation=30, ha="right")
    ax.set_ylabel("share of split (%)")
    ax.legend(frameon=False, ncol=3, loc="upper left", handlelength=1.4)
    ax.set_ylim(0, 27)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    save(fig, "fig_labels.pdf")


def fig_corr(raw):
    df = raw.copy()
    for c in RAW_COUNTS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["total_history_count"] = df[RAW_COUNTS].sum(axis=1)
    s = df.statement.astype(str)
    df["statement_char_len"] = s.str.len()
    df["statement_word_count"] = s.str.split().apply(len)
    df["avg_word_length"] = df.statement_char_len / df.statement_word_count.replace(0, 1)
    df["is_true"] = df.label.isin(["half-true", "mostly-true", "true"]).astype(int)
    cols = RAW_COUNTS + ["total_history_count", "statement_char_len",
                         "statement_word_count", "avg_word_length", "is_true"]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(COL, 2.95))
    im = ax.imshow(corr.values, cmap="Greys", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90, fontsize=6)
    ax.set_yticklabels(cols, fontsize=6)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.values[i, j]
            ax.text(j, i, "%.2f" % v, ha="center", va="center", fontsize=4.6,
                    color="white" if abs(v) > 0.55 else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_linewidth(0.5)
    save(fig, "fig_corr.pdf")


def fig_leakage():
    with open(os.path.join(ROOT, "experiments", "results", "leakage_analysis.json")) as fh:
        rep = json.load(fh)["history_only_rf_6class"]
    names = ["raw LIAR\ncounts", "naive\nrepair", "leak-free\nOOF history", "majority\nclass"]
    keys = ["leaky_raw_counts", "prescribed_subtract_own_label", "leak_free_oof_history",
            "majority_class"]
    vals = [rep[k]["test_acc"] for k in keys]
    fig, ax = plt.subplots(figsize=(COL, 1.95))
    bars = ax.bar(range(4), vals, 0.6,
                  color=["0.35", "0.2", "0.72", "white"],
                  edgecolor="black", linewidth=0.7,
                  hatch=["///", "xxx", "", ".."])
    ax.axhline(rep["majority_class"]["test_acc"], linestyle="--", linewidth=0.7,
               color="black")
    ax.set_xticks(range(4))
    ax.set_xticklabels(names, fontsize=6.8)
    ax.set_ylabel("six-class test accuracy")
    ax.set_ylim(0, 0.78)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, "%.4f" % v,
                ha="center", fontsize=7)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    save(fig, "fig_leakage.pdf")


def fig_modality():
    df = pd.read_csv(os.path.join(ROOT, "experiments", "results", "all_baselines.csv"))
    modes = ["text_only", "metadata_only", "hybrid"]
    nice = ["text only", "metadata only", "hybrid"]
    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 2.25))
    for ax, task, major, ylim, title in zip(
            axes, ["6class", "binary"], [0.2081, 0.5666],
            [(0.15, 0.325), (0.50, 0.725)],
            ["(a) six-class task", "(b) binary task"]):
        sub = df[df.task == task]
        models = sorted(sub.model.unique())
        x = np.arange(len(models))
        w = 0.26
        for i, (m, nm) in enumerate(zip(modes, nice)):
            vals = [sub[(sub.model == mo) & (sub.feature_mode == m)].test_acc.values[0]
                    for mo in models]
            ax.bar(x + (i - 1) * w, vals, w, label=nm, color=GREY[i],
                   edgecolor="black", linewidth=0.5, hatch=HATCH[i])
        ax.axhline(major, linestyle="--", linewidth=0.8, color="black",
                   label="majority class" if task == "binary" else None)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace(" ", "\n") for m in models], fontsize=6.2)
        ax.set_ylim(*ylim)
        ax.set_ylabel("test accuracy")
        ax.set_title(title)
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, ncol=3, loc="upper left", handlelength=1.4,
                   fontsize=6.8)
    axes[1].legend(frameon=False, loc="upper left", handlelength=1.8, fontsize=6.8)
    save(fig, "fig_modality.pdf")


def fig_confusion_and_history(tr, va, te):
    cfg = load_config(os.path.join(ROOT, "configs", "baseline_config.yaml"))["baseline_experiment"]
    pipe = LIARFeaturePipeline(
        text_column=cfg.get("text_column", "statement"),
        numeric_columns=cfg.get("numeric_features"),
        categorical_columns=cfg.get("categorical_features"),
        tfidf_params=cfg.get("tfidf"),
        feature_mode="hybrid",
    )
    pipe.fit(tr)
    X_tr, X_te = pipe.transform(tr), pipe.transform(te)
    model = get_baseline_model("multinomial_nb", cfg["models"]["multinomial_nb"], 42)
    model.fit(X_tr, tr.label_idx.values)
    cm = confusion_matrix(te.label_idx.values, model.predict(X_te))
    cmn = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(COL, 2.85))
    im = ax.imshow(cmn, cmap="Greys", vmin=0, vmax=0.55)
    ax.set_xticks(range(6))
    ax.set_yticks(range(6))
    ax.set_xticklabels(LABELS, rotation=45, ha="right", fontsize=6.5)
    ax.set_yticklabels(LABELS, fontsize=6.5)
    ax.set_xlabel("predicted label")
    ax.set_ylabel("true label")
    for i in range(6):
        for j in range(6):
            ax.text(j, i, "%.2f" % cmn[i, j], ha="center", va="center", fontsize=5.8,
                    color="white" if cmn[i, j] > 0.33 else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_linewidth(0.5)
    save(fig, "fig_confusion.pdf")

    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 2.05))
    means = te.groupby("label").credibility_score.mean().reindex(LABELS)
    sems = te.groupby("label").credibility_score.sem().reindex(LABELS)
    axes[0].bar(range(6), means.values, 0.6, yerr=sems.values, capsize=2,
                color="0.6", edgecolor="black", linewidth=0.6,
                error_kw={"linewidth": 0.7})
    axes[0].set_xticks(range(6))
    axes[0].set_xticklabels(LABELS, rotation=30, ha="right")
    axes[0].set_ylabel("credibility score")
    axes[0].axhline(0, linewidth=0.6, color="black")
    axes[0].set_title("(a) leak-free credibility score by true label (test)")
    axes[0].grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.7)
    axes[0].set_axisbelow(True)

    sizes = tr.speaker_history_size.values
    axes[1].hist(np.log10(sizes + 1), bins=30, color="0.6", edgecolor="black",
                 linewidth=0.5)
    axes[1].set_xlabel("log10(1 + out-of-fold history size)")
    axes[1].set_ylabel("training statements")
    unseen = [d.speaker_is_unseen.mean() for d in (tr, va, te)]
    axes[1].set_title("(b) speaker history support (train)")
    axes[1].text(0.97, 0.93,
                 "no usable history:\ntrain %.1f%%, valid %.1f%%, test %.1f%%"
                 % tuple(100 * u for u in unseen),
                 transform=axes[1].transAxes, ha="right", va="top", fontsize=6.5)
    axes[1].grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.7)
    axes[1].set_axisbelow(True)
    save(fig, "fig_history.pdf")


def main():
    raw = load_raw()
    tr, va, te = load_processed()
    fig_pipeline()
    fig_missing(raw)
    fig_labels(tr, va, te)
    fig_corr(raw)
    fig_leakage()
    fig_modality()
    fig_confusion_and_history(tr, va, te)


if __name__ == "__main__":
    main()
