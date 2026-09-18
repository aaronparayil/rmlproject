# Fine-Grained Fake News Detection on LIAR

A leakage-free benchmark of classical models for fine-grained credibility
classification on the **LIAR** dataset (Wang, ACL 2017), combining statement
text with speaker metadata and recomputed credit history.

## Headline finding: LIAR's credit-history columns leak the target

The credit-history columns shipped with LIAR (`barely_true_c` ... `pants_fire_c`)
cannot be used as features as distributed. Measured over the full corpus
(`python scripts/leakage_analysis.py`):

| Property | Measurement |
|---|---|
| Count vector is constant per speaker | 3314 / 3318 speakers |
| Counts equal the speaker's label tally over **train+valid+test** | 2846 / 2947 speakers |
| Single-statement speakers whose counts one-hot encode their own label | 1689 / 1701 |
| `true`-labelled single-statement speakers with all-zero counts | 355 / 355 |

So each statement is counted in its own features, and held-out labels are
visible at training time through the shared per-speaker aggregate. (There is no
`true_c` bucket, which is why `true` statements are never counted anywhere.)

The effect is large. A Random Forest given **only** the history columns:

| Variant | Test acc. (6-class) | Macro-F1 |
|---|---|---|
| Raw LIAR counts (leaky) | 0.3944 | 0.4056 |
| Naive "subtract the row's own label" | **0.6594** | 0.6633 |
| Leak-free out-of-fold history (this repo) | 0.2284 | 0.2236 |
| Majority class | 0.2081 | -- |

Note the middle row: the obvious repair makes things *worse*. Because the
per-speaker vector `B` is constant and therefore memorisable, `B - e_y`
identifies the label `y` almost deterministically.

### The fix

`src/features/credit_history.py` recomputes speaker history from scratch:

* only **training** labels ever contribute, so held-out labels are never seen;
* training rows are encoded **out-of-fold**, so no statement contributes to its
  own features;
* counts become prior-smoothed proportions, so single-statement speakers do not
  produce saturated, memorisable vectors.

The raw `*_c` columns are retained in the processed CSVs for this analysis but
are excluded from every model feature set (enforced by a regression test).

## Results

Full tables in [`experiments/results/RESULTS.md`](experiments/results/RESULTS.md).

**Default configuration** (party + state + subject tokens, default
hyperparameters) -- best test accuracy per cell:

| Task | Text only | Metadata only | Hybrid |
|---|---|---|---|
| 6-class | 0.2432 | 0.2751 | **0.2837** |
| Binary | 0.6142 | 0.6485 | **0.6610** |

**Tuned** (feature variant + hyperparameters selected on validation over 476
configurations, test scored once):

| Task | Text only | Metadata only | Hybrid |
|---|---|---|---|
| 6-class | 0.2705 | 0.2744 | **0.2845** |
| Binary | 0.6267 | 0.6329 | **0.6726** |

The 6-class hybrid result is in line with the published LIAR literature (Wang
2017 reports 0.274 for a hybrid CNN), and the hybrid setting beats both
single-modality ablations on both tasks and under both configurations.

**On the size of the tuning gain.** Tuning adds ~0.1 points (6-class) and ~1.2
points (binary). With 1283 test rows the standard error is ~1.3 points, so
these gains are suggestive rather than established. Validation ranking also
transferred poorly: on 6-class the configuration with the best validation
accuracy (0.3069) placed third from last on test, and the eventual test winner
ranked fifth on validation. Validation figures in the tuned table are recorded
for protocol transparency only -- they are inflated by the size of the search
and should not be quoted as results.

**Binary task definition.** Labels are collapsed using the standard convention:
`{half-true, mostly-true, true}` = true-leaning, `{pants-fire, false,
barely-true}` = false-leaning (majority class 0.5666).

## Project Structure

```
.
├── configs/                 # YAML configuration for data and baselines
├── data/
│   ├── raw/                 # LIAR TSV splits (train, valid, test)
│   └── processed/           # Cleaned, leakage-free engineered splits
├── Documentation/           # Research paper PDF and preliminary EDA notebooks
├── experiments/results/     # Metrics, consolidated tables, leakage analysis
├── scripts/
│   ├── prepare_data.py      # Build processed splits
│   ├── run_eda.py           # Exploratory plots
│   ├── train_baselines.py   # Train + evaluate one (task, feature-mode) cell
│   ├── summarize_results.py # Consolidate runs into paper-ready tables
│   └── leakage_analysis.py  # Reproduce the leakage evidence above
├── src/
│   ├── data/                # Loaders, preprocessing, dataset abstractions
│   ├── features/            # Credit history, text stats, feature union
│   ├── models/              # Classical model wrappers
│   └── utils/               # Logging, config, metrics
└── tests/
```

## Quick Start

```bash
pip install -r requirements.txt

python scripts/prepare_data.py          # build processed splits
python scripts/leakage_analysis.py      # reproduce the leakage evidence

# full ablation grid
for task in 6class binary; do
  for mode in hybrid text_only metadata_only; do
    python scripts/train_baselines.py --task $task --feature-mode $mode
  done
done

python scripts/summarize_results.py     # paper-ready tables
pytest                                  # test suite
```

## Scope

Implemented: the data pipeline, leakage-free feature engineering, and six
classical baselines (Logistic Regression, Linear SVM, SGD, Multinomial NB,
Random Forest, Extra Trees) across both tasks and three feature modes.

**Not implemented:** the transformer encoder and speaker/subject graph model
discussed in the accompanying paper. These would require `torch` /
`transformers` / `torch-geometric`, none of which are current dependencies. Any
such extension must consume credit history through
`LeakFreeCreditHistory`; training on the raw `*_c` columns reproduces the
inflated numbers documented above.
