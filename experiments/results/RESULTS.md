# Baseline Results (leakage-free)

All speaker credit-history features are recomputed from **training labels
only** and encoded out-of-fold; the raw LIAR `*_c` count columns are
excluded. See `src/features/credit_history.py`.

## 6class - Test Accuracy

| Model | Text only | Metadata only | Hybrid |
|---|---|---|---|
| Multinomial Naive Bayes | 0.2393 | 0.2362 | **0.2837** |
| Logistic Regression | 0.2432 | 0.2751 | 0.2611 |
| Extra Trees | 0.2175 | 0.2416 | 0.2588 |
| Random Forest | 0.2065 | 0.2440 | 0.2502 |
| Linear SVM | 0.2369 | 0.2712 | 0.2424 |
| SGD Classifier | 0.2354 | 0.2229 | 0.2330 |
| _Majority class_ | 0.2081 | -- | -- |

## 6class - Test Macro-F1

| Model | Text only | Metadata only | Hybrid |
|---|---|---|---|
| Multinomial Naive Bayes | 0.1984 | 0.2351 | **0.2773** |
| Logistic Regression | 0.2151 | 0.2704 | 0.2557 |
| Linear SVM | 0.2260 | 0.2663 | 0.2425 |
| SGD Classifier | 0.2235 | 0.2204 | 0.2320 |
| Extra Trees | 0.1335 | 0.2329 | 0.2186 |
| Random Forest | 0.1327 | 0.2349 | 0.2106 |
| _Majority class_ | 0.2081 | -- | -- |

## binary - Test Accuracy

| Model | Text only | Metadata only | Hybrid |
|---|---|---|---|
| Multinomial Naive Bayes | 0.6087 | 0.6345 | **0.6610** |
| Logistic Regression | 0.6142 | 0.6485 | 0.6477 |
| Random Forest | 0.5939 | 0.6235 | 0.6306 |
| Linear SVM | 0.6041 | 0.6454 | 0.6290 |
| Extra Trees | 0.5931 | 0.6251 | 0.6243 |
| SGD Classifier | 0.6017 | 0.5713 | 0.6228 |
| _Majority class_ | 0.5666 | -- | -- |

## binary - Test Macro-F1

| Model | Text only | Metadata only | Hybrid |
|---|---|---|---|
| Multinomial Naive Bayes | 0.5797 | 0.6198 | **0.6487** |
| Logistic Regression | 0.5906 | 0.6264 | 0.6307 |
| Linear SVM | 0.5910 | 0.6234 | 0.6191 |
| SGD Classifier | 0.5892 | 0.4956 | 0.6180 |
| Random Forest | 0.4783 | 0.5869 | 0.5719 |
| Extra Trees | 0.4532 | 0.5801 | 0.5452 |
| _Majority class_ | 0.5666 | -- | -- |

## Tuned configurations (selected on validation, scored once on test)

Feature variant, char n-grams and hyperparameters were searched over 476
configurations per task, ranked by **validation** accuracy; the test split
was scored exactly once afterwards. Validation accuracy is reported only to
document the protocol -- it is optimistically biased by the size of the
search and should not be quoted as a result.

### 6class

| Feature mode | Best model | Variant | Val acc | Test acc | Test Macro-F1 |
|---|---|---|---|---|---|
| Text only | extra_trees | base | 0.2741 | 0.2705 | 0.2209 |
| Metadata only | linear_svc | speaker_subject | 0.2796 | 0.2744 | 0.2599 |
| Hybrid | sgd_classifier | speaker | 0.2975 | 0.2845 | 0.2763 |

### binary

| Feature mode | Best model | Variant | Val acc | Test acc | Test Macro-F1 |
|---|---|---|---|---|---|
| Text only | multinomial_nb | base | 0.6316 | 0.6267 | 0.5988 |
| Metadata only | linear_svc | all_meta | 0.6363 | 0.6329 | 0.6266 |
| Hybrid | multinomial_nb | speaker | 0.6433 | 0.6726 | 0.6619 |
