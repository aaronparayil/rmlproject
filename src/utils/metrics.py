from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    target_names: Optional[List[str]] = None,
    is_binary: bool = False
) -> Dict[str, Any]:
    acc = accuracy_score(y_true, y_pred)

    # Macro averaging is always computed over all classes, for both the 6-class
    # and binary tasks, so a column labelled "macro" means the same thing
    # everywhere. Positive-class scores are reported separately under the
    # "_positive" suffix rather than overwriting the macro fields.
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    metrics = {
        "accuracy": float(acc),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(p_weighted),
        "recall_weighted": float(r_weighted),
        "f1_weighted": float(f1_weighted)
    }

    if is_binary:
        p_pos, r_pos, f1_pos, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        metrics["precision_positive"] = float(p_pos)
        metrics["recall_positive"] = float(r_pos)
        metrics["f1_positive"] = float(f1_pos)

    if y_prob is not None:
        try:
            if is_binary:
                prob = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
                metrics["auc_roc"] = float(roc_auc_score(y_true, prob))
            else:
                metrics["auc_roc_ovr"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr"))
        except Exception:
            metrics["auc_roc"] = None

    return metrics

def classification_report_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    names = target_names if target_names is not None else [str(i) for i in range(len(precision))]
    report = {}
    for i, name in enumerate(names):
        if i < len(precision):
            report[name] = {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1-score": float(f1[i]),
                "support": int(support[i])
            }
    return report
