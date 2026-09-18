from typing import Any, Dict, Optional, List
import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from scipy.sparse import issparse
from .base import BaseModel

class ClassicalModelWrapper(BaseModel):
    def __init__(self, model_name: str, estimator: Any, requires_dense: bool = False, requires_non_negative: bool = False):
        self.model_name = model_name
        self.estimator = estimator
        self.requires_dense = requires_dense
        self.requires_non_negative = requires_non_negative

    def _prepare_X(self, X: Any) -> Any:
        if self.requires_dense and issparse(X):
            return X.toarray()
        if self.requires_non_negative and issparse(X):
            X = X.copy()
            if hasattr(X, "data"):
                X.data = np.clip(X.data, 0, None)
            return X
        return X

    def fit(self, X: Any, y: np.ndarray) -> "ClassicalModelWrapper":
        X_prep = self._prepare_X(X)
        self.estimator.fit(X_prep, y)
        return self

    def predict(self, X: Any) -> np.ndarray:
        X_prep = self._prepare_X(X)
        return self.estimator.predict(X_prep)

    def predict_proba(self, X: Any) -> Optional[np.ndarray]:
        X_prep = self._prepare_X(X)
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(X_prep)
        elif hasattr(self.estimator, "decision_function"):
            dfunc = self.estimator.decision_function(X_prep)
            if dfunc.ndim == 1:
                prob = 1.0 / (1.0 + np.exp(-dfunc))
                return np.column_stack([1.0 - prob, prob])
            else:
                exp_vals = np.exp(dfunc - np.max(dfunc, axis=1, keepdims=True))
                return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
        return None

def get_baseline_model(name: str, config: Optional[Dict[str, Any]] = None, seed: int = 42) -> ClassicalModelWrapper:
    cfg = config or {}

    if name == "logistic_regression":
        estimator = LogisticRegression(
            max_iter=cfg.get("max_iter", 1000),
            C=cfg.get("C", 1.0),
            solver=cfg.get("solver", "lbfgs"),
            class_weight=cfg.get("class_weight"),
            random_state=seed
        )
        return ClassicalModelWrapper("Logistic Regression", estimator)

    elif name == "linear_svc":
        estimator = LinearSVC(
            max_iter=cfg.get("max_iter", 2000),
            C=cfg.get("C", 1.0),
            dual="auto",
            class_weight=cfg.get("class_weight"),
            random_state=seed
        )
        return ClassicalModelWrapper("Linear SVM", estimator)

    elif name == "sgd_classifier":
        estimator = SGDClassifier(
            loss=cfg.get("loss", "modified_huber"),
            max_iter=cfg.get("max_iter", 1000),
            alpha=cfg.get("alpha", 0.0001),
            class_weight=cfg.get("class_weight"),
            random_state=seed
        )
        return ClassicalModelWrapper("SGD Classifier", estimator)

    elif name == "multinomial_nb":
        estimator = MultinomialNB(alpha=cfg.get("alpha", 1.0))
        return ClassicalModelWrapper("Multinomial Naive Bayes", estimator, requires_non_negative=True)

    elif name == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=cfg.get("n_estimators", 100),
            max_depth=cfg.get("max_depth", 25),
            min_samples_leaf=cfg.get("min_samples_leaf", 1),
            class_weight=cfg.get("class_weight"),
            n_jobs=cfg.get("n_jobs", -1),
            random_state=seed
        )
        return ClassicalModelWrapper("Random Forest", estimator)

    elif name == "extra_trees":
        estimator = ExtraTreesClassifier(
            n_estimators=cfg.get("n_estimators", 100),
            max_depth=cfg.get("max_depth", 25),
            min_samples_leaf=cfg.get("min_samples_leaf", 1),
            class_weight=cfg.get("class_weight"),
            n_jobs=cfg.get("n_jobs", -1),
            random_state=seed
        )
        return ClassicalModelWrapper("Extra Trees", estimator)

    else:
        raise ValueError(f"Unknown baseline model name: {name}")

def list_available_models() -> List[str]:
    return [
        "logistic_regression",
        "linear_svc",
        "sgd_classifier",
        "multinomial_nb",
        "random_forest",
        "extra_trees"
    ]
