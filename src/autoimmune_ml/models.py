"""Leak-free modelling pipelines: feature selection -> scaling -> classifier."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC

log = logging.getLogger(__name__)


class ImportanceSelector(TransformerMixin, BaseEstimator):
    """Keep genes whose ExtraTrees importance exceeds ``threshold``.

    Replaces the notebooks' selection step, which was fitted on the full
    dataset (including test samples) before the train/test split. As a
    pipeline step it is refit on the training portion of every fold.

    Genes listed in ``always_include`` are kept regardless of importance.
    If nothing passes the threshold the ``min_features`` most important genes
    are kept so downstream models always receive input.
    """

    def __init__(self, threshold=0.003, always_include=(), min_features=5, n_estimators=300, random_state=0):
        self.threshold = threshold
        self.always_include = always_include
        self.min_features = min_features
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, y):
        X = self._as_frame(X)
        forest = ExtraTreesClassifier(
            n_estimators=self.n_estimators, random_state=self.random_state, n_jobs=-1, class_weight="balanced"
        )
        forest.fit(X.values, y)
        self.importances_ = pd.Series(forest.feature_importances_, index=X.columns)

        keep = set(self.importances_[self.importances_ > self.threshold].index)
        if len(keep) < self.min_features:
            keep |= set(self.importances_.nlargest(self.min_features).index)
        keep |= {g for g in self.always_include if g in X.columns}
        # Preserve original column order for reproducibility.
        self.selected_features_ = [c for c in X.columns if c in keep]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        return self._as_frame(X)[self.selected_features_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_features_, dtype=object)

    def _as_frame(self, X):
        if isinstance(X, pd.DataFrame):
            return X
        columns = getattr(self, "feature_names_in_", None)
        return pd.DataFrame(X, columns=columns)


def _logreg():
    return LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000)


def _svm():
    # gamma="scale" instead of the notebooks' gamma=1, which collapsed to
    # predicting a single class for every sample (see docs/methodology.md).
    # Platt-scaled probabilities (replaces the deprecated SVC(probability=True)).
    svc = SVC(C=1.0, kernel="rbf", gamma="scale", class_weight="balanced")
    return CalibratedClassifierCV(svc, method="sigmoid", cv=3, ensemble=False)


def _random_forest():
    return RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=0, n_jobs=-1)


def _xgboost():
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("xgboost is not installed; `pip install xgboost` or drop it from `models`") from exc
    return XGBClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=0, n_jobs=1,
    )


def _mlp():
    # A compact stand-in for the notebooks' 256-128-64-32 Keras network, which
    # has far more parameters than there are samples in any cohort.
    return MLPClassifier(
        hidden_layer_sizes=(64, 32), alpha=1e-2, max_iter=2000, early_stopping=False, random_state=0,
    )


MODEL_REGISTRY: dict[str, tuple[str, Callable[[], BaseEstimator]]] = {
    "logreg": ("Logistic regression", _logreg),
    "svm": ("SVM (RBF)", _svm),
    "random_forest": ("Random forest", _random_forest),
    "xgboost": ("XGBoost", _xgboost),
    "mlp": ("MLP", _mlp),
}

# Small grids used when --tune is passed (nested CV).
PARAM_GRIDS: dict[str, dict[str, list]] = {
    "logreg": {"clf__C": [0.01, 0.1, 1.0, 10.0]},
    "svm": {"clf__estimator__C": [0.1, 1.0, 10.0], "clf__estimator__gamma": ["scale", 0.01, 0.1]},
    "random_forest": {"clf__max_depth": [None, 5], "clf__min_samples_leaf": [1, 3]},
    "xgboost": {"clf__max_depth": [2, 3, 5], "clf__n_estimators": [50, 150]},
    "mlp": {"clf__alpha": [1e-3, 1e-2, 1e-1]},
}


def build_pipeline(
    model: str,
    importance_threshold: float = 0.003,
    always_include: list[str] | tuple[str, ...] = (),
    log_transform: bool = True,
    random_state: int = 0,
) -> Pipeline:
    """Return an unfitted ``Pipeline`` for ``model``.

    Steps: optional ``log1p`` -> importance-based gene selection -> z-scoring
    -> classifier. Every step is fitted on training data only.
    """
    if model not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model {model!r}; choose from {sorted(MODEL_REGISTRY)}")
    clf = MODEL_REGISTRY[model][1]()
    if "random_state" in clf.get_params():
        clf.set_params(random_state=random_state)

    steps = []
    if log_transform:
        steps.append(("log1p", FunctionTransformer(_safe_log1p, feature_names_out="one-to-one")))
    steps += [
        ("select", ImportanceSelector(importance_threshold, tuple(always_include), random_state=random_state)),
        ("scale", StandardScaler()),
        ("clf", clf),
    ]
    return Pipeline(steps)


def _safe_log1p(X):
    """log1p for non-negative count-like data; pass through data that is already log-scaled."""
    values = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
    if np.nanmin(values) < 0:
        return X
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.log1p(X)


def model_label(key: str) -> str:
    return MODEL_REGISTRY[key][0]
