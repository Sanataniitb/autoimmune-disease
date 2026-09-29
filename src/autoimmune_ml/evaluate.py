"""Cross-validated evaluation of the modelling pipelines."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedGroupKFold, StratifiedKFold

from .models import PARAM_GRIDS, build_pipeline

log = logging.getLogger(__name__)


@dataclass
class ModelResult:
    """Out-of-fold predictions and per-fold metrics for one model."""

    model: str
    fold_metrics: pd.DataFrame
    # Out-of-fold disease probability per repeat: shape (n_repeats, n_samples).
    oof_proba: np.ndarray
    selected_counts: Counter = field(default_factory=Counter)
    n_folds_total: int = 0

    def summary(self) -> pd.Series:
        stats = self.fold_metrics.drop(columns=["repeat", "fold"]).agg(["mean", "std"])
        out = {f"{metric}_{stat}": stats.loc[stat, metric] for metric in stats.columns for stat in ("mean", "std")}
        return pd.Series(out, name=self.model)

    def feature_frequency(self) -> pd.Series:
        """Fraction of folds in which each gene was selected (a stability measure)."""
        if not self.n_folds_total:
            return pd.Series(dtype=float)
        freq = pd.Series(self.selected_counts, dtype=float) / self.n_folds_total
        return freq.sort_values(ascending=False)


def classification_metrics(y_true, proba, threshold: float = 0.5) -> dict[str, float]:
    """Threshold-free (ROC AUC) and thresholded metrics for binary predictions."""
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": roc_auc_score(y_true, proba) if len(np.unique(y_true)) == 2 else np.nan,
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "f1_macro": f1_score(y_true, pred, average="macro", zero_division=0),
        "precision": precision_score(y_true, pred, zero_division=0),
        "sensitivity": recall_score(y_true, pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
    }


def cross_validate_model(
    X: pd.DataFrame,
    y: pd.Series,
    model: str,
    *,
    importance_threshold: float = 0.003,
    always_include: list[str] | tuple[str, ...] = (),
    log_transform: bool = True,
    n_splits: int = 5,
    n_repeats: int = 5,
    tune: bool = False,
    groups: pd.Series | None = None,
    random_state: int = 42,
) -> ModelResult:
    """Repeated stratified k-fold CV with all preprocessing inside each fold.

    With ``tune=True`` a small hyper-parameter grid is searched with an inner
    stratified 3-fold CV (nested CV), so the reported scores stay unbiased.

    ``groups`` (e.g. patient IDs) keeps all samples of a group in the same
    fold, so repeated samples from one patient cannot leak between train and test.
    """
    n_splits = min(n_splits, int(y.value_counts().min()))
    if n_splits < 2:
        raise ValueError("Need at least 2 samples in each class for cross-validation")

    oof = np.full((n_repeats, len(y)), np.nan)
    rows, counts, n_total = [], Counter(), 0

    for i, (train_idx, test_idx) in enumerate(_cv_splits(X, y, groups, n_splits, n_repeats, random_state)):
        repeat, fold = divmod(i, n_splits)
        estimator = build_pipeline(
            model,
            importance_threshold=importance_threshold,
            always_include=always_include,
            log_transform=log_transform,
            random_state=random_state + i,
        )
        if tune and PARAM_GRIDS.get(model):
            inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state + i)
            estimator = GridSearchCV(estimator, PARAM_GRIDS[model], cv=inner, scoring="roc_auc", n_jobs=-1)

        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        estimator.fit(X_train, y_train)
        fitted = estimator.best_estimator_ if tune and hasattr(estimator, "best_estimator_") else estimator

        proba = fitted.predict_proba(X.iloc[test_idx])[:, 1]
        oof[repeat, test_idx] = proba
        rows.append({"repeat": repeat, "fold": fold, **classification_metrics(y.iloc[test_idx], proba)})

        counts.update(fitted.named_steps["select"].selected_features_)
        n_total += 1

    return ModelResult(model, pd.DataFrame(rows), oof, counts, n_total)


def _cv_splits(X, y, groups, n_splits, n_repeats, random_state):
    if groups is None:
        yield from RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state).split(X, y)
        return
    for r in range(n_repeats):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state + r)
        yield from cv.split(X, y, groups)


def fit_final_model(X, y, model: str, **pipeline_kwargs):
    """Fit a pipeline on all samples (for inspection or scoring new samples)."""
    return clone(build_pipeline(model, **pipeline_kwargs)).fit(X, y)
