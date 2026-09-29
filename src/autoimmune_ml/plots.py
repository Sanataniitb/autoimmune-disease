"""Figures for cross-validated results."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve  # noqa: E402

from .evaluate import ModelResult  # noqa: E402
from .models import model_label  # noqa: E402


def plot_roc(results: list[ModelResult], y: pd.Series, title: str, path: Path) -> None:
    """Out-of-fold ROC curve per model (first CV repeat), AUC averaged over repeats."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for res in results:
        fpr, tpr, _ = roc_curve(y, res.oof_proba[0])
        aucs = [roc_auc_score(y, p) for p in res.oof_proba]
        ax.plot(fpr, tpr, lw=1.8, label=f"{model_label(res.model)} (AUC {np.mean(aucs):.2f} ± {np.std(aucs):.2f})")
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1)
    ax.set(xlabel="False positive rate (1 − specificity)", ylabel="True positive rate (sensitivity)",
           title=f"{title}: cross-validated ROC", xlim=(0, 1), ylim=(0, 1.01))
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(result: ModelResult, y: pd.Series, title: str, path: Path, threshold: float = 0.5) -> None:
    """Confusion matrix from out-of-fold predictions of the first CV repeat."""
    pred = (result.oof_proba[0] >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Control", "Disease"], yticklabels=["Control", "Disease"],
           xlabel="Predicted", ylabel="Actual", title=f"{title}\n{model_label(result.model)} (out-of-fold)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_feature_frequency(freq: pd.Series, title: str, path: Path, top: int = 20) -> None:
    """Horizontal bar chart of how often each gene was selected across CV folds."""
    data = freq.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(5.5, max(2.5, 0.28 * len(data) + 1)))
    ax.barh(data.index, data.values, color="#3b6fb6")
    ax.set(xlim=(0, 1), xlabel="Fraction of CV folds selected", title=f"{title}: most stable genes")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
