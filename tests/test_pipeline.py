import numpy as np
import pandas as pd
import pytest

from autoimmune_ml import build_pipeline, classification_metrics, cross_validate_model, load_expression
from autoimmune_ml.cli import main
from autoimmune_ml.evaluate import _cv_splits
from autoimmune_ml.models import MODEL_REGISTRY, ImportanceSelector


@pytest.fixture
def xy(synthetic_csv):
    return load_expression(synthetic_csv, positive_prefix="D")


def test_selector_keeps_forced_genes_and_minimum(xy):
    X, y = xy
    sel = ImportanceSelector(threshold=1.0, always_include=("GENE020", "NOT_A_GENE"), min_features=3).fit(X, y)
    assert "GENE020" in sel.selected_features_
    assert "NOT_A_GENE" not in sel.selected_features_
    assert len(sel.selected_features_) == 4
    assert list(sel.transform(X).columns) == sel.selected_features_


@pytest.mark.parametrize("model", sorted(MODEL_REGISTRY))
def test_every_model_fits_and_predicts_probabilities(xy, model):
    X, y = xy
    proba = build_pipeline(model).fit(X, y).predict_proba(X)[:, 1]
    assert proba.shape == (len(y),)
    assert np.all((proba >= 0) & (proba <= 1))


def test_cv_finds_signal(xy):
    X, y = xy
    res = cross_validate_model(X, y, "logreg", n_splits=4, n_repeats=2)
    assert len(res.fold_metrics) == 8
    assert not np.isnan(res.oof_proba).any()
    assert res.summary()["roc_auc_mean"] > 0.75
    assert res.feature_frequency().max() == 1.0


def test_no_leakage_on_random_labels():
    """With labels unrelated to expression, honest CV must stay near chance.

    The original notebooks' class-wise standardisation and whole-dataset
    feature selection would score far above 0.5 here.
    """
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(60, 200)), columns=[f"G{i}" for i in range(200)])
    y = pd.Series(rng.permutation([0, 1] * 30))
    res = cross_validate_model(X, y, "logreg", n_splits=5, n_repeats=3, log_transform=False)
    assert abs(res.summary()["roc_auc_mean"] - 0.5) < 0.15


def test_grouped_cv_never_splits_a_group():
    X = pd.DataFrame(np.zeros((24, 2)))
    y = pd.Series([1] * 12 + [0] * 12)
    groups = pd.Series([f"p{i // 2}" for i in range(24)])
    for train, test in _cv_splits(X, y, groups, n_splits=3, n_repeats=2, random_state=0):
        assert not set(groups.iloc[train]) & set(groups.iloc[test])


@pytest.mark.parametrize("model", ["logreg", "svm"])
def test_nested_tuning_runs(xy, model):
    X, y = xy
    res = cross_validate_model(X, y, model, n_splits=3, n_repeats=1, tune=True)
    assert len(res.fold_metrics) == 3


def test_classification_metrics():
    m = classification_metrics([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9])
    assert m["roc_auc"] == 0.75
    assert m["sensitivity"] == 0.5 and m["specificity"] == 0.5


def test_cli_end_to_end(tmp_path, synthetic_csv):
    cfg = tmp_path / "demo.yaml"
    cfg.write_text(
        f"name: demo\ndisplay_name: Demo\ndata_path: {synthetic_csv}\npositive_prefix: D\n"
        "always_include: [GENE024, MISSING]\n"
    )
    out = tmp_path / "results"
    assert main(["run", str(cfg), "--output", str(out), "--folds", "3", "--repeats", "1", "--models", "logreg"]) == 0
    for name in ["summary.csv", "folds_logreg.csv", "feature_stability.csv", "roc.png",
                 "confusion_matrix.png", "feature_stability.png", "run_info.json"]:
        assert (out / "demo" / name).exists(), name
    freq = pd.read_csv(out / "demo" / "feature_stability.csv", index_col="gene")["selection_frequency"]
    assert freq["GENE024"] == 1.0
