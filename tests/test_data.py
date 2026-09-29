import numpy as np
import pandas as pd
import pytest

from autoimmune_ml.data import load_expression, sample_groups


def test_load_expression_layout_and_labels(synthetic_csv):
    X, y = load_expression(synthetic_csv, positive_prefix="D")
    assert X.shape == (40, 25)
    assert list(X.index) == list(y.index)
    assert y.sum() == 20
    # Labels come from the sample name, row by row (the notebooks lost this alignment).
    assert all(y[name] == int(name.startswith("D")) for name in X.index)


def test_load_expression_cleans_input(tmp_path):
    table = pd.DataFrame(
        {
            "Unnamed: 0": [0, 1, 2],
            "Gene": ["A", "B", "A"],
            "SLE1_W0": [1.0, 2.0, 3.0],
            "healthy_1": [4.0, "n/a", 6.0],
        }
    )
    path = tmp_path / "t.csv"
    table.to_csv(path, index=False)
    X, y = load_expression(path, positive_prefix="S")
    assert list(X.columns) == ["A", "B"]  # duplicate genes averaged, index column dropped
    assert X.loc["SLE1_W0", "A"] == 2.0
    assert not X.isna().any().any()
    assert y.to_dict() == {"SLE1_W0": 1, "healthy_1": 0}


def test_single_class_is_rejected(synthetic_csv):
    with pytest.raises(ValueError, match="both classes"):
        load_expression(synthetic_csv, positive_prefix="nomatch")


def test_missing_file_has_helpful_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="data/README.md"):
        load_expression(tmp_path / "absent.csv", positive_prefix="D")


def test_sample_groups():
    idx = pd.Index(["SLE1_W0", "SLE1_W12", "SLE2_W0", "healthy_3"])
    groups = sample_groups(idx, r"^(SLE\d+)_")
    assert groups.tolist() == ["SLE1", "SLE1", "SLE2", "healthy_3"]
    assert sample_groups(idx, None) is None
