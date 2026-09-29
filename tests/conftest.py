import pytest

from autoimmune_ml import make_synthetic


@pytest.fixture
def synthetic_csv(tmp_path):
    path = tmp_path / "expr.csv"
    make_synthetic(n_samples=40, n_genes=25, n_informative=5, effect_size=1.0, seed=1).to_csv(path, index=False)
    return path
