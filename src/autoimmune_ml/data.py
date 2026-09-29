"""Loading RNA-seq expression matrices into a samples x genes design matrix."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a CSV/TSV/Excel file based on its extension."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Input data not found: {path}\n"
            "Place the expression matrix there (see data/README.md), pass --data, "
            "or generate a demo file with `autoimmune-ml synthetic`."
        )
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def load_expression(
    path: str | Path,
    positive_prefix: str,
    gene_column: str = "Gene",
) -> tuple[pd.DataFrame, pd.Series]:
    """Load a genes x samples matrix and return ``(X, y)``.

    ``X`` is a float DataFrame indexed by sample name with one column per gene.
    ``y`` is an int Series (1 = disease, 0 = control) aligned to ``X.index``;
    labels are derived from the sample-name prefix, as in the original notebooks.
    """
    raw = read_table(path)
    if gene_column not in raw.columns:
        raise ValueError(f"Column {gene_column!r} not found in {path}; columns start with {list(raw.columns[:5])}")

    raw = raw.dropna(subset=[gene_column])
    raw[gene_column] = raw[gene_column].astype(str).str.strip()
    # Drop stray index columns such as "Unnamed: 0" that some exports add.
    raw = raw.loc[:, ~raw.columns.astype(str).str.startswith("Unnamed")]

    duplicated = raw[gene_column].duplicated()
    if duplicated.any():
        log.warning("%d duplicated gene symbols; averaging their rows", int(duplicated.sum()))
        raw = raw.groupby(gene_column, sort=False).mean(numeric_only=True).reset_index()

    X = raw.set_index(gene_column).T
    X.index = X.index.astype(str)
    X.columns.name = None
    X = X.apply(pd.to_numeric, errors="coerce").astype(float)

    if X.isna().any().any():
        n_missing = int(X.isna().sum().sum())
        log.warning("%d non-numeric/missing values; imputing with per-gene median", n_missing)
        X = X.fillna(X.median())

    y = pd.Series(X.index.str.startswith(positive_prefix).astype(int), index=X.index, name="target")
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"Prefix {positive_prefix!r} gives {n_pos} disease and {n_neg} control samples; "
            f"both classes are required. Sample names look like {list(X.index[:5])}"
        )
    log.info("Loaded %d samples (%d disease, %d control) x %d genes", len(X), n_pos, n_neg, X.shape[1])
    return X, y


def sample_groups(index: pd.Index, pattern: str | None) -> pd.Series | None:
    """Map sample names to group IDs (e.g. patients) with a regex capture group.

    Samples that do not match keep their own name as group. Returns ``None``
    when ``pattern`` is empty, meaning every sample is independent.
    """
    if not pattern:
        return None
    ids = index.to_series().str.extract(pattern, expand=False)
    groups = ids.fillna(index.to_series())
    n_shared = int((groups.value_counts() > 1).sum())
    log.info("Grouping samples by %r: %d groups, %d with repeated samples", pattern, groups.nunique(), n_shared)
    return groups


def make_synthetic(
    n_samples: int = 80,
    n_genes: int = 60,
    n_informative: int = 8,
    positive_fraction: float = 0.5,
    effect_size: float = 0.5,
    positive_prefix: str = "D",
    control_prefix: str = "N",
    seed: int = 0,
) -> pd.DataFrame:
    """Create a genes x samples table in the same layout as the real inputs.

    Useful for trying the pipeline without access to the (non-public) cohort files.
    ``effect_size`` is the typical absolute log fold-change of informative genes.
    """
    rng = np.random.default_rng(seed)
    n_pos = int(round(n_samples * positive_fraction))
    labels = np.array([1] * n_pos + [0] * (n_samples - n_pos))
    expr = rng.lognormal(mean=3.0, sigma=0.6, size=(n_genes, n_samples))
    shift = effect_size * rng.uniform(0.6, 1.4, size=n_informative) * rng.choice([-1, 1], size=n_informative)
    expr[:n_informative, labels == 1] *= np.exp(shift)[:, None]

    genes = [f"GENE{i:03d}" for i in range(n_genes)]
    samples = [f"{positive_prefix}{i:03d}" if lab else f"{control_prefix}{i:03d}" for i, lab in enumerate(labels)]
    table = pd.DataFrame(np.round(expr, 3), columns=samples)
    table.insert(0, "Gene", genes)
    return table
