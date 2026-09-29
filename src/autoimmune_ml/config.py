"""Per-disease experiment configuration, loaded from YAML files in ``configs/``."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DiseaseConfig:
    """Everything needed to run the pipeline for one disease cohort.

    Attributes
    ----------
    name:
        Short identifier, used for output folder names (e.g. ``"sle"``).
    display_name:
        Human readable name used in plots and reports.
    data_path:
        Expression matrix (CSV or Excel). Genes are rows, samples are columns,
        and the gene identifiers live in ``gene_column``.
    positive_prefix:
        Sample (column) names starting with this prefix are labelled as
        disease (1); all other samples are controls (0).
    gene_column:
        Name of the column holding gene symbols.
    importance_threshold:
        Minimum ExtraTrees feature importance for a gene to be kept. Selection
        is refit inside every training fold, so it never sees test samples.
    log_transform:
        Apply ``log1p`` before modelling. Use ``true`` for raw counts and
        ``false`` for data that is already log-scaled.
    always_include:
        Genes that are always kept regardless of importance (e.g. known
        disease-associated loci). Missing genes are ignored with a warning.
    group_regex:
        Optional regex whose first capture group extracts a patient ID from
        the sample name. Samples sharing an ID are kept in the same CV fold.
    models:
        Model keys to evaluate (see :data:`autoimmune_ml.models.MODEL_REGISTRY`).
    """

    name: str
    display_name: str
    data_path: Path
    positive_prefix: str
    gene_column: str = "Gene"
    importance_threshold: float = 0.003
    log_transform: bool = True
    always_include: list[str] = field(default_factory=list)
    group_regex: str | None = None
    models: list[str] = field(default_factory=lambda: ["logreg", "svm", "random_forest", "xgboost", "mlp"])
    notes: str = ""

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DiseaseConfig":
        path = Path(path)
        with path.open() as fh:
            raw = yaml.safe_load(fh) or {}
        missing = {"name", "display_name", "data_path", "positive_prefix"} - raw.keys()
        if missing:
            raise ValueError(f"{path}: missing required keys {sorted(missing)}")
        data_path = Path(raw.pop("data_path"))
        if not data_path.is_absolute():
            # Resolve relative to the repository root (the parent of configs/).
            data_path = (path.parent.parent / data_path).resolve()
        return cls(data_path=data_path, **raw)
