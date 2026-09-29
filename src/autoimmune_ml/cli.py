"""Command line interface: ``autoimmune-ml run`` and ``autoimmune-ml synthetic``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from . import __version__
from .config import DiseaseConfig
from .data import load_expression, make_synthetic, sample_groups
from .evaluate import cross_validate_model
from .models import MODEL_REGISTRY, model_label
from .plots import plot_confusion, plot_feature_frequency, plot_roc

log = logging.getLogger("autoimmune_ml")


def run_disease(cfg: DiseaseConfig, out_root: Path, *, models=None, n_splits=5, n_repeats=5, tune=False, seed=42):
    """Run the full evaluation for one disease and write results to ``out_root/<name>/``."""
    X, y = load_expression(cfg.data_path, cfg.positive_prefix, cfg.gene_column)
    missing = [g for g in cfg.always_include if g not in X.columns]
    if missing:
        log.warning("%s: always_include genes not in data and skipped: %s", cfg.name, ", ".join(missing))

    groups = sample_groups(X.index, cfg.group_regex)

    out = out_root / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    models = models or cfg.models

    results = []
    for key in models:
        log.info("[%s] %s: %d x %d-fold CV%s", cfg.name, model_label(key), n_repeats, n_splits, " (nested tuning)" if tune else "")
        res = cross_validate_model(
            X, y, key,
            importance_threshold=cfg.importance_threshold,
            always_include=cfg.always_include,
            log_transform=cfg.log_transform,
            n_splits=n_splits, n_repeats=n_repeats, tune=tune, groups=groups, random_state=seed,
        )
        res.fold_metrics.to_csv(out / f"folds_{key}.csv", index=False)
        results.append(res)

    summary = pd.DataFrame([r.summary() for r in results])
    summary.index.name = "model"
    summary = summary.sort_values("roc_auc_mean", ascending=False)
    summary.to_csv(out / "summary.csv", float_format="%.4f")

    best = next(r for r in results if r.model == summary.index[0])
    freq = best.feature_frequency()
    freq.rename("selection_frequency").to_csv(out / "feature_stability.csv", header=True, index_label="gene")

    plot_roc(results, y, cfg.display_name, out / "roc.png")
    plot_confusion(best, y, cfg.display_name, out / "confusion_matrix.png")
    plot_feature_frequency(freq, cfg.display_name, out / "feature_stability.png")

    meta = {
        "disease": cfg.display_name,
        "data_path": str(cfg.data_path),
        "n_samples": int(len(y)),
        "n_disease": int(y.sum()),
        "n_control": int((1 - y).sum()),
        "n_genes_input": int(X.shape[1]),
        "cv": {"grouped_by": cfg.group_regex, "n_splits": n_splits, "n_repeats": n_repeats, "nested_tuning": tune, "seed": seed},
        "best_model": best.model,
        "version": __version__,
    }
    (out / "run_info.json").write_text(json.dumps(meta, indent=2))
    return summary, meta


def _format_summary(summary: pd.DataFrame) -> str:
    cols = ["roc_auc", "balanced_accuracy", "f1_macro", "sensitivity", "specificity"]
    table = pd.DataFrame(
        {c: summary[f"{c}_mean"].map("{:.2f}".format) + " ± " + summary[f"{c}_std"].map("{:.2f}".format) for c in cols}
    )
    table.index = [model_label(m) for m in summary.index]
    return table.to_string()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="autoimmune-ml", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="cross-validate models for one or more disease configs")
    run.add_argument("configs", nargs="+", type=Path, help="YAML config(s), e.g. configs/sle.yaml")
    run.add_argument("--data", type=Path, help="override data_path (only with a single config)")
    run.add_argument("--models", nargs="+", choices=sorted(MODEL_REGISTRY), help="override models in config")
    run.add_argument("--output", type=Path, default=Path("results"), help="output directory (default: results/)")
    run.add_argument("--folds", type=int, default=5, help="CV folds (default: 5)")
    run.add_argument("--repeats", type=int, default=5, help="CV repeats (default: 5)")
    run.add_argument("--tune", action="store_true", help="nested CV hyper-parameter search")
    run.add_argument("--seed", type=int, default=42)

    syn = sub.add_parser("synthetic", help="write a synthetic dataset in the expected input layout")
    syn.add_argument("--out", type=Path, default=Path("data/synthetic_demo.csv"))
    syn.add_argument("--samples", type=int, default=80)
    syn.add_argument("--genes", type=int, default=60)
    syn.add_argument("--seed", type=int, default=0)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")

    if args.command == "synthetic":
        args.out.parent.mkdir(parents=True, exist_ok=True)
        make_synthetic(n_samples=args.samples, n_genes=args.genes, seed=args.seed).to_csv(args.out, index=False)
        log.info("Wrote %s", args.out)
        return 0

    if args.data and len(args.configs) > 1:
        parser.error("--data can only be used with a single config")

    for path in args.configs:
        cfg = DiseaseConfig.from_yaml(path)
        if args.data:
            cfg.data_path = args.data
        summary, meta = run_disease(
            cfg, args.output, models=args.models, n_splits=args.folds, n_repeats=args.repeats,
            tune=args.tune, seed=args.seed,
        )
        print(f"\n{cfg.display_name}: {meta['n_disease']} disease / {meta['n_control']} control, "
              f"{meta['n_genes_input']} genes  ->  {args.output / cfg.name}")
        print(_format_summary(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
