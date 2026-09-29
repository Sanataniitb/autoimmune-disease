# Autoimmune disease classification from RNA-seq

[![tests](https://github.com/Sanataniitb/autoimmune-disease/actions/workflows/tests.yml/badge.svg)](https://github.com/Sanataniitb/autoimmune-disease/actions/workflows/tests.yml)
[![Open quickstart in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sanataniitb/autoimmune-disease/blob/main/notebooks/quickstart.ipynb)

This project trains machine-learning classifiers that separate patients with an
autoimmune disease from healthy controls, using gene expression (RNA-seq) data.
It covers four diseases:

| Disease | Config | Samples (disease / control) | Input genes |
|---|---|---|---|
| Rheumatoid arthritis (RA) | [`configs/rheumatoid_arthritis.yaml`](configs/rheumatoid_arthritis.yaml) | 63 / 43 | 56 |
| Systemic lupus erythematosus (SLE) | [`configs/sle.yaml`](configs/sle.yaml) | ~166 / ~50 | 100 |
| Type 1 diabetes (T1D) | [`configs/t1d.yaml`](configs/t1d.yaml) | ~140 / ~40 | 166 |
| Ankylosing spondylitis (AS) | [`configs/ankylosing_spondylitis.yaml`](configs/ankylosing_spondylitis.yaml) | ~26 / ~40 | 83 |

Totals come from the notebook outputs. The disease/control split is exact for RA
and estimated from the test splits for the other cohorts.

The repository has two parts:

- **`autoimmune_ml`**, a tested Python package and command-line tool. It runs a
  leak-free pipeline: gene selection → scaling → classifier, evaluated with
  repeated stratified cross-validation.
- **`notebooks/original/`**, the original exploratory Colab notebooks, kept for
  reference. They contain methodological errors that make their reported
  scores unreliable; see [docs/methodology.md](docs/methodology.md).

> [!IMPORTANT]
> This is research code. It is not a validated diagnostic tool and should not be used for clinical decisions.

## Quick start

```bash
git clone https://github.com/Sanataniitb/autoimmune-disease.git
cd autoimmune-disease
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Try the pipeline on synthetic data (no patient data needed)
autoimmune-ml synthetic
autoimmune-ml run configs/synthetic_demo.yaml
```

Example output:

```
Synthetic demo: 40 disease / 40 control, 60 genes  ->  results/synthetic_demo
                         roc_auc balanced_accuracy     f1_macro  sensitivity  specificity
Logistic regression  0.92 ± 0.06       0.84 ± 0.08  0.83 ± 0.09  0.85 ± 0.14  0.82 ± 0.14
SVM (RBF)            0.90 ± 0.07       0.81 ± 0.09  0.81 ± 0.09  0.86 ± 0.12  0.76 ± 0.13
Random forest        0.90 ± 0.07       0.81 ± 0.09  0.81 ± 0.09  0.84 ± 0.14  0.79 ± 0.13
XGBoost              0.84 ± 0.09       0.78 ± 0.07  0.78 ± 0.07  0.80 ± 0.14  0.77 ± 0.15
```

To run on a real cohort, put the expression file in `data/`
([expected names and format](data/README.md)) and run:

```bash
autoimmune-ml run configs/sle.yaml                         # one disease
autoimmune-ml run configs/*.yaml --output results/         # all diseases
autoimmune-ml run configs/t1d.yaml --data /path/to/file.xlsx --tune --repeats 10
```

Prefer a notebook? Open [`notebooks/quickstart.ipynb`](notebooks/quickstart.ipynb)
(Colab badge above). It installs the package and walks through the same steps.

### CLI options

| Option | Default | Description |
|---|---|---|
| `--models` | from config | Any of `logreg svm random_forest xgboost mlp` |
| `--folds` / `--repeats` | 5 / 5 | Repeated stratified k-fold CV |
| `--tune` | off | Nested-CV hyper-parameter search |
| `--data` | from config | Override the input file |
| `--output` | `results/` | Output directory |
| `--seed` | 42 | Random seed |

### Outputs (`results/<disease>/`)

| File | Contents |
|---|---|
| `summary.csv` | Mean ± SD of ROC AUC, balanced accuracy, F1, precision, sensitivity, specificity per model |
| `folds_<model>.csv` | Metrics for every CV fold |
| `roc.png` | Out-of-fold ROC curves for all models |
| `confusion_matrix.png` | Out-of-fold confusion matrix for the best model |
| `feature_stability.csv/.png` | How often each gene was selected across folds (candidate biomarkers) |
| `run_info.json` | Cohort size, CV settings, best model, package version |

## How it works

```
genes × samples table ──► X (samples × genes), y (from sample-name prefix)
                               │
            ┌──────── repeated stratified k-fold (grouped by patient if configured) ────────┐
            │ train fold:  log1p → ExtraTrees gene selection → StandardScaler → classifier  │
            │ test fold:   predict disease probability                                      │
            └───────────────────────────────────────────────────────────────────────────────┘
                               │
             metrics per fold · out-of-fold ROC · gene selection frequency
```

All preprocessing is fitted inside each training fold, so no information from
the test samples reaches the model. [docs/methodology.md](docs/methodology.md)
describes the pipeline in detail and lists every issue found in the original
notebooks.

### Configuring a cohort

Each disease is a small YAML file:

```yaml
name: t1d
display_name: Type 1 diabetes
data_path: "data/T1D ML Input manual feature 167.xlsx"
gene_column: Gene
positive_prefix: "lib"        # sample names starting with this are disease
log_transform: true           # raw counts → log1p
importance_threshold: 0.003   # ExtraTrees importance cut-off
always_include: [HLA-DQB1, INS, PTPN22, CTLA4, IL2RA]   # known risk genes
group_regex: null             # e.g. '^(SLE\d+)_' to group visits by patient
models: [logreg, svm, random_forest, xgboost, mlp]
```

To add a new disease, copy one of these files and change the values.

### Using the package from Python

```python
from autoimmune_ml import load_expression, cross_validate_model

X, y = load_expression("data/GSE116006_SLE_N ml input.csv", positive_prefix="S")
res = cross_validate_model(X, y, "logreg", log_transform=False, n_repeats=10)
print(res.summary()[["roc_auc_mean", "balanced_accuracy_mean"]])
print(res.feature_frequency().head(10))
```

## Key findings from the code review

The original notebooks reported test accuracies up to 0.92 (RA, neural network).
Those numbers are **not reliable**. The main reasons:

1. `standardize_features_by_class` reorders the samples by class but leaves the
   labels in their original order. Most samples end up with the wrong label
   (about 81% in RA).
2. Standardisation uses each sample's class, which is unknown for a new patient.
3. Feature selection ran on the full dataset before the train/test split. In
   simulation, this alone gives AUC ≈ 0.77 on pure noise.
4. The test set was used as the neural network's validation set.
5. ROC curves were drawn from 0/1 predictions, several metrics were printed from
   stale variables, and the SVM (`gamma=1`) predicted a single class.
6. **Study design:** the healthy controls appear to come from one external
   cohort shared across all diseases. Some genes are zero in every control but
   high in every case, which points to a batch effect. Validate with
   same-study controls or batch correction before drawing biological conclusions.

Full details and recommendations: [docs/methodology.md](docs/methodology.md).

## Repository layout

```
├── configs/                 # one YAML per disease cohort (+ synthetic demo)
├── data/                    # input files go here (git-ignored), see data/README.md
├── docs/methodology.md      # pipeline details and review of the original notebooks
├── notebooks/
│   ├── quickstart.ipynb     # Colab-ready walkthrough of the package
│   └── original/            # original exploratory notebooks (RA, SLE, T1D, AS)
├── src/autoimmune_ml/
│   ├── config.py            # YAML config → DiseaseConfig
│   ├── data.py              # loading, labelling, patient grouping, synthetic data
│   ├── models.py            # gene selector + model registry + pipeline builder
│   ├── evaluate.py          # repeated / grouped / nested CV and metrics
│   ├── plots.py             # ROC, confusion matrix, feature stability figures
│   └── cli.py               # `autoimmune-ml` command
├── tests/                   # pytest suite (runs in CI on every push)
├── pyproject.toml
└── requirements.txt
```

## Development

```bash
pip install -e ".[dev]"
pytest -q
```

CI (GitHub Actions) runs the tests on Python 3.10 and 3.12, then an end-to-end
run on synthetic data. Keep patient-level data out of commits: `data/` and
`results/` are git-ignored.

## Roadmap

- Batch-effect assessment (PCA coloured by cohort) and ComBat-seq correction
- External validation cohort per disease
- Calibration curves and decision-curve analysis
- Multi-class model across diseases (RA vs SLE vs T1D vs AS vs healthy)
- SHAP explanations for the selected genes
