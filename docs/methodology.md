# Methodology and review of the original notebooks

This document explains how the pipeline in `src/autoimmune_ml` works, and why
it differs from the original Colab notebooks in `notebooks/original/`.

## Pipeline

For each disease cohort:

```
expression table (genes × samples)
  └─ transpose → X (samples × genes), y from sample-name prefix
       └─ repeated stratified k-fold CV (grouped by patient where configured)
            for each training fold:
              log1p (raw counts only)
              → ExtraTrees importance gene selection (+ always_include genes)
              → StandardScaler
              → classifier (LogReg | SVM | Random forest | XGBoost | MLP)
            predict disease probability on the held-out fold
       └─ metrics per fold, out-of-fold ROC, gene selection frequency
```

Every step that learns from data is inside a scikit-learn `Pipeline`, so it is
refit on the training samples of each fold and never sees the test samples.

**Metrics.** ROC AUC is computed from predicted probabilities. Balanced accuracy,
macro F1, precision, sensitivity (recall) and specificity use a 0.5 threshold.
The reported value is mean ± standard deviation across all folds and repeats.
Balanced accuracy is used instead of plain accuracy because the cohorts are
imbalanced; for example SLE has about 3× more disease samples than controls.

**Hyper-parameters.** Defaults are fixed and conservative. `--tune` runs a small
grid search in an inner 3-fold CV inside each outer training fold (nested CV),
so tuning does not inflate the reported scores.

**Feature stability.** `feature_stability.csv` reports the fraction of folds in
which each gene was selected by the best model's pipeline. Genes selected in
nearly every fold are the most robust candidate biomarkers. Genes selected once
from the full dataset are not.

## Issues found in the original notebooks

The notebooks are kept unchanged in `notebooks/original/` (apart from removing
an ~8.5 MB cached TensorBoard widget from each and fixing the Colab links).
The problems below mean **the metrics in those notebooks should not be quoted
as diagnostic performance.**

### Critical: these invalidate the reported results

1. **Features and labels are misaligned.** `standardize_features_by_class`
   builds its output by concatenating all class-0 rows and then all class-1
   rows. `y` keeps the original sample order, and in every file the disease
   samples come first. After this step, row *i* of `standardized_X` usually
   belongs to a different sample than `y[i]`. For the RA file (63 disease
   followed by 43 controls), about 81% of rows get the wrong label.
2. **Standardisation uses the label.** Even with correct ordering, each class is
   centred with its own mean. This removes the between-class difference that a
   classifier should learn. It also cannot be applied to a new patient, whose
   class is unknown.
3. **Feature selection sees the test set.** `ExtraTreesClassifier` importances
   are computed on all samples before `train_test_split`. In a simulation on
   pure noise (100 samples × 200 genes, 20 seeds), choosing the top 20 genes this
   way gave a held-out logistic regression **AUC ≈ 0.77 when the true value is 0.5**.
4. **The test set is used for model selection.** The Keras model uses
   `validation_data=(X_test, y_test)` while training. Architecture, epochs and
   thresholds were then compared on the same test samples.

### Evaluation errors

5. **ROC curves from hard labels.** For logistic regression and SVM,
   `roc_curve` receives `predict()` outputs (0/1), so each "curve" has a single
   point and the AUC is really balanced accuracy.
6. **Stale variables.** In the XGBoost section, the "train F1" printout reuses
   `y_tpred` from logistic regression. In SLE, T1D and AS the SVM confusion matrix
   shows the logistic regression's `cm`. SLE cell 52 applies the threshold
   function to `y_pred`, which had already been thresholded.
7. **The optimal threshold is tuned on training predictions.** XGBoost reaches
   train AUC = 1.0, so a threshold chosen from training ROC tells you little.
8. **A single, unstratified 77/23 split.** With test sets of 14 to 50 samples,
   one split gives very noisy estimates. For example, AS has only 6 disease
   samples in its test set.

### Modelling issues

9. **SVM with `gamma=1`** on z-scored data with 27–96 features predicts one class
   for every sample (confusion matrices `[[0 10] [0 15]]` in RA). The new
   default `gamma="scale"` avoids this.
10. **Oversized neural network.** 256-128-64-32 dense layers (~55,000 parameters)
    for 66–216 samples. It is replaced by a small, regularised MLP.
11. **Class imbalance is ignored.** The new models use `class_weight="balanced"`
    where supported, and metrics are balanced.
12. **Raw counts without transformation.** RA, T1D and AS inputs are raw read
    counts spanning 0 to ~20,000. The pipeline applies `log1p` when
    `log_transform: true`.

### Study design caveats (not fixable in code)

- **Probable batch confound.** The control columns (`healthy_34`, `healthy_35`,
  …, `Health_34`, …) appear to come from a single external healthy cohort that
  is reused across diseases. Some genes are near zero in every control and high
  in every case (e.g. `CORO7-PAM16`: thousands of reads in RA samples vs 0–1 in controls).
  That pattern suggests differences in library preparation or annotation, not
  biology. A classifier can then separate *datasets* instead of *disease states*.
  Before claiming diagnostic value:
  - use controls sequenced in the same study as the cases, or apply batch
    correction (e.g. ComBat-seq) before feature selection;
  - check that top genes are not simply zero in one group;
  - validate on an independent cohort.
- **Pre-selected features.** The input files already contain only 56–166 genes,
  chosen upstream (e.g. by differential expression on the same samples). That
  step is outside this pipeline and can still bias results optimistically.
- **Repeated samples per patient.** SLE sample names (`SLE<id>_W<week>`) suggest
  several visits per patient. `group_regex` keeps a patient's samples in the
  same fold. Check that it matches your naming.
