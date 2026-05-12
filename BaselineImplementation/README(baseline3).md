# TB Drug Resistance Modeling Pipeline

This repository contains a notebook-based pipeline (`modeltrain.ipynb`) for multi-label prediction of TB drug resistance using mutation features extracted from VCF files.

## 1. Overview

The workflow in `modeltrain.ipynb` does the following:

1. Loads CRyPTIC metadata.
2. Cleans and transforms drug phenotype labels.
3. Extracts per-sample mutations from gzipped VCF files.
4. Builds a binary mutation feature matrix.
5. Trains a WDNN-style PyTorch model.
6. Evaluates using 10-fold multilabel stratified shuffle split with clinical-style metrics.

Target drugs used in the notebook:
- `RIF_BINARY_PHENOTYPE`
- `INH_BINARY_PHENOTYPE`
- `EMB_BINARY_PHENOTYPE`

## 2. Data Inputs

Main files/directories used:
- `CRyPTIC_reuse_table_20240917.csv` (metadata and labels)
- `vcf(2024)/<ENA_SAMPLE>.vcf.gz` (variant calls)

The notebook uses `ENA_SAMPLE` from metadata to map each sample to its VCF file path:
- `vcf(2024)/{ENA_SAMPLE}.vcf.gz`

## 3. Preprocessing

### 3.1 Label conversion

Function `convert_label(x)` maps phenotype values:
- `R -> 1`
- `S -> 0`
- other values -> `NaN`

Rows with missing labels in any target drug are removed:
- `df = df.dropna(subset=drug_columns)`

### 3.2 Sample filtering by available VCF

For each metadata row, the notebook checks whether the VCF exists. Samples without VCF are skipped.

### 3.3 Mutation extraction

For each non-header VCF line, a mutation token is built as:
- `"{POS}_{REF}_{ALT}"`

Each sample gets a set of mutation tokens.

## 4. Feature Engineering / Data Transformation

### 4.1 Global mutation vocabulary

All sample mutation sets are merged into a global set.

### 4.2 Frequency filtering

Mutation frequency is counted across samples and filtered with:
- `MIN_COUNT = 5`

Only mutations with count `>= 5` are retained.

### 4.3 Binary matrix creation

A binary matrix `X` is created where:
- rows = samples
- columns = filtered mutation features
- value `1` means mutation present in sample, else `0`

Labels are aligned with feature matrix index:
- `labels = labels.loc[X.index]`

## 5. Model Architecture (WDNN)

The notebook implements a PyTorch WDNN variant (`ExactWDNN`) with:

- Dense block 1: `Linear(input_dim -> 1000) -> ReLU -> BatchNorm -> Dropout(0.3)`
- Dense block 2: `Linear(1000 -> 1000) -> ReLU -> BatchNorm -> Dropout(0.3)`
- Dense block 3: `Linear(1000 -> 1000) -> ReLU -> BatchNorm -> Dropout(0.3)`
- Output: `Linear(1000 -> 3) -> Sigmoid`

Additional details:
- BatchNorm epsilon = `1e-3`
- Keras-style BN momentum translated to PyTorch momentum (`1 - 0.99 = 0.01`)
- Loss = `BCELoss + regularization_loss()`
- `regularization_loss()` includes:
  - L1 on layer weights (`kernel_l1 = 1e-4`)
  - L2 on hidden biases (`hidden_bias_l2 = 1e-3`)
  - stronger L2 on output bias (`output_bias_l2 = 1e-1`)

## 6. Training and Validation Strategy

Current evaluation strategy in the notebook:
- `MultilabelStratifiedShuffleSplit`
- `n_splits = 10`
- `test_size = 0.20` per split
- `epochs = 20`
- `batch_size = 512`
- `learning_rate = 1e-3`
- decision threshold for class prediction = `0.5`

For each fold:
1. Train model on fold training partition.
2. Evaluate on fold validation partition every epoch.
3. Track best epoch by minimum validation loss.
4. Optionally save best fold checkpoint under `checkpoints_cv/`.

## 7. Metrics Implemented

Metrics are computed per drug and macro-averaged across drugs:
- AUC (ROC AUC)
- Sensitivity (Recall, TPR): `TP / (TP + FN)`
- Specificity (TNR): `TN / (TN + FP)`
- Precision (PPV): `TP / (TP + FP)`
- NPV: `TN / (TN + FN)`

The notebook uses safe division and returns `NaN` when denominator is zero for a metric.

## 8. Results from Current Notebook Run

### 8.1 Data pipeline counts

From executed notebook outputs:
- Total metadata samples: `12,287`
- Samples after label filtering: `10,289`
- Samples processed with available VCF: `10,289`
- Total unique mutations before filtering: `1,086,959`
- Mutations after `MIN_COUNT >= 5`: `86,932`
- Final feature matrix shape: `(10,289, 86,932)`

A previous single-split cell (kept in notebook) showed:
- Train: `(7,202, 86,932)`
- Validation: `(1,543, 86,932)`
- Test: `(1,544, 86,932)`

### 8.2 10-fold CV aggregate metrics (best epoch per fold)

Overall mean +/- std across 10 folds:

| Metric | Mean | Std |
|---|---:|---:|
| AUC | 0.934938 | 0.009880 |
| Sensitivity | 0.586329 | 0.181670 |
| Specificity | 0.941320 | 0.053422 |
| Precision | 0.882883 | 0.053926 |
| NPV | 0.848141 | 0.054652 |

### 8.3 Fold-wise macro metrics (best epoch)

| Fold | Best Epoch | Best Val Loss | AUC | Sensitivity | Specificity | Precision | NPV |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 2.313971 | 0.929208 | 0.561018 | 0.819406 | 0.809404 | 0.855110 |
| 2 | 20 | 2.149102 | 0.946855 | 0.735356 | 0.941541 | 0.868469 | 0.887102 |
| 3 | 20 | 2.077448 | 0.932420 | 0.610455 | 0.974687 | 0.906987 | 0.842404 |
| 4 | 19 | 2.165723 | 0.917029 | 0.506740 | 0.941174 | 0.861231 | 0.824142 |
| 5 | 17 | 2.224462 | 0.931741 | 0.707620 | 0.937661 | 0.883706 | 0.878696 |
| 6 | 20 | 2.176839 | 0.925650 | 0.535587 | 0.971307 | 0.916803 | 0.831483 |
| 7 | 20 | 2.160563 | 0.949176 | 0.710169 | 0.960960 | 0.900239 | 0.896327 |
| 8 | 20 | 2.357307 | 0.938256 | 0.194328 | 0.995925 | 0.947849 | 0.731171 |
| 9 | 16 | 2.225171 | 0.943016 | 0.460464 | 0.987015 | 0.948749 | 0.809420 |
| 10 | 20 | 2.187304 | 0.936032 | 0.841556 | 0.883521 | 0.785394 | 0.925551 |

## 9. Checkpoints

Saved checkpoints include:
- Single-split training checkpoints (older workflow): `checkpoints/`
- CV fold-best checkpoints (current workflow): `checkpoints_cv/wdnn_fold_XX_best.pt`

Each fold checkpoint stores:
- fold index
- best epoch
- model state dict
- config
- best validation loss
- best macro metrics

## 10. How to Run

1. Open `modeltrain.ipynb`.
2. Run cells top-to-bottom.
3. Ensure the package `iterative-stratification` is installed for:
   - `from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit`

If missing, install it in the active environment and rerun the CV cell.

## 11. Notes and Caveats

- Current CV method is `MultilabelStratifiedShuffleSplit` with repeated random holdout splits (10 splits), not disjoint KFold partitions.
- Sensitivity varies substantially across folds in this run, indicating potential threshold/calibration effects and class imbalance behavior.
- For deployment, consider threshold tuning per drug and confidence interval reporting.
