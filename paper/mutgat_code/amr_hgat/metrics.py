"""
amr_hgat.metrics — Evaluation metrics for AMR prediction.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def find_best_thresholds(
    probs: np.ndarray,
    labels: np.ndarray,
    drug_cols: List[str],
    grid: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Per-drug threshold search to maximise F1 on a held-out set.
    Uses a coarse grid; should only be applied on val, never on test.
    """
    if grid is None:
        grid = np.linspace(0.1, 0.9, 17)

    thresholds: Dict[str, float] = {}
    for i, drug in enumerate(drug_cols):
        valid = ~np.isnan(labels[:, i])
        if valid.sum() < 2 or len(np.unique(labels[valid, i])) < 2:
            thresholds[drug] = 0.5
            continue
        yt = labels[valid, i].astype(int)
        yp = probs[valid, i]
        best_t, best_f1 = 0.5, -1.0
        for t in grid:
            yb = (yp >= t).astype(int)
            f = f1_score(yt, yb, zero_division=0)
            if f > best_f1:
                best_f1, best_t = f, float(t)
        thresholds[drug] = best_t
    return thresholds


def compute_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    drug_cols: List[str],
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-drug AUROC, AUPRC, Sensitivity, Specificity, F1, Threshold.
    Missing labels (NaN) are excluded from each drug's calculation.
    """
    results: Dict[str, Dict[str, float]] = {}
    for i, drug in enumerate(drug_cols):
        valid = ~np.isnan(labels[:, i])
        if valid.sum() < 2 or len(np.unique(labels[valid, i])) < 2:
            continue
        yt = labels[valid, i].astype(int)
        yp = probs[valid, i]
        th = 0.5 if thresholds is None else thresholds.get(drug, 0.5)
        yb = (yp >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(yt, yb, labels=[0, 1]).ravel()
        results[drug] = {
            "AUROC":       float(roc_auc_score(yt, yp)),
            "AUPRC":       float(average_precision_score(yt, yp)),
            "Sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
            "Specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
            "F1":          float(f1_score(yt, yb, zero_division=0)),
            "Threshold":   th,
            "n_R":         int((yt == 1).sum()),
            "n_S":         int((yt == 0).sum()),
        }
    return results


def mean_auroc(metrics: Dict[str, Dict[str, float]]) -> float:
    """Return mean AUROC across drugs in a metrics dict."""
    aurocs = [v["AUROC"] for v in metrics.values() if "AUROC" in v]
    return float(np.mean(aurocs)) if aurocs else 0.0


def aggregate_folds(
    fold_results: List[Dict[str, Dict[str, float]]]
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Aggregate per-fold metrics into mean ± std.
    Returns {drug: {metric: {"mean": ..., "std": ...}}}.
    """
    all_drugs = sorted({d for fr in fold_results for d in fr})
    out = {}
    for drug in all_drugs:
        per_fold = [fr[drug] for fr in fold_results if drug in fr]
        if not per_fold:
            continue
        metrics = list(per_fold[0].keys())
        out[drug] = {}
        for m in metrics:
            vals = [f[m] for f in per_fold]
            out[drug][m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return out
