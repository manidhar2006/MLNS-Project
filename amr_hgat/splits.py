"""
amr_hgat.splits — Multi-label stratified k-fold splits for the ablation.

Uses iterative multi-label stratification so that all drug phenotype
distributions (not just RIF) are balanced across folds.

All four arms use identical fold indices so metrics are directly comparable.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


def make_folds(
    phenotype_df: pd.DataFrame,
    isolates: List[str],
    drug_cols: List[str],
    n_splits: int = 5,
    seed: int = 42,
    primary_drug: Optional[str] = None,
) -> List[Dict[str, np.ndarray]]:
    """
    Build n_splits outer folds with multi-label stratification across all drugs.
    Each fold provides:
      train_idx, val_idx (inner ~25% of remaining), test_idx
    All indices are integer positions in the `isolates` list.

    Args:
        phenotype_df: DataFrame with isolate_id + drug label columns.
        isolates:     Ordered list of isolate IDs matching graph node order.
        drug_cols:    Drug phenotype columns.
        n_splits:     Number of outer CV folds.
        seed:         Random seed.
        primary_drug: Unused, kept for backward compatibility.

    Returns:
        List of dicts, one per fold, each with numpy arrays:
            train_idx, val_idx, test_idx
    """
    pheno_indexed = phenotype_df.set_index("isolate_id").reindex(isolates)
    y_multi = pheno_indexed[drug_cols].values.astype(float)

    # Rows where ALL drug labels are NaN cannot be stratified
    any_valid = ~np.isnan(y_multi).all(axis=1)
    valid_idx = np.where(any_valid)[0]
    invalid_idx = np.where(~any_valid)[0]

    # Replace remaining per-cell NaNs with 0 for stratification
    # (iterstrat needs finite values; NaN-masked cells will be ignored in loss)
    y_strat = np.nan_to_num(y_multi[valid_idx], nan=0.0).astype(int)

    if len(valid_idx) < n_splits * 2:
        raise RuntimeError(
            f"Not enough labeled samples ({len(valid_idx)}) for {n_splits}-fold CV."
        )

    outer_skf = MultilabelStratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=seed,
    )
    folds = []

    for fold_i, (rem_local, test_local) in enumerate(
        outer_skf.split(valid_idx, y_strat)
    ):
        rem_idx = valid_idx[rem_local]
        test_idx = valid_idx[test_local]

        y_rem = np.nan_to_num(y_multi[rem_idx], nan=0.0).astype(int)
        inner_skf = MultilabelStratifiedKFold(
            n_splits=4, shuffle=True, random_state=seed + fold_i + 1,
        )
        inner_splits = list(inner_skf.split(rem_idx, y_rem))
        train_local, val_local = inner_splits[0]
        train_idx = rem_idx[train_local]
        val_idx = rem_idx[val_local]

        # Samples with all-NaN labels go into training only
        if len(invalid_idx) > 0:
            train_idx = np.unique(np.concatenate([train_idx, invalid_idx]))

        folds.append({
            "train_idx": train_idx,
            "val_idx":   val_idx,
            "test_idx":  test_idx,
            "strat_col": "multilabel",
            "fold":      fold_i,
        })

    return folds
