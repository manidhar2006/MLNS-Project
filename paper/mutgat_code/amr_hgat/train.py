"""
amr_hgat.train — Matched training loop for all ablation arms.

Protocol (identical for A/B/C/D):
  - 180 epochs max
  - Early stopping on mean val AUROC, patience=20
  - Adam LR=5e-4, weight_decay=1e-4
  - No l2_lambda in loss (removed double-L2)
  - hidden=128, heads=2, dropout=0.3
  - Thresholds tuned on val fold only
  - Per-drug class-imbalance pos_weight clamped to [1, 10]
  - PMI+SVD embeddings recomputed per fold (training isolates only)
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from .data import compute_pmi_embeddings, append_effect_class_features
from .graph_builders import build_graph
from .metrics import (
    aggregate_folds,
    compute_metrics,
    find_best_thresholds,
    mean_auroc,
)
from .model import HGATModel, build_model
from .splits import make_folds

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _remap_iso_indices(
    full_isolates: List[str],
    graph_isolate_ids: List[str],
    idx: np.ndarray,
) -> np.ndarray:
    """
    Map fold indices defined on `full_isolates` to row indices in the current
    graph (which may omit isolates with no SNPs in the per-fold vocabulary).
    """
    gmap = {sid: j for j, sid in enumerate(graph_isolate_ids)}
    flat = np.asarray(idx).ravel().astype(np.int64)
    out: List[int] = []
    for i in flat:
        sid = full_isolates[int(i)]
        if sid in gmap:
            out.append(gmap[sid])
    return np.array(out, dtype=np.int64)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def masked_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Binary cross-entropy that ignores NaN label entries.
    No l2_lambda term (weight_decay in Adam handles regularization).
    """
    mask = ~torch.isnan(labels)
    if mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True, device=logits.device)

    if pos_weight is not None:
        weight_mat = torch.ones_like(labels)
        pw = pos_weight.unsqueeze(0).expand_as(labels)
        weight_mat = torch.where(labels == 1.0, pw, weight_mat)
        loss = F.binary_cross_entropy_with_logits(
            logits[mask], labels[mask], weight=weight_mat[mask], reduction="mean"
        )
    else:
        loss = F.binary_cross_entropy_with_logits(
            logits[mask], labels[mask], reduction="mean"
        )
    return loss


def _pos_weights(labels: torch.Tensor) -> torch.Tensor:
    pos = (labels == 1.0).sum(dim=0).float()
    neg = (labels == 0.0).sum(dim=0).float()
    return (neg / (pos + 1e-6)).clamp(min=1.0, max=10.0)


# ---------------------------------------------------------------------------
# Single fold training
# ---------------------------------------------------------------------------

def train_one_fold(
    data: HeteroData,
    model: HGATModel,
    drug_cols: List[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    epochs: int = 180,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    patience: int = 20,
    device: str = DEVICE,
    fold_num: int = 0,
    verbose: bool = True,
) -> Dict:
    """
    Train one CV fold. Returns metrics dict for the test fold.
    """
    model = copy.deepcopy(model).to(device)
    data = data.to(device)

    train_t = torch.tensor(train_idx, dtype=torch.long, device=device)
    val_t = torch.tensor(val_idx, dtype=torch.long, device=device)
    test_t = torch.tensor(test_idx, dtype=torch.long, device=device)

    y_train = data["isolate"].y[train_t]
    pw = _pos_weights(y_train).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=patience // 2, factor=0.5, min_lr=1e-6
    )

    best_val_auroc = -1.0
    best_state = None
    best_thresholds: Dict[str, float] = {d: 0.5 for d in drug_cols}
    no_improve = 0

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x_dict, data.edge_index_dict)
        loss = masked_bce_loss(logits[train_t], data["isolate"].y[train_t], pw)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(data.x_dict, data.edge_index_dict)
            val_probs = torch.sigmoid(val_logits[val_t]).cpu().numpy()
            val_labels = data["isolate"].y[val_t].cpu().numpy()

        val_metrics = compute_metrics(val_probs, val_labels, drug_cols)
        mauc = mean_auroc(val_metrics)
        scheduler.step(mauc)

        if mauc > best_val_auroc:
            best_val_auroc = mauc
            best_state = copy.deepcopy(model.state_dict())
            # Tune thresholds on val (never test)
            best_thresholds = find_best_thresholds(val_probs, val_labels, drug_cols)
            no_improve = 0
        else:
            no_improve += 1

        if verbose and epoch % 30 == 0:
            elapsed = time.time() - t0
            print(
                f"  fold={fold_num} ep={epoch:03d} loss={loss.item():.4f} "
                f"val_AUROC={mauc:.4f} best={best_val_auroc:.4f} "
                f"[{elapsed:.0f}s]"
            )

        if no_improve >= patience:
            if verbose:
                print(f"  fold={fold_num} early stop at epoch {epoch}")
            break

    # Load best checkpoint
    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        test_logits = model(data.x_dict, data.edge_index_dict)
        test_probs = torch.sigmoid(test_logits[test_t]).cpu().numpy()
        test_labels = data["isolate"].y[test_t].cpu().numpy()

    test_metrics = compute_metrics(test_probs, test_labels, drug_cols, thresholds=best_thresholds)
    test_metrics["_meta"] = {
        "fold": fold_num,
        "best_val_auroc": best_val_auroc,
        "val_thresholds": best_thresholds,
    }
    return test_metrics


# ---------------------------------------------------------------------------
# Full cross-validation
# ---------------------------------------------------------------------------

def cross_validate(
    arm: str,
    data: HeteroData,
    drug_cols: List[str],
    phenotype_df,
    isolates: List[str],
    n_splits: int = 5,
    epochs: int = 180,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    patience: int = 20,
    hidden_channels: int = 128,
    heads: int = 2,
    dropout: float = 0.3,
    seed: int = 42,
    device: str = DEVICE,
    output_dir: Optional[str] = None,
    verbose: bool = True,
    isolate_snp_df=None,
    embed_dim: int = 64,
    min_snp_count: int = 5,
    kegg_json_path: Optional[str] = None,
    effect_map: Optional[Dict] = None,
) -> Dict:
    """
    Run n_splits-fold CV for the given arm. Returns aggregated metrics.
    Optionally writes per-fold JSON to output_dir.

    When isolate_snp_df is provided, PMI+SVD embeddings are recomputed
    per fold using only training isolates (prevents data leakage).
    Falls back to the pre-built graph when isolate_snp_df is None.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    perfold_rebuild = isolate_snp_df is not None

    folds = make_folds(
        phenotype_df=phenotype_df,
        isolates=isolates,
        drug_cols=drug_cols,
        n_splits=n_splits,
        seed=seed,
    )

    if not perfold_rebuild:
        model = build_model(
            arm=arm, data=data, drug_cols=drug_cols,
            hidden_channels=hidden_channels, heads=heads, dropout=dropout,
        )

    fold_results = []
    for fold_info in folds:
        fold_i = fold_info["fold"]
        if verbose:
            print(f"\n=== Fold {fold_i + 1}/{n_splits} ===")

        if perfold_rebuild:
            train_isolate_ids = [isolates[i] for i in fold_info["train_idx"]]
            if verbose:
                print(f"  Recomputing PMI from {len(train_isolate_ids)} train isolates ...")
            fold_emb, fold_snps = compute_pmi_embeddings(
                isolate_snp_df, embed_dim=embed_dim, min_count=min_snp_count,
                train_isolates=train_isolate_ids,
            )
            if effect_map is not None:
                fold_emb = append_effect_class_features(fold_emb, fold_snps, effect_map)

            fold_data = build_graph(
                arm=arm,
                isolate_snp_df=isolate_snp_df,
                phenotype_df=phenotype_df,
                snp_embeddings=fold_emb,
                snp_list=fold_snps,
                drug_cols=drug_cols,
                kegg_json_path=kegg_json_path if arm != "A" else None,
            )
            fold_model = build_model(
                arm=arm, data=fold_data, drug_cols=drug_cols,
                hidden_channels=hidden_channels, heads=heads, dropout=dropout,
            )
            graph_ids = list(fold_data["isolate"].isolate_ids)
            train_idx = _remap_iso_indices(isolates, graph_ids, fold_info["train_idx"])
            val_idx = _remap_iso_indices(isolates, graph_ids, fold_info["val_idx"])
            test_idx = _remap_iso_indices(isolates, graph_ids, fold_info["test_idx"])
            if verbose and len(graph_ids) < len(isolates):
                print(
                    f"  [{fold_i + 1}] graph uses {len(graph_ids)}/{len(isolates)} isolates "
                    f"(SNPs outside fold train vocabulary)"
                )
        else:
            fold_data = data
            fold_model = model
            train_idx = fold_info["train_idx"]
            val_idx = fold_info["val_idx"]
            test_idx = fold_info["test_idx"]

        fold_metrics = train_one_fold(
            data=fold_data,
            model=fold_model,
            drug_cols=drug_cols,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
            device=device,
            fold_num=fold_i,
            verbose=verbose,
        )
        fold_results.append(fold_metrics)

        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            path = out / f"fold_{fold_i}.json"
            serializable = {k: v for k, v in fold_metrics.items() if k != "_meta"}
            meta = fold_metrics.get("_meta", {})
            serializable["_meta"] = {
                k: (float(v) if isinstance(v, (np.floating, float)) else v)
                for k, v in meta.items()
                if k != "val_thresholds"
            }
            with open(path, "w") as f:
                json.dump(serializable, f, indent=2)

    clean_folds = [{k: v for k, v in fr.items() if k != "_meta"} for fr in fold_results]
    summary = aggregate_folds(clean_folds)

    if verbose:
        print(f"\n=== CV Summary (arm={arm}) ===")
        for drug, mmap in summary.items():
            auroc_m = mmap.get("AUROC", {}).get("mean", 0.0)
            auroc_s = mmap.get("AUROC", {}).get("std", 0.0)
            print(f"  {drug:<25} AUROC={auroc_m:.4f}±{auroc_s:.4f}")

    return summary
