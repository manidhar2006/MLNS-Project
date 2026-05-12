"""
amr_hgat.attention — Attention interpretation following Yang et al. (2021) bbab299.

Two levels of analysis, exactly as in the paper:
  1. Type-level (gene-level) attention  — which gene types does the model attend to?
     Paper: "Averaged type-level attention scores" (Figure 3, Figure S3).
  2. Node-level (SNP-level) attention   — which individual SNPs rank highest?
     Paper: "SNP ranking" (Table 3) and isolate-level demo (Figure 4, Figure S4).

For arm D only:
  3. Pathway-level attention per drug — which pathways does each drug route through?
     (Novel extension; arm D's per-drug masked softmax over pathways.)

Paper methodology:
  - Attention is extracted by running the trained model in eval mode.
  - For type-level: average α over all training isolates, per layer, per head.
  - For SNP ranking: average β over all isolates connected to each SNP.
  - The paper trained on combined train+test for interpretability figures.
  - We extend to second-line drugs by stratifying on drug resistance phenotype.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import HeteroData

from .model import HGATModel

# Second-line drugs in our 8-drug panel (non-first-line vs INH/RIF/EMB)
SECOND_LINE_DRUGS = {
    "ETH_BINARY_PHENOTYPE": "ETH",  # Ethionamide  — inhA, fabG1
    "KAN_BINARY_PHENOTYPE": "KAN",  # Kanamycin    — rrs, eis
    "AMI_BINARY_PHENOTYPE": "AMI",  # Amikacin     — rrs, eis
    "LEV_BINARY_PHENOTYPE": "LEV",  # Levofloxacin — gyrA, gyrB
    "MXF_BINARY_PHENOTYPE": "MXF",  # Moxifloxacin — gyrA, gyrB
}

# Known resistance genes per second-line drug (for validation of attention scores)
KNOWN_RESISTANCE_GENES = {
    "ETH": ["inhA", "fabG1", "ndh"],
    "KAN": ["rrs", "eis", "gidB", "tlyA", "rrl"],
    "AMI": ["rrs", "eis", "rrl"],
    "LEV": ["gyrA", "gyrB"],
    "MXF": ["gyrA", "gyrB"],
}


# ---------------------------------------------------------------------------
# Hook registration
# ---------------------------------------------------------------------------

class _AttentionCapture:
    """
    Replaces a GATConv's .forward method so it runs with
    return_attention_weights=True and stores both the per-edge alpha tensor
    and the GATConv output tensor (per destination node).

    This is necessary because modern PyG no longer sets GATConv._alpha
    after a forward pass; the only API is the return_attention_weights kwarg.

    We also capture the output tensor to compute gene contribution magnitude,
    which is not confounded by per-isolate SNP count (unlike raw α).
    Each GATConv normalises its softmax within its own edge type, so a gene
    with 1 SNP/isolate trivially gets α=1.0 regardless of importance. The
    output L2 norm ||Σ_j α_{j→i} W h_j|| measures actual influence on the
    isolate representation.
    """

    def __init__(
        self,
        conv,
        target_dict: dict,
        target_out: dict,
        ek: tuple,
    ):
        self.conv = conv
        self.target_dict = target_dict
        self.target_out = target_out
        self.ek = ek
        self._orig = conv.forward   # bound method of GATConv

    def __call__(self, x, edge_index, *args, **kwargs):
        # Force attention weights to be returned.
        kwargs.pop("return_attention_weights", None)
        result = self._orig(x, edge_index, *args,
                            return_attention_weights=True, **kwargs)
        if isinstance(result, tuple):
            out, (_, alpha) = result
            self.target_dict[self.ek] = alpha.detach().cpu()
            self.target_out[self.ek] = out.detach().cpu()
            return out
        return result   # fallback: no attention returned (should not happen)

    def restore(self) -> None:
        self.conv.forward = self._orig


def attach_attention_hooks(
    model: HGATModel,
) -> Tuple[list, Dict[str, Dict], Dict[str, Dict]]:
    """
    Monkey-patch every GATConv in layer1 and layer2 to capture attention.

    Returns:
        captures:    list of _AttentionCapture (call cap.restore() on each)
        storage:     {'layer1': {edge_type_tuple: alpha}, 'layer2': {...}}
                      alpha shape: [n_edges, n_heads]
        storage_out: {'layer1': {edge_type_tuple: out}, 'layer2': {...}}
                      out shape: [n_isolates, hidden_dim]
                      GATConv output contribution per destination node per edge type.
    """
    storage: Dict[str, Dict] = {"layer1": {}, "layer2": {}}
    storage_out: Dict[str, Dict] = {"layer1": {}, "layer2": {}}
    captures = []
    for layer_key, layer in [("layer1", model.layer1), ("layer2", model.layer2)]:
        for ek, conv in layer.conv.convs.items():
            cap = _AttentionCapture(conv, storage[layer_key], storage_out[layer_key], ek)
            conv.forward = cap
            captures.append(cap)
    return captures, storage, storage_out


# ---------------------------------------------------------------------------
# Type-level (gene-level) attention — mirrors paper's α_τ
# ---------------------------------------------------------------------------

def compute_type_level_attention(
    attn_storage: Dict[str, Dict],
    data: HeteroData,
    layer: str = "layer1",
    iso_mask: Optional[torch.BoolTensor] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute averaged type-level (gene) attention following Eq.(1-2) of the paper.

    For each gene type g, average the incoming attention weight from g's SNP nodes
    to isolate nodes, across all isolates (or a masked subset).

    Args:
        attn_storage: output of attach_attention_hooks, captured after forward pass.
        data:         HeteroData graph (same one used in forward pass).
        layer:        'layer1' or 'layer2'.
        iso_mask:     Boolean tensor [n_isolates]. If given, average only over
                      True isolates (e.g. drug-resistant ones).

    Returns:
        dict: gene_type -> np.ndarray [n_heads]
              Mean attention weight from that gene type to isolates.
    """
    layer_alphas = attn_storage.get(layer, {})
    n_isolates = data["isolate"].x.size(0)
    results: Dict[str, np.ndarray] = {}

    for ek, alpha in layer_alphas.items():
        src_type, rel_type, dst_type = ek
        # Only consider SNP-type→isolate edges (exclude pathway/ko/drug→isolate)
        if dst_type != "isolate":
            continue
        if src_type in ("pathway", "ko", "drug", "isolate"):
            continue

        edge_index = data[ek].edge_index          # [2, n_edges]
        dst_nodes = edge_index[1]                 # isolate indices [n_edges]
        n_heads = alpha.size(1) if alpha.dim() > 1 else 1
        if alpha.dim() == 1:
            alpha = alpha.unsqueeze(1)

        # Scatter-add alpha to each destination (isolate) node
        acc = torch.zeros(n_isolates, n_heads).scatter_add(
            0,
            dst_nodes.unsqueeze(1).expand(-1, n_heads),
            alpha,
        )
        cnt = torch.zeros(n_isolates).scatter_add(
            0, dst_nodes, torch.ones(dst_nodes.size(0))
        )
        # Mean attention per isolate (0 where isolate has no edges to this gene)
        safe_cnt = cnt.clamp(min=1).unsqueeze(1)
        mean_per_iso = acc / safe_cnt          # [n_isolates, n_heads]
        mean_per_iso[cnt == 0] = 0.0

        # Apply optional isolate mask
        if iso_mask is not None:
            subset = mean_per_iso[iso_mask]
            valid = (cnt[iso_mask] > 0)
            if valid.any():
                gene_mean = subset[valid].mean(dim=0).numpy()
            else:
                gene_mean = np.zeros(n_heads)
        else:
            valid = cnt > 0
            gene_mean = mean_per_iso[valid].mean(dim=0).numpy()

        results[src_type] = gene_mean

    return results


# ---------------------------------------------------------------------------
# Node-level (SNP-level) attention — mirrors paper's β_vv'
# ---------------------------------------------------------------------------

def compute_snp_attention(
    attn_storage: Dict[str, Dict],
    data: HeteroData,
    layer: str = "layer1",
    iso_mask: Optional[torch.BoolTensor] = None,
) -> Dict[str, Tuple[np.ndarray, List[str]]]:
    """
    Compute averaged node-level (SNP) attention following Eq.(3-4) of the paper.

    For each SNP node, average its attention weight across all (masked) isolates
    it is connected to.

    Args:
        attn_storage: hook storage dict.
        data:         HeteroData graph.
        layer:        'layer1' or 'layer2'.
        iso_mask:     Boolean [n_isolates]; average only over True isolates.

    Returns:
        dict: gene_type -> (mean_attn [n_snps, n_heads], snp_id_list)
    """
    layer_alphas = attn_storage.get(layer, {})
    results: Dict[str, Tuple[np.ndarray, List[str]]] = {}

    for ek, alpha in layer_alphas.items():
        src_type, rel_type, dst_type = ek
        if dst_type != "isolate":
            continue
        if src_type in ("pathway", "ko", "drug", "isolate"):
            continue

        edge_index = data[ek].edge_index   # [2, n_edges]
        src_nodes = edge_index[0]          # SNP local indices [n_edges]
        dst_nodes = edge_index[1]          # isolate indices [n_edges]
        n_snps = data[src_type].x.size(0)
        n_heads = alpha.size(1) if alpha.dim() > 1 else 1
        if alpha.dim() == 1:
            alpha = alpha.unsqueeze(1)

        # If iso_mask is set, zero out contributions from non-selected isolates
        if iso_mask is not None:
            valid_edges = iso_mask[dst_nodes]           # [n_edges] bool
            alpha_masked = alpha * valid_edges.float().unsqueeze(1)
            cnt_weight = valid_edges.float()
        else:
            alpha_masked = alpha
            cnt_weight = torch.ones(dst_nodes.size(0))

        # Per-SNP accumulation (scatter by source = SNP node)
        acc = torch.zeros(n_snps, n_heads).scatter_add(
            0,
            src_nodes.unsqueeze(1).expand(-1, n_heads),
            alpha_masked,
        )
        cnt = torch.zeros(n_snps).scatter_add(0, src_nodes, cnt_weight)
        safe_cnt = cnt.clamp(min=1).unsqueeze(1)
        mean_attn = (acc / safe_cnt).numpy()       # [n_snps, n_heads]
        mean_attn[cnt.numpy() == 0] = 0.0

        snp_ids: List[str] = getattr(
            data[src_type], "snp_ids",
            [f"{src_type}_snp_{i}" for i in range(n_snps)],
        )
        results[src_type] = (mean_attn, snp_ids)

    return results


# ---------------------------------------------------------------------------
# SNP ranking (Table 3 equivalent)
# ---------------------------------------------------------------------------

def rank_snps(
    snp_attention: Dict[str, Tuple[np.ndarray, List[str]]],
    head: int = 0,
    top_n: int = 10,
) -> List[Tuple[str, str, float]]:
    """
    Rank all SNPs by their mean attention weight for a given head.

    Returns:
        List of (snp_id, gene_type, mean_attn) sorted descending.
    """
    entries = []
    for gene_type, (mean_attn, snp_ids) in snp_attention.items():
        h = min(head, mean_attn.shape[1] - 1) if mean_attn.ndim > 1 else 0
        scores = mean_attn[:, h] if mean_attn.ndim > 1 else mean_attn
        for snp_id, score in zip(snp_ids, scores):
            entries.append((snp_id, gene_type, float(score)))
    entries.sort(key=lambda x: -x[2])
    return entries[:top_n]




# ---------------------------------------------------------------------------
# Gene-contribution magnitude — unconfounded gene importance metric
# ---------------------------------------------------------------------------

def compute_type_level_contribution(
    storage_out: Dict[str, Dict],
    data: HeteroData,
    layer: str = "layer1",
    iso_mask: Optional[torch.BoolTensor] = None,
) -> Dict[str, float]:
    """
    Compute the mean L2 norm of each gene type's GATConv output contribution
    to isolate representations.

    This metric is NOT confounded by per-isolate SNP count (unlike raw α):
      - For gene g, GATConv computes out_g(i) = Σ_j α_{j→i} W h_j for each isolate i.
      - ||out_g(i)||₂ measures how much gene g actually influences isolate i.
      - We average this over isolates that carry ≥1 SNP of gene g.

    Returns:
        dict: gene_type -> mean contribution magnitude (float)
              Sorted descending by contribution.
    """
    results: Dict[str, float] = {}

    for ek, out in storage_out.get(layer, {}).items():
        src_type, _, dst_type = ek
        if dst_type != "isolate":
            continue
        if src_type in ("pathway", "ko", "drug", "isolate"):
            continue

        # out: [n_isolates, hidden_dim] — the contribution of this gene type to each isolate
        edge_index = data[ek].edge_index       # [2, n_edges]
        connected = torch.unique(edge_index[1])  # isolates that have ≥1 edge of this type

        norms = out.norm(dim=-1)               # [n_isolates]

        if iso_mask is not None:
            # Only include isolates that are both connected and in the mask
            mask_vals = iso_mask[connected]
            idx = connected[mask_vals]
            if idx.numel() == 0:
                continue
            results[src_type] = float(norms[idx].mean().item())
        else:
            results[src_type] = float(norms[connected].mean().item())

    return dict(sorted(results.items(), key=lambda x: -x[1]))

# ---------------------------------------------------------------------------
# Pathway-level attention (arm D only)
# ---------------------------------------------------------------------------

def extract_pathway_attention(
    model: HGATModel,
    data: HeteroData,
    drug_cols: List[str],
    iso_mask: Optional[torch.BoolTensor] = None,
) -> Dict[str, Tuple[np.ndarray, List[str]]]:
    """
    Extract per-drug pathway attention from arm D's _pathway_context_drug.

    Requires model.enable_attention_store(True) BEFORE the forward pass,
    and that a forward pass has already been run.

    Returns:
        dict: drug_col -> (mean_alpha [n_pathways], pathway_ids)
              mean_alpha: averaged over (masked) isolates.
    """
    results: Dict[str, Tuple[np.ndarray, List[str]]] = {}
    pathway_ids: List[str] = getattr(
        data["pathway"], "pathway_ids",
        [f"pathway_{i}" for i in range(data["pathway"].x.size(0))],
    )

    for drug_col in drug_cols:
        alpha = model._pathway_attention_cache.get(drug_col)  # [n_isolates, n_pathways]
        if alpha is None:
            continue

        if iso_mask is not None:
            subset = alpha[iso_mask]
            if subset.size(0) == 0:
                mean_alpha = np.zeros(alpha.size(1))
            else:
                mean_alpha = subset.mean(dim=0).numpy()
        else:
            mean_alpha = alpha.mean(dim=0).numpy()

        results[drug_col] = (mean_alpha, pathway_ids)

    return results


# ---------------------------------------------------------------------------
# Full analysis run
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_attention_analysis(
    model: HGATModel,
    data: HeteroData,
    drug_cols: List[str],
    device: str = "cpu",
    top_n_snps: int = 10,
) -> Dict[str, Dict]:
    """
    Run a complete attention analysis for all requested drugs following
    the paper's two-level approach.

    For each drug, analysis is run twice:
      - over ALL isolates (paper's default)
      - over drug-RESISTANT isolates only (second-line drug focus)

    Returns nested dict:
        {
          drug_col: {
            'all': {
              'type_level': {layer: {gene: [n_heads] mean_attn}},
              'top_snps':   {layer: {head: [(snp_id, gene, score)]}},
              'pathway':    {pathway_id: mean_attn}  # arm D only
            },
            'resistant': { same structure }
          }
        }
    """
    model.eval()
    model.to(device)
    data = data.to(device)

    n_isolates = data["isolate"].x.size(0)
    y = data["isolate"].y.cpu()             # [n_isolates, n_drugs]
    drug_to_idx = {d: i for i, d in enumerate(drug_cols)}

    # Enable pathway attention storage for arm D
    is_arm_d = model.arm == "D"
    if is_arm_d:
        model.enable_attention_store(True)

    # Monkey-patch GATConv.forward to capture attention weights + outputs
    captures, storage, storage_out = attach_attention_hooks(model)

    # Single forward pass captures everything
    _ = model(data.x_dict, data.edge_index_dict)

    # Restore original GATConv forward methods
    for cap in captures:
        cap.restore()

    # Move data back to CPU for indexing
    data = data.cpu()

    results: Dict[str, Dict] = {}

    for drug_col in drug_cols:
        d_idx = drug_to_idx.get(drug_col)
        drug_results: Dict = {"all": {}, "resistant": {}}

        for subset_key, mask in [("all", None), ("resistant", None)]:
            if subset_key == "resistant" and d_idx is not None:
                r_mask = (y[:, d_idx] == 1.0)
                if r_mask.sum() == 0:
                    continue
                iso_mask: Optional[torch.BoolTensor] = r_mask
            else:
                iso_mask = None

            type_level: Dict[str, Dict] = {}
            top_snps: Dict[str, Dict] = {}
            per_gene_snps: Dict[str, Dict] = {}   # gene -> {layer -> {head -> ranked list}}

            for layer_key in ("layer1", "layer2"):
                gene_attn = compute_type_level_attention(
                    storage, data, layer=layer_key, iso_mask=iso_mask
                )
                type_level[layer_key] = {g: v.tolist() for g, v in gene_attn.items()}

                snp_attn = compute_snp_attention(
                    storage, data, layer=layer_key, iso_mask=iso_mask
                )
                n_heads = next(
                    (v[0].shape[1] for v in snp_attn.values() if v[0].ndim > 1),
                    1,
                )
                top_snps[layer_key] = {
                    f"head{h+1}": rank_snps(snp_attn, head=h, top_n=top_n_snps)
                    for h in range(n_heads)
                }
                # Within-gene rankings: all SNPs within each gene type, ranked by attention.
                # This is the biologically informative table (avoids cross-gene α confusion).
                for gene_type, (mean_attn, snp_ids) in snp_attn.items():
                    if gene_type not in per_gene_snps:
                        per_gene_snps[gene_type] = {}
                    if layer_key not in per_gene_snps[gene_type]:
                        per_gene_snps[gene_type][layer_key] = {}
                    for h in range(n_heads):
                        h_col = min(h, mean_attn.shape[1] - 1) if mean_attn.ndim > 1 else 0
                        scores = mean_attn[:, h_col] if mean_attn.ndim > 1 else mean_attn
                        ranked_gene = sorted(
                            zip(snp_ids, scores.tolist()),
                            key=lambda x: -x[1],
                        )
                        per_gene_snps[gene_type][layer_key][f"head{h+1}"] = ranked_gene

            # Gene contribution magnitudes (unconfounded gene importance)
            gene_contrib: Dict[str, Dict] = {}
            for layer_key in ("layer1", "layer2"):
                gene_contrib[layer_key] = compute_type_level_contribution(
                    storage_out, data, layer=layer_key, iso_mask=iso_mask
                )

            subset_result = {
                "type_level": type_level,
                "top_snps": top_snps,
                "per_gene_snps": per_gene_snps,
                "gene_contribution": gene_contrib,
            }

            # Pathway attention (arm D only)
            if is_arm_d and "pathway" in data.node_types:
                pwy_attn = extract_pathway_attention(
                    model, data, [drug_col], iso_mask=iso_mask
                )
                if drug_col in pwy_attn:
                    mean_alpha, pathway_ids = pwy_attn[drug_col]
                    ranked = sorted(
                        zip(pathway_ids, mean_alpha.tolist()),
                        key=lambda x: -x[1],
                    )
                    subset_result["pathway"] = ranked

            drug_results[subset_key] = subset_result

        results[drug_col] = drug_results

    if is_arm_d:
        model.enable_attention_store(False)

    return results


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def print_type_level_attention(
    results: Dict,
    drug_col: str,
    drug_short: str,
    subset: str = "resistant",
    layer: str = "layer1",
    known_genes: Optional[List[str]] = None,
) -> None:
    """Print type-level attention sorted by mean across heads, with known-gene markers."""
    drug_data = results.get(drug_col, {}).get(subset, {})
    gene_attn = drug_data.get("type_level", {}).get(layer, {})
    if not gene_attn:
        print(f"  [no data for {drug_short} {subset} {layer}]")
        return

    # Sort by mean across heads
    sorted_genes = sorted(
        gene_attn.items(), key=lambda x: -float(np.mean(x[1]))
    )

    print(f"\n  {drug_short} | {subset} isolates | {layer} | type-level attention")
    print(f"  {'Gene':<12}  {'H1':>8}  {'H2':>8}  {'Mean':>8}  Note")
    print(f"  {'-'*12}  {'------':>8}  {'------':>8}  {'------':>8}")
    for gene, attn in sorted_genes:
        vals = attn if isinstance(attn, list) else [float(attn)]
        mean_v = float(np.mean(vals))
        h1 = f"{vals[0]:.4f}" if len(vals) > 0 else "—"
        h2 = f"{vals[1]:.4f}" if len(vals) > 1 else "—"
        note = " *known*" if known_genes and gene in known_genes else ""
        print(f"  {gene:<12}  {h1:>8}  {h2:>8}  {mean_v:>8.4f}{note}")


def print_snp_ranking(
    results: Dict,
    drug_col: str,
    drug_short: str,
    subset: str = "resistant",
    layer: str = "layer1",
    head: int = 0,
) -> None:
    """Print top-N SNP ranking for a drug/layer/head combination."""
    drug_data = results.get(drug_col, {}).get(subset, {})
    head_key = f"head{head + 1}"
    ranked = drug_data.get("top_snps", {}).get(layer, {}).get(head_key, [])
    if not ranked:
        print(f"  [no SNP data for {drug_short} {subset} {layer} {head_key}]")
        return

    print(f"\n  {drug_short} | {subset} | {layer} {head_key} | top-{len(ranked)} SNPs")
    print(f"  {'Rank':<5}  {'SNP':<30}  {'Gene':<10}  {'Score':>8}")
    print(f"  {'-'*5}  {'-'*30}  {'-'*10}  {'------':>8}")
    for rank, (snp_id, gene, score) in enumerate(ranked, 1):
        print(f"  {rank:<5}  {snp_id:<30}  {gene:<10}  {score:>8.4f}")


def print_pathway_attention(
    results: Dict,
    drug_col: str,
    drug_short: str,
    subset: str = "resistant",
    top_n: int = 10,
) -> None:
    """Print ranked pathway attention for a drug (arm D only)."""
    drug_data = results.get(drug_col, {}).get(subset, {})
    ranked = drug_data.get("pathway", [])
    if not ranked:
        print(f"  [no pathway data for {drug_short} {subset}]")
        return

    print(f"\n  {drug_short} | {subset} | pathway attention (arm D)")
    print(f"  {'Pathway':<20}  {'Attention':>10}")
    print(f"  {'-'*20}  {'----------':>10}")
    for pwy_id, score in ranked[:top_n]:
        print(f"  {pwy_id:<20}  {score:>10.4f}")
