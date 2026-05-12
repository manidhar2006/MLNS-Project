"""
amr_hgat.model — HGATModel: single configurable class for all ablation arms.

Key differences from the original notebooks:
  - No copy.deepcopy(x_dict) in forward (was wasting memory every step)
  - No unused type_attn_raw parameter
  - Layer 2 can selectively exclude edge types (for pathway->isolate hub cycle)
  - Arm D uses per-drug masked pathway attention instead of global softmax
  - Hidden=128 throughout (original KEGG notebooks used 64)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HeteroConv


class HGATLayer(nn.Module):
    """Single heterogeneous GAT layer over specified edge types."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_types: List[Tuple[str, str, str]],
        heads: int = 2,
        dropout: float = 0.3,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        assert out_channels % heads == 0, "out_channels must be divisible by heads"
        self.edge_types = edge_types
        self.dropout_p = dropout

        convs = {
            (s, r, d): GATConv(
                in_channels=(-1, -1),
                out_channels=out_channels // heads,
                heads=heads,
                dropout=dropout,
                negative_slope=negative_slope,
                add_self_loops=False,
                concat=True,
            )
            for s, r, d in edge_types
        }
        self.conv = HeteroConv(convs, aggr="sum")

        dst_types = sorted({d for _, _, d in edge_types})
        self.norms = nn.ModuleDict({d: nn.LayerNorm(out_channels) for d in dst_types})
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple, torch.Tensor],
        exclude_edge_types: Optional[List[Tuple]] = None,
    ) -> Dict[str, torch.Tensor]:
        # Optionally mask certain edge types from this layer's forward
        if exclude_edge_types:
            active_edge_index = {
                k: v for k, v in edge_index_dict.items()
                if k not in set(map(tuple, exclude_edge_types))
            }
        else:
            active_edge_index = edge_index_dict

        raw_out = self.conv(x_dict, active_edge_index)
        out = {}
        for d, h in raw_out.items():
            if h is None:
                continue
            h = self.norms[d](h)
            h = F.elu(h)
            h = self.drop(h)
            out[d] = h
        return out


class HGATModel(nn.Module):
    """
    Configurable HGAT for TB AMR prediction.

    Works for all ablation arms A–D.
    The forward pass implements:
      - Layer 1: full heterogeneous message passing
      - Layer 2: same, but with `layer2_exclude_edge_types` masked out
      - Isolate representations: iso_l1, iso_l2 from both layers
      - Pathway context (arms B/C): per-drug masked attention (arm D) or global softmax
      - Classifier: Linear([iso_l1, iso_l2, pathway_ctx] or [iso_l1, iso_l2])
    """

    def __init__(
        self,
        arm: str,
        edge_types: List[Tuple[str, str, str]],
        hidden_channels: int = 128,
        num_drugs: int = 8,
        heads: int = 2,
        dropout: float = 0.3,
        layer2_exclude_edge_types: Optional[List[Tuple]] = None,
        drug_pathway_masks: Optional[Dict[str, np.ndarray]] = None,
        drug_cols: Optional[List[str]] = None,
    ):
        """
        Args:
            arm:                        "A", "B", "C", or "D".
            edge_types:                 All edge types in the graph.
            hidden_channels:            Hidden/output dimensionality.
            num_drugs:                  Number of output drug heads.
            heads:                      Number of attention heads.
            dropout:                    Dropout rate.
            layer2_exclude_edge_types:  Edge types to mask in layer 2
                                        (typically [("pathway","has_isolate","isolate")]).
            drug_pathway_masks:         Dict drug_col -> bool array[n_pathways].
                                        Required for arm D.
            drug_cols:                  Ordered list of drug column names (arm D).
        """
        super().__init__()
        self.arm = arm.upper()
        self.hidden_channels = hidden_channels
        self.num_drugs = num_drugs
        self.layer2_exclude = layer2_exclude_edge_types or []
        self.drug_cols = drug_cols or []
        # Pathway attention cache (populated when analyse() context is active)
        self._store_pathway_attention: bool = False
        self._pathway_attention_cache: Dict[str, torch.Tensor] = {}

        self.layer1 = HGATLayer(
            in_channels=-1,
            out_channels=hidden_channels,
            edge_types=edge_types,
            heads=heads,
            dropout=dropout,
        )
        self.layer2 = HGATLayer(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            edge_types=edge_types,
            heads=heads,
            dropout=dropout,
        )

        # Pathway context branch (arms B/C/D)
        uses_pathway = self.arm in ("B", "C", "D")
        clf_in = hidden_channels * 3 if uses_pathway else hidden_channels * 2

        if uses_pathway:
            self.pathway_ctx = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, hidden_channels),
            )

        # Arm D: per-drug masked pathway attention
        if self.arm == "D" and drug_pathway_masks is not None:
            # Register masks as buffers (not parameters)
            for col, mask in drug_pathway_masks.items():
                buf_name = "mask_" + col.replace("_BINARY_PHENOTYPE", "").replace("-", "_")
                self.register_buffer(buf_name, torch.from_numpy(mask.astype(np.float32)))
            self._drug_pathway_masks = drug_pathway_masks
            # Per-drug classifiers
            self.drug_classifiers = nn.ModuleDict({
                col.replace("_BINARY_PHENOTYPE", "").replace("-", "_"): nn.Sequential(
                    nn.Linear(hidden_channels * 3, hidden_channels),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_channels, 1),
                )
                for col in self.drug_cols
            })
        else:
            self._drug_pathway_masks = None
            self.classifier = nn.Sequential(
                nn.Linear(clf_in, hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, num_drugs),
            )

    def _pathway_context_global(
        self, iso_l2: torch.Tensor, h2: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Global softmax pathway context (arms B/C)."""
        p = h2.get("pathway")
        if p is None or p.size(0) == 0:
            return torch.zeros_like(iso_l2)
        scores = torch.matmul(iso_l2, p.T) / (p.shape[1] ** 0.5)
        alpha = torch.softmax(scores, dim=1)
        p_ctx = alpha @ p
        return self.pathway_ctx(p_ctx)

    def _pathway_context_drug(
        self,
        iso_l2: torch.Tensor,
        h2: Dict[str, torch.Tensor],
        drug_col: str,
    ) -> torch.Tensor:
        """Per-drug masked pathway attention (arm D)."""
        p = h2.get("pathway")
        if p is None or p.size(0) == 0:
            return torch.zeros_like(iso_l2)

        short = drug_col.replace("_BINARY_PHENOTYPE", "").replace("-", "_")
        # Retrieve mask buffer
        buf_name = "mask_" + short
        mask = getattr(self, buf_name, None)  # [n_pathways]
        if mask is None:
            mask = torch.ones(p.size(0), dtype=torch.float32, device=p.device)

        # Dot-product attention: iso_l2 query against pathway embeddings as keys.
        # Per-drug differentiation comes from the drug-specific mask applied below.
        # Using drug_attn_heads (Linear→1) was incorrect: it scored every pathway
        # with the same scalar, producing uniform softmax within the masked set.
        scores = torch.matmul(iso_l2, p.T) / (p.shape[1] ** 0.5)  # [n_iso, n_pathways]

        # Mask: set non-drug pathways to -inf before softmax
        mask_expanded = mask.unsqueeze(0).expand_as(scores)
        scores = scores.masked_fill(mask_expanded == 0, float("-inf"))
        # Handle case where all pathways are masked out for a drug
        all_inf = (scores == float("-inf")).all(dim=1, keepdim=True)
        scores = torch.where(all_inf, torch.zeros_like(scores), scores)

        alpha = torch.softmax(scores, dim=1)  # [n_isolates, n_pathways]

        # Store pathway attention when in analysis mode
        if self._store_pathway_attention:
            self._pathway_attention_cache[drug_col] = alpha.detach().cpu()

        p_ctx = alpha @ p
        return self.pathway_ctx(p_ctx)

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple, torch.Tensor],
    ) -> torch.Tensor:
        # Layer 1: all edges
        h1 = self.layer1(x_dict, edge_index_dict)
        # Layer 2: exclude hub-cycle edges
        h2 = self.layer2(
            h1, edge_index_dict,
            exclude_edge_types=self.layer2_exclude if self.layer2_exclude else None,
        )

        # Isolate representations from both layers
        iso_l1 = h1.get("isolate")
        if iso_l1 is None:
            iso_l1 = torch.zeros(
                x_dict["isolate"].size(0), self.hidden_channels,
                device=x_dict["isolate"].device
            )
        iso_l2 = h2.get("isolate", iso_l1)

        if self.arm == "A":
            combined = torch.cat([iso_l1, iso_l2], dim=-1)
            return self.classifier(combined)

        if self.arm in ("B", "C"):
            p_ctx = self._pathway_context_global(iso_l2, h2)
            combined = torch.cat([iso_l1, iso_l2, p_ctx], dim=-1)
            return self.classifier(combined)

        # Arm D: per-drug logits assembled into [n_iso, n_drugs]
        logit_list = []
        for drug_col in self.drug_cols:
            p_ctx = self._pathway_context_drug(iso_l2, h2, drug_col)
            combined = torch.cat([iso_l1, iso_l2, p_ctx], dim=-1)
            short = drug_col.replace("_BINARY_PHENOTYPE", "").replace("-", "_")
            clf = self.drug_classifiers[short]
            logit_list.append(clf(combined))   # [n_iso, 1]
        return torch.cat(logit_list, dim=-1)   # [n_iso, n_drugs]

    def enable_attention_store(self, enable: bool = True) -> None:
        """Turn pathway attention caching on/off (arm D only)."""
        self._store_pathway_attention = enable
        if enable:
            self._pathway_attention_cache = {}

    def count_parameters(self) -> int:
        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
            and not isinstance(p, torch.nn.parameter.UninitializedParameter)
        )

    @torch.no_grad()
    def get_gene_attention(self) -> List[Dict[str, float]]:
        """Return type-level attention weight sums per gene per layer."""
        result = []
        for layer in (self.layer1, self.layer2):
            gene_attn: Dict[str, float] = {}
            for (s, r, d), conv in layer.conv.convs.items():
                # Approximate layer-level importance via L2 norm of query weights
                try:
                    w = conv.lin_src.weight if hasattr(conv, "lin_src") else None
                    importance = float(w.norm().item()) if w is not None else 1.0
                except Exception:
                    importance = 1.0
                gene = r.replace("has_", "").replace("in_", "")
                gene_attn[gene] = gene_attn.get(gene, 0.0) + importance
            # Normalize
            total = max(sum(gene_attn.values()), 1e-12)
            result.append({k: v / total for k, v in sorted(gene_attn.items(), key=lambda x: -x[1])})
        return result


def build_model(
    arm: str,
    data,
    drug_cols: List[str],
    hidden_channels: int = 128,
    heads: int = 2,
    dropout: float = 0.3,
) -> HGATModel:
    """Instantiate HGATModel from a HeteroData graph and arm string."""
    edge_types = list(data.edge_index_dict.keys())
    meta = getattr(data, "meta", {})
    layer2_exclude = meta.get("layer2_exclude_edge_types", [])
    drug_pathway_masks = meta.get("drug_pathway_masks", None)

    model = HGATModel(
        arm=arm,
        edge_types=edge_types,
        hidden_channels=hidden_channels,
        num_drugs=len(drug_cols),
        heads=heads,
        dropout=dropout,
        layer2_exclude_edge_types=layer2_exclude,
        drug_pathway_masks=drug_pathway_masks,
        drug_cols=drug_cols,
    )
    print(f"[model arm={arm}] {model.count_parameters():,} parameters, "
          f"{len(edge_types)} edge types, "
          f"layer2_exclude={[r for _,r,_ in layer2_exclude]}")
    return model
