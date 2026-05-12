"""
amr_hgat.graph_builders — Heterogeneous graph construction for ablation arms A–D.

Arms:
  A  SNP-only baseline (isolate + per-gene SNP nodes)
  B  Pathway-minimal  (+ pathway nodes, hub-pruned, +mtu00074,
                        gene→pathway edges, gene-membership pathway features,
                        layer-2 pathway->isolate edge masked)
  C  Pathway + KO     (+ KO nodes bridging pathway-orphan SNP gene types,
                        pathway-membership KO features)
  D  Pathway + KO + Drug (+ drug nodes with per-drug pathway masks)

All arms build from the same base graph via _build_base_graph().
The layer-2 masking for arm B/C/D is encoded as metadata in the returned
HeteroData via data.meta["layer2_exclude_edge_types"].
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

# Hub pathways that provide no resistance-specific signal and cause over-smoothing.
HUB_PATHWAYS: Set[str] = {"mtu01100", "mtu01110"}

# Mechanistically relevant pathways to INCLUDE even if not in original KG.
INCLUDE_PATHWAYS: Set[str] = {"mtu00074"}  # Mycolic acid biosynthesis (INH)

# Drug short-names to match against drug_cols
DRUG_SHORT_REVERSE = {
    "INH": "INH_BINARY_PHENOTYPE",
    "RIF": "RIF_BINARY_PHENOTYPE",
    "EMB": "EMB_BINARY_PHENOTYPE",
    "LEV": "LEV_BINARY_PHENOTYPE",
    "MXF": "MXF_BINARY_PHENOTYPE",
    "ETH": "ETH_BINARY_PHENOTYPE",
    "KAN": "KAN_BINARY_PHENOTYPE",
    "AMK": "AMI_BINARY_PHENOTYPE",
    "CFZ": "CFZ_BINARY_PHENOTYPE",
    "LZD": "LZD_BINARY_PHENOTYPE",
    "BDQ": "BDQ_BINARY_PHENOTYPE",
    "DLM": "DLM_BINARY_PHENOTYPE",
}
DRUG_COL_TO_SHORT = {v: k for k, v in DRUG_SHORT_REVERSE.items()}


# ---------------------------------------------------------------------------
# Base graph (shared by all arms)
# ---------------------------------------------------------------------------

def _build_base_graph(
    isolate_snp_df: pd.DataFrame,
    phenotype_df: pd.DataFrame,
    snp_embeddings: np.ndarray,
    snp_list: List[str],
    drug_cols: List[str],
) -> HeteroData:
    """
    Build isolate + per-gene SNP node types with bidirectional edges.
    This is arm A and the foundation for arms B/C/D.
    """
    snp_emb_map = {snp_id: i for i, snp_id in enumerate(snp_list)}
    df = isolate_snp_df[isolate_snp_df["snp_id"].isin(snp_emb_map)].copy()

    isolates = df["isolate_id"].unique().tolist()
    iso_to_idx = {sid: i for i, sid in enumerate(isolates)}
    genes = sorted(df["gene"].unique().tolist())
    data = HeteroData()

    # Isolate features: max-pool over PMI+SVD SNP embeddings
    iso_feats = np.zeros((len(isolates), snp_embeddings.shape[1]), dtype=np.float32)
    for iso in isolates:
        idxs = [
            snp_emb_map[s]
            for s in df[df["isolate_id"] == iso]["snp_id"]
            if s in snp_emb_map
        ]
        if idxs:
            iso_feats[iso_to_idx[iso]] = snp_embeddings[idxs].max(axis=0)

    data["isolate"].x = torch.from_numpy(iso_feats)
    phen = phenotype_df.set_index("isolate_id").reindex(isolates)[drug_cols]
    data["isolate"].y = torch.tensor(phen.values.astype(np.float32))
    data["isolate"].isolate_ids = isolates

    for gene in genes:
        gdf = df[df["gene"] == gene]
        gene_snps = gdf["snp_id"].unique().tolist()
        snp_local = {s: i for i, s in enumerate(gene_snps)}
        emb_idxs = [snp_emb_map[s] for s in gene_snps]

        data[gene].x = torch.from_numpy(snp_embeddings[emb_idxs])
        data[gene].snp_ids = gene_snps

        iso_nodes, snp_nodes = [], []
        for _, row in gdf.iterrows():
            if row["isolate_id"] in iso_to_idx and row["snp_id"] in snp_local:
                iso_nodes.append(iso_to_idx[row["isolate_id"]])
                snp_nodes.append(snp_local[row["snp_id"]])

        if iso_nodes:
            ei = torch.tensor([iso_nodes, snp_nodes], dtype=torch.long)
            data[("isolate", f"has_{gene}", gene)].edge_index = ei
            data[(gene, f"in_{gene}", "isolate")].edge_index = ei.flip(0)

    # Store arm-A layer-2 exclusion metadata (nothing to exclude)
    data.meta = {"layer2_exclude_edge_types": [], "drug_pathway_masks": None}

    n_snp = sum(data[g].x.size(0) for g in genes if hasattr(data[g], "x"))
    print(
        f"[base] {len(isolates)} isolates | {len(genes)} gene types | {n_snp} SNP nodes"
    )
    return data


# ---------------------------------------------------------------------------
# KEGG helpers
# ---------------------------------------------------------------------------

def _as_list(v) -> List[str]:
    if v is None:
        return []
    if isinstance(v, dict):
        if "pathways" in v:
            return _as_list(v["pathways"])
        return []
    if isinstance(v, (list, tuple)):
        return [str(x).strip() for x in v if str(x).strip()]
    return [str(v).strip()] if str(v).strip() else []


def _load_full_kg(kegg_json_path: str) -> dict:
    with open(kegg_json_path) as f:
        return json.load(f)


def _gene_to_pathway_lookup(kg: dict, valid_pathways: Optional[Set[str]] = None) -> Dict[str, List[str]]:
    raw = kg.get("gene_to_pathways", {})
    if not isinstance(raw, dict):
        return {}
    out = {}
    for gene, vals in raw.items():
        pws = _as_list(vals)
        if valid_pathways is not None:
            pws = [p for p in pws if p in valid_pathways]
        if pws:
            out[str(gene).strip()] = pws
    return out


def _pathways_for_gene(gene: str, lookup: Dict[str, List[str]]) -> List[str]:
    return (
        lookup.get(gene, [])
        or lookup.get(gene.lower(), [])
        or lookup.get(gene.upper(), [])
    )


def _select_pathways(kg: dict) -> List[str]:
    """
    Choose the pathway universe for arm B/C/D:
    - All pathways in the full KG gene_to_pathways values
    - INCLUDE_PATHWAYS added explicitly
    - HUB_PATHWAYS removed
    """
    seen: Set[str] = set()
    gene_to_pwys = kg.get("gene_to_pathways", {})
    if isinstance(gene_to_pwys, dict):
        for vals in gene_to_pwys.values():
            seen.update(_as_list(vals))

    # Also add from drug_to_pathways_grounded if present
    drug_grounded = kg.get("drug_to_pathways_grounded", {})
    if isinstance(drug_grounded, dict):
        for vals in drug_grounded.values():
            seen.update(_as_list(vals) if isinstance(vals, list) else [])

    seen.update(INCLUDE_PATHWAYS)
    seen -= HUB_PATHWAYS
    # Remove map-namespace refs (not mtu-specific)
    seen = {p for p in seen if p.startswith("mtu")}
    return sorted(seen)


# ---------------------------------------------------------------------------
# Arm B: Pathway-minimal
# ---------------------------------------------------------------------------

def _add_pathway_nodes(
    data: HeteroData,
    isolate_snp_df: pd.DataFrame,
    kg: dict,
) -> Tuple[HeteroData, List[str], Dict[str, int]]:
    """
    Add pathway nodes and edges to a base graph (in-place).
    Returns (data, pathways, pwy_to_idx).

    Edges added:
      isolate -> in_pathway -> pathway
      pathway -> has_isolate -> isolate  (only in layer 1; excluded in layer 2)
      gene_type -> gene_in_pathway -> pathway   (direct gene→pathway link)
      pathway -> pathway_has_gene -> gene_type   (reverse)
    """
    pathways = _select_pathways(kg)
    pwy_to_idx = {p: i for i, p in enumerate(pathways)}

    gene_to_pwys = _gene_to_pathway_lookup(kg, valid_pathways=set(pathways))

    # --- Pathway node features: gene-membership binary vectors ---
    snp_gene_types = sorted([
        nt for nt in data.node_types
        if nt not in ("isolate", "pathway", "gene", "ko", "drug")
    ])
    gene_type_to_idx = {g: i for i, g in enumerate(snp_gene_types)}
    feat_dim = max(len(snp_gene_types), 1)

    pwy_feats = np.zeros((len(pathways), feat_dim), dtype=np.float32)
    for gene_type in snp_gene_types:
        for pwy in _pathways_for_gene(gene_type, gene_to_pwys):
            if pwy in pwy_to_idx and gene_type in gene_type_to_idx:
                pwy_feats[pwy_to_idx[pwy], gene_type_to_idx[gene_type]] = 1.0

    data["pathway"].x = torch.from_numpy(pwy_feats)
    data["pathway"].pathway_ids = pathways

    # --- Isolate ↔ pathway edges ---
    iso_ids = list(getattr(data["isolate"], "isolate_ids", []))
    iso_to_idx = {iso: i for i, iso in enumerate(iso_ids)}

    ip_pairs: Set[Tuple[int, int]] = set()
    for row in isolate_snp_df[["isolate_id", "gene"]].dropna().itertuples(index=False):
        iso_id, gene = row
        if iso_id not in iso_to_idx:
            continue
        for pwy in _pathways_for_gene(gene, gene_to_pwys):
            if pwy in pwy_to_idx:
                ip_pairs.add((iso_to_idx[iso_id], pwy_to_idx[pwy]))

    if ip_pairs:
        iso_nodes = [p[0] for p in ip_pairs]
        pwy_nodes = [p[1] for p in ip_pairs]
        edge_ip = torch.tensor([iso_nodes, pwy_nodes], dtype=torch.long)
        data[("isolate", "in_pathway", "pathway")].edge_index = edge_ip
        data[("pathway", "has_isolate", "isolate")].edge_index = edge_ip.flip(0)

    # --- Gene-type ↔ pathway edges (all SNP nodes of a gene → its pathways) ---
    gp_edge_count = 0
    for gene_type in snp_gene_types:
        if not hasattr(data[gene_type], "x"):
            continue
        n_snp_nodes = data[gene_type].x.size(0)
        gene_pwys = _pathways_for_gene(gene_type, gene_to_pwys)
        valid_pwy_idxs = [pwy_to_idx[p] for p in gene_pwys if p in pwy_to_idx]
        if not valid_pwy_idxs:
            continue
        src_list, dst_list = [], []
        for snp_i in range(n_snp_nodes):
            for pi in valid_pwy_idxs:
                src_list.append(snp_i)
                dst_list.append(pi)
        if src_list:
            ei = torch.tensor([src_list, dst_list], dtype=torch.long)
            data[(gene_type, "gene_in_pathway", "pathway")].edge_index = ei
            data[("pathway", "pathway_has_gene", gene_type)].edge_index = ei.flip(0)
            gp_edge_count += len(src_list)

    print(
        f"[pathway] {len(pathways)} pathways (hubs pruned) | "
        f"{len(ip_pairs)} isolate-pathway edges | "
        f"{gp_edge_count} gene-pathway edges"
    )
    return data, pathways, pwy_to_idx


# ---------------------------------------------------------------------------
# Arm C: + KO nodes
# ---------------------------------------------------------------------------

def _add_ko_nodes(
    data: HeteroData,
    kg: dict,
    pathways: List[str],
    pwy_to_idx: Dict[str, int],
) -> HeteroData:
    """
    Add KO node type with:
      gene_type -> to_ko -> ko
      ko -> from_gene_type -> gene_type  (reverse)
      ko -> ko_to_pathway -> pathway
      pathway -> pathway_to_ko -> ko    (reverse)

    Uses bulk_links.gene_to_ko from tb_knowledge_graph_full.json to connect
    SNP gene types (e.g. "gyrA") to their KO nodes, giving pathway-orphan
    genes a functional neighborhood.
    KO features: pathway-membership binary vectors (instead of one-hot identity).
    """
    bulk = kg.get("bulk_links", {})
    gene_to_ko_raw = bulk.get("gene_to_ko", {})  # keys are "mtu:RvXXXX"

    # Build rv_id -> gene_name from gene_info
    gene_info = kg.get("gene_info", {})
    rv_to_name: Dict[str, str] = {}
    for gname, info in gene_info.items():
        rv = info.get("rv_id") or info.get("kegg_id", "")
        if rv:
            rv_to_name[rv] = gname

    # Map gene SNP node types in data to their KO IDs
    snp_gene_types = [
        nt for nt in data.node_types
        if nt not in ("isolate", "pathway", "gene", "ko")
    ]

    ko_ids_seen: Set[str] = set()
    gene_ko_pairs: List[Tuple[str, str]] = []  # (gene_node_type, ko_id)

    for gene_type in snp_gene_types:
        rv_candidates = [rv for rv, name in rv_to_name.items() if name == gene_type]
        for rv in rv_candidates:
            mtu_key = f"mtu:{rv}"
            ko_list_raw = gene_to_ko_raw.get(mtu_key, [])
            for ko_entry in ko_list_raw:
                ko_id = ko_entry.replace("ko:", "")
                if ko_id:
                    gene_ko_pairs.append((gene_type, ko_id))
                    ko_ids_seen.add(ko_id)

    if not ko_ids_seen:
        print("[ko] No KO mappings found for SNP gene types; skipping KO layer.")
        return data

    ko_list = sorted(ko_ids_seen)
    ko_to_idx = {ko: i for i, ko in enumerate(ko_list)}

    # KO node features: pathway-membership binary vectors
    pwy_set_all = set(pathways)
    ko_info_all = kg.get("ko_info", {})
    ko_feat_dim = max(len(pathways), 1)
    ko_feats = np.zeros((len(ko_list), ko_feat_dim), dtype=np.float32)
    for ko_id in ko_list:
        ko_rec = ko_info_all.get(ko_id, {})
        for entry in ko_rec.get("pathway", []):
            raw_id = entry.split()[0].strip() if isinstance(entry, str) else str(entry).strip()
            raw_id = raw_id.replace("path:", "")
            pwy_id = raw_id.replace("map", "mtu") if raw_id.startswith("map") else raw_id
            if pwy_id in pwy_set_all and pwy_id in pwy_to_idx:
                ko_feats[ko_to_idx[ko_id], pwy_to_idx[pwy_id]] = 1.0

    data["ko"].x = torch.from_numpy(ko_feats)
    data["ko"].ko_ids = ko_list

    # gene_type -> ko edges
    for gene_type, ko_id in gene_ko_pairs:
        if not hasattr(data[gene_type], "x"):
            continue
        n_snp_nodes = data[gene_type].x.size(0)
        ko_idx = ko_to_idx[ko_id]
        src = torch.arange(n_snp_nodes, dtype=torch.long)
        dst = torch.full((n_snp_nodes,), ko_idx, dtype=torch.long)
        ei = torch.stack([src, dst], dim=0)
        key_fwd = (gene_type, "to_ko", "ko")
        key_rev = ("ko", f"from_{gene_type}", gene_type)
        if key_fwd in data.edge_index_dict:
            old = data[key_fwd].edge_index
            data[key_fwd].edge_index = torch.cat([old, ei], dim=1)
            data[key_rev].edge_index = torch.cat([data[key_rev].edge_index, ei.flip(0)], dim=1)
        else:
            data[key_fwd].edge_index = ei
            data[key_rev].edge_index = ei.flip(0)

    # ko -> pathway edges (via KO records in kg)
    ko_info = kg.get("ko_info", {})
    ko_pwy_src, ko_pwy_dst = [], []
    pwy_set = set(pathways)
    for ko_id in ko_list:
        ko_idx = ko_to_idx[ko_id]
        ko_rec = ko_info.get(ko_id, {})
        pwy_entries = ko_rec.get("pathway", [])
        for entry in pwy_entries:
            raw_id = entry.split()[0].strip() if isinstance(entry, str) else str(entry).strip()
            raw_id = raw_id.replace("path:", "")
            pwy_id = raw_id.replace("map", "mtu") if raw_id.startswith("map") else raw_id
            if pwy_id in pwy_set and pwy_id in pwy_to_idx:
                ko_pwy_src.append(ko_idx)
                ko_pwy_dst.append(pwy_to_idx[pwy_id])

    if ko_pwy_src:
        ei_kp = torch.tensor([ko_pwy_src, ko_pwy_dst], dtype=torch.long)
        data[("ko", "ko_to_pathway", "pathway")].edge_index = ei_kp
        data[("pathway", "pathway_to_ko", "ko")].edge_index = ei_kp.flip(0)
        print(
            f"[ko] {len(ko_list)} KO nodes | {len(gene_ko_pairs)} gene-KO links | "
            f"{len(ko_pwy_src)} KO-pathway edges"
        )
    else:
        print(
            f"[ko] {len(ko_list)} KO nodes | {len(gene_ko_pairs)} gene-KO links | "
            f"0 KO-pathway edges (KO pathway IDs not in filtered pathway set)"
        )

    return data


# ---------------------------------------------------------------------------
# Arm D: + Drug nodes
# ---------------------------------------------------------------------------

def _add_drug_nodes(
    data: HeteroData,
    kg: dict,
    drug_cols: List[str],
    pathways: List[str],
    pwy_to_idx: Dict[str, int],
) -> Tuple[HeteroData, Dict[str, np.ndarray]]:
    """
    Add drug nodes, drug<->pathway edges, drug<->gene-SNP edges.
    Also returns drug_pathway_masks: Dict[drug_col -> bool array[n_pathways]]
    for use in drug-masked pathway attention.
    """
    drug_grounded = kg.get("drug_to_pathways_grounded", {})
    drug_to_genes = kg.get("drug_to_genes", {})

    pwy_set = set(pathways)
    n_drugs = len(drug_cols)
    drug_feats = np.eye(n_drugs, dtype=np.float32)
    data["drug"].x = torch.from_numpy(drug_feats)
    data["drug"].drug_ids = drug_cols

    drug_pathway_masks: Dict[str, np.ndarray] = {}

    dp_src, dp_dst = [], []  # drug -> pathway
    for d_idx, drug_col in enumerate(drug_cols):
        short = DRUG_COL_TO_SHORT.get(drug_col, drug_col.replace("_BINARY_PHENOTYPE", ""))
        pwys = [p for p in drug_grounded.get(short, []) if p in pwy_set]
        mask = np.zeros(len(pathways), dtype=bool)
        for pwy in pwys:
            if pwy in pwy_to_idx:
                pi = pwy_to_idx[pwy]
                dp_src.append(d_idx)
                dp_dst.append(pi)
                mask[pi] = True
        # If no drug-specific pathways found, allow all (fallback)
        if not mask.any():
            mask[:] = True
        drug_pathway_masks[drug_col] = mask

    if dp_src:
        ei_dp = torch.tensor([dp_src, dp_dst], dtype=torch.long)
        data[("drug", "drug_to_pathway", "pathway")].edge_index = ei_dp
        data[("pathway", "pathway_to_drug", "drug")].edge_index = ei_dp.flip(0)

    # drug -> gene-SNP edges
    snp_gene_types = [
        nt for nt in data.node_types
        if nt not in ("isolate", "pathway", "gene", "ko", "drug")
    ]
    gene_name_set = set(snp_gene_types)

    from collections import defaultdict
    drug_gene_edges: Dict[str, Tuple[List[int], List[int]]] = defaultdict(lambda: ([], []))
    for d_idx, drug_col in enumerate(drug_cols):
        short = DRUG_COL_TO_SHORT.get(drug_col, drug_col.replace("_BINARY_PHENOTYPE", ""))
        genes_for_drug = drug_to_genes.get(short, [])
        for gene_name in genes_for_drug:
            if gene_name in gene_name_set and hasattr(data[gene_name], "x"):
                n_snp_nodes = data[gene_name].x.size(0)
                for snp_node_i in range(n_snp_nodes):
                    drug_gene_edges[gene_name][0].append(d_idx)
                    drug_gene_edges[gene_name][1].append(snp_node_i)

    for gtype, (src_list, dst_list) in drug_gene_edges.items():
        ei = torch.tensor([src_list, dst_list], dtype=torch.long)
        data[("drug", f"drug_to_{gtype}", gtype)].edge_index = ei
        data[(gtype, f"{gtype}_to_drug", "drug")].edge_index = ei.flip(0)

    print(
        f"[drug] {n_drugs} drug nodes | {len(dp_src)} drug-pathway edges | "
        f"{sum(len(v[0]) for v in drug_gene_edges.values())} drug-gene edges"
    )
    return data, drug_pathway_masks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_graph(
    arm: str,
    isolate_snp_df: pd.DataFrame,
    phenotype_df: pd.DataFrame,
    snp_embeddings: np.ndarray,
    snp_list: List[str],
    drug_cols: List[str],
    kegg_json_path: Optional[str] = None,
) -> HeteroData:
    """
    Build the HeteroData graph for the specified ablation arm.

    Args:
        arm:            One of "A", "B", "C", "D".
        isolate_snp_df: DataFrame with isolate_id, snp_id, gene columns.
        phenotype_df:   DataFrame with isolate_id + drug label columns.
        snp_embeddings: np.ndarray [n_snps x embed_dim].
        snp_list:       Ordered list of SNP IDs (aligned to snp_embeddings rows).
        drug_cols:      List of drug label column names.
        kegg_json_path: Path to tb_knowledge_graph_full.json (required for B/C/D).

    Returns:
        HeteroData with an additional `data.meta` attribute:
          - layer2_exclude_edge_types: edge types to mask in layer 2
          - drug_pathway_masks: per-drug pathway mask (arm D) or None
    """
    arm = arm.upper()
    if arm not in ("A", "B", "C", "D"):
        raise ValueError(f"arm must be one of A/B/C/D, got: {arm}")
    if arm != "A" and kegg_json_path is None:
        raise ValueError("kegg_json_path is required for arms B/C/D.")

    # Base graph is the same for all arms
    data = _build_base_graph(
        isolate_snp_df, phenotype_df, snp_embeddings, snp_list, drug_cols
    )

    if arm == "A":
        return data

    # Load KEGG KG
    kg = _load_full_kg(kegg_json_path)

    # Arm B: + pathway nodes (hub-pruned, +mtu00074, gene→pathway edges)
    data, pathways, pwy_to_idx = _add_pathway_nodes(data, isolate_snp_df, kg)
    # Mark pathway->isolate edge as layer-2 excluded (breaks the hub cycle)
    data.meta["layer2_exclude_edge_types"] = [("pathway", "has_isolate", "isolate")]

    if arm == "B":
        return data

    # Arm C: + KO nodes
    data = _add_ko_nodes(data, kg, pathways, pwy_to_idx)

    if arm == "C":
        return data

    # Arm D: + drug nodes with per-drug pathway masks
    data, drug_pathway_masks = _add_drug_nodes(
        data, kg, drug_cols, pathways, pwy_to_idx
    )
    data.meta["drug_pathway_masks"] = drug_pathway_masks

    return data
