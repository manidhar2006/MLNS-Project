"""
amr_hgat.data — VCF parsing, PMI+SVD embeddings, phenotype loading,
                snpEff annotation integration.
"""
from __future__ import annotations

import gzip
import re
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

# ---------------------------------------------------------------------------
# Drug column constants
# ---------------------------------------------------------------------------

# 8 primary drugs (>=800 R cases in CRyPTIC 20231208)
DRUG_COLS_PRIMARY = [
    "INH_BINARY_PHENOTYPE",
    "RIF_BINARY_PHENOTYPE",
    "EMB_BINARY_PHENOTYPE",
    "LEV_BINARY_PHENOTYPE",
    "MXF_BINARY_PHENOTYPE",
    "ETH_BINARY_PHENOTYPE",
    "KAN_BINARY_PHENOTYPE",
    "AMI_BINARY_PHENOTYPE",   # Amikacin
]

# 4 secondary drugs (too few R cases for tight CIs, included but not headlined)
DRUG_COLS_SECONDARY = [
    "CFZ_BINARY_PHENOTYPE",
    "LZD_BINARY_PHENOTYPE",
    "BDQ_BINARY_PHENOTYPE",
    "DLM_BINARY_PHENOTYPE",
]

DRUG_COLS_ALL = DRUG_COLS_PRIMARY + DRUG_COLS_SECONDARY

# Short-name map for display
DRUG_SHORT = {
    "INH_BINARY_PHENOTYPE": "INH",
    "RIF_BINARY_PHENOTYPE": "RIF",
    "EMB_BINARY_PHENOTYPE": "EMB",
    "LEV_BINARY_PHENOTYPE": "LEV",
    "MXF_BINARY_PHENOTYPE": "MXF",
    "ETH_BINARY_PHENOTYPE": "ETH",
    "KAN_BINARY_PHENOTYPE": "KAN",
    "AMI_BINARY_PHENOTYPE": "AMK",
    "CFZ_BINARY_PHENOTYPE": "CFZ",
    "LZD_BINARY_PHENOTYPE": "LZD",
    "BDQ_BINARY_PHENOTYPE": "BDQ",
    "DLM_BINARY_PHENOTYPE": "DLM",
}

# ---------------------------------------------------------------------------
# MTB H37Rv gene coordinates (23-gene AMR panel from original notebooks)
# ---------------------------------------------------------------------------

MTB_GENE_REGIONS: Dict[str, Tuple[str, int, int]] = {
    "katG":  ("NC_000962.3",  2153889, 2156111),
    "fabG1": ("NC_000962.3",  1673280, 1674183),
    "inhA":  ("NC_000962.3",  1674202, 1675011),
    "ahpC":  ("NC_000962.3",  2726105, 2726950),
    "rpoB":  ("NC_000962.3",   759807,  763325),
    "rpoC":  ("NC_000962.3",   763370,  767320),
    "embA":  ("NC_000962.3",  4243310, 4245330),
    "embB":  ("NC_000962.3",  4246514, 4249810),
    "embC":  ("NC_000962.3",  4240713, 4243311),
    "pncA":  ("NC_000962.3",  2288681, 2289241),
    "gyrA":  ("NC_000962.3",     7301,    9818),
    "gyrB":  ("NC_000962.3",     5240,    7267),
    "rrs":   ("NC_000962.3",  1471846, 1473382),
    "rrl":   ("NC_000962.3",  1473382, 1475491),
    "eis":   ("NC_000962.3",  2714065, 2715698),
    "gidB":  ("NC_000962.3",  4407590, 4408192),
    "tlyA":  ("NC_000962.3",  1917960, 1918590),
    "iniB":  ("NC_000962.3",  3985480, 3987076),
    "iniA":  ("NC_000962.3",  3987102, 3988602),
    "iniC":  ("NC_000962.3",  3988727, 3989990),
    "ndh":   ("NC_000962.3",  2102479, 2103985),
    "manB":  ("NC_000962.3",  4155253, 4156591),
    "rmlD":  ("NC_000962.3",  4154020, 4155231),
}

# snpEff effect categories (ordered by severity; used for one-hot encoding)
EFFECT_CLASSES = [
    "stop_gained",
    "frameshift_variant",
    "start_lost",
    "stop_lost",
    "splice_region_variant",
    "missense_variant",
    "synonymous_variant",
    "other",
]
EFFECT_DIM = len(EFFECT_CLASSES)
EFFECT_IDX = {e: i for i, e in enumerate(EFFECT_CLASSES)}


# ---------------------------------------------------------------------------
# VCF parsing helpers
# ---------------------------------------------------------------------------

def _open_vcf(path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path, "r")


def _gt_to_af(gt: str) -> float:
    alleles = re.split(r"[/|]", gt)
    try:
        return sum(1 for a in alleles if a not in ("0", ".")) / len(alleles)
    except Exception:
        return 0.0


def _extract_af(info: str, parts: list, header: list) -> float:
    m = re.search(r"AF=([0-9.]+)", info)
    if m:
        return float(m.group(1))
    if len(parts) >= 10 and len(header) >= 10:
        fmt = parts[8].split(":") if len(parts) > 8 else []
        gt_data = parts[9].split(":") if len(parts) > 9 else []
        if fmt and gt_data:
            gi = fmt.index("GT") if "GT" in fmt else 0
            return _gt_to_af(gt_data[gi]) if gi < len(gt_data) else 0.0
    return 0.0


def _pos_to_gene(chrom: str, pos: int,
                 gene_regions: Dict[str, Tuple[str, int, int]]) -> Optional[str]:
    for gene, (g_chrom, start, end) in gene_regions.items():
        if chrom == g_chrom and start <= pos < end:
            return gene
    return None


def parse_single_sample_vcf(
    vcf_path,
    isolate_id: str,
    gene_regions=None,
    min_qual: float = 20.0,
    min_af: float = 0.75,
) -> List[Tuple[str, str, str]]:
    gene_regions = gene_regions or MTB_GENE_REGIONS
    records = []
    with _open_vcf(vcf_path) as fh:
        header: list = []
        for line in fh:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                header = line.strip().split("\t")
                continue
            parts = line.strip().split("\t")
            if len(parts) < 8:
                continue
            chrom, pos, _, ref, alt, qual, _, info = parts[:8]
            pos = int(pos)
            try:
                if float(qual) < min_qual:
                    continue
            except ValueError:
                pass
            if len(ref) != 1 or len(alt) != 1 or alt == ".":
                continue
            if _extract_af(info, parts, header) < min_af:
                continue
            gene = _pos_to_gene(chrom, pos, gene_regions)
            if gene is None:
                continue
            records.append((isolate_id, f"{gene}_{ref}{pos}{alt}", gene))
    return records


def load_vcf_directory(
    vcf_dir: str,
    gene_regions=None,
    pattern: str = "*.vcf*",
    min_qual: float = 20.0,
    min_af: float = 0.75,
    verbose: bool = False,
) -> pd.DataFrame:
    gene_regions = gene_regions or MTB_GENE_REGIONS
    vcfs = sorted(Path(vcf_dir).glob(pattern))
    if not vcfs:
        raise FileNotFoundError(f"No VCFs matching '{pattern}' in {vcf_dir}")

    all_records = []
    for vcf_path in vcfs:
        iso_id = vcf_path.name.replace(".vcf.gz", "").replace(".vcf", "")
        recs = parse_single_sample_vcf(vcf_path, iso_id, gene_regions, min_qual, min_af)
        all_records.extend(recs)
        if verbose:
            print(f"  {iso_id}: {len(recs)} SNPs")

    df = pd.DataFrame(all_records, columns=["isolate_id", "snp_id", "gene"])
    if df.empty:
        raise RuntimeError("No SNP records found after VCF parsing and filtering.")
    print(
        f"Loaded SNP table: {df['isolate_id'].nunique()} isolates, "
        f"{df['snp_id'].nunique()} unique SNPs"
    )
    return df


# ---------------------------------------------------------------------------
# Phenotype loading
# ---------------------------------------------------------------------------

def load_phenotype_table(
    phenotype_csv: str,
    drug_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Load CRyPTIC phenotype table; return (phenotype_df, available_drug_cols)."""
    drug_cols = drug_cols or DRUG_COLS_ALL
    pheno_columns = pd.read_csv(phenotype_csv, nrows=0).columns.tolist()
    available = [c for c in drug_cols if c in pheno_columns]
    if not available:
        raise ValueError(f"None of the requested drug columns found in {phenotype_csv}")

    raw = pd.read_csv(phenotype_csv)
    if "ENA_SAMPLE" not in raw.columns:
        raise KeyError("Expected ENA_SAMPLE column in phenotype table.")

    df = raw[["ENA_SAMPLE", *available]].copy()
    df = df.rename(columns={"ENA_SAMPLE": "isolate_id"})

    label_map = {"R": 1.0, "S": 0.0}
    for col in available:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.upper()
            .map(label_map)
            .astype(float)
        )
    print(f"Phenotype table: {len(df)} isolates, {len(available)} drugs: {available}")
    return df, available


# ---------------------------------------------------------------------------
# PMI + SVD embeddings
# ---------------------------------------------------------------------------

def compute_pmi_embeddings(
    isolate_snp_df: pd.DataFrame,
    embed_dim: int = 64,
    min_count: int = 5,
    train_isolates: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build PMI co-occurrence matrix from isolate-SNP incidence, apply SVD.
    Returns (embeddings [n_snps x embed_dim], snp_list).

    Args:
        train_isolates: If provided, use only these isolates for PMI computation
                        (prevents test-set leakage). SNP vocabulary is still
                        determined from train_isolates with min_count filtering.
    """
    if train_isolates is not None:
        train_set = set(train_isolates)
        working_df = isolate_snp_df[isolate_snp_df["isolate_id"].isin(train_set)]
    else:
        working_df = isolate_snp_df

    snp_counts = working_df["snp_id"].value_counts()
    valid_snps = snp_counts[snp_counts >= min_count].index.tolist()
    df = working_df[working_df["snp_id"].isin(valid_snps)].copy()

    isolates = df["isolate_id"].unique().tolist()
    snps = df["snp_id"].unique().tolist()
    iso_idx = {s: i for i, s in enumerate(isolates)}
    snp_idx = {s: i for i, s in enumerate(snps)}

    rows = [iso_idx[r] for r in df["isolate_id"]]
    cols = [snp_idx[r] for r in df["snp_id"]]
    mat = csr_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(len(isolates), len(snps)),
        dtype=np.float32,
    )

    co = (mat.T @ mat).toarray()
    total = co.sum()
    p_ij = co / total
    p_i = co.sum(axis=1, keepdims=True) / total
    p_j = co.sum(axis=0, keepdims=True) / total

    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log(p_ij / (p_i * p_j + 1e-12))
    pmi = np.nan_to_num(pmi, nan=0.0, neginf=0.0)
    pmi = np.maximum(pmi, 0.0)

    k = max(1, min(embed_dim, min(pmi.shape) - 1))
    u, s_vals, _ = svds(csr_matrix(pmi), k=k)
    embeddings = (u * np.sqrt(s_vals)).astype(np.float32)

    if embeddings.shape[1] < embed_dim:
        pad = np.zeros((embeddings.shape[0], embed_dim - embeddings.shape[1]), dtype=np.float32)
        embeddings = np.concatenate([embeddings, pad], axis=1)

    tag = "train-only" if train_isolates is not None else "full"
    print(f"PMI+SVD embeddings ({tag}): {embeddings.shape} for {len(snps)} SNPs")
    return embeddings, snps


# ---------------------------------------------------------------------------
# snpEff effect-class annotation
# ---------------------------------------------------------------------------

def _parse_ann_field(ann: str) -> str:
    """Extract the most severe effect class from a snpEff ANN= field."""
    severity_order = {e: i for i, e in enumerate(EFFECT_CLASSES)}
    best = "other"
    best_rank = severity_order["other"]
    for entry in ann.split(","):
        parts = entry.split("|")
        if len(parts) < 2:
            continue
        raw = parts[1].strip().lower()
        for cls in EFFECT_CLASSES:
            if cls in raw:
                r = severity_order[cls]
                if r < best_rank:
                    best_rank = r
                    best = cls
                break
    return best


def load_snpeff_annotations(ann_parquet: str) -> pd.DataFrame:
    """
    Load pre-computed snpEff annotation table.
    Expected columns: snp_id (gene_refPOSalt), effect_class.
    Returns DataFrame indexed by snp_id with effect_class column.
    """
    df = pd.read_parquet(ann_parquet)
    if "snp_id" not in df.columns or "effect_class" not in df.columns:
        raise ValueError("snpEff parquet must have 'snp_id' and 'effect_class' columns.")
    return df.set_index("snp_id")["effect_class"].to_dict()


def append_effect_class_features(
    snp_embeddings: np.ndarray,
    snp_list: List[str],
    effect_map: Optional[Dict[str, str]],
) -> np.ndarray:
    """
    Append EFFECT_DIM one-hot columns for effect class to snp_embeddings.
    If effect_map is None, appends all-zero columns (no-op).
    """
    n = len(snp_list)
    one_hot = np.zeros((n, EFFECT_DIM), dtype=np.float32)
    if effect_map is not None:
        for i, snp_id in enumerate(snp_list):
            cls = effect_map.get(snp_id, "other")
            one_hot[i, EFFECT_IDX.get(cls, EFFECT_IDX["other"])] = 1.0
    return np.concatenate([snp_embeddings, one_hot], axis=1)


# ---------------------------------------------------------------------------
# Unified data loader
# ---------------------------------------------------------------------------

def load_data(
    vcf_dir: str = "./data/cryptic/vcf",
    phenotype_csv: str = "./data/cryptic/CRyPTIC_reuse_table_20231208.csv",
    embed_dim: int = 64,
    min_snp_count: int = 5,
    min_qual: float = 20.0,
    min_af: float = 0.75,
    drug_cols: Optional[List[str]] = None,
    snpeff_parquet: Optional[str] = None,
    verbose: bool = False,
) -> dict:
    """
    Full data pipeline. Returns dict with:
      isolate_snp_df, phenotype_df, snp_embeddings, snp_list,
      drug_cols, effect_map (or None)
    """
    isolate_snp_df = load_vcf_directory(
        vcf_dir, min_qual=min_qual, min_af=min_af, verbose=verbose
    )
    phenotype_df, available_drug_cols = load_phenotype_table(
        phenotype_csv, drug_cols=drug_cols
    )
    snp_embeddings, snp_list = compute_pmi_embeddings(
        isolate_snp_df, embed_dim=embed_dim, min_count=min_snp_count
    )

    effect_map = None
    if snpeff_parquet and Path(snpeff_parquet).exists():
        effect_map = load_snpeff_annotations(snpeff_parquet)
        snp_embeddings = append_effect_class_features(snp_embeddings, snp_list, effect_map)
        print(f"Appended snpEff one-hot; embedding dim now {snp_embeddings.shape[1]}")
    else:
        if snpeff_parquet:
            print(f"[warn] snpEff parquet not found at {snpeff_parquet}; skipping effect features.")

    return {
        "isolate_snp_df": isolate_snp_df,
        "phenotype_df": phenotype_df,
        "snp_embeddings": snp_embeddings,
        "snp_list": snp_list,
        "drug_cols": available_drug_cols,
        "effect_map": effect_map,
    }
