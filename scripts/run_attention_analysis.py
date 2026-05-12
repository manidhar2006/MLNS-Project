#!/usr/bin/env python3
"""
scripts/run_attention_analysis.py — Attention interpretation for arm D, second-line drugs.

Follows Yang et al. (2021) bbab299 methodology exactly:
  - Retrain on the FULL dataset (all isolates, no test holdout) for interpretability,
    matching the paper's approach: "demonstrated scores on a new training set by
    combining the original training and testing datasets" (Limitations section).
  - Extract type-level (gene) attention scores and node-level (SNP) rankings
    for second-line drugs: ETH, KAN, AMI, LEV, MXF.
  - For arm D: also extract per-drug pathway attention (novel extension).

Usage:
    python scripts/run_attention_analysis.py [--epochs 150] [--output-dir runs/attention]

Output:
    runs/attention/attention_results.json   — full structured results
    runs/attention/attention_report.txt     — human-readable tables
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from amr_hgat.attention import (
    KNOWN_RESISTANCE_GENES,
    SECOND_LINE_DRUGS,
    attach_attention_hooks,
    compute_type_level_attention,
    compute_snp_attention,
    extract_pathway_attention,
    print_pathway_attention,
    print_snp_ranking,
    print_type_level_attention,
    rank_snps,
    run_attention_analysis,
)
from amr_hgat.data import (
    DRUG_COLS_PRIMARY,
    append_effect_class_features,
    compute_pmi_embeddings,
    load_data,
)
from amr_hgat.graph_builders import build_graph
from amr_hgat.model import build_model
from amr_hgat.train import masked_bce_loss, _pos_weights, DEVICE


DEFAULT_KEGG_JSON   = "./kegg_data/tb_knowledge_graph_full.json"
DEFAULT_VCF_DIR     = "./data/cryptic/vcf"
DEFAULT_PHENOTYPE   = "./data/cryptic/CRyPTIC_reuse_table_20231208.csv"
DEFAULT_OUTPUT_DIR  = "./runs/attention"

# ---------------------------------------------------------------------------
# Drug metadata: all 8 drugs + their primary resistance genes
# ---------------------------------------------------------------------------
DRUG_INFO = {
    "INH_BINARY_PHENOTYPE": ("INH", ["katG", "inhA", "fabG1", "ahpC"]),
    "RIF_BINARY_PHENOTYPE": ("RIF", ["rpoB", "rpoC"]),
    "EMB_BINARY_PHENOTYPE": ("EMB", ["embB", "embA", "embC"]),
    "LEV_BINARY_PHENOTYPE": ("LEV", ["gyrA", "gyrB"]),
    "MXF_BINARY_PHENOTYPE": ("MXF", ["gyrA", "gyrB"]),
    "ETH_BINARY_PHENOTYPE": ("ETH", ["inhA", "fabG1", "ndh"]),
    "KAN_BINARY_PHENOTYPE": ("KAN", ["rrs", "eis", "gidB", "tlyA", "rrl"]),
    "AMI_BINARY_PHENOTYPE": ("AMI", ["rrs", "eis", "rrl"]),
}

# ---------------------------------------------------------------------------
# Amino-acid annotation helpers (H37Rv coordinates)
# ---------------------------------------------------------------------------
# Gene start positions on the forward (+) strand (1-indexed, inclusive).
# For reverse-strand genes we store (end_pos, '-').
# RNA genes (rrs = 16S rRNA, rrl = 23S rRNA): no protein codons.
# These are excluded from codon-based annotation.
_RNA_GENES = {"rrs", "rrl"}

# For eis (Rv2416c, reverse strand), the clinically important resistance
# mutations are in the PROMOTER region upstream of the CDS, not coding SNPs.
# CDS end (reverse strand) ≈ 2715007; promoter mutations are at positions > 2715007.
_EIS_CDS_END = 2715007
_EIS_PROMOTER_WINDOW = 600  # bp upstream = positions 2715007..2715607

_GENE_COORDS = {
    "gyrA": (7302,    "+"),   # Rv0006  forward
    "gyrB": (5240,    "+"),   # Rv0005  forward
    "rpoB": (759807,  "+"),   # Rv0667  forward
    "rpoC": (762853,  "+"),   # Rv0668  forward (approx)
    "katG": (2156111, "-"),   # Rv1908c reverse, end=2156111
    "embB": (4247429, "+"),   # Rv3795  forward
    "embA": (4243131, "+"),   # Rv3794  forward (approx)
    "inhA": (1674204, "+"),   # Rv1484  forward
    "fabG1":(1673280, "+"),   # adjacent to inhA promoter region
    "eis":  (_EIS_CDS_END, "-"),   # Rv2416c reverse, end≈2715007
}

# Key resistance positions (codon number) per gene, as named in the literature
_KNOWN_CODONS = {
    "gyrA": {90: "A90V/T", 91: "S91P", 94: "D94G/N/H/A/Y"},
    "katG": {315: "S315T/N (main INH)"},
    "rpoB": {435: "D435V", 445: "H445Y/D/R/L/P", 450: "S450L/W (most common)",
             452: "L452P", 460: "H460D/Y"},
    "embB": {306: "M306V/I/L (most common)", 354: "G354A", 406: "G406A/D/S",
             497: "D497A"},
}

def _snp_to_aa(snp_id: str) -> str:
    """
    Attempt to decode a SNP id (e.g. gyrA_C7570T) to an amino acid annotation.

    Returns a short annotation string, e.g.:
      - "codon 90 → A90V/T"    for annotated resistance codons
      - "codon 94 (QRDR)"      for QRDR positions without a named allele
      - "promoter (−Xbp)"      for eis upstream mutations
      - ""                     for RNA genes (rrs, rrl) or unknown genes
    """
    import re
    parts = snp_id.split("_", 1)
    if len(parts) != 2:
        return ""
    gene, change = parts[0], parts[1]

    # RNA genes: no protein codons — return blank to avoid misleading "codon N" labels
    if gene in _RNA_GENES:
        return ""

    # eis: check for promoter vs CDS
    if gene == "eis":
        m = re.search(r"(\d+)", change)
        if not m:
            return ""
        pos = int(m.group(1))
        offset = _EIS_CDS_END - pos   # positive = inside/near CDS; negative handled below
        if pos > _EIS_CDS_END:
            # Upstream of CDS on reverse strand = promoter
            dist = pos - _EIS_CDS_END
            if dist <= _EIS_PROMOTER_WINDOW:
                return f"promoter (−{dist}bp)"
            return ""
        if offset >= 0:
            codon_num = offset // 3 + 1
            return f"codon {codon_num}"
        return ""

    if gene not in _GENE_COORDS:
        return ""
    start, strand = _GENE_COORDS[gene]
    m = re.search(r"(\d+)", change)
    if not m:
        return ""
    pos = int(m.group(1))
    if strand == "+":
        offset = pos - start          # 0-indexed offset in CDS
    else:
        offset = start - pos          # reverse strand: distance from end
    if offset < 0:
        return ""
    codon_num = offset // 3 + 1
    known = _KNOWN_CODONS.get(gene, {}).get(codon_num, "")
    if known:
        return f"codon {codon_num} → {known}"
    # Flag QRDR for gyrA
    if gene == "gyrA" and 88 <= codon_num <= 94:
        return f"codon {codon_num} (QRDR)"
    # Flag rpoB RRDR
    if gene == "rpoB" and 430 <= codon_num <= 520:
        return f"codon {codon_num} (RRDR)"
    return f"codon {codon_num}"


# ---------------------------------------------------------------------------
# Full-data training (paper's interpretability mode)
# ---------------------------------------------------------------------------

def train_full(
    model,
    data,
    drug_cols,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: str,
    verbose: bool = True,
) -> None:
    """
    Train on ALL isolates (no train/test split) for a fixed number of epochs.
    This matches the paper's approach for producing interpretability figures.
    """
    model = model.to(device)
    data = data.to(device)

    all_idx = torch.arange(data["isolate"].x.size(0), device=device)
    y_all = data["isolate"].y[all_idx]
    pw = _pos_weights(y_all).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x_dict, data.edge_index_dict)
        loss = masked_bce_loss(logits[all_idx], y_all, pw)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if verbose and epoch % 30 == 0:
            elapsed = time.time() - t0
            print(f"  epoch={epoch:03d}  loss={loss.item():.4f}  [{elapsed:.0f}s]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Attention analysis for arm D, second-line drugs."
    )
    parser.add_argument("--vcf-dir",        default=DEFAULT_VCF_DIR)
    parser.add_argument("--phenotype-csv",  default=DEFAULT_PHENOTYPE)
    parser.add_argument("--kegg-json",      default=DEFAULT_KEGG_JSON)
    parser.add_argument("--snpeff-parquet", default=None)
    parser.add_argument("--epochs",         type=int, default=150,
                        help="Training epochs on full dataset (default: 150).")
    parser.add_argument("--lr",             type=float, default=5e-4)
    parser.add_argument("--weight-decay",   type=float, default=1e-4)
    parser.add_argument("--output-dir",     default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-n-snps",     type=int, default=10,
                        help="Number of top SNPs to report per drug/layer/head.")
    parser.add_argument("--device",         default=DEVICE)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data — use full-dataset PMI embeddings (as paper does)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Loading data ...")
    t0 = time.time()
    bundle = load_data(
        vcf_dir=args.vcf_dir,
        phenotype_csv=args.phenotype_csv,
        embed_dim=64,
        min_snp_count=5,
        min_qual=20.0,
        min_af=0.75,
        drug_cols=DRUG_COLS_PRIMARY,
        snpeff_parquet=args.snpeff_parquet,
    )
    drug_cols = bundle["drug_cols"]
    print(f"Data loaded in {time.time()-t0:.1f}s")

    isolate_snp_df  = bundle["isolate_snp_df"]
    phenotype_df    = bundle["phenotype_df"]
    snp_embeddings  = bundle["snp_embeddings"]
    snp_list        = bundle["snp_list"]
    effect_map      = bundle.get("effect_map")

    # ------------------------------------------------------------------
    # 2. Build arm D graph on FULL dataset (all isolates as training)
    # ------------------------------------------------------------------
    print("\nBuilding arm D graph (full dataset) ...")
    t0 = time.time()
    data = build_graph(
        arm="D",
        isolate_snp_df=isolate_snp_df,
        phenotype_df=phenotype_df,
        snp_embeddings=snp_embeddings,
        snp_list=snp_list,
        drug_cols=drug_cols,
        kegg_json_path=args.kegg_json,
    )
    print(f"Graph built in {time.time()-t0:.1f}s")
    print(f"  Node types:  {data.node_types}")
    print(f"  Pathway ids: {getattr(data['pathway'], 'pathway_ids', [])}")

    # ------------------------------------------------------------------
    # 3. Train arm D on full dataset
    # ------------------------------------------------------------------
    print(f"\nTraining arm D on full dataset for {args.epochs} epochs ...")
    print("(Paper approach: 'demonstrated scores on a new training set")
    print(" by combining the original training and testing datasets')")
    model = build_model(
        arm="D",
        data=data,
        drug_cols=drug_cols,
        hidden_channels=128,
        heads=2,
        dropout=0.3,
    )
    t0 = time.time()
    train_full(
        model=model,
        data=data,
        drug_cols=drug_cols,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        verbose=True,
    )
    print(f"Training done in {time.time()-t0:.1f}s")

    # Save trained model
    model_path = out_dir / "arm_D_full_data.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # ------------------------------------------------------------------
    # 4. Run attention analysis
    # ------------------------------------------------------------------
    print("\nExtracting attention weights ...")
    results = run_attention_analysis(
        model=model,
        data=data,
        drug_cols=drug_cols,
        device=args.device,
        top_n_snps=args.top_n_snps,
    )

    # ------------------------------------------------------------------
    # 5. Load pathway names from KEGG JSON for readable output
    # ------------------------------------------------------------------
    pathway_names: Dict[str, str] = {}
    try:
        with open(args.kegg_json) as _kf:
            _kg = json.load(_kf)
        for pid, info in _kg.get("pathway_info", {}).items():
            raw = info.get("name", pid) if isinstance(info, dict) else pid
            pathway_names[pid] = (
                raw.replace(" - Mycobacterium tuberculosis H37Rv", "")
                   .replace(" - Mycobacterium tuberculosis", "")
            )
    except Exception:
        pass

    def pwy_label(pid: str) -> str:
        name = pathway_names.get(pid, "")
        return f"{pid}  {name}" if name else pid

    # ------------------------------------------------------------------
    # 6. Build report — all 8 drugs
    # ------------------------------------------------------------------
    report_lines = []
    def rprint(*args_):
        line = " ".join(str(a) for a in args_)
        print(line)
        report_lines.append(line)

    rprint()
    rprint("=" * 70)
    rprint("ATTENTION INTERPRETATION REPORT — MutGAT (ARM D) — ALL 8 DRUGS")
    rprint("Following Yang et al. (2021) bbab299 methodology")
    rprint("=" * 70)

    # Collect summary rows for the end-of-report table
    summary_rows = []   # (drug, gene, rank1_snp, rank1_score, aa_note, pathway_top)

    for drug_col in drug_cols:
        if drug_col not in DRUG_INFO:
            continue
        short, known = DRUG_INFO[drug_col]
        drug_idx = drug_cols.index(drug_col)
        y = data["isolate"].y[:, drug_idx]
        n_r = int((y == 1.0).sum())
        n_s = int((y == 0.0).sum())

        rprint()
        rprint("─" * 70)
        rprint(f" {short}  ({drug_col})   R={n_r}  S={n_s}")
        rprint(f" Known resistance genes: {', '.join(known)}")
        rprint("─" * 70)

        for subset in ["resistant"]:          # paper focuses on resistant isolates
            drug_data = results.get(drug_col, {}).get(subset, {})
            per_gene = drug_data.get("per_gene_snps", {})

            # Gene contribution data is stored in JSON but not printed:
            # Cross-gene contribution magnitude is confounded by MDR co-occurrence
            # (LEV/MXF resistant isolates are predominantly MDR, so INH/RIF resistance
            # genes dominate the delta). Drug-specific gene importance is better captured
            # by the within-gene SNP rankings and pathway attention below.

            for layer in ["layer1", "layer2"]:
                for gene_name in known:
                    ranked_within = (
                        per_gene.get(gene_name, {})
                               .get(layer, {})
                               .get("head1", [])
                    )
                    if not ranked_within:
                        continue
                    rprint()
                    rprint(f"  {short} | {subset:10s} | {layer} | {gene_name} — top SNPs (within-gene α)")
                    rprint(f"  {'Rk':<3}  {'SNP':<30}  {'α':>6}  Annotation")
                    rprint(f"  {'--':<3}  {'-'*30}  {'------':>6}  ----------")
                    for rank, (snp_id, score) in enumerate(ranked_within[:10], 1):
                        aa = _snp_to_aa(snp_id)
                        rprint(f"  {rank:<3}  {snp_id:<30}  {score:>6.4f}  {aa}")
                        # Collect for summary (layer1, rank 1 only)
                        if layer == "layer1" and rank == 1:
                            # Also store gene contribution rank for this gene
                            contrib_rank = list(
                                drug_data.get("gene_contribution", {})
                                         .get("layer1", {}).keys()
                            ).index(gene_name) + 1 if gene_name in drug_data.get(
                                "gene_contribution", {}).get("layer1", {}) else -1
                            summary_rows.append((short, gene_name, snp_id, score, aa, contrib_rank))

            # ---- Pathway attention (arm D) — resistant and all ----
            for s2 in ["all", "resistant"]:
                pwy_ranked = results.get(drug_col, {}).get(s2, {}).get("pathway", [])
                if not pwy_ranked:
                    continue
                # Top non-zero pathways only
                nonzero = [(p, sc) for p, sc in pwy_ranked if sc > 1e-4]
                if not nonzero:
                    continue
                rprint()
                rprint(f"  {short} | {s2:10s} | PATHWAY attention (arm D, top {min(5,len(nonzero))})")
                rprint(f"  {'α':>6}  Pathway")
                rprint(f"  {'------':>6}  -------")
                for pwy_id, score in nonzero[:5]:
                    rprint(f"  {score:>6.4f}  {pwy_label(pwy_id)}")

    # ------------------------------------------------------------------
    # 7. Cross-drug summary table
    # ------------------------------------------------------------------
    rprint()
    rprint("=" * 70)
    rprint("SUMMARY — Top-ranked SNP per known resistance gene (layer1, resistant)")
    rprint("=" * 70)
    rprint(f"  {'Drug':<5}  {'Gene':<7}  {'GeneRank':>8}  {'Top SNP':<30}  {'α':>6}  Annotation")
    rprint(f"  {'----':<5}  {'----':<7}  {'--------':>8}  {'-'*30}  {'------':>6}  ----------")
    for row in summary_rows:
        drug, gene, snp, score, aa = row[0], row[1], row[2], row[3], row[4]
        crank = row[5] if len(row) > 5 else -1
        rank_str = f"#{crank}" if crank > 0 else "—"
        rprint(f"  {drug:<5}  {gene:<7}  {rank_str:>8}  {snp:<30}  {score:>6.4f}  {aa}")

    rprint()
    rprint("=" * 70)

    # ------------------------------------------------------------------
    # 6. Save outputs
    # ------------------------------------------------------------------
    report_path = out_dir / "attention_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\nReport saved to {report_path}")

    # Serialise results to JSON
    def to_serializable(obj):
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, list):
            return [to_serializable(x) for x in obj]
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, tuple):
            return list(obj)
        return obj

    json_results = to_serializable(results)
    json_path = out_dir / "attention_results.json"
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"Full results saved to {json_path}")


if __name__ == "__main__":
    main()
