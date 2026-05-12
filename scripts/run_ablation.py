#!/usr/bin/env python3
"""
scripts/run_ablation.py — 5-fold CV for MutGAT paper ablation arms A–D.

Arms: A HGAT-Base, B HGAT-Path, C HGAT-KO, D MutGAT (see paper Section 3).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from amr_hgat.data import DRUG_COLS_PRIMARY, load_data
from amr_hgat.graph_builders import build_graph
from amr_hgat.train import DEVICE, cross_validate

DEFAULT_VCF = "./data/cryptic/vcf"
DEFAULT_PHENO = "./data/cryptic/CRyPTIC_reuse_table_20231208.csv"
DEFAULT_KEGG = "./kegg_data/tb_knowledge_graph_full.json"


def _print_graph_stats(data) -> None:
    print("  Node types: ", list(data.x_dict.keys()))
    print("  Edge types: ", list(data.edge_index_dict.keys()))
    meta = getattr(data, "meta", {}) or {}
    print("  meta:       ", meta.get("layer2_exclude_edge_types", []))


def run_arm(
    arm: str,
    data_bundle: dict,
    isolates: list,
    ref_graph,
    kegg_json: str | None,
    output_dir: Path,
    dry_run: bool,
    epochs: int,
    seed: int,
    embed_dim: int,
    min_snp_count: int,
    hidden_channels: int,
    heads: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    patience: int,
) -> dict | None:
    arm = arm.upper()
    print(f"\n{'=' * 60}\nARM {arm}\n{'=' * 60}")

    t0 = time.time()
    data = build_graph(
        arm=arm,
        isolate_snp_df=data_bundle["isolate_snp_df"],
        phenotype_df=data_bundle["phenotype_df"],
        snp_embeddings=data_bundle["snp_embeddings"],
        snp_list=data_bundle["snp_list"],
        drug_cols=data_bundle["drug_cols"],
        kegg_json_path=kegg_json if arm != "A" else None,
    )
    print(f"Graph built in {time.time() - t0:.1f}s")
    _print_graph_stats(data)

    if dry_run:
        return None

    arm_dir = output_dir / arm
    arm_dir.mkdir(parents=True, exist_ok=True)

    summary = cross_validate(
        arm=arm,
        data=ref_graph,
        drug_cols=data_bundle["drug_cols"],
        phenotype_df=data_bundle["phenotype_df"],
        isolates=isolates,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        patience=patience,
        hidden_channels=hidden_channels,
        heads=heads,
        dropout=dropout,
        seed=seed,
        device=DEVICE,
        output_dir=str(arm_dir),
        verbose=True,
        isolate_snp_df=data_bundle["isolate_snp_df"],
        embed_dim=embed_dim,
        min_snp_count=min_snp_count,
        kegg_json_path=kegg_json if arm != "A" else None,
        effect_map=data_bundle.get("effect_map"),
    )

    summary_path = arm_dir / "cv_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")
    print(f"ARM {arm} total time: {time.time() - t0:.1f}s")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="MutGAT matched ablation (arms A–D)")
    p.add_argument("--vcf-dir", default=DEFAULT_VCF)
    p.add_argument("--phenotype-csv", default=DEFAULT_PHENO)
    p.add_argument("--kegg-json", default=DEFAULT_KEGG)
    p.add_argument("--output-dir", default="runs")
    p.add_argument("--arms", nargs="+", default=["A", "B", "C", "D"])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--epochs", type=int, default=180)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--min-snp-count", type=int, default=5)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--snpeff-parquet", default="")
    args = p.parse_args()

    print(f"Device: {DEVICE}")
    print(f"Arms: {args.arms}")
    print(f"Config: embed_dim={args.embed_dim}, min_snp_count={args.min_snp_count}, "
          f"hidden={args.hidden}, heads={args.heads}, dropout={args.dropout}, "
          f"lr={args.lr}, weight_decay={args.weight_decay}, patience={args.patience}, "
          f"epochs={args.epochs}")

    t_load = time.time()
    print("\nLoading data ...")
    data_bundle = load_data(
        vcf_dir=args.vcf_dir,
        phenotype_csv=args.phenotype_csv,
        embed_dim=args.embed_dim,
        min_snp_count=args.min_snp_count,
        drug_cols=DRUG_COLS_PRIMARY,
        snpeff_parquet=args.snpeff_parquet or None,
        verbose=True,
    )
    print(f"Data loaded in {time.time() - t_load:.1f}s")

    ref_graph = build_graph(
        arm="A",
        isolate_snp_df=data_bundle["isolate_snp_df"],
        phenotype_df=data_bundle["phenotype_df"],
        snp_embeddings=data_bundle["snp_embeddings"],
        snp_list=data_bundle["snp_list"],
        drug_cols=data_bundle["drug_cols"],
    )
    isolates = list(ref_graph["isolate"].isolate_ids)

    out = Path(args.output_dir)
    kegg = args.kegg_json if Path(args.kegg_json).exists() else None
    if kegg is None and any(a.upper() != "A" for a in args.arms):
        sys.exit(f"KEGG JSON not found: {args.kegg_json}")

    for arm in args.arms:
        run_arm(
            arm=arm,
            data_bundle=data_bundle,
            isolates=isolates,
            ref_graph=ref_graph,
            kegg_json=kegg,
            output_dir=out,
            dry_run=args.dry_run,
            epochs=args.epochs,
            seed=args.seed,
            embed_dim=args.embed_dim,
            min_snp_count=args.min_snp_count,
            hidden_channels=args.hidden,
            heads=args.heads,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
        )


if __name__ == "__main__":
    main()
