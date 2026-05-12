#!/usr/bin/env python3
"""
scripts/annotate_vcfs.py — Run snpEff on all MTB VCFs and produce a
per-SNP effect_class parquet for use as features in amr_hgat.

Usage:
    # First install snpEff if not present and build the MTB database:
    #   snpEff build -gff3 Mycobacterium_tuberculosis_h37rv
    # Then:
    python scripts/annotate_vcfs.py \
        --vcf-dir ./vcf(2024) \
        --output  ./snpeff_annotations.parquet \
        [--snpeff-jar /path/to/snpEff.jar] \
        [--genome    Mycobacterium_tuberculosis_h37rv]

Output parquet columns: snp_id, effect_class
where snp_id matches the format produced by amr_hgat.data (gene_refPOSalt).
"""
from __future__ import annotations

import argparse
import gzip
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Effect class hierarchy (most severe first)
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

# MTB H37Rv gene regions (matching amr_hgat.data.MTB_GENE_REGIONS)
MTB_GENE_REGIONS = {
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


def _pos_to_gene(chrom: str, pos: int) -> Optional[str]:
    for gene, (g_chrom, start, end) in MTB_GENE_REGIONS.items():
        if chrom == g_chrom and start <= pos < end:
            return gene
    return None


def _open_vcf(path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path, "r")


def _parse_ann(ann_field: str) -> str:
    """
    Extract the most severe effect class from a snpEff ANN= field.
    Falls back to inferring from EFF= (older snpEff format).
    """
    severity = {e: i for i, e in enumerate(EFFECT_CLASSES)}
    best = "other"
    best_rank = severity["other"]
    for entry in ann_field.split(","):
        parts = entry.split("|")
        effect_str = parts[1].strip().lower() if len(parts) > 1 else ""
        for cls in EFFECT_CLASSES:
            if cls in effect_str:
                if severity[cls] < best_rank:
                    best_rank = severity[cls]
                    best = cls
                break
    return best


def _parse_eff_legacy(eff_field: str) -> str:
    """Fallback for older snpEff EFF= format."""
    legacy_map = {
        "stop_gained": "stop_gained",
        "frameshift_coding": "frameshift_variant",
        "start_lost": "start_lost",
        "stop_lost": "stop_lost",
        "splice_site_region": "splice_region_variant",
        "non_synonymous_coding": "missense_variant",
        "synonymous_coding": "synonymous_variant",
        "non_synonymous_start": "missense_variant",
    }
    severity = {e: i for i, e in enumerate(EFFECT_CLASSES)}
    best = "other"
    best_rank = severity["other"]
    for entry in eff_field.split(","):
        eff_type = entry.split("(")[0].strip().lower()
        mapped = legacy_map.get(eff_type, "other")
        if severity.get(mapped, 99) < best_rank:
            best_rank = severity[mapped]
            best = mapped
    return best


def run_snpeff_on_vcf(
    vcf_path: str,
    snpeff_jar: str,
    genome: str,
    snpeff_config: Optional[str] = None,
    java_opts: str = "-Xmx4g",
) -> Dict[str, str]:
    """
    Run snpEff on a single VCF. Returns {snp_id: effect_class}.
    """
    cmd = ["java", java_opts, "-jar", snpeff_jar]
    if snpeff_config:
        cmd += ["-c", snpeff_config]
    cmd += [
        "ann",
        "-noStats",
        "-quiet",
        genome,
        vcf_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [warn] snpEff failed on {vcf_path}: {result.stderr[:200]}", file=sys.stderr)
        return {}

    effects: Dict[str, str] = {}
    for line in result.stdout.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 8:
            continue
        chrom = parts[0]
        try:
            pos = int(parts[1])
        except ValueError:
            continue
        ref = parts[3]
        alt = parts[4]
        info = parts[7]

        gene = _pos_to_gene(chrom, pos)
        if gene is None:
            continue
        if len(ref) != 1 or len(alt) != 1 or alt == ".":
            continue

        snp_id = f"{gene}_{ref}{pos}{alt}"
        ann_match = re.search(r"ANN=([^;\t]+)", info)
        eff_match = re.search(r"EFF=([^;\t]+)", info)

        if ann_match:
            effect = _parse_ann(ann_match.group(1))
        elif eff_match:
            effect = _parse_eff_legacy(eff_match.group(1))
        else:
            effect = "other"

        # Only update if new effect is more severe
        existing = effects.get(snp_id, "other")
        sev = {e: i for i, e in enumerate(EFFECT_CLASSES)}
        if sev.get(effect, 99) < sev.get(existing, 99):
            effects[snp_id] = effect

    return effects


def annotate_vcf_directory(
    vcf_dir: str,
    snpeff_jar: str,
    genome: str,
    output_parquet: str,
    snpeff_config: Optional[str] = None,
    pattern: str = "*.vcf*",
) -> pd.DataFrame:
    """
    Run snpEff over all VCFs in a directory, collecting per-SNP effect classes.
    Writes output_parquet and returns the DataFrame.
    """
    vcfs = sorted(Path(vcf_dir).glob(pattern))
    if not vcfs:
        raise FileNotFoundError(f"No VCFs found in {vcf_dir} matching '{pattern}'")
    print(f"Found {len(vcfs)} VCF files.")

    all_effects: Dict[str, str] = {}
    sev = {e: i for i, e in enumerate(EFFECT_CLASSES)}

    for i, vcf_path in enumerate(vcfs):
        if i % 100 == 0:
            print(f"  Processing {i}/{len(vcfs)} ...")
        file_effects = run_snpeff_on_vcf(
            str(vcf_path), snpeff_jar, genome, snpeff_config
        )
        for snp_id, eff in file_effects.items():
            existing = all_effects.get(snp_id, "other")
            if sev.get(eff, 99) < sev.get(existing, 99):
                all_effects[snp_id] = eff

    df = pd.DataFrame(
        [{"snp_id": k, "effect_class": v} for k, v in all_effects.items()]
    )
    print(f"\nAnnotated {len(df)} unique SNPs.")
    if not df.empty:
        print("Effect class distribution:")
        print(df["effect_class"].value_counts().to_string())

    Path(output_parquet).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_parquet, index=False)
    print(f"\nSaved to {output_parquet}")
    return df


def annotate_without_snpeff(
    vcf_dir: str,
    output_parquet: str,
    pattern: str = "*.vcf*",
) -> pd.DataFrame:
    """
    Lightweight fallback: classify SNPs as missense/synonymous/frameshift
    based on variant length alone (no codon-level analysis).
    Useful when snpEff is not installed.

    Frameshift: indel (len ref != len alt after stripping to SNV)
    Stop/missense: not determinable without reference; label as 'missense_variant'
    Synonymous: cannot determine without codon table; label as 'other'

    This is a best-effort approximation. Run the real snpEff pipeline when possible.
    """
    print(
        "[warn] Running without snpEff — using length-based effect approximation. "
        "For accurate per-SNP effect classes, install snpEff and use --snpeff-jar."
    )
    vcfs = sorted(Path(vcf_dir).glob(pattern))
    records = []
    for vcf_path in vcfs:
        opener = gzip.open(vcf_path, "rt") if str(vcf_path).endswith(".gz") else open(vcf_path)
        with opener as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                parts = line.strip().split("\t")
                if len(parts) < 5:
                    continue
                chrom, pos_s, _, ref, alt = parts[:5]
                try:
                    pos = int(pos_s)
                except ValueError:
                    continue
                gene = _pos_to_gene(chrom, pos)
                if gene is None:
                    continue
                snp_id = f"{gene}_{ref}{pos}{alt}"
                # Heuristic classification
                if alt == ".":
                    continue
                if len(ref) != len(alt):
                    effect = "frameshift_variant" if abs(len(ref) - len(alt)) % 3 != 0 else "missense_variant"
                elif len(ref) == 1:
                    effect = "missense_variant"  # Cannot tell without codon table
                else:
                    effect = "other"
                records.append({"snp_id": snp_id, "effect_class": effect})

    df = pd.DataFrame(records).drop_duplicates(subset="snp_id").reset_index(drop=True)
    df.to_parquet(output_parquet, index=False)
    print(f"Saved fallback annotations for {len(df)} SNPs to {output_parquet}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Annotate MTB VCFs with snpEff effect classes.")
    parser.add_argument("--vcf-dir", required=True, help="Directory containing VCF(.gz) files.")
    parser.add_argument("--output", required=True, help="Output parquet path.")
    parser.add_argument("--snpeff-jar", default=None, help="Path to snpEff.jar.")
    parser.add_argument(
        "--genome", default="Mycobacterium_tuberculosis_h37rv",
        help="snpEff genome name (default: Mycobacterium_tuberculosis_h37rv)."
    )
    parser.add_argument("--snpeff-config", default=None, help="Path to snpEff.config.")
    parser.add_argument("--pattern", default="*.vcf*", help="VCF glob pattern.")
    parser.add_argument(
        "--no-snpeff", action="store_true",
        help="Skip snpEff; use length-based fallback (less accurate)."
    )
    args = parser.parse_args()

    if args.no_snpeff or args.snpeff_jar is None:
        annotate_without_snpeff(args.vcf_dir, args.output, args.pattern)
    else:
        annotate_vcf_directory(
            vcf_dir=args.vcf_dir,
            snpeff_jar=args.snpeff_jar,
            genome=args.genome,
            output_parquet=args.output,
            snpeff_config=args.snpeff_config,
            pattern=args.pattern,
        )


if __name__ == "__main__":
    main()
