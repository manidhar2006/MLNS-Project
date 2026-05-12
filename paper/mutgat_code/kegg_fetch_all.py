#!/usr/bin/env python3
"""
Comprehensive KEGG data fetcher for Mycobacterium tuberculosis (mtu).

Downloads a superset of what `kegg_pathways.py` collected, namely:
  - All mtu pathways (name list + per-pathway detail + per-pathway gene list)
  - All mtu genes -> pathway bulk mapping (link/pathway/mtu)
  - All mtu genes -> KO bulk mapping (link/ko/mtu)
  - All mtu genes -> EC/enzyme bulk mapping (link/enzyme/mtu)
  - All mtu genes -> module bulk mapping (link/module/mtu)  (usually empty)
  - All mtu genes -> reaction bulk mapping (link/reaction/mtu)
  - Detailed gene records for an expanded AMR-relevant gene panel
  - Drug records for TB first+second line drugs (KEGG Drug DB)
  - Drug-target links (link/target/...) and drug-disease links
  - BRITE hierarchies: mtu00001 (KO), br08303 (drug targets),
    br08402 (diseases), br08901 (pathway hierarchy), br08907 (AMR genes)
  - Tuberculosis disease record (H00342)
  - Resistance-specific pathways (mtu01501, mtu01503, mtu05152)
  - KO records for AMR gene KOs

Writes a merged knowledge graph to kegg_data/tb_knowledge_graph_full.{json,pkl}
and a human summary to kegg_data/knowledge_graph_summary_full.txt.

Idempotent; every REST call is cached in kegg_data/cache/.
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests


KEGG_BASE = "https://rest.kegg.jp"
MTU = "mtu"


# --------------------------------------------------------------------------
# Expanded AMR / TB gene panel
# Grouped by drug. Rv IDs are H37Rv locus tags used by KEGG.
# Core set matches kegg_pathways.py; extras are widely cited in the WHO AMR
# mutation catalogue (2023) and the CRyPTIC paper supplements.
# --------------------------------------------------------------------------
AMR_GENES: Dict[str, Tuple[str, List[str], str]] = {
    # Isoniazid
    "katG":  ("Rv1908c", ["INH"],         "Catalase-peroxidase (INH activation)"),
    "inhA":  ("Rv1484",  ["INH"],         "Enoyl-ACP reductase (INH target)"),
    "fabG1": ("Rv1483",  ["INH"],         "3-oxoacyl-ACP reductase"),
    "ahpC":  ("Rv2428",  ["INH"],         "Alkyl hydroperoxide reductase"),
    "ndh":   ("Rv1854c", ["INH"],         "NADH dehydrogenase"),
    "iniA":  ("Rv0342",  ["INH", "ETH"],  "Isoniazid-inducible protein IniA"),
    "iniB":  ("Rv0341",  ["INH", "ETH"],  "Isoniazid-inducible protein IniB"),
    "iniC":  ("Rv0343",  ["INH", "ETH"],  "Isoniazid-inducible protein IniC"),
    "kasA":  ("Rv2245",  ["INH"],         "beta-ketoacyl-ACP synthase A"),
    "mshA":  ("Rv0486",  ["INH", "ETH"],  "Mycothiol biosynthesis glycosyltransferase"),
    "furA":  ("Rv1909c", ["INH"],         "Ferric uptake regulator (katG co-regulator)"),
    "nat":   ("Rv3566c", ["INH"],         "Arylamine N-acetyltransferase"),

    # Ethionamide (shares activator with INH)
    "ethA":  ("Rv3854c", ["ETH"],         "FAD-containing monooxygenase (ETH activation)"),
    "ethR":  ("Rv3855",  ["ETH"],         "TetR-family repressor of ethA"),
    "mymA":  ("Rv3083",  ["ETH"],         "Monooxygenase involved in ETH activation"),

    # Rifampicin
    "rpoB":  ("Rv0667",  ["RIF"],         "RNA polymerase beta subunit"),
    "rpoC":  ("Rv0668",  ["RIF"],         "RNA polymerase beta' subunit"),
    "rpoA":  ("Rv3457c", ["RIF"],         "RNA polymerase alpha subunit"),

    # Ethambutol
    "embA":  ("Rv3794",  ["EMB"],         "Arabinosyltransferase A"),
    "embB":  ("Rv3795",  ["EMB"],         "Arabinosyltransferase B"),
    "embC":  ("Rv3793",  ["EMB"],         "Arabinosyltransferase C"),
    "embR":  ("Rv1267c", ["EMB"],         "Transcriptional activator of embCAB"),
    "manB":  ("Rv3264c", ["EMB"],         "Phosphomannomutase"),
    "rmlD":  ("Rv3266c", ["EMB"],         "dTDP-4-dehydrorhamnose reductase"),
    "aftA":  ("Rv3792",  ["EMB"],         "Arabinofuranosyltransferase A"),
    "ubiA":  ("Rv3806c", ["EMB"],         "Decaprenyl-phosphate 5-phosphoribosyltransferase"),

    # Pyrazinamide
    "pncA":  ("Rv2043c", ["PZA"],         "Pyrazinamidase/nicotinamidase"),
    "rpsA":  ("Rv1630",  ["PZA"],         "30S ribosomal protein S1 (PZA-related)"),
    "panD":  ("Rv3601c", ["PZA"],         "Aspartate 1-decarboxylase"),
    "clpC1": ("Rv3596c", ["PZA"],         "ATP-dependent Clp protease (PZA target proxy)"),

    # Fluoroquinolones
    "gyrA":  ("Rv0006",  ["FLQ"],         "DNA gyrase subunit A"),
    "gyrB":  ("Rv0005",  ["FLQ"],         "DNA gyrase subunit B"),

    # Aminoglycosides / cyclic peptides
    "rrs":   ("Rvnr01",  ["STR", "AMK", "KAN", "CAP"], "16S ribosomal RNA"),
    "rrl":   ("Rvnr02",  ["LZD"],         "23S ribosomal RNA"),
    "eis":   ("Rv2416c", ["KAN", "AMK"],  "Enhanced intracellular survival, aminoglycoside acetyltransferase"),
    "gidB":  ("Rv3919c", ["STR"],         "16S rRNA methyltransferase"),
    "tlyA":  ("Rv1694",  ["CAP"],         "23S rRNA 2'-O-methyltransferase (CAP susceptibility)"),
    "whiB7": ("Rv3197A", ["INH", "STR", "KAN", "AMK"], "Intrinsic antibiotic resistance regulator"),

    # Linezolid / oxazolidinones
    "rplC":  ("Rv0701",  ["LZD"],         "50S ribosomal protein L3"),
    "rplD":  ("Rv0702",  ["LZD"],         "50S ribosomal protein L4"),

    # Bedaquiline / clofazimine (efflux + target)
    "atpE":  ("Rv1305",  ["BDQ"],         "ATP synthase F0 subunit c (BDQ target)"),
    "mmpR5": ("Rv0678",  ["BDQ", "CFZ"],  "MarR-family regulator of mmpS5-mmpL5"),
    "pepQ":  ("Rv2535c", ["BDQ", "CFZ"],  "Proline aminopeptidase (BDQ/CFZ resistance)"),
    "atpB":  ("Rv1304",  ["BDQ"],         "ATP synthase F0 subunit a"),

    # Delamanid / pretomanid (F420-dependent bioactivation)
    "ddn":   ("Rv3547",  ["DLM", "PTM"],  "Deazaflavin-dependent nitroreductase"),
    "fgd1":  ("Rv0407",  ["DLM", "PTM"],  "F420-dependent glucose-6-phosphate dehydrogenase"),
    "fbiA":  ("Rv3261",  ["DLM", "PTM"],  "F420 biosynthesis"),
    "fbiB":  ("Rv3262",  ["DLM", "PTM"],  "F420 biosynthesis"),
    "fbiC":  ("Rv1173",  ["DLM", "PTM"],  "F420 biosynthesis"),
    "fbiD":  ("Rv2983",  ["DLM", "PTM"],  "F420 biosynthesis"),
}

# KEGG Drug (D-number) IDs for TB drugs.
# Verified 2026-04-22 against `rest.kegg.jp/find/drug/...`.
TB_DRUGS: Dict[str, str] = {
    "INH": "D00346",  # Isoniazid
    "RIF": "D00211",  # Rifampin / Rifampicin
    "RFB": "D00424",  # Rifabutin
    "RPT": "D00879",  # Rifapentine
    "EMB": "D07925",  # Ethambutol (INN)
    "PZA": "D00144",  # Pyrazinamide
    "ETH": "D00591",  # Ethionamide
    "PTH": "D01195",  # Prothionamide
    "STR": "D01350",  # Streptomycin sulfate
    "KAN": "D00866",  # Kanamycin sulfate
    "AMK": "D02543",  # Amikacin (INN)
    "CAP": "D00135",  # Capreomycin sulfate
    "LZD": "D00947",  # Linezolid
    "BDQ": "D09872",  # Bedaquiline (INN)
    "DLM": "D09785",  # Delamanid
    "PTM": "D10722",  # Pretomanid
    "CFZ": "D00278",  # Clofazimine
    "LEV": "D08120",  # Levofloxacin (INN)
    "MXF": "D08237",  # Moxifloxacin (INN)
    "OFX": "D00453",  # Ofloxacin
    "CIP": "D00186",  # Ciprofloxacin
    "CYS": "D00877",  # Cycloserine
    "PAS": "D03368",  # Calcium para-aminosalicylate (PAS)
}

# Additional resistance/disease pathways that the original fetcher missed.
EXTRA_PATHWAYS: List[str] = [
    "mtu01501",  # beta-Lactam resistance
    "mtu01503",  # Cationic antimicrobial peptide (CAMP) resistance
    "mtu05152",  # Tuberculosis (host pathogenesis)
    "mtu00061",  # Fatty acid biosynthesis (FAS-II; mycolic acid precursor)
    "mtu00071",  # Fatty acid degradation
    "mtu00072",  # Synthesis and degradation of ketone bodies
    "mtu00780",  # Biotin metabolism
    "mtu00785",  # Lipoic acid metabolism
    "mtu00250",  # Alanine, aspartate, glutamate metabolism
    "mtu00300",  # Lysine biosynthesis
    "mtu00540",  # Lipopolysaccharide biosynthesis
    "mtu00550",  # Peptidoglycan biosynthesis
    "mtu00561",  # Glycerolipid metabolism
    "mtu00563",  # Glycosylphosphatidylinositol anchor biosynthesis
    "mtu00564",  # Glycerophospholipid metabolism
    "mtu01053",  # Biosynthesis of siderophore group nonribosomal peptides
    "mtu01130",  # Biosynthesis of antibiotics
    "mtu02020",  # Two-component system
    "mtu02024",  # Quorum sensing
    "mtu02025",  # Biofilm formation
    "mtu03440",  # Homologous recombination
]

# BRITE hierarchies of interest.
BRITE_FILES: List[str] = [
    "br:br08402",  # Human diseases
    "br:br08303",  # Drug targets
    "br:br08901",  # KEGG pathway maps
    "br:br08902",  # BRITE hierarchy files
    "br:br08907",  # Antimicrobial resistance genes
    "br:mtu00001", # KEGG Orthology hierarchy for M. tuberculosis
    "br:mtu01000", # Enzymes hierarchy for mtu
    "br:mtu03036", # Chromosome and associated proteins for mtu
]

# KEGG Disease record for TB
TB_DISEASES: List[str] = ["H00342"]   # Tuberculosis


# ==========================================================================
# Fetcher
# ==========================================================================
@dataclass
class Fetcher:
    cache_dir: Path
    delay: float = 0.3
    verbose: bool = True

    def __post_init__(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._session = requests.Session()

    def _cache_name(self, endpoint: str) -> Path:
        safe = (endpoint
                .replace("/", "_")
                .replace(":", "_")
                .replace("+", "_plus_"))
        return self.cache_dir / f"{safe}.txt"

    def get(self, endpoint: str, force: bool = False) -> Optional[str]:
        cache = self._cache_name(endpoint)
        if cache.exists() and not force and cache.stat().st_size > 0:
            return cache.read_text()
        url = f"{KEGG_BASE}/{endpoint}"
        try:
            if self.verbose:
                print(f"  GET {endpoint}")
            time.sleep(self.delay)
            r = self._session.get(url, timeout=60)
            if r.status_code == 200 and r.text.strip():
                cache.write_text(r.text)
                return r.text
            elif r.status_code == 404:
                cache.write_text("")  # negative cache
                return None
            else:
                print(f"    HTTP {r.status_code} for {endpoint}")
                return None
        except Exception as e:
            print(f"    error fetching {endpoint}: {e}")
            return None


# ==========================================================================
# Parsers
# ==========================================================================
def parse_flat(data: Optional[str]) -> Dict[str, List[str]]:
    """Parse KEGG flat-file 'ENTRY' format into section -> list of lines."""
    out: Dict[str, List[str]] = defaultdict(list)
    if not data:
        return out
    current = None
    for line in data.splitlines():
        if not line or line.startswith("///"):
            continue
        if not line.startswith(" "):
            head = line[:12].strip()
            if head:
                current = head
                rest = line[12:].strip()
                if rest:
                    out[current].append(rest)
        else:
            if current:
                out[current].append(line[12:].strip() if len(line) > 12 else line.strip())
    return dict(out)


def parse_gene(data: Optional[str]) -> Dict:
    sec = parse_flat(data)
    pathways: List[Dict[str, str]] = []
    for p in sec.get("PATHWAY", []):
        parts = p.split(None, 1)
        pathways.append({"id": parts[0], "name": parts[1] if len(parts) > 1 else ""})
    ko_entries: List[Dict[str, str]] = []
    for o in sec.get("ORTHOLOGY", []):
        parts = o.split(None, 1)
        ko_entries.append({"id": parts[0], "name": parts[1] if len(parts) > 1 else ""})
    return {
        "entry": (sec.get("ENTRY", [""])[0].split()[0] if sec.get("ENTRY") else ""),
        "symbol": sec.get("SYMBOL", [""])[0] if sec.get("SYMBOL") else "",
        "name": sec.get("NAME", [""])[0] if sec.get("NAME") else "",
        "definition": sec.get("DEFINITION", [""])[0] if sec.get("DEFINITION") else "",
        "orthology": ko_entries,
        "pathways": pathways,
        "brite": sec.get("BRITE", []),
        "motif": sec.get("MOTIF", []),
        "dblinks": sec.get("DBLINKS", []),
        "position": sec.get("POSITION", [""])[0] if sec.get("POSITION") else "",
        "aaseq_len": _len_from_aaseq(sec.get("AASEQ")),
    }


def _len_from_aaseq(lines: Optional[List[str]]) -> Optional[int]:
    if not lines:
        return None
    first = lines[0]
    m = re.match(r"(\d+)", first.strip())
    return int(m.group(1)) if m else None


def parse_pathway(data: Optional[str]) -> Dict:
    sec = parse_flat(data)
    genes: List[Dict[str, str]] = []
    for g in sec.get("GENE", []):
        if not g or g.startswith("["):
            continue
        parts = g.split(None, 1)
        gene_id = parts[0].replace(f"{MTU}:", "")
        genes.append({"id": gene_id, "name": parts[1] if len(parts) > 1 else ""})
    return {
        "entry": (sec.get("ENTRY", [""])[0].split()[0] if sec.get("ENTRY") else ""),
        "name": sec.get("NAME", [""])[0] if sec.get("NAME") else "",
        "description": " ".join(sec.get("DESCRIPTION", [])),
        "class": sec.get("CLASS", [""])[0] if sec.get("CLASS") else "",
        "module": sec.get("MODULE", []),
        "drug": sec.get("DRUG", []),
        "disease": sec.get("DISEASE", []),
        "genes": genes,
        "compounds": sec.get("COMPOUND", []),
        "reference_count": len(sec.get("REFERENCE", [])),
    }


def parse_drug(data: Optional[str]) -> Dict:
    sec = parse_flat(data)
    return {
        "entry": (sec.get("ENTRY", [""])[0].split()[0] if sec.get("ENTRY") else ""),
        "name": sec.get("NAME", [""])[0] if sec.get("NAME") else "",
        "formula": sec.get("FORMULA", [""])[0] if sec.get("FORMULA") else "",
        "class": sec.get("CLASS", []),
        "target": sec.get("TARGET", []),
        "metabolism": sec.get("METABOLISM", []),
        "remark": sec.get("REMARK", []),
        "efficacy": sec.get("EFFICACY", []),
        "dblinks": sec.get("DBLINKS", []),
    }


def parse_disease(data: Optional[str]) -> Dict:
    sec = parse_flat(data)
    return {
        "entry": (sec.get("ENTRY", [""])[0].split()[0] if sec.get("ENTRY") else ""),
        "name": sec.get("NAME", [""])[0] if sec.get("NAME") else "",
        "description": " ".join(sec.get("DESCRIPTION", [])),
        "category": sec.get("CATEGORY", []),
        "pathogen": sec.get("PATHOGEN", []),
        "pathway": sec.get("PATHWAY", []),
        "gene": sec.get("GENE", []),
        "drug": sec.get("DRUG", []),
    }


def parse_ko(data: Optional[str]) -> Dict:
    sec = parse_flat(data)
    return {
        "entry": (sec.get("ENTRY", [""])[0].split()[0] if sec.get("ENTRY") else ""),
        "name": sec.get("NAME", [""])[0] if sec.get("NAME") else "",
        "definition": sec.get("DEFINITION", [""])[0] if sec.get("DEFINITION") else "",
        "pathway": sec.get("PATHWAY", []),
        "module": sec.get("MODULE", []),
        "brite": sec.get("BRITE", []),
        "disease": sec.get("DISEASE", []),
    }


def parse_bulk_link(data: Optional[str]) -> Dict[str, List[str]]:
    """Parse 'link' output (two columns). Returns {lhs: [rhs,...]}."""
    out: Dict[str, List[str]] = defaultdict(list)
    if not data:
        return {}
    for line in data.strip().split("\n"):
        parts = line.split("\t")
        if len(parts) == 2:
            out[parts[0]].append(parts[1])
    return dict(out)


# ==========================================================================
# Fetch plan
# ==========================================================================
def fetch_all(fetcher: Fetcher) -> Dict:
    print("\n[1] Listing all mtu pathways...")
    pwy_list_raw = fetcher.get(f"list/pathway/{MTU}")
    pathways: List[Tuple[str, str]] = []
    if pwy_list_raw:
        for line in pwy_list_raw.strip().split("\n"):
            if "\t" in line:
                pid, name = line.split("\t", 1)
                pathways.append((pid, name))
    print(f"  found {len(pathways)} pathways")

    print("\n[2] Detailed pathway records + pathway gene lists...")
    pathway_info: Dict[str, Dict] = {}
    pathway_to_genes: Dict[str, List[str]] = {}
    wanted_ids = set(p[0] for p in pathways) | set(EXTRA_PATHWAYS)
    for pid in sorted(wanted_ids):
        pdata = fetcher.get(f"get/{pid}")
        pathway_info[pid] = parse_pathway(pdata)
        gdata = fetcher.get(f"link/{MTU}/{pid}")
        genes = []
        if gdata:
            for line in gdata.strip().split("\n"):
                parts = line.split("\t")
                if len(parts) == 2:
                    genes.append(parts[1].replace(f"{MTU}:", ""))
        pathway_to_genes[pid] = genes

    print("\n[3] Bulk cross-reference mappings...")
    bulk = {
        "gene_to_pathway": parse_bulk_link(fetcher.get(f"link/pathway/{MTU}")),
        "gene_to_ko":      parse_bulk_link(fetcher.get(f"link/ko/{MTU}")),
        "gene_to_enzyme":  parse_bulk_link(fetcher.get(f"link/enzyme/{MTU}")),
        "gene_to_module":  parse_bulk_link(fetcher.get(f"link/module/{MTU}")),
        "gene_to_reaction":parse_bulk_link(fetcher.get(f"link/reaction/{MTU}")),
    }
    for k, v in bulk.items():
        print(f"  {k}: {len(v)} source nodes")

    print(f"\n[4] Detailed gene records for AMR panel ({len(AMR_GENES)} genes)...")
    gene_info: Dict[str, Dict] = {}
    gene_to_pathways: Dict[str, List[str]] = defaultdict(list)
    drug_to_genes: Dict[str, List[str]] = defaultdict(list)
    for gene_name, (rv, drugs, desc) in AMR_GENES.items():
        data = fetcher.get(f"get/{MTU}:{rv}")
        info = parse_gene(data)
        info["standard_name"] = gene_name
        info["rv_id"] = rv
        info["associated_drugs"] = drugs
        info["description"] = desc
        gene_info[gene_name] = info
        # direct pathways from gene page
        direct_pathways = [p["id"] for p in info["pathways"]]
        # augment via bulk link
        bulk_pathways = [
            pp.replace("path:", "")
            for pp in bulk["gene_to_pathway"].get(f"{MTU}:{rv}", [])
        ]
        merged = []
        for p in direct_pathways + bulk_pathways:
            if p not in merged:
                merged.append(p)
        gene_to_pathways[gene_name] = merged
        for drug in drugs:
            drug_to_genes[drug].append(gene_name)

    print("\n[5] KO records for AMR panel orthologs...")
    ko_info: Dict[str, Dict] = {}
    seen_kos: set = set()
    for gene_name, info in gene_info.items():
        for ortho in info["orthology"]:
            ko_id = ortho["id"]
            if ko_id and ko_id.startswith("K") and ko_id not in seen_kos:
                seen_kos.add(ko_id)
                data = fetcher.get(f"get/ko:{ko_id}")
                ko_info[ko_id] = parse_ko(data)

    print(f"\n[6] Drug records ({len(TB_DRUGS)} drugs)...")
    drug_info: Dict[str, Dict] = {}
    drug_to_targets: Dict[str, List[str]] = {}
    drug_to_pathway_links: Dict[str, List[str]] = {}
    drug_to_disease_links: Dict[str, List[str]] = {}
    for short, did in TB_DRUGS.items():
        data = fetcher.get(f"get/{did}")
        drug_info[short] = parse_drug(data)
        drug_info[short]["short_name"] = short
        drug_info[short]["kegg_id"] = did
        # cross references
        for kind, key in [
            ("target", f"link/target/dr:{did}"),
            ("pathway", f"link/pathway/dr:{did}"),
            ("disease", f"link/disease/dr:{did}"),
        ]:
            raw = fetcher.get(key)
            xs: List[str] = []
            if raw:
                for line in raw.strip().split("\n"):
                    parts = line.split("\t")
                    if len(parts) == 2:
                        xs.append(parts[1])
            if kind == "target":
                drug_to_targets[short] = xs
            elif kind == "pathway":
                drug_to_pathway_links[short] = xs
            elif kind == "disease":
                drug_to_disease_links[short] = xs

    print(f"\n[7] BRITE hierarchy files ({len(BRITE_FILES)})...")
    brite: Dict[str, Optional[str]] = {}
    for b in BRITE_FILES:
        brite[b] = fetcher.get(f"get/{b}")

    print(f"\n[8] TB disease records...")
    disease_info: Dict[str, Dict] = {}
    for d in TB_DISEASES:
        data = fetcher.get(f"get/{d}")
        disease_info[d] = parse_disease(data)

    # --------- Build drug -> pathways mapping grounded in KEGG data ---------
    # Strategy: union of (a) KEGG's drug->pathway links (often sparse for drugs),
    # (b) pathways that contain any of the gene targets for that drug,
    # (c) fallback manual mapping of mechanism-level pathways.
    drug_to_pathways_grounded: Dict[str, List[str]] = defaultdict(list)
    for short, genes in drug_to_genes.items():
        seen: set = set()
        for pid in drug_to_pathway_links.get(short, []):
            clean = pid.replace("path:", "")
            if clean not in seen:
                seen.add(clean)
                drug_to_pathways_grounded[short].append(clean)
        for g in genes:
            for p in gene_to_pathways.get(g, []):
                if p not in seen:
                    seen.add(p)
                    drug_to_pathways_grounded[short].append(p)

    kg: Dict = {
        "metadata": {
            "organism": "Mycobacterium tuberculosis H37Rv",
            "organism_code": MTU,
            "n_pathways": len(pathway_info),
            "n_genes_panel": len(gene_info),
            "n_drugs": len(drug_info),
            "n_kos_panel": len(ko_info),
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "pathway_list": pathways,
        "pathway_info": pathway_info,
        "pathway_to_genes": pathway_to_genes,
        "gene_info": gene_info,
        "gene_to_pathways": dict(gene_to_pathways),
        "drug_to_pathways_grounded": dict(drug_to_pathways_grounded),
        "drug_to_genes": dict(drug_to_genes),
        "drug_info": drug_info,
        "drug_to_targets": drug_to_targets,
        "drug_to_pathway_links": drug_to_pathway_links,
        "drug_to_disease_links": drug_to_disease_links,
        "ko_info": ko_info,
        "brite": brite,
        "disease_info": disease_info,
        "bulk_links": bulk,
    }
    return kg


# ==========================================================================
# Output helpers
# ==========================================================================
def save(kg: Dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "tb_knowledge_graph_full.json"
    pkl_path = out_dir / "tb_knowledge_graph_full.pkl"
    txt_path = out_dir / "knowledge_graph_summary_full.txt"

    with json_path.open("w") as f:
        json.dump(kg, f, indent=2, default=str)
    with pkl_path.open("wb") as f:
        pickle.dump(kg, f)

    lines: List[str] = []
    lines.append("TB (mtu) KEGG KNOWLEDGE GRAPH — FULL")
    lines.append("=" * 70)
    md = kg["metadata"]
    for k in ("organism", "organism_code", "fetched_at", "n_pathways",
              "n_genes_panel", "n_drugs", "n_kos_panel"):
        lines.append(f"{k:.<35} {md[k]}")
    lines.append("")
    lines.append("PATHWAYS")
    lines.append("-" * 70)
    pwy_sizes = [
        (pid, len(kg["pathway_to_genes"].get(pid, [])))
        for pid in sorted(kg["pathway_info"])
    ]
    for pid, n in sorted(pwy_sizes, key=lambda kv: -kv[1]):
        name = kg["pathway_info"][pid].get("name") or ""
        lines.append(f"  {pid:<10} {n:>4} genes  {name}")
    lines.append("")
    lines.append("AMR GENE PANEL")
    lines.append("-" * 70)
    for g, info in sorted(kg["gene_info"].items()):
        ps = kg["gene_to_pathways"].get(g, [])
        ko = ",".join(o["id"] for o in info.get("orthology", []))
        lines.append(
            f"  {g:<8} {info['rv_id']:<8} drugs={','.join(info['associated_drugs']):<20} "
            f"KO={ko:<12} pathways={len(ps)}"
        )
    lines.append("")
    lines.append("DRUGS")
    lines.append("-" * 70)
    for short, d in sorted(kg["drug_info"].items()):
        targets = kg["drug_to_targets"].get(short, [])
        pwys = kg["drug_to_pathways_grounded"].get(short, [])
        lines.append(
            f"  {short:<4} {d.get('kegg_id','')}  {d.get('name','')[:40]:<40} "
            f"targets={len(targets):>3} pathways={len(pwys):>3}"
        )
    lines.append("")
    lines.append("DRUG -> PATHWAYS (grounded)")
    lines.append("-" * 70)
    for short, pwys in sorted(kg["drug_to_pathways_grounded"].items()):
        lines.append(f"  {short}: {', '.join(sorted(pwys))}")
    txt_path.write_text("\n".join(lines))

    print(f"\nWrote:\n  {json_path}\n  {pkl_path}\n  {txt_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="./kegg_data")
    p.add_argument("--delay", type=float, default=0.25)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    out = Path(args.output)
    cache = out / "cache"
    fetcher = Fetcher(cache_dir=cache, delay=args.delay, verbose=not args.quiet)
    kg = fetch_all(fetcher)
    save(kg, out)


if __name__ == "__main__":
    main()
