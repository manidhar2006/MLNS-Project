#!/usr/bin/env python3
"""
KEGG Pathway Data Fetcher for TB Drug Resistance
Downloads pathway information and stores locally for HGAT-AMR model

Usage:
    python fetch_kegg_pathways.py --output ./kegg_data
"""

import os
import json
import time
import argparse
import requests
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import pickle


class KEGGFetcher:
    """Fetch TB drug pathway data from KEGG API"""
    
    def __init__(self, cache_dir: str = "./kegg_data", delay: float = 0.5):
        """
        Args:
            cache_dir: Directory to store downloaded data
            delay: Delay between API calls (seconds) to respect rate limits
        """
        self.base_url = "http://rest.kegg.jp"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay
        
        # TB organism code
        self.mtb_org = "mtu"  # M. tuberculosis H37Rv
        
        print(f"✓ KEGG Fetcher initialized")
        print(f"  Cache directory: {self.cache_dir}")
        print(f"  Organism: {self.mtb_org} (M. tuberculosis H37Rv)")
    
    def _get(self, endpoint: str) -> Optional[str]:
        """Make GET request with caching and rate limiting"""
        url = f"{self.base_url}/{endpoint}"
        cache_file = self.cache_dir / f"{endpoint.replace('/', '_')}.txt"
        
        # Check cache first
        if cache_file.exists():
            print(f"  ↻ Loading from cache: {endpoint}")
            return cache_file.read_text()
        
        # Fetch from API
        print(f"  ↓ Downloading: {endpoint}")
        try:
            time.sleep(self.delay)  # Rate limiting
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                cache_file.write_text(response.text)
                return response.text
            else:
                print(f"    ✗ Failed (status {response.status_code})")
                return None
                
        except Exception as e:
            print(f"    ✗ Error: {e}")
            return None
    
    def get_organism_info(self) -> Dict:
        """Get M. tuberculosis organism information"""
        print("\n[1] Fetching organism information...")
        data = self._get(f"info/{self.mtb_org}")
        
        info = {"organism": self.mtb_org, "name": "Mycobacterium tuberculosis H37Rv"}
        if data:
            for line in data.split('\n'):
                if line.startswith("mtu"):
                    parts = line.split(None, 1)
                    if len(parts) > 1:
                        info["name"] = parts[1].strip()
        
        return info
    
    def get_gene_info(self, gene_id: str) -> Dict:
        """Get detailed information for a gene"""
        data = self._get(f"get/{self.mtb_org}:{gene_id}")
        
        info = {
            "kegg_id": gene_id,
            "name": "",
            "definition": "",
            "pathways": [],
            "position": "",
            "sequence": ""
        }
        
        if not data:
            return info
        
        current_section = None
        for line in data.split('\n'):
            if line.startswith("NAME"):
                info["name"] = line.split(None, 1)[1].strip() if len(line.split(None, 1)) > 1 else ""
            elif line.startswith("DEFINITION"):
                info["definition"] = line.split(None, 1)[1].strip() if len(line.split(None, 1)) > 1 else ""
            elif line.startswith("PATHWAY"):
                current_section = "pathway"
                pathway_part = line.split(None, 1)[1].strip() if len(line.split(None, 1)) > 1 else ""
                if pathway_part:
                    parts = pathway_part.split(None, 1)
                    info["pathways"].append({
                        "id": parts[0],
                        "name": parts[1] if len(parts) > 1 else ""
                    })
            elif line.startswith("POSITION"):
                info["position"] = line.split(None, 1)[1].strip() if len(line.split(None, 1)) > 1 else ""
            elif line.startswith(" " * 12) and current_section == "pathway":
                pathway_part = line.strip()
                if pathway_part:
                    parts = pathway_part.split(None, 1)
                    info["pathways"].append({
                        "id": parts[0],
                        "name": parts[1] if len(parts) > 1 else ""
                    })
            elif not line.startswith(" "):
                current_section = None
        
        return info
    
    def get_pathway_info(self, pathway_id: str) -> Dict:
        """Get detailed pathway information"""
        data = self._get(f"get/{pathway_id}")
        
        info = {
            "pathway_id": pathway_id,
            "name": "",
            "description": "",
            "class": "",
            "genes": [],
            "compounds": [],
            "drugs": []
        }
        
        if not data:
            return info
        
        current_section = None
        for line in data.split('\n'):
            if line.startswith("NAME"):
                info["name"] = line.split(None, 1)[1].strip() if len(line.split(None, 1)) > 1 else ""
            elif line.startswith("DESCRIPTION"):
                info["description"] = line.split(None, 1)[1].strip() if len(line.split(None, 1)) > 1 else ""
            elif line.startswith("CLASS"):
                info["class"] = line.split(None, 1)[1].strip() if len(line.split(None, 1)) > 1 else ""
            elif line.startswith("GENE"):
                current_section = "gene"
                gene_part = line.split(None, 1)[1].strip() if len(line.split(None, 1)) > 1 else ""
                if gene_part:
                    parts = gene_part.split(None, 1)
                    info["genes"].append({
                        "id": parts[0].replace(f"{self.mtb_org}:", ""),
                        "name": parts[1] if len(parts) > 1 else ""
                    })
            elif line.startswith(" " * 12) and current_section == "gene":
                gene_part = line.strip()
                if gene_part and not gene_part.startswith("["):
                    parts = gene_part.split(None, 1)
                    info["genes"].append({
                        "id": parts[0].replace(f"{self.mtb_org}:", ""),
                        "name": parts[1] if len(parts) > 1 else ""
                    })
            elif not line.startswith(" "):
                current_section = None
        
        return info
    
    def get_pathway_genes(self, pathway_id: str) -> List[str]:
        """Get list of genes in a pathway"""
        data = self._get(f"link/{self.mtb_org}/{pathway_id}")
        
        genes = []
        if data:
            for line in data.strip().split('\n'):
                parts = line.split('\t')
                if len(parts) == 2:
                    gene_id = parts[1].replace(f'{self.mtb_org}:', '')
                    genes.append(gene_id)
        
        return genes
    
    def search_pathways_by_keyword(self, keyword: str) -> List[Dict]:
        """Search for pathways by keyword"""
        data = self._get(f"find/pathway/{keyword}")
        
        pathways = []
        if data:
            for line in data.strip().split('\n'):
                if '\t' in line:
                    parts = line.split('\t', 1)
                    pathways.append({
                        "id": parts[0].split(':')[1] if ':' in parts[0] else parts[0],
                        "name": parts[1] if len(parts) > 1 else ""
                    })
        
        return pathways


def build_tb_drug_knowledge_graph(fetcher: KEGGFetcher) -> Dict:
    """
    Build comprehensive TB drug resistance knowledge graph
    
    Returns:
        Dictionary containing:
        - gene_info: Detailed info for each resistance gene
        - pathway_info: Detailed info for each pathway
        - gene_to_pathways: Mapping of genes to pathways
        - pathway_to_genes: Mapping of pathways to genes
        - drug_to_pathways: Known drug-pathway associations
        - drug_to_genes: Known drug-gene associations
    """
    
    # ========================================================================
    # 1. Define MTB resistance genes and their KEGG IDs
    # ========================================================================
    
    MTB_RESISTANCE_GENES = {
        # Gene name: (KEGG ID, Drug association, Description)
        "katG":  ("Rv1908c", ["INH"], "Catalase-peroxidase (INH activation)"),
        "inhA":  ("Rv1484",  ["INH"], "Enoyl-ACP reductase (INH target)"),
        "fabG1": ("Rv1483",  ["INH"], "3-oxoacyl-ACP reductase"),
        "ahpC":  ("Rv2428",  ["INH"], "Alkyl hydroperoxide reductase"),
        
        "rpoB":  ("Rv0667",  ["RIF"], "RNA polymerase beta subunit"),
        "rpoC":  ("Rv0668",  ["RIF"], "RNA polymerase beta' subunit"),
        
        "embB":  ("Rv3795",  ["EMB"], "Arabinosyltransferase B"),
        "embA":  ("Rv3794",  ["EMB"], "Arabinosyltransferase A"),
        "embC":  ("Rv3793",  ["EMB"], "Arabinosyltransferase C"),
        
        "pncA":  ("Rv2043c", ["PZA"], "Pyrazinamidase/nicotinamidase"),
        
        "gyrA":  ("Rv0006",  ["FLQ"], "DNA gyrase subunit A"),
        "gyrB":  ("Rv0005",  ["FLQ"], "DNA gyrase subunit B"),
        
        "rrs":   ("Rvnr01", ["STR", "AMK", "KAN"], "16S ribosomal RNA"),
        "rrl":   ("Rvnr02", ["LNZ"], "23S ribosomal RNA"),
        
        "eis":   ("Rv2416c", ["KAN", "AMK"], "Enhanced intracellular survival"),
        "gidB":  ("Rv3919c", ["STR"], "Glucose-inhibited division protein"),
        "tlyA":  ("Rv1694",  ["CAP"], "Hemolysin"),
        
        "iniB":  ("Rv0341",  ["INH", "ETH"], "Isoniazid inductible gene"),
        "iniA":  ("Rv0342",  ["INH", "ETH"], "Isoniazid inductible gene"),
        "iniC":  ("Rv0343",  ["INH", "ETH"], "Isoniazid inductible gene"),
        
        "ndh":   ("Rv1854c", ["INH"], "NADH dehydrogenase"),
        "manB":  ("Rv3264c", ["EMB"], "Phosphomannomutase"),
        "rmlD":  ("Rv3266c", ["EMB"], "dTDP-4-dehydrorhamnose reductase"),
    }
    
    # ========================================================================
    # 2. Define TB drug pathway associations
    # ========================================================================
    
    TB_DRUG_PATHWAYS = {
        # First-line drugs
        "INH": {
            "pathways": ["mtu00650", "mtu01100", "mtu01110"],  
            "description": "Targets mycolic acid biosynthesis",
            "mechanism": "Inhibits InhA (enoyl-ACP reductase) after activation by KatG"
        },
        "RIF": {
            "pathways": ["mtu03020", "mtu01100"],
            "description": "Targets RNA synthesis",
            "mechanism": "Binds to RNA polymerase beta subunit (RpoB)"
        },
        "EMB": {
            "pathways": ["mtu00572", "mtu01100"],
            "description": "Targets arabinogalactan biosynthesis",
            "mechanism": "Inhibits arabinosyltransferases (EmbCAB)"
        },
        "PZA": {
            "pathways": ["mtu00770", "mtu00760", "mtu01100"],
            "description": "Targets energy metabolism",
            "mechanism": "Converted to pyrazinoic acid by PncA, disrupts membrane"
        },
        
        # Second-line drugs (for completeness)
        "FLQ": {  # Fluoroquinolones
            "pathways": ["mtu00230", "mtu03018"],
            "description": "Targets DNA replication",
            "mechanism": "Inhibits DNA gyrase (GyrA/GyrB)"
        },
        "STR": {  # Streptomycin
            "pathways": ["mtu03010"],
            "description": "Targets protein synthesis",
            "mechanism": "Binds 16S rRNA, causes misreading"
        },
        "KAN": {  # Kanamycin
            "pathways": ["mtu03010"],
            "description": "Targets protein synthesis",
            "mechanism": "Binds 16S rRNA"
        },
        "AMK": {  # Amikacin
            "pathways": ["mtu03010"],
            "description": "Targets protein synthesis",
            "mechanism": "Binds 16S rRNA"
        },
    }
    
    # ========================================================================
    # 3. Fetch data from KEGG
    # ========================================================================
    
    knowledge_graph = {
        "gene_info": {},
        "pathway_info": {},
        "gene_to_pathways": defaultdict(list),
        "pathway_to_genes": defaultdict(list),
        "drug_to_pathways": TB_DRUG_PATHWAYS,
        "drug_to_genes": defaultdict(list),
        "metadata": {
            "organism": "Mycobacterium tuberculosis H37Rv",
            "organism_code": "mtu",
            "n_genes": len(MTB_RESISTANCE_GENES),
            "n_drugs": len(TB_DRUG_PATHWAYS)
        }
    }
    
    # Get organism info
    print("\n" + "="*70)
    print("BUILDING TB DRUG RESISTANCE KNOWLEDGE GRAPH")
    print("="*70)
    
    org_info = fetcher.get_organism_info()
    knowledge_graph["metadata"]["organism_name"] = org_info.get("name", "")
    
    # ========================================================================
    # 4. Fetch gene information
    # ========================================================================
    
    print(f"\n[2] Fetching information for {len(MTB_RESISTANCE_GENES)} resistance genes...")
    
    for gene_name, (kegg_id, drugs, description) in MTB_RESISTANCE_GENES.items():
        print(f"\n  Processing {gene_name} ({kegg_id})...")
        
        # Get gene details
        gene_info = fetcher.get_gene_info(kegg_id)
        gene_info["standard_name"] = gene_name
        gene_info["associated_drugs"] = drugs
        gene_info["description"] = description
        
        knowledge_graph["gene_info"][gene_name] = gene_info
        
        # Map gene to pathways
        pathway_ids = [p["id"] for p in gene_info["pathways"]]
        knowledge_graph["gene_to_pathways"][gene_name] = pathway_ids
        
        # Map drugs to genes
        for drug in drugs:
            knowledge_graph["drug_to_genes"][drug].append(gene_name)
        
        print(f"    ✓ Found {len(pathway_ids)} pathways")
    
    # ========================================================================
    # 5. Fetch pathway information
    # ========================================================================
    
    # Collect all unique pathways
    all_pathways = set()
    for pathways in knowledge_graph["gene_to_pathways"].values():
        all_pathways.update(pathways)
    
    # Add drug-specific pathways
    for drug_info in TB_DRUG_PATHWAYS.values():
        all_pathways.update(drug_info["pathways"])
    
    print(f"\n[3] Fetching information for {len(all_pathways)} pathways...")
    
    for pathway_id in sorted(all_pathways):
        print(f"\n  Processing pathway {pathway_id}...")
        
        pathway_info = fetcher.get_pathway_info(pathway_id)
        knowledge_graph["pathway_info"][pathway_id] = pathway_info
        
        # Get genes in this pathway
        pathway_genes = fetcher.get_pathway_genes(pathway_id)
        knowledge_graph["pathway_to_genes"][pathway_id] = pathway_genes
        
        print(f"    ✓ Found {len(pathway_genes)} genes")
    
    # ========================================================================
    # 6. Enrich with pathway-gene relationships
    # ========================================================================
    
    print("\n[4] Enriching gene-pathway relationships...")
    
    for pathway_id, genes in knowledge_graph["pathway_to_genes"].items():
        for gene_kegg_id in genes:
            # Find which of our resistance genes is in this pathway
            for gene_name, (kegg_id, _, _) in MTB_RESISTANCE_GENES.items():
                if kegg_id == gene_kegg_id:
                    if pathway_id not in knowledge_graph["gene_to_pathways"][gene_name]:
                        knowledge_graph["gene_to_pathways"][gene_name].append(pathway_id)
    
    # ========================================================================
    # 7. Generate statistics
    # ========================================================================
    
    stats = {
        "total_genes": len(knowledge_graph["gene_info"]),
        "total_pathways": len(knowledge_graph["pathway_info"]),
        "total_drugs": len(knowledge_graph["drug_to_pathways"]),
        "genes_with_pathways": sum(1 for v in knowledge_graph["gene_to_pathways"].values() if v),
        "avg_pathways_per_gene": sum(len(v) for v in knowledge_graph["gene_to_pathways"].values()) / 
                                  len(knowledge_graph["gene_to_pathways"]) if knowledge_graph["gene_to_pathways"] else 0,
        "avg_genes_per_pathway": sum(len(v) for v in knowledge_graph["pathway_to_genes"].values()) / 
                                 len(knowledge_graph["pathway_to_genes"]) if knowledge_graph["pathway_to_genes"] else 0,
    }
    
    knowledge_graph["statistics"] = stats
    
    # Print summary
    print("\n" + "="*70)
    print("KNOWLEDGE GRAPH SUMMARY")
    print("="*70)
    print(f"  Genes:                    {stats['total_genes']}")
    print(f"  Pathways:                 {stats['total_pathways']}")
    print(f"  Drugs:                    {stats['total_drugs']}")
    print(f"  Genes with pathways:      {stats['genes_with_pathways']}")
    print(f"  Avg pathways per gene:    {stats['avg_pathways_per_gene']:.2f}")
    print(f"  Avg genes per pathway:    {stats['avg_genes_per_pathway']:.2f}")
    print("="*70)
    
    # Convert defaultdicts to regular dicts for JSON serialization
    knowledge_graph["gene_to_pathways"] = dict(knowledge_graph["gene_to_pathways"])
    knowledge_graph["pathway_to_genes"] = dict(knowledge_graph["pathway_to_genes"])
    knowledge_graph["drug_to_genes"] = dict(knowledge_graph["drug_to_genes"])
    
    return knowledge_graph


def save_knowledge_graph(kg: Dict, output_dir: str):
    """Save knowledge graph in multiple formats"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[5] Saving knowledge graph to {output_dir}...")
    
    # Save as JSON (human-readable)
    json_file = output_path / "tb_drug_knowledge_graph.json"
    with open(json_file, 'w') as f:
        json.dump(kg, f, indent=2)
    print(f"  ✓ Saved JSON: {json_file} ({json_file.stat().st_size / 1024:.1f} KB)")
    
    # Save as pickle (for Python loading)
    pickle_file = output_path / "tb_drug_knowledge_graph.pkl"
    with open(pickle_file, 'wb') as f:
        pickle.dump(kg, f)
    print(f"  ✓ Saved pickle: {pickle_file} ({pickle_file.stat().st_size / 1024:.1f} KB)")
    
    # Save summary as text
    summary_file = output_path / "knowledge_graph_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("TB DRUG RESISTANCE KNOWLEDGE GRAPH SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("STATISTICS\n")
        f.write("-" * 70 + "\n")
        for key, value in kg["statistics"].items():
            f.write(f"{key:.<40} {value}\n")
        
        f.write("\n\nDRUG-PATHWAY ASSOCIATIONS\n")
        f.write("-" * 70 + "\n")
        for drug, info in kg["drug_to_pathways"].items():
            f.write(f"\n{drug}:\n")
            f.write(f"  Description: {info['description']}\n")
            f.write(f"  Mechanism:   {info['mechanism']}\n")
            f.write(f"  Pathways:    {', '.join(info['pathways'])}\n")
        
        f.write("\n\nGENE-PATHWAY MAPPINGS\n")
        f.write("-" * 70 + "\n")
        for gene, pathways in sorted(kg["gene_to_pathways"].items()):
            f.write(f"\n{gene}:\n")
            gene_info = kg["gene_info"].get(gene, {})
            f.write(f"  KEGG ID:      {gene_info.get('kegg_id', 'N/A')}\n")
            f.write(f"  Drugs:        {', '.join(gene_info.get('associated_drugs', []))}\n")
            f.write(f"  Description:  {gene_info.get('description', 'N/A')}\n")
            f.write(f"  Pathways:     {', '.join(pathways) if pathways else 'None found'}\n")
    
    print(f"  ✓ Saved summary: {summary_file}")
    
    print(f"\n✓ Knowledge graph saved successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch KEGG pathway data for TB drug resistance analysis"
    )
    parser.add_argument(
        "--output", "-o",
        default="./kegg_data",
        help="Output directory for data (default: ./kegg_data)"
    )
    parser.add_argument(
        "--delay", "-d",
        type=float,
        default=0.5,
        help="Delay between API calls in seconds (default: 0.5)"
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force refresh cached data"
    )
    
    args = parser.parse_args()
    
    # Remove cache if force refresh
    if args.force_refresh:
        cache_dir = Path(args.output) / "cache"
        if cache_dir.exists():
            import shutil
            shutil.rmtree(cache_dir)
            print(f"✓ Cleared cache directory")
    
    # Initialize fetcher
    cache_dir = Path(args.output) / "cache"
    fetcher = KEGGFetcher(cache_dir=str(cache_dir), delay=args.delay)
    
    # Build knowledge graph
    kg = build_tb_drug_knowledge_graph(fetcher)
    
    # Save results
    save_knowledge_graph(kg, args.output)
    
    print("\n" + "="*70)
    print("✓ COMPLETE! Knowledge graph ready for use.")
    print("="*70)
    print(f"\nFiles created in {args.output}/:")
    print("  • tb_drug_knowledge_graph.json  (human-readable)")
    print("  • tb_drug_knowledge_graph.pkl   (Python object)")
    print("  • knowledge_graph_summary.txt   (summary report)")
    print("\nNext steps:")
    print("  1. Review knowledge_graph_summary.txt")
    print("  2. Load the data in your Jupyter notebook")
    print("  3. Build pathway-aware HGAT model")


if __name__ == "__main__":
    main()