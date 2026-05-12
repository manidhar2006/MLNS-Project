# MLNS-Project — MutGAT / HGAT for TB drug resistance

Heterogeneous graph attention models for predicting *Mycobacterium tuberculosis* antibiotic resistance from CRyPTIC whole-genome data, including **MutGAT**: a KEGG pathway–enriched graph with a matched four-arm ablation (SNP-only baseline through drug-aware full model). The ICML-style write-up lives under `paper/`.

## Repository layout

| Path | Contents |
|------|----------|
| `amr_hgat/` | Python package: VCF parsing, PMI+SVD embeddings, graph construction (arms A–D), `HGATModel`, training, metrics, attention helpers |
| `scripts/` | `run_ablation.py` (5-fold CV), `run_attention_analysis.py`, `aggregate_results.py`, `annotate_vcfs.py`, `fetch_cryptic_reuse_table.py` |
| `data/cryptic/` | CRyPTIC reuse phenotype CSVs; put downloaded VCFs under `data/cryptic/vcf/` (see `data/cryptic/SOURCE.txt`) |
| `kegg_data/` | KEGG knowledge graph JSON and summaries (`tb_knowledge_graph_full.json` for arms B–D). Cache dir is gitignored; regenerate with `kegg_fetch_all.py` if needed |
| `paper/` | `paper.tex`, `paper.bib`, figure; `paper/mutgat_code/` is a self-contained copy of code + bundled small data + `requirements.txt` |
| `analysis/` | `ablation_results.ipynb` for plotting CV outputs under `runs/` |
| `HANDOFF.md` | Longer operational notes (conda env, protocol details, file tree) |

## Requirements

- Python 3.10+ recommended  
- [PyTorch](https://pytorch.org/) and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) (versions must match your CUDA/CPU install)  
- `numpy`, `pandas`, `scipy`, `scikit-learn`, `iterative-stratification`  

Pinned lists: `paper/mutgat_code/requirements.txt` (same stack as the main package).

## Quick start (from repo root)

1. **Phenotypes** are under `data/cryptic/`. **VCFs**: run `download_all_vcfs.py` (or the copy in `paper/mutgat_code/scripts/`) as described in `data/cryptic/SOURCE.txt`.  
2. **Optional snpEff** annotations: see `paper/mutgat_code/docs/SNPNEFF_SETUP.txt`.  
3. **Train / evaluate** (example):

```bash
python scripts/run_ablation.py \
  --vcf-dir data/cryptic/vcf \
  --phenotype-csv data/cryptic/CRyPTIC_reuse_table_20231208.csv \
  --kegg-json kegg_data/tb_knowledge_graph_full.json \
  --output-dir runs
```

4. **Aggregate** fold JSONs: `python scripts/aggregate_results.py --runs-dir runs --output metrics_table.csv`  
5. **Plots**: open `analysis/ablation_results.ipynb`.

Dry-run (build graphs, no training): `python scripts/run_ablation.py --arms A B C D --dry-run`

## Data and ethics

CRyPTIC consortium data are subject to [EBI / CRyPTIC terms of use](https://www.ebi.ac.uk/ena/browser/view/PRJEB28461). Do not commit per-sample VCFs to git (they are large); this repo’s `.gitignore` excludes typical VCF paths under `data/cryptic/vcf/`.

## Citation

If you use the MutGAT method or this codebase, cite the paper associated with `paper/paper.tex` (and the CRyPTIC / KEGG primary sources cited in `paper/paper.bib`) once the work is published or publicly available.

## License

Add a `LICENSE` file if you intend open redistribution; none is set by default in this tree.
