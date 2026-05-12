# Project Handoff — HGAT TB AMR Matched Ablation

**Date:** 2026-04-23  
**Workspace:** `/mnt/bigssd/akshat_home/mlns/project`  
**Python env with all required packages:** `/mnt/bigssd/akshat_home/miniconda3/envs/lma/bin/python`  
(This is the only conda env that has `torch_geometric` installed.)

---

## What this project is

Research project predicting TB first-line drug resistance from CRyPTIC isolate VCFs
using a Heterogeneous Graph Attention Network (HGAT). The core question explored
across all prior conversation:

> Under a fair matched-training comparison, does pathway-aware graph structure
> (KEGG pathways / KO / drug nodes) trade a small AUROC cost for mechanism-grounded
> interpretability and better generalization to second-line drugs?

---

## Current state: code is done, training has NOT been run

Everything is implemented and dry-run verified. The actual 5-fold CV training
(`scripts/run_ablation.py`) still needs to be run. That is the only remaining step.

---

## What was built (prior conversations)

### 1. Comprehensive KEGG download (`kegg_fetch_all.py`)
- 147 mtu pathways + all cross-reference tables fetched to `kegg_data/`
- Key outputs: `kegg_data/tb_knowledge_graph_full.json` (~2.9 MB)
- 465 raw API responses cached in `kegg_data/cache/`
- Drug D-numbers verified (old `kegg_pathways.py` had several wrong IDs)

### 2. Data audit (`KEGG_DATA_AUDIT.md`)
Key findings:
- 9/23 AMR genes have zero KEGG pathway edges (genuine KEGG gap, confirmed via REST)
- Hub pathways `mtu01100`/`mtu01110` dominate attention (should be pruned)
- Old KEGG notebooks used wrong training protocol (30 epochs, LR 2e-3, test-fold thresholds)
- Drug node/edge metadata was loaded but never added to the forward graph

### 3. Ablation package (`amr_hgat/`)
| File | What it does |
|---|---|
| `data.py` | VCF parsing + PMI+SVD embeddings + phenotype loading for 12 drugs + snpEff integration |
| `splits.py` | 5-fold stratified CV with inner val split for early stopping + threshold |
| `metrics.py` | AUROC/AUPRC/F1/Sens/Spec; val-only threshold tuning; fold aggregation |
| `graph_builders.py` | `build_graph(arm, ...)` — shared base + arms B/C/D layered on top |
| `model.py` | Single `HGATModel` class; arm-configurable edge types + pathway attention |
| `train.py` | Matched protocol: 180 epochs, ES patience=20, LR 5e-4, weight_decay=1e-4 |

### 4. Scripts
| Script | What it does |
|---|---|
| `scripts/run_ablation.py` | Main driver: builds graph, runs 5-fold CV per arm, writes JSONs to `runs/` |
| `scripts/run_attention_analysis.py` | Full-dataset retrain + SNP/pathway attention (paper-style interpretability) |
| `scripts/aggregate_results.py` | Collates `runs/*/cv_summary.json` → `metrics_table.csv` |
| `scripts/annotate_vcfs.py` | Runs snpEff on VCFs → `snpeff_annotations.parquet` (optional) |
| `scripts/fetch_cryptic_reuse_table.py` | Downloads official `CRyPTIC_reuse_table_20231208.csv` from EBI into the repo root |

### 4b. One-folder reproduction bundle (`paper/mutgat_code/`)
Self-contained copy: `amr_hgat/`, scripts, `data/cryptic/` reuse CSVs, `kegg_data/*.json` (+ summaries, legacy drug KG), `requirements.txt`, `docs/SNPNEFF_SETUP.txt`, `paper_submission/` (`paper.tex`, `paper.bib`, figure; add ICML style files per `ICML_STYLE_NOTE.txt`), and `analysis/ablation_results.ipynb`. Defaults inside that folder point at bundled paths.

### 5. Analysis notebook + results doc
- `analysis/ablation_results.ipynb` — AUROC bar plots, pathway attention heatmap, attribution, mechanism hit-rates
- `ABLATION_RESULTS.md` — pre-structured results doc (tables blank, fill after training)

---

## The four ablation arms

All arms use the exact same training recipe (see Matched Protocol below).

| Arm | Nodes | Key changes vs old notebooks |
|---|---|---|
| **A** | isolate + per-gene SNPs | Baseline. Same graph as `HGAT_AMR.ipynb` but with corrected training protocol. |
| **B** | A + pathway | Hub pathways `mtu01100`/`mtu01110` removed. `mtu00074` (Mycolic acid biosynthesis) added. Dead `gene` hub node type removed. `pathway→isolate` edge masked in layer 2 to break hub over-smoothing cycle. |
| **C** | B + KO | 19 KO nodes added via `bulk_links.gene_to_ko`; bridges pathway-orphan genes (`gyrA/B`, `gidB`, `ahpC`, `tlyA`, etc.) into the functional graph with 24 KO-pathway edges. |
| **D** | C + drug | 8 drug nodes, 20 drug-pathway edges, 635 drug-gene edges. Per-drug masked pathway attention replaces the global 23-way softmax (INH head only attends INH-relevant pathways, etc.). |

---

## Matched training protocol (all arms identical)

| Setting | Value | Why changed from old notebooks |
|---|---|---|
| Epochs | 180 max | Old KEGG code: 30 (6× fewer) |
| Early stopping | patience=20 on val AUROC | Old code: none |
| LR | 5e-4 | Old KEGG code: 2e-3 (4× higher) |
| weight_decay | 1e-4 | Same |
| l2_lambda | **removed** | Old code had both weight_decay + l2_lambda (double L2) |
| hidden | 128 | Old KEGG code: 64 |
| Threshold tuning | val fold only | Old code: tuned on test fold (biased F1/Sens/Spec) |
| CV | 5-fold stratified, inner val for ES | Old code: 5-fold test-only (no val split) |
| `copy.deepcopy(x_dict)` | **removed** | Memory waste on every forward pass |
| `type_attn_raw` param | **removed** | Never applied to logits; was misleading |

---

## Drugs evaluated

**Primary (headline):** INH, RIF, EMB, LEV, MXF, ETH, KAN, AMK  
(All have ≥800 R cases in CRyPTIC 20231208 dataset)

**Secondary (reported but not headlined):** CFZ, LZD, BDQ, DLM  
(Too few R cases for tight CIs; include to show where priors might stabilize learning)

Expected AUROC directions:
- INH/RIF/EMB: all arms within ±0.003 (baseline saturates at ~0.97)
- LEV/MXF: C/D may improve via `gyrA/B` KO neighborhood
- KAN/AMK: C/D may improve via shared `mtu03010` ribosome KO structure
- ETH: benefits from `ethA` KO neighborhood + snpEff LOF signal

---

## Data files

| File | Description |
|---|---|
| `data/cryptic/vcf/` | Per-isolate VCF.gz files (download with `download_all_vcfs.py`; not stored in git) |
| `data/cryptic/CRyPTIC_reuse_table_20231208.csv` | Phenotype table; 12 drug binary labels + MICs (paper cohort) |
| `data/cryptic/CRyPTIC_reuse_table_20240917.csv` | Newer CRyPTIC table (used in some legacy notebooks) |
| `CRyPTIC_reuse_table_*.csv` (repo root) | Optional copies alongside older workflows; canonical paths for new runs are under `data/cryptic/` |
| `kegg_data/tb_knowledge_graph_full.json` | Comprehensive KEGG KG (use this, not the old one) |
| `kegg_data/tb_drug_knowledge_graph.json` | Old incomplete KG (kept for old notebook reproducibility) |
| `kegg_data/knowledge_graph_summary_full.txt` | Human-readable summary of the full KG |
| `KEGG_DATA_AUDIT.md` | Full audit of KEGG data quality for this task |

---

## How to run the ablation

```bash
PYTHON=/mnt/bigssd/akshat_home/miniconda3/envs/lma/bin/python
cd /mnt/bigssd/akshat_home/mlns/project

# Quick sanity check (builds all 4 graphs, skips training, ~10 min):
$PYTHON scripts/run_ablation.py --arms A B C D --dry-run

# Run all 4 arms (production, ~several hours on CPU, much faster on GPU):
$PYTHON scripts/run_ablation.py \
    --vcf-dir "./data/cryptic/vcf" \
    --phenotype-csv data/cryptic/CRyPTIC_reuse_table_20231208.csv \
    --kegg-json kegg_data/tb_knowledge_graph_full.json \
    --output-dir runs

# Run a single arm to test:
$PYTHON scripts/run_ablation.py --arms A --epochs 10

# Aggregate after training:
$PYTHON scripts/aggregate_results.py --runs-dir runs --output metrics_table.csv

# Then open analysis/ablation_results.ipynb for plots
```

---

## Existing notebooks (unchanged — for baseline reference)

| Notebook | What it produces |
|---|---|
| `HGAT_AMR.ipynb` | Baseline no-KEGG HGAT. INH 0.969 / RIF 0.969 / EMB 0.962 AUROC. |
| `CombinedwithKEGG/HGAT_Pathways.ipynb` | Multi-drug pathway HGAT (old protocol, 30 epochs, biased metrics). |
| `CombinedwithKEGG/HGAT_Pathways_{INH,RIF,EMB}_single_head.ipynb` | Per-drug single-head variants. |
| `ONLYKEGG/HGAT_Pathways_KEGG_INH_single_head.ipynb` | INH-only, same code as CombinedwithKEGG. |

These notebooks are **not** touched by the new code. Their results are the
"before" numbers to compare against.

---

## Key issues fixed in the new code (vs old notebooks)

1. **Unfair comparison**: old KEGG models ran 30 fixed epochs vs baseline's 180 with ES. Fixed.
2. **Test-fold threshold bias**: old code tuned thresholds on `test_idx_t`, making F1/Sens/Spec look better for KEGG. Fixed (val only).
3. **Hub over-smoothing**: `mtu01100`/`mtu01110` created 153k-edge hubs that flattened isolate representations. Fixed (pruned + layer-2 reverse edge masked).
4. **Dead gene hub**: `gene` node type received no signal from isolates/SNPs. Fixed (removed).
5. **Missing INH pathway**: `mtu00074` (Mycolic acid Biosynthesis; contains `fabG1/inhA/kasA`) was absent from the 23-pathway set. Fixed (added).
6. **Drug edges never connected**: `drug_to_pathways` was loaded but drug nodes were never added to the graph. Fixed in arm D.
7. **Double L2**: `Adam weight_decay=1e-4` + `l2_lambda=1e-4` in loss. Fixed (removed `l2_lambda`).
8. **`copy.deepcopy(x_dict)`**: wasted GPU memory on every forward pass. Fixed.
9. **`type_attn_raw`**: looked like edge-type attention but did nothing to logits. Removed.

---

## Papers / related context

- Existing notebooks are based on the CRyPTIC project (Walker 2022, Nature Medicine).
- Reference paper in `reference_paper/bbab299.pdf`.
- The research framing, after multiple review rounds, is:
  > "Under matched training, does pathway structure trade AUROC parity for
  > mechanism-grounded interpretability and better second-line drug generalization?"
  > (NOT "does adding KEGG improve AUROC on INH/RIF/EMB" — that claim was reviewed
  > and rejected; baseline saturates there.)

---

## Prior conversation plans (in `~/.cursor/plans/`)

| Plan file | Contents |
|---|---|
| `hgat_kegg_task_summary_1b615740.plan.md` | Overview of the three experimental tracks (baseline, CombinedwithKEGG, ONLYKEGG) |
| `hgat_kegg_deep_dive_c4985bad.plan.md` | Full diagnosis of why KEGG underperforms: protocol mismatch, hub over-smoothing, dead gene hub, double L2, test-fold threshold bias, ceiling effect |
| `hgat_matched_ablation_0c9a0dff.plan.md` | **The active plan** — 4-arm matched ablation specification (implemented) |

---

## What to do next in a new chat

1. **Start training**: run `scripts/run_ablation.py` (see command above). This will take hours. You can run arm A first to get a baseline, then B/C/D in parallel if you have multiple GPUs or overnight.

2. **Populate results**: after training, run `scripts/aggregate_results.py` and open `analysis/ablation_results.ipynb`.

3. **Fill in `ABLATION_RESULTS.md`**: replace the `—` placeholders with real numbers.

4. **Interpret attention heatmap** (arm D): check that RIF attends to `mtu03020`, INH to `mtu00074`, EMB to `mtu00572`, aminoglycosides to `mtu03010`. If the attention is flat/uniform, the drug-masked attention mechanism needs debugging.

5. **If KO edges prove weak** (arm C ≈ arm B): the 24 KO-pathway edges are real but thin. Consider whether to enrich by also pulling `map*→mtu*` mappings from the full KO records beyond what the current code does.

6. **Optional snpEff**: run `scripts/annotate_vcfs.py --vcf-dir ./data/cryptic/vcf --no-snpeff` for the fallback (length-based) annotation, or install snpEff and use the full pipeline (see `paper/mutgat_code/docs/SNPNEFF_SETUP.txt`). Then re-run with `--snpeff-parquet snpeff_annotations.parquet` to add `effect_class` one-hot features to SNP nodes.

---

## File tree (new files created)

```
project/
├── amr_hgat/
│   ├── __init__.py
│   ├── data.py
│   ├── graph_builders.py
│   ├── metrics.py
│   ├── model.py
│   ├── splits.py
│   └── train.py
├── data/
│   └── cryptic/
│       ├── CRyPTIC_reuse_table_20231208.csv
│       ├── CRyPTIC_reuse_table_20240917.csv
│       ├── SOURCE.txt
│       └── vcf/   (populate via download_all_vcfs.py)
├── scripts/
│   ├── annotate_vcfs.py
│   ├── run_ablation.py
│   ├── run_attention_analysis.py
│   ├── aggregate_results.py
│   └── fetch_cryptic_reuse_table.py
├── analysis/
│   └── ablation_results.ipynb
├── runs/
│   └── all_arms_summary.json  (dry-run output only; fill with real training)
├── kegg_data/
│   ├── tb_knowledge_graph_full.json   ← USE THIS (new, comprehensive)
│   ├── tb_knowledge_graph_full.pkl
│   ├── knowledge_graph_summary_full.txt
│   ├── tb_drug_knowledge_graph.json   ← old, kept for old notebook compat
│   └── cache/  (465 raw KEGG REST responses)
├── paper/
│   ├── paper.tex, paper.bib, fig_mutgat_graph.jpg
│   └── mutgat_code/   ← bundled reproduction + submission assets (see §4b)
├── kegg_fetch_all.py       ← new comprehensive fetcher
├── kegg_pathways.py        ← old fetcher (kept for reference)
├── download_all_vcfs.py    ← CRyPTIC VCF bulk download
├── KEGG_DATA_AUDIT.md      ← full KEGG data quality audit
├── ABLATION_RESULTS.md     ← results template (fill after training)
└── HANDOFF.md              ← this file
```
