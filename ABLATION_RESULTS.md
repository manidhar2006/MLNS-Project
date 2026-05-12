# HGAT Matched Ablation — Results

*This file will be populated with numbers once `scripts/run_ablation.py` completes.*
*The section structure, tables, and figure references are pre-filled.*

---

## How to run

The project requires the `lma` conda environment which has PyG (torch_geometric) installed.

```bash
# Activate the correct Python environment
PYTHON=/mnt/bigssd/akshat_home/miniconda3/envs/lma/bin/python

# 1. (Optional) Annotate SNPs with snpEff for effect_class features
$PYTHON scripts/annotate_vcfs.py \
    --vcf-dir "./data/cryptic/vcf" \
    --output  snpeff_annotations.parquet \
    --snpeff-jar /path/to/snpEff.jar
# Or use fallback (no snpEff required, less accurate):
$PYTHON scripts/annotate_vcfs.py \
    --vcf-dir "./data/cryptic/vcf" --output snpeff_annotations.parquet --no-snpeff

# 2. Run 4-arm ablation (takes ~several hours on GPU / ~10+ hours on CPU)
$PYTHON scripts/run_ablation.py \
    --vcf-dir "./data/cryptic/vcf" \
    --phenotype-csv data/cryptic/CRyPTIC_reuse_table_20231208.csv \
    --kegg-json kegg_data/tb_knowledge_graph_full.json \
    --snpeff-parquet snpeff_annotations.parquet \
    --output-dir runs

# 3. Aggregate results
$PYTHON scripts/aggregate_results.py --runs-dir runs --output metrics_table.csv

# 4. Dry-run (build graphs only, no training) to verify setup:
$PYTHON scripts/run_ablation.py --arms A B C D --dry-run
```

---

## Experimental setup

| | |
|---|---|
| Dataset | CRyPTIC 2023-12-08 (12,287 isolates) |
| Protocol | 5-fold stratified CV (RIF label), inner 20% val for ES + threshold |
| Epochs | 180 max, early stopping patience=20 on val AUROC |
| Optimizer | Adam LR=5e-4, weight_decay=1e-4 |
| Architecture | hidden=128, 2 HGAT layers, 2 attention heads, dropout=0.3 |
| Threshold | Tuned on val fold only (never on test) |
| Primary metric | AUROC (rank-based, unaffected by threshold choice) |

---

## Arms

| Arm | Description |
|---|---|
| **A** | SNP-only baseline. Isolate + per-gene SNP nodes, no biological structure beyond VCF-derived SNP co-occurrence. |
| **B** | Pathway-minimal. Arm A + pathway nodes. Hub pathways `mtu01100`/`mtu01110` pruned. `mtu00074` (Mycolic acid biosynthesis) added. Dead `gene` hub removed. `pathway→isolate` reverse edge masked in layer 2. |
| **C** | Pathway + KO. Arm B + KO node type. Connects pathway-orphan genes (`gyrA/B`, `gidB`, `ahpC`, `tlyA`, etc.) to their KEGG Orthology group and thence to pathways. |
| **D** | Pathway + KO + Drug. Arm C + drug nodes. `drug↔pathway` and `drug↔gene` edges from `kegg_data/tb_knowledge_graph_full.json`. Per-drug masked pathway attention replaces the global 23-way softmax. |

---

## Primary results: AUROC (mean ± std over 5 folds)

*Fill in after run completes. Expected format:*

| Drug | A (SNP-only) | B (Pathway-min) | C (+KO) | D (+Drug) |
|---|---|---|---|---|
| INH | — | — | — | — |
| RIF | — | — | — | — |
| EMB | — | — | — | — |
| LEV | — | — | — | — |
| MXF | — | — | — | — |
| ETH | — | — | — | — |
| KAN | — | — | — | — |
| AMK | — | — | — | — |

*Fill in from `metrics_table.csv` or `analysis/ablation_results.ipynb`.*

---

## Secondary results: AUROC for low-prevalence drugs

| Drug | R% | A | B | C | D |
|---|---|---|---|---|---|
| CFZ | 4.4 | — | — | — | — |
| LZD | 1.3 | — | — | — | — |
| BDQ | 0.9 | — | — | — | — |
| DLM | 1.6 | — | — | — | — |

---

## Interpretability: pathway attention heatmap (arm D)

See `analysis/pathway_attention_heatmap_D.png`.

Expected pattern (pre-run):
- RIF arm D → dominant attention on `mtu03020` (RNA polymerase)
- INH arm D → dominant attention on `mtu00074` (Mycolic acid biosynthesis)
- EMB arm D → dominant attention on `mtu00572` / `mtu00571` (Arabinogalactan / LAM)
- KAN/AMK arm D → dominant attention on `mtu03010` (Ribosome)

---

## Interpretability: SNP → gene → pathway attribution

Canonical mutations tested: `katG S315T` (INH), `rpoB S450L` (RIF),
`embB M306V` (EMB), `gyrA D94G` (LEV/MXF), `rrs A1401G` (AMK/KAN).

See `analysis/ablation_results.ipynb` Section 4.

---

## AUROC delta heatmap (B/C/D vs A)

See `analysis/auroc_delta_heatmap.png`.

Colour: green = better than arm A, red = worse.

---

## Key design decisions and code notes

- **`copy.deepcopy(x_dict)` removed** from `PathwayHGAT_AMR.forward`; was wasting GPU memory on every forward pass.
- **`type_attn_raw` / `type_attn_weights` removed**; the parameter existed in both the baseline and KEGG notebooks but was never applied to logits and misleadingly suggested edge-type attention was operating.
- **Double L2 removed**: the old KEGG notebooks applied both `Adam weight_decay=1e-4` and an explicit `l2_lambda=1e-4` L2 term in `masked_bce_loss`, effectively doubling regularization. Only Adam `weight_decay` is used now.
- **Threshold tuning on val only**: the old KEGG notebooks selected `best_probs` from `test_idx_t` and then called `find_best_thresholds` on those same test arrays. F1/Sens/Spec/Threshold were therefore optimistically biased. All threshold tuning now happens on the inner val split.
- **LR reduced from 2e-3 to 5e-4, epochs raised from 30 to 180**: the old KEGG models were under-trained by ~6× relative to the baseline.

---

## Conclusion (to be written after results)

*Expected conclusion based on prior analysis:*

Under a matched training protocol, all four arms achieve similar AUROC on
first-line drugs (INH/RIF/EMB), confirming that KEGG pathway structure cannot
improve over a near-saturated SNP-level baseline (0.96–0.97 AUROC). On
mid-tier drugs (LEV/MXF, KAN/AMK), arms C and D provide small positive
differences due to KO nodes connecting `gyrA/B` and `rrs` to functional
context. Arm D's drug-masked pathway attention produces mechanism-grounded
interpretability outputs — RIF models attend predominantly to `mtu03020` (RNA
polymerase), INH to `mtu00074` (Mycolic acid biosynthesis), and aminoglycosides
to `mtu03010` (Ribosome) — while maintaining parity or marginal gains in AUROC
on second-line drugs, supporting the paper framing of "parity with mechanistic
grounding" rather than a simple AUROC improvement story.
