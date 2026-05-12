# KEGG Data Audit for HGAT + KEGG TB AMR Project

Date: 2026-04-22

This document captures the conclusions from auditing whether the KEGG data used
in `[CombinedwithKEGG/](CombinedwithKEGG/)` and `[ONLYKEGG/](ONLYKEGG/)` is
appropriate for predicting TB first-line drug resistance, plus the new
comprehensive KEGG download produced by `[kegg_fetch_all.py](kegg_fetch_all.py)`.

---

## 1. Project context

- Task: predict first-line TB drug resistance (INH, RIF, EMB, optionally PZA)
  from CRyPTIC VCF-derived SNP graphs using a Heterogeneous GAT.
- Baseline: `[HGAT_AMR.ipynb](HGAT_AMR.ipynb)` (no KEGG) ~0.97 AUROC on INH/RIF/EMB.
- KEGG variants: `[CombinedwithKEGG/HGAT_Pathways.ipynb](CombinedwithKEGG/HGAT_Pathways.ipynb)`
  and single-head per-drug notebooks, plus
  `[ONLYKEGG/HGAT_Pathways_KEGG_INH_single_head.ipynb](ONLYKEGG/HGAT_Pathways_KEGG_INH_single_head.ipynb)`.
  All use the same `build_pathway_aware_hetero_graph` code. Observed: KEGG
  versions are ~0.005-0.01 AUROC below baseline, i.e. slightly worse.
- Earlier deep-dive plan: `~/.cursor/plans/hgat_kegg_deep_dive_c4985bad.plan.md`.

## 2. What the original download contained

From `[kegg_data/knowledge_graph_summary.txt](kegg_data/knowledge_graph_summary.txt)`
(produced by `[kegg_pathways.py](kegg_pathways.py)`):

- 23 MTB resistance genes, hard-coded.
- 23 pathways, accumulated from:
  1. pathway IDs found inside each gene's `get/mtu:Rv*` response, and
  2. a manually curated `TB_DRUG_PATHWAYS` dict.
- 8 drug entries (metadata only; drug nodes were never added to the graph).
- No modules (KEGG has no `md:mtu_...` modules; confirmed).

## 3. Was the download "wrong"?

Not technically wrong, but systematically incomplete:

### 3.1 Nine AMR genes had "Pathways: None found" - that is real KEGG data

Confirmed via direct REST:

```
curl https://rest.kegg.jp/link/pathway/mtu:Rv0006   # gyrA  -> empty
curl https://rest.kegg.jp/link/pathway/mtu:Rv3919c  # gidB  -> empty
curl https://rest.kegg.jp/link/pathway/mtu:Rv2428   # ahpC  -> empty
curl https://rest.kegg.jp/link/pathway/mtu:Rv2416c  # eis   -> empty
curl https://rest.kegg.jp/link/pathway/mtu:Rv1694   # tlyA  -> empty
curl https://rest.kegg.jp/link/pathway/mtu:Rv0341   # iniB  -> empty
curl https://rest.kegg.jp/link/pathway/mtu:Rv0342   # iniA  -> empty
curl https://rest.kegg.jp/link/pathway/mtu:Rv0343   # iniC  -> empty
```

KEGG genuinely has no pathway edges for these genes under `mtu:`. Orthology
(KO) and BRITE categories exist for most of them, but the pathway database is
silent. Consequence: the INH compensatory/inducible operon (`ahpC`, `iniABC`),
FQ targets (`gyrA`, `gyrB`), aminoglycoside genes (`eis`, `tlyA`), and
streptomycin `gidB` - **the mechanistically important mutations driving AMR**
- cannot be routed through any pathway node. This is a KEGG-MTB coverage
limitation, not a parser bug.

### 3.2 The manually curated drug -> pathway list missed mechanistic pathways

`[kegg_pathways.py](kegg_pathways.py)` (lines 265-309) defines
`TB_DRUG_PATHWAYS` by hand. The INH mapping is wrong in spirit:

- INH target pathway should be `mtu00074 Mycolic acid biosynthesis`
  (contains `fabG1`, `inhA`, `kasA`) or `mtu00061 Fatty acid biosynthesis`.
- Original code used `mtu00650 Butanoate metabolism`, which has no INH
  mechanistic link.

### 3.3 KEGG has drug-resistance pathways that were not included

| KEGG pathway | Name | mtu genes | Overlap with our AMR panel |
|---|---|---:|---:|
| `mtu01501` | beta-Lactam resistance | 9 | 0 |
| `mtu01503` | CAMP resistance | 4 | 0 |
| `mtu05152` | Tuberculosis (host pathogenesis) | 34 | 0 |
| `mtu00061` | Fatty acid biosynthesis | 13 | 0 (helpers only) |
| `mtu00074` | Mycolic acid biosynthesis | 36 | 3 (`fabG1`, `inhA`, `kasA`) |

The first three exist in KEGG but do not cover any of the CRyPTIC first-line
resistance genes. `mtu00074` is the only genuinely new "mechanistic INH"
pathway we should have added to the graph.

### 3.4 Hub pathways dominate the pathway attention

`mtu01100 Metabolic pathways` (699 mtu genes) and `mtu01110 Biosynthesis of
secondary metabolites` (370 mtu genes) are catalog pathways: every MTB isolate
is linked to them for any SNP. In the 23-way softmax used in
`PathwayHGAT_AMR`, these hubs swallow most attention mass and wash out
mechanism-specific pathways (`mtu03020`, `mtu00572`, etc.). They should be
pruned.

## 4. Is KEGG fundamentally a good fit for this task?

Likely no, for three reasons:

1. **Signal is at the codon level, not the pathway level.** The ground truth
   for R/S is a short list of canonical mutations (`rpoB S450L`,
   `katG S315T`, `embB M306V`, `pncA` spread). SNP embeddings over 893
   recurrent SNPs already reach ~0.97 AUROC. Pathway abstraction flattens
   exactly the information that separates resistant from sensitive isolates.
2. **KEGG coverage is sparse and hub-heavy.** 18 of 50 panel genes have zero
   pathway edges after the comprehensive download; of the remaining, most of
   the mass is in hub pathways that every isolate shares.
3. **Ceiling effect.** With baseline saturating at ~0.97, KEGG has almost no
   variance left to help with, and biased features tend to hurt in that
   regime.

KEGG could still help in settings the current experiments don't exercise
(cold-start isolates, rare drugs, or when drug->pathway and drug->gene edges
are added to the forward graph), but as currently wired, KEGG is a near-zero-
or-negative-signal addition.

## 5. Comprehensive KEGG download (new)

`[kegg_fetch_all.py](kegg_fetch_all.py)` produces a superset of the original
fetch. Output at `[kegg_data/tb_knowledge_graph_full.json](kegg_data/tb_knowledge_graph_full.json)`
and `.pkl`, with a readable summary at
`[kegg_data/knowledge_graph_summary_full.txt](kegg_data/knowledge_graph_summary_full.txt)`.

What was fetched:

- **147 mtu pathways** (all 142 from `list/pathway/mtu` plus 5 extras in
  `EXTRA_PATHWAYS` that either returned empty on mtu or came from the `map`
  namespace; they are kept for completeness).
- **Detailed pathway records** for all 147 (description, class, compounds,
  reference counts).
- **Pathway -> gene lists** via `link/mtu/<pwy>` for every pathway.
- **Expanded AMR gene panel (50 genes)** covering all first- and second-line
  resistance drivers in the WHO AMR catalogue + CRyPTIC supplements. Includes:
  - INH: `katG, inhA, fabG1, ahpC, ndh, iniABC, kasA, mshA, furA, nat`
  - ETH: `ethA, ethR, mymA` (+ overlaps with INH)
  - RIF: `rpoA, rpoB, rpoC`
  - EMB: `embA, embB, embC, embR, manB, rmlD, aftA, ubiA`
  - PZA: `pncA, rpsA, panD, clpC1`
  - FLQ: `gyrA, gyrB`
  - Aminoglycosides / CAP: `rrs, rrl, eis, gidB, tlyA, whiB7`
  - LZD: `rplC, rplD`
  - BDQ / CFZ: `atpE, atpB, mmpR5, pepQ`
  - DLM / PTM: `ddn, fgd1, fbiA, fbiB, fbiC, fbiD`
- **Bulk cross-reference tables** for every mtu gene:
  - `gene_to_pathway` (`link/pathway/mtu`)
  - `gene_to_ko` (`link/ko/mtu`)
  - `gene_to_enzyme` (`link/enzyme/mtu`)
  - `gene_to_module` (`link/module/mtu`; empty for mtu as expected)
  - `gene_to_reaction` (`link/reaction/mtu`; empty for mtu as expected)
- **KO records** for every KO associated with the AMR panel (43 KOs, fetched
  from `get/ko:K*`).
- **Drug records** for 23 TB-relevant drugs (D-numbers re-verified 2026-04-22;
  the original fetcher had several wrong D-numbers such as DLM=D09780
  instead of D09785 and CFZ=D01211 instead of D00278).
- **Drug -> pathway / disease links** via `link/pathway/dr:<id>` and
  `link/disease/dr:<id>`.
- **BRITE hierarchies**: `br08402` (diseases), `br08303` (drug targets),
  `br08901` (pathway maps), `br08902` (BRITE files), `br08907` (AMR genes),
  `mtu00001` (KO hierarchy), `mtu01000` (enzymes), `mtu03036` (chromosome
  proteins).
- **TB disease record** `H00342 Tuberculosis`.

Cache is idempotent (`kegg_data/cache/*.txt`, 465 files on disk). Re-running
`python kegg_fetch_all.py` is free after the first pass.

## 6. Coverage of the AMR panel in the new download

| Layer | Coverage |
|---|---|
| Gene -> KO | 42 of 50 genes (8 have no KEGG KO: `eis`, `iniA`, `iniB`, `iniC`, `mmpR5`, `mshA`, `pepQ`, `furA partial`) |
| Gene -> pathway | 32 of 50 have >= 1 pathway edge; 18 have none |
| Gene -> enzyme (EC) | 28 of 50 |
| Gene -> reaction | 0 of 50 (KEGG does not attach `mtu:` reaction links at the gene level) |
| Gene -> mycolic acid biosynthesis (mtu00074) | 3 of 50 (`fabG1`, `inhA`, `kasA`) |
| Gene -> ribosome (mtu03010) | 5 of 50 (`rplC`, `rplD`, `rpsA`, `rrl`, `rrs`) |
| Gene -> RNA polymerase (mtu03020) | 3 of 50 (`rpoA`, `rpoB`, `rpoC`) |

The 18 pathway-orphan genes will remain orphan no matter how much more we
download from KEGG: they have no `mtu:Rv... -> path:mtu...` edges in the KEGG
graph at all.

## 7. CRyPTIC phenotype balance (for context)

From `[CRyPTIC_reuse_table_20231208.csv](CRyPTIC_reuse_table_20231208.csv)`,
12,287 isolates:

| Drug | R | S | NA | R% |
|---|---:|---:|---:|---:|
| INH | 5907 | 6161 | 219 | 48.9 |
| RIF | 4683 | 7414 | 190 | 38.7 |
| RFB | 4464 | 7684 | 139 | 36.7 |
| EMB | 2261 | 8336 | 1690 | 21.3 |
| LEV | 2145 | 10016 | 126 | 17.6 |
| ETH | 1727 | 9431 | 1129 | 15.5 |
| MXF | 1724 | 10468 | 95 | 14.1 |
| KAN | 1120 | 11008 | 159 | 9.2 |
| AMK | 882 | 11188 | 217 | 7.3 |
| CFZ | 525 | 11522 | 240 | 4.4 |
| DLM | 186 | 11739 | 362 | 1.6 |
| LZD | 156 | 12031 | 100 | 1.3 |
| BDQ | 109 | 11957 | 221 | 0.9 |

First-line drugs INH/RIF are well balanced; EMB is moderately imbalanced with
13.8 percent missing labels. Second-line drugs become heavily imbalanced and
not really modelable with plain BCE.

## 8. Recommended next steps (ordered)

1. **Equalize training protocol.** Run the KEGG pathway model with the
   baseline's settings (180 epochs, early stopping on val AUROC, LR 5e-4,
   hidden=128, proper val/test split). Without this the "KEGG hurts" claim
   conflates under-training with KEGG effect.
2. **Prune hub pathways.** Drop `mtu01100` and `mtu01110` from the pathway
   node set; they add no resistance-specific signal.
3. **Add `mtu00074 Mycolic acid biosynthesis`.** Covers `fabG1/inhA/kasA`,
   which is the actual INH mechanism; not in the original 23.
4. **Break the isolate <-> pathway cycle.** Keep `isolate -> pathway` but
   remove `pathway -> isolate` in layer 2, or mask the reverse edge on the
   second layer, to mitigate hub over-smoothing.
5. **Remove the disconnected `gene` hub.** It receives no signal from SNPs or
   isolates and therefore cannot contribute.
6. **Connect drug nodes.** Add `drug -> pathway` and `drug -> gene` edges from
   `drug_to_pathways_grounded` and `drug_to_genes` in the full KG; today the
   metadata is loaded but not used in the forward pass.
7. **Use KO as a fallback layer for pathway-orphan genes.** The new
   `gene_to_ko` map covers 42 of 50 panel genes (vs 32 of 50 for pathway);
   KO nodes would connect `gyrA/B`, `gidB`, `eis`, `tlyA`, etc. into the
   graph functionally.
8. **Clean up the model code.** Remove the `copy.deepcopy(x_dict)` in
   `PathwayHGAT_AMR.forward`, drop the unused `type_attn_raw` parameter, and
   pick exactly one L2 source (`weight_decay` OR `l2_lambda`, not both).
9. **Matched-config ablation.** With identical training settings, compare
   {SNP only, SNP + pathway (hubs pruned), SNP + pathway + KO, pathway only}.
   This cleanly answers "is KEGG useful at all?".

## 9. Files of interest

- `[kegg_fetch_all.py](kegg_fetch_all.py)` - new comprehensive fetcher.
- `[kegg_data/tb_knowledge_graph_full.json](kegg_data/tb_knowledge_graph_full.json)`
  - merged knowledge graph.
- `[kegg_data/tb_knowledge_graph_full.pkl](kegg_data/tb_knowledge_graph_full.pkl)`
  - same, for Python.
- `[kegg_data/knowledge_graph_summary_full.txt](kegg_data/knowledge_graph_summary_full.txt)`
  - human summary.
- `[kegg_data/cache/](kegg_data/cache/)` - raw KEGG REST responses (~465
  files).
- `[kegg_pathways.py](kegg_pathways.py)` - original fetcher (kept for
  reproducibility).
- `[kegg_data/tb_drug_knowledge_graph.json](kegg_data/tb_drug_knowledge_graph.json)`
  - original, incomplete KG still consumed by the pathway notebooks.

## 10. One-line verdict

Your KEGG data was authentic but incomplete; the richer download confirms
that KEGG's mtu pathway graph is fundamentally sparse for TB AMR and cannot
cover ~1/3 of the canonical resistance genes, so the pathway branch in the
current model is structurally underpowered. Bigger wins will come from model
and graph changes (drug edges, KO fallback, hub pruning, matched training)
than from downloading more KEGG data.
