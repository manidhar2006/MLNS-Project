Now I have enough detail to give a thorough review. Here are the issues I see, from most impactful to least.

1. Transductive data leakage — the biggest problem
The entire graph (all 12,279 isolates) is passed to every forward pass. During training, the model runs message passing over all nodes, including test isolates. That means:


train.py
Lines 121-122
        logits = model(data.x_dict, data.edge_index_dict)
        loss = masked_bce_loss(logits[train_t], data["isolate"].y[train_t], pw)
Only the loss is masked to train indices. But the GATConv in layers 1 and 2 passes messages between all isolate and SNP nodes. A test isolate's embedding is shaped by its neighbors, which include training isolates that share SNPs with it — and those training isolates' representations are shaped by their labels (indirectly, through gradient updates).

This is the standard transductive GNN setup, and for node classification on a single fixed graph it's defensible. But it means:

Your AUROC numbers are optimistic relative to a truly inductive setting (new isolate arrives, predict its resistance).
The SNP nodes act as information bridges: a rare SNP shared between a train-R isolate and a test isolate leaks signal through the GAT attention weights.
This equally affects all four arms, so the relative comparison is still fair. But the absolute numbers shouldn't be compared directly against an inductive model like a random forest on the same features.
If the goal is a deployable predictor, you'd need inductive evaluation (remove test isolate nodes from the graph entirely and add them back at inference with their edges but without label-derived gradients). If the goal is the ablation comparison, this is less critical but should be disclosed.

2. The PMI+SVD embedding is questionable
The embedding pipeline does this:


data.py
Lines 273-292
    mat = csr_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(len(isolates), len(snps)),
        dtype=np.float32,
    )
    co = (mat.T @ mat).toarray()
    ...
    pmi = np.log(p_ij / (p_i * p_j + 1e-12))
    ...
    pmi = np.maximum(pmi, 0.0)   # PPMI
    k = max(1, min(embed_dim, min(pmi.shape) - 1))
    u, s_vals, _ = svds(csr_matrix(pmi), k=k)
    embeddings = (u * np.sqrt(s_vals)).astype(np.float32)
Problems:

Co-occurrence is computed across all isolates including test. The PPMI matrix captures which SNPs tend to appear together across the entire dataset. This is another form of data leakage — the embedding of a SNP encodes information about test-set co-occurrence patterns.
PMI measures co-occurrence, not function. Two SNPs that co-occur frequently (e.g., because they're in the same clonal lineage) get similar embeddings. This captures population structure (lineage), not resistance mechanism. That's useful for prediction but misleading for interpretability — the model may be partly doing lineage classification dressed up as gene-level attention.
84% of SNPs are discarded by min_count=5. You go from 5,721 unique SNPs to 893. The rare SNPs that get dropped are often the most informative for drugs like PZA, ETH, and second-line drugs where resistance is driven by diverse loss-of-function mutations spread across the gene. katG goes from 458 to 31 SNPs in the embedding.
max-pool for isolate features loses multiplicity. An isolate with 1 SNP in katG and an isolate with 5 SNPs in katG get the same isolate feature if the max embedding is the same. The number of mutations in a gene is informative (multi-hit = more likely resistant).
3. The graph topology has a fundamental asymmetry
SNP nodes are gene-local (each gene type has its own node space), but isolate nodes are global. This means:

katG SNP node 0 and rpoB SNP node 0 live in different type spaces and never directly exchange messages.
The only path between two genes is gene_A_snp → isolate → gene_B_snp (2 hops).
In arms B/C/D, pathways add a second bridge, but it's isolate → pathway → isolate, not gene → pathway → gene.
The consequence: gene-gene relationships are learned entirely through shared isolates, not through any explicit gene-level connectivity. This is why the pathway additions (B/C/D) barely change AUROC — the pathway nodes connect to isolates, not to gene-SNP nodes directly. The biological prior you're trying to encode (genes in the same pathway are functionally related) never actually creates a message-passing path between gene-level representations.

A more natural design would have pathway nodes connected to gene-type nodes:

gene_type → in_pathway → pathway
pathway → has_gene → gene_type
This would let gyrA SNP representations be informed by other genes in the same pathway.

4. The pathway context is a post-hoc readout, not a structural prior
For arms B and C, pathways participate in message passing (isolate ↔ pathway edges), but the pathway context vector is computed after both GAT layers as a separate attention readout:


model.py
Lines 188-198
    def _pathway_context_global(
        self, iso_l2: torch.Tensor, h2: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        scores = torch.matmul(iso_l2, p.T) / (p.shape[1] ** 0.5)
        alpha = torch.softmax(scores, dim=1)
        p_ctx = alpha @ p
        return self.pathway_ctx(p_ctx)
This is an attention-weighted average of pathway embeddings. Since pathway features are initialized as one-hot identity vectors (np.eye(22)), the pathway representations after 2 layers of message passing are really just aggregations of their neighboring isolate features. So p_ctx is approximately "weighted average of average isolate features grouped by pathway" — which is close to what the isolate embedding already captures.

5. Pathway features are identity vectors

graph_builders.py
Lines 209-210
    pwy_feats = np.eye(len(pathways), dtype=np.float32)
    data["pathway"].x = torch.from_numpy(pwy_feats)
Similarly for KO nodes:


graph_builders.py
Lines 303-304
    ko_feats = np.eye(len(ko_list), dtype=np.float32)
    data["ko"].x = torch.from_numpy(ko_feats)
These are 22-dim and 19-dim one-hot vectors respectively. They carry zero semantic content. The only information they gain is through message passing from isolates. Compare this to the 64-dim PMI+SVD embeddings that isolate and SNP nodes get. The pathway and KO nodes start at a massive representational disadvantage.

You could initialize pathway features with something meaningful: gene membership vectors, functional annotations, pathway size, or even pre-trained biological embeddings.

6. Stratification is on a single drug

splits.py
Lines 46-48
    if primary_drug and primary_drug in drug_cols:
        strat_col = primary_drug
    elif "RIF_BINARY_PHENOTYPE" in drug_cols:
        strat_col = "RIF_BINARY_PHENOTYPE"
Folds are stratified on RIF only. For drugs with very different resistance prevalence (AMI at 7% R vs INH at 48% R), the folds may not be well-balanced. Multi-label stratification (e.g., iterative-stratification) would be more appropriate for a multi-drug prediction task.

7. Full-graph forward pass every epoch is wasteful
Every epoch does a full forward pass over all 12,279 isolates and 893 SNP nodes, but only uses train_t indices for the loss. This is fine for small graphs but means the model is doing ~6x more compute than necessary per epoch if you could use mini-batch sampling. More importantly, it means you can't scale to larger datasets without running out of memory.

Summary of what I'd prioritize fixing
Priority	Issue	Impact on results	Fix difficulty
1
PMI embedding computed on full dataset (train+test)
Inflated absolute metrics
Medium — recompute per fold
2
No gene→pathway edges (pathways only connect at isolate level)
Pathway structure can't help gene representations
Medium — add edges in graph builder
3
84% SNP dropout from min_count=5
Missing rare resistance variants
Easy — lower threshold or use fallback features
4
Pathway/KO features are uninformative one-hots
Pathway nodes start with no semantic content
Easy — use gene-count or functional features
5
Transductive message passing over test nodes
Optimistic absolute AUROC
Hard — requires inductive split
6
Single-drug fold stratification
Imbalanced folds for rare-R drugs
Easy — use multi-label stratification
Issues 1-4 are the ones most likely to explain why B/C/D barely improve over A: the pathway structure is connected in a way that can't actually inform gene-level representations, and the features on those nodes carry no biological signal to begin with.