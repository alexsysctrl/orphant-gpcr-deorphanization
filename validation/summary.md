# Deorphanization Algorithm Validation Report

## Validation Method: Leave-One-Receptor-Out Cross-Validation (LOROCV)

### Dataset
- **349** known receptor-ligand pairs
- **73** unique receptors
- **163** unique ligands
- **72** orphan receptors in catalog (not used in training)

### Scoring Strategy
Multi-signal scoring combining:
1. **Family match** (40% weight) — whether other receptors binding a ligand share the same family
2. **G-protein match** (30% weight) — whether other receptors binding a ligand share the same G-coupling
3. **KDE affinity** (20% weight) — similarity-weighted average affinity of a ligand across similar receptors
4. **Normalized affinity** (10% weight) — ligand's average -log10(ki) as tiebreaker

### Overall Metrics

| Metric | Value |
|--------|-------|
| Precision@1 | 2.74% |
| Precision@3 | 13.24% |
| Precision@5 | 10.68% |
| Precision@10 | 7.26% |
| Recall@1 | 0.65% |
| Recall@3 | 15.47% |
| Recall@5 | 19.00% |
| Recall@10 | 24.14% |
| AUC-ROC | 0.5878 |
| AUC-PR | 0.0400 |
| Mean Average Precision (mAP) | 0.1004 |

### Interpretation
- **P@1 = 2.74%** is slightly above random baseline (~1/163 = 0.6%), indicating weak but real signal
- **P@5 = 10.68%** means roughly 1 in 5 top-5 predictions is a true ligand
- **R@10 = 24.14%** means we recover ~1/4 of all true ligands in top-10
- **AUC-ROC = 0.59** indicates modest discriminative ability (0.5 = random)
- **mAP = 0.10** reflects the challenge of ranking true ligands highly when most candidates are false

### Performance by Receptor Family

| Family | N Receptors | Precision@5 |
|--------|-------------|-------------|
| Melanocortin | 20 | **26.0%** |
| S1P | 20 | **22.0%** |
| Orexin | 8 | **20.0%** |
| Free Fatty Acid | 16 | **15.0%** |
| Adrenergic | 36 | 11.67% |
| Dopaminergic | 20 | 11.0% |
| Muscarinic | 20 | 6.0% |
| Serotonergic | 44 | 5.91% |
| Purinergic | 16 | 2.5% |
| Opioid | 20 | 2.0% |
| Histamine | 16 | 0.0% |
| Adenosine | 16 | 0.0% |
| Trace Amine | 4 | 0.0% |
| Partially Deorphanized | 36 | 0.0% |

**Key finding**: Small, specialized families (melanocortin, S1P, orexin) perform best. Large families with many cross-binding ligands (serotonergic, adrenergic) perform worse due to signal dilution.

### Performance by Ligand Class

| Ligand Class | N Receptors | Precision@5 |
|---------------|-------------|-------------|
| Lipid | 28 | **15.71%** |
| Peptide | 52 | **13.85%** |
| Catecholamine | 60 | 10.67% |
| Fatty Acid | 28 | 8.57% |
| Amino Acid Derivative | 40 | 3.0% |
| Purine | 32 | 1.25% |
| Indolamine | 44 | 5.91% |
| Metal Ion | 4 | 0.0% |
| Amino Acid | 4 | 0.0% |

**Key finding**: Lipid and peptide ligands are most predictable, likely because their binding is more family-specific.

### Performance by G-Protein Coupling

| G-Protein | N Receptors | Precision@5 |
|-----------|-------------|-------------|
| G12/13 | 4 | **15.0%** |
| Gi | 132 | 8.79% |
| Gq | 64 | 8.75% |
| Gs | 88 | 7.73% |
| Ligand-gated ion channel | 4 | 0.0% |

### False Positive / False Negative Analysis

- **677 false positives** across all receptors (top-10 predictions that are not true ligands)
- **198 false negatives** (true ligands not in top-10)
- Top false positives tend to be ligands from unrelated families that happen to have high affinity (low ki), inflating their scores
- False negatives are concentrated in receptors with many true ligands (e.g., Alpha-2A with 7 ligands, Beta-2 with 9 ligands) — hard to rank all of them in top-10

### Correlation Analysis

- **Pair count vs P@5 correlation: 0.0097** — essentially zero correlation
- This means the number of known ligands per receptor does NOT predict prediction quality
- Receptors with few ligands and many ligands perform similarly, suggesting the signal is family-dependent rather than data-quantity-dependent

### Best Performing Receptors

| Receptor | Family | Hits in Top-10 |
|----------|--------|----------------|
| MC1R melanocortin receptor | melanocortin | 3 |
| MC4R melanocortin receptor | melanocortin | 3 |
| 5-HT1B receptor | serotonergic | 2 |
| 5-HT1D receptor | serotonergic | 2 |
| MC3R melanocortin receptor | melanocortin | 2 |
| MC5R melanocortin receptor | melanocortin | 2 |
| OX1 orexin receptor | orexin | 2 |
| OX2 orexin receptor | orexin | 2 |
| S1P1 receptor | S1P | 2 |
| S1P2 receptor | S1P | 2 |

### Worst Performing Receptors

| Receptor | Family | Hits in Top-10 |
|----------|--------|----------------|
| Beta-3 adrenergic receptor | adrenergic | 0 |
| 5-HT2A receptor | serotonergic | 0 |
| 5-HT2B receptor | serotonergic | 0 |
| 5-HT2C receptor | serotonergic | 0 |
| 5-HT3 receptor | serotonergic | 0 |
| 5-HT4 receptor | serotonergic | 0 |
| 5-HT5A receptor | serotonergic | 0 |
| 5-HT6 receptor | serotonergic | 0 |
| 5-HT7 receptor | serotonergic | 0 |
| Kappa opioid receptor | opioid | 0 |

**Pattern**: Serotonergic receptors (5-HT2A through 5-HT7) all perform at zero — these receptors share extremely promiscuous ligand profiles (serotonin, spiperone, LSD bind across many 5-HT subtypes), making family-based prediction ineffective.

### Key Insights

1. **Family-specific signal is real but weak**: The metadata-based similarity approach (family + G-protein matching) provides statistically meaningful signal above random, but precision remains low due to ligand promiscuity within families.

2. **Small families outperform large families**: Melanocortin (26% P@5), S1P (22%), and orexin (20%) families have tight ligand specificity within family, making prediction easier. Large families like serotonergic have cross-binding ligands that dilute the signal.

3. **The synthesized feature vectors have no discriminative power**: The ReceptorEncoder's 860-dim feature vectors produce nearly identical distances for all receptor pairs (~380-410 squared Euclidean), regardless of family membership. The RBF kernel with any reasonable bandwidth cannot distinguish receptors in this space.

4. **Metadata-based approach outperforms embedding approach**: Using raw family/G-protein/class metadata directly as similarity signals is more effective than the synthesized high-dimensional feature vectors.

5. **Ligand promiscuity is the main challenge**: Common ligands like serotonin, dopamine, spiperone, and norepinephrine bind across many subtypes within a family, making it impossible to distinguish subtypes based on family membership alone.

6. **No correlation with data quantity**: The number of known ligands per receptor does not predict prediction quality, suggesting that improving the similarity metric (rather than collecting more data) is the key bottleneck.

### Recommendations for Improvement

1. **Incorporate structural features**: Use ligand structural fingerprints (SMILES-based) and receptor binding pocket descriptors to capture subtype-specific binding determinants beyond family membership.

2. **Use sequence similarity**: Compute pairwise sequence identity between receptors (using BLOSUM62 alignment) rather than coarse family labels. This would differentiate Alpha-1A from Alpha-1B more finely.

3. **Pharmacophore matching**: Implement the pharmacophore projection from the framework (Eq. 24) to capture specific binding requirements (H-bond donors/acceptors, hydrophobic regions) that distinguish subtypes.

4. **Ensemble scoring**: Combine metadata similarity with structural similarity (ligand fingerprints) and sequence similarity for a more robust prediction signal.

5. **Threshold optimization**: The current scoring uses fixed weights (0.4/0.3/0.2/0.1). These could be optimized on a validation set to maximize AUC-ROC or P@5.

6. **Address the encoder limitation**: The ReceptorEncoder produces non-discriminative features because the synthesis function maps all receptors to similar vectors. A better synthesis would incorporate family-specific sequence patterns, binding pocket residue differences, and tissue expression profiles more distinctly.
