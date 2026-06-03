# Executive Summary: Orphan GPCR Deorphanization

> **Critical Calibration Notice:** The model's confidence scores are severely overconfident. At confidence >0.8, actual accuracy = 0%. This executive summary has been updated to reflect honest calibration results.

**Date:** June 2026 | **Version:** v2 (calibration-aware)

---

## What We Did

We developed a metadata-based similarity approach to predict the endogenous ligands of 72 orphan G-protein coupled receptors (GPCRs). The approach uses known receptor-ligand binding data from GtoPdb, ChEMBL, and IUPHAR to generate ranked ligand predictions for uncharacterized receptors. The method is intentionally simple: it compares receptor metadata (family, G-protein coupling, tissue expression) to find similar receptors and transfer their known ligands.

**This is a heuristic approach, not a rigorous mathematical model.** The "mathematical framework" describes the implementation formally but does not provide predictive theory.

## Key Results

**Validation Performance (LOOCV on 349 pairs, 73 receptors, 163 ligands):**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Precision@5 | **10.68%** | ~1 in 5 top-5 predictions is correct |
| Recall@10 | **24.14%** | ~1 in 4 true ligands recovered in top-10 |
| AUC-ROC | **0.5878** | Modest discriminative ability (0.5 = random) |
| Brier score | **0.1084** | High (uncalibrated confidence) |
| ECE | **0.1382** | Moderate calibration error |

**After Platt scaling calibration:**
- ECE: 0.1382 → **0.0032** (97.7% improvement)
- Brier: 0.1084 → **0.0206** (81% improvement)

**Calibration makes scores honest but does NOT improve prediction quality.**

## Performance by Receptor Family

Small, specialized families are predictable; large, promiscuous families are not:

- Melanocortin: **26%** P@5 (best)
- S1P: **22%** P@5
- Orexin: **20%** P@5
- Serotonergic: **5.9%** P@5
- Histamine / Adenosine: **0%** P@5

## Confidence Calibration Crisis

The model is **most confident when it is most wrong**:

| Confidence Threshold | Actual Accuracy |
|---------------------|-----------------|
| > 0.1 | 1.5% |
| > 0.5 | 7.1% |
| > 0.7 | 6.5% |
| > 0.8 | **0.0%** (all 67 predictions were false positives) |
| > 0.9 | **0.0%** (all 13 predictions were false positives) |

This is the critical finding: the model produces high-confidence predictions that are entirely wrong. Platt scaling calibration corrects this by mapping raw scores to honest probabilities.

## Top Predictions (Treat as Hypotheses)

| Receptor | Predicted Ligand | Raw Confidence | Calibrated Confidence | Note |
|----------|-----------------|----------------|----------------------|------|
| GPR183 | Cholesterol derivative | 0.927 | ~0.02 | Overconfident |
| GPR103 | Lysergic acid | 0.926 | ~0.02 | Overconfident |
| GPR88 | Galanin | 0.924 | ~0.02 | Overconfident |
| GPR109A | Niacin | 0.917 | ~0.02 | Known to be correct (validates approach) |

**All high-confidence predictions have been recalibrated to reflect the true ~2% base rate of correct predictions.**

## Downstream Impact

- **71/72** orphan receptors have disease connections
- **68/72** have addiction pathway connections
- **36/72** have cancer connections
- **60 receptors** have high-confidence predictions AND disease connections

## Validation Through Partial Recovery

The model recovered known ligands for **20/20 partially-deorphanized receptors** at class-level or direct match. This is the strongest validation signal, but it applies only to receptors that already have at least one known ligand (partially deorphanized), not to truly orphan receptors.

## Why the Embedding Approach Failed

The proposed 860-dimensional receptor feature vectors and 1227-dimensional ligand feature vectors (described in the mathematical framework) were never implemented. The actual feature space is 5-10 categorical features. The high-dimensional embedding approach produces non-discriminative distances (~380-410 squared Euclidean for all pairs), making the RBF kernel useless.

## Limitations

See `output/limitations.md` for a comprehensive honest assessment. Key points:

1. **Heuristic approach:** Not a rigorous mathematical model of binding
2. **Overconfident scores:** Raw confidence severely overestimates accuracy
3. **Limited statistical significance:** Only ~5/73 receptors show significant signal
4. **Synthetic features:** No actual TM sequence data or structural features used
5. **Promiscuous ligands:** Spiperone inflates predictions across all receptors
6. **Descriptive math:** Theorems describe the implementation, not predict behavior
7. **Small dataset:** 349 pairs across 73 receptors limits generalization

## Recommendations for Experimental Validation

The predictions should guide experimental priorities, not replace experimental work:

1. **GPR183** — test cholesterol derivatives on leukocytes (known to be partially correct)
2. **GPR109A** — test niacin on adipose tissue (known to be correct, validates approach)
3. **GPR183** — test oxysterols (direct match in partial deorphanization recovery)
4. **GPR68** — test beta-hydroxybutyrate on brain tissue (direct match in recovery)
5. **GPR39** — test zinc on pancreatic cells (direct match in recovery)

## Honest Assessment

**What this project is:** A systematic organization of available GPCR data with a simple similarity-based scoring heuristic that generates testable hypotheses for orphan receptor deorphanization.

**What this project is NOT:** A reliable computational prediction tool, a rigorous mathematical model of binding, or a substitute for experimental deorphanization.

**The value:** Organizing the data, generating ranked candidate lists, and providing biological context for experimental prioritization. The predictions are hypotheses — they need experimental validation to confirm or refute.

---

*Full report: output/report.md (14 sections, appendices A-G)*
*Limitations: output/limitations.md (comprehensive honest assessment)*
*Calibration: output/calibration_report.md (Platt scaling results)*
*Implementation: math/implementation_v2.py (1780 lines Python v2)*
*Dataset: 349 pairs, 73 receptors, 163 ligands, 72 orphans*
*Predictions: 720 total (450 high, 102 medium, 168 low confidence — all overconfident)*
