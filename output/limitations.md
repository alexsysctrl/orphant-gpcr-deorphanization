# Limitations and Honest Assessment

## 1. This Is a Heuristic Approach, Not a Rigorous Mathematical Model

The framework described in this project is a **heuristic similarity-based approach**, not a rigorous mathematical model of receptor-ligand binding. The "theorems" and "propositions" in the mathematical framework are **descriptive statements** about the implementation — they describe what the code does, not predict novel behavior. The mathematical notation provides a formal language for describing the scoring function, but the function itself is constructed from hand-tuned weights on metadata features, not derived from biophysical principles.

The core scoring function:

```
combined = 0.35 * fam_match + 0.25 * gp_match + 0.20 * kde_aff + 0.10 * aff_normalized + 0.10 * natural_bonus + 0.05 * broad_bonus
```

This is a weighted average of five signals, where the weights (0.35, 0.25, 0.20, 0.10, 0.10, 0.05) were chosen heuristically, not optimized through principled learning. The "geometric distance," "pharmacophore compatibility," "G-protein coupling," and "family modulation" terms described in the mathematical framework are **not independently implemented** — they are collapsed into the five signals above.

## 2. Confidence Scores Are Uncalibrated and Overestimate Actual Accuracy

The raw confidence scores produced by the model are **severely overconfident**:

| Confidence Threshold | Actual Accuracy |
|---------------------|-----------------|
| > 0.1 | 1.5% |
| > 0.3 | 2.6% |
| > 0.5 | 7.1% |
| > 0.7 | 6.5% |
| > 0.8 | **0.0%** |
| > 0.9 | **0.0%** |

At confidence >0.8, **every single prediction was a false positive**. The model is most confident when it is most wrong.

After applying Platt scaling calibration (the best calibration method tested):
- ECE improved from 0.1382 to 0.0032 (97.7% improvement)
- Brier score improved from 0.1084 to 0.0206 (81% improvement)

However, **calibration does not fix the underlying prediction quality**. Even with calibrated scores, the model's precision at top-5 is only 10.68% — meaning 9 out of 10 top predictions are wrong. The calibrated scores are now *honest* about this poor performance, but they do not make the predictions better.

## 3. Only 5/73 Receptors Show Statistically Significant Predictions

Of the 73 receptors tested in leave-one-receptor-out cross-validation, only a small fraction achieve performance significantly above random chance. The majority of receptors show precision at top-5 near or below the random baseline (~0.6%), indicating that the model provides no meaningful signal for most orphan receptors.

The best-performing families (melanocortin 26%, S1P 22%, orexin 20%) benefit from tight ligand specificity within small families. Large families with promiscuous cross-binding (serotonergic 5.9%, histamine 0%, adenosine 0%) provide essentially no discriminative signal.

## 4. The Framework Provides Hypotheses for Experimental Testing, Not Definitive Predictions

All predictions should be treated as **hypotheses for experimental validation**, not as reliable computational predictions. The framework's value lies in:

- Generating a ranked list of candidate ligands for each orphan receptor
- Providing biological context (disease connections, tissue expression, G-protein coupling)
- Identifying which receptors are most promising for experimental deorphanization

The predictions are **not** reliable enough to guide drug discovery decisions without experimental validation.

## 5. Feature Encoding Is Synthetic — No Actual TM Sequence Data Used

The mathematical framework describes receptor features in terms of amino acid sequences (450-dimensional BLOSUM62 pairwise alignment), binding pocket geometry, and conservation scores. **None of these features are actually computed from real data.**

The actual implementation uses only:
- Receptor family label (categorical)
- G-protein coupling type (categorical: Gs, Gi, Gq, G12/13)
- Receptor class (categorical: A, B, C)
- Tissue expression (categorical, simplified)
- Ligand class (categorical: 14 categories)
- Ligand affinity (pKi values from GtoPdb/ChEMBL)

The 860-dimensional and 1227-dimensional feature vectors described in the mathematical framework are **not implemented**. They were proposed as a future direction but never realized. The actual feature space is effectively 5-10 dimensional categorical features.

## 6. Promiscuous Ligands (Spiperone) Were Problematic

The synthetic ligand **spiperone** appears as a top prediction for nearly every orphan receptor in the dataset, regardless of family or tissue expression. This is because spiperone is an extremely promiscuous antipsychotic that binds to dozens of receptor subtypes across multiple families.

A promiscuity penalty was added in v2 to downweight such ligands, but the penalty is insufficient to eliminate spiperone from top predictions. This is a known limitation: promiscuous ligands act as "universal predictors" that inflate confidence scores without providing specific information.

## 7. The Math Framework Describes the Implementation, Not Novel Predictions

The mathematical framework (74 equations, 5 theorems, 60 definitions) provides a formal description of the heuristic approach. However:

- The "binding affinity function" (Eq. 14) is not actually implemented as described — the four terms are collapsed into a single weighted sum
- The "kernel density estimation" (Eq. 29) is not implemented as a true KDE — it uses simple similarity-weighted averaging
- The "confidence score" (Eq. 30) is not derived from the affinity function — it uses a separate heuristic formula
- The theorems about "boundedness," "positive semi-definiteness," and "kinetic consistency" are **trivial mathematical facts** about the chosen functions, not insights about receptor-ligand binding

The mathematical framework serves as **formal documentation** of the implementation, not as a predictive theory.

## 8. Summary Assessment

| Aspect | Assessment |
|--------|-----------|
| Prediction quality | Poor (P@5 = 10.68%, AUC-ROC = 0.5878) |
| Confidence calibration | Severely overconfident; requires Platt scaling correction |
| Statistical significance | Only ~5/73 receptors show significant signal |
| Feature richness | Minimal (5-10 categorical features, not 860/1227 dimensional) |
| Mathematical rigor | Descriptive formalization, not predictive theory |
| Practical utility | Generates hypotheses for experimental testing |
| Best use case | Prioritizing which orphan receptors to study experimentally |

**The framework is a starting point for systematic orphan receptor deorphanization, not a solution.** Its value is in organizing the available data and generating testable hypotheses, not in providing reliable computational predictions.
