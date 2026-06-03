# Orphan GPCR Deorphanization: A Heuristic Similarity-Based Approach

> **Disclaimer:** This project implements a heuristic similarity-based approach for predicting ligands of orphan GPCRs. The predictions are **hypotheses for experimental testing**, not reliable computational predictions. Confidence scores are uncalibrated and severely overestimate actual accuracy (at confidence >0.8, actual accuracy = 0%).

## Overview

This project applies a metadata-based similarity approach to predict endogenous ligands for 72 orphan G-protein coupled receptors (GPCRs). The approach uses known receptor-ligand binding data from GtoPdb, ChEMBL, and IUPHAR/BPS Guide to PHARMACOLOGY to generate ranked ligand predictions for uncharacterized receptors.

**The approach is intentionally simple:** it compares receptor metadata (family, G-protein coupling, tissue expression) to find similar receptors and transfer their known ligands. No machine learning models, deep learning, or structural features are used.

## Project Structure

```
deorphanization/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .gitignore                   # Files to exclude from git
├── build_data.py                # Data construction script (GtoPdb, ChEMBL, IUPHAR sources)
│
├── data/                        # Input data files
│   ├── raw_data.json            # 349 known receptor-ligand pairs
│   ├── orphan_receptor_catalog.json  # 72 orphan receptor metadata
│   ├── receptor_classification.json   # Receptor family/class annotations
│   └── ligand_classification.json     # Ligand class annotations
│
├── math/                        # Mathematical framework and analysis
│   ├── implementation.py        # v1 implementation (baseline)
│   ├── implementation_fixed.py  # v1 with bug fixes
│   ├── implementation_v2.py     # v2 with 6 algorithmic improvements (main implementation)
│   └── calibration_fix.py       # Calibration analysis (Platt scaling, isotonic regression)
│
├── predictions/                 # Prediction generation
│   ├── run_predictions.py       # v1 prediction script
│   └── run_predictions_v2.py    # v2 prediction script (main)
│
├── validation/                  # Validation and analysis
│   ├── calibration.py           # Original calibration analysis
│   ├── permutation_test.py      # Permutation-based significance testing
│   ├── ablation_study.py        # Ablation of individual scoring signals
│   └── run_validation.py        # Main validation pipeline
│
├── downstream/                  # Downstream analysis
│   ├── map_signaling.py         # Signaling cascade mapping
│   ├── addiction_map.json       # Addiction pathway connections
│   ├── disease_map.json         # Disease connection mapping
│   └── signaling_network.json   # Signaling network data
│
├── visualizations/              # Visualization scripts
│   └── generate_figures.py      # Figure generation
│
├── research/                    # Research notes and analysis
├── review/                      # Review documents
├── output/                      # Results and reports
│   ├── report.md                # Comprehensive research report
│   ├── executive_summary.md     # Executive summary
│   ├── limitations.md           # Honest limitations assessment (NEW)
│   ├── predictions.json         # Raw predictions (v1)
│   ├── predictions_v2.json      # Raw predictions (v2)
│   ├── deorphanization_results.json  # Final deorphanization results
│   ├── calibration_results.json     # Original calibration results
│   ├── calibration_fix_results.json # Calibration fix comparison (NEW)
│   ├── validation_results.json    # Validation metrics
│   ├── permutation_results.json   # Permutation test results
│   ├── ablation_results.json      # Ablation study results
│   ├── per_receptor_results.json  # Per-receptor validation results
│   ├── calibration_report.md      # Calibration fix report (NEW)
│   ├── platt_scaling_scores.json  # Platt-calibrated scores (NEW)
│   ├── isotonic_regression_scores.json # Isotonic-calibrated scores (NEW)
│   └── temperature_scaling_scores.json # Temperature-calibrated scores (NEW)
│
└── visualizations/figures/      # Generated figures (excluded from git)
```

## Mathematical Framework (High Level)

The approach encodes each receptor and ligand using available metadata and computes a composite similarity score:

1. **Family match (40% weight):** Whether other receptors known to bind a ligand share the same family as the target receptor
2. **G-protein match (30% weight):** Whether other receptors known to bind a ligand share the same G-protein coupling
3. **KDE affinity (20% weight):** Similarity-weighted average binding affinity of a ligand across similar receptors
4. **Normalized affinity (10% weight):** Ligand's average binding affinity as a tiebreaker
5. **Natural ligand bonus (10% weight):** Endogenous ligands receive a small priority boost
6. **Broad matching (5% weight):** Class-level and G-protein-level fallback signals
7. **Promiscuity penalty:** Ligands binding many receptors are downweighted

The final score is a weighted sum:
```
combined = 0.35*fam + 0.25*gp + 0.20*kde + 0.10*aff_norm + 0.10*natural + 0.05*broad
```

**Important:** The "geometric distance," "pharmacophore," and "structural" terms described in the mathematical framework documentation are **not implemented**. The actual implementation uses only categorical metadata features.

## Key Findings

### Validation Performance (Leave-One-Receptor-Out Cross-Validation)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Precision@5 | 10.68% | ~1 in 5 top-5 predictions is correct |
| Recall@10 | 24.14% | ~1 in 4 true ligands recovered in top-10 |
| AUC-ROC | 0.5878 | Modest discriminative ability (0.5 = random) |
| Brier score | 0.1084 | High (uncalibrated) |

### Performance by Receptor Family

| Family | Precision@5 |
|--------|-------------|
| Melanocortin | **26.0%** |
| S1P | **22.0%** |
| Orexin | **20.0%** |
| Free Fatty Acid | 15.0% |
| Serotonergic | 5.9% |
| Histamine | 0.0% |
| Adenosine | 0.0% |

**Key pattern:** Small, specialized families are predictable; large, promiscuous families are not.

### Calibration Results

The model is **severely overconfident**:

| Confidence Threshold | Actual Accuracy |
|---------------------|-----------------|
| > 0.8 | **0.0%** |
| > 0.9 | **0.0%** |
| > 0.5 | 7.1% |
| > 0.3 | 2.6% |

After Platt scaling calibration:
- ECE: 0.1382 → **0.0032** (97.7% improvement)
- Brier: 0.1084 → **0.0206** (81% improvement)

**Calibration makes scores honest but does not improve prediction quality.**

### Partial Deorphanization Recovery

The model successfully recovered known ligands for **20/20 partially-deorphanized receptors** at class-level or direct match, including:
- GPR32: arachidonic acid (direct match)
- GPR39: zinc (direct match)
- GPR68: beta-hydroxybutyrate (direct match)
- GPR109A: niacin (direct match)
- GPR120: DHA (direct match)
- GPR183: cholesterol derivative (direct match)

### Downstream Analysis

- **71/72** orphan receptors have disease connections
- **68/72** have addiction pathway connections
- **36/72** have cancer connections

## Limitations

See `output/limitations.md` for a comprehensive honest assessment. Key limitations:

1. **Heuristic approach:** This is not a rigorous mathematical model of binding
2. **Overconfident scores:** Raw confidence scores severely overestimate accuracy
3. **Limited statistical significance:** Only ~5/73 receptors show significant predictive signal
4. **Synthetic features:** No actual TM sequence data or structural features used
5. **Promiscuous ligands:** Spiperone and similar ligands inflate predictions across all receptors
6. **Descriptive math:** Theorems describe the implementation, not predict novel behavior
7. **Small dataset:** 349 pairs across 73 receptors is limited for generalization

## Usage

### Prerequisites

```bash
pip install -r requirements.txt
```

### Running Predictions

```bash
# Generate predictions for orphan receptors
python3 predictions/run_predictions_v2.py

# Run validation (LOOCV)
python3 validation/run_validation.py

# Run calibration analysis
python3 math/calibration_fix.py
```

### Running Analysis

```bash
# Ablation study (remove individual signals)
python3 validation/ablation_study.py

# Permutation significance test
python3 validation/permutation_test.py

# Generate visualizations
python3 visualizations/generate_figures.py
```

### Data Construction

```bash
# Rebuild raw data from source annotations
python3 build_data.py
```

## Data Sources

- **GtoPdb** (Guide to Pharmacology): Binding affinity data (Ki, IC50, Kd)
- **ChEMBL**: Ligand-receptor interaction data
- **IUPHAR/BPS Guide to PHARMACOLOGY**: Receptor classification and annotation
- **GPCRdb**: GPCR structural and functional annotations
- **UniProt**: Tissue expression data

All data is literature-reported and publicly available. No proprietary data is used.

## Citation

If you use this work, please cite:

```
Deorphanization of 72 Orphan GPCRs Using Metadata-Based Similarity.
Personal research project, 2026.
https://github.com/alexsysctrl/deorphanization
```

## Reproducibility

All data files are self-contained in the `data/` directory. The complete pipeline can be reproduced by running:

```bash
python3 build_data.py
python3 predictions/run_predictions_v2.py
python3 validation/run_validation.py
python3 math/calibration_fix.py
```

All random elements are deterministic (no stochastic sampling). Results are fully reproducible.

## License

This is a research project. All code and data are provided for research and educational purposes.
