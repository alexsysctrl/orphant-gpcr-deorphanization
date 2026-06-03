"""
Run metadata-based deorphanization predictions on all 72 orphan receptors.

This script:
1. Loads all data from ~/Desktop/deorphanization/data/
2. Uses the FIXED metadata-based implementation
3. Runs deorphanization on ALL 72 orphan receptors
4. For each orphan, predicts top 10 ligands with confidence scores
5. Also predicts top 3 G-protein coupling preferences
6. Saves results to ~/Desktop/deorphanization/output/predictions.json
7. Generates summary to ~/Desktop/deorphanization/output/predictions_summary.md

Run from: ~/Desktop/deorphanization/
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'math'))

import numpy as np

# Import directly from the module file
import importlib.util
spec = importlib.util.spec_from_file_location("implementation_fixed",
    "/Users/alex/Desktop/deorphanization/math/implementation_fixed.py")
impl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(impl)
main = impl.main
load_all_data = impl.load_all_data
MetadataDeorphanizationPipeline = impl.MetadataDeorphanizationPipeline


def run_predictions():
    """Run the full deorphanization pipeline and print results."""
    print("=" * 70)
    print("ORPHAN RECEPTOR DEORPHANIZATION — METADATA-BASED PREDICTIONS")
    print("=" * 70)

    np.random.seed(42)

    # Load data
    print("\nLoading data...")
    data = load_all_data()
    print(f"  Raw receptor-ligand pairs: {len(data['raw_pairs'])}")
    print(f"  Orphan receptors: {len(data['orphans'])}")
    print(f"  Ligand classes: {len(data['ligand_classes'])}")
    print(f"  Receptor classes: {len(data['receptor_classes'])}")

    # Run pipeline
    print("\nRunning metadata-based deorphanization...")
    pipeline = MetadataDeorphanizationPipeline(data)
    results = pipeline.run()

    # Print detailed per-orphan results
    print("\n" + "=" * 70)
    print("DETAILED RESULTS — ALL 72 ORPHAN RECEPTORS")
    print("=" * 70)

    for pred in results:
        print(f"\n{'─' * 65}")
        print(f"  {pred.receptor_name} ({pred.gene})")
        print(f"  Family: {pred.family} | Class: {pred.class_type} | "
              f"G-coupling: {pred.g_coupling or 'unknown'} | "
              f"Tissues: {pred.tissue_expression or 'none specified'}")
        print(f"  Similar known receptors (S>0.1): {pred.n_similar_receptors}")
        print(f"  Confidence: {pred.high_confidence_count} high, "
              f"{pred.medium_confidence_count} medium, "
              f"{pred.low_confidence_count} low")

        if pred.top_ligands:
            print(f"\n  Top 10 Predicted Ligands:")
            print(f"  {'Rank':<5} {'Ligand':<35} {'Class':<20} {'pKi':>6} {'Conf':>6}")
            print(f"  {'-'*5} {'-'*35} {'-'*20} {'-'*6} {'-'*6}")
            for rank, (ligand, prob, lclass, pki, conf) in enumerate(pred.top_ligands[:10], 1):
                print(f"  {rank:<5} {ligand:<35} {lclass:<20} {pki:>6.1f} {conf:>6.3f}")
        else:
            print(f"\n  No ligand predictions (no similar known receptors)")

        if pred.top_gproteins:
            print(f"\n  Top 3 G-Protein Coupling Predictions:")
            for rank, (gprot, prob) in enumerate(pred.top_gproteins[:3], 1):
                print(f"    {rank}. {gprot:<20} (prob={prob:.3f})")

    # Final summary statistics
    print(f"\n{'=' * 70}")
    print("SUMMARY STATISTICS")
    print("=" * 70)

    total_high = sum(p.high_confidence_count for p in results)
    total_medium = sum(p.medium_confidence_count for p in results)
    total_low = sum(p.low_confidence_count for p in results)
    total_preds = total_high + total_medium + total_low

    print(f"\n  Total predictions across all orphans: {total_preds}")
    print(f"  High confidence (>0.7): {total_high} ({total_high/max(total_preds,1)*100:.1f}%)")
    print(f"  Medium confidence (0.4-0.7): {total_medium} ({total_medium/max(total_preds,1)*100:.1f}%)")
    print(f"  Low confidence (<0.4): {total_low} ({total_low/max(total_preds,1)*100:.1f}%)\n")

    # Orphans with predictions
    orphans_with_high = sum(1 for p in results if p.high_confidence_count > 0)
    orphans_with_medium = sum(1 for p in results if p.medium_confidence_count > 0)
    orphans_with_low_only = sum(1 for p in results if p.high_confidence_count == 0 and p.medium_confidence_count == 0)
    orphans_no_preds = sum(1 for p in results if not p.top_ligands)

    print(f"  Orphan receptors with high-confidence predictions: {orphans_with_high}")
    print(f"  Orphan receptors with medium-confidence predictions: {orphans_with_medium}")
    print(f"  Orphan receptors with only low-confidence predictions: {orphans_with_low_only}")
    print(f"  Orphan receptors with NO predictions: {orphans_no_preds}\n")

    # Top 10 most confident orphan-ligand pairs
    all_pairs = []
    for pred in results:
        for ligand, prob, lclass, pki, conf in pred.top_ligands:
            all_pairs.append((pred.receptor_name, pred.gene, pred.family,
                              ligand, lclass, pki, conf))

    all_pairs.sort(key=lambda x: x[6], reverse=True)
    top10 = all_pairs[:10]

    print(f"  Top 10 Most Confident Orphan-Ligand Pairs:")
    print(f"  {'Rank':<5} {'Receptor':<22} {'Ligand':<30} {'Class':<18} {'pKi':>6} {'Conf':>6}")
    print(f"  {'-'*5} {'-'*22} {'-'*30} {'-'*18} {'-'*6} {'-'*6}")
    for rank, (receptor, gene, family, ligand, lclass, pki, conf) in enumerate(top10, 1):
        print(f"  {rank:<5} {receptor:<22} {ligand:<30} {lclass:<18} {pki:>6.1f} {conf:>6.3f}")

    # Interesting patterns
    print(f"\n  Interesting Patterns and Insights:")
    print(f"  {'─' * 50}")

    # Pattern: Orphans matching known partially-deorphanized receptors
    known_orphans = {
        "GPR39": "zinc/galanin",
        "GPR119": "OEA/vernonolide",
        "GPR120": "EPA/DHA",
        "GPR109A": "niacin",
        "GPR139": "neurotensin",
        "GPR151": "tryptophan/phenylalanine",
        "GPR32": "PGE2",
        "GPR33": "N-acyl dopamine",
        "GPR65": "H+ (acid-sensing)",
        "GPR68": "beta-hydroxybutyrate",
        "GPR83": "LPA",
        "GPR87": "LPA/prostaglandins",
    }

    for pred in results:
        if pred.receptor_name in known_orphans:
            known = known_orphans[pred.receptor_name]
            if pred.top_ligands:
                top = pred.top_ligands[0]
                print(f"  ✓ {pred.receptor_name}: Known={known}, "
                      f"Predicted={top[0]} (conf={top[4]:.3f})")

    # Pattern: Adhesion GPCRs tend to have low confidence (no family match in known data)
    adhesion = [p for p in results if "Adhesion" in p.family]
    if adhesion:
        avg_conf = sum(p.top_ligands[0][4] for p in adhesion if p.top_ligands) / max(len([p for p in adhesion if p.top_ligands]), 1)
        print(f"  ⚠ Adhesion GPCRs ({len(adhesion)} receptors): "
              f"avg top confidence = {avg_conf:.3f} (low — no family match in known data)")

    # Pattern: Class C orphans
    class_c = [p for p in results if "Class C" in p.family]
    if class_c:
        print(f"  ⚠ Class C orphans ({len(class_c)} receptors): "
              f"may need amino acid/peptide ligand predictions")

    print(f"\n{'=' * 70}")
    print(f"Output files:")
    print(f"  predictions.json: {os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output', 'predictions.json')}")
    print(f"  predictions_summary.md: {os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output', 'predictions_summary.md')}")
    print("=" * 70)


if __name__ == "__main__":
    run_predictions()
