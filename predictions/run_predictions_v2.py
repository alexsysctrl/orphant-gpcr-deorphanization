"""
Run v2 deorphanization predictions on all orphan receptors.

Uses the fixed implementation with all six improvements:
  1. Promiscuity penalty (spiperone downweighted)
  2. Family hints from orphan catalog
  3. Broadened matching (class-level → G-protein → ligand-class)
  4. Tissue-only fallback (GPR25, GPR164)
  5. Improved confidence scoring
  6. Natural ligand priority

Run from: ~/Desktop/deorphanization/
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'math'))

import importlib.util
spec = importlib.util.spec_from_file_location("implementation_v2",
    "/Users/alex/Desktop/deorphanization/math/implementation_v2.py")
impl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(impl)

load_all_data = impl.load_all_data
MetadataDeorphanizationPipelineV2 = impl.MetadataDeorphanizationPipelineV2


def run_predictions():
    print("=" * 70)
    print("DEORPHANIZATION v2 — PREDICTION RUN")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    data = load_all_data()
    print(f"  Raw receptor-ligand pairs: {len(data['raw_pairs'])}")
    print(f"  Orphan receptors: {len(data['orphans'])}")

    # Run pipeline
    print("\nRunning v2 deorphanization pipeline...")
    pipeline = MetadataDeorphanizationPipelineV2(data)
    results = pipeline.run()

    # Detailed per-receptor results
    print("\n" + "=" * 70)
    print("DETAILED RESULTS — ALL ORPHAN RECEPTORS")
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
            print(f"\n  No ligand predictions")

        if pred.top_gproteins:
            print(f"\n  Top G-Protein Coupling Predictions:")
            for rank, (gprot, prob) in enumerate(pred.top_gproteins[:3], 1):
                print(f"    {rank}. {gprot:<20} (prob={prob:.3f})")

    # Summary statistics
    print(f"\n{'=' * 70}")
    print("SUMMARY STATISTICS")
    print("=" * 70)

    total_high = sum(p.high_confidence_count for p in results)
    total_medium = sum(p.medium_confidence_count for p in results)
    total_low = sum(p.low_confidence_count for p in results)
    total_preds = total_high + total_medium + total_low

    print(f"\n  Total predictions: {total_preds}")
    print(f"  High confidence (>0.7): {total_high}")
    print(f"  Medium confidence (0.4-0.7): {total_medium}")
    print(f"  Low confidence (<0.4): {total_low}\n")

    non_zero = sum(1 for p in results if any(l[4] > 0 for l in p.top_ligands))
    zero_conf = sum(1 for p in results if all(l[4] == 0 for l in p.top_ligands))
    no_preds = sum(1 for p in results if not p.top_ligands)

    print(f"  Receptors with non-zero confidence: {non_zero}")
    print(f"  Receptors with only zero confidence: {zero_conf}")
    print(f"  Receptors with NO predictions: {no_preds}\n")

    # Top 15 most confident pairs
    all_pairs = []
    for pred in results:
        for ligand, prob, lclass, pki, conf in pred.top_ligands:
            all_pairs.append((pred.receptor_name, pred.gene, pred.family,
                              ligand, lclass, pki, conf))

    all_pairs.sort(key=lambda x: x[6], reverse=True)
    top15 = all_pairs[:15]

    print(f"  Top 15 Most Confident Orphan-Ligand Pairs:")
    print(f"  {'Rank':<5} {'Receptor':<22} {'Ligand':<30} {'Class':<18} {'pKi':>6} {'Conf':>6}")
    print(f"  {'-'*5} {'-'*22} {'-'*30} {'-'*18} {'-'*6} {'-'*6}")
    for rank, (receptor, gene, family, ligand, lclass, pki, conf) in enumerate(top15, 1):
        print(f"  {rank:<5} {receptor:<22} {ligand:<30} {lclass:<18} {pki:>6.1f} {conf:>6.3f}")

    # Partially-deorphanized recovery
    print(f"\n  Partially-Deorphanized Receptor Recovery:")
    check = {
        "GPR32": ["PGE2", "15-dPGJ2", "arachidonic acid"],
        "GPR119": ["OEA", "oleic acid", "vernonolide"],
        "GPR120": ["EPA", "DHA", "palmitic acid"],
        "GPR109A": ["niacin", "nicotinamide"],
        "GPR151": ["tryptophan", "phenylalanine"],
        "GPR39": ["zinc", "galanin"],
        "GPR65": ["H+ (protons)"],
        "GPR68": ["beta-hydroxybutyrate"],
    }
    for pred in results:
        if pred.receptor_name in check:
            known = check[pred.receptor_name]
            if pred.top_ligands:
                top = pred.top_ligands[0]
                hit = any(l in known for l, _, _, _, _ in pred.top_ligands)
                status = "✓ MATCH" if hit else "✗ NO MATCH"
                print(f"    {status} {pred.receptor_name}: "
                      f"known={known}, top={top[0]} (conf={top[4]:.3f})")
            else:
                print(f"    ✗ NO PREDICTION {pred.receptor_name}")

    # Receptors that were previously 0-confidence
    print(f"\n  Previously 0-Confidence Receptors (now fixed?):")
    prev_zero = ["GPR45", "GPR75", "GPR141", "GPR142", "GPR149", "GPR150",
                 "GPR152", "GPR153", "GPR155", "GPR157", "GPR173", "GPR176"]
    for pred in results:
        if pred.receptor_name in prev_zero:
            has_nonzero = any(l[4] > 0 for l in pred.top_ligands)
            status = "✓ FIXED" if has_nonzero else "✗ STILL ZERO"
            if pred.top_ligands:
                top = pred.top_ligands[0]
                print(f"    {status} {pred.receptor_name}: "
                      f"top={top[0]} (conf={top[4]:.3f}), "
                      f"n_similar={pred.n_similar_receptors}")
            else:
                print(f"    {status} {pred.receptor_name}: no predictions")

    # GPR25 and GPR164
    print(f"\n  GPR25/GPR164 (Fix 4 - tissue-only fallback):")
    for pred in results:
        if pred.receptor_name in ["GPR25", "GPR164", "GPR176"]:
            if pred.top_ligands:
                top = pred.top_ligands[0]
                print(f"    ✓ {pred.receptor_name}: top={top[0]} (conf={top[4]:.3f}, "
                      f"class={top[2]}, tissue={pred.tissue_expression})")
            else:
                print(f"    ✗ {pred.receptor_name}: no predictions")

    # Remaining issues
    print(f"\n  Remaining Issues:")
    remaining = [p for p in results
                 if p.high_confidence_count == 0 and p.medium_confidence_count == 0]
    if remaining:
        print(f"    {len(remaining)} receptors still have no high/medium confidence:")
        for pred in remaining[:10]:
            if pred.top_ligands:
                top = pred.top_ligands[0]
                print(f"      - {pred.receptor_name}: top={top[0]} (conf={top[4]:.3f})")
            else:
                print(f"      - {pred.receptor_name}: no predictions")
    else:
        print(f"    All receptors have at least one medium or high confidence prediction.")

    print(f"\n{'='*70}")
    print(f"Output files:")
    print(f"  predictions_v2.json: {os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output', 'predictions_v2.json')}")
    print(f"  predictions_v2_summary.md: {os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output', 'predictions_v2_summary.md')}")
    print("=" * 70)


if __name__ == "__main__":
    run_predictions()
