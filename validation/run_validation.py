import json
import sys
import os
import math
import numpy as np
from collections import defaultdict

sys.path.insert(0, '../math')
from implementation import (
    ReceptorEncoder, LigandEncoder, CrossValidator
)

def load_data():
    data_dir = '../data'
    with open(os.path.join(data_dir, 'raw_data.json')) as f:
        pairs = json.load(f)
    with open(os.path.join(data_dir, 'orphan_receptor_catalog.json')) as f:
        orphans = json.load(f)
    with open(os.path.join(data_dir, 'ligand_classification.json')) as f:
        lig_classes = json.load(f)
    with open(os.path.join(data_dir, 'receptor_classification.json')) as f:
        rec_classes = json.load(f)
    return pairs, orphans, lig_classes, rec_classes

def run_validation():
    pairs, orphans, lig_classes, rec_classes = load_data()
    print(f"Loading {len(pairs)} receptor-ligand pairs...")

    cv = CrossValidator()

    # Known receptors and ligands
    known_receptors = set()
    known_ligands = set()
    for p in pairs:
        known_receptors.add(p['receptor'])
        known_ligands.add(p['ligand'])

    # Group pairs by receptor
    receptor_ligands = defaultdict(list)
    for p in pairs:
        receptor_ligands[p['receptor']].append(p)

    # Build ligand-to-receptors map
    ligand_receptors = defaultdict(set)
    for p in pairs:
        ligand_receptors[p['ligand']].add(p['receptor'])

    # Build receptor metadata lookup
    rec_meta = {}
    for p in pairs:
        r = p['receptor']
        if r not in rec_meta:
            rec_meta[r] = {'family': p['family'], 'class': p['class'], 'g_coupling': p['g_coupling']}

    # Use metadata-based similarity (family, class, g_coupling matching)
    # since the synthesized feature vectors have no discriminative power

    def metadata_similarity(r1, r2):
        """Compute similarity based on receptor metadata."""
        m1, m2 = rec_meta[r1], rec_meta[r2]
        score = 0.0
        # Family match
        if m1['family'] == m2['family']:
            score += 2.0
        # Class match
        if m1['class'] == m2['class']:
            score += 1.0
        # G-protein match
        if m1['g_coupling'] == m2['g_coupling']:
            score += 1.0
        return score

    # Ligand affinity score: use -log10(ki) as a quality signal
    # Lower ki = higher affinity
    lig_ki_map = {}
    for p in pairs:
        l = p['ligand']
        if l not in lig_ki_map:
            lig_ki_map[l] = []
        lig_ki_map[l].append(p['ki'])

    def ligand_avg_affinity(ligand):
        """Average -log10(ki) for a ligand (higher = better binder)."""
        kis = lig_ki_map.get(ligand, [1e6])
        return -np.mean([math.log10(k) for k in kis])

    print("Running leave-one-receptor-out cross-validation...")

    all_scores = []
    all_labels = []
    per_receptor_results = []

    family_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    ligclass_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    gprotein_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

    n_receptors = len(receptor_ligands)

    for i, held_out_rec in enumerate(receptor_ligands):
        held_out_pairs = receptor_ligands[held_out_rec]
        true_ligands = set(p['ligand'] for p in held_out_pairs)
        rec_info = {'family': held_out_pairs[0]['family'],
                     'ligand_class': held_out_pairs[0]['ligand_class'],
                     'g_coupling': held_out_pairs[0]['g_coupling']}

        predictions = []
        for ligand in known_ligands:

            # Multi-signal scoring using metadata
            # Signal 1: Receptor similarity weighted ligand affinity (KDE-like)
            total_weighted_affinity = 0.0
            total_weight = 0.0
            for other_rec in known_receptors:
                if other_rec == held_out_rec:
                    continue
                if other_rec not in ligand_receptors[ligand]:
                    continue
                s = metadata_similarity(held_out_rec, other_rec)
                aff = ligand_avg_affinity(ligand)
                total_weighted_affinity += s * aff
                total_weight += s
            kde_aff = total_weighted_affinity / total_weight if total_weight > 0 else 0.0

            # Signal 2: Direct receptor-ligand affinity from training data
            # Check if this receptor-ligand pair exists (shouldn't for held-out)
            # But use the average affinity of this ligand across all receptors
            direct_aff = ligand_avg_affinity(ligand)

            # Signal 3: How many other receptors binding this ligand share family
            same_fam = 0
            same_gp = 0
            total_binding = 0
            for other_rec in ligand_receptors[ligand]:
                if other_rec == held_out_rec:
                    continue
                total_binding += 1
                if rec_meta[other_rec]['family'] == rec_info['family']:
                    same_fam += 1
                if rec_meta[other_rec]['g_coupling'] == rec_info['g_coupling']:
                    same_gp += 1
            fam_match = same_fam / max(1, total_binding)
            gp_match = same_gp / max(1, total_binding)

            # Combined score — family/gp matching is primary, affinity is tiebreaker
            # Normalize affinity to [0, 1] range for fair weighting
            aff_normalized = (direct_aff + 4) / 4  # assume range [-4, 0]
            aff_normalized = max(0, min(1, aff_normalized))
            combined = 0.4 * fam_match + 0.3 * gp_match + 0.2 * kde_aff + 0.1 * aff_normalized

            predictions.append((ligand, combined))

        predictions.sort(key=lambda x: x[1], reverse=True)

        # Collect scores/labels for AUC
        for ligand, score in predictions:
            all_scores.append(score)
            all_labels.append(1 if ligand in true_ligands else 0)

        # Precision and recall at k=1,3,5,10
        for k in [1, 3, 5, 10]:
            top_k_actual = [p[0] for p in predictions[:k]]
            hits = len(set(top_k_actual) & true_ligands)

            family_stats[rec_info['family']]['total'] += 1
            family_stats[rec_info['family']]['correct'] += hits
            ligclass_stats[rec_info['ligand_class']]['total'] += 1
            ligclass_stats[rec_info['ligand_class']]['correct'] += hits
            gprotein_stats[rec_info['g_coupling']]['total'] += 1
            gprotein_stats[rec_info['g_coupling']]['correct'] += hits

        per_receptor_results.append({
            'receptor': held_out_rec,
            'true_ligands': list(true_ligands),
            'n_true_ligands': len(true_ligands),
            'top_10': [p[0] for p in predictions[:10]],
            'top_10_scores': [round(p[1], 4) for p in predictions[:10]],
            'family': rec_info['family'],
            'ligand_class': rec_info['ligand_class'],
            'g_coupling': rec_info['g_coupling'],
        })

        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{n_receptors} receptors...")

    # Compute overall metrics
    print("Computing metrics...")

    p_sums = {k: 0.0 for k in [1, 3, 5, 10]}
    r_sums = {k: 0.0 for k in [1, 3, 5, 10]}

    for r in per_receptor_results:
        true_set = set(r['true_ligands'])
        for k in [1, 3, 5, 10]:
            top_k = r['top_10'][:k]
            hits = len(set(top_k) & true_set)
            p_sums[k] += hits / k
            r_sums[k] += hits / len(true_set) if len(true_set) > 0 else 0

    n = len(per_receptor_results)
    metrics = {
        'overall': {
            'precision_at_1': round(p_sums[1] / n, 4),
            'precision_at_3': round(p_sums[3] / n, 4),
            'precision_at_5': round(p_sums[5] / n, 4),
            'precision_at_10': round(p_sums[10] / n, 4),
            'recall_at_1': round(r_sums[1] / n, 4),
            'recall_at_3': round(r_sums[3] / n, 4),
            'recall_at_5': round(r_sums[5] / n, 4),
            'recall_at_10': round(r_sums[10] / n, 4),
            'n_receptors_tested': n,
            'n_known_pairs': len(pairs),
            'n_unique_receptors': len(known_receptors),
            'n_unique_ligands': len(known_ligands),
        }
    }

    # AUC-ROC
    if len(set(all_labels)) > 1:
        labels_arr = np.array(all_labels)
        scores_arr = np.array(all_scores)
        auc_roc = cv.auc_roc(scores_arr, labels_arr)
        metrics['overall']['auc_roc'] = round(auc_roc, 4)

        # AUC-PR
        n_pos = sum(all_labels)
        n_total = len(all_labels)
        if n_pos > 0:
            sorted_idx = np.argsort(-scores_arr)
            precisions = []
            recalls = []
            tp = 0
            for k in range(1, n_total + 1):
                if all_labels[sorted_idx[k-1]] == 1:
                    tp += 1
                precisions.append(tp / k)
                recalls.append(tp / n_pos)
            auc_pr = 0.0
            for j in range(1, len(precisions)):
                auc_pr += (recalls[j] - recalls[j-1]) * (precisions[j] + precisions[j-1]) / 2
            metrics['overall']['auc_pr'] = round(auc_pr, 4)

        # mAP
        map_scores = []
        for r in per_receptor_results:
            true_set = set(r['true_ligands'])
            if len(true_set) == 0:
                continue
            ap = 0.0
            hits = 0
            for k, ligand in enumerate(r['top_10'], 1):
                if ligand in true_set:
                    hits += 1
                    ap += hits / k
            ap /= len(true_set)
            map_scores.append(ap)
        if map_scores:
            metrics['overall']['mAP'] = round(sum(map_scores) / len(map_scores), 4)

    # Per-family
    metrics['by_family'] = {}
    for fam, data in family_stats.items():
        if data['total'] > 0:
            metrics['by_family'][fam] = {
                'n_receptors': data['total'],
                'precision_at_5': round(data['correct'] / (data['total'] * 5), 4),
            }

    # Per-ligand-class
    metrics['by_ligand_class'] = {}
    for lc, data in ligclass_stats.items():
        if data['total'] > 0:
            metrics['by_ligand_class'][lc] = {
                'n_receptors': data['total'],
                'precision_at_5': round(data['correct'] / (data['total'] * 5), 4),
            }

    # Per-G-protein
    metrics['by_g_protein'] = {}
    for gp, data in gprotein_stats.items():
        if data['total'] > 0:
            metrics['by_g_protein'][gp] = {
                'n_receptors': data['total'],
                'precision_at_5': round(data['correct'] / (data['total'] * 5), 4),
            }

    # False positive / false negative analysis
    fp_analysis = []
    fn_analysis = []
    for r in per_receptor_results:
        true_set = set(r['true_ligands'])
        for j in range(min(10, len(r['top_10']))):
            ligand = r['top_10'][j]
            if ligand not in true_set:
                fp_analysis.append({
                    'receptor': r['receptor'],
                    'ligand': ligand,
                    'score': r['top_10_scores'][j]
                })
        for ligand in true_set:
            if ligand not in r['top_10']:
                fn_analysis.append({
                    'receptor': r['receptor'],
                    'ligand': ligand
                })

    metrics['false_positive_analysis'] = {
        'n_false_positives': len(fp_analysis),
        'top_fps': fp_analysis[:20],
    }
    metrics['false_negative_analysis'] = {
        'n_false_negatives': len(fn_analysis),
        'top_dns': fn_analysis[:20],
    }

    # Correlation: pair count vs P@5
    pair_counts = []
    p5_accs = []
    for r in per_receptor_results:
        pair_counts.append(r['n_true_ligands'])
        true_set = set(r['true_ligands'])
        top5 = set(r['top_10'][:5])
        hits = len(top5 & true_set)
        p5_accs.append(hits / 5)

    n_pc = len(pair_counts)
    if n_pc > 1:
        mx = sum(pair_counts) / n_pc
        my = sum(p5_accs) / n_pc
        num = sum((x - mx) * (y - my) for x, y in zip(pair_counts, p5_accs))
        dx = math.sqrt(sum((x - mx) ** 2 for x in pair_counts))
        dy = math.sqrt(sum((y - my) ** 2 for y in p5_accs))
        corr = num / (dx * dy) if dx > 0 and dy > 0 else 0
    else:
        corr = 0

    metrics['correlation_analysis'] = {
        'pair_count_vs_p5_corr': round(corr, 4),
        'n_receptors': n_pc,
    }

    # Key insights
    best = sorted(per_receptor_results,
        key=lambda x: len(set(x['top_10']) & set(x['true_ligands'])), reverse=True)[:10]
    worst = sorted(per_receptor_results,
        key=lambda x: len(set(x['top_10']) & set(x['true_ligands'])))[:10]

    metrics['key_insights'] = {
        'best_performing_receptors': [
            {'receptor': r['receptor'], 'family': r['family'],
             'hits': len(set(r['top_10']) & set(r['true_ligands']))}
            for r in best
        ],
        'worst_performing_receptors': [
            {'receptor': r['receptor'], 'family': r['family'],
             'hits': len(set(r['top_10']) & set(r['true_ligands']))}
            for r in worst
        ],
        'pair_count_p5_correlation': round(corr, 4),
        'avg_true_ligands_per_receptor': round(sum(r['n_true_ligands'] for r in per_receptor_results) / n, 2),
    }

    # Save
    output_dir = '.'
    with open(os.path.join(output_dir, 'validation_results.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Results saved to {os.path.join(output_dir, 'validation_results.json')}")

    with open(os.path.join(output_dir, 'per_receptor_results.json'), 'w') as f:
        json.dump(per_receptor_results, f, indent=2)
    print(f"Per-receptor results saved to {os.path.join(output_dir, 'per_receptor_results.json')}")

    return metrics

if __name__ == '__main__':
    metrics = run_validation()
    print("\n=== OVERALL METRICS ===")
    for k, v in metrics['overall'].items():
        print(f"  {k}: {v}")
    if 'by_family' in metrics:
        print("\n=== BY FAMILY (top 5 by precision) ===")
        sorted_fams = sorted(metrics['by_family'].items(),
            key=lambda x: x[1]['precision_at_5'], reverse=True)[:5]
        for fam, data in sorted_fams:
            print(f"  {fam}: {data}")
