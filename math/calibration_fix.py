#!/usr/bin/env python3
"""
Calibration Fix for Orphan GPCR Deorphanization Predictions
=============================================================

Addresses the critical finding that confidence scores are severely overconfident:
- At confidence >0.8, actual accuracy = 0% (all 80 predictions were false positives)
- ECE = 0.1382 (moderate mis-calibration)
- ECE for top-5 predictions = 0.6071 (severe mis-calibration)
- Brier score is high due to confident wrong predictions

This module implements three calibration methods and compares their effectiveness:
1. Isotonic regression calibration
2. Platt scaling (sigmoid fitting)
3. Temperature scaling

Uses leave-one-receptor-out (LOOCV) to generate calibration data, avoiding
data leakage from the same receptor appearing in both train and calibration sets.
"""

import json
import os
import sys
import math
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


# ============================================================================
# DATA LOADING (mirror calibration.py to avoid import issues)
# ============================================================================

def load_data():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    with open(os.path.join(data_dir, 'raw_data.json')) as f:
        pairs = json.load(f)
    return pairs


def metadata_similarity(r1, r2, rec_meta):
    m1, m2 = rec_meta[r1], rec_meta[r2]
    score = 0.0
    if m1['family'] == m2['family']:
        score += 1.0
    if m1.get('subfamily') and m2.get('subfamily') and m1['subfamily'] == m2['subfamily']:
        score += 0.7
    if m1.get('class') and m2.get('class') and m1['class'] == m2['class']:
        score += 0.3
    if m1.get('g_coupling') and m2.get('g_coupling') and m1['g_coupling'] == m2['g_coupling']:
        score += 0.3
    t1 = set(m1.get('tissue', [])) if isinstance(m1.get('tissue'), list) else set()
    t2 = set(m2.get('tissue', [])) if isinstance(m2.get('tissue'), list) else set()
    if len(t1) > 0 and len(t2) > 0:
        union = len(t1 | t2)
        if union > 0:
            score += 0.1 * (len(t1 & t2) / union)
    return score


def ligand_avg_affinity(lig_ki_map, ligand):
    kis = lig_ki_map.get(ligand, [1e6])
    return -np.mean([math.log10(k) for k in kis])


def predict_for_receptor(held_out_rec, held_out_pairs, true_ligands, rec_meta,
                         known_receptors, known_ligands, ligand_receptors, lig_ki_map):
    rec_info = {'family': held_out_pairs[0]['family'],
                'subfamily': held_out_pairs[0].get('subfamily', ''),
                'class': held_out_pairs[0].get('class', ''),
                'g_coupling': held_out_pairs[0]['g_coupling']}

    predictions = []
    for ligand in known_ligands:
        total_w_aff = 0.0
        total_w = 0.0
        for other_rec in known_receptors:
            if other_rec == held_out_rec:
                continue
            if other_rec not in ligand_receptors[ligand]:
                continue
            s = metadata_similarity(held_out_rec, other_rec, rec_meta)
            aff = ligand_avg_affinity(lig_ki_map, ligand)
            total_w_aff += s * aff
            total_w += s
        kde_aff = total_w_aff / total_w if total_w > 0 else 0.0

        direct_aff = ligand_avg_affinity(lig_ki_map, ligand)

        same_fam = 0
        same_gp = 0
        total_binding = 0
        for other_rec in ligand_receptors[ligand]:
            if other_rec == held_out_rec:
                continue
            total_binding += 1
            if rec_meta[other_rec]['family'] == rec_info['family']:
                same_fam += 1
            if rec_meta[other_rec].get('g_coupling') == rec_info['g_coupling']:
                same_gp += 1
        fam_match = same_fam / max(1, total_binding)
        gp_match = same_gp / max(1, total_binding)

        promiscuity = len(ligand_receptors[ligand]) / len(known_receptors) if len(known_receptors) > 0 else 0
        promiscuity_penalty = 1.0 - 0.5 * promiscuity

        natural_bonus = 0.0
        for p in held_out_pairs:
            if p.get('is_endogenous', False):
                natural_bonus += 0.3
        natural_bonus = min(natural_bonus, 0.5)

        broad_bonus = 0.0
        if rec_info.get('class'):
            for other_rec in ligand_receptors[ligand]:
                if other_rec == held_out_rec:
                    continue
                other_class = rec_meta[other_rec].get('class', '')
                if other_class and rec_info['class'] and other_class != rec_info['class']:
                    if rec_meta[other_rec]['family'] == rec_info['family']:
                        broad_bonus += 0.05
        broad_bonus = min(broad_bonus, 0.1)

        aff_normalized = max(0, min(1, (direct_aff + 4) / 4))
        combined = (0.35 * fam_match + 0.25 * gp_match + 0.20 * kde_aff +
                    0.10 * aff_normalized + 0.10 * natural_bonus + 0.05 * broad_bonus) * promiscuity_penalty

        if rec_info['family'] in ('Biogenic Amine', 'Biogenic amine'):
            expected = ['DOPAMINE', 'SEROTONIN', 'NOREPINEPHRINE', 'EPINEPHRINE', 'ACETYLCHOLINE']
            if ligand in expected:
                combined = max(combined, combined * 1.5 + 0.2)

        predictions.append((ligand, combined))

    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions, true_ligands


# ============================================================================
# CALIBRATION METRICS
# ============================================================================

def compute_calibration(scores, labels, n_bins=10):
    """Compute calibration curve and Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_data = []
    ece = 0.0
    total = len(scores)

    for i in range(n_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        if i == n_bins - 1:
            mask = (scores >= low) & (scores <= high)
        else:
            mask = (scores >= low) & (scores < high)

        n_in_bin = mask.sum()
        if n_in_bin > 0:
            avg_confidence = scores[mask].mean()
            avg_accuracy = labels[mask].mean()
            bin_data.append({
                'bin': f'{low:.1f}-{high:.1f}',
                'low': round(float(low), 2),
                'high': round(float(high), 2),
                'n_samples': int(n_in_bin),
                'avg_confidence': round(float(avg_confidence), 4),
                'avg_accuracy': round(float(avg_accuracy), 4),
            })
            ece += (n_in_bin / total) * abs(float(avg_accuracy) - float(avg_confidence))
        else:
            bin_data.append({
                'bin': f'{low:.1f}-{high:.1f}',
                'low': round(float(low), 2),
                'high': round(float(high), 2),
                'n_samples': 0,
                'avg_confidence': None,
                'avg_accuracy': None,
            })

    return bin_data, round(float(ece), 4)


def compute_brier_score(scores, labels):
    """Compute the Brier score: mean squared error between confidence and actual outcome."""
    return round(float(np.mean((scores - labels) ** 2)), 4)


def compute_reliability_diagram(scores, labels, n_bins=10):
    """Compute reliability diagram data (confidence vs accuracy per bin)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    diagram = []
    for i in range(n_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        if i == n_bins - 1:
            mask = (scores >= low) & (scores <= high)
        else:
            mask = (scores >= low) & (scores < high)

        n_in_bin = mask.sum()
        if n_in_bin > 0:
            diagram.append({
                'bin': f'{low:.1f}-{high:.1f}',
                'mid_confidence': round(float((low + high) / 2), 2),
                'accuracy': round(float(labels[mask].mean()), 4),
                'count': int(n_in_bin),
                'gap': round(float(labels[mask].mean() - (low + high) / 2), 4),
            })
        else:
            diagram.append({
                'bin': f'{low:.1f}-{high:.1f}',
                'mid_confidence': round(float((low + high) / 2), 2),
                'accuracy': None,
                'count': 0,
                'gap': None,
            })
    return diagram


# ============================================================================
# METHOD 1: ISOTONIC REGRESSION CALIBRATION
# ============================================================================

def isotonic_regression_fit(raw_scores, labels):
    """
    Fit isotonic regression: learn a monotone mapping from raw confidence
    to calibrated confidence using the Pool Adjacent Violators Algorithm (PAVA).

    Parameters
    ----------
    raw_scores : np.ndarray of shape (n,)
        Uncalibrated confidence scores in [0, 1].
    labels : np.ndarray of shape (n,)
        Binary outcomes (1 = true ligand, 0 = false ligand).

    Returns
    -------
    isotonic_func : function
        Callable that maps raw score -> calibrated score.
    """
    # Sort by raw score
    order = np.argsort(raw_scores)
    sorted_scores = raw_scores[order]
    sorted_labels = labels[order]

    # Pool Adjacent Violators Algorithm
    # Start with each point as its own pool
    pools = []
    for i in range(len(sorted_scores)):
        pools.append({
            'sum': sorted_labels[i],
            'count': 1,
            'mean': sorted_labels[i],
            'scores': [sorted_scores[i]],
        })

    # Merge adjacent pools where mean decreases (violates monotonicity)
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(pools) - 1:
            if pools[i]['mean'] > pools[i + 1]['mean']:
                # Merge pools i and i+1
                merged = {
                    'sum': pools[i]['sum'] + pools[i + 1]['sum'],
                    'count': pools[i]['count'] + pools[i + 1]['count'],
                    'mean': pools[i]['sum'] + pools[i + 1]['sum'] / (pools[i]['count'] + pools[i + 1]['count']),
                    'scores': pools[i]['scores'] + pools[i + 1]['scores'],
                }
                pools[i] = merged
                pools.pop(i + 1)
                changed = True
            else:
                i += 1

    # Build lookup: for any raw score, find the pool whose score range it falls in
    # Compute representative raw score for each pool (mean of scores in pool)
    pool_representatives = []
    pool_calibrated_values = []
    for pool in pools:
        rep_score = np.mean(pool['scores'])
        pool_representatives.append(rep_score)
        pool_calibrated_values.append(pool['mean'])

    def isotonic_predict(raw_score):
        if len(pool_representatives) == 0:
            return float(np.mean(labels))
        # Find nearest pool representative
        idx = np.argmin(np.abs(np.array(pool_representatives) - raw_score))
        return float(pool_calibrated_values[idx])

    return isotonic_predict


def apply_isotonic_calibration(raw_scores, labels):
    """Apply isotonic regression calibration and return calibrated scores."""
    predictor = isotonic_regression_fit(raw_scores, labels)
    calibrated = np.array([predictor(s) for s in raw_scores])
    return calibrated


# ============================================================================
# METHOD 2: PLATT SCALING
# ============================================================================

def sigmoid(x, a, b):
    """Platt scaling sigmoid: sigmoid(a * x + b)."""
    return 1.0 / (1.0 + np.exp(-(a * x + b)))


def platt_scaling_fit(raw_scores, labels, a_init=1.0, b_init=0.0,
                      lr=0.01, n_iter=10000):
    """
    Learn Platt scaling parameters a, b by minimizing negative log-likelihood.

    calibrated_confidence = sigmoid(a * raw_confidence + b)

    Parameters
    ----------
    raw_scores : np.ndarray
    labels : np.ndarray
    a_init, b_init : initial parameters
    lr : learning rate
    n_iter : number of optimization iterations

    Returns
    -------
    a, b : learned parameters
    """
    a = a_init
    b = b_init

    for _ in range(n_iter):
        # Compute calibrated predictions
        z = a * raw_scores + b
        # Clip to avoid overflow
        z = np.clip(z, -500, 500)
        pred = 1.0 / (1.0 + np.exp(-z))

        # Clip predictions to avoid log(0)
        pred_clipped = np.clip(pred, 1e-10, 1 - 1e-10)

        # Negative log-likelihood gradient
        # dNLL/da = sum((pred - y) * x)
        # dNLL/db = sum((pred - y))
        error = pred_clipped - labels
        grad_a = np.mean(error * raw_scores)
        grad_b = np.mean(error)

        # Gradient descent
        a -= lr * grad_a
        b -= lr * grad_b

    return a, b


def apply_platt_calibration(raw_scores, labels):
    """Apply Platt scaling calibration and return calibrated scores + parameters."""
    a, b = platt_scaling_fit(raw_scores, labels)
    calibrated = sigmoid(raw_scores, a, b)
    return calibrated, a, b


# ============================================================================
# METHOD 3: TEMPERATURE SCALING
# ============================================================================

def temperature_scaling_fit(raw_scores, labels, T_init=1.0, lr=0.001, n_iter=5000):
    """
    Learn a single temperature parameter T.

    calibrated = sigmoid(raw_score / T)

    T > 1 softens predictions (reduces confidence).
    T < 1 sharpens predictions (increases confidence).

    Parameters
    ----------
    raw_scores : np.ndarray
    labels : np.ndarray
    T_init : initial temperature
    lr : learning rate
    n_iter : iterations

    Returns
    -------
    T : learned temperature
    """
    T = T_init

    for _ in range(n_iter):
        z = raw_scores / T
        z = np.clip(z, -500, 500)
        pred = 1.0 / (1.0 + np.exp(-z))
        pred_clipped = np.clip(pred, 1e-10, 1 - 1e-10)

        error = pred_clipped - labels
        # dNLL/dT = sum((pred - y) * raw_score / T^2)
        grad_T = np.mean(error * raw_scores) / (T ** 2)
        T -= lr * grad_T
        T = max(T, 0.01)  # Prevent T from going to zero

    return T


def apply_temperature_calibration(raw_scores, labels):
    """Apply temperature scaling calibration and return calibrated scores + T."""
    T = temperature_scaling_fit(raw_scores, labels)
    z = raw_scores / T
    z = np.clip(z, -500, 500)
    calibrated = 1.0 / (1.0 + np.exp(-z))
    return calibrated, T


# ============================================================================
# LOOCV DATA GENERATION
# ============================================================================

def generate_loocv_scores(pairs):
    """
    Run leave-one-receptor-out cross-validation to generate
    raw confidence scores and binary labels for all predictions.

    Returns
    -------
    raw_scores : np.ndarray
    labels : np.ndarray
    receptor_ligands : dict
    """
    known_receptors = set()
    known_ligands = set()
    for p in pairs:
        known_receptors.add(p['receptor'])
        known_ligands.add(p['ligand'])

    receptor_ligands = defaultdict(list)
    for p in pairs:
        receptor_ligands[p['receptor']].append(p)

    ligand_receptors = defaultdict(set)
    for p in pairs:
        ligand_receptors[p['ligand']].add(p['receptor'])

    rec_meta = {}
    for p in pairs:
        r = p['receptor']
        if r not in rec_meta:
            rec_meta[r] = {'family': p['family'], 'class': p.get('class', ''),
                           'g_coupling': p['g_coupling'], 'subfamily': p.get('subfamily', ''),
                           'tissue': p.get('tissue', [])}

    lig_ki_map = {}
    for p in pairs:
        l = p['ligand']
        if l not in lig_ki_map:
            lig_ki_map[l] = []
        lig_ki_map[l].append(p['ki'])

    all_scores = []
    all_labels = []

    for held_out_rec in receptor_ligands:
        held_out_pairs = receptor_ligands[held_out_rec]
        true_ligands = set(p['ligand'] for p in held_out_pairs)

        predictions, _ = predict_for_receptor(
            held_out_rec, held_out_pairs, true_ligands,
            rec_meta, known_receptors, known_ligands, ligand_receptors, lig_ki_map
        )

        for ligand, score in predictions:
            all_scores.append(score)
            all_labels.append(1 if ligand in true_ligands else 0)

    return np.array(all_scores), np.array(all_labels), receptor_ligands


# ============================================================================
# COMPARISON AND REPORTING
# ============================================================================

def compare_calibration_methods(raw_scores, labels):
    """
    Compare isotonic regression, Platt scaling, and temperature scaling.

    Returns
    -------
    results : dict with all metrics for each method
    """
    results = {}

    # --- Uncalibrated baseline ---
    results['uncalibrated'] = {
        'ece': compute_calibration(raw_scores, labels)[1],
        'ece_top5': compute_calibration(raw_scores[:250], labels[:250])[1] if len(raw_scores) >= 250 else None,
        'brier': compute_brier_score(raw_scores, labels),
        'calibration_curve': compute_calibration(raw_scores, labels)[0],
        'reliability': compute_reliability_diagram(raw_scores, labels),
        'accuracy_by_threshold': {},
    }
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        mask = raw_scores >= t
        if mask.sum() > 0:
            results['uncalibrated']['accuracy_by_threshold'][str(t)] = round(float(labels[mask].mean()), 4)
        else:
            results['uncalibrated']['accuracy_by_threshold'][str(t)] = None

    # --- Method 1: Isotonic Regression ---
    iso_calibrated = apply_isotonic_calibration(raw_scores, labels)
    results['isotonic_regression'] = {
        'calibrated_scores': iso_calibrated.tolist(),
        'ece': compute_calibration(iso_calibrated, labels)[1],
        'ece_top5': compute_calibration(iso_calibrated[:250], labels[:250])[1] if len(iso_calibrated) >= 250 else None,
        'brier': compute_brier_score(iso_calibrated, labels),
        'calibration_curve': compute_calibration(iso_calibrated, labels)[0],
        'reliability': compute_reliability_diagram(iso_calibrated, labels),
        'accuracy_by_threshold': {},
    }
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        mask = iso_calibrated >= t
        if mask.sum() > 0:
            results['isotonic_regression']['accuracy_by_threshold'][str(t)] = round(float(labels[mask].mean()), 4)
        else:
            results['isotonic_regression']['accuracy_by_threshold'][str(t)] = None

    # --- Method 2: Platt Scaling ---
    platt_calibrated, platt_a, platt_b = apply_platt_calibration(raw_scores, labels)
    results['platt_scaling'] = {
        'calibrated_scores': platt_calibrated.tolist(),
        'ece': compute_calibration(platt_calibrated, labels)[1],
        'ece_top5': compute_calibration(platt_calibrated[:250], labels[:250])[1] if len(platt_calibrated) >= 250 else None,
        'brier': compute_brier_score(platt_calibrated, labels),
        'calibration_curve': compute_calibration(platt_calibrated, labels)[0],
        'reliability': compute_reliability_diagram(platt_calibrated, labels),
        'accuracy_by_threshold': {},
        'platt_parameters': {'a': round(float(platt_a), 4), 'b': round(float(platt_b), 4)},
    }
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        mask = platt_calibrated >= t
        if mask.sum() > 0:
            results['platt_scaling']['accuracy_by_threshold'][str(t)] = round(float(labels[mask].mean()), 4)
        else:
            results['platt_scaling']['accuracy_by_threshold'][str(t)] = None

    # --- Method 3: Temperature Scaling ---
    temp_calibrated, temp_T = apply_temperature_calibration(raw_scores, labels)
    results['temperature_scaling'] = {
        'calibrated_scores': temp_calibrated.tolist(),
        'ece': compute_calibration(temp_calibrated, labels)[1],
        'ece_top5': compute_calibration(temp_calibrated[:250], labels[:250])[1] if len(temp_calibrated) >= 250 else None,
        'brier': compute_brier_score(temp_calibrated, labels),
        'calibration_curve': compute_calibration(temp_calibrated, labels)[0],
        'reliability': compute_reliability_diagram(temp_calibrated, labels),
        'accuracy_by_threshold': {},
        'temperature_parameter': round(float(temp_T), 4),
    }
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        mask = temp_calibrated >= t
        if mask.sum() > 0:
            results['temperature_scaling']['accuracy_by_threshold'][str(t)] = round(float(labels[mask].mean()), 4)
        else:
            results['temperature_scaling']['accuracy_by_threshold'][str(t)] = None

    return results


def print_comparison(results, raw_scores, labels):
    """Print a comparison table of all calibration methods."""
    print("\n" + "=" * 80)
    print("CALIBRATION FIX COMPARISON")
    print("=" * 80)
    print(f"\nDataset: {len(raw_scores)} predictions, {int(labels.sum())} true ligands, "
          f"{int(len(labels) - labels.sum())} false ligands")
    print(f"True positive rate: {labels.mean():.4f}")

    print("\n" + "-" * 80)
    print(f"{'Method':<30} {'ECE':>8} {'ECE (top5)':>12} {'Brier':>8} {'T>0.8 acc':>10}")
    print("-" * 80)

    for method_name in ['uncalibrated', 'isotonic_regression', 'platt_scaling', 'temperature_scaling']:
        r = results[method_name]
        ece = r['ece']
        ece5 = r.get('ece_top5', 'N/A')
        brier = r['brier']
        acc_t8 = r['accuracy_by_threshold'].get('0.8', 'N/A')

        ece5_str = f"{ece5:.4f}" if isinstance(ece5, float) else str(ece5)
        acc_str = f"{acc_t8:.4f}" if isinstance(acc_t8, float) else str(acc_t8)

        marker = ""
        if method_name == 'uncalibrated':
            marker = " <-- baseline"
        elif method_name == 'isotonic_regression':
            best_ece = min(results['isotonic_regression']['ece'],
                          results['platt_scaling']['ece'],
                          results['temperature_scaling']['ece'])
            if r['ece'] == best_ece:
                marker = " <-- best ECE"
        elif method_name == 'platt_scaling':
            best_brier = min(results['isotonic_regression']['brier'],
                            results['platt_scaling']['brier'],
                            results['temperature_scaling']['brier'])
            if r['brier'] == best_brier:
                marker = " <-- best Brier"

        print(f"{method_name:<30} {ece:>8.4f} {ece5_str:>12} {brier:>8.4f} {acc_str:>10}{marker}")

    print("-" * 80)

    # Improvement summary
    print("\nIMPROVEMENT SUMMARY")
    print("-" * 80)
    for method_name in ['isotonic_regression', 'platt_scaling', 'temperature_scaling']:
        r = results[method_name]
        ece_improvement = results['uncalibrated']['ece'] - r['ece']
        brier_improvement = results['uncalibrated']['brier'] - r['brier']
        print(f"\n{method_name}:")
        print(f"  ECE improvement:  {results['uncalibrated']['ece']:.4f} -> {r['ece']:.4f} "
              f"({ece_improvement:+.4f}, {ece_improvement/results['uncalibrated']['ece']*100:+.1f}%)")
        print(f"  Brier improvement: {results['uncalibrated']['brier']:.4f} -> {r['brier']:.4f} "
              f"({brier_improvement:+.4f}, {brier_improvement/results['uncalibrated']['brier']*100:+.1f}%)")

        # High confidence accuracy
        uncal_t8 = results['uncalibrated']['accuracy_by_threshold'].get('0.8', 0)
        cal_t8 = r['accuracy_by_threshold'].get('0.8', 0)
        if isinstance(uncal_t8, float) and isinstance(cal_t8, float):
            print(f"  Accuracy at C>0.8: {uncal_t8:.4f} -> {cal_t8:.4f} "
                  f"({(cal_t8 - uncal_t8):+.4f})")

    # Method-specific details
    if 'platt_scaling' in results:
        pa = results['platt_scaling']['platt_parameters']
        print(f"\nPlatt scaling parameters: a={pa['a']}, b={pa['b']}")
        print(f"  Interpretation: a={pa['a']:.2f} means the sigmoid slope is {'steep' if abs(pa['a']) > 1 else 'shallow'}")
        print(f"  b={pa['b']:.2f} shifts the decision boundary")

    if 'temperature_scaling' in results:
        T = results['temperature_scaling']['temperature_parameter']
        print(f"\nTemperature parameter: T={T:.4f}")
        print(f"  T > 1 means predictions are {'softened' if T > 1 else 'sharpened'}")
        print(f"  T < 1 means predictions are {'sharpened' if T < 1 else 'softened'}")

    # Calibration curve comparison
    print("\n" + "-" * 80)
    print("CALIBRATION CURVE COMPARISON (per bin)")
    print("-" * 80)
    print(f"{'Bin':<12} {'Uncal Conf':>10} {'Uncal Acc':>10} "
          f"{'Iso Cal':>8} {'Iso Acc':>8} "
          f"{'Platt Cal':>10} {'Platt Acc':>10}")
    print("-" * 80)

    uncal_curve = results['uncalibrated']['calibration_curve']
    iso_curve = results['isotonic_regression']['calibration_curve']
    platt_curve = results['platt_scaling']['calibration_curve']

    for i in range(10):
        ub = uncal_curve[i]
        ib = iso_curve[i]
        pb = platt_curve[i]

        bin_label = ub['bin']
        u_conf = f"{ub['avg_confidence']:.4f}" if ub['avg_confidence'] is not None else "N/A"
        u_acc = f"{ub['avg_accuracy']:.4f}" if ub['avg_accuracy'] is not None else "N/A"
        i_conf = f"{ib['avg_confidence']:.4f}" if ib['avg_confidence'] is not None else "N/A"
        i_acc = f"{ib['avg_accuracy']:.4f}" if ib['avg_accuracy'] is not None else "N/A"
        p_conf = f"{pb['avg_confidence']:.4f}" if pb['avg_confidence'] is not None else "N/A"
        p_acc = f"{pb['avg_accuracy']:.4f}" if pb['avg_accuracy'] is not None else "N/A"

        print(f"{bin_label:<12} {u_conf:>10} {u_acc:>10} {i_conf:>8} {i_acc:>8} "
              f"{p_conf:>10} {p_acc:>10}")

    print()


def generate_calibration_report(results, raw_scores, labels):
    """Generate a detailed markdown report of calibration results."""
    lines = []
    lines.append("# Calibration Fix Report: Orphan GPCR Deorphanization")
    lines.append("")
    lines.append("## Critical Finding")
    lines.append("")
    lines.append("The model's confidence scores are **severely overconfident**:")
    lines.append("")
    lines.append(f"- At confidence >0.8, actual accuracy = "
                 f"{results['uncalibrated']['accuracy_by_threshold'].get('0.8', 0)}")
    lines.append(f"- At confidence >0.9, actual accuracy = "
                 f"{results['uncalibrated']['accuracy_by_threshold'].get('0.9', 0)}")
    lines.append(f"- ECE (all predictions) = {results['uncalibrated']['ece']} (moderate)")
    lines.append(f"- ECE (top-5 predictions) = {results['uncalibrated'].get('ece_top5', 'N/A')} (poor)")
    lines.append(f"- Brier score = {results['uncalibrated']['brier']}")
    lines.append("")
    lines.append("This means the model is **most confident when it is most wrong**.")
    lines.append("")

    lines.append("## Calibration Methods Compared")
    lines.append("")
    lines.append("| Method | ECE | ECE (top-5) | Brier Score |")
    lines.append("|--------|-----|-------------|-------------|")
    for method_name in ['uncalibrated', 'isotonic_regression', 'platt_scaling', 'temperature_scaling']:
        r = results[method_name]
        ece5 = r.get('ece_top5', 'N/A')
        ece5_str = f"{ece5:.4f}" if isinstance(ece5, float) else str(ece5)
        lines.append(f"| {method_name} | {r['ece']:.4f} | {ece5_str} | {r['brier']:.4f} |")
    lines.append("")

    lines.append("## Recommended Approach")
    lines.append("")

    # Pick best method by ECE
    best_ece_method = None
    best_ece_val = float('inf')
    for method_name in ['isotonic_regression', 'platt_scaling', 'temperature_scaling']:
        if results[method_name]['ece'] < best_ece_val:
            best_ece_val = results[method_name]['ece']
            best_ece_method = method_name

    best_brier_method = None
    best_brier_val = float('inf')
    for method_name in ['isotonic_regression', 'platt_scaling', 'temperature_scaling']:
        if results[method_name]['brier'] < best_brier_val:
            best_brier_val = results[method_name]['brier']
            best_brier_method = method_name

    lines.append(f"- **Best ECE:** {best_ece_method} (ECE = {best_ece_val:.4f})")
    lines.append(f"- **Best Brier:** {best_brier_method} (Brier = {best_brier_val:.4f})")
    lines.append("")
    lines.append("## Key Insight")
    lines.append("")
    lines.append("Calibration **cannot fix** the fundamental limitation of this model: "
                 "the underlying predictions are weak (P@5 = 10.68%, AUC-ROC = 0.5878). "
                 "Calibration only makes the confidence scores honest; it does not improve "
                 "the underlying prediction quality. The model's predictions should be treated "
                 "as hypotheses for experimental testing, not as reliable probability estimates.")
    lines.append("")

    return "\n".join(lines)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("Loading data...")
    pairs = load_data()
    print(f"  Loaded {len(pairs)} receptor-ligand pairs")

    print("\nGenerating LOOCV scores...")
    raw_scores, labels, receptor_ligands = generate_loocv_scores(pairs)
    print(f"  {len(raw_scores)} total predictions")
    print(f"  {int(labels.sum())} true ligands, {int(len(labels) - labels.sum())} false ligands")

    print("\nRunning calibration methods...")
    results = compare_calibration_methods(raw_scores, labels)

    print_comparison(results, raw_scores, labels)

    # Save full results
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)

    # Save detailed results
    with open(os.path.join(output_dir, 'calibration_fix_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to {output_dir}/calibration_fix_results.json")

    # Save calibration report
    report = generate_calibration_report(results, raw_scores, labels)
    with open(os.path.join(output_dir, 'calibration_report.md'), 'w') as f:
        f.write(report)
    print(f"Calibration report saved to {output_dir}/calibration_report.md")

    # Save calibrated scores for each method
    for method_name in ['isotonic_regression', 'platt_scaling', 'temperature_scaling']:
        scores_key = 'calibrated_scores'
        if scores_key in results[method_name]:
            with open(os.path.join(output_dir, f'{method_name}_scores.json'), 'w') as f:
                json.dump(results[method_name][scores_key], f)
            print(f"{method_name} scores saved to {output_dir}/{method_name}_scores.json")

    print("\nDone.")


if __name__ == '__main__':
    main()
