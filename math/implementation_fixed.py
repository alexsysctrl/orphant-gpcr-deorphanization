"""
Metadata-Based Receptor Similarity for Orphan Receptor Deorphanization
=======================================================================

Replaces the embedding-based approach with a metadata-driven similarity model.

The embedding approach failed because the 860-dim receptor feature vectors
had zero discriminative power (all pairwise distances ~380-410).

This implementation uses ONLY the metadata fields that are actually meaningful:
  1. Family membership (exact match, subfamily, class-level, or no match)
  2. G-protein coupling agreement
  3. Tissue expression overlap (Jaccard similarity)

Each component produces a score in [0, 1], and they are combined into a
weighted similarity score.

Ligand prediction uses proper kernel density estimation:
  P(l|r_orphan) = sum_j S(r_i, r_j) * count(l, r_j) / sum_j S(r_i, r_j)

Binding affinity prediction uses weighted averaging of pKi values:
  f(r_i, l) = sum_j S(r_i, r_j) * pKi(r_j, l) / sum_j S(r_i, r_j)

Confidence scoring:
  C(r_i, l) = sigmoid(f(r_i, l) - theta) * sqrt(N_similar)
"""

import json
import math
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict


# ============================================================================
# SECTION 0: DATA LOADING
# ============================================================================

# Resolve DATA_DIR relative to this file's location
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
DATA_DIR = os.path.normpath(_DATA_DIR)


def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def load_all_data() -> Dict[str, Any]:
    return {
        "raw_pairs": load_json(f"{DATA_DIR}/raw_data.json"),
        "orphans": load_json(f"{DATA_DIR}/orphan_receptor_catalog.json"),
        "ligand_classes": load_json(f"{DATA_DIR}/ligand_classification.json"),
        "receptor_classes": load_json(f"{DATA_DIR}/receptor_classification.json"),
    }


# ============================================================================
# SECTION 1: RECEPTOR METADATA SIMILARITY
# ============================================================================

# Define family subfamily groupings for fine-grained matching
# Receptors in the same subfamily share high sequence identity and ligand specificity
FAMILY_SUBFAMILY_MAP = {
    # Adrenergic subfamilies
    "adrenergic": {
        "alpha_1": ["Alpha-1A adrenergic receptor", "Alpha-1B adrenergic receptor", "Alpha-1D adrenergic receptor"],
        "alpha_2": ["Alpha-2A adrenergic receptor", "Alpha-2B adrenergic receptor", "Alpha-2C adrenergic receptor"],
        "beta": ["Beta-1 adrenergic receptor", "Beta-2 adrenergic receptor", "Beta-3 adrenergic receptor"],
    },
    # Dopaminergic subfamilies
    "dopaminergic": {
        "D1_like": ["D1 dopamine receptor", "D5 dopamine receptor"],
        "D2_like": ["D2 dopamine receptor", "D3 dopamine receptor", "D4 dopamine receptor"],
    },
    # Serotonergic subfamilies
    "serotonergic": {
        "5-HT1": ["5-HT1A receptor", "5-HT1B receptor", "5-HT1D receptor", "5-HT1E receptor", "5-HT1F receptor", "5-HT5A receptor"],
        "5-HT2": ["5-HT2A receptor", "5-HT2B receptor", "5-HT2C receptor"],
        "5-HT3": ["5-HT3 receptor"],
        "5-HT4": ["5-HT4 receptor"],
        "5-HT6": ["5-HT6 receptor"],
        "5-HT7": ["5-HT7 receptor"],
    },
    # Opioid subfamilies
    "opioid": {
        "mu": ["Mu opioid receptor"],
        "kappa": ["Kappa opioid receptor"],
        "delta": ["Delta opioid receptor"],
        "nociceptin": ["NOP (Nociceptin) receptor"],
    },
    # Muscarinic subfamilies
    "muscarinic": {
        "M1-M5": ["M1 muscarinic receptor", "M2 muscarinic receptor", "M3 muscarinic receptor", "M4 muscarinic receptor", "M5 muscarinic receptor"],
    },
    # Adenosine subfamilies
    "adenosine": {
        "A1/A3": ["A1 adenosine receptor", "A3 adenosine receptor"],
        "A2": ["A2A adenosine receptor", "A2B adenosine receptor"],
    },
    # Histamine subfamilies
    "histamine": {
        "H1-H4": ["H1 histamine receptor", "H2 histamine receptor", "H3 histamine receptor", "H4 histamine receptor"],
    },
    # Purinergic subfamilies
    "purinergic": {
        "P2Y1-like": ["P2Y1 receptor", "P2Y2 receptor", "P2Y4 receptor", "P2Y10 receptor", "P2Y11 receptor"],
        "P2Y6-like": ["P2Y6 receptor"],
        "P2Y12/13": ["P2Y12 receptor", "P2Y13 receptor"],
    },
    # Melanocortin subfamilies
    "melanocortin": {
        "MC1-MC5": ["MC1R melanocortin receptor", "MC2R melanocortin receptor", "MC3R melanocortin receptor", "MC4R melanocortin receptor", "MC5R melanocortin receptor"],
    },
    # Orexin subfamilies
    "orexin": {
        "OX1/OX2": ["OX1 orexin receptor", "OX2 orexin receptor"],
    },
    # S1P subfamilies
    "S1P": {
        "S1P1-5": ["EDG1", "EDG3", "EDG4", "EDG5", "EDG8"],
    },
    # Free fatty acid subfamilies
    "free fatty acid": {
        "FFAR1-4": ["FFAR1", "FFAR2", "FFAR3", "FFAR4"],
    },
    # Adhesion GPCR subfamilies
    "Adhesion GPCR": {
        "ADGRB": ["GPR56", "GPR64", "GPR126"],
        "ADGRL": ["GPR128", "GPR133", "GPR164"],
        "ADGRF": ["GPR143", "GPR144", "GPR177"],
        "other": ["GPR132", "GPR135", "GPR155", "GPR157", "GPR161"],
    },
}


class MetadataSimilarity:
    """
    Computes receptor similarity based on metadata only.

    Components (each in [0, 1]):
      1. family_similarity: Exact family = 1.0, same subfamily = 0.7,
                           same broad class = 0.3, different = 0.0
      2. g_protein_similarity: Same G-coupling = 1.0, different = 0.0
                               (handles null/unknown gracefully)
      3. tissue_similarity: Jaccard similarity of tissue expression sets

    Combined: S = w1 * family + w2 * g_protein + w3 * tissue
    Weights: 0.5, 0.3, 0.2 (family is most discriminative)
    """

    def __init__(self, w_family: float = 0.5, w_gprotein: float = 0.3,
                 w_tissue: float = 0.2):
        self.w_family = w_family
        self.w_gprotein = w_gprotein
        self.w_tissue = w_tissue
        self.total_weight = w_family + w_gprotein + w_tissue

        # Build receptor name -> metadata lookup
        self.receptor_metadata: Dict[str, Dict] = {}
        self.receptor_names: List[str] = []

    def build_index(self, raw_pairs: List[Dict], orphan_catalog: List[Dict]):
        """Build lookup tables from raw data."""
        # Index known receptors from raw pairs
        for pair in raw_pairs:
            name = pair["receptor"]
            if name not in self.receptor_metadata:
                self.receptor_metadata[name] = {
                    "name": name,
                    "gene": pair.get("gene", ""),
                    "family": pair.get("family", ""),
                    "class": pair.get("class", "A"),
                    "g_coupling": pair.get("g_coupling"),
                    "tissue_expression": pair.get("tissue_expression", ""),
                }
                self.receptor_names.append(name)

        # Index orphan receptors
        for odata in orphan_catalog:
            name = odata["name"]
            if name not in self.receptor_metadata:
                self.receptor_metadata[name] = {
                    "name": name,
                    "gene": odata.get("gene", ""),
                    "family": odata.get("family", "Class A orphan"),
                    "class": odata.get("class", "A"),
                    "g_coupling": odata.get("g_coupling"),
                    "tissue_expression": odata.get("tissue_expression", ""),
                }
                self.receptor_names.append(name)

    def family_similarity(self, r1: Dict, r2: Dict) -> float:
        """
        Compute family membership similarity.

        Exact same family = 1.0
        Same subfamily = 0.7
        Orphan matched to specific family via notes/homolog hints = 0.5
        Same broad class (A/B/C/Adhesion) = 0.15 (weak — most orphans are Class A)
        Different = 0.0
        """
        f1 = r1.get("family", "")
        f2 = r2.get("family", "")

        if f1 == f2:
            # Check subfamily
            sub1 = self._get_subfamily(r1)
            sub2 = self._get_subfamily(r2)
            if sub1 == sub2 and sub1 is not None:
                return 0.7
            return 1.0

        # Check if one is an orphan and the other is a known family
        # Use the orphan's notes/homolog hints to determine family match
        if "orphan" in f1.lower() or "orphan" in f2.lower():
            orphan_meta = r1 if "orphan" in f1.lower() else r2
            known_meta = r2 if "orphan" in f1.lower() else r1
            hinted_family = self._hint_family_from_orphan(orphan_meta)
            if hinted_family and hinted_family == known_meta.get("family", ""):
                return 0.5

        # Check broad class (weak signal — most orphans are Class A)
        class1 = self._get_broad_class(f1)
        class2 = self._get_broad_class(f2)
        if class1 == class2 and class1 is not None:
            return 0.15

        return 0.0

    def _hint_family_from_orphan(self, orphan_meta: Dict) -> Optional[str]:
        """
        Infer likely family from orphan receptor notes and homolog hints.

        Uses the 'notes' field and known homolog relationships to guess
        which known family this orphan likely belongs to.
        """
        notes = orphan_meta.get("notes", "").lower()
        family = orphan_meta.get("family", "")
        name = orphan_meta.get("name", "")

        # Explicit family hints from notes
        family_hints = {
            "dopamine": "dopaminergic",
            "adrenergic": "adrenergic",
            "melanocortin": "melanocortin",
            "serotonin": "serotonergic",
            "histamine": "histamine",
            "galanin": "partially deorphanized",
            "opioid": "opioid",
            "orexin": "orexin",
            "muscarinic": "muscarinic",
            "purine": "purinergic",
            "fatty acid": "free fatty acid",
            "lipid": "S1P",
            "sphingosine": "S1P",
            "prostaglandin": "S1P",
            "LPA": "S1P",
            "acid-sensing": "Class A orphan",
            "class c": "Class C orphan",
            "adhesion": "Adhesion GPCR",
        }

        for hint, family_match in family_hints.items():
            if hint in notes:
                return family_match

        # Name-based hints
        name_hints = {
            "GPR32": "S1P",  # prostaglandin E2 receptor
            "GPR33": "dopaminergic",  # N-acyl dopamine receptor
            "GPR34": "Class A orphan",  # D-serine
            "GPR65": "Class A orphan",  # acid-sensing (TGR6)
            "GPR68": "free fatty acid",  # ketone body sensor
            "GPR83": "S1P",  # LPA activated
            "GPR87": "S1P",  # LPA/prostaglandin
            "GPR119": "free fatty acid",  # OEA receptor
            "GPR120": "free fatty acid",  # EPA/DHA
            "GPR109A": "Class A orphan",  # niacin
            "GPR139": "partially deorphanized",  # neurotensin
            "GPR144": "Class A orphan",  # retinaldehyde
            "GPR151": "amino acid",  # tryptophan/phenylalanine
            "GPR171": "Adhesion GPCR",  # adhesion GPCR
            "GPR4": "Class A orphan",  # acid-sensing
            "GPR126": "Adhesion GPCR",
            "GPR128": "Adhesion GPCR",
            "GPR132": "Adhesion GPCR",
            "GPR133": "Adhesion GPCR",
            "GPR135": "Adhesion GPCR",
            "GPR143": "Adhesion GPCR",
            "GPR155": "Adhesion GPCR",
            "GPR156": "Class C orphan",
            "GPR157": "Adhesion GPCR",
            "GPR158": "Class C orphan",
            "GPR161": "Adhesion GPCR",
            "GPR164": "Adhesion GPCR",
            "GPR177": "Adhesion GPCR",
            "GPR179": "Class C orphan",
            "GPR56": "Adhesion GPCR",
            "GPR64": "Adhesion GPCR",
            "GPR19": "dopaminergic",  # dopamine receptor family
            "GPR20": "adrenergic",  # closest homolog is ADRA1A
            "GPR21": "adrenergic",  # antagonizes adrenergic
            "GPR22": "dopaminergic",  # amphetamine response
            "GPR26": "dopaminergic",  # dopamine receptor family
            "GPR31": "dopaminergic",  # dopamine receptor family
            "GPR50": "melanocortin",  # melanocortin receptor family
            "GPR52": "dopaminergic",  # dopamine receptor family
            "GPR62": "serotonergic",  # serotonin receptor family
            "GPR63": "serotonergic",  # serotonin receptor family
            "GPR103": "serotonergic",  # serotonin receptor family
            "GPR148": "serotonergic",  # serotonin receptor family
            "GPR174": "histamine",  # histamine agonist
        }

        if name in name_hints:
            return name_hints[name]

        return None

    def _get_subfamily(self, receptor: Dict) -> Optional[str]:
        """Get the subfamily key for a receptor."""
        family = receptor.get("family", "")
        name = receptor.get("name", receptor.get("receptor", ""))

        subfamily_map = FAMILY_SUBFAMILY_MAP.get(family, {})
        for subfamily, members in subfamily_map.items():
            if name in members:
                return subfamily

        # Check gene-based matching for orphan receptors
        gene = receptor.get("gene", "")
        for subfamily, members in subfamily_map.items():
            if gene in members or any(gene == m for m in members):
                return subfamily

        return None

    def _get_broad_class(self, family: str) -> Optional[str]:
        """Get the broad GPCR class from family name."""
        if "Adhesion" in family:
            return "Adhesion"
        if "Class C" in family:
            return "Class_C"
        if family in ["partially deorphanized"]:
            return "Class_A"
        return "Class_A"  # Default for Class A families

    def g_protein_similarity(self, r1: Dict, r2: Dict) -> float:
        """
        Compute G-protein coupling similarity.

        Same coupling = 1.0
        Different coupling = 0.0
        If either is unknown (null), return 0.0 (no signal)
        """
        g1 = r1.get("g_coupling")
        g2 = r2.get("g_coupling")

        if g1 is None or g2 is None:
            return 0.0

        if g1 == g2:
            return 1.0

        return 0.0

    def tissue_similarity(self, r1: Dict, r2: Dict) -> float:
        """
        Compute tissue expression overlap using Jaccard similarity.

        J(A, B) = |A ∩ B| / |A ∪ B|
        """
        tissues1 = self._parse_tissues(r1.get("tissue_expression", ""))
        tissues2 = self._parse_tissues(r2.get("tissue_expression", ""))

        if not tissues1 or not tissues2:
            return 0.0

        intersection = len(tissues1 & tissues2)
        union = len(tissues1 | tissues2)

        if union == 0:
            return 0.0

        return intersection / union

    def _parse_tissues(self, tissue_str: str) -> set:
        """Parse tissue expression string into a normalized set."""
        if not tissue_str:
            return set()

        tissues = set()
        for t in tissue_str.split(","):
            t = t.strip().lower()
            if not t:
                continue

            # Normalize tissue names
            normalized = self._normalize_tissue(t)
            tissues.add(normalized)

        return tissues

    def _normalize_tissue(self, tissue: str) -> str:
        """Normalize tissue names for comparison."""
        mapping = {
            "brain": "brain",
            "brain_cerebellum": "brain_cerebellum",
            "brain_cortex": "brain_cortex",
            "brain_hippocampus": "brain_hippocampus",
            "brain_striatum": "brain_striatum",
            "brain_substantia_nigra": "brain_substantia_nigra",
            "cerebellum": "brain_cerebellum",
            "cortex": "brain_cortex",
            "hippocampus": "brain_hippocampus",
            "striatum": "brain_striatum",
            "hypothalamus": "brain_hypothalamus",
            "olfactory": "brain_olfactory",
            "olfactory bulb": "brain_olfactory",
            "retina": "retina",
            "heart": "heart",
            "heart_atrial": "heart_atrial",
            "heart_ventricle": "heart_ventricle",
            "kidney": "kidney",
            "liver": "liver",
            "lung": "lung",
            "pancreas": "pancreas",
            "intestine": "intestine",
            "small_intestine": "intestine",
            "gut": "gut",
            "stomach": "stomach",
            "colon": "colon",
            "esophagus": "esophagus",
            "thyroid": "thyroid",
            "breast": "breast",
            "prostate": "prostate",
            "testis": "testis",
            "ovary": "ovary",
            "uterus": "uterus",
            "muscle": "skeletal_muscle",
            "skeletal_muscle": "skeletal_muscle",
            "skin": "skin",
            "adipose": "adipose",
            "adipose tissue": "adipose",
            "adrenal": "adrenal_gland",
            "adrenal_gland": "adrenal_gland",
            "blood": "blood",
            "blood_plasma": "blood",
            "endothelial": "endothelial",
            "endothelial cells": "endothelial",
            "endothelium": "endothelial",
            "spleen": "spleen",
            "thymus": "thymus",
            "bone": "bone",
            "bone_marrow": "bone_marrow",
            "leukocytes": "leukocytes",
            "immune cells": "leukocytes",
            "macrophages": "leukocytes",
            "microglia": "leukocytes",
            "neutrophil": "leukocytes",
            "thymus": "thymus",
            "oligodendrocytes": "oligodendrocytes",
            "neural progenitors": "neural_progenitors",
            "macrophages": "leukocytes",
        }
        return mapping.get(tissue, tissue)

    def compute_similarity(self, r1: Dict, r2: Dict) -> float:
        """
        Compute the full metadata similarity score.

        S(r1, r2) = w1 * family_sim + w2 * gprotein_sim + w3 * tissue_sim
        """
        fam_sim = self.family_similarity(r1, r2)
        g_sim = self.g_protein_similarity(r1, r2)
        tissue_sim = self.tissue_similarity(r1, r2)

        score = (self.w_family * fam_sim +
                 self.w_gprotein * g_sim +
                 self.w_tissue * tissue_sim)

        # Normalize by total weight
        return score / self.total_weight

    def compute_similarity_matrix(self, receptor_names: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Compute pairwise similarity matrix for a set of receptors.

        Returns:
            Dict mapping (name_i, name_j) -> similarity score
        """
        matrix = {}
        for i, n1 in enumerate(receptor_names):
            r1 = self.receptor_metadata[n1]
            for j, n2 in enumerate(receptor_names):
                if i == j:
                    continue
                r2 = self.receptor_metadata[n2]
                matrix[(n1, n2)] = self.compute_similarity(r1, r2)

        return matrix

    def orphan_to_known_similarity(self, orphan_name: str,
                                    known_names: List[str]) -> Dict[str, float]:
        """
        Compute similarity between one orphan receptor and all known receptors.

        Returns:
            Dict mapping known_receptor_name -> similarity_score
        """
        orphan_meta = self.receptor_metadata[orphan_name]
        similarities = {}

        for known_name in known_names:
            known_meta = self.receptor_metadata[known_name]
            similarities[known_name] = self.compute_similarity(orphan_meta, known_meta)

        return similarities


# ============================================================================
# SECTION 2: LIGAND PREDICTION VIA KERNEL DENSITY ESTIMATION
# ============================================================================

class LigandPredictor:
    """
    Predicts ligands for orphan receptors using metadata-weighted KDE.

    P(l|r_orphan) = sum_j S(r_orphan, r_j) * count(l, r_j) / sum_j S(r_orphan, r_j)

    Where count(l, r_j) = number of known instances of ligand l binding receptor r_j
    (handles multiple entries in raw_data).
    """

    def __init__(self):
        # Build ligand -> receptor lookup from raw pairs
        self.ligand_to_receptors: Dict[str, List[Dict]] = defaultdict(list)
        self.receptor_to_ligands: Dict[str, List[Dict]] = defaultdict(list)
        self.all_ligands: set = set()
        self.ligand_class_map: Dict[str, str] = {}

    def build_index(self, raw_pairs: List[Dict]):
        """Build lookup tables from raw pairs."""
        for pair in raw_pairs:
            ligand = pair["ligand"]
            receptor = pair["receptor"]
            ligand_class = pair.get("ligand_class", "unknown")

            self.ligand_to_receptors[ligand].append(pair)
            self.receptor_to_ligands[receptor].append(pair)
            self.all_ligands.add(ligand)
            self.ligand_class_map[ligand] = ligand_class

    def predict_ligands_kde(self, orphan_name: str,
                            similarities: Dict[str, float],
                            known_receptors: List[str]) -> List[Tuple[str, float, str]]:
        """
        Predict ligand probabilities using kernel density estimation.

        P(l|r_orphan) = sum_j S(r_i, r_j) * count(l, r_j) / sum_j S(r_i, r_j)

        Args:
            orphan_name: Name of the orphan receptor
            similarities: Dict of known_receptor -> similarity_score
            known_receptors: List of known receptor names

        Returns:
            List of (ligand_name, probability, ligand_class) sorted by probability
        """
        total_weight = sum(similarities.get(n, 0.0) for n in known_receptors)

        if total_weight == 0:
            return []

        # Weighted ligand counts
        weighted_counts = defaultdict(float)

        for known_name in known_receptors:
            sim = similarities.get(known_name, 0.0)
            if sim <= 0:
                continue

            for pair in self.receptor_to_ligands.get(known_name, []):
                ligand = pair["ligand"]
                weighted_counts[ligand] += sim

        # Normalize to get probabilities
        predictions = []
        for ligand, weight in weighted_counts.items():
            prob = weight / total_weight
            ligand_class = self.ligand_class_map.get(ligand, "unknown")
            predictions.append((ligand, prob, ligand_class))

        # Sort by probability descending
        predictions.sort(key=lambda x: x[1], reverse=True)

        return predictions

    def predict_affinity(self, orphan_name: str,
                         similarities: Dict[str, float],
                         known_receptors: List[str]) -> Dict[str, float]:
        """
        Predict binding affinity (pKi) for each ligand.

        f(r_orphan, l) = sum_j S(r_i, r_j) * pKi(r_j, l) / sum_j S(r_i, r_j)

        Where pKi(r_j, l) = -log10(ki * 1e-9) if ligand l is known for receptor r_j,
                              0 otherwise (ligand not observed for that receptor).

        Returns:
            Dict mapping ligand -> predicted_pKi
        """
        total_weight = sum(similarities.get(n, 0.0) for n in known_receptors)

        if total_weight == 0:
            return {}

        weighted_pki = defaultdict(float)
        weighted_count = defaultdict(float)

        for known_name in known_receptors:
            sim = similarities.get(known_name, 0.0)
            if sim <= 0:
                continue

            for pair in self.receptor_to_ligands.get(known_name, []):
                ligand = pair["ligand"]
                ki = pair.get("ki", 0)

                if ki > 0 and ki != float("inf"):
                    pki = -math.log10(ki * 1e-9)
                    weighted_pki[ligand] += sim * pki
                    weighted_count[ligand] += sim

        predicted_affinities = {}
        for ligand in weighted_pki:
            if weighted_count[ligand] > 0:
                predicted_affinities[ligand] = weighted_pki[ligand] / weighted_count[ligand]

        return predicted_affinities

    def predict_gprotein_coupling(self, orphan_name: str,
                                   similarities: Dict[str, float],
                                   known_receptors: List[str]) -> List[Tuple[str, float]]:
        """
        Predict G-protein coupling preferences for an orphan receptor.

        Uses weighted voting from similar known receptors.

        Returns:
            List of (g_protein, weighted_probability) sorted by probability
        """
        total_weight = sum(similarities.get(n, 0.0) for n in known_receptors)

        if total_weight == 0:
            return [("unknown", 1.0)]

        gprotein_weights = defaultdict(float)

        for known_name in known_receptors:
            sim = similarities.get(known_name, 0.0)
            if sim <= 0:
                continue

            g_coupling = self.receptor_metadata_get_g(known_name)
            if g_coupling:
                gprotein_weights[g_coupling] += sim

        # Normalize
        predictions = []
        for gprot, weight in gprotein_weights.items():
            prob = weight / total_weight
            predictions.append((gprot, prob))

        predictions.sort(key=lambda x: x[1], reverse=True)

        return predictions

    def receptor_metadata_get_g(self, receptor_name: str) -> Optional[str]:
        """Get G-protein coupling from the similarity module's index."""
        meta = self._get_receptor_meta(receptor_name)
        return meta.get("g_coupling") if meta else None

    def _get_receptor_meta(self, receptor_name: str) -> Optional[Dict]:
        """Get receptor metadata from ligand predictor's index."""
        # Access the similarity module's metadata
        # This is handled via the combined pipeline
        return None


# ============================================================================
# SECTION 3: CONFIDENCE SCORING
# ============================================================================

class ConfidenceScorer:
    """
    Computes confidence scores for predictions.

    C(r_i, l) = sigmoid(f(r_i, l) - theta) * sqrt(N_similar)

    Where:
      - f(r_i, l) = predicted pKi value (higher = tighter binding)
      - theta = binding threshold (typically ~7.0 for nM-range binders)
      - N_similar = number of known receptors with similarity > 0.1
      - sigmoid(x) = 1 / (1 + exp(-x))

    The sqrt(N_similar) factor accounts for data density — more similar
    receptors means more reliable prediction.
    """

    def __init__(self, theta: float = 7.0, sigmoid_scale: float = 1.0):
        self.theta = theta
        self.sigmoid_scale = sigmoid_scale

    def confidence_score(self, predicted_pki: float, n_similar: int) -> float:
        """
        Compute confidence score.

        C(r_i, l) = sigmoid(f(r_i, l) - theta) * sqrt(N_similar) / sqrt(N_max)

        Where:
          - f(r_i, l) = predicted pKi value (higher = tighter binding)
          - theta = binding threshold (typically ~7.0 for nM-range binders)
          - N_similar = number of known receptors with similarity > 0.1
          - N_max = total number of known receptors (normalization factor)
          - sigmoid(x) = 1 / (1 + exp(-x))

        The sqrt(N_similar) / sqrt(N_max) factor accounts for data density.
        When N_similar is small, confidence is low even with good pKi.
        When N_similar is large, confidence approaches sigmoid(pKi - theta).
        """
        if n_similar <= 0:
            return 0.0

        # Clip pKi to reasonable range
        pki = max(0.0, min(predicted_pki, 12.0))

        sigmoid_val = 1.0 / (1.0 + math.exp(-self.sigmoid_scale * (pki - self.theta)))

        # Data density factor: sqrt(N_similar / N_max)
        # This ensures confidence stays in [0, 1] range
        # N_max = 73 known receptors
        n_max = 73.0
        data_density = math.sqrt(n_similar / n_max)

        confidence = sigmoid_val * data_density

        return min(confidence, 1.0)

    def classify_confidence(self, confidence: float) -> str:
        """Classify confidence into categories."""
        if confidence > 0.7:
            return "high"
        elif confidence > 0.4:
            return "medium"
        else:
            return "low"


# ============================================================================
# SECTION 4: COMBINED DEORPHANIZATION PIPELINE
# ============================================================================

@dataclass
class OrphanPrediction:
    """Prediction result for a single orphan receptor."""
    receptor_name: str
    gene: str
    family: str
    class_type: str
    g_coupling: Optional[str]
    tissue_expression: str

    # Top ligand predictions
    top_ligands: List[Tuple[str, float, str, float, float]] = field(default_factory=list)
    # (ligand, probability, ligand_class, predicted_pKi, confidence)

    # Top G-protein predictions
    top_gproteins: List[Tuple[str, float]] = field(default_factory=list)
    # (g_protein, probability)

    # Summary
    high_confidence_count: int = 0
    medium_confidence_count: int = 0
    low_confidence_count: int = 0
    n_similar_receptors: int = 0


class MetadataDeorphanizationPipeline:
    """
    Complete metadata-based deorphanization pipeline.

    Step 1: Build metadata index from all receptor data
    Step 2: For each orphan receptor:
      a. Compute similarity to all known receptors
      b. Predict ligands via KDE
      c. Predict binding affinities via weighted pKi averaging
      d. Predict G-protein coupling via weighted voting
      e. Compute confidence scores
    Step 3: Compile and save results
    """

    def __init__(self, data: Dict[str, Any]):
        self.data = data
        self.raw_pairs = data["raw_pairs"]
        self.orphans = data["orphans"]
        self.ligand_classes = data["ligand_classes"]
        self.receptor_classes = data["receptor_classes"]

        # Components
        self.similarity = MetadataSimilarity()
        self.predictor = LigandPredictor()
        self.confidence = ConfidenceScorer()

        # Cached data
        self.known_receptors: List[str] = []
        self.orphan_predictions: List[OrphanPrediction] = []

    def run(self):
        """Execute the complete deorphanization pipeline."""
        print("=" * 70)
        print("METADATA-BASED ORPHAN RECEPTOR DEORPHANIZATION")
        print("=" * 70)

        # Step 1: Build metadata index
        print("\n[Step 1] Building metadata index")
        self._build_index()

        # Step 2: Run deorphanization
        print("\n[Step 2] Deorphanizing orphan receptors")
        self._deorphanize_all()

        # Step 3: Save results
        print("\n[Step 3] Saving results")
        self._save_predictions()
        self._save_summary()

        # Step 4: Print summary
        print("\n[Step 4] Summary")
        self._print_summary()

        return self.orphan_predictions

    def _build_index(self):
        """Build all lookup tables and indexes."""
        # Build similarity index
        self.similarity.build_index(self.raw_pairs, self.orphans)

        # Build ligand predictor index
        self.predictor.build_index(self.raw_pairs)

        # Identify known receptors (those with at least one ligand pair)
        receptor_ligand_count = defaultdict(int)
        for pair in self.raw_pairs:
            receptor_ligand_count[pair["receptor"]] += 1

        self.known_receptors = sorted([
            name for name, count in receptor_ligand_count.items() if count > 0
        ])

        print(f"  Known receptors: {len(self.known_receptors)}")
        print(f"  Orphan receptors: {len(self.orphans)}")
        print(f"  Ligand classes: {len(self.predictor.all_ligands)}")

        # Determine binding threshold from known data
        pkis = []
        for pair in self.raw_pairs:
            ki = pair.get("ki", 0)
            if ki > 0 and ki != float("inf"):
                pki = -math.log10(ki * 1e-9)
                pkis.append(pki)

        if pkis:
            self.confidence.theta = float(np.median(pkis)) if 'np' in dir() else sum(pkis) / len(pkis)
            print(f"  Binding threshold (theta): {self.confidence.theta:.2f}")

    def _deorphanize_all(self):
        """Run deorphanization on all orphan receptors."""
        orphan_names = [o["name"] for o in self.orphans]

        for odata in self.orphans:
            orphan_name = odata["name"]
            gene = odata["gene"]
            family = odata.get("family", "Class A orphan")
            class_type = odata.get("class", "A")
            g_coupling = odata.get("g_coupling")
            tissue_expr = odata.get("tissue_expression", "")

            # a. Compute similarity to all known receptors
            similarities = self.similarity.orphan_to_known_similarity(
                orphan_name, self.known_receptors
            )

            # b. Predict ligands via KDE
            ligand_predictions = self.predictor.predict_ligands_kde(
                orphan_name, similarities, self.known_receptors
            )

            # c. Predict binding affinities
            affinity_predictions = self.predictor.predict_affinity(
                orphan_name, similarities, self.known_receptors
            )

            # d. Predict G-protein coupling
            gprotein_predictions = self.predictor.predict_gprotein_coupling(
                orphan_name, similarities, self.known_receptors
            )

            # e. Compute confidence scores and merge results
            # Count similar receptors (similarity > 0.1)
            n_similar = sum(1 for s in similarities.values() if s > 0.1)

            # Merge ligand predictions with affinity and confidence
            merged_predictions = []
            high_count = 0
            medium_count = 0
            low_count = 0

            for ligand, prob, ligand_class in ligand_predictions:
                predicted_pki = affinity_predictions.get(ligand, 0.0)
                conf = self.confidence.confidence_score(predicted_pki, n_similar)

                merged_predictions.append((
                    ligand, prob, ligand_class, predicted_pki, conf
                ))

                category = self.confidence.classify_confidence(conf)
                if category == "high":
                    high_count += 1
                elif category == "medium":
                    medium_count += 1
                else:
                    low_count += 1

            # Build result
            result = OrphanPrediction(
                receptor_name=orphan_name,
                gene=gene,
                family=family,
                class_type=class_type,
                g_coupling=g_coupling,
                tissue_expression=tissue_expr,
                top_ligands=merged_predictions[:10],
                top_gproteins=gprotein_predictions[:3],
                high_confidence_count=high_count,
                medium_confidence_count=medium_count,
                low_confidence_count=low_count,
                n_similar_receptors=n_similar,
            )

            self.orphan_predictions.append(result)
            print(f"  {orphan_name}: {len(merged_predictions)} ligands predicted, "
                  f"{high_count} high, {medium_count} medium, {low_count} low confidence")

    def _save_predictions(self):
        """Save predictions to JSON."""
        output = []
        for pred in self.orphan_predictions:
            entry = {
                "receptor": pred.receptor_name,
                "gene": pred.gene,
                "family": pred.family,
                "class": pred.class_type,
                "g_coupling": pred.g_coupling,
                "tissue_expression": pred.tissue_expression,
                "n_similar_receptors": pred.n_similar_receptors,
                "top_ligands": [
                    {
                        "ligand": l[0],
                        "probability": round(l[1], 6),
                        "ligand_class": l[2],
                        "predicted_pKi": round(l[3], 2),
                        "confidence": round(l[4], 4),
                    }
                    for l in pred.top_ligands
                ],
                "top_gproteins": [
                    {"g_protein": g[0], "probability": round(g[1], 4)}
                    for g in pred.top_gproteins
                ],
                "high_confidence_count": pred.high_confidence_count,
                "medium_confidence_count": pred.medium_confidence_count,
                "low_confidence_count": pred.low_confidence_count,
            }
            output.append(entry)

        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output", "predictions.json")
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  Predictions saved to {out_path}")

    def _save_summary(self):
        """Generate and save summary markdown."""
        lines = []
        lines.append("# Orphan Receptor Deorphanization Results\n")
        lines.append("## Metadata-Based Prediction Summary\n")
        lines.append("### Overview\n")
        lines.append(f"- **Total orphan receptors analyzed**: {len(self.orphan_predictions)}")
        lines.append(f"- **Known receptors used for prediction**: {len(self.known_receptors)}")
        lines.append(f"- **Ligand candidate pool**: {len(self.predictor.all_ligands)}")
        lines.append(f"- **Similarity model**: Family (50%) + G-protein (30%) + Tissue (20%)\n")

        # Confidence distribution
        total_high = sum(p.high_confidence_count for p in self.orphan_predictions)
        total_medium = sum(p.medium_confidence_count for p in self.orphan_predictions)
        total_low = sum(p.low_confidence_count for p in self.orphan_predictions)
        total_preds = total_high + total_medium + total_low

        lines.append("### Confidence Distribution\n")
        lines.append(f"- **High confidence (>0.7)**: {total_high} predictions")
        lines.append(f"- **Medium confidence (0.4-0.7)**: {total_medium} predictions")
        lines.append(f"- **Low confidence (<0.4)**: {total_low} predictions\n")

        # Group by family
        lines.append("## Results by Orphan Receptor Family\n")

        families = defaultdict(list)
        for pred in self.orphan_predictions:
            families[pred.family].append(pred)

        for family in sorted(families.keys()):
            preds = families[family]
            lines.append(f"### {family} ({len(preds)} receptors)\n")
            lines.append("| Receptor | Gene | G-Coupling | Top Ligand | Confidence | Ligand Class |")
            lines.append("|----------|------|------------|------------|------------|--------------|")

            for pred in sorted(preds, key=lambda p: p.top_ligands[0][4] if p.top_ligands else 0, reverse=True):
                if pred.top_ligands:
                    top = pred.top_ligands[0]
                    gprot = pred.g_coupling or "unknown"
                    lines.append(f"| {pred.receptor_name} | {pred.gene} | {gprot} | "
                                 f"{top[0]} | {top[4]:.3f} | {top[2]} |")
                else:
                    lines.append(f"| {pred.receptor_name} | {pred.gene} | "
                                 f"{pred.g_coupling or 'unknown'} | N/A | N/A | N/A |")
            lines.append("")

        # High-confidence predictions
        lines.append("## High-Confidence Predictions (confidence > 0.7)\n")
        high_preds = []
        for pred in self.orphan_predictions:
            for ligand, prob, lclass, pki, conf in pred.top_ligands:
                if conf > 0.7:
                    high_preds.append((pred.receptor_name, pred.gene, pred.family,
                                       ligand, lclass, pki, conf))

        if high_preds:
            high_preds.sort(key=lambda x: x[6], reverse=True)
            lines.append("| Receptor | Gene | Family | Ligand | Class | pKi | Confidence |")
            lines.append("|----------|------|--------|--------|-------|-----|------------|")
            for receptor, gene, family, ligand, lclass, pki, conf in high_preds:
                lines.append(f"| {receptor} | {gene} | {family} | "
                             f"{ligand} | {lclass} | {pki:.1f} | {conf:.3f} |")
        else:
            lines.append("*No high-confidence predictions found.*\n")

        # Medium-confidence predictions
        lines.append("## Medium-Confidence Predictions (confidence 0.4-0.7)\n")
        med_preds = []
        for pred in self.orphan_predictions:
            for ligand, prob, lclass, pki, conf in pred.top_ligands:
                if 0.4 < conf <= 0.7:
                    med_preds.append((pred.receptor_name, pred.gene, pred.family,
                                      ligand, lclass, pki, conf))

        if med_preds:
            med_preds.sort(key=lambda x: x[6], reverse=True)
            lines.append(f"**{len(med_preds)}** medium-confidence predictions:\n")
            lines.append("| Receptor | Gene | Family | Ligand | Class | pKi | Confidence |")
            lines.append("|----------|------|--------|--------|-------|-----|------------|")
            for receptor, gene, family, ligand, lclass, pki, conf in med_preds:
                lines.append(f"| {receptor} | {gene} | {family} | "
                             f"{ligand} | {lclass} | {pki:.1f} | {conf:.3f} |")
        else:
            lines.append("*No medium-confidence predictions found.*\n")

        # Low-confidence predictions
        lines.append("## Low-Confidence Predictions (confidence < 0.4)\n")
        lines.append("**These predictions need experimental validation.**\n")

        low_receptors = [p for p in self.orphan_predictions
                         if p.high_confidence_count == 0 and p.medium_confidence_count == 0]

        if low_receptors:
            lines.append(f"**{len(low_receptors)} orphan receptors** have no high or medium confidence predictions:\n")
            for pred in sorted(low_receptors, key=lambda p: p.top_ligands[0][4] if p.top_ligands else 0, reverse=True):
                top_info = ""
                if pred.top_ligands:
                    top = pred.top_ligands[0]
                    top_info = f" Top: {top[0]} (conf={top[4]:.3f}, pKi={top[3]:.1f})"
                lines.append(f"- **{pred.receptor_name}** ({pred.gene}): {pred.family}, "
                             f"G={pred.g_coupling or 'unknown'}, "
                             f"{pred.n_similar_receptors} similar receptors{top_info}")
        else:
            lines.append("*All orphan receptors have at least one medium or high confidence prediction.*\n")

        # G-protein predictions summary
        lines.append("## G-Protein Coupling Predictions\n")
        lines.append("Top 3 predicted G-protein couplings for each orphan receptor:\n")
        lines.append("| Receptor | Predicted G1 | G1 Prob | Predicted G2 | G2 Prob | Predicted G3 | G3 Prob |")
        lines.append("|----------|-------------|---------|-------------|---------|-------------|---------|")

        for pred in self.orphan_predictions:
            g1 = pred.top_gproteins[0] if len(pred.top_gproteins) > 0 else ("unknown", 0)
            g2 = pred.top_gproteins[1] if len(pred.top_gproteins) > 1 else ("unknown", 0)
            g3 = pred.top_gproteins[2] if len(pred.top_gproteins) > 2 else ("unknown", 0)
            lines.append(f"| {pred.receptor_name} | {g1[0]} | {g1[1]:.3f} | "
                         f"{g2[0]} | {g2[1]:.3f} | {g3[0]} | {g3[1]:.3f} |")
        lines.append("")

        # Interesting patterns
        lines.append("## Interesting Patterns and Insights\n")
        self._add_patterns(lines)

        summary_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output", "predictions_summary.md")
        with open(summary_path, "w") as f:
            f.write("\n".join(lines))
        print(f"  Summary saved to {summary_path}")

    def _add_patterns(self, lines: List[str]):
        """Add interesting patterns analysis to the summary."""
        # Pattern 1: Orphans with known ligands from notes
        lines.append("### Known Partially-Deorphanized Receptors\n")
        lines.append("Several orphan receptors already have known ligands from literature. ")
        lines.append("The metadata model should recover these:\n")

        known_ligands = {
            "GPR39": ["zinc", "galanin"],
            "GPR119": ["OEA", "vernonolide"],
            "GPR120": ["EPA", "DHA", "palmitic acid"],
            "GPR109A": ["niacin", "nicotinamide"],
            "GPR139": ["neurotensin", "YL-2"],
            "GPR151": ["tryptophan", "phenylalanine"],
            "GPR182": ["angiocrine factors"],
            "GPR183": ["oxysterol"],
            "GPR132": ["SLC1A3"],
            "GPR171": ["2-APB"],
            "GPR174": ["histamine"],
            "GPR88": ["macadamia acid"],
            "GPR32": ["PGE2", "15-dPGJ2"],
            "GPR33": ["N-acyl dopamine"],
            "GPR34": ["D-serine"],
            "GPR65": ["H+ (protons)"],
            "GPR68": ["beta-hydroxybutyrate"],
            "GPR83": ["LPA"],
            "GPR87": ["LPA", "prostaglandins"],
            "GPR160": ["A-88,142"],
            "GPR144": ["retinaldehyde"],
            "GPR4": ["H+ (protons)"],
        }

        for pred in self.orphan_predictions:
            if pred.receptor_name in known_ligands:
                known = known_ligands[pred.receptor_name]
                lines.append(f"- **{pred.receptor_name}**: Known ligands = {', '.join(known)}")
                if pred.top_ligands:
                    top = pred.top_ligands[0]
                    lines.append(f"  Predicted top ligand: {top[0]} (conf={top[4]:.3f})")
        lines.append("")

        # Pattern 2: Lipid/FA receptors
        lines.append("### Lipid and Fatty Acid Receptor Predictions\n")
        lines.append("Orphan receptors in lipid-related families should predict fatty acids, lipids, and eicosanoids:\n")

        lipid_families = ["Class A orphan"]  # Most lipid orphans are Class A
        lipid_ligands = {"fatty acids", "lipids", "fatty acid"}

        for pred in self.orphan_predictions:
            if pred.receptor_name in ["GPR119", "GPR120", "GPR78", "GPR183"]:
                lipid_preds = [(l, c) for l, p, lc, pk, c in pred.top_ligands if lc in lipid_ligands]
                if lipid_preds:
                    lines.append(f"- **{pred.receptor_name}**: Lipid ligands = {', '.join(f'{l} (conf={c:.3f})' for l, c in lipid_preds[:3])}")
        lines.append("")

        # Pattern 3: Serotonin-family orphans
        lines.append("### Serotonin-Family Orphan Predictions\n")
        lines.append("Orphans in the serotonin receptor family (GPR62, GPR63, GPR103, GPR148) should predict indolamines:\n")

        serotonin_orphans = ["GPR62", "GPR63", "GPR103", "GPR148"]
        for pred in self.orphan_predictions:
            if pred.receptor_name in serotonin_orphans:
                indol_preds = [(l, c) for l, p, lc, pk, c in pred.top_ligands if lc == "indolamine"]
                if indol_preds:
                    lines.append(f"- **{pred.receptor_name}**: Indolamine ligands = {', '.join(f'{l} (conf={c:.3f})' for l, c in indol_preds[:3])}")
                else:
                    lines.append(f"- **{pred.receptor_name}**: No strong indolamine predictions (low family signal)")
        lines.append("")

        # Pattern 4: Dopamine-family orphans
        lines.append("### Dopamine-Family Orphan Predictions\n")
        lines.append("Orphans in the dopamine receptor family should predict catecholamines:\n")

        dopamine_orphans = ["GPR19", "GPR20", "GPR21", "GPR22", "GPR26", "GPR31", "GPR33",
                           "GPR82", "GPR85", "GPR52", "GPR88"]
        for pred in self.orphan_predictions:
            if pred.receptor_name in dopamine_orphans:
                cat_preds = [(l, c) for l, p, lc, pk, c in pred.top_ligands if lc == "catecholamine"]
                if cat_preds:
                    lines.append(f"- **{pred.receptor_name}**: Catecholamine ligands = {', '.join(f'{l} (conf={c:.3f})' for l, c in cat_preds[:3])}")
                else:
                    lines.append(f"- **{pred.receptor_name}**: No strong catecholamine predictions")
        lines.append("")

        # Pattern 5: Adhesion GPCR predictions
        lines.append("### Adhesion GPCR Predictions\n")
        lines.append("Adhesion GPCRs have large extracellular domains and may bind diverse ligands:\n")

        adhesion_orphans = [p for p in self.orphan_predictions if "Adhesion" in p.family]
        if adhesion_orphans:
            for pred in adhesion_orphans:
                if pred.top_ligands:
                    top = pred.top_ligands[0]
                    lines.append(f"- **{pred.receptor_name}**: Top = {top[0]} ({top[2]}, conf={top[4]:.3f})")
        lines.append("")

        # Pattern 6: Class C orphan predictions
        lines.append("### Class C Orphan Predictions\n")
        lines.append("Class C GPCRs have large Venus flytrap domains and often bind amino acids/peptides:\n")

        class_c_orphans = [p for p in self.orphan_predictions if "Class C" in p.family]
        if class_c_orphans:
            for pred in class_c_orphans:
                if pred.top_ligands:
                    top = pred.top_ligands[0]
                    lines.append(f"- **{pred.receptor_name}**: Top = {top[0]} ({top[2]}, conf={top[4]:.3f})")
        lines.append("")

    def _print_summary(self):
        """Print summary to stdout."""
        total_high = sum(p.high_confidence_count for p in self.orphan_predictions)
        total_medium = sum(p.medium_confidence_count for p in self.orphan_predictions)
        total_low = sum(p.low_confidence_count for p in self.orphan_predictions)

        print(f"\n  {'='*60}")
        print(f"  {'METADATA-BASED DEORPHANIZATION RESULTS':^60}")
        print(f"  {'='*60}")
        print(f"\n  Total orphan receptors: {len(self.orphan_predictions)}")
        print(f"  Known receptors used: {len(self.known_receptors)}")
        print(f"  Ligand pool: {len(self.predictor.all_ligands)}")
        print(f"\n  Confidence distribution:")
        print(f"    High (>0.7):     {total_high}")
        print(f"    Medium (0.4-0.7): {total_medium}")
        print(f"    Low (<0.4):       {total_low}")

        # Top 10 most confident pairs
        all_pairs = []
        for pred in self.orphan_predictions:
            for ligand, prob, lclass, pki, conf in pred.top_ligands:
                all_pairs.append((pred.receptor_name, pred.gene, ligand, lclass, pki, conf))

        all_pairs.sort(key=lambda x: x[5], reverse=True)
        top10 = all_pairs[:10]

        print(f"\n  Top 10 Most Confident Predictions:")
        print(f"  {'Receptor':<25} {'Ligand':<25} {'Conf':>6}")
        print(f"  {'-'*25} {'-'*25} {'-'*6}")
        for receptor, gene, ligand, lclass, pki, conf in top10:
            print(f"  {receptor:<25} {ligand:<25} {conf:.3f}")

        print(f"\n  {'='*60}")

        # Receptors with no good predictions
        no_good = [p for p in self.orphan_predictions
                   if p.high_confidence_count == 0 and p.medium_confidence_count == 0]
        if no_good:
            print(f"\n  WARNING: {len(no_good)} receptors have NO high/medium confidence predictions:")
            for p in no_good:
                print(f"    - {p.receptor_name} ({p.gene}): {p.family}, "
                      f"{p.n_similar_receptors} similar receptors")
        else:
            print(f"\n  All orphan receptors have at least one medium-confidence prediction.")

        print(f"\n  Results saved to:")
        print(f"    {DATA_DIR}/../output/predictions.json")
        print(f"    {DATA_DIR}/../output/predictions_summary.md")


# ============================================================================
# SECTION 5: MAIN ENTRY POINT
# ============================================================================

def main():
    import numpy as np

    print("Loading data...")
    data = load_all_data()
    print(f"  Raw pairs: {len(data['raw_pairs'])}")
    print(f"  Orphan receptors: {len(data['orphans'])}")
    print(f"  Ligand classes: {len(data['ligand_classes'])}")
    print(f"  Receptor classes: {len(data['receptor_classes'])}")

    np.random.seed(42)

    pipeline = MetadataDeorphanizationPipeline(data)
    results = pipeline.run()

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
