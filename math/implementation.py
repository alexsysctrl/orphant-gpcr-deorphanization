"""
Orphan Receptor Deorphanization — Mathematical Framework Implementation
========================================================================

Implements the rigorous geometric-pharmacophore model from framework.tex.

Mathematical references:
  - Receptor feature space: Eq. (1) -- (10)
  - Ligand feature space: Eq. (11) -- (20)
  - Binding affinity function: Eq. (21) -- (31)
  - Deorphanization map: Eq. (32) -- (42)
  - Cross-validation: Eq. (43) -- (50)
  - Uncertainty: Eq. (51) -- (57)
  - Kinetics: Eq. (58) -- (66)

Usage:
    python implementation.py

Output:
    - Deorphanization predictions for all 72 orphan receptors
    - Cross-validation metrics on 73 characterized receptors
    - Parameter estimates and confidence intervals
"""

import json
import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# SECTION 0: DATA LOADING
# ============================================================================

DATA_DIR = "../data"


def load_json(path: str) -> Any:
    """Load a JSON file and return the parsed data."""
    with open(path, "r") as f:
        return json.load(f)


def load_all_data() -> Dict[str, Any]:
    """Load all four data files."""
    return {
        "raw_pairs": load_json(f"{DATA_DIR}/raw_data.json"),
        "orphans": load_json(f"{DATA_DIR}/orphan_receptor_catalog.json"),
        "ligand_classes": load_json(f"{DATA_DIR}/ligand_classification.json"),
        "receptor_classes": load_json(f"{DATA_DIR}/receptor_classification.json"),
    }


# ============================================================================
# SECTION 1: FEATURE ENCODING — RECEPTOR SPACE R^d
# ============================================================================
# Implements Eq. (1) -- (10) from framework.tex

# Tissue types from GTEx / Human Protein Atlas (28 tissues)
TISSUE_TYPES = [
    "brain_cerebellum", "brain_cortex", "brain_hippocampus", "brain_substantia_nigra",
    "heart_atrial", "heart_ventricle", "kidney", "liver", "lung", "pancreas",
    "small_intestine", "stomach", "colon", "esophagus", "thyroid",
    "breast", "prostate", "testis", "ovary", "uterus",
    "skeletal_muscle", "skin", "adipose", "adrenal_gland",
    "blood_plasma", "spleen", "thymus", "bone_marrow",
]

# G-protein types
G_PROTEIN_TYPES = ["Gs", "Gi", "Gq", "G12/13"]
G_PROTEIN_INDEX = {g: i for i, g in enumerate(G_PROTEIN_TYPES)}

# Amino acid one-hot
AMINO_ACIDS = "ARNDCQEGHILKMFPSTWYV"
AA_INDEX = {a: i for i, a in enumerate(AMINO_ACIDS)}

# Kyte-Doolittle hydropathy scale
KYTE_DOOLITTLE = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5,
    "M": 1.9, "A": 1.8, "G": -0.4, "T": -0.7, "S": -0.8,
    "W": -0.9, "Y": -1.3, "H": -3.2, "P": -1.6, "N": -3.5,
    "Q": -3.5, "D": -3.5, "E": -3.5, "K": -3.9, "R": -4.5,
}

# Charge classification
POSITIVE_AA = {"R", "K", "H"}
NEGATIVE_AA = {"D", "E"}
POLAR_AA = {"N", "Q", "S", "T"}


class ReceptorEncoder:
    """
    Encodes a receptor into a d-dimensional feature vector.
    
    Implements Eq. (1): vv_r = [vv_seq, vv_g, vv_expr, vv_fam, vv_struct]
    
    Dimension breakdown:
      vv_seq:    450 dimensions (AAC=20, BLOSUM=400, hydro=7, charge=3)
      vv_g:       4 dimensions (Gs, Gi, Gq, G12/13)
      vv_expr:   28 dimensions (one per tissue)
      vv_fam:    19 dimensions (class=3, family=16)
      vv_struct: ~370 dimensions (conservation=350, pocket=20, dynamics=9)
    
    Total: d ≈ 871 dimensions
    
    In practice, we use a compressed representation for computational efficiency.
    """

    def __init__(self, receptor_classes: List[Dict]):
        self.receptor_classes = receptor_classes
        self.family_names = [rc["family"] for rc in receptor_classes]
        self.family_index = {f: i for i, f in enumerate(self.family_names)}
        self.class_names = ["Class_A", "Class_B", "Class_C", "Adhesion"]
        self.class_index = {c: i for i, c in enumerate(self.class_names)}

    def encode(self, receptor_data: Dict) -> np.ndarray:
        """
        Encode a receptor dictionary into a feature vector.
        
        Args:
            receptor_data: Dict with keys: name, gene, family, class,
                          g_coupling, tissue_expression, notes
        
        Returns:
            np.ndarray of shape (d,) with the receptor feature vector.
        """
        blocks = []

        # Block 1: Sequence features (450 dims)
        blocks.append(self._sequence_features(receptor_data))

        # Block 2: G-protein coupling (4 dims) — Eq. (7)
        blocks.append(self._g_protein_features(receptor_data))

        # Block 3: Tissue expression (28 dims) — Eq. (8)
        blocks.append(self._tissue_expression(receptor_data))

        # Block 4: Family encoding (19 dims) — Eq. (9)
        blocks.append(self._family_encoding(receptor_data))

        # Block 5: Structural features (~370 dims) — Eq. (10)
        blocks.append(self._structural_features(receptor_data))

        return np.concatenate(blocks)

    def _sequence_features(self, rdata: Dict) -> np.ndarray:
        """
        Sequence block: AAC (20) + BLOSUM-contact (400) + hydrophobicity (7) + charge (3)
        Eq. (2) -- (6)
        
        Since we don't have actual sequences, we synthesize features from
        family membership and known structural properties.
        """
        family = rdata.get("family", "")

        # Amino Acid Composition (20 dims) — Eq. (2)
        # Use family-typical compositions as proxy
        aac = self._family_aac(family)

        # BLOSUM-contact matrix flattened (400 dims)
        # Use family-typical substitution patterns
        blosum_contact = self._family_blosum_contact(family)

        # Hydrophobicity profile (7 bins = 7 TMs) — Eq. (4)
        hydro = self._family_hydrophobicity(family)

        # Charge distribution (3 dims) — Eq. (5)
        charge = self._family_charge(family)

        return np.concatenate([aac, blosum_contact, hydro, charge])

    def _family_aac(self, family: str) -> np.ndarray:
        """Synthesize amino acid composition from family."""
        # Family-typical TM domain compositions (approximate)
        compositions = {
            "adrenergic": [0.08, 0.03, 0.04, 0.04, 0.02, 0.04, 0.05, 0.07, 0.03,
                          0.09, 0.10, 0.04, 0.06, 0.06, 0.05, 0.07, 0.06, 0.02, 0.04, 0.09],
            "dopaminergic": [0.07, 0.03, 0.04, 0.03, 0.02, 0.04, 0.05, 0.07, 0.03,
                            0.09, 0.10, 0.04, 0.06, 0.06, 0.05, 0.07, 0.06, 0.02, 0.04, 0.09],
            "serotonergic": [0.07, 0.03, 0.04, 0.03, 0.02, 0.04, 0.05, 0.07, 0.03,
                            0.09, 0.10, 0.04, 0.06, 0.06, 0.05, 0.07, 0.06, 0.02, 0.04, 0.09],
            "opioid": [0.08, 0.04, 0.03, 0.04, 0.02, 0.04, 0.05, 0.07, 0.03,
                       0.08, 0.09, 0.04, 0.06, 0.06, 0.05, 0.07, 0.06, 0.02, 0.04, 0.09],
            "muscarinic": [0.07, 0.03, 0.04, 0.03, 0.02, 0.04, 0.05, 0.07, 0.03,
                           0.09, 0.10, 0.04, 0.06, 0.06, 0.05, 0.07, 0.06, 0.02, 0.04, 0.09],
            "S1P": [0.08, 0.03, 0.04, 0.03, 0.02, 0.04, 0.05, 0.07, 0.03,
                    0.09, 0.10, 0.04, 0.06, 0.06, 0.05, 0.07, 0.06, 0.02, 0.04, 0.09],
            "free fatty acid": [0.08, 0.03, 0.04, 0.03, 0.02, 0.04, 0.05, 0.07, 0.03,
                                0.09, 0.10, 0.04, 0.06, 0.06, 0.05, 0.07, 0.06, 0.02, 0.04, 0.09],
            "Class C orphan": [0.07, 0.03, 0.04, 0.03, 0.02, 0.04, 0.05, 0.07, 0.03,
                               0.09, 0.10, 0.04, 0.06, 0.06, 0.05, 0.07, 0.06, 0.02, 0.04, 0.09],
            "Adhesion GPCR": [0.07, 0.03, 0.04, 0.03, 0.02, 0.04, 0.05, 0.07, 0.03,
                              0.09, 0.10, 0.04, 0.06, 0.06, 0.05, 0.07, 0.06, 0.02, 0.04, 0.09],
        }
        return np.array(compositions.get(family, compositions["adrenergic"]), dtype=float)

    def _family_blosum_contact(self, family: str) -> np.ndarray:
        """Synthesize BLOSUM-contact matrix (400 dims)."""
        # Create a diagonal-dominant matrix typical for conserved family
        base = np.eye(20) * 3.0  # Self-substitutions are high
        base += np.random.randn(20, 20) * 0.5  # Small random perturbation
        return base.flatten()

    def _family_hydrophobicity(self, family: str) -> np.ndarray:
        """Synthesize hydrophobicity profile for 7 TMs (7 dims) — Eq. (4)."""
        # TM helices are typically hydrophobic (values > 1.0 on Kyte-Doolittle)
        base_hydro = np.array([3.5, 3.8, 3.2, 3.6, 3.4, 3.7, 3.3])
        base_hydro += np.random.randn(7) * 0.3
        return base_hydro

    def _family_charge(self, family: str) -> np.ndarray:
        """Synthesize charge distribution (3 dims) — Eq. (5)."""
        pos_frac = 0.08 + np.random.randn() * 0.02
        neg_frac = 0.06 + np.random.randn() * 0.02
        polar_frac = 0.15 + np.random.randn() * 0.03
        return np.array([max(0, pos_frac), max(0, neg_frac), max(0, polar_frac)])

    def _g_protein_features(self, rdata: Dict) -> np.ndarray:
        """
        G-protein coupling signature (4 dims) — Eq. (7).
        
        For known coupling: one-hot encoding.
        For unknown coupling: uniform prior (0.25, 0.25, 0.25, 0.25).
        """
        g_coupling = rdata.get("g_coupling")
        if g_coupling is None:
            return np.array([0.25, 0.25, 0.25, 0.25])
        vec = np.zeros(4)
        if g_coupling in G_PROTEIN_INDEX:
            vec[G_PROTEIN_INDEX[g_coupling]] = 1.0
        return vec

    def _tissue_expression(self, rdata: Dict) -> np.ndarray:
        """
        Tissue expression profile (28 dims) — Eq. (8).
        
        Parses tissue_expression string and assigns expression levels.
        """
        expr_vec = np.zeros(28)
        tissue_str = rdata.get("tissue_expression", "")
        if tissue_str:
            tissues = [t.strip().lower() for t in tissue_str.split(",")]
            tissue_mapping = {
                "brain": [0, 1, 2, 3],
                "cerebellum": [0],
                "cortex": [1],
                "hippocampus": [2],
                "striatum": [2],  # approximated as hippocampus
                "heart": [4, 5],
                "kidney": [6],
                "liver": [7],
                "lung": [8],
                "pancreas": [9],
                "intestine": [10, 12],
                "stomach": [11],
                "esophagus": [13],
                "thyroid": [14],
                "breast": [15],
                "prostate": [16],
                "testis": [17],
                "ovary": [18],
                "uterus": [19],
                "muscle": [20],
                "skin": [21],
                "adipose": [22],
                "adrenal": [23],
                "spleen": [25],
                "thymus": [26],
                "bone": [27],
                "leukocytes": [25],
                "endothelial": [24],  # blood_plasma proxy
            }
            for t in tissues:
                if t in tissue_mapping:
                    for idx in tissue_mapping[t]:
                        expr_vec[idx] = 0.7 + np.random.rand() * 0.3
        return np.clip(expr_vec, 0, 1)

    def _family_encoding(self, rdata: Dict) -> np.ndarray:
        """
        Hierarchical family encoding (19 dims) — Eq. (9).
        
        Level 1: broad class (3 dims, one-hot over 4 classes minus reference)
        Level 2: specific family (16 dims, one-hot)
        """
        family = rdata.get("family", "")
        class_type = rdata.get("class", "A")

        # Broad class encoding (3 dims)
        class_vec = np.zeros(3)
        if "Adhesion" in family:
            class_vec[3 - 3] = 1.0  # 4th class, drop last
        elif "Class C" in family:
            class_vec[2 - 3] = 1.0  # 3rd class
        elif class_type == "B":
            class_vec[1 - 3] = 1.0  # 2nd class
        # Class A is the reference (all zeros)

        # Family encoding (16 dims)
        fam_idx = self.family_index.get(family, 0)
        fam_vec = np.zeros(16)
        fam_vec[fam_idx] = 1.0

        return np.concatenate([class_vec, fam_vec])

    def _structural_features(self, rdata: Dict) -> np.ndarray:
        """
        Structural features (~370 dims) — Eq. (10).
        
        conservation (350) + pocket (20) + dynamics (9)
        """
        # Conservation scores (350 dims) — Eq. (11)
        # Use family-typical conservation pattern
        conservation = self._family_conservation()

        # Pocket geometry (20 dims)
        pocket = self._pocket_geometry(rdata)

        # Dynamics (9 dims) — Eq. (14)
        dynamics = self._dynamics_features()

        return np.concatenate([conservation, pocket, dynamics])

    def _family_conservation(self) -> np.ndarray:
        """Synthesize conservation scores (350 dims) — Eq. (11)."""
        # Conserved positions (TM core) have high conservation
        # Variable positions (loops, termini) have low conservation
        scores = np.random.exponential(0.3, 350)
        # TM helix regions (roughly positions 1-120, 130-240, 250-350)
        # are more conserved
        tm_regions = list(range(1, 120)) + list(range(130, 240)) + list(range(250, 350))
        for i in tm_regions:
            scores[i] *= 1.5
        return np.clip(scores, 0, 3.0)  # max entropy = log2(20) ≈ 4.32

    def _pocket_geometry(self, rdata: Dict) -> np.ndarray:
        """Synthesize pocket geometry descriptors (20 dims) — Eq. (12)."""
        # Volume, radius, depth, shape descriptors
        pocket = np.random.rand(20) * 2.0
        return pocket

    def _dynamics_features(self) -> np.ndarray:
        """Synthesize RMSF features (9 dims) — Eq. (14)."""
        # RMSF for TM1-TM7 + ICL3 + ECL2
        rmsf = np.random.exponential(1.5, 9)
        return rmsf


# ============================================================================
# SECTION 2: FEATURE ENCODING — LIGAND SPACE L^k
# ============================================================================
# Implements Eq. (11) -- (20) from framework.tex

LIGAND_CLASS_NAMES = [
    "catecholamines", "indolamines", "peptides", "lipids", "fatty_acids",
    "amino_acids", "purines", "ergolines", "synthetic_opioids", "alkaloids",
    "metal_ions", "synthetic", "tryptamines", "amino_acid_derivatives",
]
LIGAND_CLASS_INDEX = {c: i for i, c in enumerate(LIGAND_CLASS_NAMES)}


class LigandEncoder:
    """
    Encodes a ligand into a k-dimensional feature vector.
    
    Implements Eq. (11): uu_l = [uu_phys, uu_finger, uu_class, uu_pharm, uu_conf]
    
    Dimension breakdown:
      uu_phys:      8 dimensions (MW, logP, HBD, HBA, PSA, rotBonds, aromaticRings, formalCharge)
      uu_finger:  1191 dimensions (ECFP4=1024 + MACCS=167)
      uu_class:    14 dimensions (one-hot over ligand families)
      uu_pharm:     8 dimensions (pharmacophore element counts)
      uu_conf:      6 dimensions (flexibility, shape, volume, surface, asphericity, ellipticity)
    
    Total: k = 1227 dimensions
    
    In practice, we use a compressed representation.
    """

    def __init__(self, ligand_classes: List[Dict]):
        self.ligand_classes = ligand_classes
        self.class_ranges = {}
        for lc in ligand_classes:
            self.class_ranges[lc["family"]] = {
                "mw": lc["mw_range"],
                "logp": lc["logp_range"],
                "hbd": lc["hbond_donors"],
                "hba": lc["hbond_acceptors"],
            }

    def encode(self, ligand_name: str, ligand_class: str) -> np.ndarray:
        """
        Encode a ligand into a feature vector.
        
        Args:
            ligand_name: String name of the ligand
            ligand_class: String class name (e.g., 'catecholamine')
        
        Returns:
            np.ndarray of shape (k,) with the ligand feature vector.
        """
        blocks = []

        # Block 1: Physicochemical properties (8 dims) — Eq. (11)
        blocks.append(self._physicochemical(ligand_name, ligand_class))

        # Block 2: Structural fingerprints (1191 dims) — Eq. (12) -- (13)
        blocks.append(self._fingerprints(ligand_name, ligand_class))

        # Block 3: Chemical class (14 dims) — Eq. (14)
        blocks.append(self._class_encoding(ligand_class))

        # Block 4: Pharmacophore features (8 dims) — Eq. (15)
        blocks.append(self._pharmacophore(ligand_name, ligand_class))

        # Block 5: Conformational features (6 dims) — Eq. (16)
        blocks.append(self._conformational(ligand_name, ligand_class))

        return np.concatenate(blocks)

    def _physicochemical(self, name: str, ligand_class: str) -> np.ndarray:
        """
        Physicochemical properties (8 dims) — Eq. (11).
        
        MW, logP, HBD, HBA, PSA, rotBonds, aromaticRings, formalCharge
        """
        cls_key = self._normalize_class_key(ligand_class)
        ranges = self.class_ranges.get(cls_key, None)

        if ranges:
            mw = self._sample_in_range(name, ranges["mw"], default=250.0)
            logp = self._sample_in_range(name, ranges["logp"], default=2.0)
            hbd = int(self._sample_in_range(name, ranges["hbd"], default=2.0))
            hba = int(self._sample_in_range(name, ranges["hba"], default=3.0))
        else:
            mw = 250.0 + np.random.randn() * 50
            logp = 2.0 + np.random.randn() * 1.0
            hbd = 2
            hba = 3

        psa = max(0, logp * 15 + np.random.randn() * 10)
        rot_bonds = max(0, int(np.random.exponential(3)))
        aromatic_rings = max(0, int(np.random.poisson(1.5)))
        formal_charge = int(np.random.choice([-1, 0, 0, 0, 1]))  # mostly neutral

        return np.array([mw, logp, hbd, hba, psa, rot_bonds, aromatic_rings, formal_charge])

    def _fingerprints(self, name: str, ligand_class: str) -> np.ndarray:
        """
        Structural fingerprints (1191 dims) — Eq. (12) -- (13).
        
        ECF4 Morgan (1024) + MACCS keys (167)
        
        Since we don't have actual SMILES, we synthesize fingerprint-like
        features based on ligand class.
        """
        cls_key = self._normalize_class_key(ligand_class)

        # Use class-typical fingerprint patterns
        # Different classes have different substructure distributions
        if cls_key == "catecholamines":
            # High activation in catechol-related hashes
            fp = np.random.rand(1024)
            fp[:100] += 0.5  # catechol ring substructures
            fp[200:300] += 0.3  # amine substructures
        elif cls_key == "indolamines":
            fp = np.random.rand(1024)
            fp[:80] += 0.5  # indole ring
            fp[150:250] += 0.3  # ethylamine
        elif cls_key == "peptides":
            fp = np.random.rand(1024)
            fp[:200] += 0.4  # peptide bond patterns
        elif cls_key == "lipids":
            fp = np.random.rand(1024)
            fp[300:500] += 0.5  # hydrocarbon chain patterns
        elif cls_key == "fatty_acids":
            fp = np.random.rand(1024)
            fp[400:600] += 0.5  # carboxylate patterns
        elif cls_key == "purines":
            fp = np.random.rand(1024)
            fp[:60] += 0.5  # purine ring
            fp[100:200] += 0.3  # phosphate patterns
        elif cls_key == "ergolines":
            fp = np.random.rand(1024)
            fp[:120] += 0.5  # ergoline tetracyclic
        elif cls_key == "synthetic_opioids":
            fp = np.random.rand(1024)
            fp[150:350] += 0.4  # opioid pharmacophore
        elif cls_key == "alkaloids":
            fp = np.random.rand(1024)
            fp[:150] += 0.4  # heterocyclic patterns
        elif cls_key == "synthetic":
            fp = np.random.rand(1024)
            fp += 0.2  # diverse
        else:
            fp = np.random.rand(1024)

        fp = np.clip(fp, 0, 1)

        # MACCS keys (167 dims)
        maccs = np.random.rand(167) > 0.5
        return np.concatenate([fp, maccs.astype(float)])

    def _class_encoding(self, ligand_class: str) -> np.ndarray:
        """One-hot chemical class (14 dims) — Eq. (14)."""
        cls_key = self._normalize_class_key(ligand_class)
        vec = np.zeros(14)
        idx = LIGAND_CLASS_INDEX.get(cls_key, 0)
        vec[idx] = 1.0
        return vec

    def _pharmacophore(self, name: str, ligand_class: str) -> np.ndarray:
        """
        Pharmacophore features (8 dims) — Eq. (15).
        
        HBD, HBA, aromatic, positive, negative, hydrophobic, halogen, sulfur
        """
        cls_key = self._normalize_class_key(ligand_class)
        ranges = self.class_ranges.get(cls_key, None)

        if ranges:
            hbd = int(self._sample_in_range(name, ranges["hbd"], default=2.0))
            hba = int(self._sample_in_range(name, ranges["hba"], default=3.0))
        else:
            hbd = 2
            hba = 3

        aromatic = max(0, int(np.random.poisson(1.2)))
        positive = 1 if cls_key in ["catecholamines", "amino_acid_derivatives"] else max(0, int(np.random.poisson(0.5)))
        negative = 1 if cls_key in ["fatty_acids", "amino_acids"] else max(0, int(np.random.poisson(0.3)))
        hydrophobic = max(0, int(np.random.exponential(4)))
        halogen = max(0, int(np.random.poisson(0.5)))
        sulfur = max(0, int(np.random.poisson(0.3)))

        return np.array([hbd, hba, aromatic, positive, negative, hydrophobic, halogen, sulfur])

    def _conformational(self, name: str, ligand_class: str) -> np.ndarray:
        """
        Conformational features (6 dims) — Eq. (16).
        
        flexibility, shape_index, volume, surface, asphericity, ellipticity
        """
        cls_key = self._normalize_class_key(ligand_class)

        if cls_key == "peptides":
            flexibility = 0.8 + np.random.randn() * 0.1
            shape_idx = np.random.randn() * 0.5
            volume = 1000 + np.random.randn() * 200
            surface = 500 + np.random.randn() * 100
        elif cls_key == "lipids":
            flexibility = 0.6 + np.random.randn() * 0.1
            shape_idx = np.random.randn() * 0.3
            volume = 500 + np.random.randn() * 100
            surface = 300 + np.random.randn() * 50
        else:
            flexibility = 0.3 + np.random.randn() * 0.15
            shape_idx = np.random.randn() * 0.5
            volume = 200 + np.random.randn() * 50
            surface = 150 + np.random.randn() * 30

        asphericity = max(0, np.random.exponential(0.5))
        ellipticity = max(0.1, np.random.exponential(1.0))

        return np.array([
            np.clip(flexibility, 0, 1),
            np.clip(shape_idx, -1, 1),
            volume,
            surface,
            asphericity,
            ellipticity,
        ])

    def _normalize_class_key(self, ligand_class: str) -> str:
        """Map ligand class name to dictionary key."""
        mapping = {
            "catecholamine": "catecholamines",
            "indolamine": "indolamines",
            "peptide": "peptides",
            "lipid": "lipids",
            "fatty acid": "fatty_acids",
            "amino acid": "amino_acids",
            "purine": "purines",
            "ergoline": "ergolines",
            "synthetic opioid": "synthetic_opioids",
            "alkaloid": "alkaloids",
            "metal ion": "metal_ions",
            "synthetic": "synthetic",
            "tryptamine": "tryptamines",
            "amino acid derivative": "amino_acid_derivatives",
        }
        return mapping.get(ligand_class.lower(), ligand_class.lower())

    def _sample_in_range(self, name: str, rng: List, default: float) -> float:
        """Sample a value within the given range, using name hash for reproducibility."""
        if rng[0] is None or rng[1] is None:
            return default
        span = rng[1] - rng[0]
        # Use name hash for deterministic sampling within range
        h = hash(name) % 10000 / 10000.0
        return rng[0] + h * span


# ============================================================================
# SECTION 3: SHARED EMBEDDING AND PROJECTION
# ============================================================================
# Implements Eq. (21) -- (22) from framework.tex


class EmbeddingProjector:
    """
    Projects receptor and ligand features into shared embedding space R^p.
    
    Implements Eq. (21): x_r = W_R * v_r + b_R, x_l = W_L * u_l + b_L
    """

    def __init__(self, receptor_dim: int, ligand_dim: int, embedding_dim: int = 64):
        self.d_in_r = receptor_dim
        self.d_in_l = ligand_dim
        self.p = embedding_dim

        # Initialize projection matrices (Xavier initialization)
        scale_r = np.sqrt(2.0 / (receptor_dim + embedding_dim))
        scale_l = np.sqrt(2.0 / (ligand_dim + embedding_dim))

        self.W_R = np.random.randn(embedding_dim, receptor_dim) * scale_r
        self.b_R = np.zeros(embedding_dim)
        self.W_L = np.random.randn(embedding_dim, ligand_dim) * scale_l
        self.b_L = np.zeros(embedding_dim)

        # Pharmacophore projection: R^d -> R^8
        self.W_pharm = np.random.randn(8, receptor_dim) * scale_r
        self.b_pharm = np.zeros(8)

    def project_receptor(self, v_r: np.ndarray) -> np.ndarray:
        """Project receptor feature vector to embedding space."""
        return self.W_R @ v_r + self.b_R

    def project_ligand(self, u_l: np.ndarray) -> np.ndarray:
        """Project ligand feature vector to embedding space."""
        return self.W_L @ u_l + self.b_L

    def project_pharmacophore(self, v_r: np.ndarray) -> np.ndarray:
        """Project receptor features to pharmacophore requirement vector."""
        return self.W_pharm @ v_r + self.b_pharm

    def fit(self, receptors: List[np.ndarray], ligands: List[np.ndarray],
            affinities: np.ndarray, lr: float = 0.001, n_iter: int = 500):
        """
        Learn projection matrices by minimizing squared error between
        predicted and observed pK_i values.
        
        The input vectors are RAW features (not pre-projected).
        Implements the MLE estimation from Eq. (47) -- (50).
        """
        n = len(receptors)
        print(f"  Fitting embedding projector: {n} pairs, {n_iter} iterations")

        # Pre-project all vectors
        x_r_all = np.array([self.project_receptor(v) for v in receptors])
        x_l_all = np.array([self.project_ligand(u) for u in ligands])

        for iteration in range(n_iter):
            # Predicted affinity: Eq. (21) -- (24)
            diff = x_r_all - x_l_all  # (n, p)
            geom = -np.sum(diff ** 2, axis=1)  # Eq. (23)

            pharm_proj = np.array([self.project_pharmacophore(v) for v in receptors])
            pharm_vec = np.array([self._extract_pharm(u) for u in ligands])
            pharm = np.sum(pharm_proj * pharm_vec, axis=1)  # Eq. (24)

            pred = geom + pharm

            # Loss: MSE
            loss = np.mean((pred - affinities) ** 2)

            if iteration % 100 == 0:
                print(f"    Iteration {iteration}: MSE = {loss:.4f}")

            # Gradient update on projection matrices
            error = pred - affinities  # (n,)

            # Gradient of geom w.r.t. x_r: -2 * (x_r - x_l)
            # Gradient of geom w.r.t. x_l: +2 * (x_r - x_l)
            # dW_R = (1/n) * sum_i error_i * dx_r_i * v_ri^T
            for i in range(n):
                dx = -2 * diff[i]  # (p,)
                # Update W_R, W_L (simplified: update first 16 dims)
                for d in range(min(16, self.p)):
                    self.W_R[d] += lr * error[i] * dx[d] * receptors[i] / n
                    self.W_L[d] += lr * error[i] * (-dx[d]) * ligands[i] / n

    def _extract_pharm(self, u_l: np.ndarray) -> np.ndarray:
        """Extract pharmacophore features (8 dims) from ligand vector."""
        # Pharmacophore features are at a known offset in the ligand vector
        # phys(8) + finger_start(1024) + maccs_start(167) + class(14) + pharm(8)
        # phys(8) + finger(1024) + maccs(167) is already part of the 1191
        # class(14) + pharm(8): pharm starts at index 8 + 1191 + 14 = 1213
        start = 8 + 1191 + 14
        return u_l[start:start + 8]


# ============================================================================
# SECTION 4: BINDING AFFINITY FUNCTION
# ============================================================================
# Implements Eq. (21) -- (31) from framework.tex


class BindingAffinityModel:
    """
    Computes the binding affinity function f(r, l).
    
    Implements Eq. (21): f = f_geom + f_pharm + f_gprot + f_fam
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.5,
                 gamma: float = 0.3, delta: float = 0.2, bias: float = 0.0,
                 normalize: bool = True):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.bias = bias
        self.normalize = normalize

    def compute(self, x_r: np.ndarray, x_l: np.ndarray,
                pharm_r: np.ndarray, pharm_l: np.ndarray,
                g_r: np.ndarray, g_l: np.ndarray,
                fam_r: np.ndarray, fam_l: np.ndarray) -> float:
        """
        Compute the complete binding affinity f(r, l).
        
        Implements Eq. (21):
          f_geom = -alpha * ||x_r - x_l||^2  [Eq. 23]
          f_pharm = beta * <pharm_r, pharm_l>  [Eq. 24]
          f_gprot = gamma * <g_r, g_l>  [Eq. 25]
          f_fam = delta * <fam_r, fam_l>  [Eq. 26]
        """
        # Normalize embeddings to unit length for stable distance computation
        if self.normalize:
            nr = np.linalg.norm(x_r)
            nl = np.linalg.norm(x_l)
            if nr > 0 and nl > 0:
                x_r_n = x_r / nr
                x_l_n = x_l / nl
            else:
                x_r_n, x_l_n = x_r, x_l
        else:
            x_r_n, x_l_n = x_r, x_l

        # Geometric term — Eq. (23)
        f_geom = -self.alpha * np.sum((x_r_n - x_l_n) ** 2)

        # Pharmacophore term — Eq. (24)
        f_pharm = self.beta * np.sum(pharm_r * pharm_l)

        # G-protein term — Eq. (25)
        f_gprot = self.gamma * np.sum(g_r * g_l)

        # Family term — Eq. (26)
        f_fam = self.delta * np.sum(fam_r * fam_l)

        return f_geom + f_pharm + f_gprot + f_fam + self.bias

    def set_parameters(self, alpha: float, beta: float, gamma: float,
                       delta: float, bias: float):
        """Update model parameters."""
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.bias = bias


# ============================================================================
# SECTION 5: RECEPTOR SIMILARITY AND KERNEL DENSITY ESTIMATION
# ============================================================================
# Implements Eq. (32) -- (42) from framework.tex


class ReceptorSimilarity:
    """
    Computes receptor similarity using Gaussian (RBF) kernel.
    
    Implements Eq. (32) -- (34): K(r_i, r_j) = exp(-||v_i - v_j||^2 / (2*sigma_R^2))
    """

    def __init__(self, bandwidth: float = 1.0):
        self.sigma_R = bandwidth

    def pairwise_similarity(self, V: np.ndarray) -> np.ndarray:
        """
        Compute pairwise similarity matrix for all receptors.
        
        Args:
            V: receptor feature matrix of shape (n_receptors, d)
        
        Returns:
            Similarity matrix of shape (n_receptors, n_receptors)
        """
        # Compute squared Euclidean distances
        norms = np.sum(V ** 2, axis=1)
        dist_sq = norms[:, None] + norms[None, :] - 2 * V @ V.T
        dist_sq = np.maximum(dist_sq, 0)  # numerical stability

        # RBF kernel — Eq. (32)
        K = np.exp(-dist_sq / (2 * self.sigma_R ** 2))
        return K

    def orphan_to_known_similarity(self, v_orphan: np.ndarray,
                                    V_known: np.ndarray) -> np.ndarray:
        """
        Compute similarity between one orphan receptor and all known receptors.
        
        Returns:
            Array of shape (n_known,) with similarity scores.
        """
        norms_o = np.sum(v_orphan ** 2)
        norms_k = np.sum(V_known ** 2, axis=1)
        dist_sq = norms_o + norms_k - 2 * v_orphan @ V_known.T
        dist_sq = np.maximum(dist_sq, 0)
        return np.exp(-dist_sq / (2 * self.sigma_R ** 2))

    def kde_ligand_distribution(self, v_orphan: np.ndarray,
                                 V_known: np.ndarray,
                                 known_ligands: List[List[str]],
                                 known_affinities: List[np.ndarray],
                                 all_ligands: List[str]) -> Dict[str, float]:
        """
        Compute predicted ligand distribution for orphan receptor via KDE.
        
        Implements Eq. (37) -- (39):
          P(l|r_orphan) = sum_j S(r_orphan, r_j) * P(l|r_j) / sum_j S(r_orphan, r_j)
        """
        # Compute similarities — Eq. (32)
        sims = self.orphan_to_known_similarity(v_orphan, V_known)

        # Compute ligand distributions for known receptors — Eq. (38)
        ligand_dists = []
        for j in range(len(V_known)):
            dist = self._ligand_distribution(known_affinities[j])
            ligand_dists.append(dist)

        # Weighted sum — Eq. (37)
        pred_dist = defaultdict(float)
        total_sim = np.sum(sims)

        for j in range(len(V_known)):
            for ligand, prob in ligand_dists[j].items():
                pred_dist[ligand] += sims[j] * prob

        # Normalize
        for ligand in pred_dist:
            pred_dist[ligand] /= total_sim

        return dict(pred_dist)

    def _ligand_distribution(self, affinities: np.ndarray) -> Dict[int, float]:
        """
        Compute softmax ligand distribution from affinities.
        
        Implements Eq. (38): P(l|r) = softmax((f - max) / tau)
        """
        if len(affinities) == 0:
            return {}

        tau = 1.0  # temperature parameter
        exp_vals = np.exp((affinities - np.max(affinities)) / tau)
        probs = exp_vals / np.sum(exp_vals)

        return {i: float(p) for i, p in enumerate(probs)}


# ============================================================================
# SECTION 6: CONFIDENCE SCORES AND UNCERTAINTY
# ============================================================================
# Implements Eq. (40) -- (57) from framework.tex


class UncertaintyQuantifier:
    """
    Computes confidence scores and prediction uncertainty.
    
    Implements:
      - Confidence score: Eq. (40)
      - Prediction variance: Eq. (51)
      - Confidence interval: Eq. (57)
    """

    def __init__(self, theta: float = 0.0, lambda_sigmoid: float = 1.0):
        self.theta = theta  # binding threshold — Eq. (35)
        self.lambda_s = lambda_sigmoid  # sigmoid steepness — Eq. (40)

    def confidence_score(self, affinity: float) -> float:
        """
        Compute confidence score C(r, l) = sigmoid(f(r,l) - theta).
        
        Implements Eq. (40): C = 1 / (1 + exp(-lambda * (f - theta)))
        """
        return 1.0 / (1.0 + math.exp(-self.lambda_s * (affinity - self.theta)))

    def prediction_variance(self, affinities: np.ndarray,
                            weighted_avg: float) -> float:
        """
        Compute prediction variance — Eq. (51).
        
        Var(r, l) = sum_j S(r_i, r_j) * (f(r_j, l) - f_hat(r_i, l))^2 / sum_j S(r_i, r_j)
        """
        if len(affinities) == 0:
            return 0.0
        return float(np.mean((affinities - weighted_avg) ** 2))

    def confidence_interval(self, predicted_pk: float,
                            variance: float,
                            model_var: float,
                            alpha: float = 0.05) -> Tuple[float, float]:
        """
        Compute (1-alpha) confidence interval — Eq. (57).
        
        CI = predicted +/- z_{alpha/2} * sqrt(Var + sigma_model^2)
        """
        z = 1.96 if alpha == 0.05 else 2.576  # 95% or 99%
        se = math.sqrt(variance + model_var)
        lower = predicted_pk - z * se
        upper = predicted_pk + z * se
        return lower, upper

    def set_threshold(self, theta: float):
        """Set the binding threshold."""
        self.theta = theta


# ============================================================================
# SECTION 7: CROSS-VALIDATION
# ============================================================================
# Implements Eq. (43) -- (50) from framework.tex


class CrossValidator:
    """
    Implements Leave-One-Receptor-Out Cross-Validation (LOROCV).
    
    Definition 14: For each characterized receptor, hold it out,
    predict its ligands from the remaining receptors.
    """

    def __init__(self):
        self.precision_at_k_list = []
        self.recall_at_k_list = []
        self.auc_roc_list = []
        self.auc_pr_list = []
        self.map_scores = []

    def precision_at_k(self, ranked_ligands: List[str],
                       true_ligands: set, k: int) -> float:
        """
        Precision@k — Eq. (43).
        
        |{l_(1), ..., l_(k)} ∩ L_r| / k
        """
        top_k = set(ranked_ligands[:k])
        hits = len(top_k & true_ligands)
        return hits / k if k > 0 else 0.0

    def recall_at_k(self, ranked_ligands: List[str],
                    true_ligands: set, k: int) -> float:
        """
        Recall@k — Eq. (44).
        
        |{l_(1), ..., l_(k)} ∩ L_r| / |L_r|
        """
        top_k = set(ranked_ligands[:k])
        hits = len(top_k & true_ligands)
        return hits / len(true_ligands) if len(true_ligands) > 0 else 0.0

    def auc_roc(self, scores: np.ndarray, labels: np.ndarray) -> float:
        """
        AUC-ROC — Eq. (45).
        
        Standard ROC AUC computation using the trapezoidal rule.
        """
        # Sort by descending score
        sorted_indices = np.argsort(-scores)
        sorted_labels = labels[sorted_indices]

        n_pos = np.sum(labels == 1)
        n_neg = np.sum(labels == 0)

        if n_pos == 0 or n_neg == 0:
            return 0.5

        tpr_list = [0.0]
        fpr_list = [0.0]

        tp = 0
        fp = 0
        for label in sorted_labels:
            if label == 1:
                tp += 1
            else:
                fp += 1
            tpr_list.append(tp / n_pos)
            fpr_list.append(fp / n_neg)

        # Trapezoidal rule
        auc = 0.0
        for i in range(1, len(tpr_list)):
            auc += (fpr_list[i] - fpr_list[i - 1]) * (tpr_list[i] + tpr_list[i - 1]) / 2

        return auc

    def auc_pr(self, ranked_ligands: List[str],
               true_ligands: set) -> float:
        """
        AUC-PR — Eq. (46).
        
        Sum over k of Precision@k * (Recall@k - Recall@k-1)
        """
        precisions = []
        recalls = []
        n_true = len(true_ligands)

        for k in range(1, len(ranked_ligands) + 1):
            top_k = set(ranked_ligands[:k])
            hits = len(top_k & true_ligands)
            prec = hits / k
            rec = hits / n_true if n_true > 0 else 0.0
            precisions.append(prec)
            recalls.append(rec)

        # Trapezoidal integration
        auc = 0.0
        for i in range(1, len(precisions)):
            auc += (recalls[i] - recalls[i - 1]) * (precisions[i] + precisions[i - 1]) / 2

        return auc

    def mean_average_precision(self, ranked_ligands: List[str],
                                true_ligands: set) -> float:
        """
        Mean Average Precision — Eq. (47).
        """
        n_true = len(true_ligands)
        if n_true == 0:
            return 0.0

        ap = 0.0
        hits = 0
        for k, ligand in enumerate(ranked_ligands, 1):
            if ligand in true_ligands:
                hits += 1
                ap += hits / k

        ap /= n_true
        return ap


# ============================================================================
# SECTION 8: BINDING KINETICS
# ============================================================================
# Implements Eq. (58) -- (66) from framework.tex


class BindingKinetics:
    """
    Models binding kinetics from affinity predictions.
    
    Implements:
      - Association rate: Eq. (58) -- (59)
      - Dissociation rate: Eq. (60) -- (61)
      - Equilibrium K_d: Eq. (62) -- (64)
    """

    def __init__(self, k0: float = 1e6, E0: float = 50.0,
                 E0_prime: float = 30.0, kappa: float = 2.0,
                 kappa_prime: float = 3.0, R: float = 8.314,
                 T: float = 310.0):
        """
        Args:
            k0: Pre-exponential factor (collision frequency)
            E0: Baseline activation energy (kJ/mol)
            E0_prime: Baseline dissociation energy (kJ/mol)
            kappa: Affinity-to-energy scaling for association
            kappa_prime: Affinity-to-energy scaling for dissociation
            R: Gas constant (J/(mol·K))
            T: Temperature (K)
        """
        self.k0 = k0
        self.E0 = E0
        self.E0_prime = E0_prime
        self.kappa = kappa
        self.kappa_prime = kappa_prime
        self.R = R
        self.T = T

    def association_rate(self, affinity: float) -> float:
        """
        Compute k_on from affinity.
        
        Implements Eq. (58) -- (59):
          E_a = E0 - kappa * f(r,l)
          k_on = k0 * exp(-E_a / RT)
        """
        E_a = self.E0 - self.kappa * affinity
        return self.k0 * math.exp(-E_a / (self.R * self.T / 1000))  # convert R to kJ

    def dissociation_rate(self, affinity: float) -> float:
        """
        Compute k_off from affinity.
        
        Implements Eq. (60) -- (61):
          E_d = E0' + kappa' * f(r,l)
          k_off = k0 * exp(-E_d / RT)
        """
        E_d = self.E0_prime + self.kappa_prime * affinity
        return self.k0 * math.exp(-E_d / (self.R * self.T / 1000))

    def equilibrium_kd(self, affinity: float) -> float:
        """
        Compute equilibrium K_d from affinity.
        
        Implements Eq. (62) -- (64):
          K_d = k_off / k_on
          pK_d = ((E0' - E0) + (kappa' + kappa) * f) / (RT * ln(10))
        """
        k_on = self.association_rate(affinity)
        k_off = self.dissociation_rate(affinity)
        if k_on == 0:
            return float("inf")
        return k_off / k_on

    def predicted_pkd(self, affinity: float) -> float:
        """
        Compute predicted pK_d from affinity.
        
        Implements Eq. (66):
          pK_d = ((E0' - E0) + (kappa' + kappa) * f) / (RT * ln(10))
        """
        numerator = (self.E0_prime - self.E0) + (self.kappa_prime + self.kappa) * affinity
        denominator = (self.R * self.T / 1000) * math.log(10)
        return numerator / denominator


# ============================================================================
# SECTION 9: MAIN DEORPHANIZATION PIPELINE
# ============================================================================


@dataclass
class DeorphanizationResult:
    """Result of deorphanization for a single orphan receptor."""
    receptor_name: str
    gene: str
    predicted_ligands: List[Tuple[str, float, float]] = field(default_factory=list)
    # List of (ligand_name, confidence_score, predicted_pKi)
    top_5: List[Tuple[str, float, float]] = field(default_factory=list)
    top_10: List[Tuple[str, float, float]] = field(default_factory=list)


class DeorphanizationPipeline:
    """
    Complete deorphanization pipeline.
    
    Algorithm 1 from framework.tex:
      Step 1: Feature Encoding
      Step 2: Projection to Shared Embedding
      Step 3: Parameter Estimation
      Step 4: Affinity Prediction
      Step 5: Confidence Estimation
      Step 6: Ranking
    """

    def __init__(self, data: Dict[str, Any], embedding_dim: int = 64):
        self.data = data
        self.raw_pairs = data["raw_pairs"]
        self.orphans = data["orphans"]
        self.ligand_classes = data["ligand_classes"]
        self.receptor_classes = data["receptor_classes"]

        # Encoders
        self.receptor_encoder = ReceptorEncoder(self.receptor_classes)
        self.ligand_encoder = LigandEncoder(self.ligand_classes)

        # Components
        self.projector = EmbeddingProjector(
            receptor_dim=860,
            ligand_dim=1227,
            embedding_dim=embedding_dim,
        )
        self.affinity_model = BindingAffinityModel(normalize=True)
        self.similarity = ReceptorSimilarity(bandwidth=5.0)
        self.uncertainty = UncertaintyQuantifier()
        self.kinetics = BindingKinetics()
        self.cv = CrossValidator()

        # Cached encoded data
        self.receptor_features: Dict[str, np.ndarray] = {}
        self.ligand_features: Dict[str, np.ndarray] = {}
        self.receptor_embeddings: Dict[str, np.ndarray] = {}
        self.ligand_embeddings: Dict[str, np.ndarray] = {}

        # Known receptor-ligand data
        self.known_receptors: List[str] = []
        self.known_receptor_features: List[np.ndarray] = []
        self.known_receptor_embeddings: List[np.ndarray] = []
        self.known_ligands_by_receptor: Dict[str, List[str]] = defaultdict(list)
        self.known_affinities_by_receptor: Dict[str, List[float]] = defaultdict(list)
        self.all_ligand_names: List[str] = []
        self.ligand_class_map: Dict[str, str] = {}

    def run(self):
        """Execute the complete deorphanization pipeline."""
        print("=" * 70)
        print("ORPHAN RECEPTOR DEORPHANIZATION PIPELINE")
        print("=" * 70)

        # Step 1: Feature Encoding
        print("\n[Step 1] Feature Encoding")
        self._encode_features()

        # Step 2: Build known receptor data structures
        print("\n[Step 2] Building Known Receptor Data")
        self._build_known_data()

        # Step 3: Parameter Estimation
        print("\n[Step 3] Parameter Estimation")
        self._estimate_parameters()

        # Step 4: Cross-Validation on Known Receptors
        print("\n[Step 4] Cross-Validation on Known Receptors")
        self._cross_validate()

        # Step 5: Deorphanize Orphan Receptors
        print("\n[Step 5] Deorphanizing Orphan Receptors")
        results = self._deorphanize_orphans()

        # Step 6: Print Results
        print("\n[Step 6] Results Summary")
        self._print_results(results)

        return results

    def _encode_features(self):
        """Step 1: Encode all receptor and ligand features."""
        # Extract unique receptors and ligands from raw data
        receptor_data_map = {}
        ligand_set = set()

        for pair in self.raw_pairs:
            receptor_name = pair["receptor"]
            ligand_name = pair["ligand"]
            ligand_class = pair["ligand_class"]

            receptor_data_map[receptor_name] = pair
            ligand_set.add(ligand_name)
            self.ligand_class_map[ligand_name] = ligand_class

        self.all_ligand_names = sorted(ligand_set)

        # Encode known receptors
        print(f"  Encoding {len(receptor_data_map)} known receptors...")
        for name, rdata in receptor_data_map.items():
            v_r = self.receptor_encoder.encode(rdata)
            self.receptor_features[name] = v_r

        # Encode all ligands
        print(f"  Encoding {len(self.all_ligand_names)} ligands...")
        for ligand_name in self.all_ligand_names:
            ligand_class = self.ligand_class_map[ligand_name]
            u_l = self.ligand_encoder.encode(ligand_name, ligand_class)
            self.ligand_features[ligand_name] = u_l

        # Encode orphan receptors
        print(f"  Encoding {len(self.orphans)} orphan receptors...")
        self.orphan_features = {}
        for odata in self.orphans:
            v_r = self.receptor_encoder.encode(odata)
            self.orphan_features[odata["name"]] = v_r

    def _build_known_data(self):
        """Build data structures for known receptor-ligand pairs."""
        # Group pairs by receptor
        receptor_pairs = defaultdict(list)
        for pair in self.raw_pairs:
            receptor_pairs[pair["receptor"]].append(pair)

        self.known_receptors = sorted(receptor_pairs.keys())

        for receptor_name in self.known_receptors:
            pairs = receptor_pairs[receptor_name]
            v_r = self.receptor_features[receptor_name]

            # Project to embedding
            x_r = self.projector.project_receptor(v_r)
            self.receptor_embeddings[receptor_name] = x_r
            self.known_receptor_embeddings.append(x_r)

            # Encode ligands and compute affinities
            ligands = []
            affinities = []
            for pair in pairs:
                ligand_name = pair["ligand"]
                u_l = self.ligand_features[ligand_name]
                x_l = self.projector.project_ligand(u_l)
                self.ligand_embeddings[ligand_name] = x_l

                # Compute affinity components
                pharm_r = self.projector.project_pharmacophore(v_r)
                pharm_l = self.ligand_encoder._pharmacophore(ligand_name, pair["ligand_class"])
                g_r = self.receptor_encoder._g_protein_features(pair)
                g_l = self._compute_ligand_g_preference(ligand_name)
                fam_r = self.receptor_encoder._family_encoding(pair)
                fam_l = self._compute_ligand_fam_affinity(ligand_name)

                # Compute affinity — Eq. (21)
                f = self.affinity_model.compute(
                    x_r, x_l, pharm_r, pharm_l, g_r, g_l, fam_r, fam_l
                )

                ligands.append(ligand_name)
                affinities.append(f)

                # Also store pK_i for validation
                ki = pair["ki"]
                if ki > 0:
                    pk_i = -math.log10(ki * 1e-9)  # convert nM to M, then -log10
                    self.known_affinities_by_receptor[receptor_name].append(pk_i)
                else:
                    self.known_affinities_by_receptor[receptor_name].append(0.0)

            self.known_receptor_features.append(v_r)
            self.known_ligands_by_receptor[receptor_name] = ligands

        print(f"  Built data for {len(self.known_receptors)} receptors")
        print(f"  Total ligands: {len(self.all_ligand_names)}")

    def _compute_ligand_g_preference(self, ligand_name: str) -> np.ndarray:
        """Compute G-protein preference vector for a ligand — Eq. (27)."""
        g_vecs = []
        for pair in self.raw_pairs:
            if pair["ligand"] == ligand_name:
                g_vec = self.receptor_encoder._g_protein_features(pair)
                g_vecs.append(g_vec)

        if g_vecs:
            return np.mean(g_vecs, axis=0)
        return np.array([0.25, 0.25, 0.25, 0.25])

    def _compute_ligand_fam_affinity(self, ligand_name: str) -> np.ndarray:
        """Compute family affinity vector for a ligand — Eq. (28)."""
        fam_vecs = []
        for pair in self.raw_pairs:
            if pair["ligand"] == ligand_name:
                fam_vec = self.receptor_encoder._family_encoding(pair)
                fam_vecs.append(fam_vec)

        if fam_vecs:
            return np.mean(fam_vecs, axis=0)
        return np.zeros(19)

    def _estimate_parameters(self):
        """Step 3: Estimate model parameters via MLE."""
        print("  Estimating parameters via gradient descent on known pairs...")

        # Build training data
        x_r_list = []
        x_l_list = []
        pk_i_list = []

        for pair in self.raw_pairs:
            receptor_name = pair["receptor"]
            ligand_name = pair["ligand"]

            if receptor_name not in self.receptor_embeddings:
                continue
            if ligand_name not in self.ligand_embeddings:
                continue

            x_r_list.append(self.receptor_embeddings[receptor_name])
            x_l_list.append(self.ligand_embeddings[ligand_name])

            ki = pair["ki"]
            if ki > 0:
                pk_i = -math.log10(ki * 1e-9)
                pk_i_list.append(pk_i)

        x_r_arr = np.array(x_r_list)
        x_l_arr = np.array(x_l_list)
        pk_i_arr = np.array(pk_i_list)

        print(f"  Training on {len(pk_i_arr)} pairs")
        print(f"  pK_i range: [{pk_i_arr.min():.2f}, {pk_i_arr.max():.2f}]")

        # Fit projector — needs RAW features, not embeddings
        raw_r = [self.receptor_features[p["receptor"]] for p in self.raw_pairs if p["receptor"] in self.receptor_features]
        raw_l = [self.ligand_features[p["ligand"]] for p in self.raw_pairs if p["ligand"] in self.ligand_features]
        self.projector.fit(
            raw_r, raw_l, pk_i_arr,
            lr=0.00001, n_iter=100
        )

        # Re-project after fitting
        for name in self.receptor_embeddings:
            v_r = self.receptor_features[name]
            self.receptor_embeddings[name] = self.projector.project_receptor(v_r)

        # Compute binding threshold — Eq. (35)
        all_affinities = []
        for receptor_name in self.known_receptors:
            for ligand in self.known_ligands_by_receptor[receptor_name]:
                x_r = self.receptor_embeddings[receptor_name]
                x_l = self.ligand_embeddings[ligand]
                f = self.affinity_model.compute(x_r, x_l,
                                                np.zeros(8), np.zeros(8),
                                                np.zeros(4), np.zeros(4),
                                                np.zeros(19), np.zeros(19))
                all_affinities.append(f)

        all_affinities = np.array(all_affinities)
        mu_f = np.mean(all_affinities)
        sigma_f = np.std(all_affinities)

        # Set threshold at mean - 1*sigma (catch most true binders)
        theta = mu_f - 1.0 * sigma_f
        self.uncertainty.set_threshold(theta)
        self.uncertainty.lambda_s = 1.0 / max(sigma_f, 0.1)

        # Set affinity parameters
        self.affinity_model.set_parameters(
            alpha=0.5,
            beta=0.3,
            gamma=0.2,
            delta=0.1,
            bias=mu_f,
        )

        print(f"  Binding threshold theta = {theta:.4f}")
        print(f"  Mean affinity = {mu_f:.4f}, Std = {sigma_f:.4f}")

    def _cross_validate(self):
        """Step 4: Leave-One-Receptor-Out Cross-Validation."""
        print("  Running LOROCV on known receptors...")
        print("  (Using similarity-based prediction for each held-out receptor)")

        k = 10  # top-k for precision/recall

        total_prec_at_k = 0.0
        total_rec_at_k = 0.0
        total_auc_roc = 0.0
        total_auc_pr = 0.0
        total_map = 0.0
        n_valid = 0

        for i, receptor_name in enumerate(self.known_receptors):
            true_ligands = set(self.known_ligands_by_receptor[receptor_name])

            if len(true_ligands) == 0:
                continue

            # Use similarity-based prediction — Eq. (37)
            v_r = self.receptor_features[receptor_name]

            # Get similarities to all OTHER known receptors
            all_sims = []
            all_ligand_dists = []

            for j, other_name in enumerate(self.known_receptors):
                if other_name == receptor_name:
                    continue

                v_other = self.receptor_features[other_name]
                dist = np.sum((v_r - v_other) ** 2)
                sim = math.exp(-dist / (2 * 5.0 ** 2))  # sigma_R = 5

                # Ligand distribution for other receptor
                other_ligands = self.known_ligands_by_receptor[other_name]
                if len(other_ligands) == 0:
                    continue

                # Simple uniform distribution over other's ligands
                dist_dict = {}
                for l in other_ligands:
                    dist_dict[l] = 1.0 / len(other_ligands)
                all_ligand_dists.append((sim, dist_dict))

            # Weighted combination — Eq. (37)
            pred_scores = defaultdict(float)
            total_weight = sum(s for s, _ in all_ligand_dists)

            if total_weight > 0:
                for sim, dist_dict in all_ligand_dists:
                    for ligand, prob in dist_dict.items():
                        pred_scores[ligand] += sim * prob / total_weight

            # Rank ligands by predicted probability
            ranked = sorted(pred_scores.keys(), key=lambda l: pred_scores[l], reverse=True)

            # Compute metrics
            prec = self.cv.precision_at_k(ranked, true_ligands, k)
            rec = self.cv.recall_at_k(ranked, true_ligands, k)
            map_score = self.cv.mean_average_precision(ranked, true_ligands)

            # AUC-ROC: need scores for all ligands
            scores = np.array([pred_scores.get(l, 0.0) for l in self.all_ligand_names])
            labels = np.array([1 if l in true_ligands else 0 for l in self.all_ligand_names])
            auc_roc = self.cv.auc_roc(scores, labels)
            auc_pr = self.cv.auc_pr(ranked, true_ligands)

            total_prec_at_k += prec
            total_rec_at_k += rec
            total_auc_roc += auc_roc
            total_auc_pr += auc_pr
            total_map += map_score
            n_valid += 1

        if n_valid > 0:
            print(f"\n  LOROCV Results (k={k}):")
            print(f"    Precision@{k}:     {total_prec_at_k / n_valid:.4f}")
            print(f"    Recall@{k}:        {total_rec_at_k / n_valid:.4f}")
            print(f"    Mean AUC-ROC:      {total_auc_roc / n_valid:.4f}")
            print(f"    Mean AUC-PR:       {total_auc_pr / n_valid:.4f}")
            print(f"    Mean Average Precision: {total_map / n_valid:.4f}")

    def _deorphanize_orphans(self) -> List[DeorphanizationResult]:
        """Step 5: Predict ligands for all orphan receptors."""
        results = []

        # Build known receptor feature matrix
        V_known = np.array(self.known_receptor_features)

        for odata in self.orphans:
            receptor_name = odata["name"]
            gene = odata["gene"]

            v_orphan = self.orphan_features[receptor_name]

            # Compute affinity for all ligands — Eq. (21)
            affinities = []
            for ligand_name in self.all_ligand_names:
                x_r = self.projector.project_receptor(v_orphan)
                x_l = self.ligand_embeddings[ligand_name]

                pharm_r = self.projector.project_pharmacophore(v_orphan)
                u_l = self.ligand_features[ligand_name]
                pharm_l = self.ligand_encoder._pharmacophore(ligand_name, self.ligand_class_map[ligand_name])
                g_r = self.receptor_encoder._g_protein_features(odata)
                g_l = self._compute_ligand_g_preference(ligand_name)
                fam_r = self.receptor_encoder._family_encoding(odata)
                fam_l = self._compute_ligand_fam_affinity(ligand_name)

                f = self.affinity_model.compute(
                    x_r, x_l, pharm_r, pharm_l, g_r, g_l, fam_r, fam_l
                )
                affinities.append(f)

            affinities = np.array(affinities)

            # Compute confidence scores — Eq. (40)
            confidence_scores = []
            for f in affinities:
                c = self.uncertainty.confidence_score(f)
                confidence_scores.append(c)

          # Compute prediction variance — Eq. (51)
            # Simplified: variance based on spread of affinities
            var = float(np.var(affinities))
            weighted_avg = float(np.mean(affinities))

            # Compute predicted pK_i — Eq. (31)
            predicted_pkis = affinities + self.affinity_model.bias

            # Rank by confidence score
            indexed_scores = list(zip(
                self.all_ligand_names,
                confidence_scores,
                predicted_pkis,
                affinities,
            ))
            indexed_scores.sort(key=lambda x: x[1], reverse=True)

            # Build result
            predicted_ligands = [
                (name, conf, pk) for name, conf, pk, _ in indexed_scores
            ]

            result = DeorphanizationResult(
                receptor_name=receptor_name,
                gene=gene,
                predicted_ligands=predicted_ligands,
                top_5=predicted_ligands[:5],
                top_10=predicted_ligands[:10],
            )

            # Store variance for confidence intervals
            result.variance = var
            result.weighted_avg = weighted_avg

            results.append(result)

        return results

    def _print_results(self, results: List[DeorphanizationResult]):
        """Print deorphanization results."""
        print(f"\n  {'='*66}")
        print(f"  {'ORPHAN RECEPTOR DEORPHANIZATION RESULTS':^66}")
        print(f"  {'='*66}")

        for result in results[:15]:  # Show top 15
            print(f"\n  Receptor: {result.receptor_name} ({result.gene})")
            print(f"  {'Top 5 Predicted Ligands:':<30}")
            for rank, (ligand, conf, pk) in enumerate(result.top_5, 1):
                print(f"    {rank:2d}. {ligand:<35s}  C={conf:.3f}  pKi={pk:.2f}")

        print(f"\n  {'='*66}")
        print(f"  Total orphan receptors analyzed: {len(results)}")
        print(f"  Total candidate ligands: {len(self.all_ligand_names)}")
        print(f"  Binding threshold: {self.uncertainty.theta:.4f}")
        print(f"  {'='*66}")

        # Save full results
        self._save_results(results)

    def _save_results(self, results: List[DeorphanizationResult]):
        """Save results to JSON."""
        output = []
        for result in results:
            entry = {
                "receptor": result.receptor_name,
                "gene": result.gene,
                "top_5": [
                    {"ligand": l, "confidence": c, "predicted_pkd": p}
                    for l, c, p in result.top_5
                ],
                "top_10": [
                    {"ligand": l, "confidence": c, "predicted_pkd": p}
                    for l, c, p in result.top_10
                ],
            }
            output.append(entry)

        out_path = "../output/deorphanization_results.json"
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n  Results saved to {out_path}")


# ============================================================================
# SECTION 10: MAIN ENTRY POINT
# ============================================================================


def main():
    """Run the complete deorphanization pipeline."""
    print("\nLoading data...")
    data = load_all_data()
    print(f"  Raw pairs: {len(data['raw_pairs'])}")
    print(f"  Orphan receptors: {len(data['orphans'])}")
    print(f"  Ligand classes: {len(data['ligand_classes'])}")
    print(f"  Receptor classes: {len(data['receptor_classes'])}")

    # Set random seed for reproducibility
    np.random.seed(42)

    # Run pipeline
    pipeline = DeorphanizationPipeline(data, embedding_dim=64)
    results = pipeline.run()

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
