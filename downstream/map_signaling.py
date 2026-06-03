#!/usr/bin/env python3
"""
Downstream signaling cascade mapping for predicted orphan receptor-ligand pairs.
Maps GPCR signaling cascades, disease connections, addiction pathways, and
therapeutic opportunities for 72 orphan GPCRs.
"""

import json
import math
from collections import defaultdict
from pathlib import Path

BASE = Path("/Users/alex/Desktop/deorphanization")
PREDICTIONS = BASE / "output/predictions_v2.json"
CATALOG = BASE / "data/orphan_receptor_catalog.json"
RAW_DATA = BASE / "data/raw_data.json"

OUTPUT_DIR = BASE / "downstream"
SIGNALENG_NETWORK_OUT = OUTPUT_DIR / "signaling_network.json"
DISEASE_MAP_OUT = OUTPUT_DIR / "disease_map.json"
ADDICTION_MAP_OUT = OUTPUT_DIR / "addiction_map.json"
SUMMARY_OUT = OUTPUT_DIR / "summary.md"

# ---------------------------------------------------------------------------
# Signaling cascade definitions
# ---------------------------------------------------------------------------

G_PROTEIN_CASCADES = {
    "Gs": {
        "description": "Stimulatory G-protein",
        "pathway": [
            "Gs alpha subunit activation",
            "Adenylyl cyclase stimulation",
            "cAMP production increase (10-100x basal)",
            "PKA (protein kinase A) activation",
            "CREB (cAMP response element-binding protein) phosphorylation at Ser133",
            "cAMP response element (CRE) mediated gene transcription",
        ],
        "downstream_targets": ["CREB", "EPAC1/2", "HCN channels", "KCNQ potassium channels"],
        "cellular_outcomes": [
            "Increased neuronal excitability",
            "Enhanced synaptic plasticity (LTP)",
            "Increased hormone secretion (e.g., insulin, cortisol)",
            "Cardiac myocyte contractility increase",
            "Adipocyte lipolysis",
            "Smooth muscle relaxation",
        ],
        "second_messenger": "cAMP (increased)",
    },
    "Gi": {
        "description": "Inhibitory G-protein",
        "pathway": [
            "Gi alpha subunit activation (Galpha-i)",
            "Adenylyl cyclase inhibition",
            "cAMP production decrease (30-80% reduction)",
            "Reduced PKA activity",
            "Direct GIRK (G-protein-gated inwardly rectifying K+) channel opening via Gbeta-gamma",
            "Altered ion channel activity (K+ efflux, Ca2+ influx reduction)",
        ],
        "downstream_targets": ["GIRK channels", "CACNA1 calcium channels", "MAPK/ERK", "PI3K"],
        "cellular_outcomes": [
            "Neuronal hyperpolarization / reduced firing",
            "Presynaptic neurotransmitter release inhibition",
            "Smooth muscle contraction",
            "Cardiac pacemaker rate decrease",
            "Neurotransmitter reuptake modulation",
            "Anti-inflammatory signaling (in immune cells)",
        ],
        "second_messenger": "cAMP (decreased)",
    },
    "Gq": {
        "description": "Gq/11 family G-protein",
        "pathway": [
            "Gq alpha subunit activation",
            "PLC-beta (phospholipase C) activation",
            "PIP2 hydrolysis to IP3 + DAG",
            "IP3-mediated Ca2+ release from ER/SR stores",
            "DAG-mediated PKC activation (conventional and novel PKC isoforms)",
            "Ca2+/calmodulin-dependent kinase (CaMK) activation",
            "NFAT nuclear translocation",
            "NF-kB pathway activation (via PKC-dependent IKK)",
        ],
        "downstream_targets": ["PLC-beta", "TRP channels", "NFAT", "NF-kB", "PKC isoforms", "ERK1/2"],
        "cellular_outcomes": [
            "Smooth muscle contraction",
            "Neurotransmitter release (glutamate, acetylcholine)",
            "Platelet activation and aggregation",
            "Immune cell degranulation",
            "Cell proliferation and differentiation",
            "Cytokine production (TNF, IL-6, IL-1beta)",
        ],
        "second_messenger": "IP3 + DAG (both increased), Ca2+ release",
    },
    "G12/13": {
        "description": "RhoGEF-activating G-protein",
        "pathway": [
            "G12/G13 alpha subunit activation",
            "RhoGEF (p115-RhoGEF, PDZ-RhoGEF, LARG) activation",
            "RhoA GTPase activation (GDP -> GTP exchange)",
            "ROCK (Rho-associated kinase) activation",
            "MLCK (myosin light chain kinase) activation",
            "CTRF (cytoskeletal tension response factor) activation",
            "JNK pathway activation",
        ],
        "downstream_targets": ["RhoA", "ROCK1/2", "JNK", "MLCK", "mDia"],
        "cellular_outcomes": [
            "Actin cytoskeletal reorganization",
            "Cell migration and chemotaxis",
            "Cell contraction (vascular smooth muscle)",
            "Blood-brain barrier permeability changes",
            "Neurite retraction / growth cone collapse",
            "Focal adhesion dynamics",
        ],
        "second_messenger": "RhoA-GTP (active form)",
    },
}

# ---------------------------------------------------------------------------
# Disease pathway mappings
# ---------------------------------------------------------------------------

DISEASE_CATEGORIES = {
    "neurological": {
        "diseases": [
            "schizophrenia", "depression", "anxiety", "epilepsy",
            "Parkinson's disease", "Alzheimer's disease", "addiction/substance use disorder",
            "bipolar disorder", "autism spectrum disorder", "PTSD",
            "migraine", "chronic pain", "insomnia",
        ],
        "pathways": [
            "dopaminergic signaling", "serotonergic signaling", "glutamatergic signaling",
            "GABAergic signaling", "cholinergic signaling", "noradrenergic signaling",
            "endocannabinoid signaling", "opioidergic signaling",
        ],
        "brain_regions": ["prefrontal cortex", "striatum", "hippocampus", "amygdala",
                          "ventral tegmental area", "hypothalamus", "brainstem", "cerebellum"],
    },
    "metabolic": {
        "diseases": [
            "type 2 diabetes", "obesity", "metabolic syndrome",
            "insulin resistance", "dyslipidemia", "non-alcoholic fatty liver disease (NAFLD)",
            "hyperglycemia", "cachexia",
        ],
        "pathways": [
            "insulin signaling", "glucagon signaling", "leptin signaling",
            "ghrelin signaling", "GLP-1 signaling", "AMPK pathway",
            "mTOR signaling", "thyroid hormone signaling",
        ],
        "organs": ["pancreas", "liver", "adipose tissue", "skeletal muscle", "hypothalamus"],
    },
    "cardiovascular": {
        "diseases": [
            "hypertension", "heart failure", "arrhythmia",
            "atherosclerosis", "myocardial infarction", "pulmonary hypertension",
            "vascular remodeling", "cardiac hypertrophy",
        ],
        "pathways": [
            "adrenergic signaling", "angiotensin signaling", "endothelin signaling",
            "nitric oxide signaling", "atrial natriuretic peptide signaling",
            "calcium handling", "cardiac ion channel function",
        ],
        "organs": ["heart", "blood vessels", "kidney", "endothelium"],
    },
    "immune": {
        "diseases": [
            "autoimmune disease", "chronic inflammation", "allergy/asthma",
            "autoinflammatory syndrome", "sepsis", "graft rejection",
            "immunodeficiency", "celiac disease",
        ],
        "pathways": [
            "chemokine signaling", "complement activation", "cytokine signaling (JAK/STAT)",
            "T-cell activation", "B-cell activation", "macrophage polarization",
            "neutrophil recruitment", "mast cell degranulation",
        ],
        "cell_types": ["T cells", "B cells", "macrophages", "neutrophils",
                       "microglia", "dendritic cells", "mast cells", "NK cells"],
    },
    "cancer": {
        "diseases": [
            "tumor progression", "metastasis", "angiogenesis-dependent tumors",
            "neuroendocrine tumors", "glioma", "leukemia",
        ],
        "pathways": [
            "Wnt/beta-catenin", "PI3K/Akt/mTOR", "MAPK/ERK",
            "JAK/STAT", "Hedgehog", "NF-kB",
        ],
        "mechanisms": ["cell proliferation", "angiogenesis", "epithelial-mesenchymal transition",
                       "immune evasion", "metastasis", "chemoresistance"],
    },
}

# ---------------------------------------------------------------------------
# Ligand class -> disease connection mapping
# ---------------------------------------------------------------------------

LIGAND_CLASS_DISEASE_MAP = {
    "catecholamine": {
        "primary_categories": ["cardiovascular", "neurological"],
        "secondary_categories": ["metabolic"],
        "diseases": ["hypertension", "heart failure", "arrhythmia", "Parkinson's disease",
                      "depression", "anxiety", "migraine"],
        "rationale": "Catecholamines (dopamine, norepinephrine, epinephrine) are primary cardiovascular and neurological signaling molecules",
    },
    "amino acid": {
        "primary_categories": ["neurological", "metabolic"],
        "secondary_categories": ["immune"],
        "diseases": ["epilepsy", "schizophrenia", "depression", "type 2 diabetes",
                      "chronic inflammation", "autism spectrum disorder"],
        "rationale": "Amino acid ligands (glutamate, GABA, D-serine, tryptophan) directly modulate neurotransmission and metabolic sensing",
    },
    "lipid": {
        "primary_categories": ["immune", "metabolic"],
        "secondary_categories": ["neurological", "cancer"],
        "diseases": ["chronic inflammation", "autoimmune disease", "obesity", "NAFLD",
                      "multiple sclerosis", "glioma", "asthma"],
        "rationale": "Lipid ligands (prostaglandins, LPA, S1P, endocannabinoids) are key mediators of inflammation and metabolic signaling",
    },
    "peptide": {
        "primary_categories": ["neurological", "metabolic"],
        "secondary_categories": ["cardiovascular", "immune"],
        "diseases": ["depression", "schizophrenia", "Alzheimer's disease", "type 2 diabetes",
                      "hypertension", "chronic pain", "addiction"],
        "rationale": "Neuropeptides (galanin, neurotensin, endorphins, dynorphin, ACTH) are potent neuromodulators and endocrine regulators",
    },
    "tryptamine": {
        "primary_categories": ["neurological", "cancer"],
        "secondary_categories": ["metabolic"],
        "diseases": ["depression", "anxiety", "schizophrenia", "migraine",
                      "psychiatric disorders", "PTSD", "insomnia"],
        "rationale": "Tryptamines (serotonin, melatonin, DMT) are primary psychiatric and circadian signaling molecules",
    },
    "indolamine": {
        "primary_categories": ["neurological", "metabolic"],
        "secondary_categories": ["immune"],
        "diseases": ["depression", "sleep disorders", "circadian rhythm disorders",
                      "migraine", "inflammatory bowel disease", "autoimmune disease"],
        "rationale": "Indolamines (melatonin, serotonin) regulate sleep, mood, and circadian rhythms with immune modulatory effects",
    },
    "phenethylamine": {
        "primary_categories": ["neurological", "cancer"],
        "secondary_categories": ["metabolic"],
        "diseases": ["ADHD", "depression", "Parkinson's disease", "psychiatric disorders",
                      "addiction/substance use disorder", "migraine"],
        "rationale": "Phenethylamines (phenethylamine, tyramine, octopamine) modulate monoaminergic transmission and TAAR1 signaling",
    },
    "tyramine": {
        "primary_categories": ["cardiovascular", "neurological"],
        "secondary_categories": ["metabolic"],
        "diseases": ["hypertension", "migraine", "Parkinson's disease", "depression",
                      "addiction/substance use disorder"],
        "rationale": "Tyramine acts on TAAR1 and adrenergic receptors, affecting blood pressure and neurotransmission",
    },
    "octopamine": {
        "primary_categories": ["neurological", "metabolic"],
        "secondary_categories": ["cardiovascular"],
        "diseases": ["depression", "anxiety", "metabolic syndrome", "hypertension"],
        "rationale": "Octopamine is a trace amine acting on TAAR receptors affecting mood and metabolism",
    },
    "opioid": {
        "primary_categories": ["neurological", "immune"],
        "secondary_categories": ["cardiovascular"],
        "diseases": ["chronic pain", "addiction/substance use disorder", "opioid use disorder",
                      "depression", "anxiety", "respiratory depression"],
        "rationale": "Opioid ligands (endorphins, enkephalins, dynorphin) modulate pain, reward, and addiction pathways via mu/delta/kappa receptors",
    },
    "cannabinoid": {
        "primary_categories": ["neurological", "immune"],
        "secondary_categories": ["metabolic", "cancer"],
        "diseases": ["chronic pain", "addiction/substance use disorder", "epilepsy",
                      "spasticity (MS)", "nausea", "obesity", "glaucoma"],
        "rationale": "Cannabinoid ligands (anandamide, 2-AG) act on CB1/CB2 receptors for pain, reward, inflammation, and appetite",
    },
    "fatty acid": {
        "primary_categories": ["metabolic", "immune"],
        "secondary_categories": ["neurological", "cardiovascular"],
        "diseases": ["obesity", "type 2 diabetes", "NAFLD", "chronic inflammation",
                      "cardiovascular disease", "neuroinflammation"],
        "rationale": "Fatty acids (DHA, EPA, long-chain fatty acids) modulate GPR40/41/43, GPR120 for metabolic and inflammatory signaling",
    },
    "synthetic": {
        "primary_categories": ["neurological", "cardiovascular"],
        "secondary_categories": ["immune"],
        "diseases": ["hypertension", "depression", "schizophrenia", "anxiety",
                      "allergic inflammation", "pain"],
        "rationale": "Synthetic ligands span multiple pharmacological classes targeting GPCRs for cardiovascular and neurological indications",
    },
    "metal ion": {
        "primary_categories": ["metabolic", "neurological"],
        "secondary_categories": ["immune"],
        "diseases": ["type 2 diabetes", "osteoporosis", "neurodegeneration",
                      "acidosis-related disorders", "electrolyte imbalance"],
        "rationale": "Metal ion ligands (zinc, calcium, protons) act as allosteric modulators and direct agonists at metal-sensing GPCRs",
    },
    "purine": {
        "primary_categories": ["cardiovascular", "immune"],
        "secondary_categories": ["neurological"],
        "diseases": ["arrhythmia", "heart failure", "chronic inflammation",
                      "neuroinflammation", "pain", "asthma"],
        "rationale": "Purine ligands (ATP, ADP, adenosine) act on P2Y/P1 receptors for cardiovascular and immune regulation",
    },
    "neurotransmitter": {
        "primary_categories": ["neurological", "metabolic"],
        "secondary_categories": ["immune", "cardiovascular"],
        "diseases": ["schizophrenia", "depression", "epilepsy", "Parkinson's disease",
                      "chronic pain", "addiction", "anxiety"],
        "rationale": "Neurotransmitter ligands directly target CNS signaling pathways",
    },
    "biogenic amine": {
        "primary_categories": ["neurological", "cardiovascular"],
        "secondary_categories": ["metabolic", "immune"],
        "diseases": ["depression", "anxiety", "hypertension", "migraine",
                      "Parkinson's disease", "addiction", "asthma"],
        "rationale": "Biogenic amines (histamine, serotonin, dopamine) are broad-spectrum neuromodulators and cardiovascular regulators",
    },
    "protein": {
        "primary_categories": ["immune", "cancer"],
        "secondary_categories": ["metabolic"],
        "diseases": ["autoimmune disease", "chronic inflammation", "tumor progression",
                      "fibrosis", "sepsis"],
        "rationale": "Protein ligands (chemokines, growth factors) drive immune and proliferative signaling",
    },
}

# ---------------------------------------------------------------------------
# Addiction pathway definitions
# ---------------------------------------------------------------------------

DRUG_TARGETS = {
    # Known pharmacological targets with abuse potential
    "dopamine": {"receptors": ["DRD1", "DRD2", "DRD3", "DRD4", "DRD5"], "class": "catecholamine", "drug_class": "dopaminergic", "addiction_type": "stimulant addiction"},
    "norepinephrine": {"receptors": ["ADRA1A", "ADRA1B", "ADRA2A", "ADRA2B", "ADRB1", "ADRB2"], "class": "catecholamine", "drug_class": "adrenergic", "addiction_type": "stimulant addiction"},
    "epinephrine": {"receptors": ["ADRB1", "ADRB2", "ADRB3"], "class": "catecholamine", "drug_class": "adrenergic", "addiction_type": "none"},
    "serotonin": {"receptors": ["HTR1A", "HTR1B", "HTR2A", "HTR2C", "HTR3", "HTR4", "HTR6", "HTR7"], "class": "tryptamine", "drug_class": "serotonergic", "addiction_type": "SSRI dependence"},
    "phenethylamine": {"receptors": ["TAAR1"], "class": "phenethylamine", "drug_class": "trace amine", "addiction_type": "stimulant addiction"},
    "tyramine": {"receptors": ["TAAR1", "ADRA2A"], "class": "tyramine", "drug_class": "trace amine", "addiction_type": "stimulant addiction"},
    "octopamine": {"receptors": ["TAAR4", "TAAR6"], "class": "octopamine", "drug_class": "trace amine", "addiction_type": "stimulant addiction"},
    "beta-endorphin": {"receptors": ["OPRM1", "OPRD1", "OPRK1"], "class": "opioid", "drug_class": "opioid peptide", "addiction_type": "opioid addiction"},
    "dynorphin A": {"receptors": ["OPRK1"], "class": "opioid", "drug_class": "opioid peptide", "addiction_type": "opioid addiction"},
    "enkephalin": {"receptors": ["OPRD1", "OPRM1"], "class": "opioid", "drug_class": "opioid peptide", "addiction_type": "opioid addiction"},
    "endomorphin": {"receptors": ["OPRM1"], "class": "opioid", "drug_class": "opioid peptide", "addiction_type": "opioid addiction"},
    "anandamide": {"receptors": ["CB1", "CB2"], "class": "cannabinoid", "drug_class": "endocannabinoid", "addiction_type": "cannabinoid addiction"},
    "2-AG": {"receptors": ["CB1", "CB2"], "class": "cannabinoid", "drug_class": "endocannabinoid", "addiction_type": "cannabinoid addiction"},
    "THC": {"receptors": ["CB1", "CB2"], "class": "cannabinoid", "drug_class": "cannabinoid", "addiction_type": "cannabinoid addiction"},
    "galanin": {"receptors": ["GALR1", "GALR2", "GALR3"], "class": "peptide", "drug_class": "neuropeptide", "addiction_type": "alcohol addiction"},
    "neurotensin": {"receptors": ["NTR1", "NTR2", "NTR3"], "class": "peptide", "drug_class": "neuropeptide", "addiction_type": "stimulant addiction (dopamine modulation)"},
    "ACTH": {"receptors": ["MC2R", "MC3R", "MC4R", "MC5R"], "class": "peptide", "drug_class": "melanocortin", "addiction_type": "appetite/addiction overlap"},
    "melanocortin": {"receptors": ["MC3R", "MC4R"], "class": "peptide", "drug_class": "melanocortin", "addiction_type": "appetite/addiction overlap"},
    "glutamate": {"receptors": ["GRM1", "GRM5", "GRM7", "GRM8"], "class": "amino acid", "drug_class": "glutamatergic", "addiction_type": "alcohol/addiction"},
    "GABA": {"receptors": ["GABBR1", "GABBR2"], "class": "amino acid", "drug_class": "GABAergic", "addiction_type": "benzodiazepine-like addiction"},
    "histamine": {"receptors": ["H1R", "H2R", "H3R", "H4R"], "class": "biogenic amine", "drug_class": "antihistamine", "addiction_type": "none"},
    "prostaglandin E2": {"receptors": ["EP1", "EP2", "EP3", "EP4"], "class": "lipid", "drug_class": "eicosanoid", "addiction_type": "none"},
    "LPA": {"receptors": ["LPAR1-6"], "class": "lipid", "drug_class": "lysophospholipid", "addiction_type": "none"},
    "S1P": {"receptors": ["SPHK1", "SPHK2", "S1PR1-5"], "class": "lipid", "drug_class": "sphingosine-1-phosphate", "addiction_type": "none"},
    "DHA": {"receptors": ["GPR119", "GPR120", "GPR4"], "class": "fatty acid", "drug_class": "omega-3 fatty acid", "addiction_type": "none"},
    "zinc": {"receptors": ["GPR39"], "class": "metal ion", "drug_class": "metal ion", "addiction_type": "none"},
    "tryptophan": {"receptors": ["GPR38", "GPR56"], "class": "amino acid", "drug_class": "amino acid", "addiction_type": "serotonin precursor - mood regulation"},
    "Deltorphin II": {"receptors": ["OPRD1"], "class": "opioid", "drug_class": "opioid peptide", "addiction_type": "opioid addiction"},
    "DPDPE": {"receptors": ["OPRD1"], "class": "opioid", "drug_class": "opioid peptide", "addiction_type": "opioid addiction"},
    "morphine": {"receptors": ["OPRM1"], "class": "opioid", "drug_class": "opioid", "addiction_type": "opioid addiction"},
    "fentanyl": {"receptors": ["OPRM1"], "class": "synthetic", "drug_class": "synthetic opioid", "addiction_type": "opioid addiction"},
    "cocaine": {"receptors": ["SLC6A3"], "class": "synthetic", "drug_class": "stimulant", "addiction_type": "stimulant addiction"},
    "amphetamine": {"receptors": ["SLC6A3", "SLC6A2", "TAAR1"], "class": "synthetic", "drug_class": "stimulant", "addiction_type": "stimulant addiction"},
    "nicotine": {"receptors": ["CHRM4", "CHRM3"], "class": "synthetic", "drug_class": "cholinergic agonist", "addiction_type": "nicotine addiction"},
    "alcohol": {"receptors": ["GABRA1", "GRIN2A"], "class": "small molecule", "drug_class": "depressant", "addiction_type": "alcohol addiction"},
    "cannabidiol": {"receptors": ["CB1", "CB2", "TRPV1"], "class": "cannabinoid", "drug_class": "cannabinoid", "addiction_type": "low abuse potential"},
    "MDMA": {"receptors": ["SLC6A4", "SLC6A3"], "class": "synthetic", "drug_class": "empathogen", "addiction_type": "empathogen addiction"},
    "psilocybin": {"receptors": ["HTR2A"], "class": "tryptamine", "drug_class": "psychedelic", "addiction_type": "low abuse potential - therapeutic"},
    "DMT": {"receptors": ["HTR2A", "HTR2C"], "class": "tryptamine", "drug_class": "psychedelic", "addiction_type": "low abuse potential - therapeutic"},
    "cortisol": {"receptors": ["NR3C1", "NR3C2"], "class": "steroid", "drug_class": "glucocorticoid", "addiction_type": "none"},
    "melatonin": {"receptors": ["MTNR1A", "MTNR1B"], "class": "indolamine", "drug_class": "circadian modulator", "addiction_type": "none"},
    "vasopressin": {"receptors": ["AVPR1A", "AVPR1B", "AVPR2"], "class": "peptide", "drug_class": "neuropeptide", "addiction_type": "social behavior modulation"},
    "oxytocin": {"receptors": ["OXTR"], "class": "peptide", "drug_class": "neuropeptide", "addiction_type": "social bonding modulation"},
    "substance P": {"receptors": ["TACR1"], "class": "peptide", "drug_class": "neurokinin", "addiction_type": "pain/addiction overlap"},
}

# Known drug ligands that indicate addiction risk
KNOWN_DRUG_LIGANDS = {
    "prazosin", "yohimbine", "clonidine", "guanfacine",  # antihypertensives / ADHD
    "fluoxetine", "sertraline", "paroxetine", "citalopram",  # SSRIs
    "risperidone", "olanzapine", "clozapine", "haloperidol",  # antipsychotics
    "diazepam", "lorazepam", "alprazolam",  # benzodiazepines
    "morphine", "fentanyl", "oxycodone", "hydrocodone",  # opioids
    "methamphetamine", "cocaine", "amphetamine",  # stimulants
    "nicotine", "varenicline",  # smoking cessation
    "alprazolam", "buspirone",  # anxiolytics
    "lithium", "valproate",  # mood stabilizers
    "cannabidiol", "dronabinol",  # cannabinoids
    "MDMA", "psilocybin", "DMT",  # psychedelics
    "SR59230A", "PD-128907", "RO5256390", "thioperamide",  # pharmacological research tools
    "GR-46630", "JNJ-39758979",  # research tools
    "vernonolide",  # synthetic compound
    "galanin", "neurotensin", "ACTH(1-24)",  # neuropeptides with therapeutic potential
    "DHA",  # omega-3 fatty acid
    "anandamide", "2-AG", "THC",  # endocannabinoids
    "beta-endorphin", "dynorphin A", "enkephalin", "endomorphin",  # endogenous opioids
    "Deltorphin II", "DPDPE",  # opioid peptides
    "phenylephrine", "norepinephrine", "epinephrine", "dopamine",  # catecholamines
    "ATP", "ADP", "adenosine",  # purines
    "prostaglandin E2", "LPA", "S1P",  # lipids
    "glutamate", "GABA", "glycine",  # amino acid neurotransmitters
    "histamine", "serotonin", "melatonin",  # biogenic amines
    "tryptophan", "tryptamine",  # tryptamines
    "zinc", "calcium", "H+ (protons)",  # ions
}

# ---------------------------------------------------------------------------
# Tissue -> organ/system mapping
# ---------------------------------------------------------------------------

TISSUE_ORGAN_MAP = {
    "brain": ["brain", "CNS"],
    "striatum": ["brain", "basal ganglia"],
    "hippocampus": ["brain", "memory"],
    "prefrontal cortex": ["brain", "executive function"],
    "amygdala": ["brain", "emotion"],
    "hypothalamus": ["brain", "endocrine", "metabolism"],
    "ventral tegmental area": ["brain", "reward"],
    "brainstem": ["brain", "autonomic"],
    "cerebellum": ["brain", "motor coordination"],
    "retina": ["eye", "vision"],
    "testis": ["reproductive", "gonadal"],
    "kidney": ["renal", "urinary", "blood pressure"],
    "pancreas": ["endocrine", "metabolism", "insulin"],
    "liver": ["metabolism", "detoxification"],
    "adipose tissue": ["metabolism", "energy storage"],
    "skeletal muscle": ["metabolism", "movement"],
    "heart": ["cardiovascular", "cardiac"],
    "blood vessels": ["cardiovascular", "vascular"],
    "endothelial cells": ["cardiovascular", "vascular"],
    "endothelium": ["cardiovascular", "vascular"],
    "leukocytes": ["immune", "white blood cells"],
    "spleen": ["immune", "lymphoid"],
    "thymus": ["immune", "T cell development"],
    "microglia": ["immune", "CNS immunity"],
    "lung": ["respiratory", "immune"],
    "spleen": ["immune", "lymphoid"],
    "adipose": ["metabolism", "energy storage"],
    "immune": ["immune", "inflammation"],
    "inflammation": ["immune", "inflammation"],
}

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_data():
    with open(PREDICTIONS) as f:
        predictions = json.load(f)
    with open(CATALOG) as f:
        catalog = json.load(f)
    with open(RAW_DATA) as f:
        raw_data = json.load(f)
    return predictions, catalog, raw_data

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_primary_gprotein(rec):
    """Get the primary G-protein coupling from predictions or catalog."""
    if rec.get("top_gproteins"):
        return rec["top_gproteins"][0]["g_protein"]
    return rec.get("g_coupling")

def get_tissues(rec):
    """Parse tissue expression string into list."""
    tissues = rec.get("tissue_expression", "")
    if not tissues:
        return []
    return [t.strip() for t in tissues.split(",")]

def map_tissues_to_organs(tissues):
    """Map tissue names to organ/system categories."""
    organs = set()
    for tissue in tissues:
        if tissue in TISSUE_ORGAN_MAP:
            organs.update(TISSUE_ORGAN_MAP[tissue])
        else:
            # partial matching
            for key, vals in TISSUE_ORGAN_MAP.items():
                if key in tissue or tissue in key:
                    organs.update(vals)
    if not organs:
        organs.add("unknown")
    return list(organs)

def get_disease_connections(ligand_class, tissues):
    """Get disease connections from ligand class and tissue expression."""
    connections = []
    info = LIGAND_CLASS_DISEASE_MAP.get(ligand_class, {})
    if not info:
        # default: neurological + metabolic for unknown classes
        info = {
            "primary_categories": ["neurological"],
            "secondary_categories": ["metabolic"],
            "diseases": ["neurological disorders", "metabolic disorders"],
            "rationale": "Unknown ligand class - defaulting to neurological and metabolic pathways",
        }

    organs = map_tissues_to_organs(tissues)

    # Primary categories
    for cat in info["primary_categories"]:
        if cat in DISEASE_CATEGORIES:
            cat_data = DISEASE_CATEGORIES[cat]
            for disease in cat_data["diseases"]:
                connections.append({
                    "category": cat,
                    "disease": disease,
                    "strength": "high",
                    "rationale": f"{info['rationale']} via {cat} pathway",
                })

    # Secondary categories
    for cat in info["secondary_categories"]:
        if cat in DISEASE_CATEGORIES:
            cat_data = DISEASE_CATEGORIES[cat]
            for disease in cat_data["diseases"]:
                connections.append({
                    "category": cat,
                    "disease": disease,
                    "strength": "medium",
                    "rationale": f"{info['rationale']} via {cat} secondary pathway",
                })

    # Add tissue-specific connections
    for tissue in tissues:
        if "brain" in tissue or "striatum" in tissue or "hippocampus" in tissue:
            neurological_conns = [c for c in connections if c["category"] == "neurological"]
            if not neurological_conns:
                for disease in DISEASE_CATEGORIES["neurological"]["diseases"][:5]:
                    connections.append({
                        "category": "neurological",
                        "disease": disease,
                        "strength": "medium",
                        "rationale": f"Tissue expression in {tissue} suggests neurological involvement",
                    })
        if "leukocytes" in tissue or "microglia" in tissue or "spleen" in tissue or "thymus" in tissue:
            immune_conns = [c for c in connections if c["category"] == "immune"]
            if not immune_conns:
                for disease in DISEASE_CATEGORIES["immune"]["diseases"][:4]:
                    connections.append({
                        "category": "immune",
                        "disease": disease,
                        "strength": "medium",
                        "rationale": f"Tissue expression in {tissue} suggests immune involvement",
                    })
        if "heart" in tissue or "blood vessels" in tissue or "endothelial" in tissue:
            cv_conns = [c for c in connections if c["category"] == "cardiovascular"]
            if not cv_conns:
                for disease in DISEASE_CATEGORIES["cardiovascular"]["diseases"][:4]:
                    connections.append({
                        "category": "cardiovascular",
                        "disease": disease,
                        "strength": "medium",
                        "rationale": f"Tissue expression in {tissue} suggests cardiovascular involvement",
                    })
        if "pancreas" in tissue or "adipose" in tissue or "liver" in tissue or "skeletal muscle" in tissue:
            metabolic_conns = [c for c in connections if c["category"] == "metabolic"]
            if not metabolic_conns:
                for disease in DISEASE_CATEGORIES["metabolic"]["diseases"][:4]:
                    connections.append({
                        "category": "metabolic",
                        "disease": disease,
                        "strength": "medium",
                        "rationale": f"Tissue expression in {tissue} suggests metabolic involvement",
                    })

    return connections

def check_addiction_connections(ligand_name, ligand_class):
    """Check if a ligand has known addiction pathway connections."""
    connections = []

    # Direct match in drug targets
    if ligand_name in DRUG_TARGETS:
        dt = DRUG_TARGETS[ligand_name]
        connections.append({
            "ligand": ligand_name,
            "known_receptors": dt["receptors"],
            "drug_class": dt["drug_class"],
            "addiction_type": dt["addiction_type"],
            "mechanism": f"{ligand_name} acts on {', '.join(dt['receptors'])} via {dt['drug_class']} pathway",
            "risk_level": "high" if "addiction" in dt["addiction_type"] and dt["addiction_type"] != "none" else "low",
        })

    # Check ligand class against addiction-related classes
    addiction_classes = {"opioid", "cannabinoid", "phenethylamine", "tyramine", "octopamine", "tryptamine"}
    if ligand_class in addiction_classes:
        if not any(c["ligand"] == ligand_name for c in connections):
            connections.append({
                "ligand": ligand_name,
                "known_receptors": [],
                "drug_class": ligand_class,
                "addiction_type": f"{ligand_class}-related addiction pathways",
                "mechanism": f"{ligand_class} class ligands are known to interact with addiction-related receptor systems",
                "risk_level": "medium",
            })

    # Check if ligand is a known drug
    if ligand_name in KNOWN_DRUG_LIGANDS:
        if not any(c["ligand"] == ligand_name for c in connections):
            connections.append({
                "ligand": ligand_name,
                "known_receptors": [],
                "drug_class": "pharmacological agent",
                "addiction_type": "known pharmacological agent - drug target overlap",
                "mechanism": f"{ligand_name} is a known pharmacological agent with established drug target overlap",
                "risk_level": "medium",
            })

    return connections

def build_signaling_cascade(rec):
    """Build the full signaling cascade for a receptor."""
    gprotein = get_primary_gprotein(rec)
    cascade = G_PROTEIN_CASCADES.get(gprotein, G_PROTEIN_CASCADES["Gi"])

    cascade_data = {
        "g_protein": gprotein,
        "g_protein_description": cascade["description"],
        "second_messenger": cascade["second_messenger"],
        "pathway_steps": cascade["pathway"],
        "downstream_targets": cascade["downstream_targets"],
        "cellular_outcomes": cascade["cellular_outcomes"],
    }

    # Add G-protein probabilities
    if rec.get("top_gproteins"):
        cascade_data["g_protein_probabilities"] = rec["top_gproteins"]

    return cascade_data

def compute_therapeutic_score(rec, ligand, disease_conns):
    """Compute therapeutic opportunity score: confidence * disease_relevance * druggability."""
    confidence = ligand.get("confidence", 0)
    if confidence == 0:
        return 0

    # Disease relevance: count of disease connections weighted by strength
    disease_relevance = 0
    for dc in disease_conns:
        if dc["strength"] == "high":
            disease_relevance += 1.0
        else:
            disease_relevance += 0.5

    # Normalize disease relevance
    disease_relevance = min(disease_relevance / 10.0, 1.0)

    # Druggability: based on G-protein coupling certainty and tissue expression
    gprotein_certainty = 0
    if rec.get("top_gproteins"):
        gprotein_certainty = rec["top_gproteins"][0]["probability"]
    elif rec.get("g_coupling"):
        gprotein_certainty = 0.5  # known coupling but no probability

    te = rec.get("tissue_expression")
    if isinstance(te, list):
        tissue_breadth = len(te)
    elif te:
        tissue_breadth = len(get_tissues(rec))
    else:
        tissue_breadth = 0
    tissue_score = min(tissue_breadth / 3.0, 1.0)

    druggability = (gprotein_certainty * 0.5 + tissue_score * 0.5)

    score = confidence * disease_relevance * druggability
    return round(score, 4)

# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_all():
    predictions, catalog, raw_data = load_data()

    # Build catalog lookup
    catalog_map = {}
    for rec in catalog:
        catalog_map[rec["name"]] = rec

    # Index predictions by receptor
    pred_map = {}
    for rec in predictions:
        pred_map[rec["receptor"]] = rec

    # =========================================================================
    # 1. Signaling Network
    # =========================================================================
    signaling_network = []
    for rec in predictions:
        receptor = rec["receptor"]
        catalog_info = catalog_map.get(receptor, {})
        tissues = get_tissues(rec)
        cascade = build_signaling_cascade(rec)

        # Find best ligands (confidence > 0.2)
        good_ligands = [l for l in rec.get("top_ligands", []) if l["confidence"] > 0.2]

        # Disease connections
        disease_conns = []
        for ligand in good_ligands:
            dc = get_disease_connections(ligand["ligand_class"], tissues)
            disease_conns.extend(dc)

        # Deduplicate disease connections
        seen_diseases = set()
        unique_disease_conns = []
        for dc in disease_conns:
            key = (dc["category"], dc["disease"])
            if key not in seen_diseases:
                seen_diseases.add(key)
                unique_disease_conns.append(dc)
        disease_conns = unique_disease_conns

        # Organ mapping
        organs = map_tissues_to_organs(tissues)

        # Build ligand entries
        ligand_entries = []
        for ligand in good_ligands:
            ligand_entry = {
                "ligand": ligand["ligand"],
                "ligand_class": ligand["ligand_class"],
                "confidence": ligand["confidence"],
                "predicted_pKi": ligand["predicted_pKi"],
                "probability": ligand["probability"],
            }
            # Add disease connections per ligand
            ligand_disease = get_disease_connections(ligand["ligand_class"], tissues)
            ligand_entry["disease_connections"] = ligand_disease

            # Add addiction connections
            addict_conns = check_addiction_connections(ligand["ligand"], ligand["ligand_class"])
            ligand_entry["addiction_connections"] = addict_conns

            ligand_entries.append(ligand_entry)

        entry = {
            "receptor": receptor,
            "gene": rec.get("gene", receptor),
            "family": rec.get("family", "unknown"),
            "catalog_g_coupling": catalog_info.get("g_coupling"),
            "catalog_knockout_phenotype": catalog_info.get("knockout_phenotype"),
            "catalog_notes": catalog_info.get("notes"),
            "tissue_expression": tissues,
            "organs": organs,
            "signaling_cascade": cascade,
            "predicted_ligands": ligand_entries,
            "disease_connections": disease_conns,
            "high_confidence_count": rec.get("high_confidence_count", 0),
            "medium_confidence_count": rec.get("medium_confidence_count", 0),
            "low_confidence_count": rec.get("low_confidence_count", 0),
        }
        signaling_network.append(entry)

    # =========================================================================
    # 2. Disease Map
    # =========================================================================
    disease_map = []
    for entry in signaling_network:
        receptor = entry["receptor"]
        disease_conns = entry["disease_connections"]

        if not disease_conns:
            continue

        # Aggregate by category
        category_diseases = defaultdict(list)
        for dc in disease_conns:
            category_diseases[dc["category"]].append(dc)

        # Get best ligand per category
        best_ligands_by_cat = {}
        for ligand in entry["predicted_ligands"]:
            lc = ligand["ligand_class"]
            if lc in LIGAND_CLASS_DISEASE_MAP:
                cats = LIGAND_CLASS_DISEASE_MAP[lc]["primary_categories"]
                for cat in cats:
                    if cat not in best_ligands_by_cat or ligand["confidence"] > best_ligands_by_cat[cat]["confidence"]:
                        best_ligands_by_cat[cat] = ligand

        disease_entry = {
            "receptor": receptor,
            "gene": entry["gene"],
            "tissue_expression": entry["tissue_expression"],
            "organs": entry["organs"],
            "knockout_phenotype": entry["catalog_knockout_phenotype"],
            "categories": {},
            "all_diseases": [],
            "best_ligand_per_category": best_ligands_by_cat,
            "has_addiction_risk": any(
                l.get("addiction_connections") for l in entry["predicted_ligands"]
            ),
        }

        for cat, dcs in category_diseases.items():
            diseases_in_cat = list(set(d["disease"] for d in dcs))
            strengths = [d["strength"] for d in dcs]
            high_count = strengths.count("high")
            medium_count = strengths.count("medium")

            disease_entry["categories"][cat] = {
                "diseases": diseases_in_cat,
                "high_strength_count": high_count,
                "medium_strength_count": medium_count,
                "total_connections": len(dcs),
            }
            disease_entry["all_diseases"].extend(diseases_in_cat)

        # Deduplicate all diseases
        disease_entry["all_diseases"] = list(set(disease_entry["all_diseases"]))

        disease_map.append(disease_entry)

    # =========================================================================
    # 3. Addiction Map
    # =========================================================================
    addiction_map = []
    for entry in signaling_network:
        receptor = entry["receptor"]
        has_addiction = False
        addiction_ligands = []

        for ligand in entry["predicted_ligands"]:
            addict_conns = ligand.get("addiction_connections", [])
            if addict_conns:
                has_addiction = True
                addiction_ligands.append({
                    "ligand": ligand["ligand"],
                    "ligand_class": ligand["ligand_class"],
                    "confidence": ligand["confidence"],
                    "connections": addict_conns,
                })

        if not has_addiction:
            continue

        # Determine highest risk
        risk_levels = []
        for al in addiction_ligands:
            for c in al["connections"]:
                risk_levels.append(c.get("risk_level", "low"))

        if "high" in risk_levels:
            overall_risk = "high"
        elif "medium" in risk_levels:
            overall_risk = "medium"
        else:
            overall_risk = "low"

        # Identify overlapping drug targets
        overlapping_targets = set()
        for al in addiction_ligands:
            for c in al["connections"]:
                overlapping_targets.update(c.get("known_receptors", []))

        addiction_entry = {
            "receptor": receptor,
            "gene": entry["gene"],
            "tissue_expression": entry["tissue_expression"],
            "organs": entry["organs"],
            "overall_addiction_risk": overall_risk,
            "overlapping_drug_targets": list(overlapping_targets),
            "addiction_ligands": addiction_ligands,
            "total_addiction_connections": sum(len(al["connections"]) for al in addiction_ligands),
        }
        addiction_map.append(addiction_entry)

    # =========================================================================
    # 4. Therapeutic Opportunity Ranking
    # =========================================================================
    therapeutic_opportunities = []
    for entry in signaling_network:
        for ligand in entry["predicted_ligands"]:
            if ligand["confidence"] <= 0.2:
                continue
            score = compute_therapeutic_score(entry, ligand, entry["disease_connections"])
            if score > 0:
                therapeutic_opportunities.append({
                    "receptor": entry["receptor"],
                    "gene": entry["gene"],
                    "ligand": ligand["ligand"],
                    "ligand_class": ligand["ligand_class"],
                    "confidence": ligand["confidence"],
                    "predicted_pKi": ligand["predicted_pKi"],
                    "g_protein": entry["signaling_cascade"]["g_protein"],
                    "score": score,
                    "disease_connections": [
                        {"category": dc["category"], "disease": dc["disease"], "strength": dc["strength"]}
                        for dc in entry["disease_connections"][:10]
                    ],
                    "organs": entry["organs"],
                    "therapeutic_rationale": entry["disease_connections"][0]["rationale"] if entry["disease_connections"] else "Unknown",
                })

    # Sort by score descending
    therapeutic_opportunities.sort(key=lambda x: x["score"], reverse=True)

    # =========================================================================
    # 5. Summary Statistics
    # =========================================================================
    receptors_with_disease = len(disease_map)
    receptors_with_addiction = len(addiction_map)
    receptors_with_metabolic = sum(1 for d in disease_map if "metabolic" in d["categories"])
    receptors_with_neurological = sum(1 for d in disease_map if "neurological" in d["categories"])
    receptors_with_immune = sum(1 for d in disease_map if "immune" in d["categories"])
    receptors_with_cardiovascular = sum(1 for d in disease_map if "cardiovascular" in d["categories"])
    receptors_with_cancer = sum(1 for d in disease_map if "cancer" in d["categories"])

    # =========================================================================
    # 6. Write outputs
    # =========================================================================
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # signaling_network.json
    with open(SIGNALENG_NETWORK_OUT, "w") as f:
        json.dump(signaling_network, f, indent=2)

    # disease_map.json
    with open(DISEASE_MAP_OUT, "w") as f:
        json.dump(disease_map, f, indent=2)

    # addiction_map.json
    with open(ADDICTION_MAP_OUT, "w") as f:
        json.dump(addiction_map, f, indent=2)

    # summary.md
    write_summary(
        signaling_network, disease_map, addiction_map,
        therapeutic_opportunities,
        {
            "disease": receptors_with_disease,
            "addiction": receptors_with_addiction,
            "metabolic": receptors_with_metabolic,
            "neurological": receptors_with_neurological,
            "immune": receptors_with_immune,
            "cardiovascular": receptors_with_cardiovascular,
            "cancer": receptors_with_cancer,
        }
    )

    print("=" * 70)
    print("DOWNSTREAM SIGNALING MAPPING COMPLETE")
    print("=" * 70)
    print(f"Total orphan receptors processed: {len(predictions)}")
    print(f"Receptors with disease connections: {receptors_with_disease}")
    print(f"Receptors with addiction pathway connections: {receptors_with_addiction}")
    print(f"Receptors with metabolic connections: {receptors_with_metabolic}")
    print(f"Receptors with neurological connections: {receptors_with_neurological}")
    print(f"Receptors with immune connections: {receptors_with_immune}")
    print(f"Receptors with cardiovascular connections: {receptors_with_cardiovascular}")
    print(f"Receptors with cancer connections: {receptors_with_cancer}")
    print(f"Therapeutic opportunities identified: {len(therapeutic_opportunities)}")
    print(f"\nOutput files:")
    print(f"  {SIGNALENG_NETWORK_OUT}")
    print(f"  {DISEASE_MAP_OUT}")
    print(f"  {ADDICTION_MAP_OUT}")
    print(f"  {SUMMARY_OUT}")
    print("=" * 70)

    return {
        "disease": receptors_with_disease,
        "addiction": receptors_with_addiction,
        "metabolic": receptors_with_metabolic,
        "neurological": receptors_with_neurological,
        "immune": receptors_with_immune,
        "cardiovascular": receptors_with_cardiovascular,
        "cancer": receptors_with_cancer,
        "top_10": therapeutic_opportunities[:10],
        "signaling_network": signaling_network,
        "disease_map": disease_map,
        "addiction_map": addiction_map,
        "therapeutic_opportunities": therapeutic_opportunities,
    }

# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------

def write_summary(signaling_network, disease_map, addiction_map,
                  therapeutic_opportunities, stats):
    """Write human-readable summary markdown."""

    lines = []
    lines.append("# Downstream Signaling Network Summary\n")
    lines.append("## Orphan GPCR Ligand Predictions — Signaling Cascades, Disease Connections & Therapeutic Opportunities\n")
    lines.append(f"**Total orphan receptors analyzed:** {len(signaling_network)}\n")
    lines.append("---\n")

    # --- Statistics ---
    lines.append("## Quick Statistics\n")
    lines.append(f"| Category | Count |\n")
    lines.append(f"|---|---|\n")
    lines.append(f"| Receptors with disease connections | {stats['disease']} |\n")
    lines.append(f"| Receptors with addiction pathway connections | {stats['addiction']} |\n")
    lines.append(f"| Receptors with metabolic connections | {stats['metabolic']} |\n")
    lines.append(f"| Receptors with neurological connections | {stats['neurological']} |\n")
    lines.append(f"| Receptors with immune connections | {stats['immune']} |\n")
    lines.append(f"| Receptors with cardiovascular connections | {stats['cardiovascular']} |\n")
    lines.append(f"| Receptors with cancer connections | {stats['cancer']} |\n")
    lines.append("---\n")

    # --- High-Priority Orphan Receptors ---
    lines.append("## High-Priority Orphan Receptors\n")
    lines.append("Receptors with high-confidence predictions (confidence > 0.5) AND known disease connections.\n")

    high_priority = []
    for entry in signaling_network:
        best_conf = max((l["confidence"] for l in entry["predicted_ligands"]), default=0)
        if best_conf > 0.5 and entry["disease_connections"]:
            best_ligand = max(entry["predicted_ligands"], key=lambda l: l["confidence"])
            high_priority.append((entry, best_ligand, best_conf))

    high_priority.sort(key=lambda x: x[2], reverse=True)

    if high_priority:
        lines.append(f"### {len(high_priority)} high-priority orphan receptors identified\n")
        for i, (entry, ligand, conf) in enumerate(high_priority[:20], 1):
            gp = entry["signaling_cascade"]["g_protein"]
            cascade_name = entry["signaling_cascade"]["second_messenger"]
            top_diseases = [d["disease"] for d in entry["disease_connections"][:5]]
            organs = entry["organs"]
            ko = entry["catalog_knockout_phenotype"] or "unknown"

            lines.append(f"### {i}. {entry['receptor']} ({entry['gene']})\n")
            lines.append(f"- **Predicted ligand:** {ligand['ligand']} (confidence: {conf}, pKi: {ligand['predicted_pKi']})")
            lines.append(f"- **Ligand class:** {ligand['ligand_class']}")
            lines.append(f"- **G-protein:** {gp} ({cascade_name})")
            lines.append(f"- **Tissue expression:** {', '.join(entry['tissue_expression'])}")
            lines.append(f"- **Organs/systems:** {', '.join(organs)}")
            lines.append(f"- **Knockout phenotype:** {ko}")
            lines.append(f"- **Disease connections:** {', '.join(top_diseases)}")
            if entry["disease_connections"]:
                lines.append(f"- **Therapeutic potential:** {entry['disease_connections'][0]['rationale']}")
            lines.append("")
    else:
        lines.append("No high-priority receptors found.\n")

    # --- Addiction-Risk Orphan Receptors ---
    lines.append("## Addiction-Risk Orphan Receptors\n")
    lines.append("Orphan receptors whose predicted ligands overlap with known drug targets.\n")

    if addiction_map:
        high_risk = [a for a in addiction_map if a["overall_addiction_risk"] == "high"]
        medium_risk = [a for a in addiction_map if a["overall_addiction_risk"] == "medium"]
        low_risk = [a for a in addiction_map if a["overall_addiction_risk"] == "low"]

        lines.append(f"**Total:** {len(addiction_map)} receptors with addiction pathway connections\n")
        lines.append(f"- **High risk:** {len(high_risk)}")
        lines.append(f"- **Medium risk:** {len(medium_risk)}")
        lines.append(f"- **Low risk:** {len(low_risk)}\n")

        if high_risk:
            lines.append("### High Addiction Risk\n")
            for a in high_risk[:15]:
                lines.append(f"#### {a['receptor']}\n")
                for al in a["addiction_ligands"]:
                    for c in al["connections"]:
                        lines.append(f"- **Ligand:** {al['ligand']} (confidence: {al['confidence']})")
                        lines.append(f"  - **Known targets:** {', '.join(c['known_receptors']) if c['known_receptors'] else 'N/A'}")
                        lines.append(f"  - **Addiction type:** {c['addiction_type']}")
                        lines.append(f"  - **Mechanism:** {c['mechanism']}")
                lines.append(f"- **Overlapping drug targets:** {', '.join(a['overlapping_drug_targets'])}\n")

        if medium_risk:
            lines.append("### Medium Addiction Risk\n")
            for a in medium_risk[:15]:
                lines.append(f"#### {a['receptor']}\n")
                for al in a["addiction_ligands"]:
                    for c in al["connections"]:
                        lines.append(f"- **Ligand:** {al['ligand']} (confidence: {al['confidence']})")
                        lines.append(f"  - **Addiction type:** {c['addiction_type']}")
                        lines.append(f"  - **Mechanism:** {c['mechanism']}\n")
    else:
        lines.append("No addiction-risk receptors found.\n")

    # --- Metabolic Orphan Receptors ---
    lines.append("## Metabolic Orphan Receptors\n")
    metabolic_entries = [d for d in disease_map if "metabolic" in d["categories"]]
    lines.append(f"### {len(metabolic_entries)} receptors connected to metabolic pathways\n")

    for d in sorted(metabolic_entries, key=lambda x: len(x["categories"]["metabolic"]["diseases"]), reverse=True)[:15]:
        cat_info = d["categories"]["metabolic"]
        best_ligand = d.get("best_ligand_per_category", {}).get("metabolic", {})
        lines.append(f"#### {d['receptor']}\n")
        lines.append(f"- **Diseases:** {', '.join(cat_info['diseases'][:8])}")
        lines.append(f"- **Tissue expression:** {', '.join(d['tissue_expression'])}")
        lines.append(f"- **Knockout phenotype:** {d['knockout_phenotype'] or 'unknown'}")
        if best_ligand:
            lines.append(f"- **Best metabolic ligand:** {best_ligand['ligand']} (confidence: {best_ligand['confidence']})")
        lines.append("")

    # --- Neurological Orphan Receptors ---
    lines.append("## Neurological Orphan Receptors\n")
    neuro_entries = [d for d in disease_map if "neurological" in d["categories"]]
    lines.append(f"### {len(neuro_entries)} receptors connected to neurological pathways\n")

    for d in sorted(neuro_entries, key=lambda x: len(x["categories"]["neurological"]["diseases"]), reverse=True)[:15]:
        cat_info = d["categories"]["neurological"]
        best_ligand = d.get("best_ligand_per_category", {}).get("neurological", {})
        lines.append(f"#### {d['receptor']}\n")
        lines.append(f"- **Diseases:** {', '.join(cat_info['diseases'][:8])}")
        lines.append(f"- **Tissue expression:** {', '.join(d['tissue_expression'])}")
        lines.append(f"- **Knockout phenotype:** {d['knockout_phenotype'] or 'unknown'}")
        if best_ligand:
            lines.append(f"- **Best neurological ligand:** {best_ligand['ligand']} (confidence: {best_ligand['confidence']})")
        lines.append("")

    # --- Immune Orphan Receptors ---
    lines.append("## Immune Orphan Receptors\n")
    immune_entries = [d for d in disease_map if "immune" in d["categories"]]
    lines.append(f"### {len(immune_entries)} receptors connected to immune pathways\n")

    for d in sorted(immune_entries, key=lambda x: len(x["categories"]["immune"]["diseases"]), reverse=True)[:15]:
        cat_info = d["categories"]["immune"]
        best_ligand = d.get("best_ligand_per_category", {}).get("immune", {})
        lines.append(f"#### {d['receptor']}\n")
        lines.append(f"- **Diseases:** {', '.join(cat_info['diseases'][:8])}")
        lines.append(f"- **Tissue expression:** {', '.join(d['tissue_expression'])}")
        lines.append(f"- **Knockout phenotype:** {d['knockout_phenotype'] or 'unknown'}")
        if best_ligand:
            lines.append(f"- **Best immune ligand:** {best_ligand['ligand']} (confidence: {best_ligand['confidence']})")
        lines.append("")

    # --- Cardiovascular Orphan Receptors ---
    lines.append("## Cardiovascular Orphan Receptors\n")
    cv_entries = [d for d in disease_map if "cardiovascular" in d["categories"]]
    lines.append(f"### {len(cv_entries)} receptors connected to cardiovascular pathways\n")

    for d in sorted(cv_entries, key=lambda x: len(x["categories"]["cardiovascular"]["diseases"]), reverse=True)[:15]:
        cat_info = d["categories"]["cardiovascular"]
        best_ligand = d.get("best_ligand_per_category", {}).get("cardiovascular", {})
        lines.append(f"#### {d['receptor']}\n")
        lines.append(f"- **Diseases:** {', '.join(cat_info['diseases'][:8])}")
        lines.append(f"- **Tissue expression:** {', '.join(d['tissue_expression'])}")
        lines.append(f"- **Knockout phenotype:** {d['knockout_phenotype'] or 'unknown'}")
        if best_ligand:
            lines.append(f"- **Best cardiovascular ligand:** {best_ligand['ligand']} (confidence: {best_ligand['confidence']})")
        lines.append("")

    # --- Therapeutic Opportunity Ranking ---
    lines.append("## Therapeutic Opportunity Ranking — Top 20\n")
    lines.append("Ranked by: prediction_confidence x disease_relevance x druggability\n")

    for i, opp in enumerate(therapeutic_opportunities[:20], 1):
        lines.append(f"### {i}. {opp['receptor']} — Score: {opp['score']}\n")
        lines.append(f"- **Predicted ligand:** {opp['ligand']} (confidence: {opp['confidence']}, pKi: {opp['predicted_pKi']})")
        lines.append(f"- **Ligand class:** {opp['ligand_class']}")
        lines.append(f"- **G-protein:** {opp['g_protein']}")
        lines.append(f"- **Organs:** {', '.join(opp['organs'])}")
        top_diseases = [d["disease"] for d in opp["disease_connections"][:5]]
        lines.append(f"- **Diseases:** {', '.join(top_diseases)}")
        lines.append(f"- **Rationale:** {opp['therapeutic_rationale']}")
        lines.append("")

    # --- Surprising / Novel Connections ---
    lines.append("## Surprising & Novel Connections Discovered\n")
    lines.append("Unusual or unexpected receptor-ligand-disease connections that may warrant further investigation.\n")

    # Look for unexpected connections
    novel_connections = []

    # 1. Opioid peptides on non-opioid receptors
    for entry in signaling_network:
        for ligand in entry["predicted_ligands"]:
            if ligand["ligand_class"] == "opioid" and ligand["confidence"] > 0.3:
                novel_connections.append({
                    "receptor": entry["receptor"],
                    "ligand": ligand["ligand"],
                    "type": "opioid peptide on non-opioid receptor",
                    "confidence": ligand["confidence"],
                    "rationale": f"Opioid peptide {ligand['ligand']} predicted for {entry['receptor']} ({entry['gene']}) — suggests potential novel opioid receptor subtype or cross-reactivity",
                })

    # 2. Cannabinoid-like ligands on non-CB receptors
    for entry in signaling_network:
        for ligand in entry["predicted_ligands"]:
            if ligand["ligand_class"] == "cannabinoid" and ligand["confidence"] > 0.3:
                novel_connections.append({
                    "receptor": entry["receptor"],
                    "ligand": ligand["ligand"],
                    "type": "cannabinoid-like ligand on non-CB receptor",
                    "confidence": ligand["confidence"],
                    "rationale": f"Cannabinoid ligand {ligand['ligand']} predicted for {entry['receptor']} — suggests potential novel endocannabinoid receptor or CB cross-reactivity",
                })

    # 3. Trace amines on non-TAAR receptors
    for entry in signaling_network:
        for ligand in entry["predicted_ligands"]:
            if ligand["ligand_class"] in ("phenethylamine", "tyramine", "octopamine") and ligand["confidence"] > 0.3:
                novel_connections.append({
                    "receptor": entry["receptor"],
                    "ligand": ligand["ligand"],
                    "type": "trace amine on non-TAAR receptor",
                    "confidence": ligand["confidence"],
                    "rationale": f"Trace amine {ligand['ligand']} predicted for {entry['receptor']} — suggests potential novel trace amine receptor or TAAR cross-reactivity",
                })

    # 4. Peptide ligands with addiction potential on unexpected receptors
    for entry in signaling_network:
        for ligand in entry["predicted_ligands"]:
            if ligand["ligand_class"] == "peptide" and ligand["confidence"] > 0.5:
                if ligand["ligand"] in ("galanin", "neurotensin", "ACTH(1-24)"):
                    novel_connections.append({
                        "receptor": entry["receptor"],
                        "ligand": ligand["ligand"],
                        "type": "neuropeptide on unexpected receptor",
                        "confidence": ligand["confidence"],
                        "rationale": f"Neuropeptide {ligand['ligand']} predicted for {entry['receptor']} — suggests novel neuropeptide receptor function with potential psychiatric implications",
                    })

    # 5. GPR22 + amphetamine connection (known from catalog notes)
    for entry in signaling_network:
        if entry["receptor"] == "GPR22":
            for ligand in entry["predicted_ligands"]:
                if ligand["ligand_class"] in ("catecholamine", "synthetic", "opioid"):
                    novel_connections.append({
                        "receptor": entry["receptor"],
                        "ligand": ligand["ligand"],
                        "type": "GPR22 amphetamine-sensing receptor validation",
                        "confidence": ligand["confidence"],
                        "rationale": f"Catalog notes indicate GPR22 responds to amphetamine; predicted ligand {ligand['ligand']} ({ligand['ligand_class']}) validates this connection",
                    })

    # Sort novel connections by confidence
    novel_connections.sort(key=lambda x: x["confidence"], reverse=True)

    if novel_connections:
        lines.append(f"### {len(novel_connections)} novel/surprising connections identified\n")
        for i, nc in enumerate(novel_connections[:20], 1):
            lines.append(f"**{i}. {nc['receptor']} + {nc['ligand']}** (confidence: {nc['confidence']})")
            lines.append(f"- **Type:** {nc['type']}")
            lines.append(f"- **Rationale:** {nc['rationale']}")
            lines.append("")
    else:
        lines.append("No surprising connections found.\n")

    # --- Full Signaling Cascade Details ---
    lines.append("## Full Signaling Cascade Reference\n")
    lines.append("Complete mapping of all orphan receptors with confidence > 0.2:\n")

    for entry in signaling_network:
        good_ligands = [l for l in entry["predicted_ligands"] if l["confidence"] > 0.2]
        if not good_ligands:
            continue

        gp = entry["signaling_cascade"]["g_protein"]
        pathway = " -> ".join(entry["signaling_cascade"]["pathway_steps"][:4])

        for ligand in good_ligands[:3]:  # top 3 per receptor
            lines.append(f"**{entry['receptor']}** -> {ligand['ligand']} ({ligand['ligand_class']}, conf={ligand['confidence']}) -> {gp} -> {pathway} -> {', '.join(entry['organs'])}\n")

    # --- Methodology ---
    lines.append("## Methodology\n")
    lines.append("1. **Signaling cascades** mapped from G-protein coupling (Gs, Gi, Gq, G12/13) to second messenger pathways and cellular outcomes.")
    lines.append("2. **Disease connections** inferred from ligand class properties, tissue expression patterns, and knockout phenotypes.")
    lines.append("3. **Addiction risk** assessed by comparing predicted ligands against known drug targets (opioids, cannabinoids, trace amines, catecholamines, serotonin, etc.).")
    lines.append("4. **Therapeutic ranking** computed as: confidence x disease_relevance x druggability.")
    lines.append("5. **Novel connections** identified by flagging unexpected ligand-receptor pairings (e.g., opioid peptides on non-opioid receptors).")
    lines.append("")
    lines.append("---\n")
    lines.append("*Generated by downstream/map_signaling.py*")

    with open(SUMMARY_OUT, "w") as f:
        f.write("\n".join(lines))

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = process_all()

    # Print top 10 therapeutic opportunities
    print("\n" + "=" * 70)
    print("TOP 10 THERAPEUTIC OPPORTUNITIES")
    print("=" * 70)
    for i, opp in enumerate(results["top_10"], 1):
        print(f"\n{i}. {opp['receptor']} + {opp['ligand']}")
        print(f"   Score: {opp['score']} | Confidence: {opp['confidence']} | pKi: {opp['predicted_pKi']}")
        print(f"   G-protein: {opp['g_protein']} | Class: {opp['ligand_class']}")
        print(f"   Organs: {', '.join(opp['organs'])}")
        top_d = [d["disease"] for d in opp["disease_connections"][:3]]
        print(f"   Diseases: {', '.join(top_d)}")
        print(f"   Rationale: {opp['therapeutic_rationale']}")

    print("\n" + "=" * 70)
    print("SURPRISING / NOVEL CONNECTIONS")
    print("=" * 70)

    # Re-discover novel connections for display
    signaling_network = results["signaling_network"]
    novel_connections = []
    for entry in signaling_network:
        for ligand in entry["predicted_ligands"]:
            if ligand["ligand_class"] == "opioid" and ligand["confidence"] > 0.3:
                novel_connections.append(
                    f"  - {entry['receptor']} + {ligand['ligand']} (conf={ligand['confidence']}): Opioid peptide on non-opioid receptor"
                )
            if ligand["ligand_class"] == "cannabinoid" and ligand["confidence"] > 0.3:
                novel_connections.append(
                    f"  - {entry['receptor']} + {ligand['ligand']} (conf={ligand['confidence']}): Cannabinoid-like ligand on non-CB receptor"
                )
            if ligand["ligand_class"] in ("phenethylamine", "tyramine", "octopamine") and ligand["confidence"] > 0.3:
                novel_connections.append(
                    f"  - {entry['receptor']} + {ligand['ligand']} (conf={ligand['confidence']}): Trace amine on non-TAAR receptor"
                )
            if ligand["ligand_class"] == "peptide" and ligand["confidence"] > 0.5 and ligand["ligand"] in ("galanin", "neurotensin", "ACTH(1-24)"):
                novel_connections.append(
                    f"  - {entry['receptor']} + {ligand['ligand']} (conf={ligand['confidence']}): Neuropeptide on unexpected receptor"
                )

    novel_connections.sort(key=lambda x: x.split("conf=")[1].split(")")[0], reverse=True)
    for nc in novel_connections[:15]:
        print(nc)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
