"""
Symptom & Mechanism → 3D Organ Mapping
═══════════════════════════════════════════════════════════════
Maps each symptom and mechanism keyword to specific 3D anatomical
structures so NEXUS can reason in space, not just words.

Approach:
  • Symptoms have direct, primary, and adjacent organ mappings
  • Mechanisms inherit organs from disease+pathway+effect text
"""
from __future__ import annotations
import re
from typing import Dict, List, Set, Tuple


# ═══════════════════════════════════════════════════════════════
# SYMPTOM → 3D ORGAN MAPPING
# ═══════════════════════════════════════════════════════════════
# For each symptom, three tiers:
#   primary:   organs that DIRECTLY produce this symptom
#   adjacent:  organs nearby that may be involved
#   zones:     body regions for visual highlight

_FALLBACK_SYMPTOM_ORGAN_MAP: Dict[str, dict] = {

    # ── HEAD / NEUROLOGIC ──
    "headache": {
        "primary":  ["brain", "meninges", "superior_sagittal_sinus",
                     "transverse_sinus", "middle_cerebral_a"],
        "adjacent": ["cerebellum", "brainstem", "internal_carotid_a"],
        "zones":    ["head"],
    },
    "stiff_neck": {
        "primary":  ["meninges", "spinal_cord", "brainstem"],
        "adjacent": ["brain"],
        "zones":    ["head", "neck"],
    },
    "altered_mental_status": {
        "primary":  ["brain", "cerebellum", "brainstem"],
        "adjacent": ["middle_cerebral_a", "anterior_cerebral_a", "posterior_cerebral_a"],
        "zones":    ["head"],
    },
    "confusion": {
        "primary":  ["brain", "cerebellum"],
        "adjacent": ["middle_cerebral_a", "anterior_cerebral_a"],
        "zones":    ["head"],
    },
    "seizure": {
        "primary":  ["brain"],
        "adjacent": ["meninges", "cerebellum"],
        "zones":    ["head"],
    },
    "syncope": {
        "primary":  ["brain", "heart"],
        "adjacent": ["aorta", "internal_carotid_a", "basilar_a"],
        "zones":    ["head", "thorax_central"],
    },
    "dizziness": {
        "primary":  ["brain", "cerebellum", "brainstem"],
        "adjacent": ["basilar_a", "r_vertebral_a", "l_vertebral_a"],
        "zones":    ["head"],
    },
    "loss_of_consciousness": {
        "primary":  ["brain", "brainstem"],
        "adjacent": ["heart"],
        "zones":    ["head"],
    },
    "sensitivity_to_light": {
        "primary":  ["brain", "meninges"],
        "adjacent": ["ophthalmic_a"],
        "zones":    ["head"],
    },
    "weakness": {
        "primary":  ["brain", "spinal_cord", "muscles"],
        "adjacent": ["middle_cerebral_a"],
        "zones":    ["head"],
    },

    # ── CHEST / CARDIOPULMONARY ──
    "chest_pain": {
        "primary":  ["heart", "lad_a", "rca_a", "lcx_a", "left_main_coronary",
                     "aorta", "thoracic_aorta", "pericardium"
                     if False else "heart",  # pericardium not in atlas yet
                     "esophagus"],
        "adjacent": ["lungs", "r_lung", "l_lung", "pleura", "diaphragm"],
        "zones":    ["thorax_central"],
    },
    "chest_tightness": {
        "primary":  ["heart", "lad_a", "rca_a", "lungs"],
        "adjacent": ["aorta", "esophagus"],
        "zones":    ["thorax_central"],
    },
    "left_arm_pain": {
        "primary":  ["heart", "lad_a", "lcx_a", "l_arm"],
        "adjacent": ["aorta"],
        "zones":    ["thorax_central"],   # cardiac referral
    },
    "shortness_of_breath": {
        "primary":  ["lungs", "r_lung", "l_lung", "trachea",
                     "r_bronchus", "l_bronchus", "heart"],
        "adjacent": ["diaphragm", "pulm_aa", "pleura"],
        "zones":    ["thorax_central"],
    },
    "dyspnea": {
        "primary":  ["lungs", "r_lung", "l_lung", "heart"],
        "adjacent": ["trachea", "diaphragm", "pulm_aa"],
        "zones":    ["thorax_central"],
    },
    "cough": {
        "primary":  ["trachea", "r_bronchus", "l_bronchus", "lungs",
                     "r_lung", "l_lung"],
        "adjacent": ["pharynx", "diaphragm", "esophagus"],
        "zones":    ["thorax_central", "neck"],
    },
    "wheezing": {
        "primary":  ["r_bronchus", "l_bronchus", "lungs", "r_lung", "l_lung"],
        "adjacent": ["trachea"],
        "zones":    ["thorax_central"],
    },
    "hemoptysis": {
        "primary":  ["lungs", "r_lung", "l_lung", "r_bronchus", "l_bronchus",
                     "bronchial_a"],
        "adjacent": ["trachea", "pulm_aa"],
        "zones":    ["thorax_central"],
    },
    "palpitations": {
        "primary":  ["heart", "right_atrium", "left_atrium", "left_ventricle"],
        "adjacent": ["aorta"],
        "zones":    ["thorax_central"],
    },

    # ── ABDOMEN / GI ──
    "abdominal_pain": {
        "primary":  ["stomach", "duodenum", "jejunum", "ileum",
                     "asc_colon", "trans_colon", "desc_colon",
                     "liver", "gallbladder", "pancreas", "appendix"],
        "adjacent": ["sma", "ima", "celiac"],
        "zones":    ["umbilical", "epigastric"],
    },
    "epigastric_pain": {
        "primary":  ["stomach", "duodenum", "pancreas", "esophagus"],
        "adjacent": ["heart", "celiac", "gastroduodenal_a"],
        "zones":    ["epigastric"],
    },
    "ruq_pain": {
        "primary":  ["liver", "gallbladder", "bile_duct", "duodenum"],
        "adjacent": ["proper_hepatic_a", "cystic_a", "cystic_v",
                     "diaphragm", "r_kidney"],
        "zones":    ["ruq"],
    },
    "right_upper_quadrant_pain": {
        "primary":  ["liver", "gallbladder", "bile_duct"],
        "adjacent": ["proper_hepatic_a", "cystic_a", "duodenum"],
        "zones":    ["ruq"],
    },
    "luq_pain": {
        "primary":  ["spleen", "stomach", "splenic_a", "splenic_v"],
        "adjacent": ["pancreas", "diaphragm", "l_kidney"],
        "zones":    ["luq"],
    },
    "rlq_pain": {
        "primary":  ["appendix", "cecum", "ileum", "ileocolic_a"],
        "adjacent": ["asc_colon", "ovaries", "uterus", "r_ureter",
                     "external_iliac_a", "testicular_a"],
        "zones":    ["rlq"],
    },
    "right_lower_quadrant_pain": {
        "primary":  ["appendix", "cecum", "ileum"],
        "adjacent": ["ovaries", "r_ureter", "ileocolic_a"],
        "zones":    ["rlq"],
    },
    "mcburney_point": {
        "primary":  ["appendix"],
        "adjacent": ["cecum", "ileum", "ileocolic_a"],
        "zones":    ["rlq"],
    },
    "llq_pain": {
        "primary":  ["sig_colon", "desc_colon"],
        "adjacent": ["ovaries", "uterus", "l_ureter", "sigmoid_aa"],
        "zones":    ["llq"],
    },
    "left_lower_quadrant_pain": {
        "primary":  ["sig_colon", "desc_colon"],
        "adjacent": ["ovaries", "l_ureter"],
        "zones":    ["llq"],
    },
    "periumbilical_pain": {
        "primary":  ["jejunum", "ileum", "trans_colon"],
        "adjacent": ["mes_ln", "sma"],
        "zones":    ["umbilical"],
    },
    "suprapubic_pain": {
        "primary":  ["bladder", "uterus", "prostate"],
        "adjacent": ["rectum", "sig_colon"],
        "zones":    ["suprapubic"],
    },
    "pelvic_pain": {
        "primary":  ["uterus", "ovaries", "bladder", "prostate", "rectum"],
        "adjacent": ["sig_colon", "internal_iliac_a"],
        "zones":    ["pelvis"],
    },

    # ── GI symptoms ──
    "nausea": {
        "primary":  ["stomach", "duodenum", "brain"],
        "adjacent": ["esophagus", "vagus_n"],
        "zones":    ["epigastric", "head"],
    },
    "vomiting": {
        "primary":  ["stomach", "esophagus", "duodenum"],
        "adjacent": ["brain", "vagus_n"],
        "zones":    ["epigastric"],
    },
    "diarrhea": {
        "primary":  ["jejunum", "ileum", "asc_colon", "desc_colon",
                     "trans_colon", "sig_colon"],
        "adjacent": ["rectum", "mes_ln"],
        "zones":    ["umbilical"],
    },
    "constipation": {
        "primary":  ["sig_colon", "rectum", "desc_colon"],
        "adjacent": ["asc_colon", "trans_colon"],
        "zones":    ["llq", "suprapubic"],
    },
    "hematemesis": {
        "primary":  ["stomach", "esophagus", "duodenum",
                     "gastroduodenal_a", "left_gastric_a"],
        "adjacent": ["liver", "portal_vein"],
        "zones":    ["epigastric"],
    },
    "melena": {
        "primary":  ["stomach", "duodenum", "jejunum", "gastroduodenal_a"],
        "adjacent": ["ileum", "portal_vein"],
        "zones":    ["epigastric", "umbilical"],
    },
    "hematochezia": {
        "primary":  ["sig_colon", "rectum", "desc_colon"],
        "adjacent": ["asc_colon", "superior_rectal_a"],
        "zones":    ["llq", "suprapubic"],
    },

    # ── URINARY ──
    "dysuria": {
        "primary":  ["bladder", "urethra"],
        "adjacent": ["prostate", "uterus"],
        "zones":    ["suprapubic"],
    },
    "urinary_frequency": {
        "primary":  ["bladder", "urethra"],
        "adjacent": ["prostate"],
        "zones":    ["suprapubic"],
    },
    "hematuria": {
        "primary":  ["bladder", "r_kidney", "l_kidney", "r_ureter", "l_ureter"],
        "adjacent": ["urethra", "prostate"],
        "zones":    ["suprapubic", "right_flank", "left_flank"],
    },
    "flank_pain": {
        "primary":  ["r_kidney", "l_kidney", "r_ureter", "l_ureter"],
        "adjacent": ["r_renal_a", "l_renal_a"],
        "zones":    ["right_flank", "left_flank"],
    },
    "right_flank_pain": {
        "primary":  ["r_kidney", "r_ureter"],
        "adjacent": ["r_renal_a", "liver"],
        "zones":    ["right_flank"],
    },
    "left_flank_pain": {
        "primary":  ["l_kidney", "l_ureter"],
        "adjacent": ["l_renal_a", "spleen"],
        "zones":    ["left_flank"],
    },

    # ── SYSTEMIC ──
    "fever": {
        "primary":  ["brain"],     # hypothalamus thermoregulation
        "adjacent": ["spleen", "bone_marrow"],
        "zones":    [],            # systemic — no specific zone
    },
    "chills": {
        "primary":  ["brain", "muscles"],
        "adjacent": ["bone_marrow"],
        "zones":    [],
    },
    "fatigue": {
        "primary":  ["brain", "heart", "muscles"],
        "adjacent": [],
        "zones":    [],
    },
    "body_aches": {
        "primary":  ["muscles"],
        "adjacent": ["bone_marrow"],
        "zones":    [],
    },
    "sweating": {
        "primary":  ["skin", "brain"],
        "adjacent": ["heart"],
        "zones":    [],
    },

    # ── BLEEDING ──
    "bleeding_minor": {
        "primary":  ["skin"],
        "adjacent": [],
        "zones":    [],
    },
    "bleeding_severe": {
        "primary":  ["heart", "aorta", "ivc", "bone_marrow"],
        "adjacent": ["spleen", "liver"],
        "zones":    [],
    },

    # ── REPRODUCTIVE ──
    "vaginal_bleeding": {
        "primary":  ["uterus", "ovaries"],
        "adjacent": ["uterine_a", "ovarian_a"],
        "zones":    ["pelvis", "suprapubic"],
    },
    "testicular_pain": {
        "primary":  ["testicular_a"],
        "adjacent": [],
        "zones":    ["pelvis"],
    },

    # ── LIMBS ──
    "calf_pain": {
        "primary":  ["soleal_vv", "gastrocnemius_vv", "posterior_tibial_a",
                     "great_saphenous_v"],
        "adjacent": ["r_popliteal_v", "l_popliteal_v"],
        "zones":    [],
    },
    "leg_swelling": {
        "primary":  ["great_saphenous_v", "small_saphenous_v",
                     "soleal_vv", "r_femoral_v", "l_femoral_v"],
        "adjacent": ["lower_limbs", "ivc"],
        "zones":    [],
    },
}


# Load from JSON via knowledge_loader, fall back to inline data if missing
try:
    from .knowledge_loader import get_kb
    _kb = get_kb()
    SYMPTOM_ORGAN_MAP = _kb.symptom_organ_map or _FALLBACK_SYMPTOM_ORGAN_MAP
except (ImportError, Exception):
    try:
        from knowledge_loader import get_kb
        _kb = get_kb()
        SYMPTOM_ORGAN_MAP = _kb.symptom_organ_map or _FALLBACK_SYMPTOM_ORGAN_MAP
    except Exception:
        SYMPTOM_ORGAN_MAP = _FALLBACK_SYMPTOM_ORGAN_MAP


# ═══════════════════════════════════════════════════════════════
# MECHANISM TEXT → ORGAN INFERENCE
# ═══════════════════════════════════════════════════════════════
# Mechanisms don't have explicit organ fields. We infer organs from
# keywords in mechanism title, pathway, and effects.

_FALLBACK_MECH_KEYWORD_TO_ORGAN: Dict[str, List[str]] = {
    # Cardiac
    "myocardial":         ["heart", "lad_a", "rca_a", "lcx_a", "left_main_coronary"],
    "cardiac":            ["heart", "left_ventricle", "right_ventricle"],
    "coronary":           ["coronary_aa", "lad_a", "rca_a", "lcx_a"],
    "pericardial":        ["heart"],
    "atrial":             ["right_atrium", "left_atrium"],
    "ventricular":        ["left_ventricle", "right_ventricle"],

    # Vascular
    "aortic":             ["aorta", "aortic_arch", "thoracic_aorta", "abdominal_aorta"],
    "arterial":           ["aorta"],
    "atherosclerosis":    ["aorta", "lad_a", "rca_a"],
    "thrombosis":         ["soleal_vv", "great_saphenous_v"],
    "embolism":           ["pulm_aa", "r_pulm_a", "l_pulm_a"],

    # Pulmonary
    "pulmonary":          ["lungs", "r_lung", "l_lung", "pulm_aa"],
    "respiratory":        ["lungs", "trachea", "r_bronchus", "l_bronchus"],
    "bronchial":          ["r_bronchus", "l_bronchus", "bronchial_a"],
    "alveolar":           ["r_lung", "l_lung", "lungs"],
    "pneumonia":          ["lungs", "r_lung", "l_lung"],
    "asthma":             ["r_bronchus", "l_bronchus", "lungs"],
    "copd":               ["lungs", "r_bronchus", "l_bronchus"],

    # Neurologic
    "cerebral":           ["brain", "middle_cerebral_a", "anterior_cerebral_a",
                           "posterior_cerebral_a"],
    "stroke":             ["brain", "middle_cerebral_a", "internal_carotid_a"],
    "ischemic_stroke":    ["brain", "middle_cerebral_a"],
    "meningeal":          ["meninges", "brain"],
    "meningitis":         ["meninges", "brain", "spinal_cord"],
    "encephalitis":       ["brain"],
    "seizure":            ["brain"],
    "spinal":             ["spinal_cord"],

    # GI
    "gastric":            ["stomach", "left_gastric_a"],
    "esophageal":         ["esophagus"],
    "duodenal":           ["duodenum", "gastroduodenal_a"],
    "intestinal":         ["jejunum", "ileum"],
    "colonic":            ["asc_colon", "desc_colon", "trans_colon"],
    "appendiceal":        ["appendix", "cecum", "ileocolic_a"],
    "appendicitis":       ["appendix", "cecum", "ileocolic_a"],
    "gastroenteritis":    ["stomach", "jejunum", "ileum", "asc_colon"],
    "colitis":            ["asc_colon", "trans_colon", "desc_colon", "sig_colon"],
    "diverticulitis":     ["sig_colon", "desc_colon"],

    # Hepatobiliary
    "hepatic":            ["liver", "proper_hepatic_a", "hepatic_vv"],
    "liver":              ["liver"],
    "biliary":            ["gallbladder", "bile_duct", "cystic_a"],
    "cholecyst":          ["gallbladder", "cystic_a", "cystic_v"],
    "pancreatic":         ["pancreas"],
    "pancreatitis":       ["pancreas", "duodenum"],

    # Renal
    "renal":              ["r_kidney", "l_kidney", "r_renal_a", "l_renal_a"],
    "kidney":             ["r_kidney", "l_kidney"],
    "ureteral":           ["r_ureter", "l_ureter"],
    "nephritis":          ["r_kidney", "l_kidney"],
    "pyelonephritis":     ["r_kidney", "l_kidney", "r_ureter", "l_ureter"],
    "cystitis":           ["bladder"],
    "urinary":            ["bladder", "urethra", "r_kidney", "l_kidney"],

    # Endocrine
    "thyroid":            ["thyroid"],
    "adrenal":            ["r_adrenal", "l_adrenal"],
    "diabetic":           ["pancreas"],
    "diabetes":           ["pancreas"],
    "pituitary":          ["pituitary"],

    # Hematologic
    "splenic":            ["spleen", "splenic_a", "splenic_v"],
    "hematologic":        ["bone_marrow", "spleen"],
    "anemia":             ["bone_marrow"],

    # Reproductive
    "uterine":            ["uterus", "uterine_a"],
    "ovarian":            ["ovaries", "ovarian_a"],
    "testicular":         ["testicular_a"],
    "prostatic":          ["prostate"],

    # Lymphatic
    "lymphatic":          ["thoracic_duct", "cerv_ln", "axil_ln", "mes_ln"],
    "lymphadenopathy":    ["cerv_ln", "axil_ln", "ing_ln", "mes_ln"],

    # Vessels
    "carotid":            ["r_carotid_a", "l_carotid_a", "internal_carotid_a"],
    "femoral":            ["r_femoral_a", "l_femoral_a"],
    "iliac":              ["r_iliac_a", "l_iliac_a", "internal_iliac_a"],

    # Inflammation/sepsis (systemic)
    "sepsis":             ["heart", "lungs", "r_kidney", "l_kidney", "liver",
                           "spleen"],
    "septic_shock":       ["heart", "lungs", "r_kidney", "l_kidney"],
    "shock":              ["heart", "brain", "r_kidney", "l_kidney"],
}


# Load from JSON via knowledge_loader, fall back to inline data if missing
try:
    from .knowledge_loader import get_kb
    _kb = get_kb()
    MECH_KEYWORD_TO_ORGAN = _kb.mechanism_keywords or _FALLBACK_MECH_KEYWORD_TO_ORGAN
except (ImportError, Exception):
    try:
        from knowledge_loader import get_kb
        _kb = get_kb()
        MECH_KEYWORD_TO_ORGAN = _kb.mechanism_keywords or _FALLBACK_MECH_KEYWORD_TO_ORGAN
    except Exception:
        MECH_KEYWORD_TO_ORGAN = _FALLBACK_MECH_KEYWORD_TO_ORGAN


# ═══════════════════════════════════════════════════════════════
# API
# ═══════════════════════════════════════════════════════════════

def get_organs_for_symptom(symptom: str) -> dict:
    """
    Map a symptom to its 3D anatomical structures.
    Returns {primary, adjacent, zones} or empty if unknown.
    """
    s = symptom.lower().strip().replace(' ', '_')

    # Direct lookup
    if s in SYMPTOM_ORGAN_MAP:
        return dict(SYMPTOM_ORGAN_MAP[s])

    # Fuzzy match — substring
    for key, mapping in SYMPTOM_ORGAN_MAP.items():
        if s in key or key in s:
            return dict(mapping)

    # Try common synonym replacements
    synonyms = {
        "stomach_pain":       "abdominal_pain",
        "tummy_pain":         "abdominal_pain",
        "belly_pain":         "abdominal_pain",
        "throwing_up":        "vomiting",
        "loose_stools":       "diarrhea",
        "shortness_breath":   "shortness_of_breath",
        "sob":                "shortness_of_breath",
        "lightheadedness":    "dizziness",
        "passed_out":         "syncope",
        "fainting":           "syncope",
    }
    if s in synonyms:
        return dict(SYMPTOM_ORGAN_MAP.get(synonyms[s], {}))

    return {"primary": [], "adjacent": [], "zones": []}


def get_organs_for_mechanism(mechanism: dict) -> List[str]:
    """
    Infer 3D organs implicated by a mechanism.
    Pulls keywords from title, description, pathway, results, related_diseases.
    """
    organs: Set[str] = set()

    # Build searchable text from mechanism
    text_parts = [
        str(mechanism.get('mechanism_name', '')),
        str(mechanism.get('title', '')),
        str(mechanism.get('description', '')),
        ' '.join(mechanism.get('process', [])) if isinstance(mechanism.get('process'), list) else '',
        ' '.join(mechanism.get('pathway', [])) if isinstance(mechanism.get('pathway'), list) else '',
        ' '.join(mechanism.get('results', [])) if isinstance(mechanism.get('results'), list) else '',
        ' '.join(mechanism.get('effects', [])) if isinstance(mechanism.get('effects'), list) else '',
        ' '.join(mechanism.get('related_diseases', [])) if isinstance(mechanism.get('related_diseases'), list) else '',
    ]
    full_text = ' '.join(text_parts).lower()

    # Match keywords
    for keyword, organ_list in MECH_KEYWORD_TO_ORGAN.items():
        if keyword in full_text:
            organs.update(organ_list)

    # Also use related_symptoms — pull organs from each symptom
    for sym in mechanism.get('related_symptoms', []) + mechanism.get('typical_symptoms', []):
        sym_organs = get_organs_for_symptom(sym).get('primary', [])
        organs.update(sym_organs)

    return sorted(organs)


def merge_symptom_organs(symptoms: List[str]) -> dict:
    """
    Aggregate organ mapping across multiple symptoms.
    Returns:
      • primary, adjacent, zones — union sets
      • dominant_zones — zones hit by ≥50% of matched symptoms
      • zone_counts — per-zone hit count
    """
    from collections import Counter

    primary = set()
    adjacent = set()
    zones = set()
    zone_counts = Counter()
    organ_counts = Counter()
    matched = []
    unmatched = []

    for sym in symptoms:
        m = get_organs_for_symptom(sym)
        if m.get('primary') or m.get('zones'):
            primary.update(m.get('primary', []))
            adjacent.update(m.get('adjacent', []))
            zones.update(m.get('zones', []))
            for z in m.get('zones', []):
                zone_counts[z] += 1
            for o in m.get('primary', []):
                organ_counts[o] += 1
            matched.append(sym)
        else:
            unmatched.append(sym)

    # Adjacent that's also primary in another symptom → keep in primary
    adjacent -= primary

    # Dominant zones: hit by ≥50% of matched symptoms (a real localization signal)
    n = max(len(matched), 1)
    threshold = max(1, n // 2)  # at least half of symptoms
    dominant_zones = sorted(z for z, c in zone_counts.items() if c >= threshold)

    # Dominant organs: hit by ≥40% of matched symptoms
    organ_threshold = max(1, int(0.4 * n))
    dominant_organs = sorted(o for o, c in organ_counts.items() if c >= organ_threshold)

    return {
        "primary":          sorted(primary),
        "adjacent":         sorted(adjacent),
        "zones":            sorted(zones),
        "dominant_zones":   dominant_zones,
        "dominant_organs":  dominant_organs,
        "zone_counts":      dict(zone_counts),
        "organ_counts":     dict(organ_counts),
        "matched_symptoms":   matched,
        "unmatched_symptoms": unmatched,
    }


# ═══════════════════════════════════════════════════════════════
# Quick test
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== TEST 1: Single symptom mapping ===")
    for sym in ["chest pain", "headache", "rlq pain", "shortness of breath",
                 "calf_pain", "fever"]:
        m = get_organs_for_symptom(sym)
        print(f"\n  {sym}:")
        print(f"    primary:  {m['primary'][:5]}")
        print(f"    adjacent: {m['adjacent'][:5]}")
        print(f"    zones:    {m['zones']}")

    print("\n=== TEST 2: Multi-symptom aggregation ===")
    syms = ["chest pain", "shortness of breath", "left arm pain"]
    agg = merge_symptom_organs(syms)
    print(f"\n  Symptoms: {syms}")
    print(f"  → primary:  {agg['primary'][:8]}")
    print(f"  → adjacent: {agg['adjacent'][:8]}")
    print(f"  → zones:    {agg['zones']}")

    print("\n=== TEST 3: Mechanism organ inference ===")
    sample_mech = {
        "mechanism_name": "Myocardial Ischemia",
        "description": "Reduced blood flow to heart muscle from coronary artery occlusion",
        "related_diseases": ["heart_attack", "stroke"],
        "related_symptoms": ["chest_pain", "left_arm_pain"],
    }
    organs = get_organs_for_mechanism(sample_mech)
    print(f"  Mechanism: {sample_mech['mechanism_name']}")
    print(f"  → organs: {organs}")



# ═══════════════════════════════════════════════════════════════
# DERIVATION WRAPPER for SYMPTOM_ORGAN_MAP
# ═══════════════════════════════════════════════════════════════
# Old JSON had {primary, adjacent, zones}. We removed adjacent + zones.
# This wrapper lazily derives them from primary organs via DerivationEngine.

_derivation_engine = None

def _get_derivation():
    global _derivation_engine
    if _derivation_engine is None:
        try:
            from .derivation_engine import DerivationEngine
            _derivation_engine = DerivationEngine()
        except ImportError:
            try:
                from derivation_engine import DerivationEngine
                _derivation_engine = DerivationEngine()
            except ImportError:
                _derivation_engine = False  # mark as failed
    return _derivation_engine if _derivation_engine else None


def get_symptom_anatomy(symptom_id: str) -> dict:
    """
    Get a symptom's full anatomy dict — primary (from JSON) + 
    zones/adjacent (DERIVED at call time from primary).
    
    This replaces direct dict access to SYMPTOM_ORGAN_MAP[symptom_id].
    """
    info = SYMPTOM_ORGAN_MAP.get(symptom_id, {})
    if not info:
        return {}
    result = dict(info)  # copy
    
    de = _get_derivation()
    if de:
        primary = result.get('primary', [])
        if primary and 'zones' not in result:
            result['zones'] = de.derive_symptom_zones(primary)
            result['_zones_derived'] = True
        if primary and 'adjacent' not in result:
            result['adjacent'] = de.derive_symptom_adjacent(primary)
            result['_adjacent_derived'] = True
    
    return result