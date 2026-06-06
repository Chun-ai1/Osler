"""
case_targets.py — bridge from a clinical indication to the physiological target
states that the pharmacology reasoning engine (reasoning_engine.recommend) needs.

A "target" is {organ, variable, direction} expressed in the SAME state-variable
vocabulary that drugs use in drugs_pkpd.json (state_effects). reasoning_engine
matches a drug when its state_effect has the same `variable`, a compatible
`organ` (the drug's organ, or "*"), and the opposite-of-disease direction.

⚠️ Variable names below are aligned BY HAND to the names that actually appear in
drugs_pkpd.json. Matching is exact-string; a typo here silently yields no drugs.
Run `py case_targets.py` to self-check every indication produces >=1 candidate.
"""
from __future__ import annotations
from typing import Dict, List, Tuple

# canonical_indication -> {"targets": [...], "aliases": [...], "label": human title}
# direction: "low" = push the variable down, "high" = push it up.
INDICATIONS: Dict[str, Dict] = {
    "acute coronary syndrome": {
        "label": "Acute coronary syndrome (ACS / heart attack)",
        "aliases": ["acs", "heart attack", "mi", "myocardial infarction", "stemi", "nstemi"],
        "scenario": "acs",
        "targets": [
            {"organ": "heart", "variable": "myocardial_oxygen_demand", "direction": "low"},
            {"organ": "blood", "variable": "platelet_aggregation", "direction": "low"},
            {"organ": "blood", "variable": "clot_propagation_risk", "direction": "low"},
        ],
    },
    "stable angina": {
        "label": "Stable angina",
        "aliases": ["angina", "anginal pain", "chronic angina"],
        "scenario": "angina",
        "targets": [
            {"organ": "heart", "variable": "myocardial_oxygen_demand", "direction": "low"},
            {"organ": "heart", "variable": "preload", "direction": "low"},
        ],
    },
    "asthma": {
        "label": "Asthma / acute bronchospasm",
        "aliases": ["bronchospasm", "asthma attack", "wheezing", "reactive airway"],
        "scenario": "asthma",
        "targets": [
            {"organ": "lung", "variable": "bronchospasm", "direction": "low"},
            {"organ": "lung", "variable": "airflow_resistance", "direction": "low"},
        ],
    },
    "gerd": {
        "label": "GERD / acid-peptic disease",
        "aliases": ["gastroesophageal reflux disease", "reflux", "heartburn",
                    "peptic ulcer", "dyspepsia", "gastritis"],
        "scenario": "gerd",
        "targets": [
            {"organ": "gi", "variable": "gastric_acid", "direction": "low"},
        ],
    },
    "hypertension": {
        "label": "Hypertension",
        "aliases": ["htn", "high blood pressure", "elevated bp"],
        "scenario": "hypertension",
        "targets": [
            {"organ": "heart", "variable": "systemic_BP", "direction": "low"},
            {"organ": "heart", "variable": "afterload", "direction": "low"},
        ],
    },
    "anaphylaxis": {
        "label": "Anaphylaxis / acute allergic reaction",
        "aliases": ["allergic reaction", "allergy", "hives", "urticaria", "angioedema"],
        "scenario": "anaphylaxis",
        "targets": [
            {"organ": "skin", "variable": "histamine_release", "direction": "low"},
            {"organ": "vessel", "variable": "vasoconstriction", "direction": "high"},
            {"organ": "lung", "variable": "bronchospasm", "direction": "low"},
        ],
    },
    "migraine": {
        "label": "Migraine",
        "aliases": ["migraine headache", "headache"],
        "scenario": "migraine",
        "targets": [
            {"organ": "vessel", "variable": "cranial_vasodilation", "direction": "low"},
            {"organ": "brain", "variable": "neurogenic_inflammation", "direction": "low"},
        ],
    },
    "venous thromboembolism": {
        "label": "Venous thromboembolism (DVT / PE)",
        "aliases": ["dvt", "pe", "pulmonary embolism", "deep vein thrombosis", "thrombosis"],
        "scenario": "vte",
        "targets": [
            {"organ": "blood", "variable": "clot_propagation_risk", "direction": "low"},
            {"organ": "blood", "variable": "coagulation_cascade", "direction": "low"},
        ],
    },
    "bacterial infection": {
        "label": "Bacterial infection / sepsis",
        "aliases": ["sepsis", "infection", "pneumonia", "uti", "cellulitis", "bacteremia"],
        "scenario": "infection",
        "targets": [
            {"organ": "*", "variable": "infection_load", "direction": "low"},
            {"organ": "*", "variable": "bacterial_pathogen", "direction": "low"},
        ],
    },
    "acute pain": {
        "label": "Acute pain",
        "aliases": ["pain", "severe pain"],
        "scenario": "pain",
        "targets": [
            {"organ": "brain", "variable": "pain_perception", "direction": "low"},
        ],
    },
    "inflammation": {
        "label": "Inflammation / fever",
        "aliases": ["fever", "inflammatory pain", "musculoskeletal pain"],
        "scenario": "inflammation",
        "targets": [
            {"organ": "*", "variable": "inflammation", "direction": "low"},
        ],
    },
    "acute heart failure": {
        "label": "Acute heart failure / fluid overload",
        "aliases": ["fluid overload", "pulmonary edema", "chf", "congestive heart failure",
                    "volume overload"],
        "scenario": "ahf",
        "targets": [
            {"organ": "heart", "variable": "preload", "direction": "low"},
            {"organ": "lung", "variable": "pulmonary_edema", "direction": "low"},
            {"organ": "*", "variable": "fluid_overload", "direction": "low"},
        ],
    },
    "hypovolemic shock": {
        "label": "Hypovolemia / shock",
        "aliases": ["shock", "dehydration", "hypovolemia", "hemorrhagic shock"],
        "scenario": "shock",
        "targets": [
            {"organ": "*", "variable": "hypovolemia", "direction": "low"},
            {"organ": "*", "variable": "perfusion", "direction": "high"},
        ],
    },
    "hypoxia": {
        "label": "Hypoxia / respiratory failure",
        "aliases": ["respiratory failure", "hypoxemia", "low oxygen", "desaturation"],
        "scenario": "hypoxia",
        "targets": [
            {"organ": "*", "variable": "hypoxia", "direction": "low"},
            {"organ": "lung", "variable": "gas_exchange", "direction": "high"},
        ],
    },
    "hyperglycemia": {
        "label": "Hyperglycemia / DKA",
        "aliases": ["dka", "diabetic ketoacidosis", "high blood sugar", "diabetes"],
        "scenario": "hyperglycemia",
        "targets": [
            {"organ": "pancreas", "variable": "blood_glucose_elevation", "direction": "low"},
        ],
    },
}

# Reverse alias index, built once.
_ALIAS_TO_CANON: Dict[str, str] = {}
for _canon, _spec in INDICATIONS.items():
    _ALIAS_TO_CANON[_canon] = _canon
    for _a in _spec["aliases"]:
        _ALIAS_TO_CANON[_a.lower()] = _canon


def list_indications() -> List[Dict[str, str]]:
    """For the UI dropdown: [{value, label}, ...] sorted by label."""
    rows = [{"value": k, "label": v["label"]} for k, v in INDICATIONS.items()]
    return sorted(rows, key=lambda r: r["label"])


def resolve(indication: str) -> str | None:
    """Normalize a free-text indication to a canonical key, or None if unknown."""
    if not indication:
        return None
    q = indication.strip().lower()
    if q in _ALIAS_TO_CANON:
        return _ALIAS_TO_CANON[q]
    # loose contains-match as a fallback
    for alias, canon in _ALIAS_TO_CANON.items():
        if alias in q or q in alias:
            return canon
    return None


def targets_for(indication: str) -> Tuple[List[Dict], str, str]:
    """
    Returns (targets, canonical_indication, scenario).
    Raises KeyError-free: unknown indication -> ([], indication, "").
    """
    canon = resolve(indication)
    if not canon:
        return [], indication, ""
    spec = INDICATIONS[canon]
    return spec["targets"], canon, spec.get("scenario", canon)


if __name__ == "__main__":
    # Self-check: every indication must produce >=1 candidate drug.
    import reasoning_engine as RE
    drugs = RE._load("drugs_pkpd.json")["drugs"]
    print(f"Checking {len(INDICATIONS)} indications against {len(drugs)} drugs...\n")
    bad = 0
    for canon in INDICATIONS:
        targets, _, _ = targets_for(canon)
        cands = RE.mechanism_candidates(targets, drugs)
        names = [c["drug"] for c in cands]
        flag = "" if names else "  ❌ NO MATCH"
        print(f"{canon:28s} -> {len(names):2d}  {', '.join(names[:6])}{flag}")
        if not names:
            bad += 1
    print(f"\n{'ALL OK' if not bad else str(bad) + ' INDICATION(S) WITH NO DRUGS'}")
