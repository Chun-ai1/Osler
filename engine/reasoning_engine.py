"""
reasoning_engine.py — 5-layer Pharmacology Reasoning Engine orchestrator.

  pathologic state(s) → mechanism → official label → patient safety gate → explainable output

Now supports MULTIPLE target states per scenario (e.g. ACS = ischemia burden ↓ +
platelet aggregation ↓ + clot propagation risk ↓), canonical alias dedup, per-drug
mechanism chains, and label-validation-aware dose gating.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from patient_profile import PatientProfile
from drug_safety_gate import evaluate
from drug_identity import canonicalize_drug_name, dedupe_to_canonical, CANONICAL
from clinical_role import assign_clinical_role

_ETYPE_WEIGHT = {"primary": 1.0, "derived": 0.5, "symptom_relief": 0.3, "": 0.4}

_HERE = Path(__file__).parent
# Data files live in the project-root data/ folder (after the repo reorg). Fall back
# to this module's own folder so the engine also works if data sits beside it.
_DATA = _HERE.parent / "data" if (_HERE.parent / "data").exists() else _HERE
_RANK = {"ok": 0, "caution": 1, "adjust": 2, "insufficient_data": 3, "avoid": 4, "emergency": 5}
_ETYPE = {"primary": 0, "derived": 1, "symptom_relief": 2, "": 3}


def _load(p): return json.loads((_DATA / p).read_text(encoding="utf-8", errors="replace"))
def _arrow(direction): return "\u2193" if direction == "low" else "\u2191"
def _friendly(v): return v.replace("_", " ")


def _eff_direction(e): return e.get("direction") or ("low" if e.get("max_delta", 0) < 0 else "high")


def _mechanism_chain(drug_data: Dict[str, Any]) -> str:
    effs = sorted(drug_data.get("state_effects", []),
                  key=lambda e: _ETYPE.get(e.get("effect_type", ""), 3))
    return " \u2192 ".join(f"{_friendly(e['variable'])} {_arrow(_eff_direction(e))}" for e in effs)


def _norm_targets(target: Union[Dict, List[Dict]]) -> List[Dict]:
    return target if isinstance(target, list) else [target]


def mechanism_candidates(targets: List[Dict], drugs_pkpd: Dict[str, Any]) -> List[Dict[str, Any]]:
    canon_map = dedupe_to_canonical(drugs_pkpd)
    hits = []
    for name, d in canon_map.items():
        matched = []
        score = 0.0
        for eff in d.get("state_effects", []):
            md = eff.get("max_delta", 0.0)
            for t in targets:
                if eff["variable"] == t["variable"] and eff["organ"] in (t["organ"], "*") \
                   and md != 0 and ((md < 0) == (t["direction"] == "low")):
                    etype = eff.get("effect_type", "primary")
                    matched.append({"target": f"{_friendly(t['variable'])} {_arrow(t['direction'])}",
                                    "effect_type": etype})
                    score += abs(md) * _ETYPE_WEIGHT.get(etype, 0.4)
        if matched:
            hits.append({"drug": name, "data": d, "aliases": d.get("_aliases", []),
                         "matched_targets": matched, "mechanism_chain": _mechanism_chain(d),
                         "mechanism_score": round(score, 3)})
    return hits


def _indication_supported(entry, indication):
    if not entry or not indication:
        return None
    ind = indication.lower()
    inds = entry.get("indications", [])
    if inds:  # structured list is authoritative; avoids matching negations in free text
        return any(ind in i.lower() or i.lower() in ind for i in inds)
    text = entry.get("label_sections", {}).get("indications_and_usage", "").lower()
    return (ind in text) if text.strip() else None


def _final_answer(g, ind_ok) -> str:
    if g["label_status"] == "not_loaded":
        return "Mechanism match only — load the official label before any dose."
    if g["decision"] == "avoid":
        return "Do NOT recommend — contraindication/allergy conflict; clinician review required."
    if g["label_status"] == "extracted_unverified":
        return "Label loaded, but the dose rule is not human-validated — show label text only; patient-specific dosing blocked."
    if ind_ok is False:
        return "Off-label for this indication — guideline/clinician review required."
    if g["decision"] == "adjust":
        return "Show validated label dose WITH patient-specific adjustment note; clinician confirms."
    if g["decision"] == "caution":
        return "Validated label dose may be shown WITH warning; clinician confirms."
    return "Validated label dose may be shown; clinician confirms."


def _result_level(g) -> str:
    if g["label_status"] == "not_loaded":
        return "Mechanism match only · clinical label not loaded"
    if g["decision"] == "avoid":
        return "Contraindicated · do not recommend"
    if g["patient_specific_dose_allowed"]:
        return "Validated label dose available"
    if not g.get("rule_validated", False):
        return "Label loaded · dose blocked pending validation"
    return "Dose blocked by safety gate"


def recommend(patient: PatientProfile, target: Union[Dict, List[Dict]], indication: Optional[str],
              drugs_pkpd: Dict[str, Any], clinical: Dict[str, Any],
              scenario: Optional[str] = None) -> Dict[str, Any]:
    targets = _norm_targets(target)
    scenario = scenario or (indication or "")
    patient_flags = [f["flag"] for f in patient.flags()]
    clinical_canon = {canonicalize_drug_name(k): v for k, v in clinical.items()}
    results = []
    for cand in mechanism_candidates(targets, drugs_pkpd):
        drug = cand["drug"]
        entry = clinical_canon.get(drug)
        ind_ok = _indication_supported(entry, indication)
        g = evaluate(patient, entry, indication)
        role = assign_clinical_role(drug, scenario, patient_flags, patient.symptoms)
        rule = g.get("dose_rule") or {}
        dose_text = rule.get("dose_text_verbatim") if g["dose_verbatim_displayable"] else None
        val = rule.get("validation", {}) if g["patient_specific_dose_allowed"] else {}
        structured = rule.get("structured_extraction", {}) if g["patient_specific_dose_allowed"] else {}
        results.append({
            "drug": drug,
            "aliases": cand["aliases"],
            "clinical_role": role,
            "mechanism_score": cand["mechanism_score"],
            "mechanism_chain": cand["mechanism_chain"],
            "matched_targets": cand["matched_targets"],
            "indication_support": ("label-supported" if ind_ok is True else
                                   "not in label / off-label" if ind_ok is False else
                                   "unknown (no label loaded)"),
            "loaded_product_type": (entry.get("identifiers", {}).get("product_type") if entry else None),
            "result_level": _result_level(g),
            "label_status": g["label_status"],
            "dose": {"verbatim": dose_text, "route": rule.get("route"),
                     "source_section": rule.get("source_section"),
                     "patient_specific_allowed": g["patient_specific_dose_allowed"],
                     "structured": structured,
                     "validated_by": val.get("validated_by"), "validated_at": val.get("validated_at")},
            "safety": {"decision": g["decision"], "risk_level": g["risk_level"],
                       "reasons": g["reasons"], "missing_patient_data": g["missing_patient_data"]},
            "evidence": ([{"source": entry["source"].get("primary"),
                           "set_id": entry["source"].get("set_id"),
                           "retrieved_at": entry["source"].get("retrieved_at") or entry["source"].get("effective_date")}]
                         if entry and "source" in entry else []),
            "faers_signals": (entry.get("faers_signals", []) if entry else []),
            "final_answer": _final_answer(g, ind_ok),
        })
    # Rank by clinical appropriateness first, mechanism strength only as tiebreaker.
    results.sort(key=lambda r: (r["clinical_role"]["rank_priority"], -r["mechanism_score"]))
    return {
        "target_states": [f"{_friendly(t['variable'])} {_arrow(t['direction'])}" for t in targets],
        "indication": indication,
        "patient": {"age": patient.age, "renal_label": patient.renal_label(),
                    "renal": patient.renal, "allergies": patient.allergies,
                    "meds": patient.current_medications, "flags": [f["flag"] for f in patient.flags()]},
        "candidates": results,
        "_disclaimer": ("DECISION SUPPORT ONLY. Doses are label-extracted and require human validation "
                        "before any patient-specific recommendation. FAERS data is signal, not causation. "
                        "Final clinical decision requires a licensed clinician."),
    }


if __name__ == "__main__":
    drugs_pkpd = _load("drugs_pkpd.json")["drugs"]
    clinical = _load("drug_clinical_data.json")["drugs"]
    acs_targets = [
        {"organ": "heart", "variable": "myocardial_oxygen_demand", "direction": "low"},
        {"organ": "blood", "variable": "platelet_aggregation", "direction": "low"},
        {"organ": "blood", "variable": "clot_propagation_risk", "direction": "low"},
    ]
    patient = PatientProfile(age=50, weight_kg=80, egfr=84, vitals={"sbp": 120, "heart_rate": 135})
    out = recommend(patient, acs_targets, "acute coronary syndrome", drugs_pkpd, clinical)
    print("patient:", out["patient"]["renal_label"], "| flags:", out["patient"]["flags"])
    print("targets:", out["target_states"])
    for r in out["candidates"]:
        al = f" (aliases: {', '.join(r['aliases'])})" if r["aliases"] else ""
        print(f"\n• {r['drug']}{al} — {r['result_level']}")
        print(f"    mechanism: {r['mechanism_chain']}")
        print(f"    matched: {[m['target']+'/'+m['effect_type'] for m in r['matched_targets']]}")
        print(f"    decision={r['safety']['decision']}  →  {r['final_answer']}")
