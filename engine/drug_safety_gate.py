"""
drug_safety_gate.py — Safety / contraindication / adverse-event layer.

Decisions come from MATCHING patient facts against SOURCED label fields. Two
separate axes are returned:
  • decision      — clinical safety ladder (ok/caution/adjust/avoid/emergency/insufficient_data)
  • dose gating   — label_status + whether a PATIENT-SPECIFIC dose may be shown

A patient-specific dose is allowed only when the label is loaded AND its dose rule
is human_validated AND safety is not blocking. Raw label text is never auto-promoted
to a recommendation.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from patient_profile import PatientProfile

_RANK = {"ok": 0, "caution": 1, "adjust": 2, "insufficient_data": 3, "avoid": 4, "emergency": 5}
_RISK = {"ok": "low", "caution": "moderate", "adjust": "moderate",
         "avoid": "high", "emergency": "critical", "insufficient_data": "unknown"}


def _worst(a, b): return a if _RANK[a] >= _RANK[b] else b
def _sec(entry, name): return (entry.get("label_sections", {}) or {}).get(name, "") or ""


def _match_dose_rule(entry, indication) -> Optional[Dict[str, Any]]:
    rules = entry.get("dose_rules", []) or []
    if not rules:
        return None
    if indication:
        for r in rules:
            if (r.get("indication") or "").lower() in indication.lower() or \
               indication.lower() in (r.get("indication") or "").lower():
                return r
    return rules[0]


def evaluate(patient: PatientProfile, entry: Optional[Dict[str, Any]],
             indication: Optional[str] = None) -> Dict[str, Any]:
    if not entry:
        return {"decision": "insufficient_data", "risk_level": "unknown", "label_status": "not_loaded",
                "patient_specific_dose_allowed": False, "dose_verbatim_displayable": False,
                "dose_rule": None, "missing_patient_data": patient.missing_core(),
                "reasons": [{"type": "no_clinical_data", "decision": "insufficient_data",
                             "message": "official label not loaded; contraindications, interactions and "
                                        "adverse reactions cannot yet be evaluated", "source": None}]}

    drug = entry.get("identifiers", {}).get("generic_name", "drug")
    label_status = entry.get("validation_status", "extracted_unverified")
    contra = _sec(entry, "contraindications").lower()
    warns = (_sec(entry, "warnings_and_precautions") + " " + _sec(entry, "boxed_warning")).lower()
    inter = _sec(entry, "drug_interactions").lower()
    specpop = (_sec(entry, "use_in_specific_populations") + " " + _sec(entry, "pregnancy")).lower()
    substance = " ".join(entry.get("identifiers", {}).get("substance_name", [])).lower()
    pharm = " ".join(entry.get("identifiers", {}).get("pharm_class", [])).lower()

    decision = "ok"
    reasons: List[Dict[str, Any]] = []

    for allergy in patient.allergies:
        a = allergy.lower()
        if a and (a in contra or a in substance or a in pharm or a in drug.lower()):
            decision = _worst(decision, "avoid")
            reasons.append({"type": "contraindication", "decision": "avoid",
                            "message": f"patient allergy '{allergy}' matches drug/contraindication",
                            "source": "FDA label \u2014 Contraindications"})
    for med in patient.current_medications:
        if med and med in inter:
            decision = _worst(decision, "caution")
            reasons.append({"type": "drug_interaction", "decision": "caution",
                            "message": f"current medication '{med}' appears in the drug-interaction section",
                            "source": "FDA label \u2014 Drug Interactions"})
    for f in patient.flags():
        hay = contra + " " + warns + " " + specpop
        if any(kw in hay for kw in f["keywords"]):
            sev = "avoid" if any(kw in contra for kw in f["keywords"]) else "caution"
            decision = _worst(decision, sev)
            reasons.append({"type": "patient_factor", "decision": sev,
                            "message": f"{f['flag']} ({f['detail']}) is referenced in the label safety text",
                            "source": "FDA label \u2014 Contraindications/Warnings/Specific Populations"})
    if patient.renal_dose_review_needed() and any(w in (specpop + warns + contra)
                                                  for w in ("renal", "creatinine clearance", "kidney", "dialysis")):
        decision = _worst(decision, "adjust")
        reasons.append({"type": "organ_function", "decision": "adjust",
                        "message": f"{patient.renal_label()}; label discusses renal use",
                        "source": "FDA label \u2014 Use in Specific Populations"})

    dose_rule = _match_dose_rule(entry, indication)
    rule_validated = bool(dose_rule and (dose_rule.get("validation", {}).get("validated")
                                         or dose_rule.get("requires_human_validation") is False))
    dose_verbatim_displayable = bool(dose_rule and dose_rule.get("dose_text_verbatim")) \
        and label_status in ("extracted_unverified", "human_validated")
    patient_specific_dose_allowed = (decision in ("ok", "caution", "adjust")
                                     and dose_rule is not None and rule_validated
                                     and bool(dose_rule.get("dose_text_verbatim")))

    if dose_rule and not rule_validated and decision != "avoid":
        reasons.append({"type": "validation", "decision": "caution",
                        "message": "dose rule extracted from label but NOT human-validated; "
                                   "patient-specific dosing is blocked",
                        "source": "validation_status=extracted_unverified"})

    return {"decision": decision, "risk_level": _RISK[decision], "label_status": label_status,
            "patient_specific_dose_allowed": patient_specific_dose_allowed,
            "rule_validated": rule_validated,
            "dose_verbatim_displayable": dose_verbatim_displayable, "dose_rule": dose_rule,
            "missing_patient_data": patient.missing_core(),
            "reasons": reasons or [{"type": "none", "message": "no blocking factor matched in the label",
                                    "source": None}]}
