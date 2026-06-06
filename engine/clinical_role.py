"""
clinical_role.py — clinical-role gate.

Mechanism match alone must not decide ranking. A drug can mechanistically lower
bronchospasm yet be the wrong first-line choice for the scenario (e.g. epinephrine
for plain asthma). This gate assigns a clinical role + rank_priority so the engine
sorts by appropriateness first, mechanism strength only as a tiebreaker.

Lower rank_priority = higher in the list.
"""
from __future__ import annotations
from typing import Dict, List

ANAPHYLAXIS_FEATURES = {
    "hives", "urticaria", "angioedema", "lip swelling", "tongue swelling",
    "throat tightness", "stridor", "known allergen exposure", "allergen exposure",
}


def _is_asthma(scenario: str) -> bool:
    s = (scenario or "").lower()
    return "asthma" in s or "bronchospasm" in s


def detect_anaphylaxis_pattern(symptoms: List[str], patient_flags: List[str]) -> bool:
    sset = {s.lower() for s in symptoms}
    fset = {f.lower() for f in patient_flags}
    return bool(sset & ANAPHYLAXIS_FEATURES) and (
        "bronchospasm" in sset or "wheezing" in sset or "hypotension" in fset)


def assign_clinical_role(drug_name: str, scenario: str,
                         patient_flags: List[str], symptoms: List[str]) -> Dict:
    drug = (drug_name or "").lower()

    if _is_asthma(scenario):
        if drug == "albuterol":
            return {"role": "primary_candidate", "rank_priority": 1,
                    "label": "PRIMARY CANDIDATE",
                    "reason": "Label-supported first-line candidate for bronchospasm; "
                              "dose and safety require official-label ingestion."}
        if drug == "epinephrine":
            if detect_anaphylaxis_pattern(symptoms, patient_flags):
                return {"role": "emergency_context_candidate", "rank_priority": 0,
                        "label": "EMERGENCY CONTEXT CANDIDATE",
                        "reason": "Anaphylaxis pattern detected; verify official label and "
                                  "emergency safety pathway."}
            return {"role": "conditional_candidate", "rank_priority": 20,
                    "label": "CONDITIONAL CANDIDATE · NOT PROMOTED",
                    "reason": "Mechanism match exists, but no anaphylaxis pattern is established; "
                              "hypotension alone does not establish anaphylaxis or septic shock."}

    return {"role": "mechanism_candidate", "rank_priority": 10,
            "label": "MECHANISM CANDIDATE",
            "reason": "Requires official-label and clinical-context evaluation."}


if __name__ == "__main__":
    for sym in ([], ["wheezing", "hives"]):
        print(f"symptoms={sym}")
        for d in ("albuterol", "epinephrine"):
            r = assign_clinical_role(d, "asthma", ["hypotension", "tachycardia"], sym)
            print(f"  {d:12s} -> {r['label']} (priority {r['rank_priority']})")
