"""
drug_profile.py — drug-centric view (flips disease→drug into drug→everything).

Given a drug name, assemble a monograph from the layers we already have:
  • body-state effects   ← drugs_pkpd state_effects   (internal mechanism map, estimated)
  • how it's used / dose  ← label dosage_and_administration / dose_rules (FDA SPL)
  • adverse effects       ← label adverse_reactions (FDA SPL)
  • bad real-world record ← FAERS signals (openFDA drug/event, signal only)
  • who should not use it  ← label contraindications + warnings + specific populations + interactions
Each section keeps its source so the renderer can cite + hyperlink it.
"""
from __future__ import annotations
from typing import Any, Dict, List
from drug_identity import canonicalize_drug_name, dedupe_to_canonical, CANONICAL, aliases_for


def available_drugs(drugs_pkpd: Dict[str, Any]) -> List[str]:
    return sorted(dedupe_to_canonical(drugs_pkpd).keys())


def drug_classes(clinical: Dict[str, Any], top: int = 120) -> List[tuple]:
    """Distinct FDA pharm_class values across the loaded labels, with counts."""
    from collections import Counter
    cnt = Counter()
    for d in clinical.values():
        for cls in (d.get("identifiers", {}).get("pharm_class") or []):
            if cls:
                cnt[cls] += 1
    return sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[:top]


def browse(clinical: Dict[str, Any], class_filter: str = "", indication: str = "",
           limit: int = 300) -> List[Dict[str, Any]]:
    """Find drugs by FDA pharm_class and/or by label indication (disease → drug, label-based)."""
    cf = (class_filter or "").lower().strip()
    iq = (indication or "").lower().strip()
    rows = []
    for name, d in clinical.items():
        ident = d.get("identifiers", {})
        classes = " ".join(ident.get("pharm_class") or []).lower()
        inds_list = d.get("indications") or []
        inds = " ".join(inds_list).lower() + " " + \
            (d.get("label_sections", {}).get("indications_and_usage", "") or "").lower()
        if cf and cf not in classes:
            continue
        if iq and iq not in inds:
            continue
        rows.append({"drug": name,
                     "pharm_class": (ident.get("pharm_class") or [None])[0],
                     "indications": ", ".join(inds_list[:3])})
    return sorted(rows, key=lambda r: r["drug"])[:limit]


def build_profile(drug_name: str, drugs_pkpd: Dict[str, Any], clinical: Dict[str, Any]) -> Dict[str, Any]:
    canon = canonicalize_drug_name(drug_name)
    mech = dedupe_to_canonical(drugs_pkpd).get(canon, {})
    clinical_canon = {canonicalize_drug_name(k): v for k, v in clinical.items()}
    entry = clinical_canon.get(canon) or {}
    ls = entry.get("label_sections", {}) or {}
    ident = entry.get("identifiers", {}) or {}
    src = entry.get("source", {}) or {}

    label_loaded = bool(ls) and entry.get("validation_status") not in (None, "not_loaded")
    return {
        "drug": canon,
        "aliases": mech.get("_aliases") or aliases_for(canon),
        "drug_class": mech.get("drug_class"),
        "rationale": mech.get("rationale"),
        "label_loaded": label_loaded,
        "validation_status": entry.get("validation_status", "not_loaded"),
        "source": {"set_id": src.get("set_id"), "primary": src.get("primary"),
                   "retrieved_at": src.get("retrieved_at") or src.get("effective_date"),
                   "product_type": ident.get("product_type")},
        "identity": {"canonical_id": ident.get("canonical_id") or CANONICAL.get(canon, {}).get("canonical_id", canon),
                     "rxnorm_cui": ident.get("rxnorm_cui"),
                     "routes": ident.get("route") or CANONICAL.get(canon, {}).get("routes", [])},
        "indications": entry.get("indications", []) or [],
        "indications_text": ls.get("indications_and_usage"),               # 使用方法 = what it treats
        # parenthetical items the user asked for:
        "administration_text": ls.get("dosage_and_administration"),       # how it is given (folded into Dosage)
        "dose_rules": entry.get("dose_rules", []),                         # 劑量 = how much
        "adverse_reactions_text": ls.get("adverse_reactions"),            # 副作用
        "body_state_changes": mech.get("state_effects", []),              # 會改變哪些人體數據
        "faers_signals": entry.get("faers_signals", []),                  # 不好的紀錄
        "contraindications_text": ls.get("contraindications"),            # 哪些病人不建議
        "warnings_text": ls.get("warnings_and_precautions"),
        "special_populations_text": ls.get("use_in_specific_populations"),
        "interactions_text": ls.get("drug_interactions"),
    }


if __name__ == "__main__":
    import json
    from pathlib import Path
    dp = json.loads(Path("drugs_pkpd.json").read_text())["drugs"]
    cl = json.loads(Path("drug_clinical_data.json").read_text())["drugs"]
    print("available:", available_drugs(dp))
    p = build_profile("ventolin", dp, cl)
    print("\nventolin → canonical:", p["drug"], "| aliases:", p["aliases"], "| label_loaded:", p["label_loaded"])
    print("body-state changes:", [(e["organ"], e["variable"], e.get("effect_type")) for e in p["body_state_changes"]])
    print("faers:", [s["event"] for s in p["faers_signals"][:3]])
