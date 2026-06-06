"""
drug_identity.py — canonical drug identity & alias resolution.

Stops the same drug appearing as multiple candidate cards (e.g. "nitroglycerin"
and "ntg"), which would otherwise duplicate label lookups and pollute ranking.
"""
from __future__ import annotations
from typing import Dict, List

DRUG_ALIAS_MAP: Dict[str, str] = {
    "ntg": "nitroglycerin",
    "nitrostat": "nitroglycerin",
    "ventolin": "albuterol",
    "proair": "albuterol",
    "benadryl": "diphenhydramine",
    "epi pen": "epinephrine",
    "epipen": "epinephrine",
    "ppi": "omeprazole",
    "triptan": "sumatriptan",
}

# Canonical identities (extend as labels are ingested).
CANONICAL: Dict[str, Dict] = {
    "nitroglycerin": {"canonical_id": "nitroglycerin_sublingual", "generic_name": "nitroglycerin",
                      "aliases": ["ntg", "nitrostat"], "routes": ["sublingual"], "formulations": ["tablet"]},
    "aspirin": {"canonical_id": "aspirin_oral", "generic_name": "aspirin",
                "aliases": [], "routes": ["oral"], "formulations": ["tablet"]},
    "heparin": {"canonical_id": "heparin_injection", "generic_name": "heparin",
                "aliases": [], "routes": ["intravenous", "subcutaneous"], "formulations": ["solution"]},
    "albuterol": {"canonical_id": "albuterol_inhaled", "generic_name": "albuterol",
                  "aliases": ["ventolin", "proair"], "routes": ["inhalation"], "formulations": ["aerosol"]},
}


def canonicalize_drug_name(name: str) -> str:
    n = (name or "").lower().strip()
    return DRUG_ALIAS_MAP.get(n, n)


def aliases_for(canonical: str) -> List[str]:
    return CANONICAL.get(canonical, {}).get("aliases", [])


def dedupe_to_canonical(drug_map: Dict[str, Dict]) -> Dict[str, Dict]:
    """Collapse a name->data map to canonical names, preferring the canonical entry."""
    out: Dict[str, Dict] = {}
    seen_aliases: Dict[str, List[str]] = {}
    for name, data in drug_map.items():
        canon = canonicalize_drug_name(name)
        seen_aliases.setdefault(canon, [])
        if name != canon:
            seen_aliases[canon].append(name)
        # prefer the entry whose key already equals the canonical name
        if canon not in out or name == canon:
            out[canon] = data
    for canon in out:
        merged = sorted(set(seen_aliases.get(canon, []) + aliases_for(canon)))
        out[canon] = {**out[canon], "_aliases": merged}
    return out


if __name__ == "__main__":
    for n in ["NTG", "Nitrostat", "ventolin", "aspirin", "heparin"]:
        print(f"{n:12s} -> {canonicalize_drug_name(n)}")
