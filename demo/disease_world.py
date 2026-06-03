"""
disease_world.py — the LOCAL knowledge "wiki": the disease side of the shared
state-variable coordinate system.

For an indication we look up the real disease model in the organ JSONs (heart.json,
lung.json, ...) and return its perturbations (variable, delta, cause) and the
symptoms its derivation_rules produce. This is the "disease pushes the body off
equilibrium" half; the drug engine supplies the "drug pushes it back" half. Showing
both in one graph is the neuro-symbolic world model the demo is about.

When an indication has no curated disease entry, we SYNTHESIZE a minimal disease
picture by inverting the treatment targets (if treatment pushes a variable down,
the disease pushed it up) so the graph is never empty. Such entries are marked
source="inferred".
"""
from __future__ import annotations
import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import case_targets

# Organ JSONs (heart.json, lung.json, ...) live in the project root, one level up
# from this demo/ folder.
_DATA = Path(__file__).resolve().parent.parent / "data"

# canonical indication -> (organ json file, disease key inside its "diseases")
INDICATION_TO_DISEASE: Dict[str, tuple] = {
    "acute coronary syndrome": ("heart.json", "Heart Attack/MI"),
    "stable angina": ("heart.json", "Stable Angina"),
    "hypertension": ("heart.json", "Chronic Essential Hypertension"),
    "acute heart failure": ("heart.json", "Congestive Heart Failure"),
    "asthma": ("lung.json", "Asthma Exacerbation"),
    "venous thromboembolism": ("lung.json", "Acute Pulmonary Embolism"),
    "hypoxia": ("lung.json", "Acute Pulmonary Edema"),
    "gerd": ("gi.json", "Gastroesophageal Reflux Disease"),
    "migraine": ("brain.json", "Migraine"),
    "bacterial infection": ("blood.json", "Sepsis"),
    "hyperglycemia": ("pancreas.json", "Diabetic Ketoacidosis"),
    "anaphylaxis": ("skin.json", "Acute Urticaria"),
}


@lru_cache(maxsize=32)
def _load_organ(fname: str) -> dict:
    p = _DATA / fname
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}


def _symptoms_for(organ: dict, disease_key: str, limit: int = 6) -> List[str]:
    """Run the organ's derivation_rules against this disease's perturbations."""
    diseases = organ.get("diseases", {})
    dz = diseases.get(disease_key, {})
    state = {p["variable"]: p.get("delta", 0.0) for p in dz.get("perturbations", [])}
    out = []
    for rule in organ.get("derivation_rules", []):
        cond = rule.get("condition", {})
        var, op, thr = cond.get("state"), cond.get("op"), cond.get("threshold", 0)
        val = state.get(var)
        if val is None:
            continue
        hit = (op == ">" and val > thr) or (op == "<" and val < thr) or \
              (op == ">=" and val >= thr) or (op == "<=" and val <= thr)
        if hit:
            out.append(rule.get("symptom"))
        if len(out) >= limit:
            break
    return out


def for_indication(indication: str) -> Dict:
    """
    Returns the disease-side world model for an indication:
      {disease, description, source, source_file, perturbations[], symptoms[]}
    perturbations: [{variable, delta, direction, cause}]
    """
    canon = case_targets.resolve(indication) or indication
    label = next((i["label"] for i in case_targets.list_indications()
                  if i["value"] == canon), canon)

    if canon in INDICATION_TO_DISEASE:
        fname, dz_key = INDICATION_TO_DISEASE[canon]
        organ = _load_organ(fname)
        dz = organ.get("diseases", {}).get(dz_key, {})
        if dz:
            perts = []
            for p in dz.get("perturbations", []):
                delta = p.get("delta", 0.0)
                perts.append({"variable": p["variable"], "delta": delta,
                              "direction": "high" if delta >= 0 else "low",
                              "cause": p.get("cause", "")})
            return {"disease": dz_key, "indication_label": label,
                    "description": dz.get("description", ""),
                    "source": "curated", "source_file": fname,
                    "perturbations": perts,
                    "symptoms": [s for s in _symptoms_for(organ, dz_key) if s]}

    # Fallback: synthesize from inverted treatment targets.
    targets, _, _ = case_targets.targets_for(canon)
    perts = []
    for t in targets:
        # disease pushes the variable OPPOSITE to the treatment goal
        dz_dir = "high" if t["direction"] == "low" else "low"
        perts.append({"variable": t["variable"], "delta": 0.6 if dz_dir == "high" else -0.6,
                      "direction": dz_dir, "cause": "inferred from treatment goal"})
    return {"disease": label, "indication_label": label, "description": "",
            "source": "inferred", "source_file": None,
            "perturbations": perts, "symptoms": []}


if __name__ == "__main__":
    for ind in ("acute coronary syndrome", "asthma", "gerd", "acute pain"):
        w = for_indication(ind)
        print(f"\n[{ind}] -> {w['disease']} ({w['source']}, {w['source_file']})")
        for p in w["perturbations"]:
            arrow = "↑" if p["direction"] == "high" else "↓"
            print(f"   {p['variable']} {arrow} ({p['delta']:+}) {p['cause'][:40]}")
        print("   symptoms:", w["symptoms"])
