"""
weak_labeler.py — Phase 1 of the medical-JEPA roadmap.

A standalone, deterministic tool that turns ANY patient observation
(symptoms, labs, vitals) into Osler-derived organ-state estimates, each
tagged with a source and confidence.

This is the labeling logic from osler_to_trajectory.py, lifted out so it can
be reused independently — e.g. to enrich the live chat panel, to label a
MIMIC timepoint, or to build a training target. It has ONE job: observation →
weak state labels. No trajectory structure, no ML.

Three label sources, in descending confidence:
  - lab_direct   : an abnormal lab maps to a state (lab_integration.json)
  - vital_direct : a vital sign threshold maps to a state
  - mechanism    : Osler's reason() infers states from symptoms

Determinism: same input → same labels, always. No randomness, no model.

Usage:
    from weak_labeler import label_observation
    labels = label_observation(
        symptoms=["cough", "fever"],
        labs={"WBC": 15.2}, lab_flags={"WBC": "high"},
        vitals={"spo2": 92},
        context={"age": 64, "sex": "male"},
    )
"""
from __future__ import annotations
import sys, os, io, json

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for p in (_REPO, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

_LAB_MAP_PATH = os.path.join(_REPO, "medical_knowledge", "state_models", "lab_integration.json")


# ──────────────────────────────────────────────────────────────────────
# Lab inversion
# ──────────────────────────────────────────────────────────────────────

_LAB_INV = None
def _lab_inversion() -> dict:
    global _LAB_INV
    if _LAB_INV is not None:
        return _LAB_INV
    with open(_LAB_MAP_PATH, encoding="utf-8") as f:
        mapping = json.load(f).get("lab_to_state_mapping", {})
    inv = {}
    for lab, entries in mapping.items():
        if lab.startswith("_") or not isinstance(entries, list):
            continue
        rules = []
        for e in entries:
            if not isinstance(e, dict) or not e.get("state"):
                continue
            organ = e.get("organ")
            full = f"{organ}.{e['state']}" if organ else f"systemic.{e['state']}"
            if e.get("high_means_perturb") is not None:
                rules.append((full, "high", float(e["high_means_perturb"])))
            if e.get("low_means_perturb") is not None:
                rules.append((full, "low", float(e["low_means_perturb"])))
        if rules:
            inv[lab.lower()] = rules
    _LAB_INV = inv
    return inv


# ──────────────────────────────────────────────────────────────────────
# Vital rules
# ──────────────────────────────────────────────────────────────────────

_VITAL_RULES = {
    "spo2":   ("<", 94,   ("lung.gas_exchange",      "low",  0.6, 0.9, "hypoxemia")),
    "rr":     (">", 24,   ("lung.gas_exchange",      "low",  0.4, 0.7, "tachypnea")),
    "hr":     (">", 100,  ("heart.sympathetic_drive","high", 0.5, 0.6, "tachycardia")),
    "bp_sys": ("<", 90,   ("heart.cardiac_output",   "low",  0.6, 0.8, "hypotension")),
    "temp":   (">", 38.0, ("blood.infection_load",   "high", 0.5, 0.6, "fever")),
}


# ──────────────────────────────────────────────────────────────────────
# Osler mechanism
# ──────────────────────────────────────────────────────────────────────

_NX = None
def _nexus():
    global _NX
    if _NX is None:
        _so = sys.stdout
        sys.stdout = io.StringIO()
        from nexus_engine.nexus_medical import NexusMedical
        _NX = NexusMedical()
        _NX.load_knowledge()
        sys.stdout = _so
    return _NX


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────

def label_from_labs(labs: dict, lab_flags: dict | None = None) -> list:
    """[{state, value, direction, source, confidence, rationale}, ...]"""
    inv = _lab_inversion()
    out = []
    for lab, val in (labs or {}).items():
        rules = inv.get(str(lab).lower())
        if not rules:
            continue
        flag = (lab_flags or {}).get(lab, "high")
        for full_state, direction, weight in rules:
            if direction != flag:
                continue
            out.append({
                "state": full_state, "value": round(weight, 3), "direction": direction,
                "source": "lab_direct",
                "confidence": round(min(0.95, 0.6 + weight * 0.35), 3),
                "rationale": f"{lab}={val} ({flag})",
            })
    return out


def label_from_vitals(vitals: dict) -> list:
    out = []
    for v, (cmp_, thr, (state, direction, value, conf, why)) in _VITAL_RULES.items():
        if v not in (vitals or {}):
            continue
        try:
            x = float(vitals[v])
        except (TypeError, ValueError):
            continue
        hit = (x < thr) if cmp_ == "<" else (x > thr)
        if hit:
            out.append({
                "state": state, "value": value, "direction": direction,
                "source": "vital_direct", "confidence": conf,
                "rationale": f"{v}={x} {why}",
            })
    return out


def label_from_mechanism(symptoms: list, context: dict | None = None, top_k: int = 1) -> list:
    nx = _nexus()
    out, seen = [], set()
    try:
        res = nx.reason(symptoms or [], context=context or {})
    except Exception:
        return out
    for d in res.get("diagnoses", [])[:top_k]:
        for ms in d.get("_state_simulation", {}).get("matched_states", []):
            state = ms.get("state", "")
            if not state or state in seen:
                continue
            seen.add(state)
            dval, thr = ms.get("disease_value", 0.5), ms.get("user_threshold", 0.4)
            try:
                margin = abs(float(dval) - float(thr))
            except (TypeError, ValueError):
                margin = 0.0
            out.append({
                "state": state,
                "value": round(float(dval), 3) if isinstance(dval, (int, float)) else 0.5,
                "direction": ms.get("direction", "high"),
                "source": "mechanism",
                "confidence": round(min(0.7, 0.3 + margin), 3),
                "rationale": "symptoms: " + ", ".join(ms.get("supporting_symptoms", [])),
            })
    return out


def _dedupe(labels: list) -> list:
    """Same state from multiple sources → keep highest confidence."""
    best = {}
    for l in labels:
        cur = best.get(l["state"])
        if cur is None or l["confidence"] > cur["confidence"]:
            best[l["state"]] = l
    return sorted(best.values(), key=lambda l: (-l["confidence"], l["state"]))


def label_observation(symptoms=None, labs=None, vitals=None,
                      lab_flags=None, context=None) -> list:
    """
    Main entry: observation → deduped, confidence-sorted weak state labels.
    Deterministic. Direct (lab/vital) labels beat mechanism labels on overlap.
    """
    labels = []
    labels += label_from_labs(labs or {}, lab_flags)
    labels += label_from_vitals(vitals or {})
    labels += label_from_mechanism(symptoms or [], context)
    return _dedupe(labels)


if __name__ == "__main__":
    print("=== weak_labeler self-test ===\n")
    cases = [
        ("Pneumonia-ish",
         dict(symptoms=["cough", "fever", "shortness of breath"],
              labs={"WBC": 15.2, "CRP": 110}, lab_flags={"WBC": "high", "CRP": "high"},
              vitals={"spo2": 92, "temp": 38.6}, context={"age": 64, "sex": "male"})),
        ("MI-ish",
         dict(symptoms=["chest pain", "diaphoresis"],
              labs={"troponin": 2.1}, lab_flags={"troponin": "high"},
              vitals={"hr": 104}, context={"age": 58, "sex": "male"})),
    ]
    for name, kw in cases:
        labels = label_observation(**kw)
        print(f"--- {name}: {len(labels)} labels ---")
        for l in labels[:10]:
            print(f"  {l['state']:34s} {l['direction']:4s} v={l['value']:.2f} "
                  f"conf={l['confidence']:.2f} [{l['source']:12s}] ← {l['rationale']}")
        print()
