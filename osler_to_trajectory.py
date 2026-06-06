"""
osler_to_trajectory.py — Phase 0 converter.

Turns an Osler case (symptoms + optional labs/vitals/context) into a
Trajectory in the canonical schema, with weak organ-state labels derived
two ways:

  1. lab_direct / vital_direct — invert lab_integration.json: an abnormal
     lab/vital maps to an organ state with high confidence.
  2. mechanism — run Osler's reason() and read matched_states from the top
     diagnosis: states the symptoms imply, lower confidence.

IMPORTANT HONESTY CONSTRAINT
----------------------------
Cases generated here are tagged source="synthetic_osler". They are for
PRETRAINING / coverage ONLY. They cannot be used to validate clinical
performance, because they're produced by Osler's own deterministic rules —
a model trained only on them learns to mimic Osler, not to surpass it.
Real validation requires real de-identified EHR (e.g. MIMIC-IV).

Usage:
    python3 osler_to_trajectory.py            # builds a demo dataset
    # or import:
    from osler_to_trajectory import case_to_trajectory
"""

from __future__ import annotations
import sys, os, io, json

# allow running from inside medical_jepa/ or repo root
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trajectory_schema import (
    Trajectory, Event, StateSnapshot, StateLabel, write_jsonl, validate_dataset,
)

_LAB_MAP_PATH = os.path.join(
    _REPO, "medical_knowledge", "state_models", "lab_integration.json"
)


# ──────────────────────────────────────────────────────────────────────
# Lab/vital → state inversion (the DIRECT, higher-confidence labels)
# ──────────────────────────────────────────────────────────────────────

def _load_lab_inversion() -> dict:
    """
    Build {lab_name: [(organ.state, direction, weight), ...]} from
    lab_integration.json so we can turn an abnormal lab into state labels.
    """
    with open(_LAB_MAP_PATH, encoding="utf-8") as f:
        mapping = json.load(f).get("lab_to_state_mapping", {})
    inv = {}
    for lab, entries in mapping.items():
        if lab.startswith("_") or not isinstance(entries, list):
            continue
        rules = []
        for e in entries:
            if not isinstance(e, dict):
                continue
            organ = e.get("organ")
            state = e.get("state")
            if not state:
                continue
            # Some lab mappings have organ=null (systemic markers like
            # 'inflammation'). Namespace them so every state is 'organ.var'.
            full = f"{organ}.{state}" if organ else f"systemic.{state}"
            hi = e.get("high_means_perturb")
            lo = e.get("low_means_perturb")
            if hi is not None:
                rules.append((full, "high", float(hi)))
            if lo is not None:
                rules.append((full, "low", float(lo)))
        if rules:
            inv[lab.lower()] = rules
    return inv


_LAB_INVERSION = None
def lab_inversion():
    global _LAB_INVERSION
    if _LAB_INVERSION is None:
        _LAB_INVERSION = _load_lab_inversion()
    return _LAB_INVERSION


def labels_from_labs(labs: dict, abnormal_flags: dict | None = None) -> list:
    """
    Given a {lab_name: value} dict and optional {lab_name: 'high'|'low'} flags,
    produce StateLabels. If no flags given, we treat any provided lab as
    'high' abnormal (caller should pass flags for real data).
    """
    inv = lab_inversion()
    out = []
    for lab, val in (labs or {}).items():
        rules = inv.get(str(lab).lower())
        if not rules:
            continue
        flag = (abnormal_flags or {}).get(lab, "high")  # default assume high
        for full_state, direction, weight in rules:
            if direction != flag:
                continue
            out.append(StateLabel(
                state=full_state,
                value=round(weight, 3),
                direction=direction,
                source="lab_direct",
                confidence=round(min(0.95, 0.6 + weight * 0.35), 3),
                rationale=f"{lab}={val} ({flag})",
            ))
    return out


# Simple vital → state rules (vitals aren't in lab_integration consistently)
_VITAL_RULES = {
    # vital, comparator, threshold, (state, direction, value, conf, why)
    "spo2":   ("<", 94, ("lung.gas_exchange", "low", 0.6, 0.9, "hypoxemia")),
    "rr":     (">", 24, ("lung.gas_exchange", "low", 0.4, 0.7, "tachypnea")),
    "hr":     (">", 100, ("heart.sympathetic_drive", "high", 0.5, 0.6, "tachycardia")),
    "bp_sys": ("<", 90, ("heart.cardiac_output", "low", 0.6, 0.8, "hypotension")),
    "temp":   (">", 38.0, ("blood.infection_load", "high", 0.5, 0.6, "fever")),
}

def labels_from_vitals(vitals: dict) -> list:
    out = []
    for v, spec in _VITAL_RULES.items():
        if v not in (vitals or {}):
            continue
        cmp_, thr, (state, direction, value, conf, why) = spec
        x = vitals[v]
        try:
            x = float(x)
        except (TypeError, ValueError):
            continue
        hit = (x < thr) if cmp_ == "<" else (x > thr)
        if hit:
            out.append(StateLabel(
                state=state, value=value, direction=direction,
                source="vital_direct", confidence=conf,
                rationale=f"{v}={x} {why}",
            ))
    return out


# ──────────────────────────────────────────────────────────────────────
# Mechanism labels (LOWER confidence) — from Osler reason()
# ──────────────────────────────────────────────────────────────────────

_NX = None
def _nexus():
    global _NX
    if _NX is None:
        # silence the noisy load banner
        _stdout = sys.stdout
        sys.stdout = io.StringIO()
        from nexus_engine.nexus_medical import NexusMedical
        _NX = NexusMedical()
        _NX.load_knowledge()
        sys.stdout = _stdout
    return _NX


def labels_from_mechanism(symptoms: list, context: dict, top_k: int = 1) -> list:
    """
    Run Osler and read matched_states from the top-k diagnoses. These are
    states the symptoms imply via mechanism — weaker than direct lab labels.
    """
    nx = _nexus()
    out = []
    seen = set()
    try:
        res = nx.reason(symptoms, context=context or {})
    except Exception as e:
        return out
    for d in res.get("diagnoses", [])[:top_k]:
        sim = d.get("_state_simulation", {})
        for ms in sim.get("matched_states", []):
            state = ms.get("state", "")
            if not state or state in seen:
                continue
            seen.add(state)
            dval = ms.get("disease_value", 0.5)
            thr = ms.get("user_threshold", 0.4)
            # confidence scaled by how far the disease value clears threshold
            try:
                margin = abs(float(dval) - float(thr))
            except (TypeError, ValueError):
                margin = 0.0
            conf = round(min(0.7, 0.3 + margin), 3)  # capped below direct labels
            out.append(StateLabel(
                state=state,
                value=round(float(dval), 3) if isinstance(dval, (int, float)) else 0.5,
                direction=ms.get("direction", "high"),
                source="mechanism",
                confidence=conf,
                rationale="symptoms: " + ", ".join(ms.get("supporting_symptoms", [])),
            ))
    return out


def _dedupe_labels(labels: list) -> list:
    """If the same state is labeled by multiple sources, keep the highest
    confidence one (direct beats mechanism)."""
    best = {}
    for l in labels:
        cur = best.get(l.state)
        if cur is None or l.confidence > cur.confidence:
            best[l.state] = l
    return list(best.values())


# ──────────────────────────────────────────────────────────────────────
# Top-level: case → trajectory
# ──────────────────────────────────────────────────────────────────────

def case_to_trajectory(
    patient_id: str,
    symptoms: list,
    context: dict | None = None,
    labs: dict | None = None,
    vitals: dict | None = None,
    lab_flags: dict | None = None,
    timeline: list | None = None,
    outcomes: dict | None = None,
    source: str = "synthetic_osler",
) -> Trajectory:
    """
    Build a Trajectory from one case.

    `timeline`: optional list of future timepoints, each:
        {"t_hours": 6, "vitals": {...}, "labs": {...}, "symptoms": [...],
         "interventions": [{"drug": "...", "dose_mg": N, "route": "iv"}]}
      The converter will create events + state snapshots for each.

    If no timeline, just creates a single t0 snapshot.
    """
    context = context or {}
    events = []
    snapshots = []

    # ── t0 events ──
    if symptoms:
        events.append(Event(0.0, "symptoms", list(symptoms)))
    if vitals:
        events.append(Event(0.0, "vitals", dict(vitals)))
    if labs:
        events.append(Event(0.0, "labs", dict(labs)))

    # ── t0 state labels ──
    t0_labels = []
    t0_labels += labels_from_labs(labs or {}, lab_flags)
    t0_labels += labels_from_vitals(vitals or {})
    t0_labels += labels_from_mechanism(symptoms or [], context)
    snapshots.append(StateSnapshot(0.0, _dedupe_labels(t0_labels)))

    # ── future timepoints ──
    for tp in (timeline or []):
        t = float(tp.get("t_hours", 0))
        if tp.get("symptoms"):
            events.append(Event(t, "symptoms", list(tp["symptoms"])))
        if tp.get("vitals"):
            events.append(Event(t, "vitals", dict(tp["vitals"])))
        if tp.get("labs"):
            events.append(Event(t, "labs", dict(tp["labs"])))
        for iv in tp.get("interventions", []):
            events.append(Event(t, "intervention", dict(iv)))
        if tp.get("imaging"):
            events.append(Event(t, "imaging", dict(tp["imaging"])))
        # state labels at this timepoint (from whatever's observed)
        tp_labels = []
        tp_labels += labels_from_labs(tp.get("labs", {}), tp.get("lab_flags"))
        tp_labels += labels_from_vitals(tp.get("vitals", {}))
        if tp.get("symptoms"):
            tp_labels += labels_from_mechanism(tp["symptoms"], context)
        if tp_labels:
            snapshots.append(StateSnapshot(t, _dedupe_labels(tp_labels)))

    # keep events sorted
    events.sort(key=lambda e: e.t_hours)

    return Trajectory(
        patient_id=patient_id,
        source=source,
        demographics={k: context[k] for k in ("age", "sex") if k in context},
        events=events,
        state_snapshots=snapshots,
        outcomes=outcomes or {},
        meta={"generator": "osler_to_trajectory.v1"},
    )


# ──────────────────────────────────────────────────────────────────────
# Demo: build a small dataset
# ──────────────────────────────────────────────────────────────────────

def build_demo_dataset() -> list:
    """A few illustrative cases, including one multi-timepoint trajectory."""
    trajs = []

    # Case 1: pneumonia, multi-timepoint (deterioration)
    trajs.append(case_to_trajectory(
        patient_id="syn_pneumonia_001",
        symptoms=["cough", "fever", "shortness of breath"],
        context={"age": 64, "sex": "male"},
        vitals={"temp": 38.6, "hr": 112, "spo2": 93, "rr": 22},
        labs={"WBC": 15.2, "CRP": 110},
        lab_flags={"WBC": "high", "CRP": "high"},
        timeline=[
            {"t_hours": 6, "vitals": {"spo2": 89, "rr": 28},
             "labs": {"lactate": 2.4}, "lab_flags": {"lactate": "high"},
             "interventions": [{"drug": "ceftriaxone", "dose_mg": 2000, "route": "iv"},
                               {"drug": "oxygen", "route": "nasal_cannula"}]},
            {"t_hours": 24, "imaging": {"modality": "CXR", "finding": "right lower lobe opacity"}},
        ],
        outcomes={"icu_admission": True, "discharge_dx": "community-acquired pneumonia"},
    ))

    # Case 2: MI, single timepoint
    trajs.append(case_to_trajectory(
        patient_id="syn_mi_001",
        symptoms=["chest pain", "diaphoresis"],
        context={"age": 58, "sex": "male"},
        vitals={"hr": 96, "bp_sys": 148},
        labs={"troponin": 2.1},
        lab_flags={"troponin": "high"},
        outcomes={"discharge_dx": "acute myocardial infarction"},
    ))

    # Case 3: AKI / sepsis
    trajs.append(case_to_trajectory(
        patient_id="syn_aki_001",
        symptoms=["fatigue", "confusion"],
        context={"age": 72, "sex": "female"},
        vitals={"bp_sys": 88, "hr": 110, "temp": 38.9},
        labs={"creatinine": 2.8, "WBC": 18.0, "lactate": 3.1},
        lab_flags={"creatinine": "high", "WBC": "high", "lactate": "high"},
        outcomes={"icu_admission": True, "discharge_dx": "septic shock with AKI"},
    ))

    return trajs


if __name__ == "__main__":
    print("Building demo trajectory dataset from Osler...")
    ds = build_demo_dataset()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "demo_trajectories.jsonl")
    n = write_jsonl(ds, out_path)
    report = validate_dataset(ds)
    print(f"Wrote {n} trajectories to {out_path}")
    print(f"Validation: {report['valid']} valid, {report['invalid']} invalid")
    if report["errors"]:
        print("Errors:", json.dumps(report["errors"], indent=2)[:500])

    # Show label stats
    total_labels = 0
    by_source = {}
    for t in ds:
        for snap in t.state_snapshots:
            for l in snap.labels:
                total_labels += 1
                by_source[l.source] = by_source.get(l.source, 0) + 1
    print(f"\nState labels: {total_labels} total")
    for src, c in sorted(by_source.items()):
        print(f"  {src}: {c}")

    # Show one trajectory's first snapshot
    print(f"\nExample — {ds[0].patient_id} @ t0 state labels:")
    for l in ds[0].state_snapshots[0].labels[:8]:
        print(f"  {l.state:34s} {l.direction:4s} v={l.value:.2f} conf={l.confidence:.2f} [{l.source}] ← {l.rationale}")
