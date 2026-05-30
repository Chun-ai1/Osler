"""
high_risk_gate.py
─────────────────────────────────────────────────────────────────
Single source of truth for high-risk symptom detection.
Loaded from red_flags.json at import time.

All modules (app.py, nexus_medical.py, evidence_gate.py,
anatomy_bridge.py) should import from here instead of
maintaining their own copies of the symptom set.

Usage:
    from nexus_engine.high_risk_gate import (
        is_high_risk, get_freeze_syndrome, HIGH_RISK_SYMS
    )
"""
from __future__ import annotations
import json, os, glob
from typing import Optional, Dict, Set, List, Tuple


# ── Build from red_flags.json at import ──────────────────────────────────────

def _load_freeze_rules(
    paths=("red_flags.json",
           "medical_knowledge/red_flags.json")
) -> Tuple[Dict[str, str], Dict[str, int], Set[str]]:
    """
    Returns:
      symptom_to_label   – symptom string → freeze syndrome label
      symptom_to_priority– symptom string → priority (higher = more dangerous)
      high_risk_syms     – flat set of all high-risk symptom strings
    """
    for path in paths:
        if os.path.exists(path):
            try:
                rules = json.load(open(path, encoding="utf-8"))
                s2l: Dict[str, str] = {}
                s2p: Dict[str, int] = {}
                for rule in rules:
                    if not rule.get("freeze_syndrome_label"):
                        continue
                    label    = rule["freeze_syndrome_label"]
                    priority = rule.get("freeze_priority", 5)
                    for sym in rule.get("symptoms", []):
                        sym_l = sym.lower().strip()
                        # Keep highest-priority label for each symptom
                        if priority > s2p.get(sym_l, -1):
                            s2l[sym_l] = label
                            s2p[sym_l] = priority
                if s2l:
                    print(f"[high_risk_gate] Loaded {len(s2l)} high-risk symptoms from {path}")
                    return s2l, s2p, frozenset(s2l.keys())
            except Exception as e:
                print(f"[high_risk_gate] Load error ({path}): {e}")

    # Fallback (identical values — used only if file missing)
    print("[high_risk_gate] WARNING: red_flags.json not found — using built-in fallback")
    _fallback = {
        "chest pain":           ("mixed high-risk chest-pain syndrome",        10),
        "shortness of breath":  ("mixed respiratory / cardiopulmonary syndrome", 9),
        "syncope":              ("high-risk syncope syndrome",                  8),
        "fainting":             ("high-risk syncope syndrome",                  8),
        "altered mental status":("high-risk neurologic syndrome",               9),
        "confusion":            ("high-risk neurologic syndrome",               8),
        "focal weakness":       ("high-risk neurologic syndrome — rule out stroke", 9),
        "slurred speech":       ("high-risk neurologic syndrome — rule out stroke", 9),
        "worst headache":       ("high-risk headache — rule out SAH",           9),
        "thunderclap headache": ("high-risk headache — rule out SAH",           9),
        "severe abdominal pain":("high-risk abdominal syndrome",                7),
        "hematemesis":          ("high-risk GI bleeding syndrome",              7),
        "hemoptysis":           ("high-risk respiratory bleeding syndrome",     7),
        "bloody stool":         ("high-risk GI bleeding syndrome",              7),
        "melena":               ("high-risk GI bleeding syndrome",              7),
    }
    s2l = {k: v[0] for k, v in _fallback.items()}
    s2p = {k: v[1] for k, v in _fallback.items()}
    return s2l, s2p, frozenset(s2l.keys())


_SYMPTOM_TO_LABEL, _SYMPTOM_TO_PRIORITY, HIGH_RISK_SYMS = _load_freeze_rules()

# Ordered list for dominant symptom selection (highest priority first)
HIGH_RISK_PRIORITY_ORDER: List[str] = sorted(
    _SYMPTOM_TO_LABEL.keys(),
    key=lambda s: -_SYMPTOM_TO_PRIORITY.get(s, 0)
)


# ── Public API ────────────────────────────────────────────────────────────────

def _normalise(s: str) -> str:
    return s.lower().replace("_", " ").strip()


def is_high_risk(symptoms) -> bool:
    """Return True if any symptom in the list is a high-risk freeze symptom."""
    return any(_normalise(s) in HIGH_RISK_SYMS for s in symptoms if s)


def get_dominant(symptoms) -> Optional[str]:
    """Return the highest-priority high-risk symptom from the list, or None."""
    sym_lower = {_normalise(s) for s in symptoms if s}
    for candidate in HIGH_RISK_PRIORITY_ORDER:
        if candidate in sym_lower:
            return candidate
    return None


def get_freeze_syndrome(symptoms) -> Optional[str]:
    """Return the freeze syndrome label for the dominant high-risk symptom."""
    dom = get_dominant(symptoms)
    return _SYMPTOM_TO_LABEL.get(dom) if dom else None


def build_freeze_assessment(symptoms, triage_level: str = "PROMPT") -> dict:
    """
    Build a complete frozen evidence_assessment dict for use when a
    high-risk symptom is detected.  Suitable for direct assignment to
    result["evidence_assessment"].
    """
    dom   = get_dominant(symptoms) or "unknown"
    label = _SYMPTOM_TO_LABEL.get(dom, "high-risk syndrome")
    return {
        "output_state":          "syndrome",
        "syndrome_label":        label,
        "secondary_pattern":     "",
        "secondary_systems":     [],
        "detected_systems":      [],
        "etiology_allowed":      False,
        "etiology_level":        "uncertain",
        "etiology_evidence":     f"frozen by high-risk symptom: {dom}",
        "disease_label_allowed": False,
        "disease_gate_details":  {"gate_passed": False, "frozen_by": dom},
        "all_vague":             False,
        "has_red_flags":         True,
        "triage_level":          triage_level.lower(),
    }