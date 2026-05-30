"""
NEXUS Physiology Engine
═══════════════════════════════════════════════════════════════
Numerical simulation of body state — NEXUS reasons in physiological 
state space, not in word matching.

WHY THIS EXISTS:
  Word-based reasoning: "bleeding" = string token, no understanding
  State-based reasoning: "bleeding" = blood_volume -= delta, MAP drops, HR up
                        cascade triggers predictable downstream effects

This is the closest thing to "thinking in the body" without using 3D meshes.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
import copy


# ═══════════════════════════════════════════════════════════════
# Body State — what the patient looks like physiologically
# ═══════════════════════════════════════════════════════════════

@dataclass
class BodyState:
    """Numerical representation of body homeostasis."""

    # ── Hemodynamics ──
    blood_volume_ml: float = 5000.0
    map_mmhg:        float = 95.0       # mean arterial pressure
    sbp_mmhg:        float = 120.0
    dbp_mmhg:        float = 80.0
    heart_rate:      float = 72.0
    cardiac_output:  float = 5.0        # L/min

    # ── Respiratory ──
    spo2:            float = 98.0       # % oxygen saturation
    resp_rate:       float = 14.0
    pao2:            float = 95.0       # mmHg
    paco2:           float = 40.0

    # ── Chemistry / Acid-base ──
    pH:              float = 7.40
    lactate_mmol:    float = 1.0
    glucose_mg:      float = 90.0
    sodium_meq:      float = 140.0
    potassium_meq:   float = 4.0
    creatinine_mg:   float = 0.9
    bun_mg:          float = 14.0

    # ── Hematologic ──
    hemoglobin_g:    float = 14.0
    hematocrit:      float = 0.42
    platelets_k:     float = 250.0
    wbc_k:           float = 7.0

    # ── Thermoregulation ──
    temperature_c:   float = 37.0

    # ── Per-organ perfusion (0=no flow, 1=normal) ──
    organ_perfusion: Dict[str, float] = field(default_factory=lambda: {
        "brain":      1.0, "heart":     1.0, "lungs":      1.0,
        "kidney_l":   1.0, "kidney_r":  1.0, "liver":      1.0,
        "spleen":     1.0, "pancreas":  1.0, "gi_tract":   1.0,
        "skin":       1.0, "muscles":   1.0, "extremities":1.0,
    })

    # ── Per-organ function (0=failed, 1=normal) ──
    organ_function: Dict[str, float] = field(default_factory=lambda: {
        "brain":      1.0, "heart":     1.0, "lungs":      1.0,
        "kidney_l":   1.0, "kidney_r":  1.0, "liver":      1.0,
        "spleen":     1.0, "pancreas":  1.0, "gi_tract":   1.0,
    })

    # ── Active processes ──
    active_processes: Set[str] = field(default_factory=set)

    def snapshot(self) -> dict:
        """Return a flat dict for trace logging."""
        return {
            "blood_volume_ml": round(self.blood_volume_ml, 0),
            "map":             round(self.map_mmhg, 1),
            "bp":              f"{round(self.sbp_mmhg)}/{round(self.dbp_mmhg)}",
            "hr":              round(self.heart_rate, 0),
            "spo2":            round(self.spo2, 1),
            "rr":              round(self.resp_rate, 0),
            "pH":              round(self.pH, 2),
            "lactate":         round(self.lactate_mmol, 1),
            "temp_c":          round(self.temperature_c, 1),
            "hgb":             round(self.hemoglobin_g, 1),
            "perfusion_min":   round(min(self.organ_perfusion.values()), 2),
            "failed_organs":   [k for k, v in self.organ_function.items() if v < 0.4],
            "active":          sorted(self.active_processes),
        }


# ═══════════════════════════════════════════════════════════════
# Symptom / Finding → State Effect Library
# ═══════════════════════════════════════════════════════════════

_FALLBACK_SYMPTOM_EFFECTS: Dict[str, Dict] = {
    # ── Bleeding patterns ──
    "bleeding_minor": {
        "blood_volume_ml": -250, "heart_rate": +5,
        "process": "minor_blood_loss",
    },
    "bleeding_moderate": {
        "blood_volume_ml": -750, "map_mmhg": -8, "heart_rate": +15,
        "hemoglobin_g": -1.5, "hematocrit": -0.04,
        "process": "moderate_blood_loss",
    },
    "bleeding_severe": {
        "blood_volume_ml": -1500, "map_mmhg": -25, "sbp_mmhg": -30,
        "heart_rate": +30, "hemoglobin_g": -3.0, "hematocrit": -0.10,
        "lactate_mmol": +1.5, "perfusion_global": 0.75,
        "process": "hemorrhagic_shock_risk",
    },
    "hematemesis": {  # vomiting blood
        "blood_volume_ml": -800, "map_mmhg": -10, "heart_rate": +20,
        "hemoglobin_g": -2.0, "process": "upper_gi_bleed",
    },
    "melena": {  # black tarry stools
        "blood_volume_ml": -500, "hemoglobin_g": -1.5,
        "process": "upper_gi_bleed_chronic",
    },
    "hematuria_gross": {
        "blood_volume_ml": -100, "hemoglobin_g": -0.3,
        "process": "urinary_bleeding",
    },

    # ── Volume loss / Dehydration ──
    "vomiting_persistent": {
        "blood_volume_ml": -500, "potassium_meq": -0.5, "pH": +0.04,
        "heart_rate": +10, "process": "metabolic_alkalosis",
    },
    "diarrhea_severe": {
        "blood_volume_ml": -800, "potassium_meq": -0.7,
        "sodium_meq": -3, "pH": -0.06,
        "heart_rate": +12, "process": "metabolic_acidosis_volume_loss",
    },
    "polyuria": {
        "blood_volume_ml": -300, "potassium_meq": -0.3,
        "process": "osmotic_diuresis",
    },

    # ── Infection / Inflammation ──
    "fever": {
        "temperature_c": +2.0, "heart_rate": +15, "resp_rate": +3,
        "process": "febrile_response",
    },
    "high_fever": {
        "temperature_c": +3.5, "heart_rate": +25, "resp_rate": +6,
        "wbc_k": +6, "process": "severe_infection",
    },
    "chills": {
        "temperature_c": +1.0, "heart_rate": +10,
        "process": "rigors_pre_fever",
    },
    "septic_shock_signs": {
        "map_mmhg": -30, "heart_rate": +40, "lactate_mmol": +3.5,
        "perfusion_global": 0.5, "wbc_k": +8,
        "process": "distributive_shock",
    },

    # ── Respiratory ──
    "dyspnea_severe": {
        "spo2": -8, "resp_rate": +14, "heart_rate": +20,
        "pao2": -25, "process": "respiratory_distress",
    },
    "cyanosis": {
        "spo2": -15, "pao2": -35, "process": "severe_hypoxemia",
    },
    "wheezing": {
        "spo2": -3, "resp_rate": +6, "process": "bronchospasm",
    },

    # ── Cardiac ──
    "chest_pain_crushing": {
        "process": "myocardial_ischemia",
        # affects perfusion downstream via organ-specific effect
        "perfusion_organ": ("heart", 0.6),
    },
    "syncope": {
        "map_mmhg": -20, "heart_rate": -10,
        "process": "cerebral_hypoperfusion",
        "perfusion_organ": ("brain", 0.5),
    },

    # ── Neurologic ──
    "altered_mental_status": {
        "perfusion_organ": ("brain", 0.7),
        "process": "cerebral_dysfunction",
    },
    "seizure": {
        "lactate_mmol": +2.0, "pH": -0.05,
        "process": "anaerobic_metabolism_crisis",
    },

    # ── Endocrine ──
    "polydipsia_polyuria_polyphagia": {
        "glucose_mg": +180, "blood_volume_ml": -400,
        "potassium_meq": -0.4, "process": "hyperglycemia",
    },
}


# Load from JSON via knowledge_loader, fall back to inline data if missing
try:
    from .knowledge_loader import get_kb
    _kb = get_kb()
    SYMPTOM_EFFECTS = _kb.symptom_effects or _FALLBACK_SYMPTOM_EFFECTS
except (ImportError, Exception):
    try:
        from knowledge_loader import get_kb
        _kb = get_kb()
        SYMPTOM_EFFECTS = _kb.symptom_effects or _FALLBACK_SYMPTOM_EFFECTS
    except Exception:
        SYMPTOM_EFFECTS = _FALLBACK_SYMPTOM_EFFECTS


# ═══════════════════════════════════════════════════════════════
# Cascade Rules — causal chains in physiological state
# ═══════════════════════════════════════════════════════════════

def apply_cascade(state: BodyState) -> List[str]:
    """
    Apply physiological cascade rules until state stabilizes.
    Returns list of triggered cascades for trace.
    """
    triggered = []
    iterations = 0
    max_iter = 5

    while iterations < max_iter:
        changed = False
        iterations += 1

        # ── Hypotension cascade ──
        if state.map_mmhg < 65 and "hypotension_decompensation" not in state.active_processes:
            for organ in state.organ_perfusion:
                state.organ_perfusion[organ] *= 0.6
            state.lactate_mmol += 2.0
            state.heart_rate = min(state.heart_rate + 20, 180)
            state.active_processes.add("hypotension_decompensation")
            triggered.append("MAP<65 → global hypoperfusion + tachycardia")
            changed = True

        # ── Severe hypovolemia ──
        if state.blood_volume_ml < 3500 and "hemorrhagic_shock" not in state.active_processes:
            state.map_mmhg = max(state.map_mmhg - 15, 40)
            state.heart_rate = min(state.heart_rate + 25, 180)
            state.organ_perfusion["kidney_l"] *= 0.5
            state.organ_perfusion["kidney_r"] *= 0.5
            state.organ_perfusion["gi_tract"] *= 0.6
            state.active_processes.add("hemorrhagic_shock")
            triggered.append(
                f"blood<3500ml → shock: MAP={state.map_mmhg:.0f}, HR={state.heart_rate:.0f}")
            changed = True

        # ── Critical hypoxia ──
        if state.spo2 < 88 and "tissue_hypoxia" not in state.active_processes:
            state.lactate_mmol += 1.5
            state.pH -= 0.05
            for organ in state.organ_perfusion:
                state.organ_function[organ] = min(state.organ_function.get(organ, 1.0), 0.7)
            state.active_processes.add("tissue_hypoxia")
            triggered.append(f"SpO2<88% → anaerobic metabolism, lactate↑")
            changed = True

        # ── Severe acidosis ──
        if state.pH < 7.20 and "severe_acidosis" not in state.active_processes:
            state.heart_rate = max(state.heart_rate - 20, 40)  # myocardial depression
            state.map_mmhg -= 10
            state.active_processes.add("severe_acidosis")
            triggered.append("pH<7.2 → myocardial depression")
            changed = True

        # ── Per-organ failure trigger ──
        for organ, perf in state.organ_perfusion.items():
            if perf < 0.4 and organ in state.organ_function:
                if state.organ_function[organ] > 0.4:
                    state.organ_function[organ] = perf  # failure follows perfusion
                    triggered.append(f"organ_failure: {organ} (perfusion={perf:.2f})")
                    changed = True

        # ── Renal failure → metabolic consequences ──
        kidney_avg = (state.organ_function.get("kidney_l", 1.0)
                       + state.organ_function.get("kidney_r", 1.0)) / 2
        if kidney_avg < 0.5 and "renal_failure" not in state.active_processes:
            state.creatinine_mg += 2.0
            state.bun_mg += 25
            state.potassium_meq += 1.0
            state.active_processes.add("renal_failure")
            triggered.append("renal_failure → uremia + hyperkalemia")
            changed = True

        # ── Brain hypoperfusion → AMS ──
        if state.organ_perfusion.get("brain", 1.0) < 0.6:
            if "ams" not in state.active_processes:
                state.active_processes.add("ams")
                triggered.append("brain perfusion<0.6 → altered mental status")
                changed = True

        if not changed:
            break

    return triggered


# ═══════════════════════════════════════════════════════════════
# Main API — apply symptoms to a body, simulate, return state
# ═══════════════════════════════════════════════════════════════

class PhysiologyEngine:
    """
    Simulates body physiological state given symptoms/findings.
    
    USAGE:
        engine = PhysiologyEngine()
        result = engine.simulate(["bleeding_severe", "altered_mental_status"])
        print(result["state"])         # numerical body state
        print(result["cascades"])      # what physiological cascades fired
        print(result["interpretation"]) # human-readable summary
    """

    def __init__(self, atlas=None):
        self.atlas = atlas  # optional anatomy atlas reference

    def simulate(self, symptoms: List[str], 
                 baseline: Optional[BodyState] = None,
                 age: int = 45, sex: str = "any") -> dict:
        """
        Apply each symptom's physiological effects, then run cascades.
        Returns the resulting state plus an interpretation.
        """
        # Start from baseline (healthy adult by default)
        state = copy.deepcopy(baseline) if baseline else BodyState()
        
        # Age/sex adjustments to baseline
        if age >= 65:
            state.heart_rate -= 5
            state.map_mmhg += 5  # mild HTN
        if age <= 5:
            state.heart_rate += 30
            state.resp_rate += 8

        applied = []
        unmatched = []

        # ── Apply each symptom's effect ──
        for symptom in symptoms:
            sym_norm = symptom.lower().strip().replace(" ", "_")

            # Try direct match
            effect = SYMPTOM_EFFECTS.get(sym_norm)

            # Try fuzzy match
            if not effect:
                for key, eff in SYMPTOM_EFFECTS.items():
                    if sym_norm in key or key in sym_norm:
                        effect = eff
                        break

            if not effect:
                unmatched.append(symptom)
                continue

            # Apply numerical deltas
            for field_name, delta in effect.items():
                if field_name == "process":
                    state.active_processes.add(delta)
                    continue
                if field_name == "perfusion_global":
                    for organ in state.organ_perfusion:
                        state.organ_perfusion[organ] *= delta
                    continue
                if field_name == "perfusion_organ":
                    organ, factor = delta
                    if organ in state.organ_perfusion:
                        state.organ_perfusion[organ] *= factor
                    continue
                # Standard numerical attribute
                if hasattr(state, field_name):
                    setattr(state, field_name,
                            getattr(state, field_name) + delta)

            applied.append(f"{symptom} → {effect.get('process', 'state_delta')}")

        # ── Recompute SBP/DBP from MAP ──
        # MAP ≈ DBP + 1/3(SBP-DBP), simplified
        state.sbp_mmhg = state.map_mmhg + 25
        state.dbp_mmhg = state.map_mmhg - 13

        # ── Run cascade rules ──
        cascades = apply_cascade(state)

        # ── Interpret state ──
        interpretation = self._interpret(state)

        return {
            "state":          state.snapshot(),
            "applied":        applied,
            "unmatched":      unmatched,
            "cascades":       cascades,
            "interpretation": interpretation,
            "shock_index":    round(state.heart_rate / max(state.sbp_mmhg, 1), 2),
            "critical":       state.map_mmhg < 65 or state.spo2 < 90 or state.pH < 7.25,
        }

    def _interpret(self, state: BodyState) -> List[str]:
        """Generate human-readable physiological summary."""
        notes = []

        if state.blood_volume_ml < 3500:
            notes.append(f"Hypovolemic ({state.blood_volume_ml:.0f}ml, normal ~5000)")
        if state.map_mmhg < 65:
            notes.append(f"Hypotensive (MAP={state.map_mmhg:.0f}, target ≥65)")
        if state.heart_rate > 120:
            notes.append(f"Tachycardic ({state.heart_rate:.0f} bpm)")
        if state.spo2 < 92:
            notes.append(f"Hypoxic (SpO2={state.spo2:.0f}%)")
        if state.lactate_mmol > 4:
            notes.append(f"Severe lactic acidosis ({state.lactate_mmol:.1f})")
        elif state.lactate_mmol > 2:
            notes.append(f"Elevated lactate ({state.lactate_mmol:.1f})")
        if state.pH < 7.30:
            notes.append(f"Acidotic (pH={state.pH:.2f})")
        elif state.pH > 7.50:
            notes.append(f"Alkalotic (pH={state.pH:.2f})")
        if state.temperature_c > 38.5:
            notes.append(f"Febrile ({state.temperature_c:.1f}°C)")
        if state.temperature_c < 35.5:
            notes.append(f"Hypothermic ({state.temperature_c:.1f}°C)")

        failed = [k for k, v in state.organ_function.items() if v < 0.4]
        if failed:
            notes.append(f"Organ failure: {', '.join(failed)}")

        return notes if notes else ["Physiologically stable"]


# ═══════════════════════════════════════════════════════════════
# Quick test
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    eng = PhysiologyEngine()

    print("\n=== Test 1: Severe Bleeding ===")
    r = eng.simulate(["bleeding_severe", "altered_mental_status"])
    print(f"State: {r['state']}")
    print(f"Cascades: {r['cascades']}")
    print(f"Interpretation: {r['interpretation']}")
    print(f"Shock index: {r['shock_index']}")
    print(f"Critical: {r['critical']}")

    print("\n=== Test 2: Sepsis ===")
    r = eng.simulate(["high_fever", "septic_shock_signs", "altered_mental_status"])
    print(f"State: {r['state']}")
    print(f"Cascades: {r['cascades']}")
    print(f"Interpretation: {r['interpretation']}")

    print("\n=== Test 3: Severe Vomiting + Diarrhea ===")
    r = eng.simulate(["vomiting_persistent", "diarrhea_severe"])
    print(f"State: {r['state']}")
    print(f"Cascades: {r['cascades']}")
    print(f"Interpretation: {r['interpretation']}")