"""
pkpd_engine.py — standalone PK/PD pharmacology framework
========================================================

Pure pharmacology. Everything disease-/diagnosis-related has been removed:
no state_model import, no propagation, no symptom derivation, no
relieves/supports_diseases mapping, no treatment recommender. What remains is
exactly the two laws:

    dose + route + time + patient
        ↓  PK   first-order absorption + first-order elimination
    plasma concentration over time          ← "how the body handles the drug"
        ↓  normalize to a per-drug reference exposure
    C_norm(t)
        ↓  PD   Emax/Hill (separate benefit + toxicity curves)
    effect strength ∈ [0, Emax]             ← "what the drug does to the body"
        ↓  mechanism
    signed deltas on body-state variables

Modeling notes:
  • Absorption (Bateman), not bolus — concentration rises to a peak then decays.
    A route may declare ka_per_hr=null for a true IV push (bolus form).
  • Concentration is normalized (1.0 ≈ reference-dose peak in the reference
    patient), so EC50 lives on the same scale; no real-unit fabrication.
  • CL and Vd are the stored primitives; ke and t½ are derived. Renal/hepatic/
    age factors scale CL, so patient variation reaches the math.
  • Benefit and toxicity are independent Emax curves; therapeutic window =
    ec50_tox / ec50_benefit.

BodyState here is just the substrate the drug acts on (organ→variable→value,
clamped [0,1]). It carries no disease logic.

SAFETY: every PK/PD value is ESTIMATED, not clinical. Mechanistic simulation
only — never a dosing recommendation.
"""

from __future__ import annotations
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional

LN2 = math.log(2.0)


# ════════════════════════════════════════════════════════════════════
# Substrate the drug acts on
# ════════════════════════════════════════════════════════════════════

class BodyState:
    """organ → variable → value, clamped to [0, 1]. No disease logic."""
    def __init__(self):
        self.organs: Dict[str, Dict[str, float]] = {}
    def set(self, organ, variable, value):
        self.organs.setdefault(organ, {})[variable] = max(0.0, min(1.0, float(value)))
    def get(self, organ, variable, default=0.0):
        return self.organs.get(organ, {}).get(variable, default)
    def apply_delta(self, organ, variable, delta):
        self.set(organ, variable, self.get(organ, variable, 0.0) + delta)
    def copy(self):
        n = BodyState()
        for o, d in self.organs.items():
            n.organs[o] = dict(d)
        return n
    @classmethod
    def from_levels(cls, levels: Dict[tuple, float]):
        s = cls()
        for (organ, var), val in levels.items():
            s.set(organ, var, val)
        return s


# ════════════════════════════════════════════════════════════════════
# Inputs
# ════════════════════════════════════════════════════════════════════

_RENAL_FACTOR   = {"normal": 1.0, "mild": 0.8, "moderate": 0.5, "severe": 0.25}
_HEPATIC_FACTOR = {"normal": 1.0, "mild": 0.7, "impaired": 0.5, "severe": 0.35}


@dataclass
class PatientProfile:
    weight_kg: float = 70.0
    age: int = 40
    renal_function: str = "normal"
    hepatic_function: str = "normal"
    pregnancy: bool = False
    def renal_factor(self):   return _RENAL_FACTOR.get(self.renal_function, 1.0)
    def hepatic_factor(self): return _HEPATIC_FACTOR.get(self.hepatic_function, 1.0)
    def age_factor(self):     return 0.85 if self.age >= 65 else 1.0


REFERENCE_PATIENT = PatientProfile()   # the normalization yardstick


@dataclass
class DoseEvent:
    drug: str
    dose_mg: float
    route: str
    time_hr: float = 0.0


# ════════════════════════════════════════════════════════════════════
# PK — concentration over time
# ════════════════════════════════════════════════════════════════════

def half_life_hr(cl_total: float, vd_total: float) -> float:
    ke = cl_total / vd_total if vd_total > 0 else 0.0
    return (LN2 / ke) if ke > 0 else float("inf")


def _pk_params(drug_data, route, patient: PatientProfile):
    """Resolve (F, ka, Vd_total, ke). ke derived from patient-adjusted clearance."""
    pk = drug_data.get("pk", {})
    rinfo = pk.get("routes", {}).get(route) \
        or pk.get("routes", {}).get(pk.get("default_route", "")) or {}
    F  = float(rinfo.get("bioavailability", 1.0))
    ka = rinfo.get("ka_per_hr", None)              # None ⇒ IV bolus
    vd = max(float(pk.get("vd_L_per_kg", 1.0)) * patient.weight_kg, 1e-6)
    cl_base = float(pk.get("cl_L_per_hr_per_kg", 0.0)) * patient.weight_kg
    fr = float(pk.get("fraction_renal_clearance", 0.3))
    cl = cl_base * (fr * patient.renal_factor() + (1 - fr) * patient.hepatic_factor())
    cl = max(cl * patient.age_factor(), 1e-9)
    return F, (float(ka) if ka is not None else None), vd, cl / vd


def _c_raw(F, dose, ka, vd, ke, dt):
    if dt < 0:
        return 0.0
    if ka is None:                                 # IV bolus
        return (F * dose / vd) * math.exp(-ke * dt)
    if abs(ka - ke) < 1e-9:                         # flip-flop limit
        return (F * dose * ka / vd) * dt * math.exp(-ka * dt)
    return (F * dose * ka) / (vd * (ka - ke)) * (math.exp(-ke * dt) - math.exp(-ka * dt))


def _tmax(ka, ke):
    if ka is None:
        return 0.0
    return (1.0 / ka) if abs(ka - ke) < 1e-9 else math.log(ka / ke) / (ka - ke)


def _reference_cmax(drug_data):
    pk = drug_data.get("pk", {})
    ref_dose  = float(pk.get("reference_dose_mg", 1.0))
    ref_route = pk.get("reference_route", pk.get("default_route", ""))
    F, ka, vd, ke = _pk_params(drug_data, ref_route, REFERENCE_PATIENT)
    return max(_c_raw(F, ref_dose, ka, vd, ke, _tmax(ka, ke)), 1e-12)


def concentration_norm(drug_name, drug_data, dose_events, patient, t_hr) -> Optional[float]:
    """Dimensionless concentration at t (superposition). None if no PK block."""
    if not drug_data.get("pk"):
        return None
    c_ref = _reference_cmax(drug_data)
    total = 0.0
    for ev in dose_events:
        if ev.drug.lower() != drug_name.lower():
            continue
        route = ev.route or drug_data["pk"].get("default_route", "")
        F, ka, vd, ke = _pk_params(drug_data, route, patient)
        total += _c_raw(F, ev.dose_mg, ka, vd, ke, t_hr - ev.time_hr)
    return total / c_ref


# ════════════════════════════════════════════════════════════════════
# PD — concentration → effect → deltas
# ════════════════════════════════════════════════════════════════════

def emax(c, params) -> float:
    if c is None or c <= 0:
        return 0.0
    e, ec50, h = float(params.get("emax", 1.0)), float(params.get("ec50_norm", 1.0)), float(params.get("hill", 1.0))
    ch = c ** h
    denom = (ec50 ** h) + ch
    return (e * ch / denom) if denom > 0 else 0.0


def therapeutic_index(drug_data) -> Optional[float]:
    pd = drug_data.get("pd", {})
    eb, et = pd.get("benefit", {}).get("ec50_norm"), pd.get("toxicity", {}).get("ec50_norm")
    return round(et / eb, 2) if (eb and et and eb > 0) else None


def drug_response(drug_name, drug_data, dose_events, patient, t_hr) -> Dict[str, Any]:
    """Full pharmacology readout at time t. deltas carry organ='*' unresolved."""
    c = concentration_norm(drug_name, drug_data, dose_events, patient, t_hr)
    pd = drug_data.get("pd", {})
    benefit = emax(c, pd.get("benefit", {}))
    tox     = emax(c, pd.get("toxicity", {}))
    deltas = []
    for eff in drug_data.get("state_effects", []):
        deltas.append({"organ": eff["organ"], "variable": eff["variable"],
                       "delta": float(eff.get("max_delta", 0.0)) * benefit, "kind": "benefit"})
    for eff in drug_data.get("toxic_effects", []):
        deltas.append({"organ": eff["organ"], "variable": eff["variable"],
                       "delta": float(eff.get("max_delta", 0.0)) * tox, "kind": "toxicity"})
    return {"drug": drug_name, "time_hr": t_hr, "c_norm": c,
            "benefit_strength": benefit, "toxicity_strength": tox,
            "therapeutic_index": therapeutic_index(drug_data),
            "deltas": deltas, "confidence": drug_data.get("confidence", {})}


def apply_to_state(state: BodyState, deltas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Apply deltas to a BodyState, resolving organ='*' to organs holding the variable."""
    changes = []
    for d in deltas:
        organ, var, delta = d["organ"], d["variable"], d["delta"]
        targets = ([o for o, vs in state.organs.items() if var in vs] if organ == "*" else [organ])
        for o in targets:
            before = state.get(o, var)
            state.apply_delta(o, var, delta)
            changes.append({"organ": o, "variable": var, "kind": d["kind"],
                            "delta": round(delta, 4), "before": round(before, 4),
                            "after": round(state.get(o, var), 4)})
    return changes


# ════════════════════════════════════════════════════════════════════
# Timeline (stateless per step)
# ════════════════════════════════════════════════════════════════════

def simulate(drug_name, drug_data, dose_events, patient, time_points_hr,
             initial_state: Optional[BodyState] = None) -> Dict[str, Any]:
    """Concentration → effect → resulting variable levels over time.

    initial_state = starting levels for the variables of interest (e.g. a high
    bronchospasm). Each step rebuilds from it, so it's pure drug dynamics — no
    disease re-driving, no propagation. Hand the deltas to your own engine if you
    want coupling.
    """
    base = initial_state or BodyState()
    timeline = []
    for t in time_points_hr:
        r = drug_response(drug_name, drug_data, dose_events, patient, t)
        if r["c_norm"] is None:
            timeline.append({"time_hr": t, "mode": "unparameterized",
                             "note": "no pk/pd block; not simulated over time"})
            continue
        state = base.copy()
        changes = apply_to_state(state, r["deltas"])
        timeline.append({
            "time_hr": t, "c_norm": round(r["c_norm"], 4),
            "benefit_strength": round(r["benefit_strength"], 4),
            "toxicity_strength": round(r["toxicity_strength"], 4),
            "changes": changes,
            "levels": {f"{o}.{v}": round(val, 4)
                       for o, vs in state.organs.items() for v, val in vs.items()},
        })
    return {"drug": drug_name, "patient": patient.__dict__,
            "doses": [e.__dict__ for e in dose_events],
            "therapeutic_index": therapeutic_index(drug_data),
            "timeline": timeline,
            "_disclaimer": ("MECHANISTIC SIMULATION ONLY. PK/PD parameters are "
                            "ESTIMATED, not clinical. Not a dosing recommendation. "
                            "Verify against FDA label / guideline / clinician.")}


def load_drugs(path="drugs_pkpd.json") -> Dict[str, Any]:
    return json.loads(Path(path).read_text())["drugs"]


# ════════════════════════════════════════════════════════════════════
# Self-test
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    drugs = load_drugs(str(Path(__file__).with_name("drugs_pkpd.json")))
    alb = drugs["albuterol"]
    init = BodyState.from_levels({("lung", "bronchospasm"): 0.80,
                                  ("lung", "airflow_resistance"): 0.70,
                                  ("heart", "heart_rate"): 0.30})
    doses = [DoseEvent("albuterol", 2.5, "inhaled", 0.0)]
    pts = [0, 0.1, 0.25, 0.5, 1, 2, 4, 6]
    print(f"albuterol  TI={therapeutic_index(alb)}\n")
    for label, pt in [("normal renal", PatientProfile()),
                      ("severe renal", PatientProfile(renal_function="severe"))]:
        out = simulate("albuterol", alb, doses, pt, pts, init)
        print(f"── {label} ──")
        print(f"{'t(h)':>5}{'C_norm':>8}{'benefit':>9}{'tox':>7}{'bronchosp':>11}{'HR':>6}")
        for e in out["timeline"]:
            print(f"{e['time_hr']:>5}{e['c_norm']:>8.3f}{e['benefit_strength']:>9.3f}"
                  f"{e['toxicity_strength']:>7.3f}{e['levels']['lung.bronchospasm']:>11.3f}"
                  f"{e['levels']['heart.heart_rate']:>6.3f}")
        print()
    print(out["_disclaimer"])