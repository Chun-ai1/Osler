"""
NEXUS Physiology-Aware Reward Function
════════════════════════════════════════════════════════════════
Replaces the simple reward in nexus_learning_env.py.

What's new vs the basic reward:
  - SepsisCascade.simulate()  → penalise for each organ that *would* fail
                                 without correct treatment
  - ChemicalBalance            → penalise critical lab derangements
                                 (pH<7.2, MAP<65, lactate>4 …)
  - OrganDependencyGraph       → cascade penalty if a critical organ fails
  - Severity multipliers        → critical patients punish mistakes harder
  - Early-treatment bonus       → reward agent for acting before cascade

Usage (drop-in replacement for MedicalEnv._compute_reward):

    from nexus_reward import PhysiologyReward
    self._reward_fn = PhysiologyReward(nexus_medical)

    # inside step():
    reward, breakdown = self._reward_fn.compute(
        diagnosis, treatment, nexus_result, patient
    )
"""

from __future__ import annotations
from typing import Dict, List, Tuple

# Severity → base multiplier
SEVERITY_MULT = {"moderate": 1.0, "severe": 1.5, "critical": 2.0}

# Organs where failure is instantly catastrophic
CRITICAL_ORGANS = {"brain", "brainstem", "heart", "diaphragm"}

# Lab thresholds that indicate impending death
CRITICAL_LABS = {
    "pH":         ("lt", 7.2,  -2.0),   # (direction, threshold, penalty)
    "MAP":        ("lt", 65,   -2.0),
    "lactate":    ("gt", 4.0,  -1.5),
    "pO2":        ("lt", 60,   -1.5),
    "potassium":  ("gt", 6.0,  -1.0),
    "creatinine": ("gt", 3.0,  -0.5),
}


class PhysiologyReward:
    """
    Computes a multi-factor reward that includes physiology simulation.
    """

    def __init__(self, nexus_medical, atlas=None):
        self.nexus = nexus_medical
        self._atlas = atlas   # reuse from MedicalEnv — do NOT build a second one
        self._physio = None
        self._init_engines()

    # ── lazy init ──────────────────────────────────────────────
    def _init_engines(self):
        try:
            from nexus_engine.physiology_engine import PhysiologyEngine
            if self._atlas is None:
                from nexus_engine.anatomy_atlas import AnatomyAtlas
                self._atlas = AnatomyAtlas()
            self._physio = PhysiologyEngine(self._atlas)
            print("[REWARD] PhysiologyEngine loaded ✓")
        except Exception as e:
            print(f"[REWARD] PhysiologyEngine not available ({e}) — using fallback")

    # ── public API ─────────────────────────────────────────────
    def compute(
        self,
        diagnosis: str,
        treatment: str,
        nexus_result: dict,
        patient: dict,
    ) -> Tuple[float, dict]:
        """
        Returns (total_reward, breakdown_dict).
        breakdown_dict is logged per episode for analysis.
        """
        severity    = patient.get("severity", "moderate")
        mult        = SEVERITY_MULT.get(severity, 1.0)
        true_dx     = patient.get("disease", "")
        true_tx     = patient.get("correct_treatments", [])
        organ_src   = patient.get("infected_organ", "")
        pathogen    = patient.get("pathogen", "virus")
        breakdown   = {}

        # ── 1. Diagnosis accuracy ─────────────────────────────
        correct_dx = self._matches(diagnosis, true_dx)
        r_dx = 2.0 * mult if correct_dx else -0.5
        breakdown["r_diagnosis"] = round(r_dx, 3)

        # ── 2. Treatment accuracy ─────────────────────────────
        correct_tx = any(self._matches(treatment, t) for t in true_tx)
        r_tx = 1.5 * mult if correct_tx else -1.0 * mult
        breakdown["r_treatment"] = round(r_tx, 3)

        # ── 3. NEXUS consistency bonus ─────────────────────────
        cons = nexus_result.get("nexus_consistency", {}).get("consistency_score", 0.5)
        r_cons = (cons - 0.5) * 1.2   # -0.6 … +0.6
        breakdown["r_nexus_consistency"] = round(r_cons, 3)

        # ── 4. Sepsis cascade penalty (PhysiologyEngine) ───────
        r_sepsis = 0.0
        cascade_info = {}
        if self._physio and organ_src:
            try:
                sev_float = {"moderate": 0.4, "severe": 0.7, "critical": 0.9}.get(severity, 0.5)
                sim = self._physio.sepsis.simulate(
                    organ_src,
                    pathogen_severity=sev_float,
                    max_steps=8,
                )
                failure_order = sim.get("failure_order", [])
                outcome       = sim.get("outcome", "contained")
                final_press   = sim.get("final_pressure", 1.0)

                cascade_info = {
                    "outcome":       outcome,
                    "organs_failed": [f["organ"] for f in failure_order],
                    "final_pressure": final_press,
                }

                # Penalise per failed organ (more for critical organs)
                for f in failure_order:
                    crit = f.get("criticality", 0.2)
                    organ_pen = -0.4 * crit * mult
                    if f["organ"] in CRITICAL_ORGANS:
                        organ_pen *= 1.5
                    r_sepsis += organ_pen

                # Big bonus if correct treatment would have stopped the cascade
                if correct_tx and outcome in ("contained", "serious"):
                    r_sepsis += 1.0 * mult   # agent acted in time
                elif correct_tx and outcome == "critical":
                    r_sepsis += 0.4 * mult
                elif not correct_tx and outcome == "death":
                    r_sepsis -= 2.0 * mult   # patient died because of wrong tx

            except Exception as e:
                cascade_info = {"error": str(e)}

        breakdown["r_sepsis_cascade"] = round(r_sepsis, 3)
        breakdown["cascade_info"]     = cascade_info

        # ── 5. Chemical derangement penalty ───────────────────
        r_chem = 0.0
        chem_alerts = []
        if self._physio and cascade_info.get("organs_failed"):
            try:
                chem = self._physio.chemistry.apply_organ_failure(
                    cascade_info["organs_failed"]
                )
                labs = chem.get("state", {})
                for param, (direction, threshold, pen) in CRITICAL_LABS.items():
                    val = labs.get(param)
                    if val is None:
                        continue
                    if direction == "lt" and val < threshold:
                        r_chem += pen * mult
                        chem_alerts.append(f"{param}={val:.1f} (< {threshold})")
                    elif direction == "gt" and val > threshold:
                        r_chem += pen * mult
                        chem_alerts.append(f"{param}={val:.1f} (> {threshold})")
            except Exception:
                pass

        breakdown["r_chemistry"]  = round(r_chem, 3)
        breakdown["chem_alerts"]  = chem_alerts

        # ── 6. Red-flag detection bonus ────────────────────────
        flags     = nexus_result.get("nexus_red_flags", [])
        r_flags   = min(len(flags) * 0.1, 0.4)   # noticing red flags = good
        breakdown["r_red_flags"] = round(r_flags, 3)

        # ── 7. Pathogen spread penalty ─────────────────────────
        spread  = nexus_result.get("nexus_pathogen_spread", [])
        r_spread = 0.0
        for s in spread:
            if isinstance(s, dict) and s.get("organ") in CRITICAL_ORGANS:
                r_spread -= 0.2 * s.get("risk", 0.3) * mult
        breakdown["r_spread"] = round(r_spread, 3)

        # ── Total ──────────────────────────────────────────────
        total = r_dx + r_tx + r_cons + r_sepsis + r_chem + r_flags + r_spread
        total = round(max(-10.0, min(10.0, total)), 3)
        breakdown["total"] = total
        breakdown["severity"] = severity
        breakdown["correct_dx"] = correct_dx
        breakdown["correct_tx"] = correct_tx

        return total, breakdown

    @staticmethod
    def _matches(a: str, b: str) -> bool:
        a, b = a.lower().strip(), b.lower().strip()
        return a == b or a in b or b in a