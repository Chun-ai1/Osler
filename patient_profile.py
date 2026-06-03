"""
patient_profile.py — Patient layer.

Renal function uses KDIGO GFR categories (G1-G5) and explicitly does NOT claim CKD
from eGFR alone: CKD requires chronicity or a kidney-damage marker. eGFR 84 is G2
(mildly decreased GFR), NOT an impaired patient. Dose adjustment is label-dependent
and is only flagged for the safety gate when eGFR < 60 (or a damage marker is present).

Vital/lab flags use standard reference thresholds (explicit constants) to raise
caution flags only; nothing here sets a dose.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

HYPOTENSION_SBP = 90      # mmHg
TACHYCARDIA_HR  = 100     # bpm (resting)
HYPOXIA_SPO2    = 92      # %
HYPERKALEMIA_K  = 5.5     # mmol/L
RENAL_DOSE_REVIEW_EGFR = 60   # below this, renal dose adjustment may apply (label-dependent)


def classify_renal_function(egfr: Optional[float], has_kidney_damage_marker: bool = False) -> Dict[str, Any]:
    if egfr is None:
        return {"egfr": None, "gfr_category": None, "description": "unknown",
                "ckd_established_from_available_data": False,
                "dose_adjustment": "unknown — eGFR not provided"}
    if egfr >= 90:   cat, desc = "G1", "normal or high"
    elif egfr >= 60: cat, desc = "G2", "mildly decreased"
    elif egfr >= 45: cat, desc = "G3a", "mildly to moderately decreased"
    elif egfr >= 30: cat, desc = "G3b", "moderately to severely decreased"
    elif egfr >= 15: cat, desc = "G4", "severely decreased"
    else:            cat, desc = "G5", "kidney failure"
    ckd = egfr < 60 or has_kidney_damage_marker
    return {"egfr": egfr, "gfr_category": cat, "description": desc,
            "ckd_established_from_available_data": ckd,
            "dose_adjustment": "must be determined from drug label"}


@dataclass
class PatientProfile:
    age: Optional[int] = None
    sex: Optional[str] = None
    weight_kg: Optional[float] = None
    pregnancy: bool = False
    egfr: Optional[float] = None
    has_kidney_damage_marker: bool = False
    hepatic_status: str = "normal"
    allergies: List[str] = field(default_factory=list)
    current_medications: List[str] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)
    symptoms: List[str] = field(default_factory=list)
    vitals: Dict[str, float] = field(default_factory=dict)
    labs: Dict[str, float] = field(default_factory=dict)
    renal: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.renal = classify_renal_function(self.egfr, self.has_kidney_damage_marker)

    def renal_label(self) -> str:
        if self.egfr is None:
            return "eGFR not provided"
        r = self.renal
        ckd = "CKD established" if r["ckd_established_from_available_data"] \
            else "CKD not established from available data"
        return f"eGFR {self.egfr:g} mL/min/1.73m² ({r['gfr_category']}; {ckd})"

    def renal_dose_review_needed(self) -> bool:
        return self.egfr is not None and (self.egfr < RENAL_DOSE_REVIEW_EGFR or self.has_kidney_damage_marker)

    def flags(self) -> List[Dict[str, Any]]:
        out = []
        sbp, hr, spo2 = self.vitals.get("sbp"), self.vitals.get("heart_rate"), self.vitals.get("spo2")
        k = self.labs.get("potassium")
        if sbp is not None and sbp < HYPOTENSION_SBP:
            out.append({"flag": "hypotension", "detail": f"SBP {sbp:g} < {HYPOTENSION_SBP}",
                        "keywords": ["hypotension", "hypotensive", "blood pressure", "shock", "syncope"]})
        if hr is not None and hr > TACHYCARDIA_HR:
            out.append({"flag": "tachycardia", "detail": f"HR {hr:g} > {TACHYCARDIA_HR}",
                        "keywords": ["tachycardia", "tachycardic", "arrhythmia", "heart rate", "palpitation"]})
        if spo2 is not None and spo2 < HYPOXIA_SPO2:
            out.append({"flag": "hypoxia", "detail": f"SpO2 {spo2:g} < {HYPOXIA_SPO2}",
                        "keywords": ["hypoxia", "respiratory", "oxygen"]})
        if k is not None and k > HYPERKALEMIA_K:
            out.append({"flag": "hyperkalemia", "detail": f"K+ {k:g} > {HYPERKALEMIA_K}",
                        "keywords": ["hyperkalemia", "potassium"]})
        if self.renal_dose_review_needed():
            out.append({"flag": "renal_dose_review",
                        "detail": f"{self.renal_label()} — renal dose review may apply",
                        "keywords": ["renal", "kidney", "creatinine clearance", "nephro", "dialysis"]})
        if self.hepatic_status not in ("normal", None):
            out.append({"flag": "hepatic_impairment", "detail": self.hepatic_status,
                        "keywords": ["hepatic", "liver", "cirrhosis"]})
        if self.pregnancy:
            out.append({"flag": "pregnancy", "detail": "pregnant",
                        "keywords": ["pregnan", "gestation", "fetal", "teratogen"]})
        return out

    def missing_core(self) -> List[str]:
        miss = []
        if self.age is None: miss.append("age")
        if self.weight_kg is None: miss.append("weight_kg")
        if self.egfr is None: miss.append("eGFR")
        return miss

    @classmethod
    def from_json(cls, path: str) -> "PatientProfile":
        d = json.loads(Path(path).read_text())
        renal = d.get("renal", {})
        return cls(age=d.get("age"), sex=d.get("sex"), weight_kg=d.get("weight_kg"),
                   pregnancy=bool(d.get("pregnancy", False)),
                   egfr=renal.get("egfr"),
                   has_kidney_damage_marker=bool(renal.get("kidney_damage_marker", False)),
                   hepatic_status=(d.get("hepatic", {}) or {}).get("status", "normal"),
                   allergies=[a.lower() for a in d.get("allergies", [])],
                   current_medications=[m.lower() for m in d.get("current_medications", [])],
                   conditions=[c.lower() for c in d.get("conditions", [])],
                   symptoms=[s.lower() for s in d.get("symptoms", [])],
                   vitals=d.get("vitals", {}), labs=d.get("labs", {}))


if __name__ == "__main__":
    for e in (84, 50, 28):
        p = PatientProfile(age=50, egfr=e, vitals={"heart_rate": 135})
        print(p.renal_label(), "| dose-review:", p.renal_dose_review_needed(),
              "| flags:", [f["flag"] for f in p.flags()])
