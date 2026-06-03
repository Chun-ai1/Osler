"""
Clinical Context Engine

Integrates patient clinical data into NEXUS reasoning:
  - Demographics: age, sex, pregnancy status
  - Lab values: CRP, PCT, WBC, neutrophils, lymphocytes, hemoglobin, creatinine, glucose, etc.
  - Vitals: temperature, heart rate, blood pressure, respiratory rate, SpO2
  - History: chronic conditions, medications, allergies

Adjusts:
  - Disease scoring (age/sex-specific prevalence)
  - Etiology classification (labs confirm virus vs bacteria)
  - Triage level (vitals determine urgency)
  - Mechanism activation (lab markers validate specific mechanisms)
  - Risk stratification (qSOFA, SIRS criteria)

Input: patient context dict
Output: clinical adjustments for NEXUS reasoning
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
from collections import defaultdict


# ═══════════════════════════════════════════════
# LAB REFERENCE RANGES
# ═══════════════════════════════════════════════

LAB_RANGES = {
    # lab_name: (low_normal, high_normal, unit, clinical_meaning_if_high, clinical_meaning_if_low)
    "wbc":           (4.0, 11.0, "K/uL", "infection/inflammation", "immunosuppression/viral"),
    "neutrophils":   (40, 70, "%", "bacterial infection", "viral/drug-induced"),
    "lymphocytes":   (20, 40, "%", "viral infection/CLL", "immunodeficiency"),
    "hemoglobin":    (12.0, 17.0, "g/dL", "polycythemia", "anemia"),
    "platelets":     (150, 400, "K/uL", "reactive thrombocytosis", "thrombocytopenia"),
    "crp":           (0, 10, "mg/L", "inflammation/infection", None),
    "pct":           (0, 0.1, "ng/mL", "bacterial infection", None),
    "esr":           (0, 20, "mm/hr", "inflammation/infection/malignancy", None),
    "creatinine":    (0.7, 1.3, "mg/dL", "kidney dysfunction", None),
    "bun":           (7, 20, "mg/dL", "dehydration/kidney dysfunction", None),
    "glucose":       (70, 100, "mg/dL", "diabetes/stress response", "hypoglycemia"),
    "alt":           (7, 56, "U/L", "liver damage", None),
    "ast":           (10, 40, "U/L", "liver/muscle damage", None),
    "bilirubin":     (0.1, 1.2, "mg/dL", "liver dysfunction/hemolysis", None),
    "albumin":       (3.5, 5.0, "g/dL", None, "malnutrition/liver disease"),
    "potassium":     (3.5, 5.0, "mEq/L", "kidney dysfunction/acidosis", "GI loss/diuretics"),
    "sodium":        (136, 145, "mEq/L", "dehydration", "dilution/SIADH"),
    "lactate":       (0.5, 2.0, "mmol/L", "tissue hypoxia/shock", None),
    "d_dimer":       (0, 500, "ng/mL", "thrombosis/PE/DIC", None),
    "troponin":      (0, 0.04, "ng/mL", "myocardial injury", None),
    "bnp":           (0, 100, "pg/mL", "heart failure", None),
    "hba1c":         (4.0, 5.7, "%", "diabetes", None),
    "tsh":           (0.4, 4.0, "mIU/L", "hypothyroidism", "hyperthyroidism"),
}

# ═══════════════════════════════════════════════
# VITAL REFERENCE RANGES
# ═══════════════════════════════════════════════

VITAL_RANGES = {
    # vital: (low_normal, high_normal, unit, concern_if_high, concern_if_low)
    "temperature":       (36.1, 37.2, "°C", "fever/infection", "hypothermia"),
    "heart_rate":        (60, 100, "bpm", "tachycardia", "bradycardia"),
    "systolic_bp":       (90, 140, "mmHg", "hypertension", "hypotension/shock"),
    "diastolic_bp":      (60, 90, "mmHg", "hypertension", "hypotension"),
    "respiratory_rate":  (12, 20, "/min", "tachypnea/distress", "respiratory depression"),
    "spo2":              (95, 100, "%", None, "hypoxemia"),
    "map":               (70, 100, "mmHg", "hypertension", "shock/hypoperfusion"),
}

# ═══════════════════════════════════════════════
# AGE-BASED DISEASE PREVALENCE ADJUSTMENTS
# ═══════════════════════════════════════════════

AGE_ADJUSTMENTS = {
    # age_group: {disease_pattern: score_multiplier}
    "pediatric": {  # 0-17
        "meningitis": 1.3, "otitis": 1.5, "bronchiolitis": 1.5,
        "appendicitis": 1.2, "kawasaki": 2.0,
        "mi": 0.1, "stroke": 0.1, "cancer": 0.3, "copd": 0.05,
    },
    "young_adult": {  # 18-39
        "sti": 1.3, "anxiety": 1.2, "migraine": 1.3,
        "appendicitis": 1.3, "pregnancy": 1.5,
        "mi": 0.3, "cancer": 0.5, "copd": 0.2,
    },
    "middle_age": {  # 40-64
        "mi": 1.3, "diabetes": 1.3, "hypertension": 1.3,
        "cancer": 1.2, "gerd": 1.2, "gallstones": 1.3,
    },
    "elderly": {  # 65+
        "mi": 1.5, "stroke": 1.5, "pneumonia": 1.4,
        "uti": 1.3, "cancer": 1.4, "heart_failure": 1.5,
        "copd": 1.3, "atrial_fibrillation": 1.4,
        "pe": 1.3, "dvt": 1.3,
        "appendicitis": 0.7, "sti": 0.5,
    },
}

SEX_ADJUSTMENTS = {
    "female": {
        "uti": 1.5, "migraine": 1.3, "lupus": 1.5,
        "thyroid": 1.4, "gallstones": 1.3,
        "pregnancy_related": 2.0, "endometriosis": 1.5,
        "prostate": 0.0, "testicular": 0.0,
    },
    "male": {
        "mi": 1.3, "gout": 1.5, "kidney_stones": 1.3,
        "prostate": 1.5, "aortic_aneurysm": 1.3,
        "uti": 0.5, "lupus": 0.3,
        "pregnancy_related": 0.0, "endometriosis": 0.0,
    },
}


class ClinicalContext:
    """
    Processes patient clinical data and produces adjustments for NEXUS.
    """

    def __init__(self):
        pass

    def analyze(self, patient: dict) -> dict:
        """
        Analyze patient context and return clinical adjustments.

        patient = {
            "age": 65,
            "sex": "male",
            "pregnant": False,
            "labs": {"wbc": 18.0, "crp": 150, "pct": 3.5, "neutrophils": 88},
            "vitals": {"temperature": 39.2, "heart_rate": 110, "systolic_bp": 85, "spo2": 91},
            "history": ["diabetes", "hypertension"],
            "medications": ["metformin", "lisinopril"],
            "allergies": ["penicillin"],
        }
        """
        result = {
            "lab_analysis": [],
            "vital_analysis": [],
            "abnormal_labs": {},
            "abnormal_vitals": {},
            "disease_adjustments": {},
            "risk_scores": {},
            "triage_override": None,
            "warnings": [],
            "clinical_impression": [],
        }

        # ── Analyze labs ──
        labs = patient.get("labs", {})
        if labs:
            result["lab_analysis"], result["abnormal_labs"] = self._analyze_labs(labs)

        # ── Analyze vitals ──
        vitals = patient.get("vitals", {})
        if vitals:
            result["vital_analysis"], result["abnormal_vitals"] = self._analyze_vitals(vitals)

        # ── Age/sex adjustments ──
        age = patient.get("age")
        sex = patient.get("sex", "").lower()
        if age is not None:
            result["disease_adjustments"].update(self._age_adjustments(age))
        if sex:
            result["disease_adjustments"].update(self._sex_adjustments(sex))

        # ── Risk scores (qSOFA, SIRS) ──
        result["risk_scores"] = self._calculate_risk_scores(vitals, labs, age)

        # ── Triage override from vitals ──
        result["triage_override"] = self._triage_from_vitals(vitals, labs)

        # ── Medication/allergy warnings ──
        meds = patient.get("medications", [])
        allergies = patient.get("allergies", [])
        history = patient.get("history", [])
        result["warnings"] = self._check_warnings(meds, allergies, history)

        # ── Clinical impression ──
        result["clinical_impression"] = self._build_impression(result, patient)

        # ── Mechanism validation (which mechanism labs match) ──
        result["lab_validated_mechanisms"] = self._validate_mechanisms_by_labs(labs)

        return result

    def _analyze_labs(self, labs: dict) -> tuple:
        analysis = []
        abnormal = {}

        for lab_name, value in labs.items():
            if not isinstance(value, (int, float)):
                continue
            lab_key = lab_name.lower().replace(" ", "_")
            ref = LAB_RANGES.get(lab_key)
            if not ref:
                continue

            low, high, unit, high_meaning, low_meaning = ref

            if value > high:
                severity = "mildly elevated" if value <= high * 1.5 else "significantly elevated" if value <= high * 3 else "critically elevated"
                abnormal[lab_key] = {"value": value, "status": "high", "severity": severity}
                meaning = high_meaning or "abnormal"
                analysis.append(f"{lab_key}: {value} {unit} ({severity}) — suggests {meaning}")
            elif value < low:
                severity = "mildly low" if value >= low * 0.7 else "significantly low"
                abnormal[lab_key] = {"value": value, "status": "low", "severity": severity}
                meaning = low_meaning or "abnormal"
                analysis.append(f"{lab_key}: {value} {unit} ({severity}) — suggests {meaning}")
            # Normal values noted for important labs
            elif lab_key in ("pct", "troponin", "d_dimer", "lactate"):
                analysis.append(f"{lab_key}: {value} {unit} (normal)")

        return analysis, abnormal

    def _analyze_vitals(self, vitals: dict) -> tuple:
        analysis = []
        abnormal = {}

        for vital_name, value in vitals.items():
            if not isinstance(value, (int, float)):
                continue
            vital_key = vital_name.lower().replace(" ", "_")
            ref = VITAL_RANGES.get(vital_key)
            if not ref:
                continue

            low, high, unit, high_concern, low_concern = ref

            if value > high:
                abnormal[vital_key] = {"value": value, "status": "high"}
                concern = high_concern or "elevated"
                analysis.append(f"{vital_key}: {value} {unit} — {concern}")
            elif value < low:
                abnormal[vital_key] = {"value": value, "status": "low"}
                concern = low_concern or "low"
                analysis.append(f"{vital_key}: {value} {unit} — {concern}")

        return analysis, abnormal

    def _age_adjustments(self, age: int) -> dict:
        if age < 18:
            group = "pediatric"
        elif age < 40:
            group = "young_adult"
        elif age < 65:
            group = "middle_age"
        else:
            group = "elderly"
        return AGE_ADJUSTMENTS.get(group, {})

    def _sex_adjustments(self, sex: str) -> dict:
        return SEX_ADJUSTMENTS.get(sex, {})

    def _calculate_risk_scores(self, vitals: dict, labs: dict, age: int = None) -> dict:
        scores = {}

        # qSOFA (quick Sepsis-related Organ Failure Assessment)
        # 1 point each: RR >= 22, altered mentation, SBP <= 100
        qsofa = 0
        qsofa_criteria = []
        if vitals.get("respiratory_rate", 0) >= 22:
            qsofa += 1
            qsofa_criteria.append("respiratory rate >= 22")
        if vitals.get("systolic_bp", 999) <= 100:
            qsofa += 1
            qsofa_criteria.append("systolic BP <= 100")
        # Can't assess altered mentation from data alone
        scores["qsofa"] = {"score": qsofa, "max": 3, "criteria_met": qsofa_criteria,
                          "interpretation": "high sepsis risk" if qsofa >= 2 else "monitor"}

        # SIRS criteria (2+ = SIRS)
        sirs = 0
        sirs_criteria = []
        temp = vitals.get("temperature", 37)
        if temp > 38 or temp < 36:
            sirs += 1
            sirs_criteria.append(f"temperature {temp}°C")
        if vitals.get("heart_rate", 0) > 90:
            sirs += 1
            sirs_criteria.append("heart rate > 90")
        if vitals.get("respiratory_rate", 0) > 20:
            sirs += 1
            sirs_criteria.append("respiratory rate > 20")
        wbc = labs.get("wbc", 0)
        if wbc > 12 or wbc < 4:
            sirs += 1
            sirs_criteria.append(f"WBC {wbc}")
        scores["sirs"] = {"score": sirs, "max": 4, "criteria_met": sirs_criteria,
                         "interpretation": "SIRS present" if sirs >= 2 else "SIRS not met"}

        # NEWS2 (simplified)
        news = 0
        spo2 = vitals.get("spo2", 99)
        if spo2 <= 91: news += 3
        elif spo2 <= 93: news += 2
        elif spo2 <= 95: news += 1

        sbp = vitals.get("systolic_bp", 120)
        if sbp <= 90: news += 3
        elif sbp <= 100: news += 2
        elif sbp <= 110: news += 1
        elif sbp >= 220: news += 3

        hr = vitals.get("heart_rate", 75)
        if hr <= 40 or hr >= 131: news += 3
        elif hr <= 50 or hr >= 111: news += 2
        elif hr >= 91: news += 1

        if temp <= 35: news += 3
        elif temp >= 39.1: news += 2
        elif temp >= 38.1: news += 1
        elif temp <= 36: news += 1

        scores["news2"] = {"score": news, "max": 20,
                          "interpretation": "critical" if news >= 7 else "urgent" if news >= 5 else "moderate" if news >= 3 else "low"}

        return scores

    def _triage_from_vitals(self, vitals: dict, labs: dict) -> Optional[dict]:
        """Override triage level if vitals indicate emergency."""
        if not vitals:
            return None

        spo2 = vitals.get("spo2", 99)
        sbp = vitals.get("systolic_bp", 120)
        temp = vitals.get("temperature", 37)
        hr = vitals.get("heart_rate", 75)
        lactate = labs.get("lactate", 0)
        troponin = labs.get("troponin", 0)

        # Emergency criteria
        if spo2 < 90:
            return {"level": "emergency", "reason": f"SpO2 {spo2}% — severe hypoxemia"}
        if sbp < 80:
            return {"level": "emergency", "reason": f"SBP {sbp} mmHg — shock"}
        if lactate > 4:
            return {"level": "emergency", "reason": f"Lactate {lactate} — tissue hypoperfusion"}
        if troponin > 0.1:
            return {"level": "emergency", "reason": f"Troponin {troponin} — possible MI"}
        if temp > 41:
            return {"level": "emergency", "reason": f"Temperature {temp}°C — hyperpyrexia"}

        # Urgent criteria
        if spo2 < 94:
            return {"level": "urgent", "reason": f"SpO2 {spo2}% — hypoxemia"}
        if sbp < 90 or sbp > 180:
            return {"level": "urgent", "reason": f"SBP {sbp} mmHg — abnormal"}
        if hr > 120 or hr < 45:
            return {"level": "urgent", "reason": f"HR {hr} — abnormal"}
        if temp > 39.5:
            return {"level": "urgent", "reason": f"Temperature {temp}°C — high fever"}

        return None

    def _check_warnings(self, meds: list, allergies: list, history: list) -> list:
        warnings = []
        meds_lower = [m.lower() for m in meds]
        allergy_lower = [a.lower() for a in allergies]
        history_lower = [h.lower() for h in history]

        # Common drug-allergy interactions
        if "penicillin" in allergy_lower:
            for med in meds_lower:
                if any(abx in med for abx in ["amoxicillin", "ampicillin", "penicillin"]):
                    warnings.append(f"ALLERGY ALERT: Patient allergic to penicillin but taking {med}")

        # Chronic condition warnings
        if "diabetes" in history_lower:
            warnings.append("Chronic: Diabetes — monitor glucose, infection risk elevated")
        if "hypertension" in history_lower:
            warnings.append("Chronic: Hypertension — monitor BP, cardiovascular risk")
        if "copd" in history_lower or "asthma" in history_lower:
            warnings.append("Chronic: Respiratory disease — lower threshold for respiratory symptoms")
        if "immunosuppression" in history_lower or "hiv" in history_lower:
            warnings.append("Chronic: Immunocompromised — atypical presentations possible")
        if any("kidney" in h for h in history_lower):
            warnings.append("Chronic: Kidney disease — adjust drug doses, monitor creatinine")

        return warnings

    def _validate_mechanisms_by_labs(self, labs: dict) -> list:
        """Check which mechanism lab patterns match the patient's actual labs."""
        # Map from mechanism lab markers to actual lab values
        lab_mapping = {
            "crp_high": ("crp", 50, "above"),
            "crp_mild": ("crp", 10, "above"),
            "wbc_high_or_low": ("wbc", 12, "above"),
            "wbc_normal_or_low": ("wbc", 10, "below"),
            "lactate_high": ("lactate", 2, "above"),
            "blood_culture_positive": None,  # can't determine from lab values
            "alt_ast_high": ("alt", 56, "above"),
            "platelets_low": ("platelets", 150, "below"),
            "cd4_low_possible": ("cd4", 500, "below"),
            "dehydration_signs": None,
            "csf_wbc_high": None,
        }

        validated = []
        for mech_lab, mapping in lab_mapping.items():
            if mapping is None:
                continue
            lab_key, threshold, direction = mapping
            value = labs.get(lab_key)
            if value is None:
                continue
            if direction == "above" and value > threshold:
                validated.append({"mechanism_lab": mech_lab, "patient_value": value, "threshold": threshold, "match": True})
            elif direction == "below" and value < threshold:
                validated.append({"mechanism_lab": mech_lab, "patient_value": value, "threshold": threshold, "match": True})

        return validated

    def _build_impression(self, analysis: dict, patient: dict) -> list:
        impression = []
        age = patient.get("age")
        sex = patient.get("sex", "")

        if age:
            impression.append(f"{age}-year-old {sex}" if sex else f"{age}-year-old patient")

        # Lab summary
        abnormal_labs = analysis.get("abnormal_labs", {})
        if abnormal_labs:
            high_labs = [k for k, v in abnormal_labs.items() if v["status"] == "high"]
            low_labs = [k for k, v in abnormal_labs.items() if v["status"] == "low"]
            if high_labs:
                impression.append(f"Elevated: {', '.join(high_labs)}")
            if low_labs:
                impression.append(f"Low: {', '.join(low_labs)}")

        # Risk scores
        risk = analysis.get("risk_scores", {})
        qsofa = risk.get("qsofa", {})
        sirs = risk.get("sirs", {})
        news = risk.get("news2", {})

        if qsofa.get("score", 0) >= 2:
            impression.append(f"qSOFA {qsofa['score']}/3 — HIGH SEPSIS RISK")
        if sirs.get("score", 0) >= 2:
            impression.append(f"SIRS criteria met ({sirs['score']}/4)")
        if news.get("score", 0) >= 5:
            impression.append(f"NEWS2 {news['score']} — {news.get('interpretation', '')}")

        # Triage override
        override = analysis.get("triage_override")
        if override:
            impression.append(f"TRIAGE: {override['level'].upper()} — {override['reason']}")

        return impression