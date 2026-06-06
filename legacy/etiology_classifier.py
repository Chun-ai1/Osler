"""
Etiology Classifier — Virus vs Bacteria vs Non-Infectious

Implements the 4-layer clinical diagnostic approach:
  1. Clinical pattern analysis (symptom characteristics)
  2. Lab marker inference (WBC differential, CRP, PCT patterns)
  3. Mechanism matching (which mechanisms fire for these symptoms)
  4. Anatomical distribution (focal vs systemic)

Input: symptoms list + optional vitals/labs
Output: {
    "etiology": "viral" | "bacterial" | "non_infectious" | "uncertain",
    "confidence": 0.0-1.0,
    "scores": {"viral": 0.4, "bacterial": 0.3, "non_infectious": 0.2},
    "reasoning": [...],
    "lab_predictions": {...},
    "recommended_tests": [...],
}

Built from your mechanism data: 1200 bacteria + 1500 virus mechanisms
with their typical_symptoms, typical_labs, and typical_vitals.
"""

import json
import os
import glob
from collections import defaultdict, Counter


class EtiologyClassifier:

    def __init__(self):
        self._loaded = False
        # Symptom → etiology weights (learned from mechanism data)
        self.sym_viral = Counter()      # symptom → count in virus mechanisms
        self.sym_bacterial = Counter()  # symptom → count in bacteria mechanisms
        self.lab_viral = Counter()      # lab → count in virus mechanisms
        self.lab_bacterial = Counter()  # lab → count in bacteria mechanisms
        self.vital_viral = Counter()
        self.vital_bacterial = Counter()

    def load(self, knowledge_dir="medical_knowledge"):
        if self._loaded:
            return

        # Load bacteria mechanisms
        bac_path = os.path.join(knowledge_dir, "mechanisms", "bacteria_mechanisms_1200.json")
        if os.path.exists(bac_path):
            bac = json.load(open(bac_path, "r", encoding="utf-8"))
            for m in bac:
                for s in m.get("typical_symptoms", []):
                    self.sym_bacterial[s.lower()] += 1
                for lab in m.get("typical_labs", []):
                    self.lab_bacterial[lab.lower()] += 1
                for v in m.get("typical_vitals", []):
                    self.vital_bacterial[v.lower()] += 1

        # Load virus mechanisms
        vir_path = os.path.join(knowledge_dir, "mechanisms", "virus_mechanisms_1500.json")
        if os.path.exists(vir_path):
            vir = json.load(open(vir_path, "r", encoding="utf-8"))
            for m in vir:
                for s in m.get("typical_symptoms", []):
                    self.sym_viral[s.lower()] += 1
                for lab in m.get("typical_labs", []):
                    self.lab_viral[lab.lower()] += 1
                for v in m.get("typical_vitals", []):
                    self.vital_viral[v.lower()] += 1

        self._loaded = True

    def classify(self, symptoms: list, vitals: dict = None, labs: dict = None) -> dict:
        """
        Classify etiology from symptoms + optional vitals/labs.

        Args:
            symptoms: ["headache", "fever", "cough"]
            vitals: {"temperature": 39.5, "heart_rate": 110} (optional)
            labs: {"wbc": 15.0, "neutrophils_pct": 85, "crp": 120, "pct": 2.5} (optional)
        """
        self.load()
        sym_set = set(s.lower().strip().replace(" ", "_") for s in symptoms)
        reasoning = []

        # ═══ Layer 1: Clinical symptom pattern ═══
        viral_score, bacterial_score, noninf_score = 0.0, 0.0, 0.0
        # ── MINIMUM INFECTION EVIDENCE GATE ──────────────────────────────
        # If there are no symptoms that suggest an infectious process,
        # the viral/bacterial distinction is meaningless — return uncertain.
        # Single symptoms like "swelling", "chest pain", "dizziness" alone
        # cannot anchor an etiology claim regardless of classifier scores.
        _INFECTION_SUPPORTING = {
            "fever", "chills", "cough", "sore throat", "runny nose", "congestion",
            "dysuria", "frequency", "urgency", "diarrhea", "vomiting",
            "productive cough", "purulent discharge", "rash", "body aches",
            "lymphadenopathy", "night sweats", "productive_cough", "purulent_discharge",
            "sore_throat", "runny_nose", "body_aches", "night_sweats",
        }
        _has_infection_context = bool(sym_set & _INFECTION_SUPPORTING)

        layer1 = self._analyze_symptom_pattern(sym_set)
        viral_score += layer1["viral"]
        bacterial_score += layer1["bacterial"]
        noninf_score += layer1["non_infectious"]
        reasoning.extend(layer1["reasoning"])

        # ═══ Layer 2: Mechanism matching (data-driven) ═══
        layer2 = self._mechanism_score(sym_set)
        viral_score += layer2["viral"] * 0.4
        bacterial_score += layer2["bacterial"] * 0.4
        reasoning.extend(layer2["reasoning"])

        # ═══ Layer 3: Lab inference (if labs provided) ═══
        layer3_result = None
        if labs:
            layer3 = self._analyze_labs(labs)
            viral_score += layer3["viral"]
            bacterial_score += layer3["bacterial"]
            noninf_score += layer3["non_infectious"]
            reasoning.extend(layer3["reasoning"])
            layer3_result = layer3

        # ═══ Layer 4: Anatomical distribution ═══
        layer4 = self._analyze_distribution(sym_set)
        viral_score += layer4["viral"]
        bacterial_score += layer4["bacterial"]
        noninf_score += layer4["non_infectious"]
        reasoning.extend(layer4["reasoning"])

        # ═══ Normalize scores ═══
        total = max(viral_score + bacterial_score + noninf_score, 0.01)
        scores = {
            "viral": round(viral_score / total, 3),
            "bacterial": round(bacterial_score / total, 3),
            "non_infectious": round(noninf_score / total, 3),
        }

        # Determine etiology
        best = max(scores, key=scores.get)
        confidence = scores[best]

        # Gate 1: pure-nonspecific symptoms cannot support a viral/bacterial call.
        # chills, fatigue, malaise, nausea, dizziness alone are too vague —
        # they appear in both viral and bacterial disease equally.
        _PURE_NONSPECIFIC = {
            "chills", "fatigue", "malaise", "weakness", "loss of appetite",
            "nausea", "dizziness", "headache", "sweating", "feeling unwell",
            "tiredness", "lethargy",
        }
        _all_nonspecific = bool(sym_set) and all(s in _PURE_NONSPECIFIC for s in sym_set)

        # Gate 2: without any infection-supporting symptoms, viral/bacterial claim unreliable.
        if _all_nonspecific and best in ("viral", "bacterial"):
            etiology = "uncertain"
            confidence = 0.0
        elif not _has_infection_context and best in ("viral", "bacterial"):
            etiology = "uncertain"
            confidence = max(scores.values()) * 0.5
        elif confidence < 0.45:
            etiology = "uncertain"
            confidence = max(scores.values())
        else:
            etiology = best

        # ═══ Predicted lab values ═══
        lab_predictions = self._predict_labs(etiology)

        # ═══ Recommended tests ═══
        recommended = self._recommend_tests(etiology, confidence, labs is not None)

        # When uncertain: clear misleading score values so they don't
        # look like final probabilities (e.g. B=1.0 alongside uncertain).
        if etiology == "uncertain":
            output_scores = {
                "viral": 0.0,
                "bacterial": 0.0,
                "non_infectious": 0.0,
                "_note": "scores suppressed — insufficient infection evidence",
            }
        else:
            output_scores = scores

        return {
            "etiology": etiology,
            "confidence": round(confidence, 3),
            "scores": output_scores,
            "reasoning": reasoning,
            "lab_predictions": lab_predictions,
            "recommended_tests": recommended,
        }

    # ═══════════════════════════════════════
    # Layer 1: Clinical symptom patterns
    # ═══════════════════════════════════════

    def _analyze_symptom_pattern(self, sym_set: set) -> dict:
        reasoning = []
        viral, bacterial, noninf = 0.0, 0.0, 0.0

        # Viral indicators: systemic, diffuse, multi-system
        # Data-derived from virus_mechanisms_1500.json
        viral_indicators = {"congestion", "dark_urine", "dizziness", "fatigue", "hoarseness", "jaundice", "myalgia", "nerve_pain", "opportunistic_infections", "painful_lesions", "prolonged_symptoms", "rash", "recurrent_lesions", "right_upper_quadrant_pain", "runny_nose", "severe_dehydration", "sore_throat", "vesicles", "vesicular_rash", "vomiting", "wheezing"}
        viral_hits = sym_set & viral_indicators
        if viral_hits:
            viral += len(viral_hits) * 0.15
            reasoning.append(f"Viral pattern: systemic symptoms present ({', '.join(viral_hits)})")

        # Bacterial indicators: focal, localized, purulent
        # Data-derived from bacteria_mechanisms_1200.json
        # (symptoms <= 25% viral mechanisms, min 10 occurrences)
        bacterial_indicators = {"bleeding", "dysuria", "easy_bruising", "edema", "erythema", "frequency", "hypotension", "hypoxia", "lightheadedness", "necrosis", "pain", "persistent_fever", "photophobia", "pleuritic_chest_pain", "rigors", "severe_pain", "suprapubic_pain", "swelling", "syncope", "weakness", "worsening_symptoms"}
        bacterial_hits = sym_set & bacterial_indicators
        if bacterial_hits:
            bacterial += len(bacterial_hits) * 0.2
        # Cellulitis cluster: swelling only counts with erythema/warmth
        if "swelling" in sym_set and ({"erythema", "warmth", "redness"} & sym_set):
            bacterial += 0.2
            reasoning.append(f"Bacterial pattern: focal/localized symptoms ({', '.join(bacterial_hits)})")

        # Non-infectious indicators
        noninf_indicators = {
            "joint_pain", "stiffness", "reduced_range_of_motion",
            "symmetrical", "chronic", "weight_loss",
            "dry_skin", "hair_loss",  # autoimmune
            "itching", "hives",  # allergic
        }
        noninf_hits = sym_set & noninf_indicators
        if noninf_hits:
            noninf += len(noninf_hits) * 0.15
            reasoning.append(f"Non-infectious pattern: inflammatory/autoimmune features ({', '.join(noninf_hits)})")

        # Fever pattern
        if "fever" in sym_set or "chills" in sym_set:
            # Fever + multi-system = likely viral
            if len(viral_hits) >= 2:
                viral += 0.15
                reasoning.append("Fever with multiple systemic symptoms suggests viral")
            # Fever + focal = likely bacterial
            elif len(bacterial_hits) >= 1:
                bacterial += 0.2
                reasoning.append("Fever with focal symptoms suggests bacterial")
            else:
                viral += 0.1
                bacterial += 0.1

        # Headache context
        if "headache" in sym_set:
            if "neck_stiffness" in sym_set or "photophobia" in sym_set:
                bacterial += 0.3
                reasoning.append("Headache + neck stiffness/photophobia: bacterial meningitis pattern")
            elif "nausea" in sym_set or "fatigue" in sym_set:
                viral += 0.1
                reasoning.append("Headache + systemic symptoms: common in viral infections")

        # Rash context
        if "rash" in sym_set:
            if "fever" in sym_set:
                viral += 0.15
                reasoning.append("Fever + rash: common viral exanthem pattern")
            if "joint_pain" in sym_set:
                noninf += 0.2
                reasoning.append("Rash + joint pain: consider autoimmune (e.g., lupus)")

        return {"viral": viral, "bacterial": bacterial, "non_infectious": noninf, "reasoning": reasoning}

    # ═══════════════════════════════════════
    # Layer 2: Mechanism matching (data-driven)
    # ═══════════════════════════════════════

    def _mechanism_score(self, sym_set: set) -> dict:
        """Score based on how many virus vs bacteria mechanisms match these symptoms."""
        reasoning = []

        viral_match = 0
        bacterial_match = 0

        for sym in sym_set:
            vc = self.sym_viral.get(sym, 0)
            bc = self.sym_bacterial.get(sym, 0)
            viral_match += vc
            bacterial_match += bc

        total = max(viral_match + bacterial_match, 1)
        v_ratio = viral_match / total
        b_ratio = bacterial_match / total

        if viral_match > bacterial_match * 1.5:
            reasoning.append(
                f"Mechanism analysis: {viral_match} virus mechanisms vs {bacterial_match} bacteria "
                f"mechanisms match these symptoms (viral ratio: {v_ratio:.0%})")
        elif bacterial_match > viral_match * 1.5:
            reasoning.append(
                f"Mechanism analysis: {bacterial_match} bacteria mechanisms vs {viral_match} virus "
                f"mechanisms match (bacterial ratio: {b_ratio:.0%})")
        else:
            reasoning.append(
                f"Mechanism analysis: virus={viral_match}, bacteria={bacterial_match} — "
                f"both plausible, further evaluation needed")

        return {
            "viral": v_ratio,
            "bacterial": b_ratio,
            "reasoning": reasoning,
        }

    # ═══════════════════════════════════════
    # Layer 3: Lab analysis
    # ═══════════════════════════════════════

    def _analyze_labs(self, labs: dict) -> dict:
        """Analyze lab values for etiology clues."""
        reasoning = []
        viral, bacterial, noninf = 0.0, 0.0, 0.0

        wbc = labs.get("wbc")  # K/uL
        neut_pct = labs.get("neutrophils_pct")  # %
        lymph_pct = labs.get("lymphocytes_pct")  # %
        crp = labs.get("crp")  # mg/L
        pct = labs.get("pct") or labs.get("procalcitonin")  # ng/mL
        esr = labs.get("esr")  # mm/hr

        # WBC differential
        if wbc is not None:
            if wbc > 12:
                bacterial += 0.25
                reasoning.append(f"WBC {wbc:.1f} K/uL (elevated) — suggests bacterial infection")
            elif wbc < 4:
                viral += 0.2
                reasoning.append(f"WBC {wbc:.1f} K/uL (low) — may suggest viral or severe infection")
            else:
                reasoning.append(f"WBC {wbc:.1f} K/uL (normal range)")

        if neut_pct is not None:
            if neut_pct > 80:
                bacterial += 0.3
                reasoning.append(f"Neutrophils {neut_pct}% (high) — strong bacterial indicator")
            elif neut_pct < 40 and lymph_pct and lymph_pct > 40:
                viral += 0.25
                reasoning.append(f"Lymphocyte predominance ({lymph_pct}%) — typical viral pattern")

        # CRP
        if crp is not None:
            if crp > 100:
                bacterial += 0.3
                reasoning.append(f"CRP {crp} mg/L (very high) — strong bacterial infection marker")
            elif crp > 40:
                bacterial += 0.15
                viral += 0.05
                reasoning.append(f"CRP {crp} mg/L (elevated) — infection likely, bacterial more probable")
            elif crp > 10:
                viral += 0.1
                reasoning.append(f"CRP {crp} mg/L (mildly elevated) — common in viral infections")
            else:
                noninf += 0.1
                reasoning.append(f"CRP {crp} mg/L (normal) — less likely severe infection")

        # Procalcitonin — the key discriminator
        if pct is not None:
            if pct > 2.0:
                bacterial += 0.5
                reasoning.append(f"Procalcitonin {pct} ng/mL (very high) — highly specific for bacterial infection")
            elif pct > 0.5:
                bacterial += 0.3
                reasoning.append(f"Procalcitonin {pct} ng/mL (elevated) — bacterial infection likely")
            elif pct > 0.1:
                reasoning.append(f"Procalcitonin {pct} ng/mL (borderline) — possible bacterial, monitor")
            else:
                viral += 0.2
                noninf += 0.15
                reasoning.append(f"Procalcitonin {pct} ng/mL (low) — bacterial infection unlikely")

        return {"viral": viral, "bacterial": bacterial, "non_infectious": noninf, "reasoning": reasoning}

    # ═══════════════════════════════════════
    # Layer 4: Anatomical distribution
    # ═══════════════════════════════════════

    def _analyze_distribution(self, sym_set: set) -> dict:
        """Viral = systemic/diffuse, Bacterial = focal/localized."""
        reasoning = []
        viral, bacterial, noninf = 0.0, 0.0, 0.0

        # Count how many body systems are involved
        system_map = {
            "respiratory": {"cough", "wheezing", "dyspnea", "sore_throat", "congestion", "runny_nose"},
            "gi": {"nausea", "vomiting", "diarrhea", "abdominal_pain"},
            "neuro": {"headache", "dizziness", "numbness", "confusion", "photophobia"},
            "msk": {"joint_pain", "myalgia", "body_aches", "weakness", "stiffness"},
            "skin": {"rash", "redness", "itching", "swelling", "erythema"},
            "urinary": {"dysuria", "frequency", "urgency", "flank_pain"},
            "cardiac": {"chest_pain", "palpitations"},
            "systemic": {"fever", "chills", "fatigue", "malaise", "weight_loss"},
        }

        active_systems = set()
        for system, syms in system_map.items():
            if sym_set & syms:
                active_systems.add(system)

        n_systems = len(active_systems)

        if n_systems >= 3:
            viral += 0.2
            reasoning.append(
                f"Multi-system involvement ({n_systems} systems: {', '.join(sorted(active_systems))}) "
                f"— typical of systemic viral infection")
        elif n_systems == 1 and "systemic" not in active_systems:
            bacterial += 0.15
            system_name = list(active_systems)[0] if active_systems else "unknown"
            reasoning.append(
                f"Single-system involvement ({system_name}) — more consistent with focal bacterial infection")
        elif n_systems == 2:
            if "systemic" in active_systems:
                other = (active_systems - {"systemic"}).pop() if len(active_systems) > 1 else "unknown"
                reasoning.append(
                    f"Systemic symptoms + {other} involvement — could be either viral or bacterial")
                viral += 0.05
                bacterial += 0.05

        # Check for unilateral/bilateral patterns
        unilateral_hints = {"one_sided", "unilateral", "localized"}
        if sym_set & unilateral_hints:
            bacterial += 0.1
            reasoning.append("Unilateral/localized presentation suggests bacterial")

        return {"viral": viral, "bacterial": bacterial, "non_infectious": noninf, "reasoning": reasoning}

    # ═══════════════════════════════════════
    # Predictions & recommendations
    # ═══════════════════════════════════════

    def _predict_labs(self, etiology: str) -> dict:
        """Predict what lab values you'd expect for this etiology."""
        if etiology == "viral":
            return {
                "expected_wbc": "normal or low (4-10 K/uL)",
                "expected_differential": "lymphocyte predominance (>40%)",
                "expected_crp": "normal or mildly elevated (<40 mg/L)",
                "expected_pct": "low (<0.1 ng/mL)",
                "expected_esr": "normal or mildly elevated",
            }
        elif etiology == "bacterial":
            return {
                "expected_wbc": "elevated (>12 K/uL) with left shift",
                "expected_differential": "neutrophil predominance (>80%)",
                "expected_crp": "significantly elevated (>100 mg/L)",
                "expected_pct": "elevated (>0.5 ng/mL)",
                "expected_esr": "elevated",
            }
        elif etiology == "non_infectious":
            return {
                "expected_wbc": "normal or mildly elevated",
                "expected_differential": "varies by condition",
                "expected_crp": "may be elevated in autoimmune flares",
                "expected_pct": "low (<0.1 ng/mL)",
                "expected_esr": "may be elevated in autoimmune conditions",
            }
        return {}

    def _recommend_tests(self, etiology: str, confidence: float, has_labs: bool) -> list:
        """Recommend diagnostic tests based on current assessment."""
        tests = []

        if not has_labs:
            tests.append("Complete blood count (CBC) with differential")
            tests.append("C-reactive protein (CRP)")

        if confidence < 0.6:
            tests.append("Procalcitonin (PCT) — best discriminator for bacterial vs viral")

        if etiology == "bacterial":
            tests.append("Blood cultures (x2 sets) before antibiotics")
            tests.append("Site-specific culture (urine/sputum/wound as appropriate)")
        elif etiology == "viral":
            tests.append("Rapid viral panel / PCR if available")
            tests.append("Consider specific serology based on clinical picture")
        elif etiology == "non_infectious":
            tests.append("ESR (erythrocyte sedimentation rate)")
            tests.append("ANA panel if autoimmune suspected")
            tests.append("Complement levels (C3, C4)")

        return tests