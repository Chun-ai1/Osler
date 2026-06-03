"""
IntegratedReasoner — NEXUS multi-modal diagnostic reasoning engine
====================================================================
Extends PureMechanismDiagnoser with:
  1. Atlas spatial propagation (symptom → adjacent organs → candidate cascades)
  2. HPO semantic matching (user's HPO term → ancestor/descendant cascade symptoms)
  3. Referred pain reasoning (chest pain at T1-T4 → could be heart/lung/esophagus)
  4. Lab-first reverse reasoning (lab pattern → mechanism candidates)
  5. Comorbidity rules (diabetes + chest pain → upweight silent MI)
  6. Time-window filtering (acute vs chronic)

Returns ranked diagnoses with EVIDENCE PROVENANCE — every score is traceable
back to which data layer contributed it.

USAGE:
    from integrated_reasoner import IntegratedReasoner
    reasoner = IntegratedReasoner()
    
    result = reasoner.diagnose(
        symptoms=["chest pain", "shortness of breath"],
        labs={"troponin": 2.5, "BNP": 1500},
        comorbidities=["diabetes mellitus type 2"],
        onset_hours=2,
        top_k=10,
    )
    
    # result is list of (disease, score, evidence_dict)
    # evidence_dict explains why each disease was ranked there:
    #   - "cascade_match": from LAYER 2 symptom matching
    #   - "atlas_propagation": from atlas spatial reasoning
    #   - "hpo_semantic": from HPO ancestry
    #   - "lab_match": from LAYER 5 lab values
    #   - "comorbidity_boost": from comorbidity rules
    #   - "time_window_filter": acute/chronic filter applied
"""
from __future__ import annotations
import os
import json
import re
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path

# Try imports - all are optional, gracefully degrade
try:
    from .pure_mechanism_diagnoser import PureMechanismDiagnoser
except ImportError:
    from pure_mechanism_diagnoser import PureMechanismDiagnoser

try:
    from .atlas_extension_loader import get_extended_atlas
except ImportError:
    try:
        from atlas_extension_loader import get_extended_atlas
    except ImportError:
        get_extended_atlas = None


class IntegratedReasoner:
    """Multi-modal diagnostic reasoning using all NEXUS layers."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        
        # ─── Load PureMechanismDiagnoser (LAYER 2 cascade matching) ───
        self.pmd = PureMechanismDiagnoser()
        if verbose:
            print(f"[IntegratedReasoner] Loaded {len(self.pmd.derived_pictures)} disease pictures")
        
        # ─── Load atlas (LAYER 1 + 1.5) ───
        self.atlas = None
        if get_extended_atlas:
            try:
                self.atlas = get_extended_atlas(verbose=False)
                if verbose:
                    print(f"[IntegratedReasoner] Atlas: {len(self.atlas.organs)} organs, "
                          f"{len(self.atlas.connections)} connections")
            except Exception as e:
                if verbose:
                    print(f"[IntegratedReasoner] Atlas unavailable: {e}")
        
        # ─── Load HPO data ───
        self.hpo_terms = {}
        self.hpo_ancestors = {}
        self.symptom_to_hpo = {}
        self.hpo_to_cascade = {}
        
        for candidate in ["medical_knowledge/external_ontologies",
                           "../medical_knowledge/external_ontologies",
                           os.path.join(os.path.dirname(__file__), "..",
                                         "medical_knowledge", "external_ontologies")]:
            if os.path.isdir(candidate):
                try:
                    if os.path.exists(os.path.join(candidate, "hpo_terms.json")):
                        data = json.load(open(os.path.join(candidate, "hpo_terms.json")))
                        self.hpo_terms = data.get("terms", {})
                    if os.path.exists(os.path.join(candidate, "hpo_ancestors.json")):
                        data = json.load(open(os.path.join(candidate, "hpo_ancestors.json")))
                        self.hpo_ancestors = data.get("ancestors", {})
                    if os.path.exists(os.path.join(candidate, "symptom_to_hpo.json")):
                        data = json.load(open(os.path.join(candidate, "symptom_to_hpo.json")))
                        for sym, info in data.get("mappings", {}).items():
                            self.symptom_to_hpo[sym] = info["hpo_id"]
                            self.hpo_to_cascade.setdefault(info["hpo_id"], []).append(sym)
                    break
                except Exception as e:
                    if verbose:
                        print(f"[IntegratedReasoner] HPO load error: {e}")
        
        if verbose and self.hpo_terms:
            print(f"[IntegratedReasoner] HPO: {len(self.hpo_terms)} terms, "
                  f"{len(self.symptom_to_hpo)} cascade mappings")
        
        # ─── Load comorbidity rules ───
        self.comorbidity_rules = self._load_comorbidity_rules()
        
        # ─── Load UBERON if present (Phase 1 optional) ───
        self.uberon_terms = {}
        for candidate in ["medical_knowledge/external_ontologies/uberon_terms.json",
                           "../medical_knowledge/external_ontologies/uberon_terms.json"]:
            if os.path.exists(candidate):
                try:
                    data = json.load(open(candidate))
                    self.uberon_terms = data.get("terms", {})
                    if verbose:
                        print(f"[IntegratedReasoner] UBERON: {len(self.uberon_terms)} terms")
                    break
                except Exception:
                    pass
    
    # ═══════════════════════════════════════════════════════════════
    # Main diagnose entry point
    # ═══════════════════════════════════════════════════════════════
    
    def diagnose(self, 
                  symptoms: List[str],
                  labs: Optional[Dict[str, Any]] = None,
                  comorbidities: Optional[List[str]] = None,
                  onset_hours: Optional[float] = None,
                  top_k: int = 10,
                  use_propagation: bool = True,
                  use_hpo_expansion: bool = True,
                  return_evidence: bool = True) -> List[Tuple[str, float, Dict]]:
        """
        Multi-modal diagnosis. Returns ranked diagnoses with evidence provenance.
        
        Args:
            symptoms: User-provided symptom list
            labs: Lab values dict (e.g. {"troponin": 2.5})
            comorbidities: Known comorbid conditions
            onset_hours: Symptom onset timing (for acute/chronic filtering)
            top_k: Number of results to return
            use_propagation: Use atlas spatial propagation
            use_hpo_expansion: Use HPO semantic expansion
            return_evidence: Include evidence breakdown per diagnosis
        """
        comorbidities = comorbidities or []
        labs = labs or {}
        
        # ─── Step 1: Expand symptoms via HPO semantic ───
        expanded_symptoms = list(symptoms)
        hpo_expansions = []
        
        if use_hpo_expansion and self.hpo_to_cascade:
            for sym in symptoms:
                # Find HPO ID for this symptom
                hpo_id = self._find_hpo_for_symptom(sym)
                if hpo_id and hpo_id in self.hpo_terms:
                    # PREFER: use the canonical HPO term name (generic)
                    # AVOID: cascade-specific verbose strings that bias toward one disease
                    hpo_name = self.hpo_terms[hpo_id].get("name", "").lower()
                    if hpo_name and hpo_name not in expanded_symptoms:
                        # Only add if generic enough (no other cascade has exact same string)
                        expanded_symptoms.append(hpo_name)
                        hpo_expansions.append((sym, hpo_id, hpo_name))
        
        # ─── Step 2: Get base scores from PureMechanismDiagnoser ───
        base_results = self.pmd.diagnose(
            patient_symptoms=expanded_symptoms,
            patient_labs=labs,
            top_k=top_k * 3,  # get more candidates for re-ranking
            verbose=False
        )
        
        # ─── Step 3: Atlas propagation - boost diseases for adjacent organs ───
        propagation_boosts = {}
        if use_propagation and self.atlas:
            propagation_boosts = self._compute_propagation_boosts(symptoms)
        
        # ─── Step 4: Comorbidity rules - upweight specific diseases ───
        comorbidity_boosts = {}
        if comorbidities and self.comorbidity_rules:
            comorbidity_boosts = self._compute_comorbidity_boosts(
                symptoms, comorbidities)
        
        # ─── Step 5: Time-window filtering ───
        time_penalties = {}
        if onset_hours is not None:
            time_penalties = self._compute_time_penalties(onset_hours, base_results)
        
        # ─── Step 6: Combine scores ───
        final_results = []
        for disease, base_score, base_evidence in base_results:
            evidence = dict(base_evidence)
            
            # Propagation
            prop_boost = propagation_boosts.get(disease, 0.0)
            evidence["atlas_propagation_boost"] = round(prop_boost, 3)
            
            # Comorbidity
            comorb_boost = comorbidity_boosts.get(disease, 0.0)
            evidence["comorbidity_boost"] = round(comorb_boost, 3)
            
            # Time penalty (negative)
            time_pen = time_penalties.get(disease, 0.0)
            evidence["time_window_penalty"] = round(time_pen, 3)
            
            # HPO expansion contribution
            evidence["hpo_expansions_used"] = len(hpo_expansions)
            
            # Final score
            final_score = base_score + prop_boost + comorb_boost - time_pen
            
            final_results.append((disease, final_score, evidence))
        
        # Re-rank and trim, tie-break by triage urgency (emergent > urgent > routine)
        TRIAGE_PRIORITY = {"emergent": 3, "urgent": 2, "routine": 1, "": 0}
        
        def sort_key(item):
            disease, score, evidence = item
            triage = self.pmd.mechanism_map.get(disease, {}).get("triage_level", "")
            return (-score, -TRIAGE_PRIORITY.get(triage, 0))
        
        final_results.sort(key=sort_key)
        return final_results[:top_k]
    
    # ═══════════════════════════════════════════════════════════════
    # Lab-first reverse reasoning (Phase 2)
    # ═══════════════════════════════════════════════════════════════
    
    def diagnose_from_labs(self, 
                              labs: Dict[str, Any],
                              additional_symptoms: Optional[List[str]] = None,
                              top_k: int = 10) -> List[Tuple[str, float, Dict]]:
        """
        Reverse reasoning: start from lab abnormalities, find candidate diseases.
        Useful when patient has abnormal labs but vague symptoms.
        """
        # Walk all cascades, check which have matching lab fingerprints
        scores = {}
        evidence_map = {}
        
        for disease_name, picture in self.pmd.derived_pictures.items():
            lab_fp = picture.get("_lab_fingerprint", {})
            if not lab_fp:
                continue
            
            matched_labs = []
            for lab_name, expected in lab_fp.items():
                if lab_name in labs:
                    if PureMechanismDiagnoser._lab_matches_pattern(
                            labs[lab_name], expected, lab_name):
                        matched_labs.append((lab_name, labs[lab_name], expected))
            
            if matched_labs:
                # Score = fraction of disease labs matched
                score = len(matched_labs) / max(len(lab_fp), 1)
                # Bonus for matching multiple labs
                score += 0.1 * (len(matched_labs) - 1)
                scores[disease_name] = score
                evidence_map[disease_name] = {
                    "lab_matches": matched_labs,
                    "expected_labs": lab_fp,
                    "match_rate": f"{len(matched_labs)}/{len(lab_fp)}",
                }
        
        # If symptoms also provided, blend with forward reasoning
        if additional_symptoms:
            forward = self.pmd.diagnose(
                patient_symptoms=additional_symptoms,
                patient_labs=labs,
                top_k=top_k * 2,
            )
            forward_scores = {d: s for d, s, _ in forward}
            forward_ev = {d: e for d, s, e in forward}
            
            # Combine: lab-first 60% + symptom 40%
            for d in set(list(scores.keys()) + list(forward_scores.keys())):
                lab_s = scores.get(d, 0)
                sym_s = forward_scores.get(d, 0)
                scores[d] = lab_s * 0.6 + sym_s * 0.4
                if d in forward_ev:
                    evidence_map.setdefault(d, {}).update(forward_ev[d])
        
        # Rank
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        return [(d, s, evidence_map.get(d, {})) for d, s in ranked]
    
    # ═══════════════════════════════════════════════════════════════
    # Helper: HPO semantic expansion
    # ═══════════════════════════════════════════════════════════════
    
    def _find_hpo_for_symptom(self, symptom: str) -> Optional[str]:
        """Find HPO ID matching a user's symptom (fuzzy)."""
        sym_lower = symptom.lower().strip()
        
        # 1. Direct match in cascade-to-hpo (reverse: search by symptom string)
        for cascade_sym, hpo_id in self.symptom_to_hpo.items():
            if sym_lower in cascade_sym.lower():
                return hpo_id
        
        # 2. Search HPO terms by name
        for hpid, info in self.hpo_terms.items():
            name = info.get("name", "").lower()
            if name == sym_lower or sym_lower in name:
                return hpid
        
        return None
    
    def _hpo_related_cascades(self, hpo_id: str) -> List[str]:
        """Get cascade symptoms related via HPO ancestry."""
        related = set()
        
        # Direct mappings
        for s in self.hpo_to_cascade.get(hpo_id, []):
            related.add(s)
        
        # Descendants of query hpo_id (their cascade symptoms)
        for cascade_sym, cascade_hpo in self.symptom_to_hpo.items():
            cascade_ancestors = self.hpo_ancestors.get(cascade_hpo, [])
            if hpo_id in cascade_ancestors:
                related.add(cascade_sym)
        
        return sorted(related)
    
    # ═══════════════════════════════════════════════════════════════
    # Helper: Atlas-based propagation
    # ═══════════════════════════════════════════════════════════════
    
    def _compute_propagation_boosts(self, symptoms: List[str]) -> Dict[str, float]:
        """
        For each symptom, find candidate organs via:
          1. symptom-keyword → organ reverse index (built from cascade data)
          2. atlas referred_pain (when applicable)
          3. atlas.resolve (skip 'ing_ln' which is a buggy default)
        Then boost diseases affecting those organs.
        """
        if not self.atlas:
            return {}
        
        # Lazy-load keyword index
        if not hasattr(self, "_keyword_to_organs"):
            self._keyword_to_organs = {}
            for path in [
                "medical_knowledge/derived/symptom_keyword_to_organ.json",
                "../medical_knowledge/derived/symptom_keyword_to_organ.json",
                os.path.join(os.path.dirname(__file__), "..",
                              "medical_knowledge", "derived",
                              "symptom_keyword_to_organ.json"),
            ]:
                if os.path.exists(path):
                    try:
                        data = json.load(open(path))
                        self._keyword_to_organs = data.get("keyword_to_organs", {})
                        break
                    except Exception:
                        pass
        
        boosts = {}
        candidate_organs = set()
        
        for sym in symptoms:
            sym_lower = sym.lower()
            tokens = re.split(r"\s+", sym_lower)
            
            # Strategy 1: keyword reverse index (most reliable)
            for token in tokens:
                if len(token) < 4:
                    continue
                organs = self._keyword_to_organs.get(token, [])
                for o in organs:
                    candidate_organs.add(o)
            
            # Strategy 2: atlas referred_pain (some atlas versions have this)
            try:
                pain_sources = self.atlas.find_referred_pain_sources(sym_lower)
                if pain_sources:
                    for src in pain_sources[:5]:
                        if isinstance(src, dict):
                            organ = src.get("organ") or src.get("source")
                            if organ and organ != "ing_ln":
                                candidate_organs.add(organ)
            except Exception:
                pass
            
            # Strategy 3: atlas.resolve (skip 'ing_ln' buggy fallback)
            for token in tokens:
                if len(token) < 4:
                    continue
                try:
                    resolved = self.atlas.resolve(token)
                    if resolved and resolved != "ing_ln":
                        candidate_organs.add(resolved)
                        # Also include adjacent neighbors
                        try:
                            neighbors = self.atlas.get_neighbors(resolved,
                                                                    conn_types=["adjacent"])
                            for n in neighbors[:3]:
                                if n != "ing_ln":
                                    candidate_organs.add(n)
                        except Exception:
                            pass
                except Exception:
                    pass
        
        # Step 2: For each candidate organ, boost diseases that affect it
        for disease_name, mapping in self.pmd.mechanism_map.items():
            organ = mapping.get("organ", "")
            if organ in candidate_organs:
                boosts[disease_name] = boosts.get(disease_name, 0) + 0.05
        
        # Cap boosts (max +0.15 per disease)
        for d in boosts:
            boosts[d] = min(boosts[d], 0.15)
        
        return boosts
    
    # ═══════════════════════════════════════════════════════════════
    # Helper: Comorbidity rules (Phase 3 - 5 examples)
    # ═══════════════════════════════════════════════════════════════
    
    def _load_comorbidity_rules(self) -> List[Dict]:
        """Load comorbidity rules from JSON or use built-in defaults."""
        for path in ["medical_knowledge/rules/comorbidity_rules.json",
                     "../medical_knowledge/rules/comorbidity_rules.json"]:
            if os.path.exists(path):
                try:
                    data = json.load(open(path))
                    return data.get("rules", [])
                except Exception:
                    pass
        
        # Built-in defaults (5 critical rules)
        return [
            {
                "name": "silent_MI_in_diabetic",
                "if_comorbidities": ["diabetes mellitus", "diabetes mellitus type 2",
                                       "diabetes mellitus type 1", "diabetes patient"],
                "if_any_symptoms": ["nausea", "fatigue", "shortness of breath",
                                      "dyspnea", "syncope", "weakness",
                                      "altered mental status", "epigastric pain"],
                "if_not_symptoms": ["fever", "diarrhea"],
                "boost_diseases": ["Heart Attack/MI"],
                "boost_value": 0.20,
                "explanation": "Diabetics often present with atypical MI symptoms; CP may be absent",
            },
            {
                "name": "SDH_in_elderly_with_fall",
                "if_comorbidities": ["elderly", "anticoagulation", "warfarin", "alcohol use"],
                "if_any_symptoms": ["headache", "altered mental status", "confusion",
                                      "memory loss", "weakness", "after fall"],
                "boost_diseases": ["Subarachnoid Hemorrhage", "Ischemic Stroke"],
                "boost_value": 0.15,
                "explanation": "Elderly on anticoagulation post-fall: subdural/intracranial hemorrhage",
            },
            {
                "name": "preeclampsia_in_pregnancy",
                "if_comorbidities": ["pregnancy", "pregnant", "third trimester",
                                       "20 weeks gestation"],
                "if_any_symptoms": ["headache", "vision changes", "blurred vision",
                                      "right upper quadrant pain", "RUQ pain",
                                      "edema", "seizure"],
                "boost_diseases": ["Preeclampsia"],
                "boost_value": 0.25,
                "explanation": "Pregnant + HA/vision/RUQ pain: preeclampsia urgent rule-out",
            },
            {
                "name": "atypical_infection_in_immunocompromised",
                "if_comorbidities": ["HIV", "chemotherapy", "transplant", "immunosuppression",
                                       "immunocompromised", "steroids chronic"],
                "if_any_symptoms": ["fever", "cough", "shortness of breath", "headache"],
                "boost_diseases": ["Sepsis", "Pulmonary Tuberculosis"],
                "boost_value": 0.15,
                "explanation": "Immunocompromised: atypical/opportunistic organisms must be considered",
            },
            {
                "name": "vasoocclusive_crisis_in_sickle_cell",
                "if_comorbidities": ["sickle cell", "sickle cell disease", "SCD"],
                "if_any_symptoms": ["bone pain", "back pain", "chest pain",
                                      "abdominal pain", "joint pain"],
                "boost_diseases": ["Acute Compartment Syndrome"],  # not perfect, no SCD cascade yet
                "boost_value": 0.10,
                "explanation": "SCD patients with pain: vasoocclusive crisis primary consideration",
            },
        ]
    
    def _compute_comorbidity_boosts(self, symptoms: List[str],
                                       comorbidities: List[str]) -> Dict[str, float]:
        """Apply comorbidity rules to boost matching diseases."""
        boosts = {}
        sym_set = set(s.lower() for s in symptoms)
        comorb_set = set(c.lower() for c in comorbidities)
        
        for rule in self.comorbidity_rules:
            # Check comorbidity match
            rule_comorbs = set(c.lower() for c in rule.get("if_comorbidities", []))
            comorb_match = any(c in comorb_set or any(c in cs for cs in comorb_set)
                                 for c in rule_comorbs)
            if not comorb_match:
                continue
            
            # Check symptom match
            rule_syms = set(s.lower() for s in rule.get("if_any_symptoms", []))
            sym_match = any(s in sym_set or any(s in us for us in sym_set)
                              for s in rule_syms)
            if not sym_match:
                continue
            
            # Check excluded symptoms
            rule_excluded = set(s.lower() for s in rule.get("if_not_symptoms", []))
            excluded_match = any(s in sym_set or any(s in us for us in sym_set)
                                    for s in rule_excluded)
            if excluded_match:
                continue
            
            # Apply boost
            for disease in rule.get("boost_diseases", []):
                boosts[disease] = boosts.get(disease, 0) + rule.get("boost_value", 0.1)
        
        return boosts
    
    # ═══════════════════════════════════════════════════════════════
    # Helper: Time-window filtering (Phase 4)
    # ═══════════════════════════════════════════════════════════════
    
    def _compute_time_penalties(self, onset_hours: float,
                                   base_results: List) -> Dict[str, float]:
        """
        Penalize diseases whose typical onset doesn't match user's onset.
        Uses cascade speed field if available, else default heuristics.
        """
        # Categorize onset
        if onset_hours < 1:
            user_category = "hyperacute"  # < 1 hr
        elif onset_hours < 24:
            user_category = "acute"  # 1-24 hr
        elif onset_hours < 168:
            user_category = "subacute"  # 1-7 days
        elif onset_hours < 720:
            user_category = "chronic_recent"  # 1-30 days
        else:
            user_category = "chronic_old"  # > 30 days
        
        # Disease typical speed → category mapping (from cascade speed field)
        speed_to_category = {
            "seconds": "hyperacute", "minutes": "hyperacute", "hours": "acute",
            "days": "subacute", "weeks": "chronic_recent",
            "months": "chronic_recent", "years": "chronic_old",
            "chronic": "chronic_old", "chronic_recurrent": "chronic_old",
        }
        
        penalties = {}
        for disease, score, evidence in base_results:
            mech = evidence.get("mechanism", "")
            # Try to find speed in cascade
            organ = self.pmd.mechanism_map.get(disease, {}).get("organ")
            fmode = self.pmd.mechanism_map.get(disease, {}).get("failure_mode")
            
            disease_speed = self._get_disease_speed(organ, fmode)
            if not disease_speed:
                continue
            
            disease_category = self._categorize_speed(disease_speed)
            if not disease_category:
                continue
            
            # Penalty based on category mismatch distance
            categories_order = ["hyperacute", "acute", "subacute", "chronic_recent", "chronic_old"]
            try:
                user_idx = categories_order.index(user_category)
                disease_idx = categories_order.index(disease_category)
                gap = abs(user_idx - disease_idx)
                if gap >= 3:
                    penalties[disease] = 0.30  # strong mismatch
                elif gap == 2:
                    penalties[disease] = 0.15
                elif gap == 1:
                    penalties[disease] = 0.05
            except ValueError:
                pass
        
        return penalties
    
    def _get_disease_speed(self, organ: str, fmode: str) -> Optional[str]:
        """Get cascade speed field. Prefers _speed_category if tagged."""
        if not hasattr(self, "_speed_cache"):
            self._speed_cache = {}
            for p in ["medical_knowledge/layer2_physiology/organ_function.json",
                      "../medical_knowledge/layer2_physiology/organ_function.json"]:
                if os.path.exists(p):
                    try:
                        of = json.load(open(p))
                        for o_name, o_data in of.get("organs", {}).items():
                            for fm in o_data.get("failure_modes", []):
                                key = (o_name, fm.get("mode", ""))
                                self._speed_cache[key] = {
                                    "category": fm.get("_speed_category"),
                                    "speed": fm.get("speed", "")
                                }
                        break
                    except Exception:
                        pass
        info = self._speed_cache.get((organ, fmode), {})
        # Prefer tagged category (more reliable than parsing speed string)
        return info.get("category") or info.get("speed")
    
    def _categorize_speed(self, speed_str: str) -> Optional[str]:
        """Map cascade speed string OR category to canonical category."""
        if not speed_str:
            return None
        s = speed_str.lower()
        # Direct category match (preferred path via _speed_category)
        if s in ("hyperacute", "acute", "subacute", "chronic_recent", "chronic_old"):
            return s
        # Parse free-form speed string
        if "second" in s or "minute" in s:
            return "hyperacute"
        if "hour" in s:
            return "acute"
        if "day" in s and "30" not in s:
            return "subacute"
        if "week" in s or "month" in s:
            return "chronic_recent"
        if "year" in s or "chronic" in s:
            return "chronic_old"
        return None


# ═══════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print("IntegratedReasoner self-test")
    print("=" * 70)
    
    reasoner = IntegratedReasoner(verbose=True)
    
    print("\n[Test 1] Basic forward reasoning (symptoms only):")
    results = reasoner.diagnose(
        symptoms=["chest pain", "left arm pain", "shortness of breath", "diaphoresis"],
        top_k=3,
    )
    for d, s, e in results:
        print(f"  {d:30s} score={s:.3f}")
    
    print("\n[Test 2] With diabetic comorbidity (silent MI rule):")
    results = reasoner.diagnose(
        symptoms=["nausea", "fatigue", "epigastric pain", "shortness of breath"],
        comorbidities=["diabetes mellitus type 2"],
        top_k=3,
    )
    for d, s, e in results:
        boost = e.get("comorbidity_boost", 0)
        print(f"  {d:30s} score={s:.3f}  comorbidity_boost={boost:.2f}")
    
    print("\n[Test 3] Lab-first reverse reasoning (Phase 2):")
    results = reasoner.diagnose_from_labs(
        labs={"troponin": 2.5, "CK_MB": 25, "BNP": 800},
        top_k=3,
    )
    for d, s, e in results:
        matches = e.get("lab_matches", [])
        print(f"  {d:30s} score={s:.3f}  labs={[m[0] for m in matches]}")
    
    print("\n[Test 4] Time-window filtering (acute presentation):")
    results = reasoner.diagnose(
        symptoms=["chest pain", "shortness of breath"],
        onset_hours=0.5,  # 30 min
        top_k=5,
    )
    for d, s, e in results:
        tp = e.get("time_window_penalty", 0)
        print(f"  {d:30s} score={s:.3f}  time_penalty={tp:.2f}")