"""
NEXUS Medical Reasoning Engine (v6 — Full Knowledge Web)

Connects ALL Osler data sources into one reasoning graph:

  Data Sources:
    - 13 disease profiles (common_symptoms, red_flags, pathophysiology)
    - 20 symptom profiles (systems, causes, red_flags, related_symptoms)
    - 947 general mechanisms (effects, symptoms, diseases)
    - 1,200 bacteria mechanisms (effects, symptoms, diseases, pathway)
    - 1,500 virus mechanisms (effects, symptoms, diseases, pathway)
    - symptom_graph_v4.json (co-occurrence matrix)
    - organ_system_engine mappings (symptom -> organ system)
    - red_flag_engine rules
    - mechanism_symptom_map.json
    - OTC/medication rules

  Reasoning Web:
    symptom -> organ_system -> diseases (narrow search domain)
    symptom -> mechanism -> effects -> diseases (causal chain)
    symptom -> symptom_graph -> co-occurring symptoms (clinical pattern)
    symptom -> red_flags -> urgency detection
    symptom -> related_symptoms -> differential questions
    symptom -> common_causes -> root cause candidates
    disease -> common_symptoms -> coverage scoring
    disease -> red_flags -> safety checks
    mechanism -> effects -> secondary symptoms (predict what else)
    mechanism -> pathway -> category -> specialty routing
"""
from __future__ import annotations
import json, os, glob, re
from typing import Dict, List, Set, Tuple
from collections import defaultdict

# Phase B mechanism derivation (optional, won't break if missing)
try:
    from nexus_engine.mechanism_derivation import MechanismDerivationEngine
except ImportError:
    MechanismDerivationEngine = None

# Phase D: Pure mechanism diagnoser
try:
    from nexus_engine.pure_mechanism_diagnoser import PureMechanismDiagnoser
except ImportError:
    PureMechanismDiagnoser = None


# Organ system maps (from your organ_system_engine.py — absorbed into NEXUS)
SYMPTOM_TO_SYSTEM = {}  # Populated from data at load time. See _build_symptom_system_map()

# Fallback for symptoms not in any profile (minimal hardcoded safety net)
_FALLBACK_SYMPTOM_SYSTEM = {
    "chest pain": "cardiovascular",
    "fever": "systemic",
    "shortness of breath": "respiratory",
}

# Canonical system names — merge duplicates
SYSTEM_CANONICAL = {
    "gi": "gastrointestinal",
    "msk": "musculoskeletal",
    "neuro": "neurologic",
    "cardio": "cardiovascular",
    "cardiac": "cardiovascular",
    "renal": "urinary",
    "cutaneous": "dermatologic",
    "pulmonary": "respiratory",
    "circulatory": "circulatory",  # Keep separate — not same as cardiovascular
    "lymphatic": "immune",
}

def _canonicalize_systems(systems):
    """Merge duplicate system names into canonical forms."""
    result = set()
    for s in systems:
        canonical = SYSTEM_CANONICAL.get(s.lower(), s.lower())
        result.add(canonical)
    return result

SYSTEM_DISEASE_BOOST = {}  # Built from mechanism data at load time


class NexusMedical:

    def __init__(self):
        # === Disease layer ===
        self.diseases: Dict[str, dict] = {}
        self.disease_symptoms: Dict[str, Set[str]] = defaultdict(set)
        self.disease_red_flags: Dict[str, List[str]] = defaultdict(list)
        # Phase B: mechanism derivation engine
        try:
            self._mech_derive = MechanismDerivationEngine() if MechanismDerivationEngine else None
        except Exception:
            self._mech_derive = None
        self._disease_mechanism_map = None
        # Cached registry mappings dict for additional_sites lookup (lazy-loaded)
        self._dis_mech_map = None
        # Phase D: Pure mechanism diagnoser (lazy-initialized)
        self._pure_mech_diagnoser = None

        # === Symptom layer ===
        self.symptom_info: Dict[str, dict] = {}
        self.symptom_red_flags: Dict[str, List[str]] = defaultdict(list)
        self.symptom_related: Dict[str, List[str]] = defaultdict(list)
        self.symptom_causes: Dict[str, List[str]] = defaultdict(list)
        self.symptom_systems: Dict[str, List[str]] = defaultdict(list)

        # === Backward-compatibility stubs ===
        # Phase A deleted self._load_mechanisms() and the mech_info/mech_to_diseases/
        # sym_to_mechs/effect_to_symptoms attributes. These empty stubs prevent
        # AttributeError in any leftover code paths that still read them
        # (e.g. legacy startup prints, old route files, stale .pyc caches).
        # Pure-mechanism reasoning does NOT use them — the cascade is the source.
        self.mech_info: Dict[str, dict] = {}
        self.mech_to_diseases: Dict[str, List[str]] = {}
        self.sym_to_mechs: Dict[str, List[str]] = {}
        self.effect_to_symptoms: Dict[str, List[str]] = {}
        self.mech_to_effects: Dict[str, List[str]] = {}
        self.symptom_graph: Dict[str, Dict[str, float]] = {}

        # === Stats ===
        self.stats: Dict[str, int] = {}
        self._loaded = False

    # ===============================================
    # LOADING — builds the entire knowledge web
    # ===============================================

    def load_knowledge(self, knowledge_dir: str = "medical_knowledge"):
        if self._loaded:
            return
        self._load_diseases(knowledge_dir)
        self._load_symptoms(knowledge_dir)
        self._build_symptom_system_map()  # Data-driven: symptom → system from profiles
        self._build_disease_aliases()  # P2-2: short name aliases
        self._compute_stats()

        # Eager-load 3D anatomy stack for Step 5d
        # All three components must be present before first reason() call
        # so the trace shows real 3D analysis from the very first query.
        try:
            from nexus_engine.anatomy_atlas    import AnatomyAtlas
            from nexus_engine.spatial_engine   import SpatialEngine
            from nexus_engine.spatial_reasoner import SpatialReasoner
            self.atlas             = AnatomyAtlas()
            self._spatial_engine   = SpatialEngine(self.atlas)
            self._spatial_reasoner = SpatialReasoner(self._spatial_engine)
            print(f"[NEXUS] 3D reasoning ready: "
                  f"{len(self.atlas.organs)} organs, "
                  f"{len(self._spatial_engine.regions)} 3D regions, "
                  f"{len(self._spatial_reasoner.disease_anatomy)} disease fingerprints")
        except Exception as _ae:
            print(f"[NEXUS] 3D anatomy stack not available: {_ae}")
            self.atlas             = None
            self._spatial_engine   = None
            self._spatial_reasoner = None

        self._loaded = True

        s = self.stats
        print(f"[NEXUS] Knowledge loaded (pure mechanism):")
        print(f"  {s['diseases']} diseases (supplemental: triage/treatment/ICD)")
        print(f"  {s['symptom_profiles']} symptom profiles (for evidence_gate)")
        print(f"  {s['sym_disease_links']} symptom→disease links")
        print(f"  {s['total_web_connections']} total connections")

    def _load_diseases(self, base_dir):
        for folder in ["symptoms", "diseases", "diseases_pro"]:
            for fpath in glob.glob(os.path.join(base_dir, folder, "*.json")):
                try:
                    raw = json.load(open(fpath, "r", encoding="utf-8"))
                    # Support both single dict and array of dicts
                    items = raw if isinstance(raw, list) else [raw]
                    for d in items:
                        if not isinstance(d, dict):
                            continue
                        name = d.get("disease_name", "")
                        if not name:
                            continue
                        self.diseases[name] = d
                        for sym in d.get("common_symptoms", []):
                            self.disease_symptoms[name].add(sym.lower().strip())
                        for rf in d.get("red_flags", []):
                            self.disease_red_flags[name].append(rf)
                except Exception:
                    continue

    def _build_disease_aliases(self):
        """P2-2: Build short-name alias table from loaded disease names.

        NEXUS disease names (from disease JSON) are clinical:
          "Acute Viral Upper Respiratory Infection"
          "Community-acquired Pneumonia"
          "Acute Bacterial Meningitis"

        But the RL training pool and pipeline use short names:
          "flu", "pneumonia", "meningitis"

        This table maps long → short so downstream consumers (feature
        vector, consistency check, trace) can match them.

        Strategy: for each canonical disease name, extract the most
        distinctive keyword(s) and build bidirectional lookup.
        Also includes a hardcoded alias table for common mappings that
        keyword extraction can't catch (e.g. "flu" ≠ any word in
        "Acute Viral Upper Respiratory Infection").
        """
        # Hardcoded aliases for names that can't be derived by keyword overlap
        KNOWN_ALIASES = {
            "acute viral upper respiratory infection": "viral URI",
            "viral upper respiratory infection": "viral URI",
            "upper respiratory infection": "URI",
            "influenza": "influenza",
            "community-acquired pneumonia": "pneumonia",
            "bacterial pneumonia": "pneumonia",
            "viral pneumonia": "pneumonia",
            "acute bacterial meningitis": "meningitis",
            "bacterial meningitis": "meningitis",
            "acute appendicitis": "appendicitis",
            "myocardial infarction": "heart attack",
            "acute myocardial infarction": "heart attack",
            "acute gastroenteritis": "gastroenteritis",
            "viral gastroenteritis": "gastroenteritis",
            "severe sepsis": "sepsis",
            "septic shock": "sepsis",
            "bronchial asthma": "asthma",
            "migraine headache": "migraine",
            "migraine with aura": "migraine",
            "migraine without aura": "migraine",
            "covid-19": "covid",
            "sars-cov-2 infection": "covid",
            "coronavirus disease": "covid",
        }

        self._disease_short_names = {}  # long_name_lower → short_name

        for name in self.diseases:
            nl = name.lower().strip()
            # Check hardcoded aliases first
            if nl in KNOWN_ALIASES:
                self._disease_short_names[nl] = KNOWN_ALIASES[nl]
            else:
                # Fallback: use the last significant word as short name
                # (e.g. "Acute Bacterial Meningitis" → "meningitis")
                words = [w for w in nl.split() if w not in
                         {"acute", "chronic", "severe", "mild", "viral",
                          "bacterial", "fungal", "acquired", "community",
                          "non", "infectious", "infection"}]
                if words:
                    self._disease_short_names[nl] = words[-1]
                else:
                    self._disease_short_names[nl] = nl

        if self._disease_short_names:
            aliases_str = ", ".join(
                f"{k[:30]}→{v}" for k, v in
                sorted(self._disease_short_names.items())[:10]
            )
            print(f"[NEXUS] Built {len(self._disease_short_names)} disease aliases: {aliases_str}")

    def _short_name(self, disease_name: str) -> str:
        """Return the short training-pool name for a NEXUS disease name.
        Falls back to the original name if no alias exists."""
        nl = disease_name.lower().strip()
        return self._disease_short_names.get(nl, disease_name)

    def _load_symptoms(self, base_dir):
        for fpath in glob.glob(os.path.join(base_dir, "symptoms", "symptom_*.json")):
            try:
                d = json.load(open(fpath, "r", encoding="utf-8"))
                sym = d.get("symptom", "").lower().strip()
                if not sym:
                    continue
                self.symptom_info[sym] = d
                self.symptom_red_flags[sym] = d.get("red_flags", [])
                self.symptom_related[sym] = [
                    r.lower().strip() for r in d.get("related_symptoms", [])
                ]
                self.symptom_causes[sym] = d.get("common_causes", [])
                self.symptom_systems[sym] = d.get("systems", [])
            except Exception:
                continue

    def _build_symptom_system_map(self):
        """Build SYMPTOM_TO_SYSTEM from loaded symptom profiles (data-driven, not hardcoded).
        
        Selection logic:
          1. Canonicalize all system names
          2. For each symptom, pick the most specific non-vascular system first
          3. If only vascular/broad systems, use the first one
          4. Fallback for symptoms not in any profile
        """
        global SYMPTOM_TO_SYSTEM
        SYMPTOM_TO_SYSTEM = {}
        
        # Systems that are too broad to be a useful primary for AMBIGUOUS symptoms
        # (cardiovascular is kept because chest pain → cardiovascular is correct)
        broad_systems = {"immune", "systemic", "nervous", "circulatory"}
        
        for sym, info in self.symptom_info.items():
            systems = info.get("systems", [])
            if not systems:
                continue
            # Canonicalize all
            canon = [SYSTEM_CANONICAL.get(s.lower().strip(), s.lower().strip()) for s in systems]
            
            # Prefer specific systems over broad ones
            specific = [s for s in canon if s not in broad_systems]
            if specific:
                SYMPTOM_TO_SYSTEM[sym] = specific[0]
            else:
                SYMPTOM_TO_SYSTEM[sym] = canon[0]
                
        # Merge fallbacks for anything not in profiles
        for sym, sys in _FALLBACK_SYMPTOM_SYSTEM.items():
            if sym not in SYMPTOM_TO_SYSTEM:
                SYMPTOM_TO_SYSTEM[sym] = SYSTEM_CANONICAL.get(sys, sys)
        print(f"[NEXUS] Built symptom→system map: {len(SYMPTOM_TO_SYSTEM)} mappings from data")






    def _compute_stats(self):
        self.stats = {
            "diseases": len(self.diseases),
            "symptom_profiles": len(self.symptom_info),
            "sym_disease_links": sum(len(v) for v in self.disease_symptoms.values()),
            "total_web_connections": (
                sum(len(v) for v in self.disease_symptoms.values()) +
                sum(len(v) for v in self.symptom_related.values()) +
                sum(len(v) for v in self.symptom_causes.values())
            ),
        }

    # ===============================================
    # CORE REASONING: symptoms -> diseases + thinking
    # ===============================================

    def reason(self, symptoms: List[str], user_text: str = "",
               mode: str = "pure_mechanism",
               context: dict = None) -> dict:
        """
        STATE-FIRST mechanism reasoning (G.4 architecture).

        Pipeline:
          1. STATE_FIRST_SCAN  — run state_model.simulate() across ALL 141
             state-modeled diseases. Top scorers form primary candidates.
          2. CASCADE_BACKFILL — fill remaining slots with cascade-only
             diseases (e.g. cancers) that state model doesn't cover.
          3. CONTEXT_MODIFIERS — adjust scores by demographics + vitals
             (age, sex, pregnancy, etc).
          4. PHYSIOLOGY_SIM   — numerical body state (BP, HR, lactate).
          5. SPATIAL_3D       — anatomical consistency check.
          6. TRACE_BUILD      — for top-3, build full reasoning trace.

        Args:
          symptoms: list of symptom strings
          user_text: free-text input (legacy)
          mode: legacy parameter
          context: optional patient context, e.g.
            {
              "age": 29,
              "sex": "female",
              "pregnancy_status": "pregnant",
              "duration": "2 days",
              "vitals": {"bp_sys": 160, "bp_dia": 100, "hr": 105, "temp": 37.2},
              "history": ["pregnancy 32 weeks"],
              "medications": [...]
            }

        Returns dict with diagnoses[], thinking[], red_flags[], stats{}.
        Top-3 diagnoses include 'reasoning_trace' field for full explanation.
        """
        self.load_knowledge()
        context = context or {}

        # ── Normalize symptoms (G.5 #2a: support per-symptom temporal data) ──
        # Backward-compatible: accept either
        #   - list of strings:  ["chest pain", "SOB"]    (no timing info)
        #   - list of dicts:    [{"name": "cough", "onset_hours_ago": 72}, ...]
        #   - mixed:            ["fever", {"name": "rash", "onset_hours_ago": 6}]
        # Internal canonical form: list of dicts with at least "name", optional "onset_hours_ago"
        sym_records = []  # canonical: [{"name": "...", "onset_hours_ago": float|None}]
        for s in symptoms:
            if isinstance(s, str) and s.strip():
                sym_records.append({
                    "name": s.lower().strip().replace("_", " "),
                    "onset_hours_ago": None,
                })
            elif isinstance(s, dict) and s.get("name"):
                name = str(s["name"]).lower().strip().replace("_", " ")
                onset = s.get("onset_hours_ago")
                # Validate numeric onset (negative / non-numeric → None)
                if isinstance(onset, (int, float)) and onset >= 0:
                    onset_val = float(onset)
                else:
                    onset_val = None
                sym_records.append({"name": name, "onset_hours_ago": onset_val})
            # silently skip None / empty / malformed

        # Build legacy sym_list (just names) — keeps the rest of pipeline unchanged
        sym_list = [r["name"] for r in sym_records]
        sym_set = set(sym_list)

        # Compute case-level duration for backward compatibility with G.5 #2 v1.
        # If no explicit context["duration_hours"] but symptoms have onsets,
        # use the MAXIMUM onset (i.e. earliest symptom = duration of illness).
        if "duration_hours" not in context or context.get("duration_hours") is None:
            onsets = [r["onset_hours_ago"] for r in sym_records
                      if r["onset_hours_ago"] is not None]
            if onsets:
                context["duration_hours"] = max(onsets)
                context["_duration_inferred_from_symptoms"] = True

        thinking = [{
            "step": "INPUT",
            "detail": f"Patient symptoms: {sorted(sym_set)}",
            "context": context if context else "(no demographics provided)",
            "symptom_records": sym_records,  # G.5 #2a: full temporal view
        }]

        # Early return if no usable symptoms — cascade backfill would otherwise
        # produce spurious diagnoses with no input evidence.
        if not sym_list:
            thinking.append({
                "step": "INPUT_VALIDATION_FAILED",
                "detail": "No usable symptoms provided. Reason() cannot diagnose.",
            })
            return {
                "diagnoses":       [],
                "thinking":        thinking,
                "red_flags":       [],
                "triage_level":    None,
                "detected_systems":[],
                "stats":           {"mechanisms_activated": 0},
                "mode":            "state_first_g4",
                "input_symptoms":  symptoms,
                "context":         context,
            }

        # ════════════════════════════════════════════════════════════
        # Step 1: STATE_FIRST_SCAN — run state_model across ALL diseases
        # BLENDED scoring (3 layers):
        #   - alias_score:    user symptom string ∩ state-derived symptoms
        #   - evidence_score: user symptoms → inferred states ∩ disease states (C2)
        #   - bayes_score:    P(disease | symptoms) via naive Bayes (C3)
        # ════════════════════════════════════════════════════════════
        try:
            from nexus_engine.state_model import (
                STATE_MODEL_DISEASES, simulate_disease,
                expand_symptom_aliases, _REGISTRY,
                infer_state_evidence, state_evidence_match_score,
                bayesian_state_posteriors, bayesian_disease_score,
            )
            _REGISTRY.load_all()
        except ImportError as e:
            thinking.append({"step": "ERROR",
                             "detail": f"state_model not available: {e}"})
            return {"diagnoses": [], "thinking": thinking,
                    "red_flags": [], "stats": {}, "mode": mode}

        # G.4.C: pre-compute user state inference once for the whole scan
        user_state_inference = infer_state_evidence(sym_list)
        user_states = user_state_inference["state_evidence"]
        unrecognized = user_state_inference["unrecognized_symptoms"]

        # G.4.C3: pre-compute Bayesian state posteriors once
        bayes_posteriors_data = bayesian_state_posteriors(sym_list, top_k_states=15)
        thinking.append({
            "step": "STATE_EVIDENCE_INFERENCE",
            "detail": f"User symptoms imply {len(user_states)} organ states "
                      f"(rule-based) + {len(bayes_posteriors_data['posteriors'])} "
                      f"states with Bayesian posteriors. "
                      f"Unrecognized: {unrecognized if unrecognized else 'none'}.",
            "top_5_bayesian_states": [
                {"state": e["state"], "P": e["posterior"],
                 "supporting": e["supporting_symptoms"]}
                for e in bayes_posteriors_data["top_k"][:5]
            ],
        })

        state_candidates = []
        user_norm = {s.lower().replace("_", " ").strip() for s in sym_list}

        for disease_name in _REGISTRY._disease_index.keys():
            sim = simulate_disease(disease_name)
            if not sim:
                continue
            # Layer 1: alias-based symptom matching
            sim_syms = expand_symptom_aliases(sim["derived_symptoms"])
            overlap = user_norm & sim_syms
            alias_score = len(overlap) / max(len(sym_list), 1)
            alias_score = min(alias_score, 1.0)

            # Layer 2: state-evidence matching (C2 — rule-based)
            ev_match = state_evidence_match_score(sym_list, disease_name)
            evidence_score = ev_match["score"]

            # Layer 3: Bayesian state inference (C3)
            # Pass pre-computed posteriors to avoid 141× recomputation
            bayes_match = bayesian_disease_score(
                sym_list, disease_name,
                posteriors=bayes_posteriors_data["posteriors"]
            )
            bayes_score = bayes_match["score"]

            # Skip if all three methods say zero
            if alias_score <= 0 and evidence_score <= 0 and bayes_score <= 0:
                continue

            # Blend: alias is denser baseline, evidence + bayes add precision.
            # Weighting (tuned empirically on 30 vignettes):
            #   - alias 0.6  (broad symptom match — most reliable signal)
            #   - evidence 0.25 (C2 rule-based state agreement)
            #   - bayes 0.15 (C3 probabilistic — adds discrimination but sparse)
            blended_state_score = (
                0.6 * alias_score
                + 0.25 * evidence_score
                + 0.15 * bayes_score
            )
            blended_state_score = min(blended_state_score, 1.0)

            state_candidates.append((disease_name, blended_state_score, sim,
                                     overlap, sim_syms, ev_match, bayes_match,
                                     alias_score, evidence_score, bayes_score))

        # Sort by blended score
        state_candidates.sort(key=lambda x: x[1], reverse=True)

        # Funnel statistics — distinguish prior-only vs evidence-supported.
        # Helps debug: if 132/132 have score but only 20 have real evidence,
        # the ranking is dominated by Bayesian prior, not patient symptoms.
        total_scanned = len(_REGISTRY._disease_index)
        n_alias_only      = sum(1 for c in state_candidates if c[7] > 0)
        n_evidence_only   = sum(1 for c in state_candidates if c[8] > 0)
        n_bayes_only      = sum(1 for c in state_candidates if c[9] > 0)
        # "Real evidence" = at least 1 layer with score > 0
        n_any_evidence    = len(state_candidates)
        # "Strong" = at least 2 layers agree (>0)
        n_multi_layer = sum(
            1 for c in state_candidates
            if sum(1 for s in (c[7], c[8], c[9]) if s > 0) >= 2
        )
        # Above blend threshold (visible candidates)
        n_above_threshold = sum(1 for c in state_candidates if c[1] >= 0.15)

        thinking.append({
            "step": "STATE_FIRST_SCAN",
            "detail": (
                f"Scanned {total_scanned} diseases. "
                f"alias>0: {n_alias_only}. "
                f"state evidence>0: {n_evidence_only}. "
                f"Bayesian>0: {n_bayes_only}. "
                f"≥2 layers agree: {n_multi_layer}. "
                f"blended ≥0.15: {n_above_threshold}. "
                f"Top-3 shown."
            ),
            "funnel": {
                "scanned":             total_scanned,
                "alias_supported":     n_alias_only,
                "evidence_supported":  n_evidence_only,
                "bayes_supported":     n_bayes_only,
                "multi_layer_agree":   n_multi_layer,
                "above_threshold_015": n_above_threshold,
            },
            "blend_formula": "0.6×alias + 0.25×evidence + 0.15×bayes",
            "top_3": [(d, round(s, 3))
                      for d, s, *_ in state_candidates[:3]],
        })

        # Build initial diagnoses from state candidates
        diagnoses = []
        seen = set()
        for (disease, sscore, sim, overlap, sim_syms, ev_match, bayes_match,
             alias_score, evidence_score, bayes_score) in state_candidates[:15]:
            diagnoses.append({
                "disease": disease,
                "short_name": self._short_name(disease),
                "score": round(sscore, 3),
                "confidence": min(sscore, 1.0),
                "matched_symptoms": [(us, us, 1.0) for us in sorted(overlap)],
                "missing_symptoms": [],
                "evidence": [
                    f"alias: {len(overlap)}/{len(sym_list)} sym match",
                    f"state evidence (C2): {evidence_score:.2f}",
                    f"Bayesian (C3): {bayes_score:.2f}",
                ],
                "supplemental": self._get_supplemental_data(disease),
                "_state_modeled": True,
                "_state_simulation": {
                    "perturbations":    sim["perturbations"],
                    "active_state":     [{"organ": o, "variable": v,
                                          "value": round(val, 2)}
                                         for o, v, val in sim["active_state"]],
                    "derivations":      sim["derivations"],
                    "derived_symptoms": sim["derived_symptoms"],
                    "matched_symptoms": sorted(overlap),
                    "missing_symptoms": sorted(user_norm - sim_syms),
                    "alias_score":      round(alias_score, 3),
                    "evidence_score":   round(evidence_score, 3),
                    "bayesian_score":   round(bayes_score, 3),
                    "blended_score":    round(sscore, 3),
                    "matched_states":   ev_match["matched_states"],
                    "bayes_matched_states":  bayes_match.get("matched_states", []),
                    "bayes_missed_states":   bayes_match.get("missed_states", []),
                },
                "_state_evidence_match": ev_match,
                "_bayesian_match": bayes_match,
                "_mechanism": "state_model",
            })
            seen.add(disease.lower())

        # ════════════════════════════════════════════════════════════
        # Step 2: CASCADE_BACKFILL — fill with cascade-only diseases
        # (mainly cancers and others not in state model)
        # ════════════════════════════════════════════════════════════
        scores_dict = {d["disease"]: d["score"] for d in diagnoses}

        if self._pure_mech_diagnoser is None and PureMechanismDiagnoser is not None:
            self._pure_mech_diagnoser = PureMechanismDiagnoser()
        if self._pure_mech_diagnoser is not None:
            pure_results = self._pure_mech_diagnoser.diagnose(sym_list, top_k=20)
            # Build cascade score lookup
            cascade_scores = {dname: cs for dname, cs, _ in pure_results}

            # BLEND: state-modeled diseases get state_score boosted by
            # cascade_score (if cascade also matches). This way state model
            # leads, but cascade provides corroborating evidence — cascade
            # CANNOT override state_score, but adds confidence.
            for d in diagnoses:
                if not d.get("_state_modeled"):
                    continue
                cscore = cascade_scores.get(d["disease"], 0.0)
                state_score = d["score"]
                # Blend: 0.7 × state + 0.3 × cascade, capped at 1.0
                blended = min(0.7 * state_score + 0.3 * cscore, 1.0)
                d["_cascade_corroboration"] = round(cscore, 3)
                d["_blend"] = {
                    "state_score": state_score,
                    "cascade_score": cscore,
                    "blended": round(blended, 3),
                }
                d["score"] = round(blended, 3)
                d["confidence"] = min(blended, 1.0)
                scores_dict[d["disease"]] = blended

                # Preserve cascade mechanism for proper rendering
                # (deterministic_response expects "organ/failure_mode" format)
                for dname, _, evidence in pure_results:
                    if dname == d["disease"]:
                        cascade_mech = evidence.get("mechanism", "")
                        if cascade_mech:
                            d["_mechanism"] = cascade_mech
                        # Merge cascade matched_symptoms into visible field
                        # so deterministic_response sees evidence for proof
                        # strength. State-side matches are 1.0 similarity but
                        # often fewer; cascade adds breadth.
                        cmatched = evidence.get("matched_symptoms", [])
                        if cmatched:
                            d["_cascade_matched_symptoms"] = cmatched
                            # Merge into visible matched_symptoms (dedupe by name)
                            existing_names = {m[0] for m in d["matched_symptoms"]
                                              if isinstance(m, (list, tuple))}
                            for cm in cmatched:
                                if isinstance(cm, (list, tuple)) and len(cm) >= 3:
                                    if cm[0] not in existing_names:
                                        d["matched_symptoms"].append(tuple(cm))
                                        existing_names.add(cm[0])
                        break

            # Backfill cascade-only (cancers etc.)
            backfill = 0
            for disease_name, score, evidence in pure_results:
                if disease_name.lower() in seen:
                    continue
                # Cascade-only — weaker prior (×0.6 penalty)
                diagnoses.append({
                    "disease": disease_name,
                    "short_name": self._short_name(disease_name),
                    "score": round(score * 0.6, 3),
                    "confidence": min(score * 0.6, 1.0),
                    "matched_symptoms": [
                        (m, m, 1.0) for m in evidence.get("matched_symptoms", [])
                    ],
                    "missing_symptoms": [],
                    "evidence": [
                        f"cascade match (weak prior, no state model)",
                        f"mechanism: {evidence.get('mechanism', '?')}",
                    ],
                    "supplemental": self._get_supplemental_data(disease_name),
                    "_state_modeled": False,
                    "_cascade_only": True,
                    "_cascade_score": score,
                    "_mechanism": evidence.get("mechanism", ""),
                })
                scores_dict[disease_name] = score * 0.6
                backfill += 1
                if len(diagnoses) >= 20:
                    break

            thinking.append({
                "step": "CASCADE_BACKFILL",
                "detail": f"Added {backfill} cascade-only diseases. "
                          f"For state-modeled diseases, blended state×0.7 + cascade×0.3.",
            })

        # ════════════════════════════════════════════════════════════
        # Step 3: CONTEXT_MODIFIERS — demographics-based score adjustment
        # ════════════════════════════════════════════════════════════
        if context:
            self._apply_context_modifiers(diagnoses, context, thinking)
            # Resort after modifiers
            diagnoses.sort(key=lambda x: x["score"], reverse=True)

        # ════════════════════════════════════════════════════════════
        # Step 3b: G.5 MODIFIERS — temporal, mismatch, history, drug response
        # All optional — only apply if relevant context is provided.
        # G.5 #2b: symptom_records carry per-symptom temporal info for ordering
        # reasoning. Inject via context (doesn't break signature).
        # ════════════════════════════════════════════════════════════
        if "_symptom_records" not in context:
            context["_symptom_records"] = sym_records
        self._apply_g5_modifiers(diagnoses, sym_list, context, thinking)

        # Resort by score in case state_candidates ordering changed
        diagnoses.sort(key=lambda x: x["score"], reverse=True)

        # ── Step 4: Physiological State Simulation (numerical body state) ──
        _phys_state = {}
        _phys_critical = False
        try:
            if not hasattr(self, "_physio_engine") or self._physio_engine is None:
                from nexus_engine.physiology_engine import PhysiologyEngine
                self._physio_engine = PhysiologyEngine(getattr(self, "atlas", None))
            phys_result = self._physio_engine.simulate(sorted(sym_set))
            _phys_state = phys_result.get("state", {})
            _phys_critical = phys_result.get("critical", False)
            
            # Boost diseases matching critical physiologic patterns
            if _phys_critical:
                for d in diagnoses:
                    d_lower = d["disease"].lower()
                    if any(kw in d_lower for kw in
                           ["sepsis", "shock", "hemorrhage", "mi", "infarction", "acute"]):
                        d["score"] = round(d["score"] + 0.15, 3)
                        scores_dict[d["disease"]] = d["score"]
                        d["evidence"].append("physiology: critical state boost (+0.15)")
            
            thinking.append({
                "step": "PHYSIOLOGY_SIM",
                "detail": (f"State: BP={_phys_state.get('bp','?')} "
                           f"HR={_phys_state.get('hr','?')} "
                           f"lactate={_phys_state.get('lactate','?')} "
                           f"shock_idx={phys_result.get('shock_index','?')} "
                           f"critical={_phys_critical}"),
                "interpretation": phys_result.get("interpretation", []),
                "cascades":       phys_result.get("cascades", []),
                "critical":       _phys_critical,
            })
        except Exception as _pe:
            thinking.append({
                "step":   "PHYSIOLOGY_SIM",
                "detail": f"Physiology engine unavailable: {_pe}",
            })
        
        # ── Step 3: 3D Spatial Anatomical Consistency ──
        # Phase B refactor: spatial_reasoner now uses cascade-derived anatomy
        # (medical_knowledge/derived/disease_anatomy_cascade.json) instead of
        # the hardcoded _FALLBACK_DISEASE_ANATOMY.
        _spatial_state = {}
        try:
            if not hasattr(self, "_spatial_reasoner") or self._spatial_reasoner is None:
                from nexus_engine.spatial_engine import SpatialEngine
                from nexus_engine.spatial_reasoner import SpatialReasoner
                self._spatial_engine = SpatialEngine(getattr(self, "atlas", None))
                self._spatial_reasoner = SpatialReasoner(self._spatial_engine)
            
            from nexus_engine.symptom_organ_map import merge_symptom_organs
            spatial_map = merge_symptom_organs(sorted(sym_set))
            
            # Run 3D reasoning. Score adjustments use cascade-derived primary organs.
            # Apply as TIE-BREAKER (small magnitude) rather than primary scorer:
            # cascade matching already does the heavy lifting; spatial just nudges
            # ties in the anatomically-coherent direction.
            reasoning_result = self._spatial_reasoner.reason(spatial_map, scores_dict)
            
            SPATIAL_DAMPING = 0.20   # scale spatial deltas to 20% of full magnitude
            
            applied = []
            for d in diagnoses:
                d_name = d["disease"]
                d_norm = d_name.lower().replace(' ', '_').replace('/', '_').replace('-', '_')
                delta_raw = (reasoning_result["adjustments"].get(d_name, 0.0) or
                             reasoning_result["adjustments"].get(d_norm, 0.0))
                delta = delta_raw * SPATIAL_DAMPING
                if abs(delta) > 0.005:
                    d["score"] = round(max(0.0, d["score"] + delta), 3)
                    scores_dict[d_name] = d["score"]
                    expl = (reasoning_result["explanations"].get(d_name) or
                            reasoning_result["explanations"].get(d_norm, ""))
                    d["evidence"].append(f"3D nudge: {expl} (Δ={delta:+.3f})")
                    applied.append({"disease": d_name, "delta": round(delta, 3)})
            
            # Apply contradiction penalties — also damped (×0.85 instead of ×0.5)
            contradictions = set()
            contradict_names = {c[0] for c in reasoning_result["contradictions"]}
            for d in diagnoses:
                d_name = d["disease"]
                d_norm = d_name.lower().replace(' ', '_').replace('/', '_').replace('-', '_')
                if d_name in contradict_names or d_norm in contradict_names:
                    d["score"] = round(d["score"] * 0.85, 3)
                    scores_dict[d_name] = d["score"]
                    d["evidence"].append("3D: anatomical contradiction (×0.85)")
                    contradictions.add(d_name)
            
            _spatial_state = {
                "primary_organs":      sorted(spatial_map.get("primary", [])),
                "zones_active":        sorted(spatial_map.get("zones", [])),
                "applied_adjustments": applied,
                "contradictions":      sorted(contradictions),
            }
            
            thinking.append({
                "step":   "SPATIAL_3D",
                "detail": (f"3D (cascade-derived anatomy): "
                           f"{len(applied)} adjustments, "
                           f"{len(contradictions)} contradictions"),
                "result": _spatial_state,
            })
        except Exception as _se:
            thinking.append({
                "step":   "SPATIAL_3D",
                "detail": f"Spatial reasoner unavailable: {_se}",
            })
        
        # ── Step 4: Re-rank after enrichment ──
        diagnoses.sort(key=lambda x: -x["score"])
        
        thinking.append({
            "step": "RANKING",
            "detail": f"Top {len(diagnoses)} cascade-derived diagnoses (after physiology + 3D)",
            "top_3": [(d["disease"], d["score"]) for d in diagnoses[:3]],
        })
        
        # ── Red flags from cascade derivation ONLY (no hardcoded lists) ──
        top_red_flags = []
        if diagnoses:
            top_red_flags = diagnoses[0].get("_derived_red_flags", [])
        
        # ── Triage from supplemental data (allowed per spec) ──
        top_triage = ""
        if diagnoses and diagnoses[0].get("supplemental"):
            top_triage = diagnoses[0]["supplemental"].get("triage_level", "")
        
        # ── Stats for AgentOrchestrator ──
        stats = {
            "symptoms_input":       len(sym_set),
            "diseases_considered":  len(diagnoses),
            "mechanisms_activated": len(pure_results),  # one cascade per top-K disease
            "physiology_critical":  _phys_critical,
            "spatial_adjustments":  len(_spatial_state.get("applied_adjustments", [])),
            "spatial_contradictions": len(_spatial_state.get("contradictions", [])),
        }
        
        # ── Cascade-derived detected_systems ──
        # Pure mechanism: derive system membership FROM the cascade matches'
        # organs (via atlas), not from symptom_*.json files. This is the
        # correct architecture — the cascade knows what system each disease
        # affects via the organ→system atlas mapping.
        _ATLAS_TO_EG_SYS = {
            "cardiovascular": "cardiovascular", "respiratory": "respiratory",
            "gi": "gastrointestinal", "neurologic": "neurologic",
            "renal": "urinary", "hepatobiliary": "hepatobiliary",
            "endocrine": "endocrine", "integumentary": "cutaneous",
            "msk": "musculoskeletal", "musculoskeletal": "musculoskeletal",
            "hematologic": "hematologic",
            "immune": "immune", "reproductive": "reproductive",
            "lymphatic": "immune",
            "sensory": "sensory",   # NEW: eye, retina, middle_ear
        }
        # Registry uses generic organ names; atlas uses anatomical names.
        # Same mapping used by spatial_reasoner and anatomy_knowledge_loader.
        # Now expanded — most organs added to atlas, so this is just for true aliases.
        _REGISTRY_TO_ATLAS = {
            "bronchus":          ["r_bronchus"],
            "kidney":            ["r_kidney"],
            "small_intestine":   ["duodenum"],
            "colon":             ["asc_colon"],
            "coronary_arteries": ["coronary_aa"],
            "pericardium":       ["heart"],
            "pulmonary_arteries":["pulm_aa"],
            "systemic_immune":   ["spleen", "bone_marrow"],
            "throat":            ["pharynx"],
            "nose":              ["pharynx"],
            "upper_respiratory_tract": ["pharynx"],
            "vasculature_systemic": ["aorta"],
            "spinal_nerves":     ["spinal_cord"],
            "cauda_equina":      ["spinal_cord"],
            "bone":              ["bone_marrow"],
            "lymph_node":        ["cerv_ln"],
            "ovary":             ["ovaries"],
            "sig_colon":         ["sigmoid"],
        }

        # Load disease_mechanism_map to access additional_sites
        # (cached on first reason() call; cheap if already loaded)
        if not hasattr(self, '_dis_mech_map') or self._dis_mech_map is None:
            import json as _json
            try:
                with open('medical_knowledge/registry/disease_mechanism_map.json') as _f:
                    self._dis_mech_map = _json.load(_f).get('mappings', {})
            except Exception:
                self._dis_mech_map = {}

        detected_systems = []
        seen_sys = set()
        for d in diagnoses[:5]:
            mech = d.get('_mechanism', '')
            if '/' not in mech:
                continue
            organ = mech.split('/')[0].strip()

            # Build candidate organs list: primary (via alias) + additional_sites
            atlas_candidates = []
            if organ in self.atlas.organs:
                atlas_candidates.append(organ)
            else:
                atlas_candidates.extend(_REGISTRY_TO_ATLAS.get(organ, []))

            # Pull additional_sites from registry (multi-organ enrichment)
            dx_name = d.get('disease', '')
            reg_entry = self._dis_mech_map.get(dx_name, {}) if dx_name else {}
            for site in (reg_entry.get('additional_sites') or []):
                if site in self.atlas.organs and site not in atlas_candidates:
                    atlas_candidates.append(site)

            # Collect all system memberships from candidate organs
            for ao in atlas_candidates:
                if ao in self.atlas.organs:
                    atlas_sys = (self.atlas.organs[ao].system or "").lower()
                    # Organ-specific override BEFORE system→eg_sys mapping.
                    # Some atlas organs have ambiguous system (e.g. pharynx is
                    # both digestive and upper-airway; atlas labels it "gi",
                    # but clinically presenting with pharyngitis is respiratory).
                    if ao in ("pharynx", "larynx", "epiglottis", "tonsils"):
                        eg_sys = "respiratory"
                    else:
                        eg_sys = _ATLAS_TO_EG_SYS.get(atlas_sys, atlas_sys)
                    if eg_sys and eg_sys not in seen_sys:
                        detected_systems.append(eg_sys)
                        seen_sys.add(eg_sys)

        # ════════════════════════════════════════════════════════════
        # Step 6: TRACE_BUILD — build reasoning trace for top-5
        # (Frontend displays up to 5 diagnoses; each needs a trace so its
        #  body diagram + mechanism chain render. simulate_disease is cached,
        #  so extending 3→5 is cheap.)
        # ════════════════════════════════════════════════════════════
        for d in diagnoses[:5]:
            d["reasoning_trace"] = self._build_reasoning_trace(
                d, sym_list, context
            )

        # ════════════════════════════════════════════════════════════
        # Step 7: LAB_RECOMMENDATION — suggest labs that disambiguate top-3
        # ════════════════════════════════════════════════════════════
        lab_recommendations = {}
        try:
            from nexus_engine.state_model import recommend_labs_for_top_diagnoses
            lab_recommendations = recommend_labs_for_top_diagnoses(
                diagnoses, sym_list, top_k=3, max_labs_per_disease=3
            )
            if lab_recommendations.get("labs_aggregated"):
                thinking.append({
                    "step": "LAB_RECOMMENDATION",
                    "detail": (f"{len(lab_recommendations['labs_aggregated'])} "
                                f"lab(s) suggested based on missing state evidence"),
                    "top_labs": [
                        {"lab": l["lab"], "priority": l["max_priority"],
                         "supports": l["supports_diagnoses"]}
                        for l in lab_recommendations["labs_aggregated"][:5]
                    ],
                })
        except Exception as _e:
            print(f"[NEXUS] lab recommendation skipped: {_e}")

        # ════════════════════════════════════════════════════════════
        # Step 8: TREATMENT_RECOMMENDATION (G.5 #10)
        # Mechanism-derived: from disease perturbations → treatment targets →
        # drugs whose state_effects counteract perturbations. No hardcoding.
        # ════════════════════════════════════════════════════════════
        treatment_recommendations = {}
        try:
            from nexus_engine.state_model import recommend_treatment_for_top_diagnoses
            treatment_recommendations = recommend_treatment_for_top_diagnoses(
                diagnoses, top_k=3, max_drugs_per_disease=4
            )
            if treatment_recommendations.get("drugs_aggregated"):
                thinking.append({
                    "step": "TREATMENT_RECOMMENDATION",
                    "detail": (f"{len(treatment_recommendations['drugs_aggregated'])} "
                                f"drug(s) recommended via mechanism (state-effect matching)"),
                    "top_drugs": [
                        {"drug": d["drug"], "class": d.get("drug_class", "?"),
                         "priority": d["max_priority"],
                         "supports": d["supports_diagnoses"]}
                        for d in treatment_recommendations["drugs_aggregated"][:5]
                    ],
                })
        except Exception as _e:
            print(f"[NEXUS] treatment recommendation skipped: {_e}")

        return {
            "diagnoses":         diagnoses,
            "thinking":          thinking,
            "red_flags":         top_red_flags,
            "triage_level":      top_triage,
            "detected_systems":  detected_systems,
            "predicted_symptoms": [],
            "root_causes":       [],
            "otc_hints":         [],
            "pathogen_spread":   [],
            "suggested_questions": [],
            "nexus_consistency": self._compute_consistency(diagnoses, detected_systems,
                                                            len(state_candidates), 0),
            "stats":             stats,
            "mode":              "state_first_g4",
            "engine":            "StateFirstReasoner (G.4) + PhysiologyEngine + SpatialReasoner",
            "input_symptoms":    symptoms,
            "context":           context,
            "physiology_state":  _phys_state,
            "spatial_state":     _spatial_state,
            "lab_recommendations":       lab_recommendations,
            "treatment_recommendations": treatment_recommendations,
        }


    def _compute_consistency(self, top_diseases, detected_systems, mechs, ctx_mechs):
        """NEXUS self-consistency: how coherent is the evidence?"""
        if mechs >= 20:     f_mech = 0.4
        elif mechs >= 5:    f_mech = 0.3
        elif mechs >= 1:    f_mech = 0.2
        elif ctx_mechs > 0: f_mech = 0.1
        else:               f_mech = 0.0

        _dis_sys = {
            name.lower().strip(): d.get("system", "")
            for name, d in self.diseases.items()
            if isinstance(d, dict)
        }
        dx_systems = set()
        for d in top_diseases:
            name = d.get("disease", "").lower().strip()
            sys  = _dis_sys.get(name, "")
            if not sys:
                for k, v in _dis_sys.items():
                    if (name in k or k in name) and len(min(name, k, key=len)) > 4:
                        sys = v; break
            if sys:
                dx_systems.add(sys)

        if len(dx_systems) == 1:   f_disease = 0.4
        elif len(dx_systems) == 2: f_disease = 0.2
        elif top_diseases:         f_disease = 0.1
        else:                      f_disease = 0.0

        f_sys = 0.2 if detected_systems else 0.0
        score = round(min(f_mech + f_disease + f_sys, 1.0), 2)
        return {
            "consistency_score":     score,
            "disease_systems":       sorted(dx_systems),
            "detected_systems":      sorted(detected_systems),
            "agreed_systems":        sorted(dx_systems & set(detected_systems)),
            "mechs_direct":          mechs,
            "mechs_context":         ctx_mechs,
            "reliable":              score >= 0.4 and mechs >= 1,
        }


    # ===============================================
    # Helpers
    # ===============================================







    def _tok(text: str) -> list:
        return [w for w in re.split(r'[\s_\-/,]+', text.lower().strip()) if len(w) > 1]


    # ════════════════════════════════════════════════════════════════
    # G.4: Context modifiers (demographics + vitals)
    # ════════════════════════════════════════════════════════════════
    def _apply_context_modifiers(self, diagnoses: list, context: dict,
                                  thinking: list) -> None:
        """Adjust diagnosis scores based on patient context.

        Context fields supported:
          - age (int)
          - sex (str: 'male', 'female')
          - pregnancy_status (str: 'pregnant', 'not_pregnant', 'unknown')
          - vitals (dict: bp_sys, bp_dia, hr, rr, temp, spo2)
          - history (list of strings)
          - medications (list of strings)
        """
        age = context.get("age")
        sex = (context.get("sex") or "").lower()
        pregnant = context.get("pregnancy_status", "").lower() == "pregnant"
        vitals = context.get("vitals", {}) or {}
        history = context.get("history", []) or []
        history_text = " ".join(str(h).lower() for h in history)

        modifiers_applied = []

        for d in diagnoses:
            dname_lower = d["disease"].lower()
            score_before = d["score"]
            multiplier = 1.0
            reasons = []

            # ── Pregnancy-specific diseases ──
            if any(k in dname_lower for k in ["preeclampsia", "eclampsia",
                                                "gestational", "hellp"]):
                if pregnant or "pregnan" in history_text:
                    multiplier *= 2.0
                    reasons.append("pregnancy context strongly supports")
                elif sex == "female" and age and 15 <= age <= 50:
                    multiplier *= 0.5  # plausible but unknown pregnancy
                    reasons.append("reproductive age but pregnancy not confirmed")
                else:
                    multiplier *= 0.05  # impossible
                    reasons.append("not pregnant — disease impossible")

            # ── PCOS, fibroids, ectopic ──
            if any(k in dname_lower for k in ["pcos", "polycystic",
                                                "fibroid", "ectopic"]):
                if sex == "male":
                    multiplier *= 0.0
                    reasons.append("male patient — female-specific disease impossible")

            # ── Prostate diseases (M only) ──
            if any(k in dname_lower for k in ["bph", "prostat"]):
                if sex == "female":
                    multiplier *= 0.0
                    reasons.append("female patient — prostate disease impossible")

            # ── Pediatric-specific ──
            if any(k in dname_lower for k in ["kawasaki", "intussus",
                                                "pyloric stenosis", "neonatal",
                                                "bronchiolitis"]):
                if age and age > 18:
                    multiplier *= 0.2
                    reasons.append(f"adult ({age}) — pediatric disease unlikely")

            # ── Age-related (Alzheimer, Parkinson, PAD, AAA) ──
            if any(k in dname_lower for k in ["alzheimer", "dementia",
                                                "parkinson"]):
                if age and age < 50:
                    multiplier *= 0.3
                    reasons.append(f"young ({age}) — age-related disease unlikely")

            # ── Pediatric appendicitis / bronchiolitis nudge ──
            if "bronchiolitis" in dname_lower:
                if age and age < 2:
                    multiplier *= 1.5
                    reasons.append(f"infant ({age}) — typical age for bronchiolitis")

            # ── Vital signs: shock pattern → boost sepsis/anaphylaxis/cardiogenic ──
            bp_sys = vitals.get("bp_sys")
            hr = vitals.get("hr")
            temp = vitals.get("temp")
            spo2 = vitals.get("spo2")

            if bp_sys and bp_sys < 90:
                # Hypotensive
                if "sepsis" in dname_lower or "anaphyl" in dname_lower:
                    multiplier *= 1.5
                    reasons.append("hypotension supports shock state")
                if "hypertensive emergency" in dname_lower:
                    multiplier *= 0.0
                    reasons.append("hypotensive — hypertensive emergency excluded")

            if bp_sys and bp_sys > 160:
                if "hypertensive emergency" in dname_lower or \
                   "preeclampsia" in dname_lower:
                    multiplier *= 1.5
                    reasons.append("severe HTN supports diagnosis")
                if "vasovagal" in dname_lower or "hypotens" in dname_lower:
                    multiplier *= 0.2
                    reasons.append("hypertensive — vasovagal/hypotensive excluded")

            if hr and hr > 130:
                if "atrial fibrillation" in dname_lower or "svt" in dname_lower:
                    multiplier *= 1.3
                    reasons.append("tachycardia supports arrhythmia")
                if "sepsis" in dname_lower:
                    multiplier *= 1.2

            if temp and temp > 38.5:
                if any(k in dname_lower for k in ["sepsis", "meningit",
                                                    "pneumonia", "appendicit"]):
                    multiplier *= 1.2
                    reasons.append("fever supports infection")

            if spo2 and spo2 < 90:
                if any(k in dname_lower for k in ["embolism", "pneumonia",
                                                    "pneumothorax", "ards"]):
                    multiplier *= 1.3
                    reasons.append("hypoxia supports lung pathology")

            # ── Apply multiplier ──
            if abs(multiplier - 1.0) > 0.001:
                new_score = round(score_before * multiplier, 3)
                d["score"] = max(0.0, min(new_score, 1.5))  # allow >1 then re-cap
                d["confidence"] = min(d["score"], 1.0)
                d["_context_modifier"] = {
                    "before": score_before,
                    "after": d["score"],
                    "multiplier": round(multiplier, 2),
                    "reasons": reasons,
                }
                modifiers_applied.append(
                    f"{d['disease']}: ×{multiplier:.2f} ({'; '.join(reasons)})"
                )

        if modifiers_applied:
            thinking.append({
                "step": "CONTEXT_MODIFIERS",
                "detail": f"Applied {len(modifiers_applied)} demographic/vital adjustments",
                "adjustments": modifiers_applied[:10],
            })

    # ════════════════════════════════════════════════════════════════
    # G.5: combined modifiers — temporal, mismatch, history, drug response
    # ════════════════════════════════════════════════════════════════
    def _apply_g5_modifiers(self, diagnoses: list, user_symptoms: list,
                             context: dict, thinking: list) -> None:
        """Apply G.5 modifiers (temporal, mismatch, history, drug) to scores.

        Each modifier multiplier is computed independently then combined.
        Diagnoses are mutated in place. Skipped silently if relevant context
        fields are missing.
        """
        from nexus_engine.state_model import (
            temporal_compatibility_score, symptom_mismatch_penalty,
            history_disease_modifier, drug_response_modifier,
            negative_evidence_modifier, symptom_ordering_compatibility,
        )

        context = context or {}
        duration_hours = context.get("duration_hours")
        history = context.get("history") or []
        drug_responses = context.get("drug_responses") or []
        denied_symptoms = context.get("denied_symptoms") or []
        symptom_records = context.get("_symptom_records") or []
        # Only run ordering check if AT LEAST ONE symptom has timing info
        # (avoid noise when user gave list-of-strings = no temporal data)
        has_temporal_info = any(r.get("onset_hours_ago") is not None
                                  for r in symptom_records)

        # Mismatch penalty applies even without explicit context
        applied_count = 0

        for d in diagnoses:
            dname = d["disease"]
            score_before = d["score"]
            combined_mult = 1.0
            mod_details = {}

            # G.5 #3a — Mismatch penalty (DISABLED — state model coverage too partial
            #         to apply this safely. See note below.)
            # mismatch = symptom_mismatch_penalty(user_symptoms, dname)

            # G.5 #3b — Negative evidence (patient explicitly denies symptom)
            # ENABLED — uses derivation rule confidence, so high-confidence
            # symptom denial is strong (predictable) signal.
            if denied_symptoms:
                neg = negative_evidence_modifier(dname, denied_symptoms)
                if neg["multiplier"] < 1.0:
                    combined_mult *= neg["multiplier"]
                    mod_details["negative_evidence"] = neg

            # G.5 #2a — Temporal compatibility (case-level duration_hours)
            if duration_hours:
                temporal = temporal_compatibility_score(dname, duration_hours)
                if abs(temporal["multiplier"] - 1.0) > 0.001:
                    combined_mult *= temporal["multiplier"]
                    mod_details["temporal"] = temporal

            # G.5 #2b — Symptom ordering compatibility (per-symptom timing)
            # Distinguishes monophasic acute / subacute / chronic-with-exacerbation
            if has_temporal_info:
                ordering = symptom_ordering_compatibility(dname, symptom_records)
                if abs(ordering["multiplier"] - 1.0) > 0.001:
                    combined_mult *= ordering["multiplier"]
                    mod_details["symptom_ordering"] = ordering

            # G.5 #4 — History modifier
            if history:
                hist_mod = history_disease_modifier(dname, history)
                if abs(hist_mod["multiplier"] - 1.0) > 0.001:
                    combined_mult *= hist_mod["multiplier"]
                    mod_details["history"] = hist_mod

            # G.5 #5 — Drug response
            if drug_responses:
                drug_mod = drug_response_modifier(dname, drug_responses)
                if abs(drug_mod["multiplier"] - 1.0) > 0.001:
                    combined_mult *= drug_mod["multiplier"]
                    mod_details["drug_response"] = drug_mod

            if abs(combined_mult - 1.0) > 0.001:
                new_score = round(score_before * combined_mult, 3)
                d["score"] = max(0.0, min(new_score, 1.5))
                d["confidence"] = min(d["score"], 1.0)
                d["_g5_modifiers"] = {
                    "before": score_before,
                    "after": d["score"],
                    "combined_multiplier": round(combined_mult, 2),
                    "details": mod_details,
                }
                applied_count += 1

        if applied_count:
            thinking.append({
                "step": "G5_MODIFIERS",
                "detail": f"Applied G.5 score adjustments to {applied_count} diseases "
                          f"(temporal/mismatch/history/drug)",
            })

    # ════════════════════════════════════════════════════════════════
    # G.4: Reasoning trace builder
    # ════════════════════════════════════════════════════════════════
    def _build_reasoning_trace(self, diagnosis: dict, user_symptoms: list,
                                 context: dict) -> dict:
        """Build full mechanism reasoning trace for a single diagnosis.

        Output structure (the 'why this diagnosis' explanation):
          {
            "1_disease_perturbation": "Disease X causes ↑ischemia (+0.8) in heart",
            "2_state_propagation": [...derivation chain...],
            "3_explains_symptoms": [list of matched user symptoms + how],
            "4_vital_signs_check": "consistent / not provided / inconsistent",
            "5_anatomy_check": "consistent / inconsistent",
            "6_missing_evidence": [...what would strengthen diagnosis...],
            "summary": "1-sentence overall verdict"
          }
        """
        if not diagnosis.get("_state_modeled"):
            return {
                "summary": "Cascade-only diagnosis — no mechanism trace available.",
                "note": "This disease lacks a state model; ranking based on "
                        "symptom-pattern matching (weaker evidence).",
            }

        sim = diagnosis.get("_state_simulation", {})
        perturbations = sim.get("perturbations", [])
        active_state = sim.get("active_state", [])
        derivations = sim.get("derivations", [])
        matched = sim.get("matched_symptoms", [])
        missing = sim.get("missing_symptoms", [])
        derived_syms = sim.get("derived_symptoms", [])

        trace = {}

        # 1. Disease perturbation
        if perturbations:
            ptext = []
            for p in perturbations[:5]:
                organ = p.get("organ", "?")
                var = p.get("variable", p.get("state", "?"))
                delta = p.get("delta", "?")
                cause = p.get("cause", "")
                sign = "↑" if (isinstance(delta, (int, float)) and delta > 0) else "↓"
                ptext.append(f"{sign}{var} ({delta:+.2f}) in {organ}: {cause}")
            trace["1_disease_perturbation"] = ptext

        # 2. State propagation chain (from derivations)
        propagation_steps = []
        for deriv in derivations[:8]:
            if isinstance(deriv, dict):
                organ = deriv.get("from_organ", "?")
                sym = deriv.get("symptom", "?")
                rationale = deriv.get("rationale", "")
                propagation_steps.append(
                    f"[{organ}] state perturbation → {sym} ({rationale[:80]})"
                )
        if propagation_steps:
            trace["2_state_propagation"] = propagation_steps

        # 2b. Cross-organ cascade — the medically interesting spread.
        # _state_simulation doesn't carry propagation_trace, so pull it from a
        # fresh simulate_disease() call (cached — cheap). This is the "how the
        # disease spreads through the body" chain doctors reason about
        # (e.g. heart failure → kidney hypoperfusion → fluid retention).
        cascade = []
        spread_edges = []  # for frontend organ-to-organ arrows
        try:
            from nexus_engine.state_model import simulate_disease as _sim_dz
            _full_sim = _sim_dz(diagnosis.get("disease", "")) or {}
            _prop = _full_sim.get("propagation_trace", [])
        except Exception:
            _prop = []
        for step in _prop:
            name = step.get("name", "")
            effect = step.get("effect", "")
            trigger = step.get("trigger", "")
            if "→" not in name:
                continue
            left, right = name.split("→", 1)
            def _organ_of(tok):
                tok = tok.split(":")[-1].strip()
                tok = tok.split(".")[0].strip()
                if tok in ("r_lung", "l_lung", "left_lung", "right_lung"):
                    return "lung"
                return tok
            src_organ = _organ_of(left)
            dst_organ = _organ_of(right)
            cascade.append({
                "from": left.split(":")[-1].strip(),
                "to": right.strip(),
                "trigger": trigger,
                "effect": effect,
            })
            if src_organ != dst_organ and src_organ and dst_organ:
                spread_edges.append({"from": src_organ, "to": dst_organ,
                                      "via": right.split(".")[-1].strip()
                                      if "." in right else ""})
        if cascade:
            trace["2b_cross_organ_cascade"] = cascade
        if spread_edges:
            seen_e = set()
            uniq = []
            for e in spread_edges:
                key = (e["from"], e["to"])
                if key not in seen_e:
                    seen_e.add(key)
                    uniq.append(e)
            trace["spread_edges"] = uniq

        # 3. Explains symptoms
        explained = []
        for user_sym in user_symptoms:
            us_norm = user_sym.lower().strip()
            if us_norm in matched:
                # Find which derivation explained this
                explainer = None
                for deriv in derivations:
                    if isinstance(deriv, dict):
                        deriv_sym = deriv.get("symptom", "").lower()
                        if deriv_sym == us_norm or us_norm in deriv_sym \
                           or deriv_sym in us_norm:
                            explainer = deriv.get("rationale",
                                                   "state model derivation")
                            break
                explained.append({
                    "user_reported": user_sym,
                    "explained_by": explainer or "matched via alias",
                })
            else:
                explained.append({
                    "user_reported": user_sym,
                    "explained_by": "NOT EXPLAINED by this disease — see missing evidence",
                })
        trace["3_explains_symptoms"] = explained

        # 3b. DIAGNOSTIC BRIDGE — the REVERSE direction: symptom → inferred
        # organ state → why that points to this disease. This is the clinical
        # "how did we get FROM your symptoms TO this diagnosis" chain.
        # Built from matched_states (each carries supporting_symptoms +
        # disease_value vs the user_threshold = how strong the evidence is).
        bridge = []
        sim_ms = diagnosis.get("_state_simulation", {}).get("matched_states", [])
        for ms in sim_ms:
            state_full = ms.get("state", "")            # e.g. "lung.gas_exchange"
            organ = state_full.split(".")[0] if "." in state_full else ""
            state_name = state_full.split(".")[-1] if "." in state_full else state_full
            direction = ms.get("direction", "")          # "high" / "low"
            sup = ms.get("supporting_symptoms", []) or []
            dval = ms.get("disease_value")
            thr = ms.get("user_threshold")
            # strength: how far disease_value clears the threshold
            strength = "strong"
            try:
                if dval is not None and thr is not None:
                    margin = abs(float(dval) - float(thr))
                    strength = ("strong" if margin >= 0.25 else
                                "moderate" if margin >= 0.1 else "weak")
            except (TypeError, ValueError):
                pass
            if not sup:
                continue
            bridge.append({
                "symptoms": sup,                  # which user symptom(s) point here
                "organ": organ,
                "state": state_name,
                "direction": "elevated" if direction == "high" else
                             "reduced" if direction == "low" else direction,
                "disease_value": dval,
                "threshold": thr,
                "strength": strength,
            })
        if bridge:
            trace["3b_diagnostic_bridge"] = bridge

        # 4. Vital signs check
        vitals = context.get("vitals", {})
        if vitals:
            cm = diagnosis.get("_context_modifier", {})
            if cm:
                vital_reasons = [r for r in cm.get("reasons", [])
                                 if any(k in r.lower() for k in
                                       ["hypoten", "hyperten", "tachy",
                                        "fever", "hypox"])]
                if vital_reasons:
                    trace["4_vital_signs_check"] = f"consistent: {'; '.join(vital_reasons)}"
                else:
                    trace["4_vital_signs_check"] = "no vital-sign red flags"
            else:
                trace["4_vital_signs_check"] = "vitals provided but no specific signal"
        else:
            trace["4_vital_signs_check"] = "vital signs not provided"

        # 5. Anatomy check (3D spatial reasoning result if any)
        spatial = diagnosis.get("_3d_evidence")
        if spatial:
            trace["5_anatomy_check"] = f"anatomy consistent: {spatial}"
        else:
            trace["5_anatomy_check"] = "anatomy: not explicitly evaluated"

        # 6. Missing evidence — what would strengthen diagnosis
        unmatched_derived = [s for s in derived_syms
                             if s.lower() not in
                             {m.lower() for m in matched}]
        if unmatched_derived:
            trace["6_missing_evidence"] = {
                "would_strengthen": unmatched_derived[:6],
                "note": "These symptoms are predicted by the model but not "
                        "reported by patient. Ask about them.",
            }

        # Summary
        match_ratio = len(matched) / max(len(user_symptoms), 1)
        if match_ratio >= 0.7:
            verdict = "Strong match"
        elif match_ratio >= 0.4:
            verdict = "Moderate match"
        else:
            verdict = "Weak match"
        trace["summary"] = (
            f"{verdict}: state model explains {len(matched)}/{len(user_symptoms)} "
            f"reported symptoms via {len(perturbations)} initial perturbations."
        )

        return trace


    def _get_supplemental_data(self, disease_name):
        """
        For pure mechanism mode: get non-derivable supplemental data
        (treatment, triage, ICD) from disease_0001 if available.
        These fields CAN'T be derived from cascade alone.
        """
        try:
            if not hasattr(self, 'diseases') or not self.diseases:
                return {}
            for d in (self.diseases.values() if isinstance(self.diseases, dict) else self.diseases):
                if d.get('disease_name') == disease_name:
                    return {
                        "icd_code": d.get('icd_code', ''),
                        "triage_level": d.get('triage_level', ''),
                        "treatment": d.get('treatment', []),
                        "diagnostic_criteria": d.get('diagnostic_criteria', []),
                        "prevalence": d.get('prevalence', d.get('prevalence_text', '')),
                    }
        except Exception:
            pass
        return {}

    def get_mechanism_picture(self, disease_name: str) -> dict:
        """
        Phase B: derive a clinical picture for a disease from LAYER 2-4 mechanism data.
        Returns dict with derived common_symptoms, red_flags, complications, etc.
        Used as AUGMENTATION (transparency) — does NOT replace hardcoded diagnosis logic.
        """
        if not getattr(self, '_mech_derive', None):
            return {}
        if getattr(self, '_disease_mechanism_map', None) is None:
            try:
                import json as _json
                _path = "medical_knowledge/registry/disease_mechanism_map.json"
                if not os.path.exists(_path):
                    _path = "../" + _path
                with open(_path) as _f:
                    _data = _json.load(_f)
                self._disease_mechanism_map = _data.get("mappings", {})
            except Exception:
                self._disease_mechanism_map = {}
        m = self._disease_mechanism_map.get(disease_name)
        if not m:
            return {}
        return self._mech_derive.derive_disease(
            primary_organ=m.get("organ", ""),
            failure_mode=m.get("failure_mode", ""),
            pathogen=m.get("pathogen"),
        )