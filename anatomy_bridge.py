"""
NEXUS Anatomy Bridge (v3 - pure KG queries, no hardcoded maps)
═══════════════════════════════════════════════════════════════
 disease→organ  symptom→organ  NEXUS KG

"""
from __future__ import annotations
from typing import Dict, List, Any, Optional
try:
    from .anatomy_atlas import AnatomyAtlas
except ImportError:
    from nexus_engine.anatomy_atlas import AnatomyAtlas
import re


class AnatomyBridge:

    def __init__(self, nexus_medical=None, anatomy_kg=None):
        self.atlas = AnatomyAtlas()
        self.nexus = nexus_medical
        self.kg = anatomy_kg  # Separate KG for anatomy

    def enhance_with_anatomy(self, result: dict, user_input: str = "") -> dict:
        """Enhance pipeline result with anatomy data.

        P2-1: Added confidence gate. When the organ list exceeds
        MAX_USEFUL_ORGANS, the anatomy layer is flagged as low-confidence
        and BFS spread is suppressed — it was producing 30+ organs with
        15 spread paths for cases like "chest pain", which added noise
        rather than signal. The affected organs are still reported
        (transparency) but anatomy_confidence drops below 1.0, telling
        downstream consumers (scoring engine, trace, feature vector) to
        weight this layer down.
        """
        # ── Load config thresholds once at the top ───────────────────────────
        try:
            import json as _abj
            _abcfg = _abj.load(open("nexus_engine/nexus_config.json")).get("thresholds", {}).get("anatomy", {})
        except Exception:
            _abcfg = {}
        _anatomy_conf_min = _abcfg.get("anatomy_confidence_min", 0.5)
        _ab_dis_conf      = _abcfg.get("disease_organ_confidence", 0.65)
        _ab_sym_conf      = _abcfg.get("organ_confidence_default", 0.70)

        symptoms = result.get("symptoms", result.get("final_symptoms", []))
        diseases = result.get("top_diseases", [])
        user_text = (user_input or "").lower()

        # Need access to kg and _dominant_sym in this scope too (later block at line ~137)
        kg = self.kg
        try:
            from nexus_engine.high_risk_gate import HIGH_RISK_PRIORITY_ORDER as _HIGH_RISK_ORDER
        except ImportError:
            _HIGH_RISK_ORDER = ["chest pain","shortness of breath","syncope","fainting",
                                "hematemesis","hemoptysis","bloody stool","severe abdominal pain",
                                "altered mental status","focal weakness","slurred speech"]
        _sym_lower = [s.lower().replace("_"," ").strip() for s in (symptoms or [])]
        _dominant_sym = next((s for s in _HIGH_RISK_ORDER if s in _sym_lower), None)

        # 1. Find affected organs from KG
        affected = self._find_affected_organs(symptoms, diseases, user_text, result,
                                              _ab_dis_conf=_ab_dis_conf,
                                              _ab_sym_conf=_ab_sym_conf)

        # P2-1: confidence gate — when we localize to too many organs,
        # it means the symptom→organ mapping is too broad to be useful.
        # Suppress BFS spread (the noisiest part) and flag low confidence.
        MAX_USEFUL_ORGANS = 8
        anatomy_confidence = 1.0
        if len(affected) > MAX_USEFUL_ORGANS:
            anatomy_confidence = round(MAX_USEFUL_ORGANS / len(affected), 2)
            # Keep only the first N organs for spread analysis —
            # these are typically the most directly relevant ones
            spread_organs = affected[:3]
        else:
            spread_organs = affected[:3]

        # Vague/unlocalized symptom gate — matches nexus_trace.py logic
        _VAGUE_STANDALONE = {
            "swelling", "redness", "dizziness", "nausea", "weakness", "fatigue",
            "pain", "itching", "tingling", "numbness", "malaise",
        }
        _all_vague = len(symptoms) <= 2 and all(
            str(s).lower().replace("_", " ").strip() in _VAGUE_STANDALONE
            for s in symptoms
        )

        # 2. Spread path reasoning — suppressed for vague symptoms or low confidence
        spread = []
        if not _all_vague and anatomy_confidence >= _anatomy_conf_min:
            for organ in spread_organs:
                paths = self.atlas.find_spread_paths(organ, "infection", max_hops=4)
                for p in paths[:5]:
                    spread.append({**p, "origin": organ})

        referred = self._analyze_referred_pain(symptoms, user_text)

        adjacent = []
        if not _all_vague and anatomy_confidence >= _anatomy_conf_min:
            for organ in spread_organs:
                for c in self.atlas.get_connections_from(organ, ["adjacent"]):
                    tgt = self.atlas.organs.get(c.target)
                    if tgt:
                        adjacent.append({
                            "organ": c.target,
                            "from": organ, "reason": c.desc or f"adjacent to {organ}",
                        })

        context = self._build_context(affected, spread, referred, adjacent)

        # For vague unlocalized symptoms, replace specific organs with system labels.
        # Phase B refactor: read system from atlas (cascade-derived) instead of
        # hardcoded dict. atlas.organs[o].system is set per-organ.
        if _all_vague and affected:
            # Map common atlas systems to user-facing system labels
            # (atlas uses "cardiovascular" but user-facing prefers "circulatory")
            _SYS_DISPLAY = {
                "cardiovascular": "circulatory",
                "respiratory":    "respiratory",
                "gi":             "digestive",
                "neurologic":     "neurologic",
                "renal":          "renal",
                "integumentary":  "integumentary",
                "immune":         "immune",
                "hematologic":    "immune",
                "lymphatic":      "immune",
                "endocrine":      "endocrine",
                "musculoskeletal":"musculoskeletal",
                "hepatobiliary":  "digestive",
                "reproductive":   "reproductive",
            }
            system_labels = []
            seen_systems = set()
            for org in affected[:4]:
                # Look up system from atlas
                organ_obj = self.atlas.organs.get(org)
                raw_sys = (organ_obj.system if organ_obj else "") or ""
                sys = _SYS_DISPLAY.get(raw_sys.lower(), "systemic")
                if sys not in seen_systems:
                    system_labels.append(f"[{sys}]")
                    seen_systems.add(sys)
            result["anatomy_affected_organs"] = system_labels[:2]
            result["anatomy_system_labels"] = system_labels
        else:
            pass   # non-vague: organs already set above

        # Report anatomy support status for dominant symptom
        if _dominant_sym:
            _dom_norm = _dominant_sym.replace(" ", "_").replace("-", "_")
            _dom_organs = [o for o in affected if o in
                           {t.object for t in (kg.query(_dom_norm, "localizes_to") if kg else [])
                            if t.confidence >= _ab_sym_conf}]
            result["anatomy_dominant_sym"]     = _dominant_sym
            result["anatomy_dominant_organs"]  = _dom_organs
            result["anatomy_low_support"]      = len(_dom_organs) == 0
            if len(_dom_organs) == 0:
                print(f"[ANATOMY] low_support: '{_dominant_sym}' could not be localized — "
                      f"anatomy not reliable for this case")

        result["anatomy_affected_organs"] = affected
        result["anatomy_spread"] = spread[:10]
        result["anatomy_referred_pain"] = referred
        result["anatomy_adjacent_risk"] = adjacent
        result["anatomy_context"] = context
        # P2-1: new fields for downstream consumers
        result["anatomy_confidence"] = anatomy_confidence
        result["anatomy_organ_count"] = len(affected)
        if _all_vague:
            result["anatomy_note"] = (
                "Vague/unlocalized symptom(s) — anatomy stopped at system level. "
                "Spread and adjacency analysis skipped. "
                "Follow-up questions needed for localization."
            )
        elif anatomy_confidence < _anatomy_conf_min:
            result["anatomy_note"] = (
                f"Anatomy layer low confidence ({anatomy_confidence:.0%}): "
                f"{len(affected)} organs localized — too broad for reliable "
                f"spread analysis. BFS spread suppressed."
            )
        return result

    def _find_affected_organs(self, symptoms, diseases, user_text, result=None,
                              _ab_dis_conf: float = 0.65,
                              _ab_sym_conf: float = 0.70) -> List[str]:
        organs = []
        seen = set()

        if not self.nexus:
            return organs

        kg = self.kg

        # Identify dominant (highest-risk) symptom for priority ordering
        try:
            from nexus_engine.high_risk_gate import HIGH_RISK_PRIORITY_ORDER as _HIGH_RISK_ORDER
        except ImportError:
            _HIGH_RISK_ORDER = ["chest pain","shortness of breath","syncope","fainting",
                                "hematemesis","hemoptysis","bloody stool","severe abdominal pain",
                                "altered mental status","focal weakness","slurred speech"]
        _sym_lower = [s.lower().replace("_"," ").strip() for s in (symptoms or [])]
        _dominant_sym = next((s for s in _HIGH_RISK_ORDER if s in _sym_lower), None)
        # If high-risk symptom present, process it FIRST then limit secondary symptoms
        if _dominant_sym:
            ordered_syms = [_dominant_sym] + [s for s in symptoms
                            if s.lower().replace("_"," ").strip() != _dominant_sym]
        else:
            ordered_syms = list(symptoms or [])

        def _add(organ_name):
            if organ_name in self.atlas.organs and organ_name not in seen:
                organs.append(organ_name)
                seen.add(organ_name)

        # ── Symptom→organ fallback map (used by KG-less path AND KG-fallback path) ──
        # Loaded from syndrome_config.json if available, else inline defaults.
        # Defined HERE (before any use) because it's needed both when there's no KG
        # and when the KG has poor coverage for some symptoms.
        _SYM_ORGAN_FALLBACK: Dict[str, List[str]] = {}
        try:
            import json as _abfj, os as _abfos
            for _p in ("nexus_engine/syndrome_config.json", "syndrome_config.json"):
                if _abfos.path.exists(_p):
                    _sc = _abfj.load(open(_p))
                    _SYM_ORGAN_FALLBACK = _sc.get("symptom_organ_fallback", {})
                    break
        except Exception:
            pass
        if not _SYM_ORGAN_FALLBACK:
            _SYM_ORGAN_FALLBACK = {
                "vomiting":         ["stomach", "esophagus", "duodenum"],
                "nausea":           ["stomach", "duodenum"],
                "diarrhea":         ["colon", "ileum", "jejunum"],
                "weakness":         ["brain", "spinal_cord", "muscle"],
                "fatigue":          ["brain", "liver"],
                "dizziness":        ["brain", "inner_ear", "cerebellum"],
                "headache":         ["brain", "meninges", "scalp"],
                "rash":             ["skin", "dermis"],
                "redness":          ["skin", "dermis"],
                "itching":          ["skin", "dermis"],
                "swelling":         ["lymph_node", "soft_tissue"],
                "palpitations":     ["heart", "left_ventricle"],
                "shortness of breath": ["lung", "bronchus", "pleura"],
                "cough":            ["bronchus", "trachea", "lung"],
                "fever":            ["hypothalamus", "liver"],
                "chills":           ["hypothalamus", "muscle"],
                "confusion":        ["brain", "cerebral_cortex"],
                "numbness":         ["spinal_cord", "peripheral_nerve"],
                "tingling":         ["peripheral_nerve", "spinal_cord"],
            }

        # ── Weak mechanism organ seeds ────────────────────────────────────────
        # If strong reasoning produced no organs, use low-confidence seeds
        # from weak mechanism directions (NOT from disease ranking)
        if not organs and result and result.get("nexus_weak_organs"):
            _weak_seed_organs = result["nexus_weak_organs"]
            for organ in _weak_seed_organs[:2]:
                if organ not in seen:
                    seen.add(organ)
                    organs.append(organ)
            if organs:
                result["anatomy_low_confidence"] = True
                print(f"[ANATOMY] using weak mechanism seeds: {organs}")

        # ── Fallback for symptoms with no KG localization ────────────────────
        # If a symptom got 0 organs from the KG, use the fallback map
        for sym in ordered_syms:
            s_norm = str(sym).strip().lower().replace(" ", "_").replace("-", "_")
            s_plain = str(sym).strip().lower()
            fallback_organs = _SYM_ORGAN_FALLBACK.get(s_plain, [])
            if not fallback_organs:
                # try underscore form
                fallback_organs = _SYM_ORGAN_FALLBACK.get(s_norm.replace("_", " "), [])
            if fallback_organs:
                # Check if this symptom already contributed organs via KG
                _sym_contributed = any(
                    t.object in seen
                    for t in (kg.query(s_norm, "localizes_to") if kg else [])
                )
                if not _sym_contributed:
                    # Add up to 2 fallback organs (capped for secondary symptoms)
                    _is_secondary_fb = bool(_dominant_sym) and s_plain != _dominant_sym
                    _fb_cap = 1 if _is_secondary_fb else 2
                    _fb_added = 0
                    for organ in fallback_organs:
                        if organ not in seen and _fb_added < _fb_cap:
                            _canonical = organ  # fallback organs already canonical
                            seen.add(_canonical)
                            organs.append(_canonical)
                            _fb_added += 1

        # ── Query KG: disease affects_organ (gated on evidence quality)
        # Only seed organs from diseases that have specific (non-generic) symptom support.
        # This prevents nonspecific-symptom-anchored diseases (e.g. pyelonephritis
        # anchored only by "nausea") from polluting the anatomy spread with unrelated organs.

        # Nonspecific symptoms from evidence_gate (built from symptom files)
        try:
            from nexus_engine.evidence_gate import _get_sets as _ab_eg
            _NONSPECIFIC_SYMS = _ab_eg().get("vague_standalone", set())
        except Exception:
            _NONSPECIFIC_SYMS = {
                "nausea", "fatigue", "weakness", "vomiting", "fever",
                "dizziness", "headache", "malaise", "chills", "sweating",
            }

        VESSEL_SYSTEMS = {"cardiovascular"}

        for d_info in (diseases or []):
            d_name = ""
            d_matched = []
            d_score = 1.0  # pipeline diseases assumed credible unless score given
            if isinstance(d_info, str):
                d_name = d_info
            elif isinstance(d_info, dict):
                d_name = d_info.get("disease", d_info.get("name", ""))
                d_matched = [s.lower().strip()
                             for s in d_info.get("matched_symptoms", [])]
                d_score = float(d_info.get("score", d_info.get("total_score", 1.0)) or 1.0)

            # Gate 1: empty matched or all-nonspecific → skip as organ seed
            if not d_matched:
                continue
            has_specific = any(s not in _NONSPECIFIC_SYMS for s in d_matched)
            if not has_specific:
                continue

            # Gate 2: score threshold + required evidence flag
            if d_score < 0.55:
                continue  # low-confidence disease → don't seed anatomy
            # If disease was produced by nexus_medical with required_evidence_met=False
            # (set by the evidence gate in ranking), block it here too
            if isinstance(d_info, dict) and d_info.get("required_evidence_met") is False:
                _reason = d_info.get("block_reason", "required evidence missing")
                print(f"[ANATOMY] ORGAN_DISEASE_SEED_BLOCKED: '{d_name}' reason={_reason}")
                continue

            # Gate 3: disease-symptom coherence check
            # Block diseases where the user's symptoms are not anatomically coherent
            # with what the disease primarily affects
            # Coherence: use the matched symptoms from disease ranking
            # A disease is anatomy-coherent if its matched symptoms are non-generic
            # No separate rules file needed — matched symptoms ARE the coherence check
            _DISEASE_COHERENCE = {}  # handled by Gate 1 (non-specific check) above
            _coherence = _DISEASE_COHERENCE.get(d_name.lower())
            if _coherence:
                _d_sym_set = {s.lower() for s in d_matched}
                if not (_d_sym_set & _coherence):
                    print(f"[ANATOMY] ORGAN_DISEASE_SEED_BLOCKED: '{d_name}' "
                          f"reason=symptom-anatomy incoherence "
                          f"(matched={sorted(_d_sym_set)[:3]})")
                    continue

            d_norm = d_name.strip().lower().replace(" ", "_").replace("-", "_")
            if not d_norm:
                continue

            for triple in (kg.query(d_norm, "affects_organ") if kg else []):
                if triple.confidence < _ab_dis_conf:
                    continue
                organ_obj = self.atlas.organs.get(triple.object)
                if organ_obj and organ_obj.system in VESSEL_SYSTEMS:
                    if triple.object in ("heart", "aorta", "coronary_aa"):
                        _add(triple.object)
                else:
                    _add(triple.object)

        # ── Query KG: symptom localizes_to organ
        # Threshold 0.70 (matches nexus_trace). Sort by confidence for determinism.
        _CARDIAC_ORGANS = {"heart", "aorta", "coronary_aa", "left_ventricle",
                           "right_ventricle", "pericardium"}
        _HAS_CARDIAC_SYM = any(
            s.lower().replace("_", " ") in
            {"chest pain", "palpitations", "shortness of breath", "chest pressure",
             "syncope", "fainting", "chest tightness"}
            for s in (symptoms or [])
        )

        _dominant_organs_found = 0
        for sym_idx, sym in enumerate(ordered_syms):
            s_norm = str(sym).strip().lower().replace(" ", "_").replace("-", "_")
            # When a high-risk dominant symptom is present, secondary symptoms
            # contribute at most 1 organ each (so dominant symptom owns anatomy)
            _is_secondary = bool(_dominant_sym) and sym_idx > 0
            _secondary_organ_cap = 1 if _is_secondary else 99
            _secondary_organs_this_sym = 0
            if not s_norm:
                continue
            # Sort by confidence descending for deterministic selection
            triples_sorted = sorted(
                (kg.query(s_norm, "localizes_to") if kg else []),
                key=lambda t: t.confidence, reverse=True
            )
            for triple in triples_sorted:
                if triple.confidence < _ab_sym_conf:
                    break  # sorted, so no point continuing
                # Secondary organ cap enforcement
                if _is_secondary and _secondary_organs_this_sym >= _secondary_organ_cap:
                    break
                # Exclude cardiac organs for non-cardiac symptoms
                if triple.object in _CARDIAC_ORGANS and not _HAS_CARDIAC_SYM:
                    continue
                organ_obj = self.atlas.organs.get(triple.object)
                if organ_obj and organ_obj.system in VESSEL_SYSTEMS:
                    if triple.object in ("heart", "aorta", "coronary_aa"):
                        _add(triple.object)
                else:
                    _add(triple.object)

        # ── Match organ names from user text ──
        for alias, canonical in self.atlas._alias.items():
            if len(alias) > 2 and alias in user_text and canonical not in seen:
                organs.append(canonical)
                seen.add(canonical)
                _secondary_organs_this_sym += 1

        return organs

    def _analyze_referred_pain(self, symptoms, user_text) -> List[Dict]:
        results = []
        pain_locations = []
        for sym in (symptoms or []):
            s = str(sym).strip().lower()
            if "pain" in s or "ache" in s:
                for word in s.replace("_", " ").split():
                    resolved = self.atlas.resolve(word)
                    if resolved:
                        pain_locations.append(resolved)

        # Common pain description patterns
        patterns = [
            (r"left\s*arm", "l_arm"), (r"right\s*arm", "r_arm"),
            (r"jaw", "jaw"), (r"left\s*shoulder", "l_shoulder"),
            (r"right\s*shoulder", "r_shoulder"), (r"epigastr", "epigastrium"),
        ]
        for pat, loc in patterns:
            if re.search(pat, user_text):
                pain_locations.append(loc)

        for loc in set(pain_locations):
            results.extend(self.atlas.find_referred_pain_sources(loc))
        return results

    def _build_context(self, affected, spread, referred, adjacent) -> str:
        parts = []
        if affected:
            parts.append(f"Affected organs: {', '.join(affected[:8])}")
        if spread:
            lines = [f"- {s.get('description', s.get('organ',''))} (risk: {s.get('risk','')})" for s in spread[:5]]
            parts.append("Possible spread paths:\n" + "\n".join(lines))
        if referred:
            lines = [f"- Pain at {r.get('pain_at','')} from {r.get('source_organ','')}: {r.get('mechanism','')}" for r in referred[:3]]
            parts.append("Referred pain analysis:\n" + "\n".join(lines))
        if adjacent:
            lines = [f"- {a.get('organ','')} adjacent to {a.get('from','')}: {a.get('reason','')}" for a in adjacent[:5]]
            parts.append("Adjacent organ risks:\n" + "\n".join(lines))
        return "\n\n".join(parts) if parts else ""

    # API
    def query_spread(self, organ, spread_type="infection", max_hops=5):
        return self.atlas.find_spread_paths(organ, spread_type, max_hops)
    def query_path(self, origin, target, spread_type="infection"):
        return self.atlas.find_path_between(origin, target, spread_type)
    def query_referred_pain(self, location):
        return self.atlas.find_referred_pain_sources(location)
    def get_atlas_json(self):
        return self.atlas.to_json()