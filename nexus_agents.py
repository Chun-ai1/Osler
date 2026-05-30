"""
nexus_agents.py — NEXUS Multi-Agent Pipeline
═════════════════════════════════════════════════════════════════
Replaces the monolithic enhance_pipeline_result() with a chain of
independent agents. Each agent:
  - receives AgentState (shared mutable context)
  - runs its logic
  - writes its result back to AgentState
  - can be inspected, tested, and rewarded independently

Pipeline order:
  IntakeAgent → MechanismAgent → DiseaseRankingAgent →
  RedFlagAgent → EvidenceGateAgent → QuestionAgent →
  AnswerComposer → Verifier → RewardTrace
"""
from __future__ import annotations
import json, os, time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


# ─────────────────────────────────────────────────────────────────────────────
# SHARED STATE  (passed through every agent, mutated in place)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentState:
    # ── Input ────────────────────────────────────────────────────────────────
    selected_symptoms: List[str] = field(default_factory=list)
    user_text:         str       = ""
    raw_result:        Dict      = field(default_factory=dict)   # from nexus_runner

    # ── IntakeAgent output ───────────────────────────────────────────────────
    final_symptoms:    List[str] = field(default_factory=list)
    severity_hints:    Dict      = field(default_factory=dict)   # {symptom: "severe"|"mild"}
    duration_hints:    Dict      = field(default_factory=dict)   # {symptom: "3 days"}
    is_vague_only:     bool      = False

    # ── MechanismAgent output ────────────────────────────────────────────────
    mechanisms_activated:  int       = 0
    viral_mech_count:      int       = 0
    bacterial_mech_count:  int       = 0
    mechanism_evidence:    List[Dict]= field(default_factory=list)  # [{title, domain, ...}]

    # ── DiseaseRankingAgent output ───────────────────────────────────────────
    top_diseases:      List[Dict] = field(default_factory=list)  # [{disease, score, ...}]
    detected_systems:  List[str]  = field(default_factory=list)
    consistency_score: float      = 0.0

    # ── RedFlagAgent output ──────────────────────────────────────────────────
    observed_red_flags:  List[Dict] = field(default_factory=list)  # A-bucket
    risk_escalators:     List[Dict] = field(default_factory=list)  # B-bucket
    watch_for_flags:     List[Dict] = field(default_factory=list)  # C-bucket
    triage_level:        str        = "MODERATE"
    high_risk_freeze:    bool       = False
    freeze_dominant:     str        = ""
    freeze_syndrome:     str        = ""
    allow_pills:         bool       = True

    # ── EvidenceGateAgent output ─────────────────────────────────────────────
    output_state:         str   = "triage_only"   # "triage_only"|"syndrome"|"label"
    syndrome_label:       str   = ""
    secondary_pattern:    str   = ""
    etiology_allowed:     bool  = False
    etiology_level:       str   = "uncertain"
    disease_label_allowed:bool  = False

    # ── EtiologyAgent output ─────────────────────────────────────────────────
    etiology:             str   = "uncertain"   # "viral"|"bacterial"|"non_infectious"|"uncertain"
    etiology_confidence:  float = 0.0
    etiology_scores:      Dict  = field(default_factory=dict)
    etiology_reasoning:   List  = field(default_factory=list)

    # ── QuestionAgent output ─────────────────────────────────────────────────
    follow_up_questions:  List[str] = field(default_factory=list)
    needs_more_info:      bool      = True

    # ── AnswerComposer output ────────────────────────────────────────────────
    final_answer:         str   = ""
    otc_suggestions:      List  = field(default_factory=list)

    # ── Verifier output ──────────────────────────────────────────────────────
    verifier_passed:      bool      = True
    verifier_issues:      List[str] = field(default_factory=list)
    verifier_patches:     List[str] = field(default_factory=list)

    # ── RewardTrace ──────────────────────────────────────────────────────────
    agent_log:            List[Dict] = field(default_factory=list)
    total_time_ms:        float      = 0.0

    def log(self, agent: str, decision: str, detail: str = "", score: float = 1.0):
        self.agent_log.append({
            "agent":    agent,
            "decision": decision,
            "detail":   detail,
            "score":    score,
            "ts_ms":    round(time.time() * 1000),
        })

    def to_result_dict(self) -> dict:
        """Convert AgentState back to the result dict format app.py expects."""
        return {
            "symptoms":                self.final_symptoms,
            "final_symptoms":          self.final_symptoms,
            "triage":                  {"level": self.triage_level},
            "allow_pills":             self.allow_pills,
            "_high_risk_freeze":       self.high_risk_freeze,
            "_freeze_dominant":        self.freeze_dominant,
            "_freeze_syndrome":        self.freeze_syndrome,
            "red_flag_block": {
                "red_flags":            self.observed_red_flags + self.risk_escalators + self.watch_for_flags,
                "observed_red_flags":   self.observed_red_flags,
                "risk_escalators":      self.risk_escalators,
                "watch_for_flags":      self.watch_for_flags,
                "observed_count":       len(self.observed_red_flags),
                "red_flag_count":       len(self.observed_red_flags) + len(self.risk_escalators),
            },
            "evidence_assessment": {
                "output_state":          self.output_state,
                "syndrome_label":        self.syndrome_label,
                "secondary_pattern":     self.secondary_pattern,
                "detected_systems":      self.detected_systems,
                "etiology_allowed":      self.etiology_allowed,
                "etiology_level":        self.etiology_level,
                "disease_label_allowed": self.disease_label_allowed,
                "all_vague":             self.is_vague_only,
                "has_red_flags":         bool(self.observed_red_flags),
            },
            "nexus_etiology": {
                "etiology":   self.etiology,
                "confidence": self.etiology_confidence,
                "scores":     self.etiology_scores,
            },
            "nexus_detected_systems":      self.detected_systems,
            "nexus_suggested_questions":   self.follow_up_questions,
            "nexus_otc_hints":             self.otc_suggestions if self.allow_pills else [],
            "nexus_stats":                 {"mechanisms_activated": self.mechanisms_activated},
            "nexus_consistency":           {"consistency_score": self.consistency_score, "reliable": self.consistency_score > 0},
            "top_diseases":                self.top_diseases,
            "_verifier_passed":            self.verifier_passed,
            "_verifier_issues":            self.verifier_issues,
            "_agent_log":                  self.agent_log,
            "_total_time_ms":             self.total_time_ms,
        }


# ─────────────────────────────────────────────────────────────────────────────
# BASE AGENT
# ─────────────────────────────────────────────────────────────────────────────

class BaseAgent:
    name = "BaseAgent"

    def run(self, state: AgentState) -> AgentState:
        t0 = time.time()
        try:
            self._run(state)
            state.log(self.name, "ok", score=1.0)
        except Exception as e:
            state.log(self.name, "error", str(e), score=0.0)
            print(f"[{self.name}] ERROR (non-blocking): {e}")
        state.log(self.name, "timing", f"{(time.time()-t0)*1000:.1f}ms")
        return state

    def _run(self, state: AgentState):
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 1: INTAKE
# ─────────────────────────────────────────────────────────────────────────────

class IntakeAgent(BaseAgent):
    """
    Extract structured symptom data from user input.
    - Merges selected UI symptoms with text-extracted symptoms
    - Detects severity/duration hints from free text
    - Flags vague-only presentations
    Output: state.final_symptoms, severity_hints, duration_hints, is_vague_only
    """
    name = "IntakeAgent"

    _VAGUE = {
        "swelling", "redness", "dizziness", "nausea", "weakness", "fatigue",
        "pain", "itching", "tingling", "numbness", "malaise", "chills",
        "loss of appetite", "feeling unwell",
    }
    _SEVERITY_WORDS = {
        "severe": "severe", "very": "severe", "extreme": "severe", "worst": "severe",
        "moderate": "moderate", "mild": "mild", "slight": "mild", "little": "mild",
        "sharp": "severe", "dull": "mild",
    }
    _DURATION_PATTERNS = [
        (r"(\d+)\s*days?",   "days"),
        (r"(\d+)\s*hours?",  "hours"),
        (r"(\d+)\s*weeks?",  "weeks"),
        (r"since\s+(\w+)",   "onset"),
    ]

    def _run(self, state: AgentState):
        import re

        # Merge: selected_symptoms are source of truth
        syms = list(state.selected_symptoms)

        # Text extraction: supplement only (don't replace)
        text_syms = state.raw_result.get("text_symptoms", [])
        existing = {s.lower().strip() for s in syms}
        for s in text_syms:
            if s.lower().strip() not in existing:
                syms.append(s)

        state.final_symptoms = syms

        # Severity hints from free text
        text = state.user_text.lower()
        sev = {}
        for sym in syms:
            for word, level in self._SEVERITY_WORDS.items():
                if word in text and sym.lower() in text:
                    sev[sym] = level
                    break
        state.severity_hints = sev

        # Duration hints
        dur = {}
        for pattern, unit in self._DURATION_PATTERNS:
            m = re.search(pattern, text)
            if m:
                dur["_general"] = f"{m.group(1) if m.lastindex else m.group()} {unit}"
                break
        state.duration_hints = dur

        # Vague-only flag
        sym_set = {s.lower().strip() for s in syms}
        state.is_vague_only = bool(sym_set) and all(s in self._VAGUE for s in sym_set)

        state.log(self.name, "symptoms_extracted",
                  f"{len(syms)} symptoms, vague_only={state.is_vague_only}")


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 2: MECHANISM
# ─────────────────────────────────────────────────────────────────────────────

class MechanismAgent(BaseAgent):
    """
    Count and classify mechanism evidence for the symptom set.
    Reads from nexus_medical's loaded mechanism data via the nexus instance.
    Output: state.mechanisms_activated, viral_mech_count, bacterial_mech_count
    """
    name = "MechanismAgent"

    def __init__(self, nexus_instance=None):
        self.nexus = nexus_instance

    def _run(self, state: AgentState):
        # Pull from raw_result (already computed by nexus_medical.reason())
        stats = state.raw_result.get("nexus_stats") or {}
        state.mechanisms_activated  = stats.get("mechanisms_activated", 0)
        _context                     = stats.get("mechanisms_context", 0)
        _total_matched               = stats.get("mechanisms_total_matched",
                                                  state.mechanisms_activated)

        # Viral/bacterial split from etiology scores
        etiol  = state.raw_result.get("nexus_etiology") or {}
        scores = etiol.get("scores") or {}
        total  = state.mechanisms_activated
        state.viral_mech_count     = int(total * scores.get("viral", 0))
        state.bacterial_mech_count = int(total * scores.get("bacterial", 0))

        # Pull top mechanism evidence strings for display
        thinking = state.raw_result.get("nexus_thinking") or []
        state.mechanism_evidence = [
            {"step": s.get("step"), "detail": s.get("detail", "")}
            for s in thinking
            if s.get("step") in ("MECHANISM_CHAIN", "EFFECT_BRIDGE", "GRAPH_SYNERGY")
        ]

        state.log(self.name, "mechanisms_counted",
                  f"total={total} viral≈{state.viral_mech_count} bact≈{state.bacterial_mech_count}")


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 3: DISEASE RANKING
# ─────────────────────────────────────────────────────────────────────────────

class DiseaseRankingAgent(BaseAgent):
    """
    Rank diseases by evidence score and detect body systems.
    Output: state.top_diseases, detected_systems, consistency_score
    """
    name = "DiseaseRankingAgent"

    def _run(self, state: AgentState):
        state.top_diseases     = state.raw_result.get("nexus_diagnoses", [])[:8]
        state.detected_systems = state.raw_result.get("nexus_detected_systems", [])
        _nc_raw = state.raw_result.get("nexus_consistency") or 0
        state.consistency_score = (_nc_raw.get("consistency_score", 0.0)
                                   if isinstance(_nc_raw, dict) else float(_nc_raw))

        _blocked = state.raw_result.get("nexus_blocked_diseases", [])
        _blocked_str = (f" | blocked={[b['disease'] for b in _blocked[:2]]}"
                        if _blocked else "")
        state.log(self.name, "diseases_ranked",
                  f"top={state.top_diseases[0].get('disease','?') if state.top_diseases else 'none'}"
                  f"{_blocked_str} consistency={state.consistency_score:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 4: RED FLAG  (independent from disease ranking)
# ─────────────────────────────────────────────────────────────────────────────

class RedFlagAgent(BaseAgent):
    """
    Independent emergency triage. Runs even if other agents fail.
    Uses high_risk_gate for freeze detection and red_flags.json for rule matching.
    Output: state.observed_red_flags, risk_escalators, watch_for_flags,
            triage_level, high_risk_freeze, freeze_dominant, freeze_syndrome
    """
    name = "RedFlagAgent"

    def _run(self, state: AgentState):
        # Pull from raw_result (already computed by nexus_runner + app.py merge)
        rfd = state.raw_result.get("red_flag_block") or {}
        state.observed_red_flags = rfd.get("observed_red_flags", [])
        state.risk_escalators    = rfd.get("risk_escalators", [])
        state.watch_for_flags    = rfd.get("watch_for_flags", [])

        # Triage
        triage = state.raw_result.get("triage") or {}
        state.triage_level = triage.get("level", "MODERATE").upper()

        # High-risk freeze
        state.high_risk_freeze = bool(state.raw_result.get("_high_risk_freeze"))
        state.freeze_dominant  = state.raw_result.get("_freeze_dominant", "")
        state.freeze_syndrome  = state.raw_result.get("_freeze_syndrome", "")

        # Pills
        state.allow_pills = bool(state.raw_result.get("allow_pills", True))
        if state.high_risk_freeze:
            state.allow_pills = False

        state.log(self.name, "triage_assessed",
                  f"level={state.triage_level} freeze={state.high_risk_freeze} "
                  f"A={len(state.observed_red_flags)} B={len(state.risk_escalators)} "
                  f"C={len(state.watch_for_flags)}")


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 5: EVIDENCE GATE
# ─────────────────────────────────────────────────────────────────────────────

class EvidenceGateAgent(BaseAgent):
    """
    Apply 4-layer evidence gate to determine output granularity.
    Output: state.output_state, syndrome_label, etiology_allowed,
            disease_label_allowed, etiology_level, etiology, etiology_confidence
    """
    name = "EvidenceGateAgent"

    def _run(self, state: AgentState):
        # If freeze is already active, set conservative gate immediately
        if state.high_risk_freeze:
            state.output_state         = "syndrome"
            state.syndrome_label       = state.freeze_syndrome
            state.etiology_allowed     = False
            state.etiology_level       = "uncertain"
            state.disease_label_allowed= False
            state.etiology             = "uncertain"
            state.etiology_confidence  = 0.0
            state.etiology_scores      = {}
            state.log(self.name, "freeze_applied", state.freeze_syndrome)
            return

        # Use evidence_assessment from raw_result (computed by evidence_gate module)
        ea = state.raw_result.get("evidence_assessment") or {}
        state.output_state          = ea.get("output_state", "triage_only")
        state.syndrome_label        = ea.get("syndrome_label", "")
        state.secondary_pattern     = ea.get("secondary_pattern", "")
        state.etiology_allowed      = ea.get("etiology_allowed", False)
        state.etiology_level        = ea.get("etiology_level", "uncertain")
        state.disease_label_allowed = ea.get("disease_label_allowed", False)

        # Etiology from classifier
        etiol = state.raw_result.get("nexus_etiology") or {}
        state.etiology            = etiol.get("etiology", "uncertain")
        state.etiology_confidence = float(etiol.get("confidence") or 0)
        state.etiology_scores     = etiol.get("scores") or {}
        state.etiology_reasoning  = etiol.get("reasoning") or []

        state.log(self.name, "gate_applied",
                  f"state={state.output_state} syndrome='{state.syndrome_label}' "
                  f"etiol={state.etiology_allowed}/{state.etiology_level}")



# ─────────────────────────────────────────────────────────────────────────────
# AGENT 6.5: QUESTION AGENT — active follow-up generation (G.5 #10)
# ─────────────────────────────────────────────────────────────────────────────

class QuestionAgent(BaseAgent):
    """
    Generate active follow-up questions that disambiguate top diagnoses.

    Strategy:
      1. Pull top-3 diagnoses' `reasoning_trace.6_missing_evidence.would_strengthen`
      2. Score each missing symptom by DISCRIMINATIVE power:
         - +2 if it strengthens top-1 ONLY (not top-2/3)
         - +1 if it strengthens top-1 + only one of top-2/3
         -  0 if it strengthens all of top-1/2/3 equally (no discrimination)
      3. Pick top 2-3 questions, phrase as natural follow-ups
      4. Skip if top-1 score > 0.7 (confident enough, no need to ask)

    Output: state.follow_up_questions = [
        {"question": "...", "asks_about": "symptom", "disambiguates": ["DzA", "DzB"]}
    ]
    """
    name = "QuestionAgent"

    # Symptom → natural-language question phrasing
    # If symptom not in map, fall back to generic "Have you experienced X?"
    QUESTION_TEMPLATES = {
        "diaphoresis": "Have you been sweating a lot, especially cold sweats?",
        "nausea": "Have you felt nauseous or wanted to vomit?",
        "vomiting": "Have you actually vomited?",
        "shortness of breath": "Have you had any trouble breathing or felt short of breath?",
        "pleuritic chest pain": "Does the chest pain get worse when you breathe in deeply?",
        "tachycardia": "Has your heart been racing or beating fast?",
        "fever": "Have you had a fever or felt feverish?",
        "chills": "Have you had chills or shivers?",
        "headache": "Have you had a headache, and if so where in the head?",
        "thunderclap headache": "Did the headache come on suddenly — worst of your life?",
        "neck stiffness": "Can you touch your chin to your chest comfortably?",
        "photophobia": "Does bright light bother your eyes more than usual?",
        "vision changes": "Have you noticed any changes in your vision — blurring, spots, loss?",
        "focal weakness": "Have you had weakness on one side of the body?",
        "slurred speech": "Has your speech been slurred or hard to understand?",
        "abdominal pain": "Where exactly is the abdominal pain located?",
        "right upper quadrant pain": "Is the abdominal pain in the upper right area, just below the ribs?",
        "rebound tenderness": "Does pressing on your belly and quickly releasing cause sharper pain?",
        "dysuria burning": "Does it burn or sting when you urinate?",
        "urinary frequency": "Have you been urinating more often than usual?",
        "back pain": "Have you had any back pain — and if so, what area?",
        "flank pain": "Have you had pain in your sides or lower back?",
        "tearing chest or back pain": "Does the pain feel like tearing or ripping?",
        "blood pressure differential between arms": "Has anyone measured blood pressure in both arms?",
        "calf pain": "Have you noticed pain or swelling in either calf?",
        "leg swelling": "Has one of your legs been more swollen than the other?",
        "edema": "Have you noticed swelling anywhere — face, hands, or legs?",
        "proteinuria with hypertension": "If you've had urine tested recently, was there protein in it?",
        "syncope": "Have you fainted or come close to fainting?",
        "altered mental status": "Has anyone said you seem confused or not yourself?",
        "rash": "Have you noticed any rash or skin changes?",
        "joint pain": "Have you had pain in any joints?",
        "fatigue": "Have you felt unusually tired or drained?",
        "weight loss": "Have you lost weight unintentionally?",
        "cough": "Have you had a cough, and is anything coming up?",
        "productive cough": "When you cough, is anything coming up?",
        "hemoptysis": "Have you coughed up any blood?",
        "diarrhea": "Have you had loose stools or diarrhea?",
        "bloody stool": "Have you noticed any blood in your stool?",
    }

    def run(self, state: AgentState) -> AgentState:
        t0 = time.time()
        top_diseases = state.top_diseases or []

        # Skip if no good candidates
        if not top_diseases or len(top_diseases) < 2:
            state.log(self.name, "skipped",
                      "<2 candidates — nothing to disambiguate")
            return state

        # Decision: skip if top-1 is CLEARLY ahead of top-2.
        # Use *gap* not absolute score — G.5 multipliers can inflate scores
        # above 1.0, but if top-2 is also high, ambiguity remains.
        top1_score = top_diseases[0].get("score", 0.0)
        top2_score = top_diseases[1].get("score", 0.0) if len(top_diseases) >= 2 else 0.0
        score_gap = top1_score - top2_score
        # Skip only if top-1 leads by a wide margin AND top-1 is reasonably high
        if score_gap >= 0.3 and top1_score >= 0.6:
            state.log(self.name, "skipped",
                      f"top-1 {top1_score:.2f} leads top-2 {top2_score:.2f} "
                      f"by {score_gap:.2f} — confident enough")
            return state

        # Gather missing evidence per top-3 disease
        # Each entry: {disease, missing: [symptom1, symptom2, ...]}
        per_disease_missing = []
        for d in top_diseases[:3]:
            trace = d.get("reasoning_trace", {})
            missing_evidence = trace.get("6_missing_evidence", {})
            if isinstance(missing_evidence, dict):
                missing = missing_evidence.get("would_strengthen", []) or []
            else:
                missing = []
            per_disease_missing.append({
                "disease": d.get("disease", "?"),
                "score":   d.get("score", 0.0),
                "missing": [str(m).lower().strip() for m in missing if m],
            })

        if not any(pd["missing"] for pd in per_disease_missing):
            state.log(self.name, "skipped",
                      "no missing-evidence info in reasoning traces")
            return state

        # Score each candidate symptom by discriminative power
        # symptom appears in disease #i's missing list → it would strengthen i
        # Best questions: appear in top-1 but NOT in top-2/3 (or only one of them)
        all_missing_syms = set()
        for pd in per_disease_missing:
            all_missing_syms.update(pd["missing"])

        symptom_scores = []
        for sym in all_missing_syms:
            in_top1 = sym in per_disease_missing[0]["missing"]
            in_top2 = (len(per_disease_missing) > 1
                       and sym in per_disease_missing[1]["missing"])
            in_top3 = (len(per_disease_missing) > 2
                       and sym in per_disease_missing[2]["missing"])

            # Discriminative score:
            #   strengthens only top-1 → +2 (highly disambiguating)
            #   strengthens top-1 + one other → +1
            #   strengthens all (or none of top-1) → 0
            score = 0
            disambiguates = []
            if in_top1 and not in_top2 and not in_top3:
                score = 2
                disambiguates = [per_disease_missing[0]["disease"]]
            elif in_top1 and (in_top2 ^ in_top3):  # XOR — exactly one of 2/3
                score = 1
                disambiguates = [per_disease_missing[0]["disease"]]
                if in_top2:
                    disambiguates.append(per_disease_missing[1]["disease"])
                if in_top3:
                    disambiguates.append(per_disease_missing[2]["disease"])
            # If symptom strengthens only top-2 (not top-1), still useful to ask
            elif not in_top1 and (in_top2 or in_top3):
                score = 1
                if in_top2:
                    disambiguates.append(per_disease_missing[1]["disease"])
                if in_top3:
                    disambiguates.append(per_disease_missing[2]["disease"])

            if score > 0:
                symptom_scores.append((sym, score, disambiguates))

        # Sort by discriminative score desc, then by alphabetical for stability
        symptom_scores.sort(key=lambda x: (-x[1], x[0]))

        # Build up to 3 follow-up questions
        questions = []
        for sym, score, disambig in symptom_scores[:3]:
            template = self.QUESTION_TEMPLATES.get(sym)
            if template:
                question_text = template
            else:
                # Fallback phrasing
                question_text = f"Have you experienced {sym}?"
            questions.append({
                "question":      question_text,
                "asks_about":    sym,
                "disambiguates": disambig,
                "score":         score,
            })

        state.follow_up_questions = questions
        state.log(self.name, "generated",
                  f"{len(questions)} discriminative questions "
                  f"(top symptom: {questions[0]['asks_about']!r})"
                  if questions else "no high-value questions found",
                  score=1.0 if questions else 0.5)
        state.total_time_ms += round((time.time() - t0) * 1000, 1)
        return state


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 7: ANSWER COMPOSER
# ─────────────────────────────────────────────────────────────────────────────

class AnswerComposer(BaseAgent):
    """
    Compose the final patient-facing answer from gate-approved state.
    Calls deterministic_response.generate_response() with the assembled result dict.
    Output: state.final_answer, state.otc_suggestions
    """
    name = "AnswerComposer"

    def __init__(self, user_input: str = ""):
        self.user_input = user_input

    def _run(self, state: AgentState):
        try:
            from deterministic_response import generate_response
            result_dict = state.to_result_dict()

            # Under high-risk freeze: strip disease candidates before composing
            # so generate_response cannot write disease names into the answer
            if state.high_risk_freeze:
                result_dict["nexus_diagnoses"]         = []
                result_dict["nexus_diagnoses_display"] = []
                result_dict["top_diseases"]            = []
                result_dict["nexus_etiology"]          = {
                    "etiology": "uncertain", "confidence": 0.0, "scores": {}}

            # Enrich with anatomy and other fields from raw_result
            for key in ("anatomy_affected_organs", "anatomy_dominant_sym",
                        "anatomy_low_support", "nexus_predicted_symptoms",
                        "nexus_recommended_tests", "nexus_otc_gating"):
                if key in state.raw_result:
                    result_dict[key] = state.raw_result[key]

            # Only pass diagnoses when label is allowed
            if state.disease_label_allowed:
                for key in ("nexus_diagnoses_display", "nexus_diagnoses"):
                    if key in state.raw_result:
                        result_dict[key] = state.raw_result[key]

            state.final_answer    = generate_response(result_dict, self.user_input)
            state.otc_suggestions = (state.raw_result.get("nexus_otc_hints", [])
                                     if state.allow_pills else [])
            state.log(self.name, "answer_composed", f"length={len(state.final_answer)} chars")
        except Exception as e:
            state.final_answer = f"[AnswerComposer error: {e}]"
            state.log(self.name, "compose_error", str(e), score=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 8: VERIFIER
# ─────────────────────────────────────────────────────────────────────────────

class Verifier(BaseAgent):
    """
    Independent audit of the composed answer.
    Checks:
      - Over-diagnosis: disease label when gate said no
      - Red flag leak: OTC shown when freeze is active
      - Red flag miss: high-risk symptom in input but triage not elevated
      - Etiology leak: viral/bacterial stated when etiol_ok=False
      - Consistency: syndrome label matches detected systems
    Output: state.verifier_passed, state.verifier_issues, state.verifier_patches
    """
    name = "Verifier"

    @staticmethod
    def _load_disease_names():
        """Load disease names from disease_0001.json — auto-updates as you add diseases."""
        # Generic/safety words that appear in disease names but are NOT diseases.
        # Adding them to disease_names triggers false DISEASE_LABEL_LEAK errors.
        _SKIP_GENERIC = {
            "emergency", "syndrome", "disease", "disorder", "infection",
            "attack", "episode", "event", "condition", "failure",
            "crisis", "type", "acute", "chronic", "media",
        }
        try:
            import json as _dj, os as _do
            # Search in proper locations (medical_knowledge/diseases/ is canonical)
            candidates = (
                "medical_knowledge/diseases/disease_0001.json",
                "../medical_knowledge/diseases/disease_0001.json",
                "disease_0001.json",          # legacy fallback
                "../disease_0001.json",       # legacy fallback
            )
            for _dp in candidates:
                if _do.path.exists(_dp):
                    data = _dj.load(open(_dp, encoding="utf-8"))
                    names = set()
                    for d in (data if isinstance(data, list) else []):
                        n = d.get("disease_name", "").lower().strip()
                        if n:
                            names.add(n)
                            # Also add short alias (last word) — but skip generic words
                            parts = n.split()
                            if len(parts) > 1:
                                last = parts[-1]
                                if last not in _SKIP_GENERIC and len(last) > 4:
                                    names.add(last)
                    return names
        except Exception:
            pass
        return {  # fallback subset
            "influenza", "flu", "pneumonia", "gastroenteritis", "appendicitis",
            "meningitis", "pyelonephritis", "cellulitis", "sepsis", "covid",
        }

    _DISEASE_NAMES = None  # loaded lazily on first Verifier._run()
    _ETIOL_WORDS = {"viral infection", "bacterial infection", "virus", "bacteria"}

    def _run(self, state: AgentState):
        # Lazy-load disease names from disease_0001.json
        if Verifier._DISEASE_NAMES is None:
            Verifier._DISEASE_NAMES = Verifier._load_disease_names()

        issues  = []
        patches = []
        answer  = state.final_answer.lower()

        # Check 1: OTC leak under freeze
        if state.high_risk_freeze and state.otc_suggestions:
            issues.append("OTC_LEAK: OTC suggestions present during high-risk freeze")
            state.otc_suggestions = []
            patches.append("Cleared OTC suggestions due to high-risk freeze")

        # Check 2: Disease label leak — flag AND suppress
        if not state.disease_label_allowed:
            _clean_answer = state.final_answer
            for name in self._DISEASE_NAMES:
                if name in answer and f"seek care for {name}" not in answer:
                    if f"rule out {name}" not in answer:
                        issues.append(f"DISEASE_LABEL_LEAK: '{name}' stated but label_ok=False")
                        patches.append(f"Removed disease label: {name}")
                        # Case-insensitive removal from answer
                        import re as _re
                        _clean_answer = _re.sub(
                            r'' + _re.escape(name) + r'',
                            "this condition", _clean_answer, flags=_re.IGNORECASE)
            if len(patches) > 0:
                state.final_answer = _clean_answer
                answer = _clean_answer.lower()  # update for subsequent checks

        # Check 3: Etiology leak
        if not state.etiology_allowed:
            for w in self._ETIOL_WORDS:
                if w in answer:
                    issues.append(f"ETIOLOGY_LEAK: '{w}' in answer but etiol_ok=False")

        # Check 4: Red flag miss
        _HR_SYMS = {"chest pain", "shortness of breath", "syncope", "bloody stool",
                    "hematemesis", "altered mental status", "focal weakness"}
        sym_set = {s.lower().strip() for s in state.final_symptoms}
        has_hr = bool(sym_set & _HR_SYMS)
        if has_hr and state.triage_level not in ("PROMPT", "URGENT", "EMERGENCY", "EMERGENT"):
            issues.append(
                f"TRIAGE_MISS: high-risk symptom present but triage={state.triage_level}")

        # Check 5: Freeze consistency
        if state.high_risk_freeze and state.syndrome_label != state.freeze_syndrome:
            issues.append(
                f"FREEZE_INCONSISTENCY: syndrome='{state.syndrome_label}' "
                f"vs freeze_syndrome='{state.freeze_syndrome}'")
            state.syndrome_label = state.freeze_syndrome
            patches.append("Aligned syndrome_label with freeze_syndrome")

        state.verifier_passed  = len(issues) == 0
        state.verifier_issues  = issues
        state.verifier_patches = patches

        score = 1.0 if state.verifier_passed else max(0.3, 1.0 - len(issues) * 0.2)
        state.log(self.name, "verified",
                  f"passed={state.verifier_passed} issues={len(issues)}", score=score)
        if issues:
            print(f"[Verifier] {len(issues)} issue(s): {issues}")


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 9: REWARD TRACE
# ─────────────────────────────────────────────────────────────────────────────

class RewardTrace(BaseAgent):
    """
    Summarise the pipeline run as a structured trace record.
    Records what each agent decided, which were good, and flags problems.
    Output: state.agent_log enriched, state.total_time_ms
    """
    name = "RewardTrace"

    def _run(self, state: AgentState):
        if state.agent_log:
            t_start = state.agent_log[0].get("ts_ms", 0)
            t_end   = state.agent_log[-1].get("ts_ms", t_start)
            state.total_time_ms = t_end - t_start

        # Print a compact trace
        print(f"[NEXUS-AGENT] pipeline complete in {state.total_time_ms:.0f}ms")
        print(f"  intake:   {len(state.final_symptoms)} symptoms  vague={state.is_vague_only}")
        _ctx = (state.raw_result.get("nexus_stats") or {}).get("mechanisms_context", 0)
        print(f"  mechs:    {state.mechanisms_activated} direct + {_ctx} context activated")
        print(f"  red_flag: A={len(state.observed_red_flags)} B={len(state.risk_escalators)} "
              f"C={len(state.watch_for_flags)} freeze={state.high_risk_freeze}")
        print(f"  gate:     state={state.output_state} syndrome='{state.syndrome_label}' "
              f"etiol={state.etiology_allowed}/{state.etiology_level}")
        print(f"  verifier: passed={state.verifier_passed}"
              + (f" ISSUES: {state.verifier_issues}" if state.verifier_issues else ""))


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

class AgentOrchestrator:
    """
    Runs the full agent pipeline in order.
    Each agent receives the shared AgentState and mutates it.
    Existing nexus_medical/nexus_runner results are consumed via state.raw_result.
    """

    def run(
        self,
        selected_symptoms: List[str],
        user_text: str,
        raw_result: dict,
        nexus_instance=None,
    ) -> AgentState:

        state = AgentState(
            selected_symptoms = selected_symptoms,
            user_text         = user_text,
            raw_result        = raw_result,
        )

        pipeline = [
            IntakeAgent(),
            MechanismAgent(nexus_instance),
            DiseaseRankingAgent(),
            RedFlagAgent(),
            EvidenceGateAgent(),
            QuestionAgent(),                  # G.5 #10: active follow-up
            AnswerComposer(user_text),
            Verifier(),
            RewardTrace(),
        ]

        for agent in pipeline:
            agent.run(state)

        return state