"""
Deterministic Response Generator v3

Fixes from Gemini feedback:
  - Follow-up questions ranked by clinical relevance to THIS case
  - Tests conditional on symptoms, not template
  - DVT/red flags expressed as conditional if context missing
  - Uncertainty expressed when etiology is uncertain
  - Front-end and back-end severity aligned

Modes:
  - PATIENT mode: risk + flags + questions + care. NO disease names.
  - CLINICAL mode: full differential for doctors.
  - PROOF mode (overlay): appends cascade-proof block showing how each
    diagnosis was derived from organ→failure_mode→cascade→matched symptoms.
    Activated by mode="patient_proof", "clinical_proof", or proof=True kwarg.
"""


def generate_response(result: dict, user_input: str = "", mode: str = "patient",
                      proof: bool = False) -> str:
    """
    Generate the user-facing or clinical response, optionally appending a
    cascade-proof block.

    Args:
        result:     dict with reason() + evidence_gate + anatomy_bridge output
        user_input: original free-text input (for context)
        mode:       "patient" | "clinical" | "patient_proof" | "clinical_proof"
        proof:      shortcut to enable proof block on either mode

    Returns:
        rendered text (markdown). When proof is on, a "═══ PROOF ═══" section
        is appended showing the cascade chain for each diagnosis.
    """
    # Normalize: support both `mode="patient_proof"` and `proof=True`
    base_mode = mode
    if mode.endswith("_proof"):
        proof = True
        base_mode = mode.replace("_proof", "")

    if base_mode == "clinical":
        body = _clinical_response(result, user_input)
    else:
        body = _patient_response(result, user_input)

    if proof:
        body = body.rstrip() + "\n\n" + _proof_block(result)

    return body


# ═══════════════════════════════════════════════
# PATIENT MODE
# ═══════════════════════════════════════════════

def _patient_response(result: dict, user_input: str = "") -> str:
    parts = []
    symptoms = result.get("symptoms", result.get("final_symptoms", []))
    # Evidence assessment from 4-layer gate
    ea = result.get("evidence_assessment") or {}
    _output_state        = ea.get("output_state", "syndrome")
    _syndrome_label      = ea.get("syndrome_label", "")
    _etiology_allowed    = ea.get("etiology_allowed", True)
    _etiology_level      = ea.get("etiology_level", "uncertain")  # "uncertain"|"leaning"|"confirmed"
    _disease_label_ok    = ea.get("disease_label_allowed", False)
    _all_vague           = ea.get("all_vague", False)
    # 3-bucket red flag reads
    _rfd           = result.get("red_flag_block") or {}
    red_flags      = _rfd.get("observed_red_flags", [])   # A: triage-relevant
    risk_escalators= _rfd.get("risk_escalators", [])      # B: caution wording
    watch_for_flags= _rfd.get("watch_for_flags", [])      # C: UI "seek care if..."
    # Legacy fallback
    if not red_flags and not risk_escalators:
        red_flags = result.get("red_flags", result.get("nexus_red_flags", []))
    triage = result.get("triage", {})
    predicted = result.get("nexus_predicted_symptoms", [])
    systems = result.get("nexus_detected_systems", [])
    otc = result.get("otc_suggestions", result.get("nexus_otc_hints", []))
    questions = result.get("nexus_suggested_questions", [])
    etiology = result.get("nexus_etiology", {})
    danger = result.get("nexus_danger_bias", 1.0)
    combo_danger = result.get("nexus_combo_danger", 1.0)
    combos = result.get("nexus_combos", [])
    recommended_tests = result.get("nexus_recommended_tests", [])

    if not symptoms:
        return "I wasn't able to identify specific symptoms. Could you describe how you feel in more detail?"

    sym_set = set(s.lower().replace("_", " ") for s in symptoms)
    sym_text = _join_natural(symptoms)
    parts.append(f"Thank you for sharing your symptoms. I've received: **{sym_text}**.")

    # ── RISK LEVEL ──
    level = triage.get("level", "routine").lower()
    effective_danger = max(danger, combo_danger)

    # ── HIGH-PRIORITY SYMPTOM OVERRIDE ──
    # Certain symptoms must NEVER default to routine self-care
    HIGH_PRIORITY_SYMPTOMS = {
        "chest pain": {
            "min_triage": "urgent",
            "red_flags": [
                "If chest pain is crushing, pressure-like, or radiating to arm/jaw/back — call emergency services",
                "If accompanied by shortness of breath, sweating, or fainting — this is a medical emergency",
                "If pain worsens with exertion — seek immediate evaluation",
            ],
            "questions": [
                "What does the chest pain feel like? (pressure/squeezing, sharp/stabbing, burning, or aching)",
                "How long has the chest pain lasted? Is it constant or comes and goes?",
                "Do you have shortness of breath, sweating, or lightheadedness with the chest pain?",
                "Does the pain spread to your arm, jaw, neck, or back?",
                "Does it get worse with physical activity or deep breathing?",
            ],
        },
        "shortness of breath": {
            "min_triage": "urgent",
            "red_flags": [
                "Sudden onset shortness of breath requires urgent evaluation",
                "If accompanied by chest pain or calf swelling — seek emergency care (PE risk)",
            ],
            "questions": [
                "Did the shortness of breath come on suddenly or gradually?",
                "Is it worse when lying down or with exertion?",
            ],
        },
        "numbness": {
            "min_triage": "urgent",
            "red_flags": [
                "If numbness is on one side of the face, arm, or leg — seek emergency care immediately (stroke risk)",
                "If numbness started suddenly and is accompanied by weakness, speech difficulty, or vision changes — call emergency services",
                "If you cannot feel one side of your body normally — this requires urgent evaluation",
            ],
            "questions": [
                "Is the numbness on one side of your body, or both sides?",
                "Where exactly is the numbness — face, arm, hand, leg, or a specific area?",
                "Did the numbness start suddenly or come on gradually?",
                "Do you also have any weakness, difficulty speaking, or vision changes?",
            ],
        },
        "weakness": {
            "min_triage": "urgent",
            "red_flags": [
                "If weakness is on one side of the body — seek emergency care immediately (stroke risk)",
                "If accompanied by numbness, speech difficulty, or vision changes — call emergency services",
            ],
            "questions": [
                "Is the weakness on one side of your body or both sides?",
                "Did it start suddenly or gradually?",
                "Do you also have numbness, difficulty speaking, or vision changes?",
            ],
        },
    }

    forced_questions = []
    forced_red_flags = []
    forced_triage = None

    for sym_key, rules in HIGH_PRIORITY_SYMPTOMS.items():
        if any(sym_key in s for s in sym_set):
            forced_triage = rules["min_triage"]
            forced_red_flags.extend(rules["red_flags"])
            forced_questions.extend(rules["questions"])

    # Apply triage override from high-priority symptoms
    if forced_triage == "emergency" or level == "emergency" or effective_danger >= 2.0:
        parts.append(
            "### ⚠️ Immediate Medical Attention Needed\n\n"
            "This combination of symptoms may indicate a serious condition. "
            "**Please go to an emergency room or call emergency services now.**"
        )
    elif forced_triage == "urgent" or level == "urgent" or effective_danger >= 1.4 or len(red_flags) >= 3:
        parts.append(
            "### ⚡ Prompt Medical Evaluation Recommended\n\n"
            "This symptom pattern warrants medical attention. "
            "**I recommend seeing a doctor as soon as possible.** "
            "If symptoms worsen suddenly, go to the emergency room."
        )
    else:
        parts.append(
            "### ✅ Monitor and Self-Care\n\n"
            "Based on the information provided, this does not appear to be an emergency. "
            "However, please monitor your symptoms and see a doctor if they persist or worsen."
        )

    # ── RED FLAGS (case-specific filtering) ──
    # Only show red flags that are clinically relevant to THIS case's symptoms.
    # Generic template flags (like "chest pain" for a diarrhea case) dilute trust.
    all_flags = forced_red_flags + [rf.get("red_flag", rf.get("trigger", "")) for rf in red_flags[:4] if rf.get("red_flag", rf.get("trigger", "")) and len(rf.get("red_flag", rf.get("trigger", ""))) > 3]

    # Case-specific filter: remove flags that reference symptoms the user doesn't have
    # UNLESS the flag is a genuine escalation warning (e.g. "if X develops, seek care")
    # Also filter out bare symptom names — "diarrhea" alone is not a red flag,
    # "bloody diarrhea" or "severe dehydration from diarrhea" would be.
    ESCALATION_KEYWORDS = {"if ", "call", "seek", "go to", "emergency", "develops"}
    filtered_flags = []
    for f in all_flags:
        f_lower = f.lower().strip()

        # Skip bare symptom names used as red flags — too coarse
        if f_lower in sym_set:
            continue

        # Keep escalation-style flags (they describe what to WATCH for)
        if any(kw in f_lower for kw in ESCALATION_KEYWORDS):
            filtered_flags.append(f)
            continue
        # For simple symptom-name flags, only keep if the symptom is in the case
        # or closely related to symptoms in the case
        is_relevant = any(s in f_lower for s in sym_set)
        if is_relevant:
            filtered_flags.append(f)
        # else: skip — this flag references a symptom not in this case

    # Convert bare red-flag labels to patient-readable conditional sentences.
    # Internal labels like "worst headache of life" should never render verbatim.
    _RF_LABEL_TO_SENTENCE = {
        "worst headache of life":   "If this becomes the worst headache you have ever had, seek emergency care immediately.",
        "thunderclap headache":     "If the headache started suddenly and severely ('thunderclap'), seek emergency care immediately.",
        "sudden severe headache":   "If the headache started suddenly and severely, seek emergency care immediately.",
        "signs of meningitis":      "If you develop a stiff neck, sensitivity to light, or high fever alongside headache, seek urgent care.",
        "signs of stroke":          "If you notice sudden weakness, face drooping, arm weakness, or speech difficulty, call emergency services (FAST).",
        "loss of consciousness":    "If you faint or lose consciousness, seek emergency care.",
        "seizure":                  "If you experience a seizure, seek emergency care immediately.",
        "difficulty breathing":     "If breathing becomes difficult or laboured, seek urgent medical care.",
        "signs of sepsis":          "If you develop high fever with rapid heart rate and confusion, seek emergency care.",
    }

    def _rf_to_sentence(flag_text: str) -> str:
        """Convert bare label to patient sentence, or return as-is if already a sentence."""
        fl = flag_text.lower().strip()
        # Already a sentence (contains verb or action word)
        if any(kw in fl for kw in ("if ", "seek", "call", "go to", "contact", "emergency")):
            return flag_text
        # Known label mapping
        for label, sentence in _RF_LABEL_TO_SENTENCE.items():
            if label in fl:
                return sentence
        # Unknown bare label — wrap generically
        return f"If {flag_text.lower()} occurs or worsens, seek medical care."

    if filtered_flags:
        seen = set()
        deduped = []
        for f in filtered_flags:
            sentence = _rf_to_sentence(f)
            if sentence not in seen:
                seen.add(sentence)
                deduped.append(sentence)

        direct = [f for f in deduped if not any(rf.get("conditional") for rf in red_flags if rf.get("red_flag", "") == f)]
        conditional_rf = [f for f in deduped if f not in direct]

        if direct:
            parts.append("**Warning signs — seek medical care if any apply:**")
            for ft in direct:
                parts.append(f"- {ft}")
        if conditional_rf:
            parts.append("**Watch out for these situations:**")
            for ft in conditional_rf:
                parts.append(f"- {ft}")

    # ── BODY SYSTEMS (filtered: only show systems with strong symptom support) ──
    if systems:
        friendly = {
            "cardiac": "heart", "cardiovascular": "heart and blood vessels",
            "respiratory": "lungs and breathing", "neurologic": "nervous system",
            "gastrointestinal": "digestive system", "gi": "digestive system",
            "renal": "kidneys", "urinary": "urinary system",
            "musculoskeletal": "muscles and joints",
            "cutaneous": "skin", "dermatologic": "skin",
            "immune": "immune system",
            "hepatobiliary": "liver and gallbladder",
            "endocrine": "hormonal system",
            "hematologic": "blood system",
            "reproductive": "reproductive system",
        }
        # Only show systems that have at least one strong symptom match.
        # "nervous" appearing for itching+cough is misleading — filter it.
        STRONG_SYS_SYMS = {
            "neurologic": {"headache", "dizziness", "numbness", "confusion", "weakness", "seizure", "tingling"},
            "nervous": {"headache", "dizziness", "numbness", "confusion", "weakness", "seizure", "tingling"},
            "immune": {"fever", "night sweats", "lymphadenopathy"},
        }
        filtered_systems = []
        for s in systems:
            check = STRONG_SYS_SYMS.get(s)
            if check is None:
                # No filter rule → always include (respiratory, dermatologic, etc.)
                filtered_systems.append(s)
            elif sym_set & check:
                # Has strong symptom support → include
                filtered_systems.append(s)
            # else: skip (e.g. "nervous" for itching+cough)

        sys_friendly = list(set(friendly.get(s, s) for s in filtered_systems))[:3]
        if sys_friendly:
            # Build anatomy-aware sentence using organ data if available
            _anatomy_organs = result.get("anatomy_affected_organs") or []
            _dominant_sym   = result.get("anatomy_dominant_sym", "")
            _low_support    = result.get("anatomy_low_support", False)
            _sys_label      = _join_natural(sys_friendly)

            if _anatomy_organs and not _low_support:
                # Rich sentence: name the symptom, system, and specific structures
                _organ_names = [o.replace("_", " ") for o in _anatomy_organs[:3]]
                _organ_str   = _join_natural(_organ_names)
                if _dominant_sym:
                    parts.append(
                        f"\nYour **{_dominant_sym}** appears to involve the **{_sys_label}**. "
                        f"NEXUS identified related structures including the {_organ_str}, "
                        f"but did not find enough evidence to suggest an emergency from the "
                        f"information provided."
                    )
                else:
                    parts.append(
                        f"\nYour symptoms appear to involve the **{_sys_label}** — "
                        f"NEXUS identified related structures including the {_organ_str}."
                    )
            elif _low_support and _dominant_sym:
                # Anatomy couldn't localize the main symptom well
                parts.append(
                    f"\nYour **{_dominant_sym}** appears to involve the **{_sys_label}**. "
                    f"Anatomical localization was limited — more symptom detail would help "
                    f"identify the specific area involved."
                )
            else:
                # Fallback: original simple sentence
                parts.append(f"\nYour symptoms may involve the **{_sys_label}**.")

    # ── SYNDROME FRAMING (state=syndrome, etiol not yet allowable) ──
    if _output_state == "syndrome" and _syndrome_label and not _disease_label_ok:
        _secondary = ea.get("secondary_pattern", "")
        # Build a conservative hypothesis frame using the gate-approved syndrome label
        if not _etiology_allowed:
            _ea_etiol_ev = ea.get("etiology_evidence", "")
            _ea_lvl      = ea.get("etiology_level", "")
            if _ea_lvl == "weak_lean":
                # Weak lean: acknowledge hypothesis but don't commit
                _syn_text = _syndrome_label
                if _secondary and _secondary != _syndrome_label:
                    _syn_text += f" with {_secondary.replace(' syndrome','').strip()} features"
                parts.append(
                    f"\nThe symptom pattern is most consistent with a **{_syn_text}**. "
                    "This is a working hypothesis — more information is needed before a "
                    "specific diagnosis or cause can be identified."
                )
            # else (uncertain): case_hypothesis narrative handles framing
        elif _etiology_allowed and _etiology_level == "leaning":
            # Leaning: can suggest direction but not commit
            parts.append(
                f"\nThe pattern suggests a **{_syndrome_label}**, with a leaning toward "
                "a viral process — though this isn't confirmed without more information."
            )

    # ── ETIOLOGY + UNCERTAINTY ──
    # Phase A: etiology_classifier deleted. Etiology field is always {} now.
    # Pure mechanism reasoning does not classify viral/bacterial/non_infectious.
    # If the cascade matches a specific infectious disease, that's exposed via
    # the disease name itself (subject to evidence_gate label permission).

    # ── FOLLOW-UP QUESTIONS ──
    # Load question templates from questions.json (single source of truth)
    import json as _qj, os as _qos
    _Q_DATA = {}
    for _qpath in ("nexus_engine/questions.json", "nexus_engine/questions.json"):
        if _qos.path.exists(_qpath):
            try:
                _Q_DATA = _qj.load(open(_qpath, encoding="utf-8"))
            except Exception:
                pass
            break

    # G.5 #10: Surface QuestionAgent's discriminative questions if any.
    # QuestionAgent produces questions only when top-1 vs top-2 is ambiguous
    # AND when there's missing-evidence info from reasoning_trace.
    # We treat its output as an ADDITIVE section (doesn't replace HIGH_PRIORITY).
    _agent_questions = questions  # from line 80
    if _agent_questions and isinstance(_agent_questions, list) and len(_agent_questions) > 0:
        # Format: list of dicts {question, asks_about, disambiguates, score}
        # or list of plain strings (legacy)
        formatted_qs = []
        for q in _agent_questions[:3]:
            if isinstance(q, dict) and "question" in q:
                formatted_qs.append(q["question"])
            elif isinstance(q, str):
                formatted_qs.append(q)
        if formatted_qs:
            parts.append("\n**To help me narrow this down, please tell me:**")
            for q in formatted_qs:
                parts.append(f"- {q}")

    # NOTE: Legacy follow-up question generation removed per pure-mechanism design.
    # bridge_hypothesis, case_hypothesis hardcoded paths deleted.
    hyp_gate = "suggest_care"

    # ── WATCH FOR ──
    if predicted:
        # Filter out generic ones, prioritize clinically important
        important_watch = _filter_important_predictions(predicted, sym_set)
        if important_watch:
            parts.append(f"\n**Watch for:** {_join_natural(important_watch[:4])}. If any develop, seek medical care.")

    # ── WHAT TO DO ──
    parts.append("\n**What to do now:**")
    # Use the SAME triage logic as the header — include forced_triage
    _effective_urgent = (forced_triage in ("urgent", "emergency") or
                         level in ("urgent", "emergency") or
                         effective_danger >= 1.4)
    _effective_emergency = (forced_triage == "emergency" or
                            level == "emergency" or
                            effective_danger >= 2.0)
    _has_high_acuity = any(s in sym_set for s in
        ("chest pain", "numbness", "weakness", "shortness of breath"))

    if _effective_emergency:
        parts.append("- **Seek emergency medical care immediately**")
        parts.append("- Do not drive yourself — call for help")
    elif _effective_urgent:
        if _has_high_acuity:
            parts.append("- **Seek same-day medical evaluation** — do not wait")
            parts.append("- If symptoms worsen or you feel worse: go to the ER or call emergency services")
        else:
            parts.append("- See a doctor today or first thing tomorrow")
            parts.append("- Go to the ER if symptoms worsen suddenly")
            parts.append("- Rest and stay hydrated in the meantime")
    else:
        parts.append("- Rest and stay hydrated")
        parts.append("- Monitor symptoms over the next 24-48 hours")
        parts.append("- See a doctor if symptoms persist or get worse")

    # ── OTC (with safety gating) ──
    has_chest_pain = any("chest pain" in s or "chest_pain" in s for s in sym_set)

    if has_chest_pain:
        parts.append("\n*OTC medications are not recommended when chest pain is present. Please see a doctor first.*")
    elif otc and level not in ("emergency",) and effective_danger < 2.0 and not forced_triage == "emergency":
        # OTC combination gate: prefer items that address multiple symptoms,
        # not just the highest-scoring single symptom.
        # e.g. cough + itching + abdominal → don't only show cough suppressants
        _SYMPTOM_OTC_MAP = {
            "cough":        {"dextromethorphan", "guaifenesin", "honey"},
            "nausea":       {"bismuth", "ginger", "dramamine", "dimenhydrinate",
                             "diphenhydramine"},  # diphenhydramine helps nausea too
            "itching":      {"diphenhydramine", "calamine", "hydrocortisone", "cetirizine",
                             "loratadine", "antihistamine", "benadryl"},
            "rash":         {"hydrocortisone", "calamine", "diphenhydramine", "antihistamine",
                             "cetirizine", "loratadine"},
            "redness":      {"hydrocortisone", "calamine", "diphenhydramine", "antihistamine"},
            "diarrhea":     {"loperamide", "bismuth"},
            "abdominal pain": {"bismuth", "antacid", "simethicone",
                              "diphenhydramine"},  # antihistamines can help GI cramping
            "fever":        {"acetaminophen", "ibuprofen", "paracetamol"},
            "headache":     {"acetaminophen", "ibuprofen"},
            "sore throat":  {"throat lozenges", "acetaminophen"},
            "congestion":   {"pseudoephedrine", "saline", "nasal"},
            "vomiting":     {"bismuth", "ginger", "dimenhydrinate"},
        }
        def _otc_score(item: str) -> int:
            """Count how many of the patient's symptoms this OTC addresses."""
            item_lower = item.lower()
            return sum(
                1 for sym, keywords in _SYMPTOM_OTC_MAP.items()
                if sym in sym_set and any(kw in item_lower for kw in keywords)
            )
        # Sort OTCs by symptom coverage; keep top 3
        otc_scored = [(o, _otc_score(o)) for o in otc if isinstance(o, str)]
        otc_scored.sort(key=lambda x: -x[1])
        # Always include at least one for the primary symptom; prefer multi-symptom coverage
        safe_otc = [o for o, _ in otc_scored][:3]

        # Safety gate: suppress OTC when risky combinations present
        has_rash = any("rash" in s for s in sym_set)
        has_neuro = any(s in sym_set for s in ("tingling", "numbness", "confusion", "weakness"))
        has_gi_bleed_risk = any(s in sym_set for s in ("bloody stool", "vomiting"))

        # Don't suggest NSAIDs when rash + neuro (possible drug reaction / autoimmune)
        if has_rash and has_neuro:
            safe_otc = [o for o in safe_otc if "naproxen" not in o.lower() and "ibuprofen" not in o.lower() and "nsaid" not in o.lower()]
            if not safe_otc:
                parts.append("\n*OTC medications are not recommended until a doctor evaluates this symptom combination.*")

        # Don't suggest NSAIDs with GI bleed risk
        if has_gi_bleed_risk:
            safe_otc = [o for o in safe_otc if "naproxen" not in o.lower() and "ibuprofen" not in o.lower()]

        if safe_otc:
            otc_text = _join_natural(safe_otc)
            parts.append(f"\n**For symptom relief:** {otc_text} may help.")
            otc_lower = " ".join(safe_otc).lower()
            if "loperamide" in otc_lower:
                parts.append("*Avoid anti-diarrheal medication if you have fever, bloody stool, or severe abdominal pain.*")
            parts.append("Always follow label instructions and check with a pharmacist if you take other medications.")

    # ── TESTS (justified from workup_engine, with fallback to old template) ──
    # P1-2: prefer the new recommended_workup field — it has justification
    # chains and was built from red flags + differential + etiology, not
    # templates. Falls back to _conditional_tests when workup_engine wasn't
    # available in the pipeline run.
    workup_items = result.get("recommended_workup", [])
    if workup_items:
        try:
            from workup_engine import format_workup_for_user
            workup_text = format_workup_for_user(workup_items)
        except ImportError:
            workup_text = "\n".join(
                f"- {it.get('test', '?')} — {it.get('rationale', '')}"
                for it in workup_items
            )
        if workup_text:
            parts.append("\n**If you see a doctor, these may be relevant:**")
            parts.append(workup_text)
    else:
        # Fallback: old template-based tests (for backward compatibility
        # when pipeline didn't have workup_engine available)
        smart_tests = _conditional_tests(symptoms, etiology, triage)
        if smart_tests:
            parts.append("\n**If you see a doctor, these may be relevant:**")
            for test in smart_tests[:4]:
                parts.append(f"- {test}")

    # ── DISCLAIMER ──
    parts.append(
        "\n---\n*This assessment is generated by Osler's reasoning engine "
        "and is for informational purposes only. It is not a medical diagnosis. "
        "Please consult a healthcare provider for proper evaluation and treatment.*"
    )

    return "\n\n".join(parts)


# ═══════════════════════════════════════════════
# SMART FOLLOW-UP QUESTIONS
# ═══════════════════════════════════════════════

def _rank_questions_for_case(symptoms: list, raw_questions: list, predicted: list) -> list:
    """
    Rank follow-up questions balanced across hypothesis buckets.

    For ambiguous symptom sets (e.g. dizziness + nausea), questions should
    sample from ALL plausible hypothesis directions — not just the top-ranked
    etiology from the classifier. Each bucket contributes at most 1-2 questions
    so no single hypothesis dominates.

    Buckets (in priority order):
      1. Symptom clarification  — character/location of what's already reported
      2. Dehydration/intake     — most actionable for GI/dizzy presentations
      3. Vestibular/positional  — key differentiator for dizziness
      4. Systemic/infectious    — fever, recent illness (1 question max)
      5. Medication/trigger     — new meds, anxiety, stress
      6. Red flag screen        — safety net
    """
    sym_set = set(s.lower().replace("_", " ") for s in symptoms)
    buckets = {}   # bucket_name → [question, ...]

    # ── Bucket 0: Neurologic localization (highest priority) ──
    # Numbness/weakness/tingling need localization before anything else.
    # Unilateral presentation = potential stroke → must ask first.
    neuro_screen = []
    has_numbness  = any("numbness" in s or "numb" in s for s in sym_set)
    has_weakness  = any("weakness" in s or "weak" in s for s in sym_set)
    has_tingling  = any("tingling" in s for s in sym_set)
    has_neuro_sym = has_numbness or has_weakness or has_tingling

    if has_numbness:
        neuro_screen.append("Is the numbness on one side of your body (face, arm, or leg), or is it on both sides?")
        neuro_screen.append("Did the numbness start suddenly, or come on gradually over minutes/hours?")
        neuro_screen.append("Do you also have any weakness, difficulty speaking, or changes in your vision?")
    elif has_weakness:
        neuro_screen.append("Is the weakness on one side of your body, or both sides?")
        neuro_screen.append("Did it start suddenly or gradually?")
    elif has_tingling:
        neuro_screen.append("Is the tingling on one side of your body or both sides, and where exactly?")
    buckets["neuro_screen"] = neuro_screen[:1]  # single highest-value question

    # ── Bucket 0.5: Skin / redness clarification (always fires if present) ──
    # Redness and rash localization are high clinical value — they must not be
    # bumped by dizziness or dehydration questions. Separate bucket ensures they appear.
    skin_clarify = []
    if any("rash" in s for s in sym_set):
        skin_clarify.append("Where is the rash? Is it spreading, raised, or itchy?")
    if any("redness" in s for s in sym_set):
        skin_clarify.append("Where exactly is the redness — skin, eyes, one spot, or widespread?")
        if any("rash" not in s for s in sym_set):  # redness without rash → also ask itch/spread
            skin_clarify.append("Is the redness itchy, painful, or spreading?")
    buckets["skin_clarify"] = skin_clarify[:1]

    # ── Bucket 1: Other symptom clarification ──
    clarify = []
    if any("dizziness" in s for s in sym_set) and not skin_clarify:
        # Only ask dizziness clarification when skin isn't the priority
        clarify.append("Is the dizziness more like lightheadedness/faintness, or a spinning sensation?")
    if any("swelling" in s for s in sym_set):
        clarify.append("Where is the swelling? (one leg, both legs, face, or all over)")
    if any("joint" in s for s in sym_set):
        clarify.append("Is the joint pain in one joint or multiple joints, and which ones?")
    if any("pain" in s for s in sym_set):
        if not any(x in s for s in sym_set for x in ("abdominal", "chest", "head", "joint")):
            clarify.append("Where exactly is the pain?")
    buckets["clarify"] = clarify[:1]

    # ── Bucket 2: Dehydration / intake ──
    # Only ask about fluids/intake when GI symptoms are present.
    # Dizziness alone does NOT imply dehydration — especially when numbness is also present.
    dehydration = []
    has_gi = any(s in sym_set for s in ("nausea", "vomiting", "diarrhea", "abdominal pain"))
    has_dizzy = any("dizziness" in s for s in sym_set)
    # Gate: require GI symptom OR dizziness WITHOUT neurologic symptoms
    if has_gi or (has_dizzy and not has_neuro_sym):
        # Neutral question — don't presuppose vomiting/diarrhea the patient didn't report
        has_vomiting = any("vomiting" in s for s in sym_set)
        has_diarrhea = any("diarrhea" in s for s in sym_set)
        if has_vomiting or has_diarrhea:
            dehydration.append("Have you been able to keep fluids down? Are you urinating normally?")
        else:
            # Nausea/dizziness only — ask about intake without implying vomiting
            dehydration.append("Have you been able to eat and drink normally, or has the nausea made that difficult?")
        dehydration.append("Are you urinating less than usual, or feeling very thirsty or dry-mouthed?")
    buckets["dehydration"] = dehydration[:1]

    # ── Bucket 3: Vestibular / positional ──
    vestibular = []
    if has_dizzy:
        vestibular.append("Is the dizziness worse when you stand up quickly, turn your head, or change positions?")
        vestibular.append("Have you had any recent ear pain, hearing changes, or a feeling of fullness in your ears?")
    buckets["vestibular"] = vestibular[:1]

    # ── Bucket 4: Systemic / infectious (max 1) ──
    # Only ask about symptoms the patient hasn't already reported.
    systemic = []
    has_fever    = any("fever" in s for s in sym_set)
    has_chills   = any("chills" in s for s in sym_set)
    has_prodrome = any(s in sym_set for s in ("fever", "chills", "body aches", "fatigue", "malaise"))
    if not has_prodrome:
        # Ask about infection signs only if none already present
        systemic.append("Have you had any fever, chills, or felt like you might be coming down with something?")
    elif has_chills and not has_fever:
        # Chills present but fever not confirmed — ask about measured temp
        systemic.append("Have you measured your temperature? Do you have an actual fever?")
    elif has_fever and not has_chills:
        systemic.append("Have you had any chills or shaking along with the fever?")
    if any(s in sym_set for s in ("nausea", "vomiting", "diarrhea")):
        systemic.append("Have you eaten anything unusual, or has anyone around you had similar symptoms?")
    if any(s in sym_set for s in ("joint pain", "rash")):
        systemic.append("Have you had any recent illness or infection in the past few weeks?")
    buckets["systemic"] = systemic[:1]

    # ── Bucket 5: Medication / trigger ──
    medication = []
    medication.append("Have you started any new medications, supplements, or changed your dosage recently?")
    if has_dizzy or has_gi:
        medication.append("Have you been under significant stress recently, or noticed any anxiety?")
    buckets["medication"] = medication[:1]

    # ── Bucket 6: Red flag screen ──
    red_flags = []
    if any("diarrhea" in s for s in sym_set):
        red_flags.append("Is there any blood in your stool?")
    if any("vomit" in s for s in sym_set):
        red_flags.append("Is there any blood in what you're vomiting?")
    buckets["red_flags"] = red_flags[:1]

    # Assemble: one from each bucket in priority order, up to 5 total
    questions = []
    seen = set()
    # When rash is present, medication/exposure question is highest clinical value
    # (need to distinguish infectious from drug/immune reaction)
    _has_rash = any("rash" in s or "redness" in s or "hives" in s
                     or "itching" in s for s in sym_set)  # redness = rash equivalent for priority
    _bucket_order = (
        # Skin symptoms: skin_clarify first, then medication (allergic vs infectious distinction)
        ["skin_clarify", "neuro_screen", "medication", "clarify", "systemic", "dehydration", "vestibular", "red_flags"]
        if _has_rash else
        ["skin_clarify", "neuro_screen", "clarify", "dehydration", "vestibular", "systemic", "medication", "red_flags"]
    )
    for bucket in _bucket_order:
        for q in buckets.get(bucket, []):
            k = q.lower().strip()
            if k not in seen and len(questions) < 5:
                questions.append(q)
                seen.add(k)

    return questions


def _filter_important_predictions(predicted: list, sym_set: set) -> list:
    """
    Return watch-for items that are:
      1. Clinically significant (not trivial)
      2. Actually relevant to the current symptom set
      3. Not already present (patient would notice)

    Avoids the generic "bleeding, chest pain, back pain" template by
    gating each watch-for item on whether the current case makes it
    plausible.
    """
    # Truly universal red flags — shown for any symptom set
    # Keep this list SHORT: these must be things a patient can monitor at home
    # without needing clinical context. Avoid alarm-level items for routine cases.
    # Load ALWAYS_WATCH and GATED_WATCH from red_flags.json (data-driven)
    import json as _drj, os as _dros
    _ALWAYS_WATCH: set = set()
    _GATED_WATCH:  dict = {}
    for _rf_path in ("nexus_engine/red_flags.json", "nexus_engine/red_flags.json"):
        if _dros.path.exists(_rf_path):
            try:
                for _rule in _drj.load(open(_rf_path)):
                    if _rule.get("bucket") != "watch_for" or not _rule.get("watch_for"):
                        continue
                    _wf   = _rule["watch_for"]
                    _syms = _rule.get("symptoms", [])
                    if not _syms or _rule.get("always"):
                        _ALWAYS_WATCH.add(_wf)
                    else:
                        _GATED_WATCH[_wf] = set(_syms)
                break
            except Exception:
                pass
    # Fallback if file not found
    if not _ALWAYS_WATCH and not _GATED_WATCH:
        _ALWAYS_WATCH = {"confusion", "fainting"}
        _GATED_WATCH  = {
            "chest pain":               {"chest pain", "palpitations", "shortness of breath"},
            "bloody or dark stool":     {"diarrhea", "abdominal pain"},
            "vomiting blood":           {"vomiting"},
            "severe abdominal pain":    {"abdominal pain", "nausea", "vomiting", "diarrhea"},
            "high fever":               {"fever", "chills", "nausea", "vomiting"},
            "inability to keep fluids down": {"nausea", "vomiting"},
            "severe dehydration":       {"nausea", "vomiting", "diarrhea", "dizziness"},
            "fainting":                 {"dizziness", "nausea", "weakness", "palpitations"},
            "jaundice":                 {"abdominal pain", "nausea"},
        }
    ALWAYS_WATCH = _ALWAYS_WATCH
    GATED_WATCH  = _GATED_WATCH

    # Symptoms that should NEVER appear as watch-for items (too generic / alarming without cause)
    NEVER_WATCH = {
        "back pain", "bleeding", "pain", "weakness", "fatigue",
        "headache",  # too common/generic to be a meaningful watch-for
    }

    result = []
    seen = set()

    def _add(item):
        k = item.lower().strip()
        if k not in seen and k not in sym_set and k not in NEVER_WATCH:
            seen.add(k)
            result.append(item)

    # Pass 1: always-watch items that aren't already present
    for w in ALWAYS_WATCH:
        if w not in sym_set:
            _add(w)

    # Pass 2: gated watch items — only if current symptoms make them plausible
    for watch_item, gate_syms in GATED_WATCH.items():
        if watch_item.lower() in sym_set:
            continue  # already present
        if any(g in s for s in sym_set for g in gate_syms):
            _add(watch_item)

    # Pass 3: from NEXUS predicted symptoms (already filtered by mechanism logic)
    # Only include if they pass the never-watch gate
    # PREDICTED_IMPORTANT: gated on system relevance
    # Map each predicted symptom to the system evidence required to show it
    # PREDICTED_GATED: predicted symptoms that only show as watch-for
    # when at least one gate symptom is present in the current case.
    # Format: {watch_item_substring: {required_symptoms}}
    # Empty set = always show; non-empty = need ANY one of these in sym_set.
    PREDICTED_GATED = {
        "bloody diarrhea":     {"diarrhea", "abdominal pain"},
        "blood in stool":      {"diarrhea", "abdominal pain"},
        "severe dehydration":  {"nausea", "vomiting", "diarrhea", "dizziness"},
        "shortness of breath": {"cough", "chest pain", "wheezing"},
        "fever":               set(),
        "confusion":           set(),
        "dehydration":         {"nausea", "vomiting", "diarrhea", "dizziness"},
        "persistent vomiting": {"nausea", "vomiting"},
    }
    for p in predicted[:10]:
        p_clean = p.replace("_", " ").lower()
        if p_clean in sym_set or p_clean in seen:
            continue
        for watch_item, gate_syms in PREDICTED_GATED.items():
            if watch_item in p_clean:
                # Fixed gate logic: check if ANY gate symptom is in sym_set
                if not gate_syms or any(gs in sym_set for gs in gate_syms):
                    _add(p_clean)
                break

    return result[:4]


# ═══════════════════════════════════════════════
# CONDITIONAL TESTS
# ═══════════════════════════════════════════════

def _conditional_tests(symptoms: list, etiology: dict, triage: dict) -> list:
    """Generate condition-specific test recommendations, not template."""
    sym_set = set(s.lower().replace("_", " ") for s in symptoms)
    tests = []
    level = triage.get("level", "routine").lower()

    # Basic blood work only if there's a reason
    has_infection_signs = any(s in sym_set for s in ("fever", "chills", "night sweats"))
    has_diarrhea = any(s in sym_set for s in ("diarrhea", "bloody stool"))
    has_gi_general = any(s in sym_set for s in ("vomiting", "abdominal pain"))
    has_nausea_only = "nausea" in sym_set and not has_diarrhea and not has_gi_general
    has_respiratory = any(s in sym_set for s in ("cough", "shortness of breath", "wheezing"))
    has_cardiac = any(s in sym_set for s in ("chest pain", "palpitations"))
    has_joint = any(s in sym_set for s in ("joint pain",))
    has_swelling = any(s in sym_set for s in ("swelling",))
    has_redness = any(s in sym_set for s in ("redness", "rash"))
    has_dizziness = any(s in sym_set for s in ("dizziness",))

    if has_infection_signs or level in ("urgent", "emergency"):
        tests.append("Blood count (CBC) with differential — to assess for infection or inflammation")
    if has_infection_signs:
        tests.append("C-reactive protein (CRP) — to gauge inflammation severity")

    if has_diarrhea:
        tests.append("Stool studies — if diarrhea persists >3 days, is bloody, or accompanied by fever")
    if (has_diarrhea or has_gi_general) and has_infection_signs:
        tests.append("Electrolytes and kidney function — if dehydration is a concern")

    if has_respiratory:
        tests.append("Chest X-ray and pulse oximetry — if breathing difficulty is significant")
    if has_cardiac:
        tests.append("ECG — to assess heart rhythm")
        if any(s in sym_set for s in ("palpitations",)):
            tests.append("Electrolytes, thyroid function — if palpitations are persistent")

    if has_joint and has_swelling:
        tests.append("Inflammatory markers (CRP, ESR) — if joint symptoms persist")
        tests.append("Consider rheumatologic workup if joint pain involves multiple joints")

    if has_swelling and not has_joint:
        tests.append("Kidney function and urinalysis — if swelling is generalized")

    if has_redness and not has_diarrhea and not has_respiratory:
        tests.append("Examination of the area of redness — to determine if rash, irritation, or infection")
    if has_dizziness:
        tests.append("Vitals and hydration assessment — if dizziness persists or worsens")

    # Only suggest PCT if there's real infection concern + it's urgent
    et_type = etiology.get("etiology", "uncertain") if etiology else "uncertain"
    if et_type == "bacterial" or (has_infection_signs and level in ("urgent", "emergency")):
        tests.append("Procalcitonin (PCT) — if bacterial infection is suspected and needs confirmation")

    # Fallback if nothing triggered
    if not tests:
        tests.append("Basic blood work (CBC, metabolic panel) if symptoms persist beyond a few days")

    return tests


# ═══════════════════════════════════════════════
# CLINICAL MODE
# ═══════════════════════════════════════════════

def _clinical_response(result: dict, user_input: str = "") -> str:
    parts = []
    symptoms = result.get("symptoms", result.get("final_symptoms", []))
    diagnoses = result.get("nexus_diagnoses", result.get("top_diseases", []))
    # 3-bucket red flag reads
    _rfd           = result.get("red_flag_block") or {}
    red_flags      = _rfd.get("observed_red_flags", [])   # A: triage-relevant
    risk_escalators= _rfd.get("risk_escalators", [])      # B: caution wording
    watch_for_flags= _rfd.get("watch_for_flags", [])      # C: UI "seek care if..."
    # Legacy fallback
    if not red_flags and not risk_escalators:
        red_flags = result.get("red_flags", result.get("nexus_red_flags", []))
    triage = result.get("triage", {})
    systems = result.get("nexus_detected_systems", [])
    etiology = result.get("nexus_etiology", {})
    physiology = result.get("nexus_physiology", {})
    recommended_tests = result.get("nexus_recommended_tests", [])

    parts.append(f"### Clinical Assessment: {_join_natural(symptoms)}")
    parts.append(f"**Triage:** {triage.get('level', 'routine').upper()}")

    if systems:
        parts.append(f"**Systems:** {', '.join(systems)}")

    # NOTE: etiology (viral/bacterial/non-infectious) display removed.
    # etiology_classifier deleted per pure-mechanism design.

    if red_flags:
        parts.append("**Red flags:**")
        for rf in red_flags[:5]:
            parts.append(f"- {rf.get('red_flag', rf.get('trigger', ''))}")

    if diagnoses:
        parts.append("\n**Differential diagnosis (ranked):**")
        for i, dx in enumerate(diagnoses[:5], 1):
            name = dx.get("disease", dx.get("disease_name", "?"))
            score = dx.get("score", 0)
            matched = dx.get("matched_symptoms", [])
            missing = dx.get("missing_symptoms", [])
            evidence = dx.get("evidence", [])
            line = f"{i}. **{name}** (score: {score})"
            if matched: line += f"\n   - Matched: {', '.join(matched)}"
            if missing: line += f"\n   - Unexplained: {', '.join(missing)}"
            if evidence: line += f"\n   - Evidence: {evidence[0]}"
            parts.append(line)

    # Conditional tests — prefer justified workup when available
    workup_items = result.get("recommended_workup", [])
    if workup_items:
        parts.append("\n**Recommended investigations (justified):**")
        for it in workup_items:
            justifications = ", ".join(it.get("justified_by", []))
            parts.append(f"- [{it.get('priority', '?')}] {it.get('test', '?')} — {it.get('rationale', '')}  *(← {justifications})*")
    else:
        smart_tests = _conditional_tests(symptoms, etiology, triage)
        if smart_tests:
            parts.append("\n**Recommended investigations (conditional):**")
            for test in smart_tests:
                parts.append(f"- {test}")

    # ── REASONING SUMMARY ───────────────────────────────────────────────────────
    _symptoms_disp  = sorted(sym_set)[:6]
    _detected_sys   = ea.get("detected_systems", systems or [])[:2]
    _obs_rfs        = red_flags  # already A-bucket
    _esc_count      = len(risk_escalators)
    _is_frozen      = result.get("_high_risk_freeze", False)
    _nc = result.get("nexus_consistency") or 0
    _consistency = _nc.get("consistency_score", 0.0) if isinstance(_nc, dict) else float(_nc or 0)
    _mechs          = int((result.get("nexus_stats") or {}).get("mechanisms_activated", 0))
    _et_type        = (etiology or {}).get("etiology", "uncertain") if etiology else "uncertain"
    _triage_level   = triage.get("level", "MODERATE").upper()

    # Evidence strength
    if _is_frozen:
        _ev_str = "bypassed — high-risk mode active"
    elif _consistency >= 0.65 and _mechs > 20:
        _ev_str = "moderate–strong"
    elif _consistency >= 0.45 or _mechs > 10:
        _ev_str = "moderate (limited by symptom count)"
    else:
        _ev_str = "limited — few symptoms provided"

    # Uncertainty
    if _is_frozen:
        _unc_str = "high — safety mode active, evaluation required"
    elif _output_state == "label":
        _unc_str = "low–moderate"
    elif _output_state == "syndrome":
        _unc_str = "moderate — pattern identified, specific cause unclear"
    else:
        _unc_str = "high — insufficient data for pattern recognition"

    # Red flag status
    if _obs_rfs:
        _rf_str = f"⚠️ observed: {', '.join(r.get('name','').replace('_',' ') for r in _obs_rfs[:2])}"
    elif _esc_count:
        _rf_str = f"{_esc_count} risk escalator(s) — monitor closely"
    else:
        _rf_str = "no emergency red flags detected from current input"

    # Cause direction
    if not _etiology_allowed or _et_type == "uncertain":
        _cause_str = "undetermined — insufficient discriminating evidence"
    elif _etiology_level in ("weak_lean",):
        _cause_str = f"weak lean toward {_et_type}"
    else:
        _cause_str = f"{_et_type} ({_etiology_level})"

    # Suggested path
    if _triage_level in ("EMERGENCY", "URGENT", "PROMPT"):
        _path_str = "seek care promptly — do not delay"
    elif _triage_level == "MODERATE":
        _path_str = "monitor + self-care; escalate if symptoms worsen"
    else:
        _path_str = "self-care; follow up if no improvement in 48–72 hours"

    _syn_display = _syndrome_label if _syndrome_label else "undifferentiated"

    parts.append(
        "\n---\n"
        "**Reasoning Summary**\n"
        f"- **Symptoms detected:** {', '.join(_symptoms_disp) or 'none reported'}\n"
        f"- **Main system:** {', '.join(_detected_sys) or 'undetermined'}\n"
        f"- **Pattern:** {_syn_display}\n"
        f"- **Red flag status:** {_rf_str}\n"
        f"- **Cause direction:** {_cause_str}\n"
        f"- **Evidence strength:** {_ev_str}\n"
        f"- **Uncertainty:** {_unc_str}\n"
        f"- **Suggested path:** {_path_str}"
    )

    return "\n\n".join(parts)


# ═══════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════

def _join_natural(items: list) -> str:
    items = [str(i).replace("_", " ") for i in items if i]
    if not items: return ""
    if len(items) == 1: return items[0]
    if len(items) == 2: return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _dedup_merged(*lists) -> list:
    """Merge any number of question lists, deduplicating by content.
    Lists given first take priority. Duplicates detected by:
      - exact lowercase match
      - keyword overlap after stripping punctuation (1 match for short questions,
        2 matches for longer ones)
    """
    import re as _re

    def _norm(q):
        return _re.sub(r"[*][*](.+?)[*][*]", r"\1", q).lower().strip()

    def _keywords(text: str) -> set:
        # Strip trailing punctuation from each word before comparing
        words = _re.sub(r"[^\w\s]", " ", text).split()
        return {w for w in words if len(w) > 3}

    result = []
    seen_norms = set()   # exact normalised strings
    seen_kw_sets = []    # keyword sets for overlap checking

    for lst in lists:
        for q in (lst or []):
            q_clean = q.strip() if isinstance(q, str) else str(q)
            q_norm  = _norm(q_clean)
            if q_norm in seen_norms:
                continue
            q_kw = _keywords(q_norm)
            # Threshold: 1 keyword overlap for short questions, 2 for longer ones
            _threshold = 1 if len(q_kw) <= 3 else 2
            duplicate = any(
                len(q_kw & ex_kw) >= _threshold
                for ex_kw in seen_kw_sets
            )
            if not duplicate:
                result.append(q_clean)
                seen_norms.add(q_norm)
                seen_kw_sets.append(q_kw)

    return result

# ═══════════════════════════════════════════════════════════════
# PROOF MODE — Cascade chain visualization
# ═══════════════════════════════════════════════════════════════
# Reads pre-computed data from `result` produced by:
#   - NexusMedical.reason()       (diagnoses, _mechanism, matched_symptoms, thinking)
#   - evidence_gate.assess_evidence() (output_state, disease_label_allowed, gate_details)
#   - AnatomyBridge.enhance_with_anatomy() (anatomy_affected_organs, anatomy_context)
#
# This is a FORMATTER ONLY. It does NOT re-run any reasoning. If the data isn't
# in `result`, the corresponding proof section is omitted gracefully.


def _proof_block(result: dict) -> str:
    """Render the 6-section cascade proof block."""
    lines = []
    lines.append("═══════════════════════════════════════════════════════════")
    lines.append("           CASCADE PROOF — How this conclusion was derived")
    lines.append("═══════════════════════════════════════════════════════════")

    diagnoses = (result.get("nexus_diagnoses")
                 or result.get("diagnoses")
                 or [])
    ea = result.get("evidence_assessment") or {}
    symptoms = result.get("symptoms", result.get("final_symptoms", []))

    if not diagnoses:
        lines.append("\nNo diagnoses produced — reasoning did not converge on any cascade.")
        return "\n".join(lines)

    # Render proof for top diagnosis (and up to 2 differentials)
    for i, dx in enumerate(diagnoses[:3]):
        if i == 0:
            lines.append("\n┌─ PRIMARY ─────────────────────────────────────────────────")
        else:
            lines.append(f"\n┌─ DIFFERENTIAL #{i+1} ────────────────────────────────────")
        lines.extend(_proof_for_diagnosis(dx, symptoms, result, ea, is_primary=(i == 0)))
        lines.append("└─────────────────────────────────────────────────────────")

    return "\n".join(lines)


def _classify_proof_strength(dx: dict, result: dict, ea: dict,
                              is_primary: bool) -> dict:
    """
    Two-layer proof classification.

    LAYER 1 — cascade_support: how strong is the underlying mechanism evidence?
       Dimensions: cascade_match (D1), anatomy (D4), physiology (D5)
       This answers: "Is the cascade reasoning itself credible?"

    LAYER 2 — final_label_proof: should we announce this as a disease label?
       Dimensions: evidence_gate (D2), differential_clarity (D3)
       This answers: "Is it safe to publish the disease name to the user?"

    Why two layers (not one combined score)?
       — A high-risk freeze case (chest pain → MI) can have STRONG cascade
         support but the final label must be WEAK (we won't say "Heart Attack"
         without lab/EKG confirmation).
       — A single combined score would hide this distinction and mislead users.

    Returns:
      {
        "level":            overall level (worst of the two layers)
        "cascade_support":  {"level", "score", "max_score", "reasons", "dims"}
        "final_label_proof":{"level", "score", "max_score", "reasons", "dims"}
        "weakest":          string explaining which layer/dim limits the overall
      }

    Calibrations applied (per latest design decisions):
      A. cascade_match thresholds RELAXED — verbose cascade strings produce
         match scores in 0.5-0.7 range; we don't want to mis-grade clean cases
         just because the cascade strings are wordy.
      B. stable physiology for outpatient disease is SUPPORTIVE (level 2), not
         capped at moderate. Outpatient disease ≈ no triage_level==critical AND
         no high-risk freeze AND label was allowed.
    """
    disease_name = dx.get("disease", "?")
    matched = dx.get("matched_symptoms", []) or []
    score   = dx.get("score", 0)
    details = (ea or {}).get("disease_gate_details") or {}
    label_ok    = (ea or {}).get("disease_label_allowed", False)
    output_state = (ea or {}).get("output_state", "")
    frozen_by   = details.get("frozen_by", "")
    supplemental = dx.get("supplemental") or {}

    # ════════════════════════════════════════════════════════════════════════
    # LAYER 1 — cascade_support (mechanism evidence)
    # ════════════════════════════════════════════════════════════════════════
    L1_dims = {}

    # D1. cascade_match — Calibration A: relaxed thresholds
    # The cascade strings are often verbose ("shortness of breath dyspnea")
    # producing fuzzy match scores in 0.5-0.7 range even for genuinely matching
    # user symptoms. Original strict threshold (need 0.8) penalized clean cases.
    # New: count matches with score >= 0.5 as "good enough"; require at least
    # one strong (>= 0.7) to be "strong overall".
    good_matches = sum(1 for m in matched
                       if isinstance(m, (list, tuple)) and len(m) >= 3 and m[2] >= 0.5)
    strong_matches = sum(1 for m in matched
                         if isinstance(m, (list, tuple)) and len(m) >= 3 and m[2] >= 0.7)
    total_matches = len(matched)
    if good_matches >= 3 and strong_matches >= 1:
        L1_dims["cascade_match"] = (2, f"{good_matches} good matches ({strong_matches} strong)")
    elif good_matches >= 2:
        L1_dims["cascade_match"] = (1, f"{good_matches} good matches")
    elif total_matches >= 1:
        L1_dims["cascade_match"] = (0, f"only {total_matches} match(es), all weak similarity")
    else:
        L1_dims["cascade_match"] = (0, "no symptom matches recorded")

    # D4. anatomical coherence
    spatial = result.get("spatial_state") or {}
    adjustments = spatial.get("applied_adjustments", []) or []
    dx_adj = next((a for a in adjustments if a.get("disease") == disease_name), None)
    contradictions = spatial.get("contradictions", []) or []
    sys_match = details.get("dis_system_match", None)
    if disease_name in contradictions:
        L1_dims["anatomy"] = (0, "3D atlas CONTRADICTS this diagnosis")
    elif dx_adj and dx_adj.get("delta", 0) > 0 and sys_match:
        L1_dims["anatomy"] = (2, f"spatial supports (Δ={dx_adj['delta']:+.2f}) + system match")
    elif (dx_adj and dx_adj.get("delta", 0) > 0) or sys_match:
        L1_dims["anatomy"] = (1, "spatial OR system match (not both)")
    else:
        L1_dims["anatomy"] = (1, "neutral — no anatomical support or contradiction")

    # D5. physiology coherence — Calibration B (refined):
    # Stable vitals are SUPPORTIVE for outpatient/urgent disease.
    # Only treat stable vitals as CONTRADICTING when the disease's intrinsic
    # severity is "critical/emergent" (e.g. Sepsis, DKA, Anaphylaxis, Cardiac
    # Arrest) — diseases where stable vitals would be medically impossible.
    #
    # IMPORTANT: do NOT use `frozen_by` here. Freezing means "we won't publish
    # a label without lab confirmation"; it does not mean "the patient must
    # be physiologically critical right now." A patient walking in with
    # chest pain (frozen) can have totally stable vitals — early MI presents
    # this way constantly.
    phys = result.get("physiology_state") or {}
    intrinsic_critical = (supplemental.get("triage_level", "") or "").lower() \
                         in ("critical", "emergent")
    if phys.get("critical"):
        L1_dims["physiology"] = (2, "critical vitals — consistent with severe cascade")
    elif phys.get("failed_organs"):
        L1_dims["physiology"] = (2, f"organ failure simulated: {phys['failed_organs']}")
    elif phys and intrinsic_critical:
        # Disease is intrinsically critical but vitals are stable → contradicts
        L1_dims["physiology"] = (0, "stable vitals contradict expected critical state")
    elif phys:
        # Calibration B: stable vitals for outpatient/urgent disease = SUPPORTIVE
        L1_dims["physiology"] = (2, "stable vitals — consistent with outpatient disease")
    else:
        L1_dims["physiology"] = (1, "physiology not evaluated")

    # ════════════════════════════════════════════════════════════════════════
    # LAYER 2 — final_label_proof (publishability of the disease label)
    # ════════════════════════════════════════════════════════════════════════
    L2_dims = {}

    # D2. evidence_gate
    contract = details.get("contract_gate", False)
    sgate    = details.get("score_gate", False)
    if label_ok and contract and sgate:
        L2_dims["evidence_gate"] = (2, "both contract_gate and score_gate passed")
    elif label_ok and (contract or sgate):
        which = "contract_gate" if contract else "score_gate"
        L2_dims["evidence_gate"] = (1, f"only {which} passed")
    elif frozen_by:
        # Hard rule #4 from spec: high-risk freeze can NEVER be "strong" final label.
        L2_dims["evidence_gate"] = (0, f"high-risk freeze by '{frozen_by}' — label REFUSED")
    elif output_state == "syndrome":
        L2_dims["evidence_gate"] = (0, "label refused; reported as syndrome")
    elif output_state == "triage_only":
        L2_dims["evidence_gate"] = (0, "label refused; only triage info given")
    else:
        L2_dims["evidence_gate"] = (0, "label not allowed")

    # D3. differential_clarity (margin)
    diagnoses = result.get("nexus_diagnoses") or result.get("diagnoses") or []
    margin = 0
    if len(diagnoses) >= 2:
        margin = diagnoses[0].get("score", 0) - diagnoses[1].get("score", 0)
    elif len(diagnoses) == 1:
        margin = diagnoses[0].get("score", 0)
    if not is_primary:
        L2_dims["differential_clarity"] = (1, "differential candidate (not primary)")
    elif margin >= 0.30:
        L2_dims["differential_clarity"] = (2, f"margin {margin:.2f} — clear winner")
    elif margin >= 0.15:
        L2_dims["differential_clarity"] = (1, f"margin {margin:.2f} — modest separation")
    else:
        L2_dims["differential_clarity"] = (0, f"margin {margin:.2f} — close differentials")

    # ════════════════════════════════════════════════════════════════════════
    # AGGREGATION per layer
    # ════════════════════════════════════════════════════════════════════════
    levels = ["weak", "moderate", "strong"]

    def _aggregate(layer_dims: dict, layer_name: str) -> dict:
        scores = [s for s, _ in layer_dims.values()]
        total = sum(scores)
        max_total = 2 * len(layer_dims)
        weakest_s = min(scores)
        weakest_d = next(name for name, (s, _) in layer_dims.items() if s == weakest_s)

        # Bin: hi-3rd → strong, mid-3rd → moderate, low-3rd → weak
        if total >= max_total - 1:    # 9/10 or higher of 6 → 5+ of 6
            binned = "strong"
        elif total >= max_total // 2:  # 6/10 or 3/6
            binned = "moderate"
        else:
            binned = "weak"

        # Safety guard: weak dim caps overall at weak; moderate-only caps at moderate
        if weakest_s == 0:
            level = "weak"
        elif weakest_s == 1 and binned == "strong":
            level = "moderate"
        else:
            level = binned

        return {
            "level":     level,
            "score":     total,
            "max_score": max_total,
            "reasons":   [f"{name}: {levels[s]} — {why}" for name, (s, why) in layer_dims.items()],
            "dims":      {name: levels[s] for name, (s, _) in layer_dims.items()},
            "weakest":   f"{layer_name}.{weakest_d}",
        }

    L1 = _aggregate(L1_dims, "cascade_support")
    L2 = _aggregate(L2_dims, "final_label_proof")

    # ════════════════════════════════════════════════════════════════════════
    # Overall level = worst of the two layers (medically conservative).
    # Hard rule #4: high-risk freeze cannot show "strong" final label.
    # This is already enforced by L2_dims["evidence_gate"] = (0, frozen).
    # ════════════════════════════════════════════════════════════════════════
    overall_rank = min(levels.index(L1["level"]), levels.index(L2["level"]))
    overall_level = levels[overall_rank]
    if levels.index(L1["level"]) <= levels.index(L2["level"]):
        weakest = L1["weakest"]
    else:
        weakest = L2["weakest"]

    return {
        "level":              overall_level,
        "cascade_support":    L1,
        "final_label_proof":  L2,
        "weakest":            weakest,
        # Legacy fields (for backward-compat with existing renderer):
        "score":              L1["score"] + L2["score"],
        "max_score":          L1["max_score"] + L2["max_score"],
        "reasons":            L1["reasons"] + L2["reasons"],
        "dim_levels":         {**L1["dims"], **L2["dims"]},
    }


def _classify_user_symptom_roles(user_symptoms: list, dx: dict, ea: dict) -> list:
    """
    For each user symptom, return a dict of flags showing what role it played
    in producing this diagnosis.

    Returns a list of dicts (same order as user_symptoms), each:
        {
          "symptom":              <user-provided string>,
          "derived_from_cascade": bool — symptom appears in the cascade for this disease
          "matched_by_user":      bool — always True (it IS the user's symptom)
          "used_for_score":       bool — contributed to cascade-match score (>= 0.5 match)
          "used_for_gate":        bool — counted as anchor or support by evidence_gate
          "match_score":          float — actual cascade match score (0.0 if no match)
          "cascade_string":       str — the cascade-side string it matched to (if any)
          "high_risk_freeze":     bool — this symptom triggered high-risk freeze
        }

    Pure formatter — reads existing fields. No recomputation.
    """
    matched = dx.get("matched_symptoms", []) or []
    details = (ea or {}).get("disease_gate_details") or {}
    frozen_by = (details.get("frozen_by") or "").lower().strip()
    anchor_syms = {s.lower().strip() for s in (details.get("anchor_syms") or [])}
    support_syms = {s.lower().strip() for s in (details.get("support_syms") or [])}
    gate_relevant = anchor_syms | support_syms

    # Build a lookup: user_symptom_lower → (cascade_string, match_score)
    # matched_symptoms entries are tuples: (user_sym, cascade_sym, match_score)
    match_lookup = {}
    for m in matched:
        if isinstance(m, (list, tuple)) and len(m) >= 3:
            user_s = str(m[0]).lower().strip()
            match_lookup[user_s] = (str(m[1]), float(m[2]))
        elif isinstance(m, (list, tuple)) and len(m) >= 2:
            user_s = str(m[0]).lower().strip()
            match_lookup[user_s] = (str(m[1]), 1.0)

    roles = []
    for s in (user_symptoms or []):
        s_norm = str(s).lower().strip().replace("_", " ")
        cascade_str, match_score = match_lookup.get(s_norm, ("", 0.0))
        derived_from_cascade = bool(cascade_str)
        # used_for_score: the cascade matcher counted this in its scoring
        used_for_score = match_score >= 0.5
        # used_for_gate: evidence_gate counted it as anchor or support
        # (anchor_syms / support_syms come from primary-system anchor matching,
        # which is independent from cascade match)
        used_for_gate = s_norm in gate_relevant
        high_risk_freeze = (s_norm == frozen_by) if frozen_by else False

        roles.append({
            "symptom":              s,
            "derived_from_cascade": derived_from_cascade,
            "matched_by_user":      True,
            "used_for_score":       used_for_score,
            "used_for_gate":        used_for_gate,
            "match_score":          round(match_score, 2),
            "cascade_string":       cascade_str,
            "high_risk_freeze":     high_risk_freeze,
        })

    return roles


def _proof_for_diagnosis(dx: dict, user_symptoms: list, result: dict,
                         ea: dict, is_primary: bool) -> list:
    """Render proof sections 1-8 for one diagnosis. Returns list of lines."""
    lines = []
    disease_name = dx.get("disease", "?")
    score = dx.get("score", 0)
    mechanism = dx.get("_mechanism", "")   # "bladder / infection_cystitis"
    matched_syms = dx.get("matched_symptoms", [])
    missing_syms = dx.get("missing_symptoms", [])
    supplemental = dx.get("supplemental") or {}

    # Compute proof strength (read-only — uses pre-computed result fields)
    strength = _classify_proof_strength(dx, result, ea, is_primary)
    _LVL_BADGE = {"strong": "[STRONG]", "moderate": "[MODERATE]", "weak": "[WEAK]"}
    cs_badge = _LVL_BADGE.get(strength["cascade_support"]["level"], "[?]")
    fl_badge = _LVL_BADGE.get(strength["final_label_proof"]["level"], "[?]")

    lines.append(f"│ Diagnosis:   {disease_name}")
    lines.append(f"│ Score:       {score}    (cascade-match strength)")
    lines.append(f"│ Cascade:     {cs_badge}   (mechanism evidence)")
    lines.append(f"│ Label proof: {fl_badge}   (publishability)")

    # ── Section 1: Cascade source (organ + failure mode + additional sites) ──
    lines.append(f"│")
    if dx.get("_state_modeled"):
        lines.append(f"│ 1. CASCADE SOURCE  [⚡ STATE MODEL — symptoms generated from "
                     f"physiology state, not cascade lookup]")
    else:
        lines.append(f"│ 1. CASCADE SOURCE")
    if mechanism and "/" in mechanism:
        organ, fmode = [p.strip() for p in mechanism.split("/", 1)]
        lines.append(f"│    organ:        {organ}")
        lines.append(f"│    failure_mode: {fmode}")
    elif mechanism:
        lines.append(f"│    mechanism:    {mechanism}")
    else:
        lines.append(f"│    (no mechanism field — cascade source unknown)")

    # Show additional_sites if the disease maps to multiple anatomical sites
    # (e.g. Gout affects mtp + ankle + knee). This data lives in the registry
    # and is enriched into anatomy_affected_organs by anatomy_bridge.
    # Read directly from spatial_state if present.
    spatial_state = result.get("spatial_state") or {}
    primary_organs_list = spatial_state.get("primary_organs", []) or []
    # Find this disease's specific primary list
    adjustments = spatial_state.get("applied_adjustments", []) or []
    dx_adj = next((a for a in adjustments
                   if a.get("disease") == disease_name), None)
    dx_organs = (dx_adj or {}).get("organs", []) or []
    # The cascade-source organ (Section 1's "organ:" line) is the primary;
    # the rest are additional anatomical sites.
    if mechanism and "/" in mechanism and dx_organs:
        primary_organ = mechanism.split("/", 1)[0].strip()
        extras = [o for o in dx_organs if o != primary_organ]
        if extras:
            lines.append(f"│    also affects: {', '.join(extras[:5])}")
            if len(extras) > 5:
                lines.append(f"│                  … and {len(extras) - 5} more sites")

    # ── Section 2: Derived symptoms (cascade OR state model) ────────────────
    # For state-modeled diseases, the symptoms come from physiology state
    # simulation (perturbation → state variables → derivation rules).
    # For everything else, fall back to cascade output (matched + missing).
    lines.append(f"│")
    state_modeled = dx.get("_state_modeled", False)
    state_sim = dx.get("_state_simulation") or {}

    if state_modeled and state_sim:
        lines.append(f"│ 2. DERIVED SYMPTOMS  (from state model — physiology simulation)")
        lines.append(f"│    [state model] derived via perturbation → body state → rules")
        derivations = state_sim.get("derivations", [])
        if derivations:
            for d in derivations[:10]:
                sym = d.get("symptom", "?")
                why = d.get("rationale", "")
                lines.append(f"│    • {sym}")
                if why:
                    lines.append(f"│        ({why})")
        else:
            for s in state_sim.get("derived_symptoms", [])[:10]:
                lines.append(f"│    • {s}")
        # Also show the active body state that drove these
        active = state_sim.get("active_state", [])
        if active:
            lines.append(f"│    body state variables (active):")
            for v in active[:8]:
                lines.append(f"│      {v['organ']}.{v['variable']:30s} {v['value']:.2f}")
    else:
        lines.append(f"│ 2. DERIVED SYMPTOMS  (full cascade output for this disease)")
        all_cascade_syms = []
        for entry in matched_syms:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                all_cascade_syms.append(entry[1])
        for s in missing_syms:
            if isinstance(s, str):
                all_cascade_syms.append(s)
        if all_cascade_syms:
            for s in all_cascade_syms[:10]:
                lines.append(f"│    • {s}")
            if len(all_cascade_syms) > 10:
                lines.append(f"│    … and {len(all_cascade_syms) - 10} more")
        else:
            lines.append(f"│    (no cascade symptoms recorded)")

    # ── Section 3: Matched user symptoms (with per-symptom role flags) ──────
    # Q3: each user symptom is tagged with what role it played:
    #   derived_from_cascade — appears in this disease's cascade output
    #   used_for_score       — contributed to the cascade-match score
    #   used_for_gate        — counted by evidence_gate (anchor or support)
    #   high_risk_freeze     — triggered the high-risk freeze
    lines.append(f"│")
    lines.append(f"│ 3. MATCHED USER SYMPTOMS  (cascade output ∩ patient input)")
    roles = _classify_user_symptom_roles(user_symptoms, dx, ea)
    if not roles:
        lines.append(f"│    (no user symptoms recorded)")
    else:
        for r in roles:
            sym = r["symptom"]
            flags = []
            if r["derived_from_cascade"]:
                flags.append("cascade✓")
            else:
                flags.append("cascade✗")
            if r["used_for_score"]:
                flags.append("score✓")
            else:
                flags.append("score✗")
            if r["used_for_gate"]:
                flags.append("gate✓")
            else:
                flags.append("gate✗")
            if r["high_risk_freeze"]:
                flags.append("⚠ HIGH-RISK FREEZE")
            flag_str = "  ".join(flags)
            if r["cascade_string"]:
                lines.append(f"│    '{sym}'  →  '{r['cascade_string']}' "
                             f"(match={r['match_score']:.2f})")
                lines.append(f"│      flags: {flag_str}")
            else:
                lines.append(f"│    '{sym}'  →  (no cascade match for this disease)")
                lines.append(f"│      flags: {flag_str}")
        # Summary line
        n_cascade = sum(1 for r in roles if r["derived_from_cascade"])
        n_score   = sum(1 for r in roles if r["used_for_score"])
        n_gate    = sum(1 for r in roles if r["used_for_gate"])
        n_total   = len(roles)
        lines.append(f"│    ───")
        lines.append(f"│    summary: {n_cascade}/{n_total} in cascade · "
                     f"{n_score}/{n_total} drive score · "
                     f"{n_gate}/{n_total} drive gate")

    # ── Section 4: Anatomy support (KG-derived organ identification) ────────
    # Note: anatomy "spread paths" are CONTEXT (what could be affected if the
    # disease progresses), not direct diagnostic proof. The actual disease
    # localization comes from the cascade's organ (Section 1). Spread organs
    # are downstream possibilities, listed for the patient's awareness.
    affected = result.get("anatomy_affected_organs", [])
    spread = result.get("anatomy_spread", [])
    referred = result.get("anatomy_referred_pain", [])
    anatomy_conf = result.get("anatomy_confidence", None)
    lines.append(f"│")
    lines.append(f"│ 4. ANATOMY SUPPORT  (KG: symptom→organ, disease→organ)")
    if affected:
        lines.append(f"│    affected_organs: {', '.join(affected[:6])}")
        if len(affected) > 6:
            lines.append(f"│                     … and {len(affected) - 6} more")
    else:
        lines.append(f"│    affected_organs: (none — anatomy bridge did not run or KG empty)")
    if spread:
        first_target = spread[0].get('organ', '?') if isinstance(spread[0], dict) else '?'
        lines.append(f"│    spread_paths:    {len(spread)} (e.g. → {first_target})")
        lines.append(f"│                     [spread organs = context only, not diagnostic proof]")
    if referred:
        lines.append(f"│    referred_pain:   {len(referred)} source(s)")
    if anatomy_conf is not None:
        lines.append(f"│    confidence:      {anatomy_conf}")

    # ── Section 5: Spatial consistency (3D atlas reasoning) ─────────────────
    spatial = result.get("spatial_state") or {}
    lines.append(f"│")
    lines.append(f"│ 5. SPATIAL CONSISTENCY  (3D atlas — cascade-derived fingerprints)")
    primary_organs = spatial.get("primary_organs", [])
    if primary_organs:
        lines.append(f"│    primary_organs:  {', '.join(primary_organs[:5])}")
    zones = spatial.get("zones_active", [])
    if zones:
        lines.append(f"│    zones_active:    {', '.join(zones[:5])}")
    # Find this diagnosis's specific spatial adjustment
    adjustments = spatial.get("applied_adjustments", [])
    dx_adj = next((a for a in adjustments if a.get("disease") == disease_name), None)
    if dx_adj:
        delta = dx_adj.get("delta", 0)
        sign = "+" if delta > 0 else ""
        verdict = "supports" if delta > 0 else ("contradicts" if delta < 0 else "neutral")
        lines.append(f"│    score adjustment: {sign}{delta}  ({verdict} this diagnosis)")
    elif primary_organs:
        lines.append(f"│    score adjustment: 0  (no specific 3D fingerprint match)")
    contradictions = spatial.get("contradictions", [])
    if disease_name in contradictions:
        lines.append(f"│    ⚠ CONTRADICTION: 3D anatomy contradicts this diagnosis")
    if not primary_organs and not adjustments:
        lines.append(f"│    (spatial reasoning did not run)")

    # ── Section 6: Physiology state (numerical body simulation) ─────────────
    phys = result.get("physiology_state") or {}
    lines.append(f"│")
    lines.append(f"│ 6. PHYSIOLOGY STATE  (numerical body simulation)")
    if phys:
        bp = phys.get("bp", "?")
        hr = phys.get("hr", "?")
        spo2 = phys.get("spo2", "?")
        rr = phys.get("rr", "?")
        lactate = phys.get("lactate", "?")
        temp = phys.get("temp_c", "?")
        critical = phys.get("critical", False)
        failed = phys.get("failed_organs", [])
        lines.append(f"│    vitals:    BP {bp}  HR {hr}  SpO2 {spo2}  RR {rr}  T {temp}°C")
        lines.append(f"│    lactate:   {lactate} mmol/L")
        if failed:
            lines.append(f"│    failed organs: {', '.join(failed)}")
        if critical:
            lines.append(f"│    ⚠ CRITICAL physiology state")
        else:
            lines.append(f"│    state:     stable (no organ failure simulated)")
    else:
        lines.append(f"│    (physiology engine did not run)")

    # ── Section 7: Lab support (always honest about availability) ───────────
    lines.append(f"│")
    lines.append(f"│ 7. LAB SUPPORT")
    # Check if user provided any lab values
    user_labs = result.get("user_labs") or result.get("labs") or {}
    if user_labs:
        lines.append(f"│    provided:    {len(user_labs)} value(s)")
        for k, v in list(user_labs.items())[:5]:
            lines.append(f"│      {k}: {v}")
    else:
        lines.append(f"│    provided:    NOT PROVIDED")
        # Show diagnostic_criteria.lab_tests = what WOULD support this diagnosis
        dx_crit = supplemental.get("diagnostic_criteria") or {}
        expected_labs = dx_crit.get("lab_tests") or []
        if expected_labs:
            lines.append(f"│    would expect (diagnostic criteria for this disease):")
            for t in expected_labs[:5]:
                lines.append(f"│      • {t}")
        else:
            lines.append(f"│    (no diagnostic lab criteria recorded for this disease)")

    # ── Section 8: Evidence gate decision ───────────────────────────────────
    lines.append(f"│")
    lines.append(f"│ 8. EVIDENCE GATE  (final decision: announce label or downgrade)")
    if ea:
        state = ea.get("output_state", "?")
        label_ok = ea.get("disease_label_allowed", False)
        details = ea.get("disease_gate_details") or {}
        lines.append(f"│    output_state:    {state}")
        lines.append(f"│    label_allowed:   {label_ok}")
        if details:
            anchor_count = details.get("anchor_count", "?")
            top_score = details.get("top_score", "?")
            top_margin = details.get("top_margin", "?")
            contract = details.get("contract_gate", "?")
            score_gate = details.get("score_gate", "?")
            sys_match = details.get("dis_system_match", "?")
            frozen_by = details.get("frozen_by", "")
            review_blocked = details.get("review_status_blocked", False)
            lines.append(f"│    anchor_count:    {anchor_count}")
            lines.append(f"│    top_score:       {top_score}  (margin: {top_margin})")
            lines.append(f"│    contract_gate:   {contract}  (sys_match={sys_match})")
            lines.append(f"│    score_gate:      {score_gate}")
            if frozen_by:
                lines.append(f"│    frozen_by:       '{frozen_by}'  (high-risk symptom override)")
            if review_blocked:
                lines.append(f"│    ⚠ REVIEW STATUS:  UNREVIEWED — cascade pending medical review")
                lines.append(f"│                       label refused regardless of gate strength")
                lines.append(f"│                       (this is a safety guard, not a flaw in cascade match)")
        if not label_ok:
            syndrome = ea.get("syndrome_label", "")
            lines.append(f"│    → label refused; reported as syndrome: '{syndrome}'")
    else:
        lines.append(f"│    (evidence_assessment not in result — gate did not run)")

    # ── Section 9: Uncertainty ──────────────────────────────────────────────
    lines.append(f"│")
    lines.append(f"│ 9. UNCERTAINTY")
    uncertainty_items = []
    if missing_syms:
        uncertainty_items.append(f"missing cascade symptoms ({len(missing_syms)}):")
        for s in missing_syms[:5]:
            uncertainty_items.append(f"  • {s}")
        if len(missing_syms) > 5:
            uncertainty_items.append(f"  … and {len(missing_syms) - 5} more")
    if not user_labs:
        uncertainty_items.append("no lab values provided to confirm/refute")
    # Differential proximity
    if is_primary:
        diagnoses = result.get("nexus_diagnoses") or result.get("diagnoses") or []
        if len(diagnoses) >= 2:
            top_score = diagnoses[0].get("score", 0)
            close_alts = [
                d for d in diagnoses[1:4]
                if (top_score - d.get("score", 0)) < 0.15
            ]
            if close_alts:
                uncertainty_items.append(f"close differentials (margin < 0.15):")
                for d in close_alts:
                    uncertainty_items.append(f"  • {d.get('disease', '?')} "
                                             f"(score={d.get('score', 0)})")
    if phys.get("critical"):
        uncertainty_items.append(f"physiology is CRITICAL — diagnosis subject to "
                                  f"urgent re-evaluation")
    if not uncertainty_items:
        lines.append(f"│    (none recorded — high-confidence cascade match)")
    else:
        for item in uncertainty_items:
            lines.append(f"│    {item}")

    # ── Section 10: Proof strength (two-layer) ──────────────────────────────
    lines.append(f"│")
    lines.append(f"│ 10. PROOF STRENGTH  → {strength['level'].upper()}")
    lines.append(f"│     limited by:  {strength['weakest']}")
    cs = strength["cascade_support"]
    fl = strength["final_label_proof"]
    lines.append(f"│")
    lines.append(f"│     ┌─ Layer 1: cascade_support → {cs['level'].upper():9s} "
                 f"({cs['score']}/{cs['max_score']})")
    lines.append(f"│     │  (Is the mechanism evidence credible?)")
    for reason in cs["reasons"]:
        lines.append(f"│     │  · {reason}")
    lines.append(f"│     │")
    lines.append(f"│     └─ Layer 2: final_label_proof → {fl['level'].upper():9s} "
                 f"({fl['score']}/{fl['max_score']})")
    lines.append(f"│        (Is it safe to announce this disease name?)")
    for reason in fl["reasons"]:
        lines.append(f"│        · {reason}")

    return lines