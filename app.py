"""
app.py — Osler Medical AI
Pipeline replaced by nexus_runner.py (thin symptom+triage layer).
All disease reasoning done by NEXUS.
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import json
import uuid
from datetime import datetime, timedelta

# ── Flask ─────────────────────────────────────────────────────────────────────
from flask import (
    Flask, render_template, request, session,
    Response, jsonify, after_this_request,
)
from flask_session import Session

# ── NEXUS runner (replaces ai_doctor_pipeline) ────────────────────────────────
from nexus_runner import ai_doctor_pipeline          # drop-in compatible shim

# ── Deterministic patient-facing response ────────────────────────────────────
from deterministic_response import generate_response

# ── NEXUS reasoning engine ────────────────────────────────────────────────────
from nexus_engine.nexus_routes import nexus_bp, init_nexus, get_learner
from nexus_engine.anatomy_bridge import AnatomyBridge
from nexus_engine.nexus_trace import NexusTracedBridge, print_trace

# ── Safety gate ───────────────────────────────────────────────────────────────
from safety_switch import (
    DEFAULT_SAFE_MODE,
    enforce_medication_gate,
    sanitize_answer,
    build_safety_event,
)

# ── Symptom / follow-up infrastructure ───────────────────────────────────────
from symptom_loader import load_symptom_db
from followup_slots import set_symptom_db, get_slot_rules
from followup_state import FollowupState
from multi_followup_engine import update_followup_loop

# ── OTC / pill DB ─────────────────────────────────────────────────────────────
from medical_pills import load_pill_db, PILL_DB

# ── DNA / variant layer ───────────────────────────────────────────────────────
from variant_layer import parse_23andme_txt, parse_vcf

# ── Optional: FAISS (removed — stubbed) ──────────────────────────────────────
try:
    import faiss
except ImportError:
    faiss = None

# ── Optional: PDF reader ──────────────────────────────────────────────────────
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

# ══════════════════════════════════════════════════════════════════════════════
#   FLASK APP CONFIG
# ══════════════════════════════════════════════════════════════════════════════
app = Flask(__name__)
app.secret_key = "your_super_secret_key_here"
app.permanent_session_lifetime = timedelta(hours=1)
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_FILE_DIR"] = "./flask_sessions"
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_USE_SIGNER"] = True
Session(app)

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
SYM_DIR   = os.path.join(BASE_DIR, "medical_knowledge", "symptoms")
SYMPTOM_DIR = SYM_DIR

# ══════════════════════════════════════════════════════════════════════════════
#   STARTUP — load databases once
# ══════════════════════════════════════════════════════════════════════════════
load_pill_db()

db = load_symptom_db("medical_knowledge/symptoms")
if isinstance(db, dict):
    db = list(db.values())
set_symptom_db(db)
get_slot_rules()

print(f"[BOOT] pill_db={len(PILL_DB)}  symptom_db={len(db)}  slot_rules={len(get_slot_rules())}")

# ══════════════════════════════════════════════════════════════════════════════
#   NEXUS ENGINE INIT
# ══════════════════════════════════════════════════════════════════════════════
nexus_instance = init_nexus(
    symptom_dir="medical_knowledge/symptoms",
    mechanism_path="medical_knowledge/mechanisms/mechanisms_rag_final.json",
)
app.register_blueprint(nexus_bp)

anatomy_bridge = AnatomyBridge(nexus_instance)

# ── Wire 3D Spatial Visualization (now that nexus_instance exists) ──
try:
    from nexus_engine.spatial_visualization import spatial_bp, init_spatial
    from nexus_engine.spatial_engine import SpatialEngine
    _spatial_atlas = anatomy_bridge.atlas if hasattr(anatomy_bridge, 'atlas') else None
    _spatial_engine_instance = SpatialEngine(_spatial_atlas)
    init_spatial(_spatial_engine_instance, _spatial_atlas)
    app.config["nexus_instance"] = nexus_instance   # for /reason_3d endpoint
    app.register_blueprint(spatial_bp)
    print("[APP] 3D Spatial Visualization Blueprint registered (/spatial/*)")
except Exception as _se:
    print(f"[APP] Spatial visualization not available: {_se}")

# ── Static route for 3D viewer ──
@app.route('/anatomy3d')
def anatomy_3d_viewer():
    from flask import send_from_directory
    return send_from_directory('static', 'anatomy_3d.html')

_anatomy_kg = None
try:
    from nexus_engine.anatomy_knowledge_loader import AnatomyKnowledgeLoader
    from nexus_engine.nexus_core import KnowledgeGraph
    # Correct init: loader needs atlas (from anatomy_bridge) and kg instance
    _anatomy_kg = KnowledgeGraph()
    loader = AnatomyKnowledgeLoader(anatomy_bridge.atlas, _anatomy_kg)
    loader.load_all("medical_knowledge")
    triple_count = len(getattr(_anatomy_kg, "triples", []))
    print(f"[BOOT] anatomy KG loaded: {triple_count} triples")
    # Wire the KG into anatomy_bridge so _find_affected_organs can use it
    anatomy_bridge.kg = _anatomy_kg
    print("[BOOT] anatomy KG wired into anatomy_bridge")
except ImportError as e:
    print(f"[BOOT] anatomy KG module missing (non-blocking): {e}")
    _anatomy_kg = None
except TypeError as e:
    print(f"[BOOT] anatomy KG init error (non-blocking): {e}")
    _anatomy_kg = None
except Exception as e:
    print(f"[BOOT] anatomy KG unavailable (non-blocking): {e}")
    _anatomy_kg = None

traced_bridge = NexusTracedBridge(anatomy_bridge, nexus_instance, anatomy_kg=_anatomy_kg)

# ══════════════════════════════════════════════════════════════════════════════
#   SESSION HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def get_sid() -> str:
    if "sid" not in session:
        session["sid"] = str(uuid.uuid4())[:8]
    return session["sid"]

def _history_path(sid: str) -> str:
    os.makedirs("logs", exist_ok=True)
    return f"logs/history_{sid}.json"

def _followup_path(sid: str) -> str:
    os.makedirs("logs", exist_ok=True)
    return f"logs/followup_{sid}.json"

def load_history_server(sid: str):
    path = _history_path(sid)
    try:
        if os.path.exists(path):
            return json.load(open(path, encoding="utf-8"))
    except Exception:
        pass
    return []

def save_history_server(sid: str, history: list):
    try:
        with open(_history_path(sid), "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[history] save failed: {e}")

def get_followup_state() -> FollowupState:
    path = _followup_path(get_sid())
    try:
        if os.path.exists(path):
            # Try .load() class method first (if FollowupState supports it)
            if hasattr(FollowupState, "load"):
                return FollowupState.load(path)
            # Fallback: load JSON and reconstruct
            data = json.load(open(path, encoding="utf-8"))
            state = FollowupState(target_questions=3)
            if isinstance(data, dict):
                for k, v in data.items():
                    try:
                        setattr(state, k, v)
                    except Exception:
                        pass
            return state
    except Exception as e:
        print(f"[followup] load failed (using fresh state): {e}")
    return FollowupState(target_questions=3)

def save_followup_state(state: FollowupState):
    path = _followup_path(get_sid())
    try:
        # Try .save() method first (if FollowupState supports it)
        if hasattr(state, "save"):
            state.save(path)
            return
        # Fallback: serialize to JSON manually
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {k: v for k, v in vars(state).items()
                if not k.startswith("_") and isinstance(v, (str, int, float, bool, list, dict))}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, default=str)
    except Exception as e:
        print(f"[followup] save failed (non-blocking): {e}")

def history_to_pipeline(history_rows: list) -> list:
    """Convert session history to the format nexus_runner expects."""
    result = []
    for row in (history_rows or []):
        if row.get("user"):
            result.append({"role": "user",      "content": row["user"]})
        if row.get("ai"):
            result.append({"role": "assistant", "content": row["ai"]})
    return result

def user_denies_other_symptoms(text: str) -> bool:
    t = text.lower().strip()
    return t in {"no", "nope", "none", "no other symptoms", "that's all",
                 "that is all", "nothing else", "no more"}


# ══════════════════════════════════════════════════════════════════════════════
#   G.5 #2c — NEXUS MULTI-TURN STATE
#   Accumulate per-session symptoms/denials/answers across NEXUS turns.
#   Independent from FollowupState (which is for free-text chat).
# ══════════════════════════════════════════════════════════════════════════════

def _nexus_turn_path(sid: str) -> str:
    """Path to JSON file holding per-session NEXUS multi-turn accumulator."""
    os.makedirs("logs", exist_ok=True)
    return f"logs/nexus_turn_{sid}.json"


NEXUS_TURN_STATE_SCHEMA = 1   # increment when state shape changes


def get_nexus_turn_state() -> dict:
    """Load NEXUS multi-turn accumulator for current session.

    Schema (v1):
      {
        "schema_version":    int,
        "symptoms":          [...],           # accumulated (with onset if known)
        "denied_symptoms":   [...],
        "context":           {...},           # age/sex/vitals/history/etc.
        "turn_count":        int,
        "asked_about":       [...],           # symptom names already asked
        "answers":           {sym: bool},     # which past asks user answered
        "last_diagnoses":    [...],           # for context
      }

    On schema mismatch, returns a fresh state (silent reset).
    """
    fresh_state = {
        "schema_version":   NEXUS_TURN_STATE_SCHEMA,
        "symptoms":         [],
        "denied_symptoms":  [],
        "context":          {},
        "turn_count":       0,
        "asked_about":      [],
        "answers":          {},
        "last_diagnoses":   [],
    }
    path = _nexus_turn_path(get_sid())
    try:
        if os.path.exists(path):
            loaded = json.load(open(path, encoding="utf-8"))
            # Schema mismatch → fresh state (avoid crashing on old/future shape)
            if loaded.get("schema_version") != NEXUS_TURN_STATE_SCHEMA:
                return fresh_state
            # Fill any missing keys with defaults (forward-compat)
            for k, v in fresh_state.items():
                loaded.setdefault(k, v)
            return loaded
    except Exception:
        pass
    return fresh_state


def save_nexus_turn_state(state: dict) -> None:
    """Persist NEXUS multi-turn state to disk."""
    try:
        # Always stamp current schema version on save
        state["schema_version"] = NEXUS_TURN_STATE_SCHEMA
        path = _nexus_turn_path(get_sid())
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[nexus_turn] save failed: {e}")


def reset_nexus_turn_state() -> None:
    """Clear NEXUS multi-turn state (start fresh)."""
    path = _nexus_turn_path(get_sid())
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
#   G.4 D — Patient context extraction (demographics + vitals)
# ══════════════════════════════════════════════════════════════════════════════

def _extract_patient_context(request_data: dict) -> dict:
    """Extract patient demographics + vitals from request body for nexus.reason().

    Accepts the JSON body of /ai_doctor or /chat_stream. Pulls the "context"
    object if present, with backward-compatible fallback to flat fields
    (e.g. "age" at top level). Returns a clean dict ready for reason(context=).

    All fields are optional. Empty dict if no context provided — reason()
    will just skip the context modifier step in that case.

    Validates basic types and ranges; rejects malformed input silently
    rather than crashing the request.
    """
    if not isinstance(request_data, dict):
        return {}

    # Prefer nested "context" key, fall back to flat fields
    raw = request_data.get("context")
    if not isinstance(raw, dict):
        # Backward-compat: pick up flat top-level fields if no context object
        raw = {}
        for k in ("age", "sex", "pregnancy_status", "vitals",
                  "history", "medications"):
            if k in request_data:
                raw[k] = request_data[k]

    ctx = {}

    # age: int 0-120
    age = raw.get("age")
    if isinstance(age, (int, float)) and 0 <= age <= 120:
        ctx["age"] = int(age)

    # sex: lowercase string
    sex = raw.get("sex")
    if isinstance(sex, str) and sex.lower() in {"male", "female", "other", "unknown"}:
        ctx["sex"] = sex.lower()

    # pregnancy_status
    preg = raw.get("pregnancy_status")
    if isinstance(preg, str) and preg.lower() in {"pregnant", "not_pregnant",
                                                      "unknown", "postpartum"}:
        ctx["pregnancy_status"] = preg.lower()

    # vitals: dict of numeric measurements
    vitals = raw.get("vitals")
    if isinstance(vitals, dict):
        clean_vitals = {}
        for k, v in vitals.items():
            if not isinstance(k, str):
                continue
            if isinstance(v, (int, float)) and v > 0:
                clean_vitals[k] = float(v) if isinstance(v, float) else v
        if clean_vitals:
            ctx["vitals"] = clean_vitals

    # history: list of strings (free text)
    history = raw.get("history")
    if isinstance(history, list):
        clean_hist = [str(h).strip() for h in history
                      if isinstance(h, (str, int, float)) and str(h).strip()]
        if clean_hist:
            ctx["history"] = clean_hist[:20]  # cap to avoid huge inputs

    # medications: list of strings
    meds = raw.get("medications")
    if isinstance(meds, list):
        clean_meds = [str(m).strip() for m in meds
                      if isinstance(m, (str, int, float)) and str(m).strip()]
        if clean_meds:
            ctx["medications"] = clean_meds[:30]

    # G.5 #2: duration_hours (numeric, for temporal reasoning)
    duration = raw.get("duration_hours")
    if isinstance(duration, (int, float)) and duration > 0:
        ctx["duration_hours"] = float(duration)

    # G.5 #5: drug_responses (list of {drug, effect, symptom})
    drs = raw.get("drug_responses")
    if isinstance(drs, list):
        clean_drs = []
        for dr in drs[:20]:  # cap to 20
            if not isinstance(dr, dict):
                continue
            drug = str(dr.get("drug", "")).strip().lower()
            effect = str(dr.get("effect", "")).strip().lower()
            sym = str(dr.get("symptom", "")).strip().lower()
            if drug and effect:
                clean_drs.append({"drug": drug, "effect": effect, "symptom": sym})
        if clean_drs:
            ctx["drug_responses"] = clean_drs

    # G.5 #3: denied_symptoms (list of strings — patient explicitly denies these)
    denied = raw.get("denied_symptoms")
    if isinstance(denied, list):
        clean_denied = [str(s).strip().lower() for s in denied
                        if isinstance(s, (str, int, float)) and str(s).strip()]
        if clean_denied:
            ctx["denied_symptoms"] = clean_denied[:30]  # cap to 30

    return ctx


# Canonical 22 state-model organs (for trace parsing)
_STATE_MODEL_ORGANS = {
    "heart", "lung", "brain", "liver", "kidney", "gi", "bladder", "skin",
    "muscle", "bone", "thyroid", "spleen", "pancreas", "gallbladder",
    "adrenal", "reproductive", "vessel", "eye", "ear", "peripheral_nerve",
    "upper_airway", "blood",
}


def _extract_organs_from_trace(trace: dict) -> list:
    """Pull affected organ names out of a diagnosis reasoning_trace.

    Section 1 perturbation lines look like:
      "↑ischemia (+0.85) in heart: coronary artery occlusion ..."
    We grab the token after ' in ' and before ':'. Returns deduped list
    of canonical state-model organ names (so the frontend can highlight them).
    """
    organs = []
    seen = set()
    for line in (trace.get("1_disease_perturbation") or []):
        s = str(line)
        # find " in <organ>:" pattern
        idx = s.find(" in ")
        if idx < 0:
            continue
        rest = s[idx + 4:]
        colon = rest.find(":")
        organ = (rest[:colon] if colon >= 0 else rest).strip().lower()
        # normalize ("upper airway" -> "upper_airway")
        organ = organ.replace(" ", "_")
        if organ in _STATE_MODEL_ORGANS and organ not in seen:
            seen.add(organ)
            organs.append(organ)
    return organs


# ══════════════════════════════════════════════════════════════════════════════
#   ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def home():
    session.permanent = True
    get_sid()
    return render_template("chat.html")


@app.get("/api/symptoms")
def api_symptoms():
    """Return all known symptoms for the UI symptom picker."""
    if not os.path.isdir(SYM_DIR):
        return jsonify({"symptoms": [], "error": f"not found: {SYM_DIR}"}), 404

    items, seen = [], {}
    for fn in os.listdir(SYM_DIR):
        if not fn.endswith(".json"):
            continue
        try:
            obj  = json.load(open(os.path.join(SYM_DIR, fn), encoding="utf-8"))
            name = (
                (obj.get("symptom") or "").strip()
                or (obj.get("name") or "").strip()
                or os.path.splitext(fn)[0].replace("_", " ")
            )
            if not name:
                continue
            key = name.lower()
            if key not in seen:
                seen[key] = True
                items.append({
                    "symptom":          name,
                    "medical_term":     obj.get("medical_term", ""),
                    "category":         obj.get("category", ""),
                    "system":           obj.get("system", ""),
                    "common_causes":    obj.get("common_causes", []),
                    "red_flags":        obj.get("red_flags", []),
                    "related_symptoms": obj.get("related_symptoms", []),
                })
        except Exception as e:
            print(f"[api_symptoms] {fn}: {e}")

    return jsonify({"symptoms": sorted(items, key=lambda x: x["symptom"].lower())})


@app.post("/upload_dna")
def upload_dna():
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400
    file     = request.files["file"]
    filename = file.filename.lower()
    os.makedirs("uploads", exist_ok=True)
    save_path = os.path.join("uploads", file.filename)
    file.save(save_path)

    if filename.endswith(".txt"):
        dna_variants, source = parse_23andme_txt(save_path), "23andme"
    elif filename.endswith(".vcf"):
        dna_variants, source = parse_vcf(save_path), "vcf"
    else:
        return jsonify({"error": "unsupported file type"}), 400

    session["dna_variants"] = dna_variants
    return jsonify({"status": "ok", "source": source,
                    "variant_count": len(dna_variants),
                    "example": dict(list(dna_variants.items())[:5])})


@app.post("/nexus_turn")
def nexus_turn():
    """G.5 #2c: Multi-turn NEXUS reasoning endpoint.

    Accumulates symptoms/denials/context across turns of patient interaction.
    Each turn returns updated diagnoses + follow-up questions. User answers
    feed back into next turn's input.

    Request body — supports three modes (controlled by `action`):

    1. action="start" (or omitted): begin a NEW case. Resets prior turn state.
       {
         "action":   "start",
         "symptoms": [...] or [{"name": "...", "onset_hours_ago": N}, ...],
         "context":  {age/sex/vitals/history/...}
       }

    2. action="answer": user answered a follow-up question.
       {
         "action":             "answer",
         "answers": [          # one or more
           {"symptom": "diaphoresis", "present": true},
           {"symptom": "left arm pain", "present": false, "onset_hours_ago": null}
         ]
       }
       The accumulator adds present=true ones to symptoms,
       present=false ones to denied_symptoms.

    3. action="reset": clear session state.
       {"action": "reset"}

    Response:
      {
        "result": {
          "turn":             int,
          "symptoms":         [...],
          "denied_symptoms":  [...],
          "nexus_diagnoses":  [...],
          "nexus_suggested_questions": [...],
          "nexus_reasoning_trace": [...],
          "patient_context":  {...},
          "lab_recommendations": {...},
          "session_id":       str
        }
      }
    """
    data    = request.get_json(silent=True) or {}
    action  = (data.get("action") or "start").lower()

    # ── action=reset ─────────────────────────────────────────────────────
    if action == "reset":
        reset_nexus_turn_state()
        return jsonify({"result": {"status": "reset", "session_id": get_sid()}})

    state = get_nexus_turn_state()

    # ── action=start: fresh case ─────────────────────────────────────────
    if action == "start":
        # Reset and seed
        reset_nexus_turn_state()
        state = get_nexus_turn_state()

        # Initial symptoms (mixed str / dict OK — NEXUS reason() handles)
        symptoms_in = data.get("symptoms") or []
        state["symptoms"] = symptoms_in if isinstance(symptoms_in, list) else []

        # Patient context
        state["context"] = _extract_patient_context(data)

    # ── action=answer: incremental update ────────────────────────────────
    elif action == "answer":
        answers = data.get("answers") or []
        if not isinstance(answers, list):
            answers = []
        for ans in answers:
            if not isinstance(ans, dict):
                continue
            sym = str(ans.get("symptom", "")).strip().lower()
            if not sym:
                continue
            present = ans.get("present")
            # Mark this symptom as answered
            state["answers"][sym] = bool(present) if present is not None else None
            if present is True:
                # Add to symptoms (with optional onset)
                onset = ans.get("onset_hours_ago")
                if isinstance(onset, (int, float)) and onset >= 0:
                    state["symptoms"].append({"name": sym, "onset_hours_ago": float(onset)})
                else:
                    state["symptoms"].append(sym)
            elif present is False:
                # Add to denied symptoms
                if sym not in state["denied_symptoms"]:
                    state["denied_symptoms"].append(sym)

        # If user provided extra context updates, merge them
        new_ctx = _extract_patient_context(data)
        if new_ctx:
            # Merge — new values override
            state["context"] = {**state["context"], **new_ctx}

    else:
        return jsonify({"error": f"unknown action: {action!r}"}), 400

    # ── Compose context for nexus_instance.reason() ──────────────────────
    full_context = dict(state["context"])  # copy
    if state["denied_symptoms"]:
        full_context["denied_symptoms"] = state["denied_symptoms"]

    # ── Run NEXUS reason() ───────────────────────────────────────────────
    try:
        _reason_out = nexus_instance.reason(
            state["symptoms"], "", context=full_context
        )
    except Exception as e:
        return jsonify({"error": f"reason() failed: {e}"}), 500

    state["turn_count"] = int(state.get("turn_count", 0)) + 1
    state["last_diagnoses"] = [
        {"disease": d["disease"], "score": d["score"]}
        for d in _reason_out.get("diagnoses", [])[:5]
    ]

    # ── Run agent pipeline for QuestionAgent ──────────────────────────────
    questions = []
    try:
        from nexus_engine.nexus_agents import AgentOrchestrator
        # Build symptom name list for orchestrator (it expects strings)
        sym_names = []
        for s in state["symptoms"]:
            if isinstance(s, dict):
                sym_names.append(s.get("name", ""))
            else:
                sym_names.append(str(s))
        sym_names = [s for s in sym_names if s]

        orch = AgentOrchestrator()
        agent_state = orch.run(
            selected_symptoms=sym_names, user_text="",
            raw_result={"nexus_diagnoses":   _reason_out.get("diagnoses", []),
                        "nexus_consistency": _reason_out.get("nexus_consistency", {}),
                        "nexus_thinking":    _reason_out.get("thinking", []),
                        "nexus_stats":       _reason_out.get("stats", {})},
            nexus_instance=nexus_instance,
        )
        raw_questions = agent_state.follow_up_questions or []
        # FILTER OUT questions already asked
        already_asked = set(state.get("asked_about", []))
        for q in raw_questions:
            sym = q.get("asks_about", "") if isinstance(q, dict) else ""
            if sym and sym in already_asked:
                continue
            questions.append(q)
            if sym:
                already_asked.add(sym)
        state["asked_about"] = list(already_asked)
    except Exception as e:
        print(f"[nexus_turn] QuestionAgent skipped: {e}")

    save_nexus_turn_state(state)

    return jsonify({
        "result": {
            "turn":              state["turn_count"],
            "symptoms":          state["symptoms"],
            "denied_symptoms":   state["denied_symptoms"],
            "nexus_diagnoses":   _reason_out.get("diagnoses", []),
            "nexus_suggested_questions": questions,
            "nexus_reasoning_trace": [
                {"disease": d["disease"], "trace": d.get("reasoning_trace", {})}
                for d in _reason_out.get("diagnoses", [])[:3]
                if d.get("reasoning_trace")
            ],
            "patient_context":   state["context"],
            "lab_recommendations": _reason_out.get("lab_recommendations", {}),
            "session_id":        get_sid(),
        }
    })


@app.post("/ai_doctor")
def ai_doctor():
    """Simple JSON endpoint (no streaming).

    Accepts optional patient context for mechanism-based reasoning.
    Behaviorally consistent with /chat_stream — same agent pipeline,
    same QuestionAgent, same Verifier, same EvidenceGate. Difference is
    only response format (JSON vs SSE stream).

    Request body:
      {
        "message": "...",                  # required: free-text or symptoms
        "dna_variants": {...},             # optional: genetics
        "context": {                       # optional: patient context
          "age": 32,
          "sex": "male" | "female",
          "pregnancy_status": "pregnant" | "not_pregnant" | "unknown",
          "vitals": {"bp_sys": 160, "bp_dia": 100, "hr": 105, ...},
          "history": ["pregnancy 32 weeks"],
          "medications": ["metformin"],
          "duration_hours": 1.0,          # G.5: temporal reasoning
          "drug_responses": [             # G.5: drug response inference
            {"drug": "nitroglycerin", "effect": "relieved", "symptom": "chest pain"}
          ]
        }
      }
    """
    data        = request.get_json(silent=True) or {}
    user_input  = (data.get("message") or "").strip()
    dna_variants = data.get("dna_variants") or session.get("dna_variants", {})
    context     = {"dna_variants": dna_variants}

    # G.4 D: extract patient context for mechanism reasoning
    patient_context = _extract_patient_context(data)

    result = ai_doctor_pipeline(user_input, context=context)

    try:
        # ── Step 1: nexus_medical.reason() with patient context ──
        _reason_out = nexus_instance.reason(
            result.get("symptoms", result.get("final_symptoms", [])),
            user_input or "",
            context=patient_context)
        result["nexus_diagnoses"]   = _reason_out.get("diagnoses", [])
        result["nexus_thinking"]    = _reason_out.get("thinking", [])
        result["nexus_red_flags"]   = _reason_out.get("red_flags", [])
        result["nexus_stats"]       = _reason_out.get("stats", {})
        result["nexus_detected_systems"] = _reason_out.get("detected_systems", [])
        result["nexus_consistency"] = _reason_out.get("nexus_consistency", {})
        # G.4 E: surface reasoning_trace from top-3 diagnoses
        result["nexus_reasoning_trace"] = [
            {"disease": d["disease"], "trace": d.get("reasoning_trace", {})}
            for d in _reason_out.get("diagnoses", [])[:3]
            if d.get("reasoning_trace")
        ]
        result["patient_context"]   = patient_context  # echo back for transparency

        # ── Step 2: Run agent pipeline (QuestionAgent, Verifier, etc.) ──
        # Same orchestrator as /chat_stream for behavioral consistency.
        try:
            from nexus_engine.nexus_agents import AgentOrchestrator
            selected_symptoms = result.get("symptoms",
                                            result.get("final_symptoms", []))
            _orchestrator = AgentOrchestrator()
            _agent_state  = _orchestrator.run(
                selected_symptoms = selected_symptoms,
                user_text         = user_input or "",
                raw_result        = result,
                nexus_instance    = nexus_instance,
            )
            # Merge agent state back into result (same pattern as /chat_stream)
            _agent_result = _agent_state.to_result_dict()
            result.update(_agent_result)
            # Don't include _agent_state object in JSON response (non-serializable)
        except ImportError as _e:
            print(f"[NEXUS] ai_doctor: AgentOrchestrator unavailable: {_e}")
        except Exception as _e:
            # Agent pipeline failure shouldn't block the response —
            # we still have nexus_diagnoses from step 1.
            print(f"[NEXUS] ai_doctor: agent pipeline failed: {_e}")
    except Exception as e:
        print("[NEXUS] ai_doctor reason failed:", e)

    # Strip non-serializable fields before returning
    result.pop("_nexus_instance", None)
    result.pop("_agent_state", None)
    return jsonify({"result": result}), 200


@app.post("/chat_stream")
def chat_stream():
    """Main streaming chat endpoint."""
    data             = request.get_json(silent=True) or {}
    user_input       = (data.get("message") or "").strip()
    selected_symptoms = data.get("selected_symptoms") or []
    if not isinstance(selected_symptoms, list):
        selected_symptoms = []

    if not user_input:
        return Response("Please say something.", mimetype="text/plain")

    safe_mode = os.environ.get("SAFE_MODE", DEFAULT_SAFE_MODE).lower()

    # 1. Load history
    sid     = get_sid()
    history = load_history_server(sid)

    # 2. FollowupState
    state = get_followup_state()
    state.consume_simple_answers(user_input)

    old = getattr(state, "ctx", {}) if isinstance(getattr(state, "ctx", {}), dict) else {}
    ctx = {
        "dna_variants":        session.get("dna_variants", {}),
        "selected_symptoms":   selected_symptoms,
        "slots":               old.get("slots", {}),
        "facts":               old.get("facts", {}),
        "followup_slot_counts":old.get("followup_slot_counts", {}),
        "followup_history":    old.get("followup_history", []),
    }
    state.ctx = ctx

    # 3. NexusRunner — symptom extraction + triage + red flags
    # single source of truth: selected_symptoms + text extraction merged inside nexus_runner
    result = ai_doctor_pipeline(
        user_input=user_input,
        mode="patient",
        context=ctx,
        history=history_to_pipeline(history),
    )

    # Input sync validation — warn if UI selection differs from extracted symptoms
    _result_syms  = set(s.lower().strip() for s in (result.get("symptoms") or []))
    _selected_set = set(s.lower().strip() for s in selected_symptoms)
    if _selected_set and not _selected_set.issubset(_result_syms):
        _missing = _selected_set - _result_syms
        print(f"[WARN][INPUT SYNC] selected symptoms not in result: {_missing}")
        # Force merge: add any missing selected symptoms to result
        merged = list(result.get("symptoms", []))
        for s in selected_symptoms:
            if s.lower().strip() not in _result_syms:
                merged.append(s.lower().strip())
        result["symptoms"] = merged
        result["final_symptoms"] = merged
        print(f"[INPUT SYNC] corrected symptoms: {merged}")

    # 3.4. Pre-enhance high-risk freeze — loaded from high_risk_gate (red_flags.json)
    try:
        from nexus_engine.high_risk_gate import (
            is_high_risk       as _app_is_hr,
            get_dominant       as _app_get_dom,
            get_freeze_syndrome as _app_get_fs,
            build_freeze_assessment as _app_build_fa,
        )
        _pre_hr_active   = _app_is_hr(result.get("symptoms") or [])
        _pre_hr_dominant = _app_get_dom(result.get("symptoms") or []) or ""
    except ImportError:
        _HR_FB = {"chest pain","shortness of breath","syncope","fainting",
                  "altered mental status","confusion","focal weakness","slurred speech",
                  "worst headache","thunderclap headache","severe abdominal pain",
                  "hematemesis","hemoptysis","bloody stool","melena"}
        _s_lower       = {s.lower().replace("_"," ").strip() for s in (result.get("symptoms") or [])}
        _pre_hr_active = bool(_s_lower & _HR_FB)
        _pre_hr_dominant = next((s for s in ["chest pain","shortness of breath","syncope"]
                                 if s in _s_lower), "")
        def _app_get_fs(syms): return "high-risk syndrome"
        def _app_build_fa(syms, triage="PROMPT"): return {
            "output_state":"syndrome","syndrome_label":"high-risk syndrome",
            "etiology_allowed":False,"disease_label_allowed":False,"has_red_flags":True}

    if _pre_hr_active:
        _freeze_label = _app_get_fs(result.get("symptoms") or []) or "high-risk syndrome"
        result["evidence_assessment"] = _app_build_fa(
            result.get("symptoms") or [],
            triage_level=(result.get("triage") or {}).get("level","PROMPT")
        )
        result["evidence_assessment"]["etiology_evidence"] = f"pre-enhance freeze: {_pre_hr_dominant}"
        result["_high_risk_freeze"]  = True
        result["_freeze_dominant"]   = _pre_hr_dominant
        result["_freeze_syndrome"]   = _freeze_label
        result["nexus_otc_hints"]    = []
        print(f"[PRE-ENHANCE FREEZE] '{_pre_hr_dominant}' → "
              f"syndrome='{_freeze_label}' etiol/OTC/label suppressed")

    # 3.5. NEXUS agent pipeline
    # G.4 D: extract patient context (age/sex/vitals/history) for reasoning
    patient_context = _extract_patient_context(data)
    try:
        # Try new agent orchestrator first
        try:
            # Run nexus_medical.reason() FIRST so raw_result has mechanism data
            if nexus_instance:
                try:
                    _reason_out = nexus_instance.reason(
                        selected_symptoms, user_input or "",
                        context=patient_context)
                    result["nexus_diagnoses"]       = _reason_out.get("diagnoses", [])
                    result["nexus_blocked_diseases"]= _reason_out.get("blocked_diseases", [])
                    result["nexus_thinking"]        = _reason_out.get("thinking", [])
                    result["nexus_stats"]           = _reason_out.get("stats", {})
                    result["nexus_consistency"]     = _reason_out.get("nexus_consistency",
                                                        {"consistency_score": 0.0})
                    result["nexus_detected_systems"]= _reason_out.get("detected_systems", [])
                    result["nexus_etiology"]        = _reason_out.get("nexus_etiology", {})
                    # G.5 #9/#10: surface lab + treatment recommendations
                    result["lab_recommendations"]       = _reason_out.get(
                        "lab_recommendations", {})
                    result["treatment_recommendations"] = _reason_out.get(
                        "treatment_recommendations", {})
                    result["nexus_suggested_questions"] = _reason_out.get(
                        "suggested_questions", [])
                    result["nexus_predicted_symptoms"]  = _reason_out.get(
                        "predicted_symptoms", [])
                    result["nexus_otc_hints"]       = _reason_out.get("otc_hints", [])
                    result["nexus_weak_questions"]  = _reason_out.get(
                        "nexus_weak_questions", [])
                    result["nexus_weak_organs"]     = _reason_out.get(
                        "nexus_weak_organs", [])
                    # G.4 E: surface reasoning_trace from top-5 diagnoses
                    result["nexus_reasoning_trace"] = [
                        {"disease": d["disease"], "trace": d.get("reasoning_trace", {})}
                        for d in _reason_out.get("diagnoses", [])[:5]
                        if d.get("reasoning_trace")
                    ]
                    result["patient_context"] = patient_context
                except Exception as _re:
                    print(f"[NEXUS] reason() failed: {_re}")

            # Optional: NexusAgentConnector neural boost (requires trained checkpoint)
            try:
                from nexus_agent_connector import agent_diagnose
                _neural = agent_diagnose(selected_symptoms)
                if _neural and _neural.get("status") == "ok":
                    # Boost diseases confirmed by neural model
                    _neural_top = {d["disease"].lower(): d["probability"]
                                   for d in _neural.get("top3", [])}
                    _cur_dx = result.get("nexus_diagnoses", [])
                    for d in _cur_dx:
                        _name = d.get("disease","").lower()
                        if _name in _neural_top:
                            d["score"] = round(
                                d["score"] * 0.7 + _neural_top[_name] * 0.3, 3)
                            d["neural_confidence"] = _neural_top[_name]
                    result["nexus_diagnoses"] = sorted(
                        _cur_dx, key=lambda x: -x.get("score",0))
                    result["_neural_active"] = True
            except (ImportError, FileNotFoundError):
                pass  # no checkpoint yet — skip silently

            from nexus_engine.nexus_agents import AgentOrchestrator
            _orchestrator = AgentOrchestrator()
            _agent_state  = _orchestrator.run(
                selected_symptoms = selected_symptoms,
                user_text         = user_input or "",
                raw_result        = result,
                nexus_instance    = nexus_instance,
            )
            # Merge agent state back into result
            _agent_result = _agent_state.to_result_dict()
            result.update(_agent_result)
            result["_agent_state"]   = _agent_state
            result["_nexus_instance"]= nexus_instance
        except ImportError as e:
            # AgentOrchestrator is required in pure-mechanism architecture.
            # If this import fails, there's a serious deployment issue.
            print(f"[NEXUS] CRITICAL: AgentOrchestrator import failed: {e}")
            raise
        print("[NEXUS] enhanced:  mechs=",
              result.get("nexus_stats", {}).get("mechanisms_activated", 0),
              " consistency=",
              round(result.get("nexus_consistency", {}).get("consistency_score", 0), 2))

        # ── Unify red flags: 3-bucket architecture ─────────────────────────
        # Bucket A — observed_red_flags:  already-happening danger → upgrades triage
        # Bucket B — risk_escalators:     possible risk, needs followup → caution only
        # Bucket C — watch_for_flags:     future warning for UI only → no triage impact
        import re as _re
        def _rf_key(name: str) -> str:
            return _re.sub(r"[^a-z0-9]", "", name.lower())

        nexus_rfs  = result.get("nexus_red_flags") or []
        runner_rfs = (result.get("red_flag_block") or {}).get("red_flags", [])

        observed_rfs  : list = []   # A — confirmed danger signals
        risk_escalators: list = []  # B — possible risk, not yet confirmed
        watch_for_flags: list = []  # C — UI-only "seek care if..."
        seen_keys: set = set()      # dedup by normalized rule name
        seen_watch: set = set()     # dedup C watch_for by normalized text

        def _norm_watch(text: str) -> str:
            """Canonical key for watch_for text — maps synonyms to same bucket."""
            import re as _re
            t = _re.sub(r"[^a-z0-9 ]", "", text.lower().strip())
            # Synonym map — prevents "vomiting blood" + "blood in vomit" both showing
            _synonyms = {
                "vomiting blood": "hematemesis",
                "blood in vomit": "hematemesis",
                "throwing up blood": "hematemesis",
                "bloody vomit": "hematemesis",
                "blood in stool": "hematochezia",
                "bloody stool": "hematochezia",
                "bloody diarrhea": "hematochezia",
                "dark stool": "melena",
                "black stool": "melena",
                "tarry stool": "melena",
                "passing out": "syncope",
                "loss of consciousness": "syncope",
                "fainting": "syncope",
            }
            for phrase, canonical in _synonyms.items():
                if phrase in t:
                    return canonical
            # Fall back to normalized text
            return _re.sub(r"[\s]+", "_", t)

        def _safe_bucket(name: str, bucket: str) -> str:
            """Hard validator: name semantics must match bucket.
            Reviewer rule: *_watch → C, *_risk/*_mixed/*_escalat → B.
            Prevents false triage escalation from misrouted flags."""
            n = name.lower()
            if (n.endswith("_watch") or n.startswith("watch_")) and bucket != "watch_for":
                return "watch_for"
            if any(k in n for k in ("_risk", "_mixed", "_pattern", "_escalat"))                and bucket == "observed_red_flag":
                return "combo_risk"
            return bucket

        for rf in runner_rfs:
            name   = rf.get("name") or ""
            key    = _rf_key(name)
            bucket = rf.get("bucket", "observed_red_flag")
            if not key or key in seen_keys:
                continue
            seen_keys.add(key)
            entry = {**rf, "source": "runner"}
            bucket = _safe_bucket(name, bucket)   # hard semantic validation
            if bucket == "watch_for":
                _wf_text = entry.get("watch_for", name)
                _wf_key  = _norm_watch(_wf_text)
                if _wf_key not in seen_watch:
                    seen_watch.add(_wf_key)
                    watch_for_flags.append(entry)
            elif bucket == "combo_risk":
                risk_escalators.append(entry)
            else:
                observed_rfs.append(entry)   # observed_red_flag

        # Symptom tokens from current request (for relevance filtering)
        _sym_tokens = set()
        for s in selected_symptoms + list(_result_syms):
            _sym_tokens.update(w for w in s.lower().split() if len(w) > 3)

        # Observed symptoms (A-bucket) — escalators must not duplicate these
        _observed_sym_names = {r.get("name","").lower() for r in observed_rfs}
        _observed_watch_syms = set()
        for r in observed_rfs:
            for s in r.get("symptoms", []):
                _observed_watch_syms.add(s.lower().strip())

        def _nexus_rf_relevant(name: str) -> bool:
            """Only include NEXUS escalators that:
            1. Relate to an actual symptom (relevance filter)
            2. Are not already captured in A-observed (dedup filter)
            3. Are not too generic/noisy for the symptom set (specificity filter)
            """
            if not name:
                return False
            name_l = name.lower()

            # Dedup: if this escalator text matches an A-observed symptom, skip it
            for obs_sym in _observed_watch_syms | _result_syms:
                if len(obs_sym) > 4 and obs_sym in name_l:
                    return False

            # Noise filter: reject escalators that require high-specificity evidence
            # not supported by the current symptom set
            _NOISE_PATTERNS = {
                "pathological fracture", "bone pain", "pathological",
                "malignancy", "cancer", "tumor", "mass",
                "compartment syndrome", "osteomyelitis",
                "dissection",   # unless chest pain + tearing present
                "embolism",     # unless unilateral leg swelling + SOB
                "tamponade",    # unless chest pain + hypotension
            }
            if any(np in name_l for np in _NOISE_PATTERNS):
                # Check if supporting evidence exists
                _DISSECT_SUPPORT  = {"tearing", "chest pain", "back pain", "hypertension"}
                _EMBOLISM_SUPPORT = {"leg swelling", "shortness of breath", "unilateral"}
                if "dissection" in name_l and not (_result_syms & _DISSECT_SUPPORT):
                    return False
                if "embolism" in name_l and not (_result_syms & _EMBOLISM_SUPPORT):
                    return False
                if any(np in name_l for np in {"pathological fracture","bone pain",
                                               "malignancy","cancer","tumor","mass",
                                               "compartment","osteomyelitis"}):
                    return False

            name_words = set(w for w in name_l.split() if len(w) > 3)
            # Accept if ≥1 symptom word appears in the red flag text
            if name_words & _sym_tokens:
                return True
            # Accept known safe generic escalation patterns
            safe_patterns = {"dehydrat", "worsening", "radiating", "radiat",
                             "sepsis", "systemic", "unable", "cannot",
                             "sweating", "diaphor", "syncope", "faint"}
            if any(p in name_l for p in safe_patterns):
                return True
            return False

        for rf in nexus_rfs:
            name = rf.get("red_flag") or rf.get("condition") or rf.get("name") or ""
            key  = _rf_key(name)
            if not key or key in seen_keys:
                continue
            # Filter: only relevant escalators (prevents e.g. fracture from joint pain alone)
            if not _nexus_rf_relevant(name):
                print(f"[RED_FLAG_SYNC] dropped irrelevant nexus escalator: {name[:50]}")
                continue
            seen_keys.add(key)
            # NEXUS combo/pattern signals → risk_escalators, not observed
            risk_escalators.append({"name": name, "priority": 1.5,
                                     "source": "nexus", "bucket": "combo_risk"})

        all_rfs = observed_rfs + risk_escalators + watch_for_flags
        result["red_flag_block"] = {
            "red_flags"        : all_rfs,
            "observed_red_flags": observed_rfs,
            "risk_escalators"  : risk_escalators,
            "watch_for_flags"  : watch_for_flags,
            "red_flag_count"   : len(all_rfs),
            "observed_count"   : len(observed_rfs),
            "highest_priority" : max((r.get("priority", 1.0) for r in all_rfs), default=0.0),
        }

        # Triage upgrade: ONLY observed_red_flags justify a triage increase
        # risk_escalators → add caution wording only
        # watch_for_flags → UI display only
        from nexus_runner import _TRIAGE_ORDER
        current_level = (result.get("triage") or {}).get("level", "MODERATE").upper()
        if observed_rfs and _TRIAGE_ORDER.get(current_level, 1) < _TRIAGE_ORDER.get(
                observed_rfs[0].get("triage_floor", "MODERATE").upper(), 2):
            new_level = observed_rfs[0].get("triage_floor", "PROMPT").upper()
            result.setdefault("triage", {})["level"] = new_level
            print(f"[RED_FLAG_SYNC] triage → {new_level} (observed red flag: "
                  f"{observed_rfs[0].get('name','')})")
        elif risk_escalators and not observed_rfs:
            print(f"[RED_FLAG_SYNC] {len(risk_escalators)} risk escalators — "
                  f"triage unchanged, caution wording added")

        # Log with full attribution — reviewer requested explicit source tracing
        def _rf_desc(r):
            return (f"{r.get('name') or r.get('red_flag','?')[:40]}"
                    f"[{r.get('source','?')}]")

        print(f"[RED_FLAG_SYNC] A={len(observed_rfs)} observed "
              f"B={len(risk_escalators)} escalators "
              f"C={len(watch_for_flags)} watch_for")
        if observed_rfs:
            print(f"  [A-observed]   " + ", ".join(_rf_desc(r) for r in observed_rfs))
        if risk_escalators:
            print(f"  [B-escalators] " + ", ".join(_rf_desc(r) for r in risk_escalators[:3]))
        if watch_for_flags:
            print(f"  [C-watch_for]  " + ", ".join(
                r.get('watch_for', r.get('name','?'))[:40] for r in watch_for_flags))

        try:
            get_learner().feedback(result)
        except Exception:
            pass
    except Exception as e:
        print("[NEXUS] enhance failed:", e)
        # Fail-closed for high-risk: preserve pre-enhance freeze, suppress all reasoning
        if _pre_hr_active:
            print(f"[NEXUS] HIGH-RISK CASE: enhance crashed but freeze is preserved — "
                  f"'{_pre_hr_dominant}' shell active")
            # Ensure freeze state is not lost due to crash
            result["_high_risk_freeze"]  = True
            result["nexus_otc_hints"]    = []
            result["nexus_etiology"]     = {
                "etiology": "uncertain", "confidence": 0.0,
                "scores": {"viral": 0.0, "bacterial": 0.0, "non_infectious": 0.0},
                "_status": "suppressed_by_high_risk_freeze",
            }
            result["nexus_diagnoses_display"] = []

    # 3.6. Anatomy trace
    try:
        result = traced_bridge.enhance_with_trace(result, user_input=user_input)
        print("[ANATOMY] organs:", result.get("anatomy_affected_organs", [])[:3],
              "spread:", len(result.get("anatomy_spread", [])),
              "trace_steps:", result.get("nexus_trace", {}).get("total_steps", 0))
        if result.get("nexus_trace"):
            print_trace(result["nexus_trace"], verbose=False)
    except Exception as e:
        print("[ANATOMY] enhance failed (non-blocking):", e)

    # 3.7. Save trace
    try:
        if result.get("nexus_trace") and result.get("case_id"):
            os.makedirs("logs", exist_ok=True)
            trace_record = {
                "case_id":    result["case_id"],
                "nexus_trace":result["nexus_trace"],
                "ts":         result.get("nexus_trace", {}).get("input", {}).get("user_text", ""),
            }
            with open("logs/nexus_traces.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(trace_record, ensure_ascii=False, default=str) + "\n")
    except Exception as e:
        print("[NEXUS-TRACE] save failed:", e)

    result = enforce_medication_gate(result, safe_mode)

    # 4. FollowupState — build next question
    symptoms = result.get("symptoms") or []
    state.add_symptoms(symptoms)
    new_qs = state.build_followup_questions(symptoms, max_new=5)
    state.add_questions(new_qs)

    force_finalize = (
        (state.last_slot == "any_other_symptoms" and
         user_input.lower().strip() in {"no", "nope", "none", "no other symptoms"})
        or user_denies_other_symptoms(user_input)
    )

    next_q = None
    if not force_finalize:
        nxt = state.get_next_question()
        if isinstance(nxt, dict):
            next_q = nxt.get("question")
        elif isinstance(nxt, str):
            next_q = nxt

    save_followup_state(state)

    # 5. Pill card handling
    pill_cards = []
    pack = result.get("pill_pack", {}) or {}
    for item in (pack.get("recommended") or []):
        if item.get("card"):
            pill_cards.append(item["card"])
        else:
            p = item.get("pill") or {}
            pill_cards.append({
                "name":     p.get("english_name") or p.get("name") or "OTC option",
                "why":      item.get("reason") or "",
                "warnings": item.get("warnings") or item.get("contraindications") or [],
            })

    if not pill_cards:
        result["pills"]     = ""
        result["pill_pack"] = {"recommended": []}

    # 5.5. Final-answer mode: suppress questions when user provided explicit symptoms
    # If the user selected symptoms via checkbox (selected_symptoms), we have enough
    # information to give a direct answer — don't ask more questions.
    _has_explicit_symptoms = len(selected_symptoms) >= 1
    if _has_explicit_symptoms or force_finalize:
        # Strip question fields so generate_response produces a final answer
        result["nexus_suggested_questions"] = []
        result["followup_questions"] = []
        result["questions"] = []

    # 6. Deterministic patient-facing response
    best_answer = generate_response(result, user_input)
    best_answer, removed = sanitize_answer(best_answer, result, safe_mode)
    result["_safety_removed"] = removed
    result["_safety_event"]   = build_safety_event(result, safe_mode)

    collected_reply = {"text": ""}

    # G.5: build a compact reasoning payload to send AFTER the text stream.
    # Frontend splits on the sentinel, renders text as markdown, and renders
    # this JSON as the step-by-step reasoning panel (NEXUS is not a black box).
    _reasoning_payload = {
        "input_symptoms":   result.get("symptoms", []),
        "context":          patient_context,
        "thinking":         result.get("nexus_thinking", []),
        "detected_systems": result.get("nexus_detected_systems", []),
        "diagnoses": [
            {
                "disease":          d.get("disease"),
                "score":            d.get("score"),
                "reasoning_trace":  d.get("reasoning_trace", {}),
                "_g5_modifiers":    d.get("_g5_modifiers", {}),
                "affected_organs":  _extract_organs_from_trace(d.get("reasoning_trace", {})),
            }
            for d in (result.get("nexus_diagnoses", []) or [])[:5]
        ],
        "lab_recommendations":       result.get("lab_recommendations", {}),
        "treatment_recommendations": result.get("treatment_recommendations", {}),
        "suggested_questions":       result.get("nexus_suggested_questions", []),
    }

    REASONING_SENTINEL = "\n\u241E\u241ENEXUS_REASONING\u241E\u241E\n"  # ␞␞ markers — won't appear in normal text

    def generate():
        chunk_size = 24
        for i in range(0, len(best_answer), chunk_size):
            chunk = best_answer[i:i + chunk_size]
            collected_reply["text"] += chunk
            yield chunk
        # After the human-readable answer, emit the sentinel + reasoning JSON.
        try:
            yield REASONING_SENTINEL
            yield json.dumps(_reasoning_payload, ensure_ascii=False)
        except Exception as _e:
            print(f"[chat_stream] reasoning payload skipped: {_e}")

    @after_this_request
    def save(response):
        hist = load_history_server(get_sid())
        hist.append({
            "user":           user_input,
            "ai":             collected_reply["text"],
            "safety":         result.get("_safety_event", {}),
            "safety_removed": result.get("_safety_removed", []),
        })
        save_history_server(get_sid(), hist)
        return response

    return Response(generate(), mimetype="text/plain")


@app.post("/reset_followup")
def reset_followup():
    p = _followup_path(get_sid())
    if os.path.exists(p):
        os.remove(p)
    return {"status": "reset"}, 200


@app.post("/save_history")
def save_history():
    data = request.get_json() or {}
    user = (data.get("user") or "").strip()
    ai   = (data.get("ai") or "").strip()
    if not user or not ai:
        return {"status": "ignored"}, 200

    history = session.get("history")
    if not isinstance(history, list):
        history = []

    history.append({"user": user, "time": datetime.now().isoformat(), "ai": ai})
    session.modified = True

    os.makedirs("memory/logs", exist_ok=True)
    try:
        with open("memory/logs/session_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("[History] log write failed:", e)

    os.makedirs("memory/interactions", exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    try:
        with open(f"memory/interactions/{ts}.json", "w", encoding="utf-8") as f:
            json.dump({"timestamp": ts, "user": user, "ai": ai},
                      f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("[History] interaction write failed:", e)

    print(f"[History] {len(history)} items saved")
    return {"status": "saved"}, 200


# ══════════════════════════════════════════════════════════════════════════════
#   ANATOMY API  (3-D viewer)
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/anatomy")
def api_anatomy():
    data     = request.get_json(silent=True) or {}
    symptoms = data.get("symptoms", [])
    if not symptoms:
        return jsonify({"affected_organs": [], "organ_positions": {}, "organ_systems": {}})

    # Confidence thresholds for KG-derived organ localization. Previously
    # referenced an undefined `_cfg_anat` (NameError → endpoint always crashed).
    _LOCALIZE_MIN = 0.55
    _DISEASE_MIN  = 0.65

    try:
        out             = nexus_instance.reason(symptoms)
        detected_systems = out.get("detected_systems", [])

        affected = set()
        if _anatomy_kg:
            for sym in symptoms:
                s_norm = str(sym).strip().lower().replace(" ", "_")
                for t in _anatomy_kg.query(s_norm, "localizes_to"):
                    if t.confidence >= _LOCALIZE_MIN and t.object in anatomy_bridge.atlas.organs:
                        affected.add(t.object)
            for dx in out.get("diagnoses", [])[:3]:
                d_norm = dx["disease"].strip().lower().replace(" ", "_").replace("-", "_")
                for t in _anatomy_kg.query(d_norm, "affects_organ"):
                    if t.confidence >= _DISEASE_MIN and t.object in anatomy_bridge.atlas.organs:
                        affected.add(t.object)

        # Fallback: if the KG produced nothing (or isn't loaded), derive
        # affected organs from the diagnosis reasoning traces — the same
        # source the chat reasoning panel uses, so the two stay consistent.
        if not affected:
            for dx in out.get("diagnoses", [])[:5]:
                for org in _extract_organs_from_trace(dx.get("reasoning_trace", {})):
                    if org in anatomy_bridge.atlas.organs:
                        affected.add(org)

        positions, systems = {}, {}
        for org_name in affected:
            org = anatomy_bridge.atlas.organs.get(org_name)
            if not org:
                continue
            # pos_3d is the numeric coordinate tuple; `position` is a text label
            # ("midline"/"left"), which is NOT plottable. Use pos_3d.
            pos = getattr(org, "pos_3d", None)
            if pos:
                positions[org_name] = list(pos)
            systems[org_name] = getattr(org, "system", "")

        spread = []
        for org_name in list(affected)[:3]:
            try:
                for p in anatomy_bridge.atlas.find_spread_paths(org_name, "infection", max_hops=3)[:3]:
                    spread.append({"from": org_name, "to": p["organ"],
                                    "path": p["path"], "hops": p["hops"]})
            except Exception:
                pass

        connections = []
        for org_name in affected:
            try:
                for conn in anatomy_bridge.atlas.get_connections_from(org_name):
                    if conn.conn_type in ("arterial", "venous", "portal"):
                        connections.append({"from": conn.source, "to": conn.target,
                                            "type": conn.conn_type})
            except Exception:
                pass

        return jsonify({
            "symptoms":          symptoms,
            "affected_organs":   sorted(affected),
            "organ_positions":   positions,
            "organ_systems":     systems,
            "spread_paths":      spread[:10],
            "vessel_connections":connections[:20],
            "detected_systems":  detected_systems,
            "nexus_diagnoses":   [{"disease": d["disease"], "score": d["score"]}
                                  for d in out.get("diagnoses", [])[:5]],
        })
    except Exception as e:
        print(f"[ANATOMY_API] error: {e}")
        return jsonify({"affected_organs": [], "organ_positions": {}, "error": str(e)})


@app.get("/api/anatomy/atlas")
def api_anatomy_atlas():
    try:
        organs = {
            name: {"name": name, "system": org.system, "region": org.region,
                   "pos": list(org.pos_3d) if org.pos_3d else [0, 0, 0],
                   "functions": org.functions}
            for name, org in anatomy_bridge.atlas.organs.items()
        }
        connections = [
            {"from": c.source, "to": c.target, "type": c.conn_type, "desc": c.desc}
            for c in anatomy_bridge.atlas.connections
        ]
        return jsonify({"organs": organs, "connections": connections,
                        "total_organs": len(organs),
                        "total_connections": len(connections)})
    except Exception as e:
        return jsonify({"error": str(e)})


@app.get("/anatomy")
def anatomy_page():
    # templates/anatomy3d.html doesn't exist; the working 3D viewer is the
    # static file served at /anatomy3d. Redirect there instead of 500-ing.
    from flask import redirect
    return redirect("/anatomy3d")


@app.post("/api/anatomy/spread")
def api_pathogen_spread():
    data     = request.get_json(silent=True) or {}
    organs   = data.get("organs", [])
    pathogen = data.get("pathogen", "infection")
    max_hops = min(int(data.get("max_hops", 7)), 10)
    if not organs:
        return jsonify({"spread_paths": [], "error": "no organs provided"}), 400
    try:
        # nexus_engine.pathogen_tracker doesn't exist; use the atlas's own
        # spread-path search, which is what the rest of the app relies on.
        spread_paths = []
        for origin in organs:
            if origin not in anatomy_bridge.atlas.organs:
                continue
            for p in anatomy_bridge.atlas.find_spread_paths(
                    origin, pathogen, max_hops=max_hops):
                spread_paths.append({
                    "from": origin,
                    "to":   p.get("organ"),
                    "path": p.get("path"),
                    "hops": p.get("hops"),
                })
        return jsonify({"origin_organs": organs, "pathogen": pathogen,
                        "spread_paths": spread_paths[:30]})
    except Exception as e:
        return jsonify({"spread_paths": [], "error": str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)