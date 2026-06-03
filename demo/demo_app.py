"""
demo_app.py — Drug-recommendation agent demo (for clinicians).

  import a case (free text or form)  ─▶  agent parses + symbolic engine recommends
  ─▶ mind-map reasoning graph (case ▸ disease ▸ targets ▸ drugs)  ─▶ chat to ask why.

The symbolic engine (reasoning_engine) makes every recommendation; the LLM only
parses the case and explains the result. Knowledge sources: drugs_pkpd.json (drug
deltas), the organ JSONs via disease_world (local disease "wiki"), and optional
live openFDA labels via agent.enrich_openfda.

Run:  py -m pip install -r requirements.txt
      py demo_app.py   →  http://127.0.0.1:5000
Chat/agent-parse use an API key entered in the top-right box (or OPENAI_API_KEY env).
Without a key it still runs: rule-based parsing + reasoning graph (chat degraded).
"""
from __future__ import annotations
import sys
import json
from pathlib import Path

# This demo (demo/) reuses the pharmacology engine in engine/ and data in data/.
_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT / "engine", _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from flask import Flask, request, jsonify, Response

import reasoning_engine as RE
import case_targets
import agent
import llm_client

_HERE = Path(__file__).parent
app = Flask(__name__)

DRUGS_PKPD = RE._load("drugs_pkpd.json")["drugs"]


def _load_optional(name: str) -> dict:
    p = _HERE / name
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8", errors="replace")).get("drugs", {})
    except Exception:
        return {}


CLINICAL = _load_optional("demo_clinical_data.json")
SAMPLE_CASES = json.loads((_HERE / "sample_cases.json").read_text(encoding="utf-8"))["cases"]

# patient_id -> last analyze bundle, used to ground chat.
_CACHE: dict[str, dict] = {}


@app.route("/")
def index():
    return Response((_HERE / "case_demo.html").read_text(encoding="utf-8"), mimetype="text/html")


@app.route("/api/cases")
def api_cases():
    return jsonify({"cases": SAMPLE_CASES, "indications": case_targets.list_indications(),
                    "env_llm": llm_client.available()})


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    d = request.get_json(force=True) or {}
    out = agent.analyze(d.get("fields") or {}, DRUGS_PKPD, dict(CLINICAL),
                        text=d.get("text") or "", api_key=(d.get("api_key") or "").strip() or None,
                        provider=d.get("provider") or None, use_openfda=bool(d.get("use_openfda")))
    if "error" in out:
        return jsonify(out), 400
    pid = d.get("patient_id") or out["indication"]
    out["patient_id"] = pid
    _CACHE[pid] = out
    return jsonify(out)


@app.route("/api/analyze_stream", methods=["POST"])
def api_analyze_stream():
    """NDJSON stream: one JSON object per line, one per workflow step, ending with
    {type:final, bundle}. Lets the UI render the agent's steps live (Claude-Code style)."""
    d = request.get_json(force=True) or {}
    pid = d.get("patient_id") or "p"
    api_key = (d.get("api_key") or "").strip() or None
    provider = d.get("provider") or None
    # deterministic fixed pipeline, or LLM tool-calling orchestration
    if d.get("orchestrator") == "llm":
        import orchestrator
        stream_fn = orchestrator.analyze_stream_llm
    else:
        stream_fn = agent.analyze_stream

    def gen():
        trace = []
        try:
            for ev in stream_fn(
                    d.get("fields") or {}, DRUGS_PKPD, dict(CLINICAL),
                    text=d.get("text") or "", api_key=api_key,
                    provider=provider, use_openfda=bool(d.get("use_openfda"))):
                if ev["type"] == "step" and not ev.get("running"):
                    trace.append({"icon": ev["icon"], "title": ev["title"], "detail": ev.get("detail", "")})
                if ev["type"] == "final":
                    ev["bundle"]["trace"] = trace
                    ev["bundle"]["patient_id"] = pid
                    _CACHE[pid] = ev["bundle"]
                yield json.dumps(ev, ensure_ascii=False) + "\n"
        except Exception as e:  # never break the stream silently
            yield json.dumps({"type": "error", "error": f"{type(e).__name__}: {e}"}) + "\n"

    return Response(gen(), mimetype="application/x-ndjson")


_SYSTEM = (
    "You are the explanation assistant for a clinician-facing drug-recommendation system. "
    "Your ONLY job is to explain, in clear clinical language, the recommendation that the "
    "symbolic reasoning engine has ALREADY computed (provided below).\n"
    "Rules:\n"
    "1) Do not make independent medical decisions, change the ranking, or suggest drugs not "
    "in the candidate list.\n"
    "2) Explain only from the provided data: disease perturbations (the local world model), "
    "mechanism_chain, matched_targets, clinical_role, safety, final_answer.\n"
    "3) Never invent a dose; only cite dose.verbatim when present, and note a clinician must "
    "confirm it.\n"
    "4) If asked about a drug not in the list, say it was not matched and why "
    "(mechanism mismatch / not in the drug knowledge base).\n"
    "5) delta magnitudes are estimates; direction is more reliable. Decision support only — "
    "a licensed clinician makes the final call.\n"
    "Answer concisely, in the user's language, citing specific drugs and target variables."
)


def _ground(bundle: dict) -> str:
    res = bundle["result"]
    dm = bundle.get("disease_model", {})
    p = res["patient"]
    lines = [f"Indication: {res.get('indication_label', res.get('indication'))}",
             f"Treatment targets: {', '.join(res.get('target_states', []))}",
             f"Disease world-model ({dm.get('source')}): {dm.get('disease')}",
             "  perturbations: " + ", ".join(
                 f"{pp['variable']} {'↑' if pp['direction'] == 'high' else '↓'}"
                 for pp in dm.get("perturbations", [])),
             "  symptoms: " + ", ".join(dm.get("symptoms", [])),
             f"Patient: age={p.get('age')}, renal={p.get('renal_label')}, "
             f"allergies={p.get('allergies')}, meds={p.get('meds')}, flags={p.get('flags')}",
             f"Mode: {'mechanism-only' if res.get('mechanism_only') else 'mechanism + label safety gate'}",
             "", "Ranked candidates:"]
    for i, c in enumerate(res.get("candidates", []), 1):
        dose = c.get("dose", {})
        dose_txt = dose.get("verbatim") if dose.get("patient_specific_allowed") else None
        reasons = "; ".join(r.get("message", "") for r in c.get("safety", {}).get("reasons", []))
        lines.append(
            f"{i}. {c['drug']} [{c['clinical_role'].get('label')}] score={c.get('mechanism_score')}\n"
            f"   mechanism: {c.get('mechanism_chain')}\n"
            f"   matched: {[m['target'] + '/' + m['effect_type'] for m in c.get('matched_targets', [])]}\n"
            f"   safety: {c['safety'].get('decision')} ({reasons})\n"
            f"   dose: {dose_txt or 'not shown'}\n"
            f"   conclusion: {c.get('final_answer')}")
    return "\n".join(lines)


@app.route("/api/chat", methods=["POST"])
def api_chat():
    d = request.get_json(force=True) or {}
    pid = d.get("patient_id")
    messages = d.get("messages") or []
    api_key = (d.get("api_key") or "").strip() or None
    provider = d.get("provider") or None
    # Prefer grounding sent by the client (survives server restarts); fall back to cache.
    bundle = None
    if d.get("result"):
        bundle = {"result": d["result"], "disease_model": d.get("disease_model") or {}}
    elif pid in _CACHE:
        bundle = _CACHE[pid]
    if not bundle:
        return jsonify({"reply": "Please analyze a case first so I can explain its result.",
                        "ok": False})
    system = _SYSTEM + "\n\n=== CURRENT CASE RESULT ===\n" + _ground(bundle)
    ok, text = llm_client.chat(system, messages, api_key=api_key, provider=provider)
    return jsonify({"reply": text, "ok": ok})


if __name__ == "__main__":
    print(f"[demo] drugs={len(DRUGS_PKPD)} clinical={len(CLINICAL)} cases={len(SAMPLE_CASES)} "
          f"env_llm={'on' if llm_client.available() else 'off'}")
    app.run(debug=True, port=5000)
