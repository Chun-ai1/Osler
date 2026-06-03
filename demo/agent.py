"""
agent.py — the neuro-symbolic orchestrator.

Flow (LLM never decides; it only parses language and explains):
  free-text case ──parse──▶ PatientProfile + indication
                              │
        case_targets.targets_for(indication)
                              ▼
        reasoning_engine.recommend()  ──▶ ranked drugs + reasoning chain   (SYMBOLIC)
                              │
        disease_world.for_indication()──▶ disease perturbations + symptoms  (LOCAL WIKI)
                              │
        (optional) clinical_data.build_entry() ─▶ live openFDA label/FAERS  (EXTERNAL WIKI)
                              ▼
                    build_graph() ──▶ mind-map nodes + edges

Returns one bundle the frontend renders: profile, result, disease_model, graph.
"""
from __future__ import annotations
from typing import Dict, List, Optional

import reasoning_engine as RE
import case_targets
import case_parser
import disease_world
from patient_profile import PatientProfile


def build_patient(d: Dict) -> PatientProfile:
    vit = d.get("vitals") or {}
    labs = d.get("labs") or {}
    return PatientProfile(
        age=d.get("age"), sex=d.get("sex") or None, weight_kg=d.get("weight_kg"),
        egfr=d.get("egfr"), hepatic_status=d.get("hepatic_status") or "normal",
        pregnancy=bool(d.get("pregnancy")),
        allergies=[a.lower() for a in d.get("allergies", [])],
        current_medications=[m.lower() for m in d.get("current_medications", [])],
        conditions=[c.lower() for c in d.get("conditions", [])],
        symptoms=[s.lower() for s in d.get("symptoms", [])],
        vitals={k: float(v) for k, v in vit.items() if v is not None},
        labs={k: float(v) for k, v in labs.items() if v is not None})


def analyze_stream(fields: Dict, drugs_pkpd: Dict, clinical: Dict,
                   text: Optional[str] = None, api_key: Optional[str] = None,
                   provider: Optional[str] = None, use_openfda: bool = False):
    """
    Generator that yields the agent's workflow one step at a time (Claude-Code style):
      {"type":"step", icon, title, detail, running?}   # running=True → still working
      {"type":"error", "error": ...}
      {"type":"final", "bundle": {...}}                 # full result at the end
    """
    import time
    clinical = dict(clinical)

    # ── 1. parse ───────────────────────────────────────────────
    parser_used = "form"
    if text and text.strip():
        parsed = case_parser.parse(text, api_key, provider)
        parser_used = parsed.pop("_parser", "rules")
        merged = dict(parsed)
        for k, v in fields.items():  # explicit form fields override parsed
            if v not in (None, "", [], {}):
                merged[k] = v
        fields = merged
    src = "LLM (" + (provider or "openai") + ")" if parser_used == "llm" else \
          ("rules (regex)" if parser_used == "rules" else "structured form")
    _kv = [f"{k}={fields.get(k)}" for k in ("age", "sex", "egfr") if fields.get(k) is not None]
    yield {"type": "step", "icon": "🧩", "title": f"Parse case · {src}",
           "detail": f"indication={fields.get('indication') or '—'}; " + ", ".join(_kv)}

    # ── 2. targets ─────────────────────────────────────────────
    indication = fields.get("indication") or ""
    targets, canon, scenario = case_targets.targets_for(indication)
    if not targets:
        yield {"type": "error", "error": f"Could not map indication “{indication}” to treatment targets. "
               "Known: " + ", ".join(i["label"] for i in case_targets.list_indications())}
        return
    yield {"type": "step", "icon": "🎯", "title": f"Map indication → {len(targets)} treatment targets",
           "detail": ", ".join(f"{t['variable'].replace('_', ' ')} "
                               f"{'↓' if t['direction'] == 'low' else '↑'}" for t in targets)}

    # ── 3. symbolic recommendation ─────────────────────────────
    patient = build_patient(fields)

    def _recommend():
        t0 = time.perf_counter()
        r = RE.recommend(patient, targets, canon, drugs_pkpd, clinical, scenario=scenario)
        r["indication_label"] = next((i["label"] for i in case_targets.list_indications()
                                     if i["value"] == canon), canon)
        r["mechanism_only"] = not clinical
        return r, (time.perf_counter() - t0) * 1000

    result, ms = _recommend()
    yield {"type": "step", "icon": "⚙️",
           "title": f"Symbolic engine ranked {len(result['candidates'])} drugs · {ms:.1f} ms · 0 LLM calls",
           "detail": " > ".join(f"{c['drug']}({c['safety']['decision']})" for c in result["candidates"])}

    # ── 4. disease world-model (local wiki) ────────────────────
    disease_model = disease_world.for_indication(canon)
    yield {"type": "step", "icon": "📖", "title": f"Disease world-model · {disease_model['source']}",
           "detail": f"{disease_model['disease']}"
                     + (f" (data/{disease_model['source_file']})" if disease_model.get('source_file') else "")
                     + f" · {len(disease_model['perturbations'])} perturbations"}

    # ── 5. optional live openFDA ───────────────────────────────
    openfda_loaded = None
    if use_openfda:
        yield {"type": "step", "icon": "🌐", "title": "openFDA · fetching live labels…",
               "detail": "querying api.fda.gov (up to a few seconds)", "running": True}
        names = [c["drug"] for c in result["candidates"]][:4]
        live = enrich_openfda(names, canon, api_key=api_key)
        if live:
            clinical.update(live)
            result, _ = _recommend()
            openfda_loaded = sorted(live.keys())
            yield {"type": "step", "icon": "🌐", "title": f"openFDA · fetched {len(live)} live label(s)",
                   "detail": ", ".join(openfda_loaded)}
        else:
            yield {"type": "step", "icon": "🌐", "title": "openFDA · no live labels",
                   "detail": "offline or no match; using bundled demo labels"}

    graph = build_graph(result, disease_model, targets)
    yield {"type": "final", "bundle": {
        "indication": canon, "parser": parser_used, "fields": fields,
        "result": result, "disease_model": disease_model, "graph": graph,
        "openfda_loaded": openfda_loaded}}


def analyze(fields: Dict, drugs_pkpd: Dict, clinical: Dict,
            text: Optional[str] = None, api_key: Optional[str] = None,
            provider: Optional[str] = None, use_openfda: bool = False) -> Dict:
    """Non-streaming convenience wrapper: collects analyze_stream into one bundle."""
    trace, bundle = [], None
    for ev in analyze_stream(fields, drugs_pkpd, clinical, text, api_key, provider, use_openfda):
        if ev["type"] == "error":
            return {"error": ev["error"]}
        if ev["type"] == "step" and not ev.get("running"):
            trace.append({"icon": ev["icon"], "title": ev["title"], "detail": ev.get("detail", "")})
        if ev["type"] == "final":
            bundle = ev["bundle"]
    bundle["trace"] = trace
    return bundle


def enrich_openfda(drug_names: List[str], indication: str,
                   api_key: Optional[str] = None) -> Dict:
    """Best-effort live openFDA label fetch. Returns {drug: entry}. Never raises."""
    out = {}
    try:
        import clinical_data
    except Exception:
        return out
    for name in drug_names:
        try:
            entry = clinical_data.build_entry(name, api_key=api_key, indication=indication)
            if entry:
                out[name] = entry
        except Exception:
            continue
    return out


# ── mind-map graph ─────────────────────────────────────────────────────────
def _arrow(direction: str) -> str:
    return "↓" if direction == "low" else "↑"


def build_graph(result: Dict, disease_model: Dict, targets: List[Dict]) -> Dict:
    """Nodes/edges for vis-network. Groups drive color/shape on the client."""
    nodes: List[Dict] = []
    edges: List[Dict] = []
    p = result["patient"]

    pat_id = "pat"
    nodes.append({"id": pat_id, "label": f"Patient\n{p.get('age') or '?'} y", "group": "patient"})
    ind_id = "ind"
    nodes.append({"id": ind_id, "label": result.get("indication_label", result.get("indication")),
                  "group": "indication"})
    edges.append({"from": pat_id, "to": ind_id, "label": "presents with"})

    # patient warning flags
    for f in (p.get("flags") or []):
        fid = f"flag:{f}"
        nodes.append({"id": fid, "label": f, "group": "flag"})
        edges.append({"from": pat_id, "to": fid, "dashes": True})

    # disease (local wiki) + its perturbations + symptoms
    dz_id = "dz"
    nodes.append({"id": dz_id, "label": disease_model.get("disease"),
                  "group": "disease",
                  "title": (disease_model.get("description") or "")[:160]})
    edges.append({"from": ind_id, "to": dz_id,
                  "label": "modeled as" + (" (inferred)" if disease_model.get("source") == "inferred" else "")})
    pert_var_to_node = {}
    for pp in disease_model.get("perturbations", []):
        nid = f"pert:{pp['variable']}"
        nodes.append({"id": nid, "label": f"{pp['variable'].replace('_', ' ')} {_arrow(pp['direction'])}",
                      "group": "pathology", "title": pp.get("cause", "")})
        edges.append({"from": dz_id, "to": nid, "label": "perturbs"})
        pert_var_to_node[pp["variable"]] = nid
    for i, s in enumerate(disease_model.get("symptoms", [])[:5]):
        sid = f"sym:{i}"
        nodes.append({"id": sid, "label": s, "group": "symptom"})
        edges.append({"from": dz_id, "to": sid, "label": "causes"})

    # treatment targets
    tgt_friendly_to_node = {}
    for t in targets:
        friendly = f"{t['variable'].replace('_', ' ')} {_arrow(t['direction'])}"
        tid = f"tgt:{t['variable']}"
        nodes.append({"id": tid, "label": friendly, "group": "target"})
        edges.append({"from": ind_id, "to": tid, "label": "treatment goal"})
        tgt_friendly_to_node[friendly] = tid
        # shared-coordinate link: same variable that the disease perturbed
        if t["variable"] in pert_var_to_node:
            edges.append({"from": pert_var_to_node[t["variable"]], "to": tid,
                          "label": "cancels", "dashes": True, "color": "#0D9488"})

    # drugs -> the targets they hit, colored by safety decision
    for c in result.get("candidates", []):
        did = f"drug:{c['drug']}"
        nodes.append({"id": did, "label": c["drug"], "group": f"drug-{c['safety']['decision']}",
                      "title": c.get("final_answer", "")})
        hit_any = False
        for m in c.get("matched_targets", []):
            tnode = tgt_friendly_to_node.get(m["target"])
            if tnode:
                edges.append({"from": did, "to": tnode, "label": m["effect_type"]})
                hit_any = True
        if not hit_any:  # keep the node connected
            edges.append({"from": did, "to": ind_id, "dashes": True})

    return {"nodes": nodes, "edges": edges}


if __name__ == "__main__":
    import json
    from pathlib import Path
    drugs = RE._load("drugs_pkpd.json")["drugs"]
    try:
        _cd = Path(__file__).resolve().parent / "demo_clinical_data.json"
        clinical = json.loads(_cd.read_text(encoding="utf-8"))["drugs"]
    except Exception:
        clinical = {}
    text = ("64M crushing chest pain, diaphoresis, acute coronary syndrome. "
            "BP 88/54, HR 112, eGFR 72.")
    out = analyze({}, drugs, clinical, text=text)
    print("parser:", out["parser"], "| indication:", out["indication"])
    print("drugs:", [c["drug"] + "/" + c["safety"]["decision"] for c in out["result"]["candidates"]])
    print("disease:", out["disease_model"]["disease"],
          "| perturbations:", len(out["disease_model"]["perturbations"]))
    print("graph: nodes", len(out["graph"]["nodes"]), "edges", len(out["graph"]["edges"]))
