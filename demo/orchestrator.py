"""
orchestrator.py — LLM-driven (tool-calling) version of the agent workflow.

Instead of a fixed pipeline (parse → targets → recommend → disease → openFDA), the LLM
decides WHICH tool to call and in WHAT order, via OpenAI function-calling. The tools
wrap the SAME symbolic functions, so the engine still makes every clinical decision —
the LLM only orchestrates the flow. Each tool call is streamed as a workflow step.

Falls back to the deterministic agent.analyze_stream when:
  • no OpenAI key (tool-calling needs one), or provider != openai, or any error.

Yields the same event shape as agent.analyze_stream:
  {"type":"step", icon, title, detail, running?} / {"type":"error",...} / {"type":"final","bundle":...}
"""
from __future__ import annotations
import json
import time
from typing import Dict, Optional

import reasoning_engine as RE
import case_parser
import case_targets
import disease_world
import agent

_TOOLS = [
    {"type": "function", "function": {
        "name": "parse_case",
        "description": "Extract structured patient fields and the clinical indication from the "
                       "free-text case. Must be called first.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "recommend_drugs",
        "description": "Run the symbolic pharmacology engine to rank candidate drugs for the parsed "
                       "case (maps the indication to physiological targets, scores drugs, applies "
                       "safety gates). Call after parse_case.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "get_disease_model",
        "description": "Load the local disease world-model (perturbations + derived symptoms) for the "
                       "indication, from the organ knowledge base.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "fetch_openfda_labels",
        "description": "Fetch live FDA labels for the recommended drugs to refine the safety gate and "
                       "show validated doses. Optional; needs internet.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "finish",
        "description": "Call when the analysis is complete and drugs have been recommended.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
]

_SYS = (
    "You are a clinical drug-recommendation agent. You DECIDE the workflow by calling tools; the "
    "tools run a symbolic engine that makes the actual clinical decisions. Plan sensibly: parse the "
    "case first, then recommend drugs, then load the disease model; fetch openFDA labels if useful; "
    "then finish. Do not answer in prose — only call tools. Always recommend_drugs before finish."
)


class _Run:
    def __init__(self, text, drugs_pkpd, clinical, api_key, provider, use_openfda):
        self.text, self.drugs, self.clinical = text, drugs_pkpd, dict(clinical)
        self.api_key, self.provider, self.use_openfda = api_key, provider, use_openfda
        self.fields = self.targets = self.canon = self.scenario = None
        self.result = self.disease_model = self.openfda_loaded = None

    # each tool returns (summary_for_llm, step_event)
    def parse_case(self):
        parsed = case_parser.parse(self.text, self.api_key, self.provider)
        self.parser = parsed.pop("_parser", "rules")
        self.fields = parsed
        ind = self.fields.get("indication") or "—"
        src = {"llm": "LLM", "rules": "rules", "form": "form"}.get(self.parser, self.parser)
        return (f"parsed: indication={ind}, age={self.fields.get('age')}, "
                f"allergies={self.fields.get('allergies')}",
                {"icon": "🧩", "title": f"parse_case · {src}", "detail": f"indication={ind}"})

    def recommend_drugs(self):
        if not self.fields:
            return "error: call parse_case first", {"icon": "⚠️", "title": "recommend_drugs (skipped)",
                                                    "detail": "parse_case not run yet"}
        self.targets, self.canon, self.scenario = case_targets.targets_for(self.fields.get("indication") or "")
        if not self.targets:
            return (f"error: unknown indication '{self.fields.get('indication')}'",
                    {"icon": "⚠️", "title": "recommend_drugs", "detail": "indication not mapped"})
        patient = agent.build_patient(self.fields)
        t0 = time.perf_counter()
        self.result = RE.recommend(patient, self.targets, self.canon, self.drugs, self.clinical,
                                   scenario=self.scenario)
        ms = (time.perf_counter() - t0) * 1000
        self.result["indication_label"] = next((i["label"] for i in case_targets.list_indications()
                                                if i["value"] == self.canon), self.canon)
        self.result["mechanism_only"] = not self.clinical
        agent._attach_rationale(self.result, self.drugs)
        names = [c["drug"] for c in self.result["candidates"]]
        return (f"ranked {len(names)} drugs: " + ", ".join(
                    f"{c['drug']}({c['safety']['decision']})" for c in self.result["candidates"]),
                {"icon": "⚙️", "title": f"recommend_drugs · {len(names)} drugs · {ms:.1f} ms",
                 "detail": " > ".join(names)})

    def get_disease_model(self):
        canon = self.canon or case_targets.resolve(self.fields.get("indication") or "") if self.fields else None
        if not canon:
            return "error: no indication yet", {"icon": "⚠️", "title": "get_disease_model",
                                                "detail": "no indication"}
        self.disease_model = disease_world.for_indication(canon)
        dm = self.disease_model
        return (f"disease {dm['disease']} ({dm['source']}): "
                + ", ".join(p["variable"] + ("↑" if p["direction"] == "high" else "↓")
                            for p in dm["perturbations"]),
                {"icon": "📖", "title": f"get_disease_model · {dm['source']}",
                 "detail": f"{dm['disease']} · {len(dm['perturbations'])} perturbations"})

    def fetch_openfda_labels(self):
        if not self.result:
            return "error: recommend_drugs first", {"icon": "⚠️", "title": "fetch_openfda_labels",
                                                    "detail": "no drugs yet"}
        names = [c["drug"] for c in self.result["candidates"]][:4]
        live = agent.enrich_openfda(names, self.canon, api_key=self.api_key)
        if live:
            self.clinical.update(live)
            self.recommend_drugs()  # re-run with live labels
            self.openfda_loaded = sorted(live.keys())
            return (f"fetched {len(live)} labels: " + ", ".join(self.openfda_loaded),
                    {"icon": "🌐", "title": f"fetch_openfda_labels · {len(live)} live",
                     "detail": ", ".join(self.openfda_loaded)})
        return ("no live labels (offline/no match)",
                {"icon": "🌐", "title": "fetch_openfda_labels · none", "detail": "using demo labels"})


def _impls(run):
    return {n: getattr(run, n) for n in ("parse_case", "recommend_drugs",
                                         "get_disease_model", "fetch_openfda_labels")}


def _exec_tool(impls, name):
    """Returns (summary_for_llm, step_event_or_None, is_finish)."""
    if name == "finish":
        return "done", None, True
    fn = impls.get(name)
    if not fn:
        return "unknown tool", {"icon": "⚠️", "title": name, "detail": "unknown tool"}, False
    summary, step = fn()
    return summary, step, False


def _loop_openai(run, impls, text, key, use_openfda):
    import os
    from openai import OpenAI
    client = OpenAI(api_key=key)
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    hint = "Fetch openFDA labels too." if use_openfda else "Skip openFDA."
    messages = [{"role": "system", "content": _SYS},
                {"role": "user", "content": f"Case:\n{text}\n\n{hint} Recommend drugs."}]
    for _ in range(8):
        yield {"type": "step", "icon": "🧠", "title": "LLM deciding next step…",
               "detail": "OpenAI tool-calling", "running": True}
        msg = client.chat.completions.create(model=model, messages=messages, tools=_TOOLS,
                                             tool_choice="auto", temperature=0).choices[0].message
        if not msg.tool_calls:
            break
        messages.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [{"id": tc.id, "type": "function",
                                         "function": {"name": tc.function.name,
                                                      "arguments": tc.function.arguments or "{}"}}
                                        for tc in msg.tool_calls]})
        stop = False
        for tc in msg.tool_calls:
            summary, step, is_finish = _exec_tool(impls, tc.function.name)
            if step:
                yield {"type": "step", **step}
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": summary})
            stop = stop or is_finish
        if stop:
            break


def _loop_gemini(run, impls, text, key, use_openfda):
    import os
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=key)
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    decls = [types.FunctionDeclaration(name=t["function"]["name"], description=t["function"]["description"],
                                       parameters=types.Schema(type="OBJECT", properties={}))
             for t in _TOOLS]
    config = types.GenerateContentConfig(system_instruction=_SYS,
                                         tools=[types.Tool(function_declarations=decls)], temperature=0)
    chat = client.chats.create(model=model, config=config)
    hint = "Fetch openFDA labels too." if use_openfda else "Skip openFDA."
    send = f"Case:\n{text}\n\n{hint} Recommend drugs."
    for _ in range(8):
        yield {"type": "step", "icon": "🧠", "title": "LLM deciding next step…",
               "detail": "Gemini tool-calling", "running": True}
        resp = chat.send_message(send)
        parts = (resp.candidates[0].content.parts if resp.candidates else []) or []
        fcs = [p.function_call for p in parts if getattr(p, "function_call", None)]
        if not fcs:
            break
        frs, stop = [], False
        for fc in fcs:
            summary, step, is_finish = _exec_tool(impls, fc.name)
            if step:
                yield {"type": "step", **step}
            frs.append(types.Part.from_function_response(name=fc.name, response={"result": summary}))
            stop = stop or is_finish
        if stop:
            break
        send = frs  # send the function responses back as the next turn


def analyze_stream_llm(fields, drugs_pkpd, clinical, text=None, api_key=None,
                       provider=None, use_openfda=False):
    """LLM tool-calling orchestration (OpenAI or Gemini). Falls back to deterministic on any issue."""
    prov = (provider or "openai").lower()
    loop = {"openai": _loop_openai, "gemini": _loop_gemini}.get(prov)
    if not api_key or loop is None:
        yield from agent.analyze_stream(fields, drugs_pkpd, clinical, text, api_key, provider, use_openfda)
        return

    run = _Run(text or "", drugs_pkpd, clinical, api_key, provider, use_openfda)
    impls = _impls(run)
    try:
        yield from loop(run, impls, text or "", api_key, use_openfda)
    except Exception as e:  # any LLM/tool failure → finish deterministically with what we have
        yield {"type": "step", "icon": "⚠️", "title": "LLM orchestration error — completing deterministically",
               "detail": f"{type(e).__name__}: {e}"}

    # Safety net: ensure a recommendation exists even if the LLM misbehaved.
    if not run.result:
        if not run.fields:
            run.parse_case()
        run.recommend_drugs()
    if run.result is None:
        yield {"type": "error", "error": "Could not produce a recommendation (check the indication)."}
        return
    if run.disease_model is None:
        run.get_disease_model()
    graph = agent.build_graph(run.result, run.disease_model, run.targets)
    yield {"type": "final", "bundle": {
        "indication": run.canon, "parser": getattr(run, "parser", "llm"),
        "fields": run.fields, "result": run.result, "disease_model": run.disease_model,
        "graph": graph, "openfda_loaded": run.openfda_loaded, "orchestration": prov}}
