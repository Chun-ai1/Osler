"""
NEXUS Reasoning Trace (rewritten for pure-mechanism architecture)
═══════════════════════════════════════════════════════════════
Records each step of NEXUS pure-mechanism reasoning + anatomy enrichment.

Input format: result dict produced by app.py /chat_stream containing:
  - nexus_diagnoses     (from nexus.reason())
  - nexus_thinking      (CASCADE_MATCH / PHYSIOLOGY_SIM / SPATIAL_3D / RANKING steps)
  - nexus_stats         (mechanism / spatial counts)
  - nexus_consistency   (self-consistency score)
  - symptoms / final_symptoms
  - anatomy_*           (filled by anatomy_bridge later)

Output: result['nexus_trace'] = {steps, summary, input, total_ms}
"""
from __future__ import annotations
import time
from typing import Dict, List, Any


class NexusTrace:
    def __init__(self):
        self.steps: List[Dict[str, Any]] = []
        self.start_time = time.time()
        self._step_counter = 0

    def step(self, phase: str, action: str, result: Any = None, detail: str = ""):
        self._step_counter += 1
        elapsed = (time.time() - self.start_time) * 1000
        entry = {
            "step":   self._step_counter,
            "phase":  phase,
            "action": action,
            "detail": detail,
            "ms":     round(elapsed, 2),
        }
        if result is not None:
            if isinstance(result, list):
                entry["result"] = result[:10] if len(result) > 10 else result
                entry["result_count"] = len(result)
            elif isinstance(result, dict):
                entry["result"] = {k: v for k, v in list(result.items())[:8]}
            else:
                entry["result"] = str(result)[:200]
        self.steps.append(entry)
        return entry

    def to_dict(self) -> Dict:
        total_ms = (time.time() - self.start_time) * 1000
        return {
            "total_steps": len(self.steps),
            "total_ms":    round(total_ms, 2),
            "steps":       self.steps,
        }


class NexusTracedBridge:
    """
    Wraps AnatomyBridge + NEXUS pure-mechanism reasoning with full trace output.

    The NEW reason() returns thinking steps:
      INPUT, CASCADE_MATCH, PHYSIOLOGY_SIM, SPATIAL_3D, RANKING

    This bridge unfolds those steps into a human-readable trace and adds
    anatomy enrichment (KG queries, spread paths, referred pain).
    """

    def __init__(self, bridge, nexus=None, anatomy_kg=None):
        self.bridge     = bridge
        self.atlas      = bridge.atlas
        self.nexus      = nexus or bridge.nexus
        self.anatomy_kg = anatomy_kg

    def enhance_with_trace(self, result: dict, user_input: str = "") -> dict:
        trace = NexusTrace()
        symptoms = result.get("symptoms", result.get("final_symptoms", []))

        # ── Phase 1: INPUT ──
        trace.step("INPUT", "Received symptoms", symptoms)
        trace.step("INPUT", "User text",
                   detail=(user_input[:200] if user_input else "(empty)"))

        # ── Phase 2: NEXUS PURE-MECHANISM REASONING ──
        if self.nexus:
            n_diseases = len(getattr(self.nexus, 'diseases', {}))
            trace.step("NEXUS", "Pure-mechanism reasoning loaded",
                       detail=f"{n_diseases} disease supplementals "
                              f"(triage/treatment/ICD only — cascade is the reasoning source)")

        # Show new reason() thinking steps (CASCADE_MATCH / PHYSIOLOGY_SIM / SPATIAL_3D / RANKING)
        nexus_thinking = result.get("nexus_thinking", [])
        for step in nexus_thinking:
            if isinstance(step, dict):
                step_name = step.get("step", "")
                detail    = step.get("detail", "")
                extra     = (step.get("top_3") or step.get("result")
                             or step.get("interpretation") or step.get("cascades"))
                trace.step(f"NEXUS_{step_name}", detail, result=extra)

        # NEXUS diagnoses (cascade-derived)
        nexus_dx = result.get("nexus_diagnoses", [])
        if nexus_dx:
            trace.step("NEXUS_DX", "Cascade-derived diagnoses (ranked)",
                       result=[{"disease": d.get("disease", ""),
                                "score":   d.get("score", 0),
                                "mechanism": d.get("_mechanism", "")}
                               for d in nexus_dx[:5]],
                       detail=f"{len(nexus_dx)} candidates")

        # Self-consistency
        consistency = result.get("nexus_consistency", {})
        if isinstance(consistency, dict) and consistency:
            score = consistency.get("consistency_score", 0)
            trace.step("NEXUS_CONSISTENCY",
                       f"Self-consistency: {score:.0%}",
                       result={
                           "score":            score,
                           "detected_systems": consistency.get("detected_systems", []),
                           "disease_systems":  consistency.get("disease_systems", []),
                           "agreed_systems":   consistency.get("agreed_systems", []),
                           "mechs_direct":     consistency.get("mechs_direct", 0),
                           "reliable":         consistency.get("reliable", False),
                       })

        # ── Phase 3: ORGAN IDENTIFICATION (anatomy KG) ──
        trace.step("ANATOMY", "Starting organ identification",
                   detail="Querying anatomy KG for symptom→organ and disease→organ")

        affected_organs = []
        seen = set()
        kg = self.anatomy_kg

        # Disease → organ (from KG)
        nexus_diseases = result.get("nexus_diagnoses", [])
        for d_info in nexus_diseases[:5]:
            d_name = d_info.get("disease", "") if isinstance(d_info, dict) else str(d_info)
            d_norm = d_name.strip().lower().replace(" ", "_").replace("-", "_")
            if not d_norm or not kg:
                continue
            organ_triples = kg.query(d_norm, "affects_organ")
            organs_found = []
            for t in organ_triples:
                if t.confidence >= 0.65 and t.object in self.atlas.organs:
                    org = self.atlas.organs[t.object]
                    # Restrict cardiovascular over-tagging
                    if org.system == "cardiovascular" and t.object not in ("heart", "aorta", "coronary_aa"):
                        continue
                    organs_found.append({"organ": t.object, "confidence": t.confidence})
                    if t.object not in seen:
                        affected_organs.append(t.object)
                        seen.add(t.object)
            if organs_found:
                trace.step("ORGAN_FROM_DISEASE",
                           f"Disease '{d_name}' affects {len(organs_found)} organs",
                           result=[o["organ"] for o in organs_found[:6]],
                           detail=f"KG: {d_norm} → affects_organ")

        # Symptom → organ (from KG)
        for sym in symptoms:
            s_norm = str(sym).strip().lower().replace(" ", "_")
            if not s_norm or not kg:
                continue
            organ_triples = kg.query(s_norm, "localizes_to")
            organs_found = []
            for t in organ_triples:
                if t.confidence >= 0.55 and t.object in self.atlas.organs:
                    org = self.atlas.organs[t.object]
                    if org.system == "cardiovascular" and t.object not in ("heart", "aorta", "coronary_aa"):
                        continue
                    organs_found.append(t.object)
                    if t.object not in seen:
                        affected_organs.append(t.object)
                        seen.add(t.object)
            if organs_found:
                trace.step("ORGAN_FROM_SYMPTOM",
                           f"Symptom '{sym}' localizes to {len(organs_found)} organs",
                           result=organs_found[:6])

        trace.step("ORGAN_SUMMARY",
                   f"Total affected organs: {len(affected_organs)}",
                   result=affected_organs[:10])

        # ── Phase 4: SPREAD PATHS (BFS through vasculature) ──
        all_spread = []
        if affected_organs:
            trace.step("SPREAD", "Infection-spread analysis",
                       detail=f"BFS from top {min(3, len(affected_organs))} organs, max 4 hops")
            for organ in affected_organs[:3]:
                paths = self.atlas.find_spread_paths(organ, "infection", max_hops=4)
                if paths:
                    trace.step("SPREAD_BFS",
                               f"From {organ}: {len(paths)} reachable organs",
                               result=[{"target": p["organ"], "hops": p["hops"],
                                        "path":   " → ".join(p["path"][:5])}
                                       for p in paths[:4]],
                               detail=paths[0].get("description", "") if paths else "")
                    for p in paths[:5]:
                        all_spread.append({**p, "origin": organ})

        # ── Phase 5: REFERRED PAIN ──
        referred = []
        try:
            referred = self.bridge._analyze_referred_pain(symptoms, user_input or "")
        except Exception:
            pass
        if referred:
            trace.step("REFERRED_PAIN",
                       f"Found {len(referred)} referred-pain sources",
                       result=[{"pain_at": r.get("pain_at", ""),
                                "source":  r.get("source_organ", "")}
                               for r in referred])
        else:
            trace.step("REFERRED_PAIN", "No referred-pain patterns detected")

        # ── Phase 6: ADJACENT ORGAN RISK ──
        adjacent = []
        for organ in affected_organs[:3]:
            try:
                for c in self.atlas.get_connections_from(organ, ["adjacent"]):
                    tgt = self.atlas.organs.get(c.target)
                    if tgt:
                        adjacent.append({"organ": c.target, "from": organ,
                                         "reason": c.desc or f"adjacent to {organ}"})
            except Exception:
                continue
        if adjacent:
            trace.step("ADJACENT",
                       f"{len(adjacent)} adjacent organs at risk",
                       result=[{"organ": a["organ"], "from": a["from"]}
                               for a in adjacent[:5]])

        # ── Phase 7: CONTEXT (built by anatomy_bridge) ──
        context = ""
        try:
            context = self.bridge._build_context(affected_organs, all_spread, referred, adjacent)
            trace.step("CONTEXT", "Built anatomy context",
                       detail=f"{len(context)} chars")
        except Exception:
            pass

        # ── Build trace data ──
        trace_data = trace.to_dict()
        trace_data["input"] = {
            "symptoms":  symptoms,
            "user_text": (user_input or "")[:200],
        }
        consistency_score = 0
        if isinstance(consistency, dict):
            consistency_score = consistency.get("consistency_score", 0)
        trace_data["summary"] = {
            "organs_found":           affected_organs[:15],
            "organs_count":           len(affected_organs),
            "spread_paths":           len(all_spread),
            "referred_pain_count":    len(referred),
            "adjacent_risk_count":    len(adjacent),
            "kg_triples_total":       len(kg) if kg else 0,
            "nexus_diagnoses_count":  len(result.get("nexus_diagnoses", [])),
            "consistency_score":      consistency_score,
            "thinking_steps":         len(nexus_thinking),
        }
        result["nexus_trace"]            = trace_data
        result["anatomy_affected_organs"] = affected_organs
        result["anatomy_spread"]          = all_spread[:10]
        result["anatomy_referred_pain"]   = referred
        result["anatomy_adjacent_risk"]   = adjacent
        result["anatomy_context"]         = context
        return result


def print_trace(trace_data: dict, verbose: bool = True):
    """Pretty-print a NEXUS trace dict."""
    print()
    print("+----------------------------------------------------------+")
    print("|             NEXUS Pure-Mechanism Reasoning Trace          |")
    print("+----------------------------------------------------------+")
    inp = trace_data.get("input", {})
    print(f"\n  Input:")
    print(f"    Symptoms: {inp.get('symptoms', [])}")
    if inp.get("user_text"):
        print(f"    Text: {inp['user_text'][:80]}")
    print(f"\n  Steps ({trace_data.get('total_steps', 0)}):")
    print(f"  {'~'*56}")
    for s in trace_data.get("steps", []):
        phase  = s["phase"]
        action = s["action"]
        ms     = s.get("ms", 0)
        prefix = {
            "INPUT": ">>", "NEXUS": "**",
            "NEXUS_CASCADE_MATCH": "##", "NEXUS_PHYSIOLOGY_SIM": "##",
            "NEXUS_SPATIAL_3D": "##", "NEXUS_RANKING": "##",
            "NEXUS_DX": "**", "NEXUS_CONSISTENCY": "==",
            "ANATOMY": "~~", "ORGAN_FROM_DISEASE": "->",
            "ORGAN_FROM_SYMPTOM": "->", "ORGAN_SUMMARY": "--",
            "SPREAD": "~~", "SPREAD_BFS": "->",
            "REFERRED_PAIN": "!!", "ADJACENT": "->",
            "CONTEXT": "--",
        }.get(phase, "  ")
        print(f"  {prefix} [{ms:6.1f}ms] {phase}: {action}")
        if verbose:
            detail = s.get("detail", "")
            if detail:
                print(f"               {detail}")
            if "result" in s:
                r = s["result"]
                if isinstance(r, list):
                    for item in r[:4]:
                        if isinstance(item, dict):
                            line = ", ".join(f"{k}={v}" for k, v in item.items())
                            print(f"               -> {line}")
                        else:
                            print(f"               -> {item}")
                    if s.get("result_count", 0) > 4:
                        print(f"               ... and {s['result_count'] - 4} more")
                elif isinstance(r, dict):
                    for k, v in r.items():
                        print(f"               {k}: {v}")
    summary = trace_data.get("summary", {})
    if summary:
        print(f"\n  {'='*56}")
        print(f"  Summary:")
        print(f"    Thinking steps:   {summary.get('thinking_steps', 0)}")
        print(f"    Organs found:     {summary.get('organs_count', 0)}")
        print(f"    Spread paths:     {summary.get('spread_paths', 0)}")
        print(f"    Referred pain:    {summary.get('referred_pain_count', 0)}")
        print(f"    Adjacent risk:    {summary.get('adjacent_risk_count', 0)}")
        print(f"    KG triples:       {summary.get('kg_triples_total', 0)}")
        print(f"    Total time:       {trace_data.get('total_ms', 0):.1f}ms")
        print()