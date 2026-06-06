"""
jepa_osler_bridge.py — connect CTM-JEPA (the temporal encoder) to Osler
(your main symbolic reasoning module) at INFERENCE time.

Flow (Direction A):
    patient timeline
      → CTM-JEPA  : current latent state + predicted future state + biggest changes
      → Osler     : reason() on observed data  (mechanism + diagnoses)
      → fuse      : reconcile JEPA-decoded states vs Osler mechanism states
      → filter    : keep only JEPA transitions that are mechanistically plausible
      → red flags : check observed vitals + JEPA-predicted trajectory against thresholds
      → handoff   : one clean payload for the LLM explainer

KEY: JEPA's latent dims ARE Osler states. vocab.json was generated from
nexus_engine.state_model (see build_vocab.py), so state_head's outputs are already
`organ.variable` names Osler understands — no translation layer is needed.

Osler is reached through the SAME API your code already uses
(NexusMedical().load_knowledge(); nx.reason(symptoms, context)). If nexus_engine
isn't importable (e.g. this sandbox), the adapter falls back to a clearly-marked
stub so the JEPA half still runs.

    python3 jepa_osler_bridge.py --ckpt checkpoints/medical_jepa_ctm.pt \
        --data new_patient.jsonl --delta_h 24 --json
"""
from __future__ import annotations
import argparse, os, sys, json, types

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)            # llm_project/  — where nexus_engine lives
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from trajectory_schema import read_jsonl
from featurize import load_vocab
from infer_patient import load_model, run_patient
from rules_common import extract_atoms
from mechanism_filter import classify

# lab reference ranges → high/low flag (so abnormal labs can be sent to the
# Osler-grounded labeler). Only labs we can flag are forwarded.
LAB_REF = {  # name(lower): (low_below, high_above)
    "wbc": (4, 11), "creatinine": (None, 1.3), "bun": (None, 20), "platelets": (150, None),
    "hgb": (12, None), "inr": (None, 1.4), "total_bilirubin": (None, 1.2),
    "glucose": (70, 180), "lactate": (None, 2.0), "sodium": (135, 145),
    "potassium": (3.5, 5.1), "anion_gap": (None, 12), "crp": (None, 10),
    "troponin": (None, 0.04), "bicarbonate": (22, None),
}


def lab_flags(labs):
    """{name: 'high'|'low'} for abnormal labs only (skips normals/unknowns)."""
    out = {}
    for name, val in (labs or {}).items():
        ref = LAB_REF.get(str(name).lower())
        if not ref:
            continue
        lo, hi = ref
        try: x = float(val)
        except (TypeError, ValueError): continue
        if hi is not None and x > hi: out[name] = "high"
        elif lo is not None and x < lo: out[name] = "low"
    return out


# ── Osler adapter: real engine if available, labelled stub otherwise ──────
class OslerAdapter:
    def __init__(self):
        self.available = False
        self._nx = None
        self.why_unavailable = None
        import io
        _so = sys.stdout
        try:
            sys.stdout = io.StringIO()
            from nexus_engine.nexus_medical import NexusMedical
            self._nx = NexusMedical(); self._nx.load_knowledge()
            self.available = True
        except Exception as e:
            self.available = False
            self.why_unavailable = f"{type(e).__name__}: {e}"
        finally:
            sys.stdout = _so

    def reason(self, symptoms, context):
        if not self.available:
            return {"diagnoses": [], "_stub": True}
        try:
            return self._nx.reason(symptoms or [], context=context or {})
        except Exception as e:
            return {"diagnoses": [], "_error": str(e)}

    def mechanism_states(self, symptoms, context, top_k=3):
        """[{state, direction}] implied by Osler's reasoning over observed symptoms."""
        res = self.reason(symptoms, context)
        out, seen = [], set()
        for d in res.get("diagnoses", [])[:top_k]:
            for ms in d.get("_state_simulation", {}).get("matched_states", []):
                s = ms.get("state")
                if s and s not in seen:
                    seen.add(s)
                    out.append({"state": s, "direction": ms.get("direction", "high")})
        return out, res

    def grounded_states(self, symptoms, labs, flags, vitals, context):
        """(1) Feed observed labs/vitals (+symptoms) through the Osler-grounded
        labeler. lab/vital → state uses lab_integration.json (no engine needed);
        symptom → state uses Osler reason() when available. Returns
        [{state, direction, source, confidence}], deduped by state."""
        out = []
        try:
            from weak_labeler import label_from_labs, label_from_vitals
            try: out += label_from_labs(labs or {}, flags or {})
            except Exception: pass              # lab_integration.json missing → skip
            out += label_from_vitals(vitals or {})
            if self.available and symptoms:
                from weak_labeler import label_from_mechanism
                try: out += label_from_mechanism(symptoms, context)
                except Exception: pass
        except Exception:
            return []
        best = {}
        for l in out:
            s = l["state"]
            if s not in best or l.get("confidence", 0) > best[s].get("confidence", 0):
                best[s] = l
        return [{"state": k, "direction": v.get("direction"),
                 "source": v.get("source"), "confidence": v.get("confidence")}
                for k, v in best.items()]


# ── observed signals from the raw events ──────────────────────────────────
def observed(traj, t_ctx):
    symptoms, last_vit, labs = [], {}, {}
    for e in traj.events:
        if e.t_hours > t_ctx:
            continue
        if e.kind == "symptoms" and isinstance(e.values, list):
            symptoms += [str(s) for s in e.values]
        elif e.kind == "vitals" and isinstance(e.values, dict):
            for k, v in e.values.items():
                try: last_vit[str(k).lower()] = float(v)
                except (TypeError, ValueError): pass
        elif e.kind == "labs" and isinstance(e.values, dict):
            for k, v in e.values.items():
                try: labs[k] = float(v)        # keep latest value (events time-ordered)
                except (TypeError, ValueError): pass
    return sorted(set(symptoms)), last_vit, labs


RED = {  # vital -> (comparator, threshold, message)  — transparent placeholder
    "spo2": ("<", 92, "hypoxemia"), "rr": (">", 28, "tachypnea"),
    "bp_sys": ("<", 90, "hypotension"), "hr": (">", 130, "severe tachycardia"),
    "temp": (">", 39.5, "high fever"),
}
# observable latent dims whose worsening (predicted) should raise a forward flag
FORWARD_WATCH = {
    "lung.gas_exchange": "falling", "heart.cardiac_output": "falling",
    "kidney.filtration": "falling", "heart.ischemia": "rising",
}


def red_flags(last_vit, jepa):
    cur = []
    for v, (cmp_, thr, msg) in RED.items():
        if v in last_vit:
            x = last_vit[v]
            if (x < thr) if cmp_ == "<" else (x > thr):
                cur.append({"signal": v, "value": x, "msg": msg})
    pred = []
    chg = {c["state"]: c for c in jepa["biggest_predicted_changes"]}
    for st, bad_dir in FORWARD_WATCH.items():
        c = chg.get(st)
        if c and c["direction"] == bad_dir:
            pred.append({"state": st, "direction": bad_dir, "delta": c["value"]})
    return cur, pred


def reconcile(jepa_current, osler_states):
    j = {s["state"] for s in jepa_current}
    o = {s["state"] for s in osler_states}
    return {"agreed": sorted(j & o), "jepa_only": sorted(j - o), "osler_only": sorted(o - j)}


def plausible_transitions(traj, t_ctx, jepa, vidx):
    """Per-patient use of the mechanism filter: which JEPA-predicted changes are
    mechanistically supported by what we actually observed in this patient?"""
    atoms = sorted(extract_atoms(traj, t_ctx))
    keep = []
    for c in jepa["biggest_predicted_changes"]:
        st = vidx.get(c["state"], {})
        rule = {"context_pattern": atoms or ["(no_atoms)"],
                "predicted_transition": {"state": c["state"], "organ": st.get("organ"),
                                          "direction": c["direction"]}}
        cls, reasons = classify(rule, vidx)
        if cls == "plausible":
            keep.append({"state": c["state"], "direction": c["direction"],
                         "delta": c["value"], "reason": reasons[0]})
    return atoms, keep


def bridge_one(model, vocab, encoder, traj, osler, args, device):
    vidx = {e["state"]: e for e in vocab["states"]}
    jepa = run_patient(model, vocab, encoder, traj,
                       types.SimpleNamespace(delta_h=args.delta_h, t_context=args.t_context,
                                             top_k=args.top_k), device)
    t_ctx = jepa["t_context_h"]
    symptoms, last_vit, labs = observed(traj, t_ctx)
    flags = lab_flags(labs)
    context = {k: traj.demographics.get(k) for k in ("age", "sex") if k in traj.demographics}

    # (1) feed observed labs/vitals (+symptoms) through the Osler-grounded labeler
    osler_grounded = osler.grounded_states(symptoms, labs, flags, vitals=last_vit, context=context)
    # symptom-driven diagnoses (only non-empty when symptoms exist)
    osler_states, osler_raw = osler.mechanism_states(symptoms, context)
    cur_flags, pred_flags = red_flags(last_vit, jepa)
    atoms, approved = plausible_transitions(traj, t_ctx, jepa, vidx)
    # (2) reconcile JEPA-decoded current states against the Osler-grounded states
    rec = reconcile(jepa["current_top_states"], osler_grounded)

    return {
        "patient_id": traj.patient_id,
        "t_context_h": t_ctx,
        "observed": {"symptoms": symptoms, "last_vitals": last_vit,
                     "abnormal_labs": flags, "atoms": atoms},
        "jepa": {
            "current_top_states": jepa["current_top_states"],
            "future_top_states": jepa["future_top_states"],
            "biggest_predicted_changes": jepa["biggest_predicted_changes"],
            "severity_proxy_current": jepa["severity_proxy_current"],
            "severity_proxy_future": jepa["severity_proxy_future"],
        },
        "osler": {"available": osler.available,
                  "grounded_states": osler_grounded,
                  "mechanism_states": osler_states,
                  "n_diagnoses": len(osler_raw.get("diagnoses", [])),
                  "why_unavailable": osler.why_unavailable},
        "reconciliation": rec,
        "red_flags": {"current_observed": cur_flags, "predicted_forward": pred_flags},
        "plausible_predicted_transitions": approved,
        "handoff_to_explainer": {
            "diagnoses": [d.get("diagnosis") or d.get("name")
                          for d in osler_raw.get("diagnoses", [])[:3]] if osler.available else [],
            "agreed_states": rec["agreed"],
            "current_concern": [s["state"] for s in jepa["current_top_states"][:3]],
            "forward_concern": [t["state"] for t in approved[:3]],
            "red_flags": cur_flags + pred_flags,
            "uncertain": jepa["severity_proxy_current"] == jepa["severity_proxy_future"],
        },
        "disclaimer": "STATE REPRESENTATION + symbolic reasoning support — not a diagnosis.",
    }


def pretty(r):
    o = r["osler"]
    print(f"\n=== {r['patient_id']} (≤{r['t_context_h']}h) ===")
    print(f"observed symptoms: {', '.join(r['observed']['symptoms']) or '—'}")
    print(f"observed atoms: {', '.join(r['observed']['atoms']) or '—'}")
    print(f"abnormal labs: {r['observed']['abnormal_labs'] or '—'}")
    print(f"Osler: available={o['available']} | grounded_states="
          f"{[s['state'] for s in o['grounded_states'][:5]]} | diagnoses={o['n_diagnoses']}")
    rec = r["reconciliation"]
    print(f"reconciliation: agreed={rec['agreed'][:5]} | jepa_only={rec['jepa_only'][:3]} | "
          f"osler_only={rec['osler_only'][:3]}")
    print("plausible predicted transitions (JEPA ∩ mechanism):")
    for t in r["plausible_predicted_transitions"][:5]:
        print(f"  → {t['state']:30s} {t['direction']:7s} Δ={t['delta']:+.3f}  ({t['reason']})")
    rf = r["red_flags"]
    print(f"red flags: current={[f['signal'] for f in rf['current_observed']]} "
          f"predicted={[f['state'] for f in rf['predicted_forward']]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(_HERE, "checkpoints", "medical_jepa_ctm.pt"))
    ap.add_argument("--data", required=True)
    ap.add_argument("--delta_h", type=float, default=24.0)
    ap.add_argument("--t_context", type=float, default=None)
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, vocab, encoder = load_model(args.ckpt, device)
    osler = OslerAdapter()
    trajs = read_jsonl(args.data)
    results = [bridge_one(model, vocab, encoder, t, osler, args, device) for t in trajs]

    if args.json:
        print(json.dumps(results if len(results) > 1 else results[0], ensure_ascii=False, indent=2))
    else:
        print(f"[bridge] encoder={encoder} | Osler available={osler.available} | {len(trajs)} patient(s)")
        if not osler.available:
            print("[bridge] NOTE: Osler not active → reasoning is STUBBED. Reason: "
                  f"{osler.why_unavailable}")
            print("[bridge] fix: ensure llm_project/ is importable and nexus_engine + "
                  "medical_knowledge/ are present, then re-run.")
        for r in results:
            pretty(r)


if __name__ == "__main__":
    main()