"""
infer_patient.py — run a trained medical-JEPA on ONE (or many) patient
timelines and read off: current organ-state estimates, predicted future-state
changes, an (uncalibrated) severity signal, and which inputs could not be
grounded. Output is meant to be handed to the Osler symbolic engine — it is a
STATE REPRESENTATION, never a diagnosis.

Key difference from probe_eval*.py: this does NOT need state_snapshots / labels.
It tokenises the patient's events directly, so a brand-new unlabeled patient
works. (probe_eval needs labels because it scores against them.)

Usage:
    python3 infer_patient.py --ckpt checkpoints/medical_jepa_ctm.pt \
        --data new_patient.jsonl --delta_h 24 --top_k 10
    # add --json to emit machine-readable output for the Osler engine
"""
from __future__ import annotations
import argparse, os, sys, json

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    import torch
except ImportError:
    sys.exit("[infer] needs torch (the same env you trained in).")

from trajectory_schema import read_jsonl
from featurize import load_vocab, Featurizer
from featurize_seq import SeqFeaturizer
from train_jepa import MedicalJEPA


# ──────────────────────────────────────────────────────────────────────
def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    enc = ck.get("encoder", "mlp")
    model = MedicalJEPA(ck["n_states"], z_dim=ck["z_dim"], encoder=enc,
                        in_dim=ck.get("in_dim"), seq_dims=ck.get("seq_dims")).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck["vocab"], enc


def _seq_batch(tokens, device):
    """Pad a single token list into the encoder's batch dict (B=1)."""
    L = max(1, len(tokens))
    modality = torch.zeros(1, L, dtype=torch.long, device=device)
    feat_id  = torch.zeros(1, L, dtype=torch.long, device=device)
    value    = torch.zeros(1, L, dtype=torch.float32, device=device)
    t_hours  = torch.zeros(1, L, dtype=torch.float32, device=device)
    pad_mask = torch.ones(1, L, dtype=torch.bool, device=device)
    for ti, tk in enumerate(tokens):
        modality[0, ti] = int(tk["modality"]); feat_id[0, ti] = int(tk["feat_id"])
        value[0, ti] = float(tk["value"]);     t_hours[0, ti] = float(tk["t_hours"])
        pad_mask[0, ti] = False
    return {"modality": modality, "feat_id": feat_id,
            "value": value, "t_hours": t_hours, "pad_mask": pad_mask}


def encode_patient(model, vocab, encoder, traj, t_context, device):
    """Build context up to t_context and return z (1, z_dim). No labels needed."""
    if encoder in ("transformer", "ctm"):
        toks = SeqFeaturizer(vocab).tokens_up_to(traj, t_context)
        ctx = _seq_batch(toks, device)
        n_input = len(toks)
    else:  # mlp
        vec = Featurizer(vocab).context_vector(traj, t_context)
        ctx = torch.tensor([vec], dtype=torch.float32, device=device)
        n_input = int(sum(1 for x in vec if x != 0.0))
    with torch.no_grad():
        z = model.context_encoder(ctx)
    return z, n_input


def top_states(vec, states, k):
    rows = [(abs(v), v, states[i]) for i, v in enumerate(vec.tolist())]
    rows.sort(reverse=True, key=lambda r: r[0])
    return [{"state": s["state"], "organ": s.get("organ"),
             "direction": "high" if v >= 0 else "low",
             "value": round(v, 3), "observability": s.get("observability")}
            for _, v, s in rows[:k]]


def observed_inputs(traj, t_context):
    labs, vitals = set(), set()
    for e in traj.events:
        if e.t_hours > t_context:
            continue
        if e.kind == "labs" and isinstance(e.values, dict):
            labs |= {str(k).lower() for k in e.values}
        elif e.kind == "vitals" and isinstance(e.values, dict):
            vitals |= {str(k).lower() for k in e.values}
    return labs, vitals


def ungrounded_anchors(vocab, labs, vitals, limit=12):
    """Observable (lab/vital) states whose measuring input was NOT provided —
    i.e. dims the model can only infer indirectly for this patient."""
    out = []
    for s in vocab["states"]:
        if s.get("observability") not in ("lab", "vital"):
            continue
        s_labs = {str(x).lower() for x in (s.get("labs") or [])}
        s_vit = str(s.get("vital")).lower() if s.get("vital") else None
        has = bool(s_labs & labs) or (s_vit in vitals if s_vit else False)
        if not has:
            out.append(s["state"])
    return out[:limit], len(out)


# ──────────────────────────────────────────────────────────────────────
def run_patient(model, vocab, encoder, traj, args, device):
    states = vocab["states"]
    obs_idx = [e["index"] for e in states if e["observability"] in ("lab", "vital")]

    # context cutoff: last event time unless the user pins one
    ev_times = [e.t_hours for e in traj.events]
    t_ctx = args.t_context if args.t_context is not None else (max(ev_times) if ev_times else 0.0)

    z, n_input = encode_patient(model, vocab, encoder, traj, t_ctx, device)
    delta = torch.tensor([[args.delta_h]], dtype=torch.float32, device=device)
    with torch.no_grad():
        cur = model.state_head(z)[0]
        fut = model.state_head(model.predictor(z, delta))[0]
    cur = torch.nan_to_num(cur); fut = torch.nan_to_num(fut)
    change = fut - cur

    # severity proxy = mean magnitude on observable dims (UNCALIBRATED)
    def sev(v): return round(float(sum(abs(v[i]) for i in obs_idx) / max(1, len(obs_idx))), 4)
    labs, vitals = observed_inputs(traj, t_ctx)
    miss, n_miss = ungrounded_anchors(vocab, labs, vitals)

    result = {
        "patient_id": traj.patient_id,
        "t_context_h": t_ctx,
        "n_input_tokens": n_input,
        "delta_h": args.delta_h,
        "current_top_states": top_states(cur, states, args.top_k),
        "future_top_states": top_states(fut, states, args.top_k),
        "biggest_predicted_changes": top_states(change, states, args.top_k),
        "severity_proxy_current": sev(cur),
        "severity_proxy_future": sev(fut),
        "ungrounded_anchor_inputs": {"examples": miss, "count": n_miss},
    }
    return result


def pretty(r):
    print(f"\n=== {r['patient_id']}  (context ≤ {r['t_context_h']}h, "
          f"{r['n_input_tokens']} input tokens) ===")
    print("current top states:")
    for s in r["current_top_states"]:
        print(f"  {s['state']:38s} {s['direction']:4s} {s['value']:+.3f}  [{s['observability']}]")
    print(f"predicted future states (+{r['delta_h']}h):")
    for s in r["future_top_states"]:
        print(f"  {s['state']:38s} {s['direction']:4s} {s['value']:+.3f}")
    print("biggest predicted changes (rising/falling):")
    for s in r["biggest_predicted_changes"]:
        arrow = "↑rising" if s["value"] >= 0 else "↓falling"
        print(f"  {s['state']:38s} {arrow:8s} Δ={s['value']:+.3f}")
    print(f"severity proxy (UNCALIBRATED): current={r['severity_proxy_current']} "
          f"future={r['severity_proxy_future']}")
    print(f"ungrounded anchor inputs ({r['ungrounded_anchor_inputs']['count']} dims "
          f"with no measured input), e.g.: {', '.join(r['ungrounded_anchor_inputs']['examples'][:6])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(_HERE, "checkpoints", "medical_jepa_ctm.pt"))
    ap.add_argument("--data", required=True, help="trajectory JSONL (labels optional)")
    ap.add_argument("--delta_h", type=float, default=24.0)
    ap.add_argument("--t_context", type=float, default=None,
                    help="encode events up to this hour (default: last event)")
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--json", action="store_true", help="emit JSON for the Osler engine")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, vocab, encoder = load_model(args.ckpt, device)
    trajs = read_jsonl(args.data)
    if not trajs:
        sys.exit("[infer] no trajectories in --data")

    results = [run_patient(model, vocab, encoder, t, args, device) for t in trajs]
    if args.json:
        print(json.dumps(results if len(results) > 1 else results[0], ensure_ascii=False, indent=2))
    else:
        print(f"[infer] encoder={encoder} | {len(trajs)} patient(s)")
        for r in results:
            pretty(r)
        print("\n[infer] STATE REPRESENTATION ONLY — not a diagnosis. "
              "Feed to the Osler symbolic engine for red-flag / guideline / "
              "safety checks and explanation.")


if __name__ == "__main__":
    main()
