"""
probe_eval.py — Phase 3: evaluate a trained medical-JEPA.

⚠️  REQUIRES PyTorch + a trained checkpoint from train_jepa.py.
    Does not run in the Osler container.

A JEPA is only as good as what its latent can predict. This runs the probes
from MEDICAL_JEPA_ASSESSMENT.md:

  Probe A — observable-state recovery: for the directly-observable latent
            dims, how well does the decoded state match the label? (R²/MAE)
            This is the core sanity check: is the latent learning real state?

  Probe D — Osler mechanism consistency: feed the model's predicted top states
            back into Osler and check whether Osler agrees they explain the
            input symptoms. (Neuro-symbolic consistency.)

  (Probes B/future-lab and C/deterioration need real longitudinal data with
   outcomes — stubbed here, wired the same way once you have MIMIC.)

Run:
    python3 probe_eval.py --ckpt checkpoints/medical_jepa.pt --data <held_out.jsonl>
"""
from __future__ import annotations
from ctm_encoder import CTMEventEncoder
import argparse, os, sys, json

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    sys.exit("[probe_eval] needs torch. Run where you have it (see train_jepa.py).")

from featurize import build_examples
from featurize_seq import build_seq_examples, SeqFeaturizer, N_MODALITIES, N_DRUG_BUCKETS
from train_jepa import MedicalJEPA, make_batches, make_seq_batches
from ctm_encoder import CTMEventEncoder
from trajectory_schema import read_jsonl


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    enc = ck.get("encoder", "mlp")

    # Safety: infer CTM checkpoint from state_dict keys if metadata is wrong
    model_keys = ck.get("model", {}).keys()
    if any(k.startswith("context_encoder.nlm_") or k.startswith("context_encoder.sink") for k in model_keys):
        enc = "ctm"
    
    model = MedicalJEPA(ck["n_states"], z_dim=ck["z_dim"], encoder=enc,
                        in_dim=ck.get("in_dim"), seq_dims=ck.get("seq_dims")).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck["vocab"], enc


def _encode_one(model, example, encoder, device):
    """Run the context encoder on a single example, return z (1, z_dim)."""
    if encoder in ("transformer", "ctm"):
        batch, *_ = next(make_seq_batches([example], 1, device, shuffle=False))
        return model.context_encoder(batch)
    else:
        ctx = torch.tensor([example["context"]], dtype=torch.float32, device=device)
        return model.context_encoder(ctx)




def _finite_float(x):
    try:
        v = float(x)
    except Exception:
        return None
    if v != v or v in (float("inf"), float("-inf")):
        return None
    return v

def probe_A_state_recovery(model, examples, vocab, device, encoder):
    """For observable dims, decoded value vs labeled value. NaN-safe."""
    obs_idx = [e["index"] for e in vocab["states"]
               if e["observability"] in ("lab", "vital")]
    if not obs_idx:
        return {"note": "no observable dims in vocab"}
    records = []
    with torch.no_grad():
        for e in examples:
            tgt = torch.tensor(e["target_values"], device=device, dtype=torch.float32)
            msk = torch.tensor(e["target_mask"], device=device, dtype=torch.float32)
            pred = model.state_head(_encode_one(model, e, encoder, device))[0]
            pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
            tgt = torch.nan_to_num(tgt, nan=0.0, posinf=0.0, neginf=0.0)
            for i in obs_idx:
                if msk[i] > 0:
                    records.append((float(pred[i].item()), float(tgt[i].item())))
    if not records:
        return {"note": "no observable labels in eval set"}
    ybar = sum(y for _, y in records) / len(records)
    sse = sum((p - y) ** 2 for p, y in records)
    sse_tot = sum((y - ybar) ** 2 for _, y in records)
    mae = sum(abs(p - y) for p, y in records) / len(records)
    r2 = 1 - (sse / sse_tot) if sse_tot > 0 else float("nan")
    return {"n_labels": int(len(records)), "MAE": round(mae, 4),
            "R2_vs_mean": round(r2, 4)}

def probe_B_future_state(model, examples, vocab, device, encoder, change_thresh=0.1):
    """Future-state prediction vs persistence baseline, NaN-safe."""
    obs_idx = [e["index"] for e in vocab["states"]
               if e["observability"] in ("lab", "vital")]
    fut = [e for e in examples if e.get("delta_h", 0) > 0]
    if not fut:
        return {"note": "no future-state examples; needs multi-timepoint data"}
    ctx_state = {}
    for e in examples:
        if e.get("delta_h", 0) == 0:
            ctx_state[(e["patient_id"], e["t_context"])] = e["target_values"]

    ae_pred = ae_persist = cnt = 0.0
    c_pred = c_persist = c_cnt = 0.0
    with torch.no_grad():
        for e in fut:
            z = _encode_one(model, e, encoder, device)
            z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
            dlt = torch.tensor([[e["delta_h"]]], dtype=torch.float32, device=device)
            pred_future = model.state_head(model.predictor(z, dlt))[0]
            pred_persist = model.state_head(z)[0]
            pred_future = torch.nan_to_num(pred_future, nan=0.0, posinf=0.0, neginf=0.0)
            pred_persist = torch.nan_to_num(pred_persist, nan=0.0, posinf=0.0, neginf=0.0)
            tgt = torch.nan_to_num(torch.tensor(e["target_values"], device=device, dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
            msk = torch.tensor(e["target_mask"], device=device, dtype=torch.float32)
            base = ctx_state.get((e["patient_id"], e["t_context"]))
            for i in obs_idx:
                if msk[i] <= 0:
                    continue
                ep = abs((pred_future[i] - tgt[i]).item())
                eb = abs((pred_persist[i] - tgt[i]).item())
                ae_pred += ep; ae_persist += eb; cnt += 1
                if base is not None:
                    bval = _finite_float(base[i])
                    if bval is not None and abs(tgt[i].item() - bval) >= change_thresh:
                        c_pred += ep; c_persist += eb; c_cnt += 1
    out = {"n_future_labels": int(cnt),
           "all__MAE_forward": round(ae_pred / cnt, 4) if cnt else None,
           "all__MAE_persistence": round(ae_persist / cnt, 4) if cnt else None,
           "all__forward_beats_persistence": (ae_pred < ae_persist) if cnt else None}
    if c_cnt:
        out.update({
            "changed__n": int(c_cnt),
            "changed__MAE_forward": round(c_pred / c_cnt, 4),
            "changed__MAE_persistence": round(c_persist / c_cnt, 4),
            "changed__forward_beats_persistence": c_pred < c_persist,
            "note": "CHANGED-ONLY is the fair dynamics test; beating persistence there ⇒ model predicts real state changes, not just stasis"})
    else:
        out["note"] = "no states changed >= threshold between bins — try longer Δ (48/72h) or smaller bins"
    return out

def probe_C_deterioration(model, examples, vocab, device, encoder):
    """
    Deterioration / outcome prediction. Two modes:

    (1) Logistic probe (preferred): freeze the encoder, extract latent z for
        every example, train a logistic regression z→outcome with patient-level
        train/test split, report AUROC. This is the standard way to ask "does
        the latent carry outcome signal?"

    (2) Severity-proxy fallback (if sklearn missing or too few examples):
        compare mean |predicted state| between outcome groups.

    Requires per-example 'outcome' (bad=ICU/death). Stub if absent.
    """
    have = [e for e in examples if e.get("outcome") is not None]
    if not have:
        return {"note": "no per-example outcomes; needs real EHR with icu/death flags"}
    pos = sum(1 for e in have if e["outcome"])
    neg = len(have) - pos
    if pos == 0 or neg == 0:
        return {"note": f"only one outcome class present ({pos} pos / {neg} neg); "
                        "need both, and ideally balanced sampling"}

    # extract latents
    import numpy as np
    Z, y, pids = [], [], []
    with torch.no_grad():
        for e in have:
            z = _encode_one(model, e, encoder, device)[0].cpu().numpy()
            Z.append(z); y.append(1 if e["outcome"] else 0); pids.append(e["patient_id"])
    Z = np.nan_to_num(np.array(Z), nan=0.0, posinf=0.0, neginf=0.0); y = np.array(y)

    # try the proper logistic probe with PATIENT-LEVEL split
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        uniq = sorted(set(pids))
        import random as _r
        _r.Random(0).shuffle(uniq)
        cut = max(1, int(len(uniq) * 0.7))
        train_p = set(uniq[:cut])
        tr = [i for i, p in enumerate(pids) if p in train_p]
        te = [i for i, p in enumerate(pids) if p not in train_p]
        if tr and te and len(set(y[tr])) == 2 and len(set(y[te])) == 2:
            clf = LogisticRegression(max_iter=1000, class_weight="balanced")
            clf.fit(Z[tr], y[tr])
            auroc = roc_auc_score(y[te], clf.predict_proba(Z[te])[:, 1])
            return {"mode": "logistic_probe", "n_pos": int(pos), "n_neg": int(neg),
                    "test_AUROC": round(float(auroc), 4),
                    "note": "AUROC>0.5 ⇒ latent carries outcome signal; "
                            "patient-level split prevents leakage"}
    except ImportError:
        pass

    # fallback: severity proxy
    obs_idx = [e["index"] for e in vocab["states"]
               if e["observability"] in ("lab", "vital")]
    sev_pos, sev_neg = [], []
    with torch.no_grad():
        for e in have:
            pred = model.state_head(_encode_one(model, e, encoder, device))[0]
            sev = sum(abs(pred[i].item()) for i in obs_idx) / max(1, len(obs_idx))
            (sev_pos if e["outcome"] else sev_neg).append(sev)
    mp = sum(sev_pos) / len(sev_pos); mn = sum(sev_neg) / len(sev_neg)
    return {"mode": "severity_proxy", "n_pos": pos, "n_neg": neg,
            "mean_severity_bad": round(mp, 4), "mean_severity_good": round(mn, 4),
            "separation": round(mp - mn, 4),
            "note": "install sklearn for the proper logistic-probe AUROC"}


def probe_D_mechanism_consistency(model, examples, vocab, device, encoder, top_k=5):
    """
    Decode predicted top states; check (offline, structurally) that they're
    real Osler states and that the highest-confidence ones cluster on a single
    organ system (a weak proxy for coherence without re-running Osler here).
    For full consistency you'd call Osler.reason() and compare — left as a hook.
    """
    states = vocab["states"]
    coherent = total = 0
    with torch.no_grad():
        for e in examples:
            pred = model.state_head(_encode_one(model, e, encoder, device))[0]
            mag = pred.abs()
            top = torch.topk(mag, min(top_k, len(states))).indices.tolist()
            organs = [states[i]["organ"] for i in top]
            # coherent if the top states concentrate (>=60%) on one organ
            if organs:
                dom = max(set(organs), key=organs.count)
                if organs.count(dom) / len(organs) >= 0.6:
                    coherent += 1
            total += 1
    return {"checked": total,
            "single_organ_coherent": coherent,
            "coherence_rate": round(coherent / total, 3) if total else 0.0,
            "note": "structural proxy; full check calls Osler.reason() on decoded states"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(_HERE, "checkpoints", "medical_jepa_mlp.pt"))
    ap.add_argument("--data", default=os.path.join(_HERE, "osler_coverage_trajectories.jsonl"))
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, vocab, encoder = load_model(args.ckpt, device)
    if encoder in ("transformer", "ctm"):
        examples, _ = build_seq_examples(args.data)
    else:
        examples, _ = build_examples(args.data)

    # attach per-example outcome for Probe C.
    # Prefer the MEANINGFUL 'deterioration' label; fall back to death.
    # (icu_admission is provenance, not a label — every ICU patient has it.)
    out_map = {}
    for t in read_jsonl(args.data):
        o = t.outcomes or {}
        if "deterioration" in o:
            out_map[t.patient_id] = bool(o["deterioration"])
        elif "death" in o:
            out_map[t.patient_id] = bool(o["death"])
        else:
            out_map[t.patient_id] = None
    for e in examples:
        e["outcome"] = out_map.get(e["patient_id"])

    print(f"[probe_eval] encoder={encoder} | {len(examples)} eval examples\n")

    print("Probe A — observable-state recovery (recovers weak labels):")
    print("  ", probe_A_state_recovery(model, examples, vocab, device, encoder))
    print("\nProbe B — future-state prediction (learned DYNAMICS?):")
    print("  ", probe_B_future_state(model, examples, vocab, device, encoder))
    print("\nProbe C — deterioration/outcome signal in latent:")
    print("  ", probe_C_deterioration(model, examples, vocab, device, encoder))
    print("\nProbe D — mechanism consistency (structural proxy):")
    print("  ", probe_D_mechanism_consistency(model, examples, vocab, device, encoder))
    print("\n[probe_eval] Probe A recovers weak labels; B/C are the ones that test "
          "whether the model learned real patient dynamics + outcome signal.")


if __name__ == "__main__":
    main()
