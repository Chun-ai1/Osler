"""
eval_generalization.py — the honest test: does the JEPA generalize to diseases
it never saw in training, or did it memorize?

Loads a checkpoint, evaluates Probe-A-style observable-state recovery on:
  - the TRAINING diseases (upper bound / memorization ceiling)
  - the HELD-OUT diseases, bucketed by state-coverage tier:
      high (>=80% states seen in train)  → fair generalization test
      mid  (40-80%)
      low  (<40%)                          → low score EXPECTED here

A model that memorized will show: high train score, collapsing held-out score.
A model that learned mechanism will: hold up on the high-coverage held-out tier.

Usage:
    python3 eval_generalization.py --ckpt checkpoints/medical_jepa_mlp.pt
"""
from __future__ import annotations
import argparse, os, sys, json, collections

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    import torch
except ImportError:
    sys.exit("[eval_gen] needs torch.")

from featurize import build_examples
from featurize_seq import build_seq_examples
from train_jepa import MedicalJEPA, make_seq_batches
from probe_eval import load_model, _encode_one
from trajectory_schema import read_jsonl


def _disease_of(example_pid, traj_by_pid):
    t = traj_by_pid.get(example_pid)
    return t.outcomes.get("discharge_dx", example_pid) if t else example_pid


def observable_recovery(model, examples, vocab, device, encoder):
    """Per-example MAE on observable, labeled states. Returns list of (mae)."""
    obs_idx = [e["index"] for e in vocab["states"]
               if e["observability"] in ("lab", "vital")]
    per_ex = []
    with torch.no_grad():
        for e in examples:
            tgt = torch.tensor(e["target_values"], device=device)
            msk = torch.tensor(e["target_mask"], device=device)
            pred = model.state_head(_encode_one(model, e, encoder, device))[0]
            errs = []
            for i in obs_idx:
                if msk[i] > 0:
                    errs.append(abs((pred[i] - tgt[i]).item()))
            if errs:
                per_ex.append(sum(errs) / len(errs))
    return per_ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(_HERE, "checkpoints", "medical_jepa_mlp.pt"))
    ap.add_argument("--train", default=os.path.join(_HERE, "train_synthetic.jsonl"))
    ap.add_argument("--heldout", default=os.path.join(_HERE, "heldout_synthetic.jsonl"))
    ap.add_argument("--report", default=os.path.join(_HERE, "split_report.json"))
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, vocab, encoder = load_model(args.ckpt, device)
    build = build_seq_examples if encoder == "transformer" else build_examples

    # coverage tier per held-out disease
    rep = json.load(open(args.report))
    tier = {}
    for d in rep["heldout_diseases"]:
        c = d["coverage"]
        tier[d["disease"]] = "high" if c >= 0.8 else "mid" if c >= 0.4 else "low"

    # map patient_id → disease
    def pid_to_dz(path):
        return {t.patient_id: t.outcomes.get("discharge_dx", t.patient_id)
                for t in read_jsonl(path)}

    print(f"[eval_gen] encoder={encoder}\n")

    # training set (memorization ceiling)
    tr_ex, _ = build(args.train)
    tr_mae = observable_recovery(model, tr_ex, vocab, device, encoder)
    print(f"TRAIN diseases (memorization ceiling):")
    print(f"  examples={len(tr_mae)}  mean MAE={_mean(tr_mae):.4f}")

    # held-out, bucketed by tier
    ho_ex, _ = build(args.heldout)
    ho_pid_dz = pid_to_dz(args.heldout)
    buckets = collections.defaultdict(list)
    mae_list = observable_recovery_with_pid(model, ho_ex, vocab, device, encoder)
    for e, mae in zip(ho_ex, mae_list):
        if mae is None:
            continue
        dz = ho_pid_dz.get(e["patient_id"], "?")
        buckets[tier.get(dz, "low")].append(mae)

    print(f"\nHELD-OUT diseases (never seen in training), by state-coverage tier:")
    for t in ("high", "mid", "low"):
        vals = buckets.get(t, [])
        if vals:
            note = {"high": "← fair generalization test",
                    "mid": "", "low": "← low expected (no basis)"}[t]
            print(f"  {t:>4s} coverage: examples={len(vals):>3d}  mean MAE={_mean(vals):.4f}  {note}")

    print("\n[eval_gen] Interpretation:")
    print("  If 'high' held-out MAE ≈ train MAE → model generalizes via mechanism.")
    print("  If 'high' held-out MAE >> train MAE → model memorized disease patterns.")
    print("  On SYNTHETIC data this only tests the pipeline's generalization behavior,")
    print("  not clinical validity (labels come from Osler's own rules).")


def observable_recovery_with_pid(model, examples, vocab, device, encoder):
    """Same as observable_recovery but aligned 1:1 with examples that HAVE labels.
    Returns a list matching `examples` order, with None where no observable label."""
    obs_idx = [e["index"] for e in vocab["states"]
               if e["observability"] in ("lab", "vital")]
    out = []
    with torch.no_grad():
        for e in examples:
            tgt = torch.tensor(e["target_values"], device=device)
            msk = torch.tensor(e["target_mask"], device=device)
            pred = model.state_head(_encode_one(model, e, encoder, device))[0]
            errs = [abs((pred[i] - tgt[i]).item()) for i in obs_idx if msk[i] > 0]
            out.append(sum(errs) / len(errs) if errs else None)
    return out  # aligned 1:1 with examples; None where no observable label


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


if __name__ == "__main__":
    main()
