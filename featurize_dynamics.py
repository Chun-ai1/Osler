"""
featurize.py — turn trajectories into model-ready feature/target arrays.

Bridges the gap between the human-readable Trajectory JSONL and the numeric
tensors a JEPA needs. Pure Python + (optional) numpy — no torch — so it's
testable in any environment. The training script (train_jepa.py) imports these.

For each trajectory we produce, per timepoint:
  - context features X(t): symptom multi-hot + lab values + vital values + age/sex
  - state-label target  Y(t): a vector over the frozen latent vocab, with a
    mask of which states are actually labeled (most are unlabeled at any t)
  - confidence weights  W(t): per-state label confidence (for weighted loss)

The JEPA uses these to build:
  - masked-state prediction (hide some labeled states, predict them)
  - future-state prediction (encode up to t, predict labels at t+Δ)
"""
from __future__ import annotations
import sys, os, json

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from trajectory_schema import read_jsonl

try:
    import numpy as np
    _HAVE_NP = True
except ImportError:
    _HAVE_NP = False


def load_vocab(path: str | None = None) -> dict:
    path = path or os.path.join(_HERE, "vocab.json")
    with open(path) as f:
        return json.load(f)


class Featurizer:
    """Stateless-ish helper that maps observations <-> fixed-width vectors."""

    def __init__(self, vocab: dict):
        self.vocab = vocab
        self.state_index = {e["state"]: e["index"] for e in vocab["states"]}
        self.n_states = len(vocab["states"])
        self.symptom_index = {s: i for i, s in enumerate(vocab["input_symptoms"])}
        self.lab_index = {l: i for i, l in enumerate(vocab["input_labs"])}
        self.vital_index = {v: i for i, v in enumerate(vocab["input_vitals"])}
        self.n_sym = len(self.symptom_index)
        self.n_lab = len(self.lab_index)
        self.n_vit = len(self.vital_index)
        # context feature width: symptoms(multi-hot) + labs(value+present) + vitals(value+present) + age + sex
        self.context_dim = self.n_sym + 2 * self.n_lab + 2 * self.n_vit + 2

    # ── context: observations up to and including time t ──
    def context_vector(self, traj, t_hours: float):
        """Aggregate all events with t<=t_hours into one context vector."""
        sym = [0.0] * self.n_sym
        lab_val = [0.0] * self.n_lab
        lab_present = [0.0] * self.n_lab
        vit_val = [0.0] * self.n_vit
        vit_present = [0.0] * self.n_vit
        for e in traj.events:
            if e.t_hours > t_hours:
                continue
            if e.kind == "symptoms":
                for s in e.values:
                    if s in self.symptom_index:
                        sym[self.symptom_index[s]] = 1.0
            elif e.kind == "labs":
                for lab, v in e.values.items():
                    if lab in self.lab_index:
                        idx = self.lab_index[lab]
                        try:
                            lab_val[idx] = float(v)
                        except (TypeError, ValueError):
                            lab_val[idx] = 0.0
                        lab_present[idx] = 1.0
            elif e.kind == "vitals":
                for vit, v in e.values.items():
                    if vit in self.vital_index:
                        idx = self.vital_index[vit]
                        try:
                            vit_val[idx] = float(v)
                        except (TypeError, ValueError):
                            vit_val[idx] = 0.0
                        vit_present[idx] = 1.0
        age = float(traj.demographics.get("age", 0)) / 100.0
        sex = 1.0 if str(traj.demographics.get("sex", "")).lower().startswith("m") else 0.0
        return sym + lab_val + lab_present + vit_val + vit_present + [age, sex]

    # ── target: state labels at a snapshot ──
    def target_vectors(self, snapshot):
        """Return (values, mask, confidence) over the latent vocab."""
        values = [0.0] * self.n_states
        mask = [0.0] * self.n_states     # 1 where a label exists
        conf = [0.0] * self.n_states
        for lab in snapshot.labels:
            idx = self.state_index.get(lab.state)
            if idx is None:
                continue
            signed = lab.value if lab.direction == "high" else -lab.value
            values[idx] = signed
            mask[idx] = 1.0
            conf[idx] = lab.confidence
        return values, mask, conf


def build_examples(jsonl_path: str, vocab_path: str | None = None):
    """
    Produce a flat list of training examples. Each example is a dict:
      {context, target_values, target_mask, target_conf, t_context, t_target, patient_id}
    Includes both:
      - same-time examples (context@t, target@t)            → masked-state task
      - future examples    (context@t, target@t' for t'>t)  → future-state task
    """
    vocab = load_vocab(vocab_path)
    fz = Featurizer(vocab)
    trajs = read_jsonl(jsonl_path)
    examples = []
    for tr in trajs:
        snaps = sorted(tr.state_snapshots, key=lambda s: s.t_hours)
        for i, snap in enumerate(snaps):
            ctx = fz.context_vector(tr, snap.t_hours)
            vals, mask, conf = fz.target_vectors(snap)
            if sum(mask) == 0:
                continue
            # same-time (masked-state task)
            examples.append({
                "patient_id": tr.patient_id, "t_context": snap.t_hours,
                "t_target": snap.t_hours, "delta_h": 0.0,
                "context": ctx,
                "target_context": ctx,
                "context_values": vals, "context_mask": mask, "context_conf": conf,
                "target_values": vals, "target_mask": mask, "target_conf": conf,
            })
            # future tasks: context now, target at each later snapshot
            for j in range(i + 1, len(snaps)):
                fut = snaps[j]
                fctx = fz.context_vector(tr, fut.t_hours)
                fv, fm, fc = fz.target_vectors(fut)
                if sum(fm) == 0:
                    continue
                examples.append({
                    "patient_id": tr.patient_id, "t_context": snap.t_hours,
                    "t_target": fut.t_hours, "delta_h": fut.t_hours - snap.t_hours,
                    "context": ctx,
                    "target_context": fctx,
                    "context_values": vals, "context_mask": mask, "context_conf": conf,
                    "target_values": fv,
                    "target_mask": fm, "target_conf": fc,
                })
    return examples, vocab


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", nargs="?", default=os.path.join(_HERE, "osler_coverage_trajectories.jsonl"))
    args = ap.parse_args()

    examples, vocab = build_examples(args.jsonl)
    print(f"Built {len(examples)} training examples from {os.path.basename(args.jsonl)}")
    print(f"  context_dim = {Featurizer(vocab).context_dim}")
    print(f"  latent (target) dim = {len(vocab['states'])}")
    same = sum(1 for e in examples if e['delta_h'] == 0)
    fut = len(examples) - same
    print(f"  same-time (masked-state) examples: {same}")
    print(f"  future-state examples: {fut}")
    if examples:
        e = examples[0]
        labeled = int(sum(e['target_mask']))
        print(f"  example 0: patient={e['patient_id']} t={e['t_context']}h "
              f"labeled_states={labeled}/{len(e['target_mask'])}")
    # save a featurized cache (JSON for portability)
    out = os.path.join(_HERE, "featurized_examples.json")
    with open(out, "w") as f:
        json.dump({"context_dim": Featurizer(vocab).context_dim,
                   "latent_dim": len(vocab["states"]),
                   "n_examples": len(examples),
                   "examples": examples[:50]}, f)  # sample only, to keep size sane
    print(f"  wrote sample cache: {out} (first 50 examples)")
