"""
featurize_seq.py — event-stream featurization for the Transformer encoder.

The MLP encoder (featurize.py) collapses all events up to time t into ONE
fixed vector. A Transformer instead consumes the events as a SEQUENCE of
tokens, so it can attend across individual observations and use their relative
timing. This module produces that token sequence.

One token per atomic observation:
  - each symptom            → token (modality=SYMPTOM, feat_id=symptom index)
  - each (lab, value)       → token (modality=LAB,     feat_id=lab index, value=value)
  - each (vital, value)     → token (modality=VITAL,   feat_id=vital index, value=value)
  - each drug               → token (modality=DRUG,    feat_id=hashed drug id)
Each token also carries t_hours (its time) so the encoder can build a temporal
position embedding.

Output per example mirrors featurize.build_examples but with a `seq` instead of
a flat `context`:
  seq = [ {modality, feat_id, value, t_hours}, ... ]     (variable length)
Targets (state labels + mask + conf) are identical to the MLP path, so the
training loop and probes are unchanged.
"""
from __future__ import annotations
import sys, os, json, hashlib

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from trajectory_schema import read_jsonl
from featurize import load_vocab, Featurizer

# modality ids (0 reserved for PAD)
MOD_PAD, MOD_SYMPTOM, MOD_LAB, MOD_VITAL, MOD_DRUG = 0, 1, 2, 3, 4
N_MODALITIES = 5
N_DRUG_BUCKETS = 256   # drugs hashed into this many buckets (open vocab)


def _drug_bucket(name: str) -> int:
    h = int(hashlib.md5(str(name).lower().encode()).hexdigest(), 16)
    return h % N_DRUG_BUCKETS


class SeqFeaturizer:
    def __init__(self, vocab: dict):
        self.vocab = vocab
        self.symptom_index = {s: i for i, s in enumerate(vocab["input_symptoms"])}
        self.lab_index = {l: i for i, l in enumerate(vocab["input_labs"])}
        self.vital_index = {v: i for i, v in enumerate(vocab["input_vitals"])}
        # feat_id ranges per modality are kept separate via (modality, feat_id);
        # the model has one embedding table per modality.
        self.n_sym = len(self.symptom_index)
        self.n_lab = len(self.lab_index)
        self.n_vit = len(self.vital_index)
        # reuse the MLP featurizer for the target side (identical labels)
        self._tgt = Featurizer(vocab)
        self.n_states = self._tgt.n_states

    def tokens_up_to(self, traj, t_hours: float) -> list:
        """All event-tokens with t <= t_hours, time-ordered."""
        toks = []
        for e in traj.events:
            if e.t_hours > t_hours:
                continue
            if e.kind == "symptoms":
                for s in e.values:
                    if s in self.symptom_index:
                        toks.append({"modality": MOD_SYMPTOM,
                                     "feat_id": self.symptom_index[s],
                                     "value": 0.0, "t_hours": e.t_hours})
            elif e.kind == "labs":
                for lab, v in e.values.items():
                    if lab in self.lab_index:
                        toks.append({"modality": MOD_LAB,
                                     "feat_id": self.lab_index[lab],
                                     "value": _safe_float(v), "t_hours": e.t_hours})
            elif e.kind == "vitals":
                for vit, v in e.values.items():
                    if vit in self.vital_index:
                        toks.append({"modality": MOD_VITAL,
                                     "feat_id": self.vital_index[vit],
                                     "value": _safe_float(v), "t_hours": e.t_hours})
            elif e.kind == "intervention":
                drug = e.values.get("drug")
                if drug:
                    toks.append({"modality": MOD_DRUG,
                                 "feat_id": _drug_bucket(drug),
                                 "value": 0.0, "t_hours": e.t_hours})
            # imaging/note: skipped here (no clean numeric token); add later.
        toks.sort(key=lambda x: x["t_hours"])
        return toks

    def target_vectors(self, snapshot):
        return self._tgt.target_vectors(snapshot)


def _safe_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def build_seq_examples(jsonl_path: str, vocab_path: str | None = None):
    """Same example list as featurize.build_examples, but each example has
    `seq` (list of token dicts) instead of `context` (flat vector)."""
    vocab = load_vocab(vocab_path)
    fz = SeqFeaturizer(vocab)
    trajs = read_jsonl(jsonl_path)
    examples = []
    for tr in trajs:
        snaps = sorted(tr.state_snapshots, key=lambda s: s.t_hours)
        for i, snap in enumerate(snaps):
            seq = fz.tokens_up_to(tr, snap.t_hours)
            if not seq:
                continue
            vals, mask, conf = fz.target_vectors(snap)
            if sum(mask) == 0:
                continue
            examples.append({
                "patient_id": tr.patient_id, "t_context": snap.t_hours,
                "t_target": snap.t_hours, "delta_h": 0.0,
                "seq": seq, "target_values": vals,
                "target_mask": mask, "target_conf": conf,
            })
            for j in range(i + 1, len(snaps)):
                fut = snaps[j]
                fv, fm, fc = fz.target_vectors(fut)
                if sum(fm) == 0:
                    continue
                examples.append({
                    "patient_id": tr.patient_id, "t_context": snap.t_hours,
                    "t_target": fut.t_hours, "delta_h": fut.t_hours - snap.t_hours,
                    "seq": seq, "target_values": fv,
                    "target_mask": fm, "target_conf": fc,
                })
    return examples, vocab


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", nargs="?",
                    default=os.path.join(_HERE, "demo_trajectories.jsonl"))
    args = ap.parse_args()
    examples, vocab = build_seq_examples(args.jsonl)
    print(f"Built {len(examples)} seq examples from {os.path.basename(args.jsonl)}")
    if examples:
        e = examples[0]
        print(f"  example 0: patient={e['patient_id']} seq_len={len(e['seq'])} "
              f"labeled_states={int(sum(e['target_mask']))}")
        print("  first tokens:")
        names = {MOD_SYMPTOM: "SYM", MOD_LAB: "LAB", MOD_VITAL: "VIT", MOD_DRUG: "DRUG"}
        for tk in e["seq"][:8]:
            print(f"    {names.get(tk['modality'],'?'):4s} feat={tk['feat_id']:4d} "
                  f"val={tk['value']:7.2f} t={tk['t_hours']}h")
        lens = [len(x["seq"]) for x in examples]
        print(f"  seq length: min={min(lens)} max={max(lens)} mean={sum(lens)/len(lens):.1f}")
