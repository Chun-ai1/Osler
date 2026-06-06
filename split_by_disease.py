"""
split_by_disease.py — make a held-out synthetic eval by DISEASE, not by event.

The point (per the roadmap): some diseases appear ONLY in the test set, never
in training. If the model scores well on held-out diseases, it learned general
mechanism (symptom→state mapping that transfers); if it collapses, it just
memorized per-disease patterns. Random event-level splitting can't show this
because the same disease leaks into both sides.

We also report, for each held-out disease, how much of its organ-state
"vocabulary" was seen in training — so a low score on a disease whose states
never appear in training is expected (no basis to generalize), while a low
score on a disease whose states ARE well-represented is the real failure signal.

Usage:
    python3 split_by_disease.py --in osler_coverage_trajectories.jsonl \
        --frac_heldout 0.2 --seed 0
    → train_synthetic.jsonl, heldout_synthetic.jsonl, split_report.json
"""
from __future__ import annotations
import argparse, os, sys, json, random, collections

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from trajectory_schema import read_jsonl, write_jsonl


def _states_of(traj):
    return {l.state for s in traj.state_snapshots for l in s.labels}


def split(in_path, frac_heldout=0.2, seed=0):
    ds = read_jsonl(in_path)
    rng = random.Random(seed)

    # each trajectory is one disease; shuffle diseases and hold out a fraction
    order = list(range(len(ds)))
    rng.shuffle(order)
    n_hold = max(1, int(len(ds) * frac_heldout))
    hold_idx = set(order[:n_hold])

    train = [ds[i] for i in range(len(ds)) if i not in hold_idx]
    heldout = [ds[i] for i in range(len(ds)) if i in hold_idx]

    # which states does training cover?
    train_states = set()
    for t in train:
        train_states |= _states_of(t)

    # per held-out disease: state coverage by training
    report = {"n_total": len(ds), "n_train": len(train), "n_heldout": len(heldout),
              "frac_heldout": frac_heldout, "seed": seed,
              "train_state_coverage": len(train_states), "heldout_diseases": []}
    for t in heldout:
        st = _states_of(t)
        seen = st & train_states
        report["heldout_diseases"].append({
            "disease": t.outcomes.get("discharge_dx", t.patient_id),
            "n_states": len(st),
            "states_seen_in_train": len(seen),
            "coverage": round(len(seen) / len(st), 2) if st else 0.0,
        })
    # sort report by coverage so it's easy to read who SHOULD be predictable
    report["heldout_diseases"].sort(key=lambda d: -d["coverage"])

    return train, heldout, report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=os.path.join(_HERE, "osler_coverage_trajectories.jsonl"))
    ap.add_argument("--frac_heldout", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    train, heldout, report = split(args.inp, args.frac_heldout, args.seed)
    write_jsonl(train, os.path.join(_HERE, "train_synthetic.jsonl"))
    write_jsonl(heldout, os.path.join(_HERE, "heldout_synthetic.jsonl"))
    json.dump(report, open(os.path.join(_HERE, "split_report.json"), "w"), indent=2, ensure_ascii=False)

    print(f"train_synthetic.jsonl:   {report['n_train']} diseases")
    print(f"heldout_synthetic.jsonl: {report['n_heldout']} diseases (NONE in training)")
    print(f"training covers {report['train_state_coverage']} distinct states\n")
    print("Held-out diseases, by how much of their state vocab training saw:")
    print(f"  {'disease':<38s} states  seen  coverage")
    for d in report["heldout_diseases"]:
        print(f"  {d['disease'][:36]:<38s} {d['n_states']:>4d}  {d['states_seen_in_train']:>4d}  {d['coverage']:>6.0%}")
    # the punchline
    well = [d for d in report["heldout_diseases"] if d["coverage"] >= 0.8]
    poor = [d for d in report["heldout_diseases"] if d["coverage"] < 0.4]
    print(f"\n  {len(well)} held-out diseases have >=80% of their states seen in training")
    print(f"    → these are the fair test of mechanism generalization")
    print(f"  {len(poor)} have <40% → low scores there are EXPECTED (no basis to generalize)")
