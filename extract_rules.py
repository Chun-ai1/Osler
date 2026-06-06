"""
extract_rules.py — mine model-discovered clinical transition rules from a
trained CTM-JEPA.

For each antecedent (a single atom like `spo2_down`, or a co-occurring pair like
`spo2_down & rr_up`) we ask: among windows where the antecedent holds, which
latent states does the model push, and by how much vs windows where it doesn't?
The shift is a Cohen's d effect size; the sign agreement is the confidence.

Output: rules_candidates.json — a list of checkable rules, every one tagged
status="hypothesis_not_clinical_rule".

    python3 extract_rules.py --ckpt checkpoints/medical_jepa_ctm.pt \
        --data mimic_probe_bc_24h_train_nonan.jsonl --delta_h 24
"""
from __future__ import annotations
import argparse, os, sys, json, itertools
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from rules_common import build_window_table, all_atoms


def _mask(rows, antecedent):
    return np.array([all(a in r["atoms"] for a in antecedent) for r in rows], dtype=bool)


def _stats_for_state(D, m, s_idx):
    g1 = D[m, s_idx]; g0 = D[~m, s_idx]
    if len(g1) == 0 or len(g0) == 0:
        return None
    m1, m0 = float(g1.mean()), float(g0.mean())
    sd = float(np.sqrt(((g1.var() * len(g1)) + (g0.var() * len(g0))) / (len(g1) + len(g0)))) or 1e-6
    cohen_d = (m1 - m0) / sd
    direction = "rising" if m1 >= 0 else "falling"
    conf = float(np.mean(np.sign(g1) == np.sign(m1))) if len(g1) else 0.0
    return {"mean_delta_with": round(m1, 4), "mean_delta_without": round(m0, 4),
            "cohen_d": round(cohen_d, 4), "direction": direction, "confidence": round(conf, 3)}


def mine(rows, vocab, antecedents, states, top_states=3,
         min_windows=8, min_patients=4, min_abs_effect=0.15):
    D = np.array([r["delta"] for r in rows])
    out = []
    for ante in antecedents:
        m = _mask(rows, ante)
        n_w = int(m.sum())
        n_p = len({rows[i]["patient_id"] for i in np.where(m)[0]})
        if n_w < min_windows or n_p < min_patients:
            continue
        scored = []
        for si in range(D.shape[1]):
            st = _stats_for_state(D, m, si)
            if st and abs(st["cohen_d"]) >= min_abs_effect:
                scored.append((abs(st["cohen_d"]) * st["confidence"], si, st))
        scored.sort(reverse=True, key=lambda x: x[0])
        for _, si, st in scored[:top_states]:
            s = states[si]
            out.append({
                "rule_id": f"{'__'.join(ante)}->{s['state']}",
                "context_pattern": list(ante),
                "predicted_transition": {"state": s["state"], "organ": s.get("organ"),
                                          "direction": st["direction"], "horizon_h": None},
                "support": {"n_patients": n_p, "n_windows": n_w,
                             "effect_size_cohen_d": st["cohen_d"], "confidence": st["confidence"],
                             "mean_delta_with": st["mean_delta_with"],
                             "mean_delta_without": st["mean_delta_without"]},
                "observability": s.get("observability"),
                "status": "hypothesis_not_clinical_rule",
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(_HERE, "checkpoints", "medical_jepa_ctm.pt"))
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=os.path.join(_HERE, "rules_candidates.json"))
    ap.add_argument("--delta_h", type=float, default=24.0)
    ap.add_argument("--trend_window", type=float, default=12.0)
    ap.add_argument("--max_windows", type=int, default=6)
    ap.add_argument("--max_seq", type=int, default=None)
    ap.add_argument("--max_pairs", type=int, default=12, help="top co-occurring atom pairs to test")
    ap.add_argument("--top_states", type=int, default=3)
    ap.add_argument("--min_windows", type=int, default=8)
    ap.add_argument("--min_patients", type=int, default=4)
    args = ap.parse_args()

    rows, vocab, encoder = build_window_table(
        args.ckpt, args.data, delta_h=args.delta_h, trend_window=args.trend_window,
        max_windows=args.max_windows, max_seq=args.max_seq)
    states = vocab["states"]
    atoms = all_atoms(rows)

    # antecedents: every single atom + the most frequent co-occurring pairs
    singles = [(a,) for a in atoms]
    from collections import Counter
    pair_count = Counter()
    for r in rows:
        for p in itertools.combinations(sorted(r["atoms"]), 2):
            pair_count[p] += 1
    pairs = [p for p, _ in pair_count.most_common(args.max_pairs)]
    antecedents = singles + pairs

    rules = mine(rows, vocab, antecedents, states, top_states=args.top_states,
                 min_windows=args.min_windows, min_patients=args.min_patients)
    rules.sort(key=lambda r: -abs(r["support"]["effect_size_cohen_d"]) * r["support"]["confidence"])
    for r in rules:
        r["predicted_transition"]["horizon_h"] = args.delta_h

    payload = {"_meta": {"ckpt": os.path.basename(args.ckpt), "data": os.path.basename(args.data),
                          "encoder": encoder, "delta_h": args.delta_h,
                          "trend_window": args.trend_window, "max_windows": args.max_windows,
                          "max_seq": args.max_seq, "n_windows": len(rows),
                          "n_rules": len(rules),
                          "note": "model-discovered dynamics; NOT clinical rules"},
               "rules": rules}
    json.dump(payload, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"[extract_rules] {len(rules)} candidates -> {args.out}")
    for r in rules[:10]:
        s = r["support"]
        print(f"  {' & '.join(r['context_pattern']):28s} -> {r['predicted_transition']['state']:30s} "
              f"{r['predicted_transition']['direction']:7s} d={s['effect_size_cohen_d']:+.2f} "
              f"conf={s['confidence']:.2f} (n_pt={s['n_patients']}, n_w={s['n_windows']})")
    if rules:
        confs = sorted(r["support"]["confidence"] for r in rules)
        med_conf = confs[len(confs) // 2]
        max_eff = max(abs(r["support"]["effect_size_cohen_d"]) for r in rules)
        if med_conf >= 0.98 or max_eff < 0.1:
            print(f"  ⚠ COLLAPSE SIGNATURE (median confidence={med_conf:.2f}, "
                  f"max|d|={max_eff:.2f}). Near-constant model output makes every rule "
                  "trivially 'consistent' and clinically meaningless. This is expected "
                  "for a toy/under-trained checkpoint — the pipeline is being demonstrated, "
                  "not the rules. After real training, expect mixed confidences (<1.0), "
                  "plausible consequents, and a MIX of stable / subgroup_specific verdicts.")


if __name__ == "__main__":
    main()
