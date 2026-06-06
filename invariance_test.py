"""
invariance_test.py — stress-test each mined rule across patient subgroups.

A rule that only holds for, say, <65-year-olds is a subgroup-specific rule, not a
universal one — and that matters before any symbolic layer relies on it. For each
rule we re-measure the predicted transition within age / sex / baseline-severity
subgroups and report where it holds, where it flips, and where there isn't enough
support to tell.

    python3 invariance_test.py --ckpt checkpoints/medical_jepa_ctm.pt \
        --data mimic_probe_bc_24h_train_nonan.jsonl --rules rules_candidates.json
"""
from __future__ import annotations
import argparse, os, sys, json
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from rules_common import build_window_table

SUBGROUP_DIMS = ["age", "sex", "baseline_severity"]


def _idx_of_state(vocab, name):
    for e in vocab["states"]:
        if e["state"] == name:
            return e["index"]
    return None


def test_rule(rows, vocab, rule, min_windows=4):
    ante = rule["context_pattern"]
    si = _idx_of_state(vocab, rule["predicted_transition"]["state"])
    want = rule["predicted_transition"]["direction"]
    if si is None:
        return None
    sel = [r for r in rows if all(a in r["atoms"] for a in ante)]
    if not sel:
        return {"global": None, "by_subgroup": {}, "verdict": "no_support"}

    def measure(subset):
        if len(subset) < min_windows:
            return {"n_windows": len(subset), "n_patients": len({r["patient_id"] for r in subset}),
                    "insufficient": True}
        d = np.array([r["delta"][si] for r in subset])
        m = float(d.mean())
        conf = float(np.mean(np.sign(d) == (1 if want == "rising" else -1)))
        return {"n_windows": len(subset),
                "n_patients": len({r["patient_id"] for r in subset}),
                "mean_delta": round(m, 4),
                "direction": "rising" if m >= 0 else "falling",
                "matches_rule": (("rising" if m >= 0 else "falling") == want),
                "confidence": round(conf, 3)}

    glob = measure(sel)
    by = {}
    for dim in SUBGROUP_DIMS:
        for val in sorted({r["subgroups"].get(dim) for r in sel if r["subgroups"].get(dim)}):
            by[f"{dim}={val}"] = measure([r for r in sel if r["subgroups"].get(dim) == val])

    # verdict: stable if every testable subgroup matches the rule direction and
    # keeps >=50% of the global effect magnitude; else subgroup-specific.
    g_mag = abs(glob.get("mean_delta", 0.0)) if glob and not glob.get("insufficient") else 0.0
    holds, fails, untested = [], [], []
    for k, v in by.items():
        if v.get("insufficient"):
            untested.append(k)
        elif v["matches_rule"] and abs(v["mean_delta"]) >= 0.5 * g_mag:
            holds.append(k)
        else:
            fails.append(k)
    verdict = "stable" if (not fails and holds) else ("subgroup_specific" if holds else "unstable")
    return {"global": glob, "by_subgroup": by,
            "holds_in": holds, "fails_in": fails, "untested": untested, "verdict": verdict}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(_HERE, "checkpoints", "medical_jepa_ctm.pt"))
    ap.add_argument("--data", required=True)
    ap.add_argument("--rules", default=os.path.join(_HERE, "rules_candidates.json"))
    ap.add_argument("--report", default=os.path.join(_HERE, "invariance_report.md"))
    ap.add_argument("--out", default=os.path.join(_HERE, "rules_with_invariance.json"))
    ap.add_argument("--min_windows", type=int, default=4)
    args = ap.parse_args()

    payload = json.load(open(args.rules))
    meta = payload["_meta"]; rules = payload["rules"]
    rows, vocab, encoder = build_window_table(
        args.ckpt, args.data, delta_h=meta["delta_h"], trend_window=meta["trend_window"],
        max_windows=meta["max_windows"], max_seq=meta.get("max_seq"))

    annotated = []
    counts = {"stable": 0, "subgroup_specific": 0, "unstable": 0, "no_support": 0}
    for rule in rules:
        inv = test_rule(rows, vocab, rule, min_windows=args.min_windows)
        rule["invariance"] = inv
        if inv: counts[inv["verdict"]] = counts.get(inv["verdict"], 0) + 1
        annotated.append(rule)

    json.dump({"_meta": meta, "rules": annotated}, open(args.out, "w"), indent=2, ensure_ascii=False)

    lines = ["# Invariance report", "",
             f"- checkpoint: `{meta['ckpt']}`  | data: `{meta['data']}`  | encoder: {encoder}",
             f"- horizon Δ = {meta['delta_h']}h | windows = {len(rows)} | rules tested = {len(rules)}",
             f"- verdicts: {counts}", "",
             "> Rules describe the model's learned dynamics, stress-tested across "
             "age / sex / baseline-severity subgroups. None is a clinical rule.", ""]
    for verdict in ("stable", "subgroup_specific", "unstable"):
        sel = [r for r in annotated if r.get("invariance", {}).get("verdict") == verdict]
        if not sel:
            continue
        lines.append(f"## {verdict}  ({len(sel)})")
        for r in sel[:25]:
            inv = r["invariance"]; pt = r["predicted_transition"]; sup = r["support"]
            lines.append(f"- **{' & '.join(r['context_pattern'])} → {pt['state']} {pt['direction']}** "
                         f"(d={sup['effect_size_cohen_d']:+.2f}, conf={sup['confidence']:.2f}, "
                         f"n_pt={sup['n_patients']})")
            if inv.get("holds_in"):  lines.append(f"  - holds in: {', '.join(inv['holds_in'])}")
            if inv.get("fails_in"):  lines.append(f"  - FAILS in: {', '.join(inv['fails_in'])}")
            if inv.get("untested"):  lines.append(f"  - untested (low support): {', '.join(inv['untested'])}")
        lines.append("")
    open(args.report, "w").write("\n".join(lines))
    print(f"[invariance_test] verdicts={counts}")
    print(f"[invariance_test] report -> {args.report}")
    print(f"[invariance_test] annotated rules -> {args.out}")
    if encoder and counts["stable"] == 0 and len(rules) > 0:
        print("  note: 0 stable rules — with a collapsed/toy checkpoint this is expected; "
              "the pipeline itself is what's being demonstrated.")


if __name__ == "__main__":
    main()
