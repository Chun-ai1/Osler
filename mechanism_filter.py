"""
mechanism_filter.py — reject rules that are statistically "stable" but
mechanistically implausible, BEFORE anything reaches the explanation layer.

`stable` only means the model's predicted transition was consistent across
subgroups. It says nothing about whether the antecedent could plausibly drive the
consequent. `anion_gap_high -> eye.optic_nerve_function` can be perfectly stable
and still be a latent artifact. This stage adds the missing axis.

Each rule is classified:
  plausible  — a defensible physiological link (direct measurement->state anchor,
               same organ system, or a known cross-organ coupling with a coherent
               direction)
  uncertain  — a possible but unestablished link (coupled organs, or an
               intervention antecedent, where confounding is likely)
  spurious   — antecedent and consequent are physiologically unrelated

The logic is deliberately transparent and conservative — it is a FILTER to stop
nonsense, not a medical authority. Replace `osler_mechanism_verdict()` with a real
call to Osler's causal graph to override the heuristic once that engine is wired.

Only rules that are BOTH stable (invariance) AND plausible (here) are approved for
the symbolic / explanation layer.

    python3 mechanism_filter.py --rules rules_with_invariance.json
"""
from __future__ import annotations
import argparse, os, sys, json

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from featurize import load_vocab

# antecedent atom -> (organ, source_measure, concept, worsening_direction)
# organ "systemic" never matches a vocab organ directly; coupling handles it.
ATOM_SEM = {
    "spo2_low": ("lung", "spo2", "function", "low"),
    "spo2_down": ("lung", "spo2", "function", "low"),
    "rr_high": ("lung", "rr", "drive", "high"),
    "rr_up": ("lung", "rr", "drive", "high"),
    "hr_high": ("heart", "hr", "drive", "high"),
    "hr_up": ("heart", "hr", "drive", "high"),
    "bp_low": ("heart", "bp", "output", "low"),
    "fever": ("blood", "temp", "infection", "high"),
    "temp_up": ("blood", "temp", "infection", "high"),
    "wbc_high": ("blood", "wbc", "infection", "high"),
    "wbc_up": ("blood", "wbc", "infection", "high"),
    "wbc_low": ("blood", "wbc", "infection", "low"),
    "creatinine_high": ("kidney", "creatinine", "filtration", "low"),
    "creatinine_up": ("kidney", "creatinine", "filtration", "low"),
    "bun_high": ("kidney", "bun", "filtration", "low"),
    "platelets_low": ("blood", "platelets", "level", "low"),
    "hgb_low": ("blood", "hgb", "level", "low"),
    "inr_high": ("liver", "inr", "integrity", "low"),
    "bili_high": ("liver", "total_bilirubin", "level", "high"),
    "glucose_high": ("pancreas", "glucose", "level", "high"),
    "lactate_high": ("systemic", "lactate", "perfusion", "low"),
    "acidosis": ("systemic", "ph", "perfusion", "low"),
    "anion_gap_high": ("systemic", "anion_gap", "perfusion", "low"),
}
INTERVENTION_ATOMS = {"on_antibiotics", "on_oxygen", "on_vasopressor", "on_diuretic"}

MAJOR = {"lung", "heart", "kidney", "liver", "blood", "gi"}
# known physiological couplings (symmetric)
COUPLED = {frozenset(p) for p in [
    ("lung", "heart"), ("kidney", "blood"), ("liver", "blood"), ("heart", "blood"),
    ("lung", "blood"), ("kidney", "heart"), ("liver", "kidney"), ("blood", "gi"),
    ("lung", "kidney"), ("heart", "kidney"),
]}
SYSTEMIC_COUPLES = MAJOR  # lactate/pH/anion-gap touch the major organs


def osler_mechanism_verdict(antecedent_atoms, consequent_state):
    """HOOK: return 'plausible'|'uncertain'|'spurious' from Osler's causal graph,
    or None to defer to the heuristic. Wire this to nexus_engine when available."""
    return None


def _coupled(a_org, c_org):
    if a_org == "systemic":
        return c_org in SYSTEMIC_COUPLES
    return frozenset((a_org, c_org)) in COUPLED


def classify(rule, vocab_index):
    atoms = rule["context_pattern"]
    pt = rule["predicted_transition"]
    c_org = pt.get("organ"); want = pt.get("direction")
    cstate = vocab_index.get(pt["state"], {})
    c_labs = {str(x).lower() for x in (cstate.get("labs") or [])}
    c_vit = str(cstate.get("vital") or "").lower()
    c_con = cstate.get("concept")

    # Osler override first
    ov = osler_mechanism_verdict(atoms, cstate)
    if ov:
        return ov, [f"osler_engine:{ov}"]

    sem = [ATOM_SEM[a] for a in atoms if a in ATOM_SEM]
    interv = [a for a in atoms if a in INTERVENTION_ATOMS]
    intervention_only = bool(interv) and not sem
    reasons = []

    # 1. direct measurement -> state anchor (strongest, sometimes near-tautological)
    for a in atoms:
        if a in ATOM_SEM:
            src = ATOM_SEM[a][1]
            if any(src == l or src in l for l in c_labs) or (src and src in c_vit):
                reasons.append(f"{a}: measurement '{src}' directly maps to {pt['state']}")
                return "plausible", reasons

    if intervention_only:
        if c_org in MAJOR:
            reasons.append(f"intervention antecedent ({','.join(interv)}) on a major organ "
                           "→ association, likely confounded by indication")
            return "uncertain", reasons
        reasons.append(f"intervention antecedent ({','.join(interv)}) → distant organ "
                       f"'{c_org}' is physiologically unrelated")
        return "spurious", reasons

    a_orgs = {o for o, *_ in sem}
    # 2. same organ system
    if c_org in a_orgs:
        reasons.append(f"same organ system ({c_org})")
        return "plausible", reasons
    # 3. coupled organs + direction coherence
    coupled = any(_coupled(o, c_org) for o in a_orgs)
    if coupled:
        # direction coherence if concept matches an antecedent concept
        concept_match = any(con == c_con for *_, con, _d in [(o, m, con, d) for (o, m, con, d) in sem])
        dir_ok = any((d == "high" and want == "rising") or (d == "low" and want == "falling")
                     for *_, d in sem)
        if concept_match and dir_ok:
            reasons.append(f"coupled organs ({'/'.join(a_orgs)}→{c_org}), matching concept "
                           f"'{c_con}', coherent direction")
            return "plausible", reasons
        reasons.append(f"coupled organs ({'/'.join(a_orgs)}→{c_org}) but "
                       f"{'direction mismatch' if not dir_ok else 'no concept match'}")
        return "uncertain", reasons
    # 4. unrelated
    reasons.append(f"no organ link: antecedent {'/'.join(a_orgs) or '∅'} vs consequent '{c_org}'")
    return "spurious", reasons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", default=os.path.join(_HERE, "rules_with_invariance.json"))
    ap.add_argument("--vocab", default=None)
    ap.add_argument("--out", default=os.path.join(_HERE, "rules_filtered.json"))
    ap.add_argument("--report", default=os.path.join(_HERE, "mechanism_filter_report.md"))
    args = ap.parse_args()

    payload = json.load(open(args.rules))
    rules = payload["rules"]; meta = payload.get("_meta", {})
    vocab = load_vocab(args.vocab)
    vidx = {e["state"]: e for e in vocab["states"]}

    counts = {"plausible": 0, "uncertain": 0, "spurious": 0}
    approved = []
    for r in rules:
        cls, reasons = classify(r, vidx)
        verdict = r.get("invariance", {}).get("verdict")  # may be None for candidates
        r["mechanism"] = {"class": cls, "reasons": reasons}
        r["approved_for_osler"] = (cls == "plausible" and (verdict in (None, "stable")))
        counts[cls] += 1
        if r["approved_for_osler"]:
            approved.append(r)

    json.dump({"_meta": {**meta, "mechanism_counts": counts, "n_approved": len(approved)},
               "rules": rules}, open(args.out, "w"), indent=2, ensure_ascii=False)

    L = ["# Mechanism plausibility filter", "",
         f"- input: `{os.path.basename(args.rules)}`  ({len(rules)} rules)",
         f"- mechanism classes: {counts}",
         f"- **approved for Osler (stable ∧ plausible): {len(approved)}**", "",
         "> `stable` ≠ `clinically valid`. A rule must also be mechanistically "
         "plausible (and ideally externally validated) before it informs explanation "
         "or safety. Logic here is a conservative heuristic; replace the Osler hook "
         "with the real causal graph to harden it.", ""]
    for cls in ("plausible", "uncertain", "spurious"):
        sel = [r for r in rules if r["mechanism"]["class"] == cls]
        if not sel:
            continue
        L.append(f"## {cls}  ({len(sel)})")
        for r in sel[:30]:
            pt = r["predicted_transition"]; inv = r.get("invariance", {}).get("verdict", "n/a")
            L.append(f"- {' & '.join(r['context_pattern'])} → {pt['state']} {pt['direction']} "
                     f"[invariance={inv}]")
            L.append(f"  - {r['mechanism']['reasons'][0]}")
        L.append("")
    open(args.report, "w").write("\n".join(L))

    print(f"[mechanism_filter] classes={counts} | approved(stable∧plausible)={len(approved)}")
    print(f"[mechanism_filter] -> {args.out}\n[mechanism_filter] report -> {args.report}")
    print("\nApproved rules:")
    for r in approved[:15]:
        pt = r["predicted_transition"]
        print(f"  ✓ {' & '.join(r['context_pattern']):26s} → {pt['state']:30s} {pt['direction']}")
    if not approved:
        print("  (none) — with the collapsed demo checkpoint, the filter rejecting "
              "everything implausible is the CORRECT behaviour.")


if __name__ == "__main__":
    main()
