"""
rules_common.py — shared machinery for rule discovery + invariance testing.

A "window" is one (patient, context-time) decision point. For each window we record:
  - antecedent ATOMS  : binary facts derived from the RAW events (observed, not
                        the model) e.g. spo2_down, rr_up, wbc_high, on_antibiotics
  - model DELTA       : the CTM-JEPA's predicted change in every latent state over
                        a horizon Δ  (state_head(predictor(z,Δ)) - state_head(z))
  - subgroup keys     : age band, sex, baseline-severity band

A rule candidate then says: "when antecedent A holds, the model predicts latent
state S moves in direction d over Δ." extract_rules.py mines these; invariance_test.py
checks whether each one survives across subgroups. Nothing here is a clinical claim —
these are descriptions of the model's learned dynamics, to be stress-tested.
"""
from __future__ import annotations
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch
from trajectory_schema import read_jsonl
from featurize_seq import SeqFeaturizer
from infer_patient import load_model, _seq_batch   # reuse the verified encode path

# ── antecedent atom definitions (observed, model-independent) ──────────────
LEVEL_RULES = {  # atom : (name, comparator, threshold)
    "spo2_low":        ("spo2", "<", 94),
    "rr_high":         ("rr", ">", 22),
    "hr_high":         ("hr", ">", 100),
    "bp_low":          ("bp_sys", "<", 90),
    "fever":           ("temp", ">", 38.0),
    "wbc_high":        ("WBC", ">", 11),
    "wbc_low":         ("WBC", "<", 4),
    "creatinine_high": ("creatinine", ">", 1.3),
    "bun_high":        ("BUN", ">", 20),
    "platelets_low":   ("platelets", "<", 150),
    "hgb_low":         ("Hgb", "<", 10),
    "inr_high":        ("INR", ">", 1.4),
    "bili_high":       ("total_bilirubin", ">", 1.2),
    "glucose_high":    ("glucose", ">", 180),
    "lactate_high":    ("lactate", ">", 2.0),
    "acidosis":        ("pH", "<", 7.35),
    "anion_gap_high":  ("anion_gap", ">", 12),
}
TREND_RULES = {  # atom : (name, direction, min_abs_change)
    "spo2_down":       ("spo2", "down", 1.0),
    "rr_up":           ("rr", "up", 1.0),
    "hr_up":           ("hr", "up", 3.0),
    "creatinine_up":   ("creatinine", "up", 0.2),
    "temp_up":         ("temp", "up", 0.3),
    "wbc_up":          ("WBC", "up", 1.0),
}
ABX = ("ceftriaxone", "vancomycin", "piperacillin", "tazobactam", "meropenem",
       "cefepime", "azithromycin", "metronidazole", "levofloxacin", "ciprofloxacin",
       "amoxicillin", "ampicillin")
PRESSORS = ("norepinephrine", "epinephrine", "vasopressin", "phenylephrine", "dopamine")
DIURETICS = ("furosemide", "bumetanide", "torsemide")


def _series(traj, name, t_lo, t_hi):
    """(t, value) for one vital/lab name within (t_lo, t_hi], time-ordered."""
    out = []
    for e in traj.events:
        if e.t_hours <= t_lo or e.t_hours > t_hi:
            continue
        if e.kind in ("vitals", "labs") and isinstance(e.values, dict):
            for k, v in e.values.items():
                if str(k).lower() == name.lower():
                    try: out.append((e.t_hours, float(v)))
                    except (TypeError, ValueError): pass
    out.sort(key=lambda x: x[0])
    return out


def extract_atoms(traj, t_context, trend_window=12.0):
    atoms = set()
    lo = t_context - trend_window
    for atom, (name, cmp_, thr) in LEVEL_RULES.items():
        s = _series(traj, name, lo - 1e9, t_context)   # last value up to t_context
        if s:
            x = s[-1][1]
            if (x < thr) if cmp_ == "<" else (x > thr):
                atoms.add(atom)
    for atom, (name, direction, mc) in TREND_RULES.items():
        s = _series(traj, name, lo, t_context)
        if len(s) >= 2:
            d = s[-1][1] - s[0][1]
            if (d <= -mc) if direction == "down" else (d >= mc):
                atoms.add(atom)
    for e in traj.events:
        if e.t_hours > t_context or e.kind != "intervention" or not isinstance(e.values, dict):
            continue
        blob = " ".join(str(v).lower() for v in e.values.values())
        if any(a in blob for a in ABX): atoms.add("on_antibiotics")
        if "oxygen" in blob or "cannula" in blob or e.values.get("route") == "nasal_cannula": atoms.add("on_oxygen")
        if any(p in blob for p in PRESSORS): atoms.add("on_vasopressor")
        if any(d in blob for d in DIURETICS): atoms.add("on_diuretic")
    return atoms


def _baseline_severity(atoms):
    """Crude observed-severity score = count of abnormal level atoms present."""
    return sum(1 for a in atoms if a in LEVEL_RULES)


def subgroups_of(traj, atoms, sev_median):
    age = traj.demographics.get("age")
    sex = str(traj.demographics.get("sex", "")).lower()
    g = {}
    if isinstance(age, (int, float)):
        g["age"] = "<40" if age < 40 else "40-65" if age <= 65 else ">65"
    if sex.startswith("m"): g["sex"] = "male"
    elif sex.startswith("f"): g["sex"] = "female"
    g["baseline_severity"] = "high" if _baseline_severity(atoms) > sev_median else "low"
    return g


def build_window_table(ckpt, data, delta_h=24.0, trend_window=12.0,
                       max_windows=6, max_seq=None, device=None, verbose=True):
    """Return (rows, vocab, encoder). Each row: dict with patient_id, atoms,
    delta (list[n_states]), subgroups."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab, encoder = load_model(ckpt, device)
    sf = SeqFeaturizer(vocab)
    trajs = read_jsonl(data)
    delta = torch.tensor([[delta_h]], dtype=torch.float32, device=device)

    # first pass: collect baseline-severity values for the median split
    sev_vals = []
    prelim = []
    for tr in trajs:
        times = sorted({s.t_hours for s in tr.state_snapshots}) or \
                sorted({e.t_hours for e in tr.events})
        times = times[:max_windows]
        for t in times:
            atoms = extract_atoms(tr, t, trend_window)
            sev_vals.append(_baseline_severity(atoms))
            prelim.append((tr, t, atoms))
    sev_median = sorted(sev_vals)[len(sev_vals)//2] if sev_vals else 0

    rows = []
    for tr, t, atoms in prelim:
        toks = sf.tokens_up_to(tr, t)
        if max_seq and len(toks) > max_seq:
            toks = toks[-max_seq:]
        batch = _seq_batch(toks, device) if encoder in ("transformer", "ctm") else None
        if batch is None:  # mlp fallback
            from featurize import Featurizer
            vec = Featurizer(vocab).context_vector(tr, t)
            ctx = torch.tensor([vec], dtype=torch.float32, device=device)
        else:
            ctx = batch
        with torch.no_grad():
            z = model.context_encoder(ctx)
            cur = model.state_head(z)[0]
            fut = model.state_head(model.predictor(z, delta))[0]
        d = torch.nan_to_num(fut - cur).cpu().tolist()
        rows.append({"patient_id": tr.patient_id, "t_context": t, "atoms": sorted(atoms),
                     "delta": d, "subgroups": subgroups_of(tr, atoms, sev_median)})
    if verbose:
        print(f"[rules_common] {len(rows)} windows | {len(trajs)} patients | "
              f"encoder={encoder} | Δ={delta_h}h | sev_median={sev_median}")
    return rows, vocab, encoder


def all_atoms(rows):
    s = set()
    for r in rows: s |= set(r["atoms"])
    return sorted(s)
