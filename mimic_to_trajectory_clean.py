"""
mimic_to_trajectory.py — convert MIMIC-IV into the canonical trajectory format.

This is the bridge from REAL de-identified ICU data to the JEPA training
format. It emits the exact same `Trajectory` JSONL that the synthetic
generators do, so train_jepa.py / featurize.py work unchanged — just point
--data at the output.

═══════════════════════════════════════════════════════════════════════════
 STATUS: SKELETON.  The control flow, time-alignment, event assembly, weak
 labeling, and train/eval split are all complete and correct. What is NOT
 filled in (because it requires actual MIMIC access to verify) are the
 MIMIC-specific ID→concept MAPPING TABLES. Every place you must fill in is
 marked  # TODO[MIMIC].  Do not trust the placeholder itemids — verify them
 against your copy of d_labitems / d_items.
═══════════════════════════════════════════════════════════════════════════

MIMIC-IV tables used (hosp + icu modules):
  patients            subject_id, gender, anchor_age
  admissions          hadm_id, admittime, dischtime, hospital_expire_flag
  icustays            stay_id, intime, outtime
  labevents           subject_id, hadm_id, itemid, charttime, valuenum, flag
  d_labitems          itemid → label (lab name)
  chartevents         stay_id, itemid, charttime, valuenum     (vitals live here)
  d_items             itemid → label (vital/chart name)
  prescriptions       hadm_id, drug, starttime, dose_val_rx, route
  diagnoses_icd       hadm_id, icd_code        (weak outcome label)
  d_icd_diagnoses     icd_code → long_title

Recommended: load via the official MIMIC-IV Postgres/BigQuery schema, or the
CSVs. This skeleton reads CSVs with pandas for portability; swap in SQL if you
have the database.

Usage (once mappings are filled and you have the CSVs):
    pip install pandas
    python3 mimic_to_trajectory.py \
        --mimic_dir /path/to/mimic-iv/ \
        --out mimic_trajectories.jsonl \
        --max_patients 5000
"""
from __future__ import annotations
import argparse, os, sys, json

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from trajectory_schema import (
    Trajectory, Event, StateSnapshot, StateLabel, write_jsonl, validate_dataset,
)
from weak_labeler import label_observation


# ══════════════════════════════════════════════════════════════════════
#  MAPPING TABLES — fill these in against YOUR MIMIC copy.   # TODO[MIMIC]
# ══════════════════════════════════════════════════════════════════════

# 1. MIMIC d_labitems.label  →  the lab name the weak_labeler knows.
#    The labeler's known lab vocabulary (67 names) is in lab_integration.json.
#    MIMIC lab labels are messy ("Creatinine", "Creatinine, Whole Blood", ...);
#    map each relevant one to the canonical name. Unmapped labs are ignored.
#    VERIFY every entry against d_labitems — these are PLAUSIBLE GUESSES ONLY.
LAB_LABEL_TO_CANON = {
    # MIMIC label (lowercased)        : canonical name used by weak_labeler
    "white blood cells":               "WBC",
    "wbc":                             "WBC",
    "creatinine":                      "creatinine",
    "urea nitrogen":                   "BUN",
    "c-reactive protein":              "CRP",
    "troponin t":                      "troponin",
    "troponin i":                      "troponin",
    "alanine aminotransferase (alt)":  "ALT",
    "asparate aminotransferase (ast)": "AST",
    "hemoglobin":                      "Hgb",
    "platelet count":                  "platelets",
    "lactate":                         "lactate",
    "bilirubin, total":                "total_bilirubin",
    "sodium":                          "sodium",
    "potassium":                       "potassium",
    "bicarbonate":                     "bicarbonate",
    "ph":                              "pH",
    "po2":                             "PaO2",
    "pco2":                            "PaCO2",
    "lipase":                          "lipase",
    "amylase":                         "amylase",
    "inr(pt)":                         "INR",
    "ptt":                             "PTT",
    "albumin":                         "albumin",
    "glucose":                         "glucose",
    "anion gap":                       "anion_gap",
    # TODO[MIMIC]: extend to cover the labs you care about, verify spellings.
}

# 2. MIMIC chartevents itemid (vitals)  →  the vital name the labeler knows.
#    These itemids are from the commonly-cited MIMIC-IV metavision set but
#    MUST be verified against d_items in your copy — itemids differ by version.
VITAL_ITEMID_TO_CANON = {
    # itemid : canonical vital
    220045: "hr",       # Heart Rate                      # TODO[MIMIC] verify
    220179: "bp_sys",   # Non Invasive BP systolic        # TODO[MIMIC] verify
    220180: "bp_dia",   # Non Invasive BP diastolic       # TODO[MIMIC] verify
    220210: "rr",       # Respiratory Rate                # TODO[MIMIC] verify
    220277: "spo2",     # O2 saturation pulseoxymetry     # TODO[MIMIC] verify
    223761: "temp",     # Temperature Fahrenheit          # TODO[MIMIC] verify (convert!)
    223762: "temp",     # Temperature Celsius             # TODO[MIMIC] verify
}
# vitals needing unit conversion → return (canon, converter_fn)
def _f_to_c(f):  # Fahrenheit → Celsius
    return (float(f) - 32.0) * 5.0 / 9.0
VITAL_CONVERTERS = {
    223761: _f_to_c,    # temp F → C   # TODO[MIMIC] confirm which itemid is F
}

# 3. Abnormal-direction inference for labs. The weak_labeler wants a flag
#    'high'/'low' per lab. MIMIC labevents has a `flag` column ('abnormal')
#    but not direction. Use reference ranges, or fall back to: compare to a
#    midpoint. Minimal version: treat MIMIC flag='abnormal' + value above the
#    canonical high-cutoff as 'high'. Provide cutoffs you trust.   # TODO[MIMIC]
LAB_HIGH_CUTOFF = {
    "WBC": 11.0, "creatinine": 1.3, "BUN": 20, "CRP": 10, "troponin": 0.04,
    "ALT": 56, "AST": 40, "lactate": 2.0, "total_bilirubin": 1.2,
    "lipase": 160, "glucose": 140, "Hgb": 17.0,   # Hgb: also check LOW
    # TODO[MIMIC]: set the cutoffs your institution/MIMIC uses.
}
LAB_LOW_CUTOFF = {
    "Hgb": 12.0, "platelets": 150, "sodium": 135, "potassium": 3.5,
    "bicarbonate": 22, "albumin": 3.5, "GFR": 60,
    # TODO[MIMIC]: extend.
}


# ══════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════

def _lab_flag(canon_name, value):
    """Return 'high' / 'low' / None for a lab value using the cutoffs above."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    hi = LAB_HIGH_CUTOFF.get(canon_name)
    lo = LAB_LOW_CUTOFF.get(canon_name)
    if hi is not None and v > hi:
        return "high"
    if lo is not None and v < lo:
        return "low"
    return None  # within range → not an abnormal observation


def _hours_since(t0, t):
    """Hours between two pandas Timestamps (or ISO strings)."""
    import pandas as pd
    if not isinstance(t0, pd.Timestamp):
        t0 = pd.to_datetime(t0)
    if not isinstance(t, pd.Timestamp):
        t = pd.to_datetime(t)
    return (t - t0).total_seconds() / 3600.0


# ══════════════════════════════════════════════════════════════════════
#  Core conversion: one ICU stay → one Trajectory
# ══════════════════════════════════════════════════════════════════════

def build_trajectory_for_stay(
    subject_id, stay_id, hadm_id,
    demographics,           # {"age":, "sex":}
    intime,                 # stay start timestamp (t0)
    labs_df,                # rows for this hadm: charttime, canon_lab, valuenum
    vitals_df,              # rows for this stay: charttime, canon_vital, valuenum
    rx_df,                  # rows for this hadm: starttime, drug, dose, route
    outcomes,               # {"icu_admission":bool, "death":bool, "discharge_dx":str}
    bin_hours=6.0,          # aggregate observations into time bins
):
    """
    Assemble events, bin them into timepoints, weak-label each bin's state.
    bin_hours controls temporal resolution (6h is a reasonable ICU default).
    """
    events = []

    # ── lab events ──
    lab_by_bin = {}   # bin_t -> {canon: (value, flag)}
    skipped_nonfinite_labs = 0
    for _, r in labs_df.iterrows():
        # Real EHR data can contain NaN / inf in valuenum. Never write those
        # into JSONL, because they poison featurize/train loss and produce NaN.
        try:
            value = float(r["valuenum"])
        except (TypeError, ValueError):
            skipped_nonfinite_labs += 1
            continue
        if value != value or value in (float("inf"), float("-inf")):
            skipped_nonfinite_labs += 1
            continue

        t = _hours_since(intime, r["charttime"])
        if t < 0:
            continue
        b = round(t / bin_hours) * bin_hours
        canon = r["canon_lab"]
        flag = _lab_flag(canon, value)
        events.append(Event(t, "labs", {canon: value}))
        lab_by_bin.setdefault(b, {})[canon] = (value, flag)

    # ── vital events ──
    vit_by_bin = {}
    skipped_nonfinite_vitals = 0
    for _, r in vitals_df.iterrows():
        try:
            value = float(r["valuenum"])
        except (TypeError, ValueError):
            skipped_nonfinite_vitals += 1
            continue
        if value != value or value in (float("inf"), float("-inf")):
            skipped_nonfinite_vitals += 1
            continue

        t = _hours_since(intime, r["charttime"])
        if t < 0:
            continue
        b = round(t / bin_hours) * bin_hours
        canon = r["canon_vital"]
        events.append(Event(t, "vitals", {canon: value}))
        vit_by_bin.setdefault(b, {})[canon] = value

    # ── intervention events ──
    for _, r in rx_df.iterrows():
        t = _hours_since(intime, r["starttime"])
        if t < 0:
            continue

        drug = str(r.get("drug", "")).strip().lower()
        if not drug or drug == "nan":
            continue

        dose = r.get("dose_val_rx")
        route = r.get("route")

        # Clean dose: never write NaN into JSONL
        if dose is None or dose != dose:
            dose = None
        else:
            dose = str(dose).strip()
            if not dose or dose.lower() == "nan":
                dose = None

        # Clean route: never write NaN into JSONL
        if route is None or route != route:
            route = None
        else:
            route = str(route).strip()
            if not route or route.lower() == "nan":
                route = None

        payload = {"drug": drug}
        if dose is not None:
            payload["dose"] = dose
        if route is not None:
            payload["route"] = route

        events.append(Event(t, "intervention", payload))

    events.sort(key=lambda e: e.t_hours)

    # ── weak-label each time bin ──
    snapshots = []
    all_bins = sorted(set(lab_by_bin) | set(vit_by_bin))
    for b in all_bins:
        labs = {k: v for k, (v, f) in lab_by_bin.get(b, {}).items()}
        lab_flags = {k: f for k, (v, f) in lab_by_bin.get(b, {}).items() if f}
        vitals = vit_by_bin.get(b, {})
        # NOTE: MIMIC has no clean symptom list; symptoms stay empty here.
        # (Symptoms could be NLP-extracted from notes — separate project.)
        labels = label_observation(symptoms=[], labs=labs, vitals=vitals,
                                   lab_flags=lab_flags,
                                   context=demographics)
        if labels:
            snapshots.append(StateSnapshot(
                float(b),
                [StateLabel(**l) for l in labels],
            ))

    return Trajectory(
        patient_id=f"mimic_{subject_id}_{stay_id}",
        source="real_ehr",
        demographics=demographics,
        events=events,
        state_snapshots=snapshots,
        outcomes=outcomes,
        meta={"hadm_id": int(hadm_id), "bin_hours": bin_hours,
              "generator": "mimic_to_trajectory.v1_skeleton"},
    )


# ══════════════════════════════════════════════════════════════════════
#  Driver: load CSVs, join, iterate stays
# ══════════════════════════════════════════════════════════════════════

def convert(mimic_dir, out_path, max_patients=None, bin_hours=6.0):
    import pandas as pd

    hosp = os.path.join(mimic_dir, "hosp")
    icu = os.path.join(mimic_dir, "icu")

    print("[mimic] loading dimension tables...")
    patients = pd.read_csv(os.path.join(hosp, "patients.csv.gz"))
    admissions = pd.read_csv(os.path.join(hosp, "admissions.csv.gz"),
                             parse_dates=["admittime", "dischtime"])
    icustays = pd.read_csv(os.path.join(icu, "icustays.csv.gz"),
                           parse_dates=["intime", "outtime"])
    d_labitems = pd.read_csv(os.path.join(hosp, "d_labitems.csv.gz"))

    # build itemid → canonical lab via the label map
    lab_label = dict(zip(d_labitems["itemid"],
                         d_labitems["label"].astype(str).str.lower()))
    labitem_to_canon = {
        itemid: LAB_LABEL_TO_CANON[lbl]
        for itemid, lbl in lab_label.items()
        if lbl in LAB_LABEL_TO_CANON
    }
    print(f"[mimic] mapped {len(labitem_to_canon)} lab itemids to canonical names")
    if not labitem_to_canon:
        print("[mimic] WARNING: no labs mapped — fill LAB_LABEL_TO_CANON. # TODO[MIMIC]")

    # choose patient subset
    stays = icustays.merge(patients[["subject_id", "gender", "anchor_age"]],
                           on="subject_id", how="left")
    stays = stays.merge(admissions[["hadm_id", "dischtime", "hospital_expire_flag"]],
                        on="hadm_id", how="left")
    if max_patients:
        keep = stays["subject_id"].drop_duplicates().head(max_patients)
        stays = stays[stays["subject_id"].isin(keep)]
    print(f"[mimic] {len(stays)} ICU stays to convert")

    # NOTE: for scale, read labevents/chartevents in chunks filtered by id.
    # This skeleton shows the logic on the assumption you can load per-stay
    # slices (or pre-filter the big tables). For the full dataset, replace
    # these with chunked reads or SQL queries.   # TODO[MIMIC]
    print("[mimic] loading labevents / chartevents (filter to mapped itemids)...")
    lab_usecols = ["subject_id", "hadm_id", "itemid", "charttime", "valuenum", "flag"]
    labevents = pd.read_csv(os.path.join(hosp, "labevents.csv.gz"),
                            usecols=lab_usecols, parse_dates=["charttime"])
    labevents = labevents[labevents["itemid"].isin(labitem_to_canon)]
    labevents["valuenum"] = pd.to_numeric(labevents["valuenum"], errors="coerce")
    labevents = labevents[labevents["valuenum"].notna()]
    labevents = labevents[labevents["valuenum"].apply(lambda x: x not in (float("inf"), float("-inf")))]
    labevents["canon_lab"] = labevents["itemid"].map(labitem_to_canon)

    chart_usecols = ["stay_id", "itemid", "charttime", "valuenum"]
    chartevents = pd.read_csv(os.path.join(icu, "chartevents.csv.gz"),
                              usecols=chart_usecols, parse_dates=["charttime"])
    chartevents = chartevents[chartevents["itemid"].isin(VITAL_ITEMID_TO_CANON)]
    chartevents["valuenum"] = pd.to_numeric(chartevents["valuenum"], errors="coerce")
    chartevents = chartevents[chartevents["valuenum"].notna()]
    chartevents = chartevents[chartevents["valuenum"].apply(lambda x: x not in (float("inf"), float("-inf")))]
    # apply converters (e.g. temp F→C) then map to canonical
    def _conv(row):
        fn = VITAL_CONVERTERS.get(row["itemid"])
        return fn(row["valuenum"]) if fn else row["valuenum"]
    chartevents["valuenum"] = chartevents.apply(_conv, axis=1)
    chartevents["canon_vital"] = chartevents["itemid"].map(VITAL_ITEMID_TO_CANON)

    prescriptions = pd.read_csv(os.path.join(hosp, "prescriptions.csv.gz"),
                                usecols=["hadm_id", "drug", "starttime",
                                         "dose_val_rx", "route"],
                                parse_dates=["starttime"])

    trajectories = []
    for _, s in stays.iterrows():
        demo = {"age": int(s["anchor_age"]) if s["anchor_age"] == s["anchor_age"] else 0,
                "sex": "male" if str(s["gender"]).upper().startswith("M") else "female"}
        labs_df = labevents[labevents["hadm_id"] == s["hadm_id"]]
        vit_df = chartevents[chartevents["stay_id"] == s["stay_id"]]
        rx_df = prescriptions[prescriptions["hadm_id"] == s["hadm_id"]]
        if labs_df.empty and vit_df.empty:
            continue
        outcomes = {
            "icu_admission": True,
            "death": bool(s.get("hospital_expire_flag", 0)),
            # discharge_dx: join diagnoses_icd if you want a weak dx label  # TODO[MIMIC]
        }
        traj = build_trajectory_for_stay(
            s["subject_id"], s["stay_id"], s["hadm_id"],
            demo, s["intime"], labs_df, vit_df, rx_df, outcomes, bin_hours)
        if traj.state_snapshots:
            trajectories.append(traj)

    n = write_jsonl(trajectories, out_path)
    rep = validate_dataset(trajectories)
    print(f"[mimic] wrote {n} trajectories → {out_path}")
    print(f"[mimic] validation: {rep['valid']} valid, {rep['invalid']} invalid")

    # split train/eval BY PATIENT (never leak a patient across the split)
    if trajectories:
        import random
        subjects = sorted({t.patient_id.split("_")[1] for t in trajectories})
        random.Random(0).shuffle(subjects)
        cut = int(len(subjects) * 0.85)
        train_subj = set(subjects[:cut])
        train = [t for t in trajectories if t.patient_id.split("_")[1] in train_subj]
        eval_ = [t for t in trajectories if t.patient_id.split("_")[1] not in train_subj]
        write_jsonl(train, out_path.replace(".jsonl", "_train.jsonl"))
        write_jsonl(eval_, out_path.replace(".jsonl", "_eval.jsonl"))
        print(f"[mimic] split by patient: {len(train)} train / {len(eval_)} eval")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic_dir", required=True, help="path to mimic-iv/ (has hosp/ and icu/)")
    ap.add_argument("--out", default=os.path.join(_HERE, "mimic_trajectories.jsonl"))
    ap.add_argument("--max_patients", type=int, default=None)
    ap.add_argument("--bin_hours", type=float, default=6.0)
    args = ap.parse_args()
    try:
        import pandas  # noqa
    except ImportError:
        sys.exit("[mimic] needs pandas: pip install pandas")
    if not os.path.isdir(args.mimic_dir):
        sys.exit(f"[mimic] --mimic_dir not found: {args.mimic_dir}\n"
                 "Point it at your MIMIC-IV folder (containing hosp/ and icu/).")
    convert(args.mimic_dir, args.out, args.max_patients, args.bin_hours)
