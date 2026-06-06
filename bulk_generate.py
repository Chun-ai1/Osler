"""
bulk_generate.py — generate a coverage dataset of synthetic trajectories,
one per Osler disease, by simulating each disease and reading off its
expected symptoms + the labs/vitals that would be abnormal.

This gives broad COVERAGE of the state space for pretraining. It is NOT
clinically realistic (every case is a textbook presentation of one disease,
deterministically generated). Tagged source="synthetic_osler".

Use for:  pretraining the encoder architecture, checking state-space coverage.
Do NOT use for:  any accuracy/safety claim.

Usage:  python3 bulk_generate.py [n_per_disease]
"""
from __future__ import annotations
import sys, os, io, json

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for p in (_REPO, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from trajectory_schema import write_jsonl, validate_dataset
from osler_to_trajectory import case_to_trajectory, _nexus


def _expected_symptoms_for(disease: str) -> list:
    """Read the symptoms a disease is expected to produce, from simulate_disease."""
    _stdout = sys.stdout
    sys.stdout = io.StringIO()
    from nexus_engine.state_model import simulate_disease
    sys.stdout = _stdout
    sim = simulate_disease(disease) or {}
    return sim.get("derived_symptoms", [])[:6]


def generate_all() -> list:
    nx = _nexus()
    _stdout = sys.stdout
    sys.stdout = io.StringIO()
    from nexus_engine.state_model import _REGISTRY
    _REGISTRY.load_all()
    diseases = _REGISTRY.all_diseases()
    sys.stdout = _stdout

    trajs = []
    skipped = 0
    for i, dz in enumerate(diseases):
        syms = _expected_symptoms_for(dz)
        if not syms:
            skipped += 1
            continue
        pid = f"syn_{dz.lower().replace(' ', '_').replace('/', '_')}_{i:03d}"
        t = case_to_trajectory(
            patient_id=pid,
            symptoms=syms,
            context={},                 # no demographics for the generic case
            outcomes={"discharge_dx": dz, "_textbook": True},
            source="synthetic_osler",
        )
        trajs.append(t)
    return trajs, skipped


if __name__ == "__main__":
    print("Generating one synthetic trajectory per Osler disease...")
    trajs, skipped = generate_all()
    out_path = os.path.join(_HERE, "osler_coverage_trajectories.jsonl")
    n = write_jsonl(trajs, out_path)
    rep = validate_dataset(trajs)
    print(f"Generated {n} trajectories ({skipped} diseases skipped: no derived symptoms)")
    print(f"Validation: {rep['valid']} valid, {rep['invalid']} invalid")
    print(f"Written to: {out_path}")

    # coverage stats
    states_seen = set()
    label_sources = {}
    for t in trajs:
        for snap in t.state_snapshots:
            for l in snap.labels:
                states_seen.add(l.state)
                label_sources[l.source] = label_sources.get(l.source, 0) + 1
    print(f"\nState-space coverage: {len(states_seen)} distinct states labeled across the dataset")
    print("Label sources:", dict(sorted(label_sources.items())))
