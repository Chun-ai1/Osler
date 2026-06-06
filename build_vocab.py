"""
build_vocab.py — freeze the JEPA latent vocabulary.

Before training, the latent space (the set of organ states the model predicts)
must be FIXED and versioned, so every training run, checkpoint, and probe agrees
on what dimension N means. This script writes `vocab.json`.

It also builds the vocabularies for the INPUT side: the set of distinct
symptoms, labs, and vitals that appear, so the encoder's embedding tables have
fixed sizes.

Tiers (from vocabulary_critique.md):
  - "core"  : states used by >=3 diseases  (~52)  — dense, recommended first model
  - "full"  : states used by >=2 diseases  (~75)  — fuller latent
  - "all"   : all 200 states                       — includes sparse tail (not recommended)

Usage:  python3 build_vocab.py [core|full|all]   (default: full)
"""
from __future__ import annotations
import sys, os, io, json, collections

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for p in (_REPO, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)


def build(tier: str = "full") -> dict:
    _so = sys.stdout
    sys.stdout = io.StringIO()
    from nexus_engine.state_model import _REGISTRY, simulate_disease
    _REGISTRY.load_all()
    diseases = _REGISTRY.all_diseases()
    sys.stdout = _so

    # 1. state usage
    state_dz = collections.defaultdict(set)
    input_symptoms = set()
    for dz in diseases:
        sim = simulate_disease(dz)
        if not sim:
            continue
        for p in sim.get("perturbations", []):
            o, v = p.get("organ"), p.get("variable")
            if o and v:
                state_dz[f"{o}.{v}"].add(dz)
        for s in sim.get("derived_symptoms", []):
            input_symptoms.add(s)

    min_dz = {"core": 3, "full": 2, "all": 0}[tier]
    states = sorted(s for s, dz in state_dz.items() if len(dz) >= min_dz)
    if tier == "all":  # include even unused
        obs = json.load(open(os.path.join(_HERE, "state_observability.json")))
        states = sorted(set(states) | {r["state"] for r in obs})

    # 2. observability + concept tag per state (concept = the shared keyword)
    obs = {r["state"]: r for r in json.load(open(os.path.join(_HERE, "state_observability.json")))}
    CONCEPTS = ["inflammation", "function", "perfusion", "infection", "ischemia",
                "damage", "injury", "integrity", "tone", "level", "output",
                "drive", "pressure", "filtration", "obstruction", "edema"]
    def concept_of(state):
        var = state.split(".", 1)[1] if "." in state else state
        for c in CONCEPTS:
            if c in var:
                return c
        return "other"

    state_entries = []
    for i, s in enumerate(states):
        r = obs.get(s, {})
        state_entries.append({
            "index": i,
            "state": s,
            "organ": s.split(".")[0],
            "concept": concept_of(s),
            "observability": r.get("observability", "mechanism_only"),
            "labs": r.get("labs", []),
            "vital": r.get("vital"),
            "n_diseases": len(state_dz.get(s, set())),
        })

    # 3. input vocabularies (fixed embedding-table sizes)
    # labs/vitals come from the labeler's known set
    lab_map = json.load(open(os.path.join(_REPO, "medical_knowledge", "state_models",
                                          "lab_integration.json")))["lab_to_state_mapping"]
    labs = sorted(k for k in lab_map if not k.startswith("_"))
    vitals = ["hr", "bp_sys", "bp_dia", "spo2", "temp", "rr", "gcs", "map", "lactate"]
    organs = sorted({e["organ"] for e in state_entries})
    concepts = sorted({e["concept"] for e in state_entries})

    return {
        "_meta": {
            "tier": tier,
            "min_diseases": min_dz,
            "n_states": len(state_entries),
            "note": "Frozen JEPA vocabulary. Index→state mapping is stable; "
                    "do not reorder. Regenerate only when knowledge base changes, "
                    "and bump version.",
            "version": 1,
        },
        "states": state_entries,                  # the LATENT space (prediction target)
        "input_symptoms": sorted(input_symptoms), # input embedding table
        "input_labs": labs,
        "input_vitals": vitals,
        "organs": organs,
        "concepts": concepts,
    }


if __name__ == "__main__":
    tier = sys.argv[1] if len(sys.argv) > 1 else "full"
    vocab = build(tier)
    out = os.path.join(_HERE, "vocab.json")
    json.dump(vocab, open(out, "w"), indent=2, ensure_ascii=False)
    m = vocab["_meta"]
    print(f"Wrote {out}")
    print(f"  tier={m['tier']}  latent states={m['n_states']}")
    print(f"  input symptoms={len(vocab['input_symptoms'])}  "
          f"labs={len(vocab['input_labs'])}  vitals={len(vocab['input_vitals'])}")
    print(f"  organs={len(vocab['organs'])}  concepts={len(vocab['concepts'])}")
    # observability split of the chosen latent
    import collections
    split = collections.Counter(e["observability"] for e in vocab["states"])
    print(f"  latent observability: {dict(split)}")
