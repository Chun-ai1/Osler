# Osler · Rx

A clinician-facing **drug-recommendation agent** built on a symbolic pharmacology
reasoning engine. Import a patient case → get ranked, **explainable** drug
recommendations → inspect the reasoning as a mind-map → chat to ask why.

The repository was reorganized (see structure below). The recommendation logic is
100% symbolic and deterministic; an LLM is used only to parse free-text cases and to
explain results in chat — it never makes a medical decision.

> ⚠️ **Decision support / research demo only — not clinically validated.** The drug and
> disease `delta` values are hand-authored estimates; `data/demo_clinical_data.json` is
> illustrative, not real FDA labels. Direction (↑/↓) is reliable; magnitudes are soft.
> A licensed clinician makes the final call. See `demo/README.md`.

## Quick start

```bat
py -m pip install -r requirements.txt
py demo\demo_app.py
```

Open http://127.0.0.1:5000 . Paste an API key in the top-right box to enable the chat /
free-text parsing (OpenAI by default, or Gemini); without a key it still runs with
rule-based parsing + the full reasoning graph.

## Repository structure

```
Osler/
├── demo/        The product: web UI + agent orchestrator.
│               demo_app.py (Flask), agent.py, case_parser.py, case_targets.py,
│               disease_world.py, llm_client.py, case_demo.html, sample_cases.json,
│               demo_clinical_data.json, README.md
│
├── engine/      The live symbolic pharmacology engine the demo depends on (7 modules):
│               reasoning_engine.py, patient_profile.py, drug_safety_gate.py,
│               drug_identity.py, clinical_role.py, drug_profile.py, clinical_data.py
│
├── data/        All knowledge/data files (.json, .npz): drugs_pkpd.json (drug deltas),
│               the organ state-models (heart.json, lung.json, …), and the rest of the
│               legacy knowledge base.
│
├── legacy/      The original disease-DIAGNOSIS system (nexus_*), the old web app
│               (app.py, chat.html, render_*), experimental RL/GNN code, tests, and
│               other modules NOT used by the drug-recommendation demo. Archived as-is;
│               see "Legacy" below.
│
├── ARCHITECTURE_REVIEW.md   Full algorithm review of the original codebase.
├── requirements.txt
└── README.md
```

### How it fits together (live path)

```
demo/demo_app.py ──uses──▶ demo/agent.py
        │                      ├─ case_parser   (free-text → fields; LLM or rules)
        │                      ├─ case_targets   (indication → treatment targets)
        │                      ├─ engine/reasoning_engine.recommend()   ◀── data/drugs_pkpd.json
        │                      └─ disease_world  (indication → disease model ◀── data/<organ>.json)
        └──chat──▶ llm_client (OpenAI / Gemini), grounded in the engine result
```

`demo/demo_app.py` puts `engine/` on `sys.path`; the engine reads from `data/`. Nothing
in the live path imports anything from `legacy/`.

## Legacy

`legacy/` holds the original disease-diagnosis system and old web UI. It is **archived,
not wired to run after the reorg** — its modules still import each other, but their data
loads expect files beside them (the data now lives in `data/`). The diagnosis system was
already incomplete (it expected a `medical_knowledge/` tree that isn't present). To
revive a legacy module, run it with both `legacy/` and `data/` discoverable, or update
its data paths to point at `../data`. The drug-recommendation product does not need any
of it.
```
