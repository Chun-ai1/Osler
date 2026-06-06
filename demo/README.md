# Osler·Rx — Drug Recommendation Agent (demo)

A clinician-facing demo: **import a patient case → get ranked drug recommendations →
see the full reasoning as a mind-map → chat to ask why.**

It reuses the project's existing **symbolic pharmacology engine** (`reasoning_engine.py`
in the project root) unchanged. The recommendation, ranking, safety gating and graph are
100% computed by that engine — fast, deterministic, explainable. An LLM is used **only**
to (a) parse messy free-text case notes and (b) explain the result in chat. The LLM never
makes a medical decision.

> Everything in this `demo/` folder is new. No legacy file in the project root was modified.

---

## ⚠️ Accuracy — read this first

This is a **methodology / architecture demo, NOT a clinically validated tool.**

- The drug `delta` values (`drugs_pkpd.json`) and the disease perturbations (organ JSONs)
  are **hand-authored estimates** marked `review_status: unreviewed`.
- `demo_clinical_data.json` is **illustrative, hand-written label text — not real FDA labels.**
- Therefore: **direction (↑/↓) is reliable; exact magnitudes are soft.** Use it for
  *explainable relative ranking*, never for absolute dose/efficacy claims.
- **The AI agent does NOT make it more medically accurate.** Accuracy comes from the
  *knowledge base*, not the LLM. To make it production-grade you must:
  1. tick **"fetch live openFDA labels"** to pull real FDA label + FAERS data (already wired);
  2. have clinicians validate/anchor the `delta` knowledge base to guidelines;
  3. align the variable vocabulary between drugs and disease models.

Decision support only. A licensed clinician makes the final call.

---

## Run

From the **project root** (`E:\jingbinqian\Osler`):

```bat
py -m pip install -r requirements.txt
py demo\demo_app.py
```

Open http://127.0.0.1:5000 .

**Chat / free-text parsing** need an LLM key — paste it in the **top-right box**
(OpenAI by default, or pick Gemini). Without a key the app still runs fully: rule-based
case parsing + the symbolic reasoning graph; only the chat falls back to a notice.

Optional env instead of the box:
```bat
set OPENAI_API_KEY=sk-...
:: or:  set GEMINI_API_KEY=...   &&   set LLM_PROVIDER=gemini
```

---

## How to use

1. **Import a case** (left): click a preset, or paste a free-text note like
   `64M crushing chest pain, acute coronary syndrome. BP 88/54, HR 112, eGFR 72.`
   then **Analyze case**.
2. **Reasoning graph** (right): `patient ▸ disease (real perturbations + symptoms) ▸
   treatment targets ▸ drugs` colored by safety verdict. Click a drug node to jump to its card.
3. **Chat** (center): ask "Why is aspirin first?", "Why is nitroglycerin avoided?",
   "What is the disease doing to the body?". Answers are grounded in the engine output.
4. **Patients** (left): each analyzed case becomes a switchable patient with its own
   graph and chat history.

---

## What is computed vs. what uses AI

| Step | Engine | Uses LLM? |
|---|---|---|
| Drug match, score, ranking, safety gate, dose gating | `reasoning_engine` + `drug_safety_gate` + `clinical_role` (symbolic) | ❌ never |
| Disease world-model (perturbations, symptoms) | `disease_world` reads the organ JSONs | ❌ never |
| Reasoning mind-map | `agent.build_graph` | ❌ never |
| Parse a **free-text** case → fields + indication | `case_parser` | ✅ if key set (else regex rules) |
| "Why this drug?" chat | `llm_client` | ✅ if key set (else fallback notice) |

A preset case sends structured fields, so it triggers **no** LLM — that's why it's instant.
"Instant" = the symbolic engine runs in ~0.2 ms, not that it is hardcoded; change the BP or
add an allergy and the verdicts change.

---

## Files

| File | Role |
|---|---|
| `demo_app.py` | Flask backend: `/`, `/api/cases`, `/api/analyze`, `/api/chat` |
| `agent.py` | Orchestrator: parse → recommend → disease world-model → build graph; optional openFDA |
| `case_parser.py` | Free-text case → structured fields (LLM or deterministic rules) |
| `case_targets.py` | Indication → physiological treatment targets (15 indications) |
| `disease_world.py` | Local "wiki": indication → real disease perturbations + symptoms from organ JSONs |
| `llm_client.py` | Provider-agnostic LLM (OpenAI / Gemini), per-request key, graceful no-key fallback |
| `case_demo.html` | Single-page UI: patient roster · import · chat · mind-map graph |
| `sample_cases.json` | 5 preset cases |
| `demo_clinical_data.json` | Illustrative (non-clinical) label data so the safety gate + dose path demonstrate |

See `../ARCHITECTURE_REVIEW.md` for the full algorithm review of the wider codebase.
