# State Observability Spec

> The supervision map for the medical-JEPA latent space.
> For every one of Osler's 200 organ-state variables: can it be measured
> directly (labs/vitals), or only inferred (mechanism)? This determines how —
> and whether — each latent dimension can be supervised during training.
>
> Generated from a live audit. Machine-readable version: `state_observability.json`.

---

## Why this document exists

A JEPA's latent space is only useful if you can supervise it (at least
partially) or learn it self-supervised. Osler hands us a ready-made latent
vocabulary — 200 `organ.state` variables — but **they are not equally
knowable from data**. Before anyone trains a model, they need to know which
latent dimensions have a measurable ground-truth signal and which are pure
inference. That's this spec.

The headline:

| Observability | States | Share |
|---|---|---|
| **Lab-observable** (an abnormal lab maps to it) | 47 | 23% |
| **Vital-observable** (a vital sign threshold maps to it) | 2 | 1% |
| **Mechanism-only** (no direct measurement; inferred from symptoms/other states) | 151 | 75% |
| **Total** | 200 | 100% |

**Three-quarters of the latent space has no direct measurement.** This is the
single most important fact for planning the training objective.

---

## What this means for training

1. **You cannot supervise 75% of the latent directly.** Those 151 states can
   only be learned through (a) self-supervised objectives like masked-event
   prediction, or (b) indirect supervision — predicting the downstream
   observables they cause, and letting backprop shape the hidden state.

2. **The 49 observable states are your anchors.** They're where you can put a
   real regression/classification loss. They should be weighted heavily in
   early training and used as the primary probes for "is the latent learning
   real physiology?"

3. **The mechanism-only states risk circular training.** If you label them
   using Osler's rules and then train to predict those labels, the model just
   re-derives Osler. For these states, prefer self-supervision and treat any
   Osler-derived label as a weak prior, not ground truth.

---

## Observability by organ

Sorted by observability rate. This directly informs the "start with 5 organs"
recommendation — the top of this table is where supervision is cleanest.

| Organ | States | Lab-obs | Vital-obs | Mechanism-only | Obs % |
|---|---|---|---|---|---|
| blood | 9 | 8 | 0 | 1 | **88%** |
| kidney | 9 | 5 | 0 | 4 | **55%** |
| liver | 8 | 4 | 0 | 4 | **50%** |
| heart | 15 | 4 | 2 | 9 | **40%** |
| pancreas | 8 | 3 | 0 | 5 | 37% |
| bladder | 6 | 2 | 0 | 4 | 33% |
| ear | 7 | 2 | 0 | 5 | 28% |
| eye | 9 | 2 | 0 | 7 | 22% |
| peripheral_nerve | 9 | 2 | 0 | 7 | 22% |
| vessel | 9 | 2 | 0 | 7 | 22% |
| gallbladder | 6 | 1 | 0 | 5 | 16% |
| gi | 13 | 2 | 0 | 11 | 15% |
| adrenal | 7 | 1 | 0 | 6 | 14% |
| skin | 7 | 1 | 0 | 6 | 14% |
| thyroid | 7 | 1 | 0 | 6 | 14% |
| upper_airway | 7 | 1 | 0 | 6 | 14% |
| brain | 15 | 2 | 0 | 13 | 13% |
| muscle | 8 | 1 | 0 | 7 | 12% |
| bone | 9 | 1 | 0 | 8 | 11% |
| reproductive | 11 | 1 | 0 | 10 | 9% |
| lung | 15 | 1 | 0 | 14 | **6%** |
| spleen | 6 | 0 | 0 | 6 | **0%** |

### The surprising one: lung is only 6% observable

Lung has 15 states but only **one** (`gas_exchange`, via SpO2) is directly
measurable in this mapping. States like `alveolar_inflammation`,
`airway_inflammation`, `consolidation`, `mucus_production` are all
mechanism-only. For a respiratory-focused model you'd want to **add imaging
(CXR/CT) and ABG as observation modalities** — without them, most lung states
are unsupervised. This contradicts the intuition that lung is "easy"; it's
easy to *diagnose* clinically but its internal states are poorly captured by
routine labs.

### The clean ones: blood, kidney, liver, heart

These four are where labs map densely to states:
- **blood (88%):** WBC→infection_load, Hgb→hemoglobin_level, platelets, CRP→inflammation_marker, etc.
- **kidney (55%):** creatinine/BUN/GFR→filtration, electrolytes.
- **liver (50%):** AST/ALT→hepatocyte_inflammation, bilirubin, albumin.
- **heart (40%):** troponin→ischemia, BNP, plus HR/BP vitals.

**Recommendation:** the MVP latent should be **blood + kidney + liver + heart**,
not the originally-proposed heart/lung/kidney/blood/gi. Swap lung→liver and
defer gi (15% observable) until imaging/endoscopy modalities exist. This is a
small change from the proposal but it's grounded in the actual supervision map.

---

## The 49 directly-observable states (the anchors)

These are the states a model can be supervised on today, from labs/vitals
alone. (Full list with the exact labs in `state_observability.json`.)

Representative examples:

| State | Observable via |
|---|---|
| blood.infection_load | WBC |
| blood.hemoglobin_level | Hgb |
| blood.inflammation_marker | CRP, ESR |
| blood.coagulation_function | INR, PT, PTT |
| kidney.filtration | creatinine, BUN, GFR |
| liver.hepatocyte_inflammation | AST, ALT |
| heart.ischemia | troponin |
| heart.cardiac_output | BP (vital) |
| heart.sympathetic_drive | HR (vital) |
| lung.gas_exchange | SpO2 (vital) |
| adrenal.cortisol_level | cortisol |

---

## Modalities that would expand observability

If the goal is to supervise more of the latent, these observation types would
help most (in rough order of impact):

1. **Imaging (CXR, CT, ultrasound)** — would make lung consolidation/opacity,
   gallbladder stones, vessel findings observable. Biggest single unlock,
   especially for lung (6%→much higher).
2. **ABG (arterial blood gas)** — pH, pCO2, pO2, lactate → multiple lung +
   metabolic states.
3. **ECG features** — heart electrical/rhythm states.
4. **Urinalysis** — bladder, kidney detail.
5. **Endoscopy / pathology** — GI mucosal states.

Without these, ~75% of the latent stays mechanism-only no matter how much
tabular EHR you have.

---

## How to read `state_observability.json`

```json
{
  "state": "blood.infection_load",
  "organ": "blood",
  "n_diseases": 9,           // how many diseases perturb this state
  "observability": "lab",    // "lab" | "vital" | "mechanism_only"
  "labs": ["WBC"],           // which labs observe it (empty if not lab-obs)
  "vital": null              // which vital observes it (null if not vital-obs)
}
```

`n_diseases` matters too: a state observable AND used by many diseases is a
high-value anchor; a mechanism-only state used by one disease is low-value and
a candidate for pruning (see the vocabulary critique).
