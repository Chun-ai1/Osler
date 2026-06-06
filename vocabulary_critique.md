# State Vocabulary Critique

> An engineering critique of Osler's 200-state vocabulary, viewed as the
> candidate latent space for a medical JEPA. What's redundant, what's too
> sparse, where granularity is uneven, and what to do before training.
>
> Generated from a live audit. Companion to `state_observability.md`.

---

## TL;DR

The vocabulary is good *as a symbolic reasoning substrate* but needs cleanup
*as an ML latent space*. Three problems:

1. **Sparsity** — 102/200 states (51%) are used by exactly one disease; 23 are
   used by none. Half the vocabulary barely participates.
2. **Redundancy** — the same concept (`inflammation`, `function`, etc.) is
   re-encoded per-organ 27+ times, with no shared representation.
3. **Granularity imbalance** — organs range from 6 to 15 states with no
   consistent principle for the split.

None of these break Osler's symbolic reasoning. All of them would hurt a naive
JEPA trained on this vocabulary as-is.

---

## Problem 1: Sparsity

| Usage | States | Share |
|---|---|---|
| Used by 0 diseases (baseline-only) | 23 | 12% |
| Used by exactly 1 disease | 102 | 51% |
| Used by 2+ diseases | 75 | 37% |

**51% single-use + 12% unused = 63% of the vocabulary is low-participation.**

### Why this hurts a JEPA

A latent dimension that only ever moves for one disease gives the model almost
no signal to learn a general representation — it's effectively a
disease-specific indicator, not a reusable state. The model will either ignore
it (wasted capacity) or memorize the one disease (overfitting).

### The lowest-value states (prune candidates)

**98 states are both mechanism-only AND used by ≤1 disease.** These have no
direct supervision *and* almost no usage — the weakest dimensions in the whole
space. Examples:

```
adrenal.acth_drive, adrenal.aldosterone_level, adrenal.catecholamine_output,
bladder.detrusor_irritability, bladder.mucosal_barrier, bladder.nerve_irritation,
blood.rbc_count, ...
```

### Recommendation

- For a **first JEPA**, restrict the latent to states used by ≥2 diseases
  (75 states) — or even ≥3 (52 states). This is the dense, learnable core.
- Keep the long tail in Osler's symbolic engine (it's fine there) but don't
  force the neural model to represent it.
- The 23 unused states are baseline physiology placeholders (`cortex_function`
  defaulting to 1.0); they carry no disease signal and should be excluded from
  the latent entirely.

---

## Problem 2: Redundancy

The concept **"inflammation" appears in 27 separate states**, one per organ/site:

```
bladder.inflammation              lung.airway_inflammation
blood.inflammation_marker         lung.alveolar_inflammation
bone.inflammation                 muscle.inflammation
bone.synovial_inflammation        pancreas.exocrine_inflammation
brain.meningeal_inflammation      pancreas.peripancreatic_inflammation
brain.neuronal_inflammation       peripheral_nerve.inflammation
eye.anterior_chamber_inflammation reproductive.inflammation
eye.conjunctival_inflammation     skin.inflammation
gallbladder.inflammation          spleen.inflammation
gi.mucosal_inflammation           thyroid.gland_inflammation
heart.pericardial_inflammation    upper_airway.nasal_inflammation
kidney.parenchymal_inflammation   upper_airway.pharyngeal_inflammation
liver.hepatocyte_inflammation     upper_airway.sinus_inflammation
                                  vessel.wall_inflammation
```

Each is a separate latent dimension with no shared structure. The model has to
independently learn "inflammation behaves like X" 27 times.

### Why this hurts a JEPA

Inflammation has shared physiology regardless of site (cytokines, CRP rise,
local swelling). Encoding it as 27 unrelated dimensions:
- wastes capacity (the model relearns the same dynamics per organ),
- loses the cross-organ correlation (systemic inflammatory response affects
  many sites together — the vocabulary can't express that coupling),
- makes the latent harder to probe (no single "inflammation" axis).

Same pattern, smaller scale, for other shared concepts: `_function` (14
states), `_integrity` (8), `_output` (3), `_drive` (3).

### Recommendation

Two options, pick based on effort:
- **Light:** add a `concept` tag to each state (`inflammation`, `perfusion`,
  `function`, …) so the model can share an embedding across same-concept states
  while keeping the organ-specific dimension. Minimal JSON change.
- **Heavier:** factor the latent as `(organ, concept, value)` — a structured
  latent where organ and concept have their own embeddings that combine. More
  principled, more work, better representation.

The light option is enough for a first model and is backward-compatible with
Osler.

---

## Problem 3: Granularity imbalance

States per organ range 6→15 with no consistent principle:

| Most states | | Fewest states | |
|---|---|---|---|
| brain | 15 | spleen | 6 |
| lung | 15 | gallbladder | 6 |
| heart | 15 | bladder | 6 |
| gi | 13 | thyroid | 7 |
| reproductive | 11 | ear | 7 |

This isn't necessarily wrong — the brain *is* more complex than the gallbladder
— but the split looks driven by "how much did we happen to model this organ"
rather than a consistent level of abstraction. Some specific tells:

- **lung has 15 states but 6% observability** — it's finely subdivided
  (`airway_inflammation` vs `alveolar_inflammation` vs `consolidation` vs
  `mucus_production`) yet almost none of those subdivisions are measurable.
  High granularity with no supervision = the worst combination for ML.
- **spleen has 6 states and 0% observability** — coarsely modeled and entirely
  unsupervised.

### Why this hurts a JEPA

Uneven granularity means the latent over-represents some organs and
under-represents others for reasons unrelated to clinical importance or data
availability. The model spends capacity proportional to "how much Osler modeled
this organ," not "how much signal exists."

### Recommendation

- Set a target granularity per organ tied to **observability + disease usage**,
  not modeling enthusiasm. An organ with 15 mechanism-only states (lung) should
  either gain observation modalities (imaging) or collapse some states.
- For the first JEPA, normalize: cap each organ's contribution, or weight the
  loss inversely to an organ's state count so brain doesn't dominate by sheer
  dimension count.

---

## Naming consistency (minor)

Suffix conventions are mostly consistent but not fully:
- `_function` (14), `_integrity` (8), `_level` (5), `_drive` (3),
  `_output` (3), `_capacity` (3), `_tone` (1)
- `blood.inflammation_marker` (singular) is the only `_marker`; everything else
  uses bare `inflammation`. Small inconsistency, worth normalizing.

Not an ML problem, but a clean vocabulary is easier to maintain and tag.

---

## Suggested pre-training cleanup checklist

In priority order, before any serious JEPA training:

1. **Define the trainable subset.** States used by ≥2 (or ≥3) diseases. ~75 (or
   ~52) states. Exclude the 23 unused + the single-use mechanism-only tail.
2. **Add concept tags.** Tag each state with its shared concept (inflammation,
   perfusion, function, …) so the model can share representation.
3. **Decide lung's fate.** Either add imaging/ABG modalities (best) or collapse
   its 15 mechanism-only states. As-is it's high-dimension, no-signal.
4. **Normalize granularity in the loss.** Weight per-organ contribution so
   dimension count doesn't equal importance.
5. **Normalize naming.** Minor, do it while doing 1–2.

Items 1–2 are the high-impact ones. They can be done as JSON/metadata changes
without touching Osler's reasoning code, and they'd make the difference between
a JEPA that learns a clean physiological latent and one that wastes most of its
capacity on sparse, redundant, unsupervised dimensions.

---

## What NOT to change

Don't "fix" these in Osler's symbolic engine — they're only problems for the
*neural latent*, not for symbolic reasoning:

- The sparse single-use states are fine for explaining rare diseases.
- The per-organ inflammation states give precise, explainable traces (the chat
  panel benefits from `lung.alveolar_inflammation` being distinct from
  `gi.mucosal_inflammation`).
- The granularity reflects real modeling effort and is useful clinically.

The right move is a **separate latent-space definition for the JEPA** that
*derives from* Osler's vocabulary (with the subsetting + tagging above), rather
than editing Osler's vocabulary in place. Keep the symbolic engine's richness;
give the neural model a cleaner, denser projection of it.
