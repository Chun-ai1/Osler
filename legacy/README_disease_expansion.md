# Osler Disease Expansion v1

This patch adds **42** new disease models across **18** organ-state JSON files and adds **42** corresponding temporal entries to `disease_timecourse.json`.

## Installation
Replace the matching JSON files inside `medical_knowledge/state_models/` with the files in this folder, then run your existing validation suite. Keep a copy of the original files before replacement.

## Validation performed here
- All expanded files parse as valid JSON.
- Every new perturbation variable exists in its target organ's `state_variables`.
- No existing disease entry was overwritten.
- `validation_errors`: []
- Interpretability audit: all 42 added models connect to at least one existing symptom-derivation or propagation state.

## Important boundary
All added entries are marked `unreviewed`. They extend mechanism coverage for research/testing; they are **not** clinical validation and should not be used to support diagnosis or treatment claims until reviewed and tested.

## Added models by file

### bladder.json
- Interstitial Cystitis/Bladder Pain Syndrome
- Urethritis

### blood.json
- Disseminated Intravascular Coagulation
- Anemia of Chronic Disease
- Hemolytic Anemia

### bone.json
- Osteomyelitis
- Osteoarthritis
- Calcium Pyrophosphate Deposition Disease

### brain.json
- Viral Encephalitis
- Transient Ischemic Attack

### ear.json
- Otitis Externa
- Meniere Disease
- Benign Paroxysmal Positional Vertigo

### eye.json
- Anterior Uveitis
- Cataract
- Central Retinal Artery Occlusion
- Optic Neuritis

### heart.json
- Acute Myocarditis
- Ventricular Tachycardia

### kidney.json
- Hydronephrosis from Urinary Obstruction
- Acute Interstitial Nephritis

### liver.json
- Hepatic Encephalopathy
- Primary Biliary Cholangitis

### lung.json
- Viral Pneumonia
- Aspiration Pneumonitis

### muscle.json
- Dermatomyositis
- Toxic Statin Myopathy

### pancreas.json
- Pancreatic Adenocarcinoma
- Exocrine Pancreatic Insufficiency

### reproductive.json
- Acute Endometritis
- Tubo-Ovarian Abscess

### skin.json
- Necrotizing Fasciitis
- Herpes Zoster

### spleen.json
- Splenic Rupture
- Hypersplenism

### thyroid.json
- Subacute Thyroiditis
- Toxic Multinodular Goiter

### upper_airway.json
- Acute Viral Rhinitis
- Chronic Rhinosinusitis

### vessel.json
- Abdominal Aortic Aneurysm
- Acute Limb Ischemia
- Superficial Thrombophlebitis
