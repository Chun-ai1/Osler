"""
NEXUS Self-Learning Medical Environment
════════════════════════════════════════════════════════════════
Wraps your existing NEXUS engine as a Gym-style RL environment.

How it works:
  1. MedicalEnv.reset()  → generates a patient (symptoms only, dx hidden)
  2. NexusMedical.reason() → agent's symbolic world model (replaces naive Q-table)
  3. Agent picks diagnosis + treatment
  4. Reward = NEXUS consistency score + survival outcome + pathogen spread penalty
  5. NexusLearner.feedback() → writes new triples back into KnowledgeGraph
  6. Memory buffer → learn_from_cases() replays on every N episodes

Dependencies (your existing files, no changes needed):
  nexus_engine/nexus_medical.py
  nexus_engine/nexus_learning_bridge.py
  nexus_engine/nexus_core.py
  nexus_engine/anatomy_atlas.py
  nexus_engine/pathogen_tracker.py
  nexus_engine/etiology_classifier.py   (optional)
  nexus_engine/physiology_engine.py     (optional)
"""

from __future__ import annotations
import json
import os
import random
import math
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────
# 1.  PATIENT GENERATOR  (the "environment")
# ─────────────────────────────────────────────────────────────

# Single canonical template per disease — clean vocabulary separation.
# Multi-template expansion caused cross-contamination between confusable pairs,
# inflating shared symptom counts and collapsing pair discrimination.
# The noise curriculum provides training diversity without template pollution.
PATIENT_POOL = [
    {"disease": "pneumonia",
     "symptoms": ["cough", "fever", "shortness of breath", "chest pain", "fatigue"],
     "severity": "severe", "pathogen": "bacteria", "infected_organ": "lungs",
     "correct_treatments": ["antibiotics", "oxygen therapy"]},

    {"disease": "flu",
     "symptoms": ["fever", "body aches", "fatigue", "cough", "headache"],
     "severity": "moderate", "pathogen": "virus", "infected_organ": "lungs",
     "correct_treatments": ["antivirals", "rest", "fluids"]},

    {"disease": "appendicitis",
     "symptoms": ["abdominal pain", "nausea", "fever", "vomiting"],
     "severity": "severe", "pathogen": "bacteria", "infected_organ": "appendix",
     "correct_treatments": ["surgery", "antibiotics"]},

    {"disease": "meningitis",
     "symptoms": ["headache", "stiff neck", "fever", "confusion", "nausea"],
     "severity": "critical", "pathogen": "bacteria", "infected_organ": "meninges",
     "correct_treatments": ["antibiotics", "steroids", "iv fluids"]},

    {"disease": "heart attack",
     "symptoms": ["chest pain", "left arm pain", "shortness of breath", "nausea", "sweating"],
     "severity": "critical", "pathogen": "non-infectious", "infected_organ": "heart",
     "correct_treatments": ["aspirin", "pci", "nitroglycerin"]},

    {"disease": "gastroenteritis",
     "symptoms": ["diarrhea", "vomiting", "abdominal pain", "nausea", "fever"],
     "severity": "moderate", "pathogen": "virus", "infected_organ": "stomach",
     "correct_treatments": ["fluids", "rest", "antiemetics"]},

    {"disease": "sepsis",
     "symptoms": ["fever", "confusion", "shortness of breath", "body aches", "weakness"],
     "severity": "critical", "pathogen": "bacteria", "infected_organ": "spleen",
     "correct_treatments": ["antibiotics", "iv fluids", "vasopressors"]},

    {"disease": "asthma",
     "symptoms": ["wheezing", "shortness of breath", "chest tightness", "cough"],
     "severity": "moderate", "pathogen": "non-infectious", "infected_organ": "lungs",
     "correct_treatments": ["inhaler", "bronchodilators", "steroids"]},

    {"disease": "migraine",
     "symptoms": ["headache", "nausea", "dizziness", "sensitivity to light"],
     "severity": "moderate", "pathogen": "non-infectious", "infected_organ": "brain",
     "correct_treatments": ["triptans", "nsaids", "rest"]},

    {"disease": "covid",
     "symptoms": ["fever", "cough", "fatigue", "shortness of breath", "loss of smell"],
     "severity": "moderate", "pathogen": "virus", "infected_organ": "lungs",
     "correct_treatments": ["antivirals", "oxygen therapy", "rest"]},
]


TREATMENT_OPTIONS = [
    "antibiotics", "antivirals", "antifungals",
    "oxygen therapy", "iv fluids", "fluids", "rest",
    "aspirin", "steroids", "nsaids", "triptans",
    "inhaler", "bronchodilators", "surgery",
    "pci", "nitroglycerin", "vasopressors",
    "antiemetics", "analgesics",
]


def _patch_nexus_atlas_cache(nexus_instance):
    """
    Monkey-patch NexusMedical._predict_pathogen_spread to use a cached atlas.
    Prevents AnatomyAtlas() from being rebuilt on every episode.
    Call once after nexus.load_knowledge().
    """
    import types
    try:
        from nexus_engine.anatomy_atlas import AnatomyAtlas
        from nexus_engine.pathogen_tracker import PathogenTracker
    except ModuleNotFoundError:
        try:
            from anatomy_atlas import AnatomyAtlas
            from pathogen_tracker import PathogenTracker
        except ModuleNotFoundError:
            return  # can't patch, skip silently

    # Build atlas ONCE and close over it
    _cached_atlas   = AnatomyAtlas()
    _cached_tracker = PathogenTracker(_cached_atlas)

    def _fast_predict(self, sym_set, top_diseases, detected_systems):
        gi_symptoms      = {"diarrhea","vomiting","nausea","abdominal pain","bloating"}
        resp_symptoms    = {"cough","shortness of breath","wheezing","sore throat"}
        neuro_symptoms   = {"headache","numbness","dizziness","weakness","confusion"}
        cardiac_symptoms = {"chest pain","palpitations"}

        infected_organs = []
        pathogen_type   = "virus"

        if sym_set & gi_symptoms:
            infected_organs.append("stomach")
            if "diarrhea" in sym_set:
                infected_organs.append("ileum")
        if sym_set & resp_symptoms:
            infected_organs.append("lungs")
            if "sore throat" in sym_set:
                infected_organs.append("pharynx")
        if sym_set & neuro_symptoms:
            infected_organs.append("brain")
        if sym_set & cardiac_symptoms:
            infected_organs.append("heart")
        if "fever" in sym_set and not infected_organs:
            infected_organs.append("spleen")
        if "swelling" in sym_set:
            infected_organs.append("r_femoral_v")
        if "systemic" in detected_systems or "immune" in detected_systems:
            pathogen_type = "bacteria" if "fever" in sym_set and len(sym_set) <= 3 else "virus"
        if not infected_organs:
            return []

        infected_organs = list(set(infected_organs))[:3]
        result = _cached_tracker.track_multiple(infected_organs, pathogen_type, max_hops=6)

        predictions = []
        for oar in result.get("combined_risk", [])[:8]:
            predictions.append({
                "organ": oar["organ"], "risk": oar["risk"],
                "via": oar["via"], "path": oar.get("path",""),
                "from": oar.get("infected_from",""),
            })
        for organ, sources in result.get("convergence_points", {}).items():
            predictions.append({
                "organ": organ, "risk": 0.95, "via": "convergence",
                "path": f"Reachable from {' and '.join(sources)}",
                "from": "multiple",
                "warning": f"{organ} at high risk from {len(sources)} sources",
            })
        for imp in result.get("individual_results",[{}])[0].get("clinical_implications",[]):
            predictions.append({"clinical_note": imp})
        return predictions

    nexus_instance._predict_pathogen_spread = types.MethodType(_fast_predict, nexus_instance)
    print("[PATCH] NexusMedical._predict_pathogen_spread now uses cached AnatomyAtlas ✓")


class MedicalEnv:
    """
    Gym-style environment powered by NEXUS.

    obs = {
        "symptoms": [...],              # visible to agent
        "nexus_result": {...},          # NEXUS 16-step reasoning output
        "etiology": {...},              # virus/bacteria/non-infectious
        "pathogen_spread": [...],       # predicted organ spread
    }
    """

    def __init__(self, nexus_medical, noise_p: float = 0.15):
        self.nexus = nexus_medical
        self.noise_p = noise_p
        self._current_patient = None
        self._episode = 0

        # Build atlas ONCE, share with everything that needs it
        self._atlas = self._load_atlas()
        self._etiology = self._load_etiology()
        self._physiology = self._load_physiology()

        # Patch NexusMedical to use our cached atlas (stops [ANATOMY] spam)
        _patch_nexus_atlas_cache(self.nexus)

    # ── optional modules ──────────────────────────────────────
    def _load_atlas(self):
        try:
            from nexus_engine.anatomy_atlas import AnatomyAtlas
            return AnatomyAtlas()
        except Exception:
            return None

    def _load_etiology(self):
        try:
            from nexus_engine.etiology_classifier import EtiologyClassifier
            return EtiologyClassifier()
        except Exception:
            return None

    def _load_physiology(self):
        try:
            from nexus_engine.physiology_engine import PhysiologyEngine
            if self._atlas:
                return PhysiologyEngine(self._atlas)
        except Exception:
            pass
        return None

    # ── core ──────────────────────────────────────────────────
    def reset(self) -> dict:
        """Generate a new patient. Returns observation (no true dx)."""
        self._episode += 1
        template = random.choice(PATIENT_POOL)
        # Add symptom noise to force generalisation
        symptoms = list(template["symptoms"])
        noise_syms = ["fatigue", "dizziness", "weakness", "loss of appetite", "insomnia"]
        for ns in noise_syms:
            if random.random() < self.noise_p and ns not in symptoms:
                symptoms.append(ns)

        self._current_patient = {
            **template,
            "symptoms": symptoms,
            "age": random.randint(18, 80),
            "episode": self._episode,
        }

        # Run NEXUS reasoning on symptoms (this IS the observation encoder)
        nexus_result = self._run_nexus(symptoms)

        # Etiology (optional)
        etiology = {}
        if self._etiology:
            try:
                etiology = self._etiology.classify(symptoms)
            except Exception:
                pass

        # Pathogen spread prediction
        spread = nexus_result.get("nexus_pathogen_spread", [])

        # Cache nexus_result so step() can reuse it without re-running NEXUS
        self._current_nexus_result = nexus_result

        return {
            "symptoms": symptoms,
            "nexus_result": nexus_result,
            "etiology": etiology,
            "pathogen_spread": spread,
        }

    def step(self, diagnosis: str, treatment: str) -> Tuple[dict, float, bool, dict]:
        """
        Agent acts: picks a diagnosis and treatment.
        Returns (next_obs, reward, done, info).
        Reuses the nexus_result already computed in reset() — no double NEXUS call.
        """
        p = self._current_patient
        # Use cached nexus_result from reset() — already in _current_obs
        nexus_result = getattr(self, "_current_nexus_result", self._run_nexus(p["symptoms"]))

        reward = self._compute_reward(diagnosis, treatment, nexus_result)

        info = {
            "true_disease": p["disease"],
            "true_treatment": p["correct_treatments"],
            "true_pathogen": p["pathogen"],
            "severity": p["severity"],
            "nexus_result": nexus_result,
            "correct_diagnosis": self._matches(diagnosis, p["disease"]),
            "correct_treatment": any(
                self._matches(treatment, t) for t in p["correct_treatments"]
            ),
            "reward": reward,
        }

        # Record this episode to case_records.jsonl for NexusLearner replay
        self._log_case(p["symptoms"], diagnosis, treatment, nexus_result, reward)

        return None, reward, True, info   # done=True: one episode = one patient

    # ── reward function ───────────────────────────────────────
    def _compute_reward(self, diagnosis: str, treatment: str, nexus_result: dict) -> float:
        """
        Simplified reward — guaranteed positive for correct, negative for wrong.
        No spread penalty (it was always firing and drowning the signal).

          +1.0  correct diagnosis
          +1.0  correct treatment
          -1.0  wrong diagnosis
          -1.0  wrong treatment
        Range: [-2, +2]. Positive only when agent is right.
        No NEXUS-agreement bonus (removed — amplifies NEXUS biases).
        """
        p = self._current_patient
        r = 0.0

        correct_dx = self._matches(diagnosis, p["disease"])
        correct_tx = any(self._matches(treatment, t) for t in p["correct_treatments"])

        r += 1.0 if correct_dx else -1.0
        r += 1.0 if correct_tx else -1.0

        # No NEXUS-agreement bonus — it amplifies existing NEXUS biases
        # onto the pair discrimination boundary. Pure correct/wrong signal only.

        return round(r, 3)

    # ── NEXUS integration ─────────────────────────────────────
    def _run_nexus(self, symptoms: List[str]) -> dict:
        """Call NexusMedical.enhance_pipeline_result() — the full 16-step chain."""
        try:
            pipeline_result = {
                "symptoms": symptoms,
                "final_symptoms": symptoms,
                "top_diseases": [],
                "reasoning": "",
            }
            return self.nexus.enhance_pipeline_result(pipeline_result)
        except Exception as e:
            return {"error": str(e), "nexus_diagnoses": [], "nexus_consistency": {}}

    # ── helpers ───────────────────────────────────────────────
    @staticmethod
    def _matches(a: str, b: str) -> bool:
        a, b = a.lower().strip(), b.lower().strip()
        return a == b or a in b or b in a

    @staticmethod
    def _log_case(symptoms, diagnosis, treatment, nexus_result, reward):
        os.makedirs("auto_learning", exist_ok=True)
        record = {
            "final_symptoms": symptoms,
            "top_diseases": nexus_result.get("nexus_diagnoses", [])[:5],
            "agent_diagnosis": diagnosis,
            "agent_treatment": treatment,
            "reward_final": {"final": reward},
        }
        with open("case_records.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


# ─────────────────────────────────────────────────────────────
# 2.  RL AGENT  (uses NEXUS output as state, not raw symptoms)
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# Neural DQN Agent  (replaces flat Q-table)
# ─────────────────────────────────────────────────────────────
# Pure-numpy two-layer network — no PyTorch/TF dependency.
# State vector = numeric features extracted from NEXUS output.
# Output = Q-value for each (disease, treatment) action pair.
# ─────────────────────────────────────────────────────────────

import numpy as np

_DISEASES    = [p["disease"] for p in PATIENT_POOL]   # 10 diseases
_TREATMENTS  = TREATMENT_OPTIONS                        # 19 treatments
_N_DX        = len(_DISEASES)                          # 10
_N_TX        = len(_TREATMENTS)                        # 19
_N_ACTIONS   = _N_DX + _N_TX                          # used for compat only
_N_FEATURES  = 128

# Per-disease key differentiating symptoms — used for loss reweighting
# These are symptoms that strongly identify THIS disease over its confusables
_DISEASE_DIFFERENTIATORS = {
    "pneumonia":       {"chest pain", "cough", "fatigue"},
    "flu":             {"body aches", "headache"},
    "appendicitis":    {"abdominal pain", "vomiting", "fever"},
    "meningitis":      {"stiff neck", "confusion", "fever"},
    "heart attack":    {"left arm pain", "sweating", "chest pain", "shortness of breath"},
    "gastroenteritis": {"diarrhea", "vomiting", "abdominal pain"},
    "sepsis":          {"confusion", "weakness", "body aches"},
    "asthma":          {"wheezing", "chest tightness"},
    "migraine":        {"sensitivity to light", "dizziness"},
    "covid":           {"loss of smell", "shortness of breath"},
}  # updated by world-model feature vector


def _action_idx(dx: str, tx: str) -> Tuple[int, int]:
    di = _DISEASES.index(dx)   if dx  in _DISEASES   else 0
    ti = _TREATMENTS.index(tx) if tx in _TREATMENTS  else 0
    return di, ti


def _idx_to_action(dx_idx: int, tx_idx: int) -> Tuple[str, str]:
    return _DISEASES[dx_idx % _N_DX], _TREATMENTS[tx_idx % _N_TX]


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


# ── NEXUS symptom-to-system map (cached at module load) ──────
try:
    from nexus_engine.nexus_medical import SYMPTOM_TO_SYSTEM as _SYM2SYS
except ModuleNotFoundError:
    try:
        from nexus_medical import SYMPTOM_TO_SYSTEM as _SYM2SYS
    except ModuleNotFoundError:
        _SYM2SYS = {}

_SYSTEMS  = ["respiratory","cardiovascular","neurologic","gi","systemic","immune","unknown"]
_COMMON   = ["fever","cough","headache","chest pain","abdominal pain",
             "nausea","shortness of breath","fatigue","diarrhea","vomiting",
             "wheezing","body aches","stiff neck","confusion","left arm pain",
             "sweating","loss of smell","sensitivity to light","palpitations","weakness"]

# Feature layout (128-dim):
#   [0:10]   NEXUS disease probability scores (world-model primary)
#   [10:20]  NEXUS top-10 rank scores (position-weighted)
#   [20:27]  Organ system one-hot (from NEXUS symptom-system map)
#   [27:32]  NEXUS mechanism signals (red flags, consistency, spread risk)
#   [32:40]  Pathogen spread — critical organs at risk (anatomy reasoning)
#   [40:77]  Symptom one-hot (37 symptoms — NOISY, secondary signal)
#   [60:128] Zero padding
_N_FEATURES = 128


def _build_feature_vector(obs: dict, symptom_dropout: float = 0.0,
                          nexus_score_dropout: float = 0.0) -> np.ndarray:
    """
    World-model-first feature vector.

    symptom_dropout:     probability of zeroing each symptom bit (0=clean, 0.5=noisy)
    nexus_score_dropout: if >0, zero positions [0:10] (NEXUS disease scores).
                         Used in late training phases to break NEXUS-following bias
                         and force classifier to rely on symptom differentiators.
    """
    nr      = obs.get("nexus_result", {})
    dx_list = nr.get("nexus_diagnoses", [])
    vec     = np.zeros(_N_FEATURES, dtype=np.float32)

    # ── [0:10] NEXUS disease probability scores ───────────────
    # These ARE the world model output — mechanism chains → disease scores
    # Normalise to sum=1 (softmax-like) so relative ranking is preserved
    raw_scores = np.zeros(_N_DX, dtype=np.float32)
    for d in dx_list[:20]:
        name  = d.get("disease","").lower()
        score = float(d.get("score", 0))
        for j, known in enumerate(_DISEASES):
            known_words = set(known.split())
            name_words  = set(name.replace("-"," ").split())
            if known_words & name_words or known in name or name in known:
                raw_scores[j] = max(raw_scores[j], score)
                break
    total = raw_scores.sum()
    if nexus_score_dropout > 0.0 and random.random() < nexus_score_dropout:
        pass   # zero out NEXUS scores — force symptom-driven decision
    else:
        vec[0:_N_DX] = raw_scores / (total + 1e-9)   # normalised probabilities

    # ── [10:20] Reserved (top-3 NEXUS scores removed — redundant with [0:10]) ─
    # Ablation showed removing these +3% accuracy — they add noise not signal.
    # Slots 10-19 remain zero.

    # ── [20:27] Organ system from NEXUS symptom-system map ────
    sys_votes = np.zeros(len(_SYSTEMS), dtype=np.float32)
    for sym in obs.get("symptoms", []):
        sys_name = _SYM2SYS.get(sym, "unknown")
        if sys_name in _SYSTEMS:
            sys_votes[_SYSTEMS.index(sys_name)] += 1.0
    if sys_votes.sum() > 0:
        vec[20:27] = sys_votes / sys_votes.sum()  # distribution over systems

    # ── [27:32] NEXUS mechanism signals ───────────────────────
    flags = nr.get("nexus_red_flags", [])
    vec[27] = min(len(flags) / 5.0, 1.0)                                      # red flag intensity
    vec[28] = nr.get("nexus_consistency",{}).get("consistency_score", 0.5)    # reasoning consistency
    vec[29] = min(len(nr.get("nexus_suggested_questions", [])) / 5.0, 1.0)    # uncertainty proxy
    vec[30] = min(len(nr.get("nexus_root_causes", [])) / 3.0, 1.0)            # causal depth
    # Pathogen type from etiology
    etio = obs.get("etiology", {})
    etio_map = {"virus": 0.33, "bacteria": 0.67, "non-infectious": 1.0}
    vec[31] = etio_map.get(str(etio.get("top_etiology","")).lower(), 0.5)

    # ── [32:40] Anatomy — critical organs at risk ─────────────
    CRITICAL = ["brain","heart","lungs","meninges","liver","kidney","spleen","brainstem"]
    spread = nr.get("nexus_pathogen_spread", [])
    organ_risk = {s.get("organ",""): s.get("risk", 0) for s in spread if isinstance(s, dict)}
    for k, organ in enumerate(CRITICAL):
        vec[32 + k] = float(organ_risk.get(organ, 0.0))

    # ── [40:60] Symptom one-hot with differentiator weighting ─
    # High-value differentiating symptoms get 2x weight
    # so they can override shared symptom signals (e.g. chest pain > abdominal pain)
    _DIFFERENTIATORS = {
        # Cardiovascular emergencies
        "left arm pain":       2.5,   # almost pathognomonic for MI
        "chest pain":          1.8,   # cardiovascular system
        "sweating":            1.5,   # autonomic — heart attack / sepsis
        # Neurological emergencies
        "stiff neck":          2.5,   # meningitis
        "confusion":           2.0,   # sepsis / meningitis
        "sensitivity to light":1.8,  # migraine
        "dizziness":           1.3,
        "headache":            1.5,   # flu differentiator
        # Respiratory
        "wheezing":            2.0,   # asthma
        "loss of smell":       2.5,   # covid
        # GI
        "abdominal pain":      1.8,   # appendicitis / GI
        "diarrhea":            1.8,   # gastroenteritis
        # Systemic
        "body aches":          1.8,   # flu / sepsis — matches chest pain weight
        "weakness":            1.3,
        "shortness of breath": 1.4,
    }

    symptoms = obs.get("symptoms", [])
    for k, sym in enumerate(_COMMON):
        if sym in symptoms:
            if random.random() >= symptom_dropout:
                weight = _DIFFERENTIATORS.get(sym, 1.0)
                vec[40 + k] = min(weight, 2.5)  # cap at 2.5 to avoid domination

    # ── [60:70] Red-flag symptom combo signals ────────────────
    # Only computed when symptom_dropout is low (clean/partial phases)
    # During heavy noise training, combo signals are also suppressed
    # to prevent overfitting to specific symptom co-occurrence patterns
    if symptom_dropout <= 0.2:
        sym_set = set(symptoms)
    else:
        # During noisy training, compute from a subset to avoid overfitting
        sym_set = set(s for s in symptoms if random.random() > symptom_dropout * 0.5)
    # MI combo: chest pain + left arm pain (or sweating)
    if "chest pain" in sym_set and "left arm pain" in sym_set:
        vec[60] = 2.0
    elif "chest pain" in sym_set and "sweating" in sym_set:
        vec[60] = 1.5
    # Meningitis combo: headache + stiff neck (+ fever)
    if "headache" in sym_set and "stiff neck" in sym_set:
        vec[61] = 2.0
    if "headache" in sym_set and "stiff neck" in sym_set and "fever" in sym_set:
        vec[61] = 2.5
    # Sepsis combo: fever + confusion
    if "fever" in sym_set and "confusion" in sym_set:
        vec[62] = 1.8
    # Asthma combo: wheezing + shortness of breath
    if "wheezing" in sym_set and "shortness of breath" in sym_set:
        vec[63] = 2.0
    # GI combo: abdominal pain + nausea/vomiting (without chest pain)
    if "abdominal pain" in sym_set and ("nausea" in sym_set or "vomiting" in sym_set):
        if "chest pain" not in sym_set and "left arm pain" not in sym_set:
            vec[64] = 1.5   # GI signal (not cardiac)
    # Covid: loss of smell (pathognomonic)
    if "loss of smell" in sym_set:
        vec[65] = 2.5
    # Flu: body aches + headache + fever (without respiratory focus)
    if "body aches" in sym_set and "headache" in sym_set and "fever" in sym_set:
        vec[66] = 1.5

    return vec


# ─────────────────────────────────────────────────────────────
# Supervised classifier (replaces DQN)
# ─────────────────────────────────────────────────────────────
# This is the RIGHT approach for single-step medical diagnosis:
#   Input:  64-dim NEXUS feature vector
#   Output: probability over 10 diseases + 19 treatments
#   Loss:   cross-entropy  (not TD / Bellman)
#   Update: SGD with momentum on every episode
#
# Why not DQN:
#   Each episode = one patient = one correct label.
#   That's a classification problem. DQN is for sequential decisions.
#   CE loss converges in hundreds of steps; DQN needs thousands.
# ─────────────────────────────────────────────────────────────

class _Classifier:
    """Two-layer softmax classifier with SGD+momentum. Pure numpy."""

    def __init__(self, lr=5e-3, momentum=0.9):
        self.lr  = lr
        self.mom = momentum
        rng = np.random.default_rng(42)
        # He init — 128-dim input, 256-dim hidden (bigger for world-model features)
        self.W1  = rng.normal(0, np.sqrt(2/128), (128, 256)).astype(np.float32)
        self.b1  = np.zeros(256, dtype=np.float32)
        self.Wdx = rng.normal(0, np.sqrt(2/256), (256, _N_DX)).astype(np.float32)
        self.bdx = np.zeros(_N_DX, dtype=np.float32)
        self.Wtx = rng.normal(0, np.sqrt(2/256), (256, _N_TX)).astype(np.float32)
        self.btx = np.zeros(_N_TX, dtype=np.float32)
        # Momentum buffers (auto-sized from weights)
        self.vW1=np.zeros_like(self.W1);   self.vb1=np.zeros_like(self.b1)
        self.vWdx=np.zeros_like(self.Wdx); self.vbdx=np.zeros_like(self.bdx)
        self.vWtx=np.zeros_like(self.Wtx); self.vbtx=np.zeros_like(self.btx)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self._x = x
        self._h = np.maximum(0, x @ self.W1 + self.b1)  # (256,)
        return self._h @ self.Wdx + self.bdx, self._h @ self.Wtx + self.btx

    def predict(self, x: np.ndarray) -> Tuple[int, int]:
        dx_logits, tx_logits = self.forward(x)
        return int(np.argmax(dx_logits)), int(np.argmax(tx_logits))

    def train_step(self, x: np.ndarray, dx_label: int, tx_label: int,
                   dx_weight: float = 1.0, tx_weight: float = 1.0):
        """
        One cross-entropy gradient step.
        dx_weight / tx_weight scale the loss (use reward signal here).
        """
        # Forward
        h       = np.maximum(0, x @ self.W1 + self.b1)  # (256,)
        dx_prob = _softmax(h @ self.Wdx + self.bdx)
        tx_prob = _softmax(h @ self.Wtx + self.btx)

        # CE gradient: dL/dlogit = prob - one_hot(label)
        ddx          = dx_prob.copy(); ddx[dx_label] -= 1.0
        dtx          = tx_prob.copy(); dtx[tx_label] -= 1.0

        # Scale by reward signal (positive reward → learn this, negative → unlearn)
        ddx *= dx_weight
        dtx *= tx_weight

        # Backprop through heads
        dWdx = h[:, None] * ddx[None, :]   # (256, N_DX)
        dbdx = ddx
        dWtx = h[:, None] * dtx[None, :]   # (256, N_TX)
        dbtx = dtx

        # Backprop through trunk (256-dim)
        dh = (ddx @ self.Wdx.T + dtx @ self.Wtx.T) * (h > 0)
        dW1 = x[:, None] * dh[None, :]    # (128, 256)
        db1 = dh

        # SGD + momentum update
        def _step(param, vel, grad):
            vel[:] = self.mom * vel - self.lr * grad
            param += vel

        _step(self.Wdx, self.vWdx, dWdx)
        _step(self.bdx, self.vbdx, dbdx)
        _step(self.Wtx, self.vWtx, dWtx)
        _step(self.btx, self.vbtx, dbtx)
        _step(self.W1,  self.vW1,  dW1)
        _step(self.b1,  self.vb1,  db1)

        return float(-np.log(dx_prob[dx_label] + 1e-9)),                float(-np.log(tx_prob[tx_label] + 1e-9))

    def contrastive_step(self, x: np.ndarray, true_label: int,
                         confusable_label: int, strength: float = 0.5):
        """
        Contrastive gradient step for overlap training.

        Pushes true_label UP and confusable_label DOWN simultaneously.
        This directly addresses the "shared symptoms dominate" problem:
        the network learns to widen the margin between confusable diseases.

        strength: how hard to push the confusable label down (0.0-1.0)
        """
        h       = np.maximum(0, x @ self.W1 + self.b1)
        dx_prob = _softmax(h @ self.Wdx + self.bdx)

        # Standard CE gradient for true label (push UP)
        ddx = dx_prob.copy()
        ddx[true_label] -= 1.0

        # Margin penalty: push confusable label DOWN
        # If prob[confusable] > prob[true] - margin, add extra gradient
        margin = 0.2
        if dx_prob[confusable_label] > dx_prob[true_label] - margin:
            # Extra push: increase confusable gradient proportionally
            ddx[confusable_label] += strength * dx_prob[confusable_label]

        # Backprop
        dWdx = h[:, None] * ddx[None, :]
        dbdx = ddx
        dh   = (ddx @ self.Wdx.T) * (h > 0)
        dW1  = x[:, None] * dh[None, :]
        db1  = dh

        def _step(param, vel, grad):
            vel[:] = self.mom * vel - self.lr * grad
            param += vel

        _step(self.Wdx, self.vWdx, dWdx)
        _step(self.bdx, self.vbdx, dbdx)
        _step(self.W1,  self.vW1,  dW1)
        _step(self.b1,  self.vb1,  db1)

        margin_gap = float(dx_prob[true_label] - dx_prob[confusable_label])
        return margin_gap


class NexusRLAgent:
    """
    Supervised classifier agent for single-step medical diagnosis.

    Uses cross-entropy loss instead of DQN/TD-learning.
    The reward signal scales the gradient:
      positive reward  → strengthen this (dx, tx) pair
      negative reward  → weaken this (dx, tx) pair (gradient reversal)

    Same external API as the DQN version so train() needs no changes.
    """

    def __init__(
        self,
        alpha: float = 5e-3,       # SGD lr
        gamma: float = 0.95,       # kept for API compat, not used
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.9994,
        replay_every: int = 50,
        batch_size: int = 64,
        warmup: int = 300,
    ):
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.replay_every  = replay_every
        self.batch_size    = batch_size
        self.warmup        = warmup

        self._clf    = _Classifier(lr=alpha)
        self._episode = 0
        self.memory: deque = deque(maxlen=5000)
        self.Q: Dict[str, Dict[str, float]] = {}  # stub for len() reporting

    # ── state encoding ────────────────────────────────────────
    def encode_state(self, obs: dict) -> np.ndarray:
        """Returns a float32 numpy feature vector."""
        vec = _build_feature_vector(obs)
        key = str(obs.get("nexus_result",{}).get("nexus_diagnoses",[{}])[0:1])
        if key not in self.Q:
            self.Q[key] = {}
        return vec

    # ── action selection ──────────────────────────────────────
    def choose_action(self, state_vec: np.ndarray) -> Tuple[str, str]:
        """Epsilon-greedy over classifier logits."""
        if self._episode < self.warmup or random.random() < self.epsilon:
            return (random.choice(_DISEASES), random.choice(_TREATMENTS))
        di, ti = self._clf.predict(state_vec)
        return _idx_to_action(di, ti)

    def action_key(self, dx: str, tx: str) -> str:
        return f"{dx}::{tx}"

    # ── Classifier update ─────────────────────────────────────
    def update(self, state_vec: np.ndarray, dx: str, tx: str,
               reward: float, next_state_vec):
        di, ti = _action_idx(dx, tx)
        self.memory.append((state_vec.copy(), di, ti, float(reward)))
        self._episode += 1

        if self._episode < self.warmup:
            return

        # Scale gradient by reward sign:
        #   correct action (positive reward) → reinforce
        #   wrong action   (negative reward) → suppress (gradient flip)
        r = float(reward)
        # Only train on POSITIVE reward episodes — wrong guesses carry no signal
        # in supervised mode (the correct label is always known from the training loop).
        # Negative gradient reversal causes asymmetric pair bias.
        if r > 0:
            w = max(r / 2.5, 0.1)
            self._clf.train_step(state_vec, di, ti, dx_weight=w, tx_weight=w)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def replay(self, batch_size: int = 64):
        """Replay buffer: re-train on recent episodes weighted by reward."""
        n = min(batch_size, len(self.memory))
        if n < 16:
            return
        batch = random.sample(list(self.memory), n)
        for sv, di, ti, rw in batch:
            # Only replay positive-reward episodes
            # Negative replay with reversed gradient causes pair bias
            if rw > 0:
                w = max(rw / 2.5, 0.1)
                old_lr = self._clf.lr
                self._clf.lr = self._clf.lr * 0.3
                self._clf.train_step(sv, di, ti, dx_weight=w, tx_weight=w)
                self._clf.lr = old_lr


# ─────────────────────────────────────────────────────────────
# 3.  TRAINING LOOP  (puts it all together)
# ─────────────────────────────────────────────────────────────

def train(
    episodes: int = 500,
    print_every: int = 50,
    learn_every: int = 200,  # NEXUS KG replay (expensive — don't do too often)
    verbose: bool = True,
):
    """
    Full training loop:
      env (MedicalEnv) ↔ agent (NexusRLAgent) ↔ NexusLearner feedback

    After every `learn_every` episodes, NexusLearner re-reads case_records.jsonl
    and writes new KG triples — so the NEXUS reasoning actually improves over time.
    """

    # ── Init NEXUS ────────────────────────────────────────────
    import sys, os as _os
    # Works whether file lives at project root OR inside nexus_engine/
    _here = _os.path.dirname(_os.path.abspath(__file__))
    if _os.path.basename(_here) == 'nexus_engine':
        sys.path.insert(0, _os.path.dirname(_here))  # add project root
    try:
        from nexus_engine.nexus_medical import NexusMedical
        from nexus_engine.nexus_learning_bridge import NexusLearner
    except ModuleNotFoundError:
        from nexus_medical import NexusMedical          # fallback: already inside nexus_engine
        from nexus_learning_bridge import NexusLearner

    print("[TRAIN] Loading NEXUS knowledge web...")
    nexus = NexusMedical()
    nexus.load_knowledge()
    learner = NexusLearner(nexus)

    # Clear stale case records from previous runs — they poison the KG
    # with old wrong answers and inflate learn_from_cases overhead
    _case_file = "case_records.jsonl"
    if os.path.exists(_case_file):
        os.remove(_case_file)
        print(f"[TRAIN] Cleared stale {_case_file}")

    env = MedicalEnv(nexus, noise_p=0.15)
    # Use class defaults — don't override epsilon_decay here
    agent = NexusRLAgent(replay_every=learn_every)

    # ── Tracking ──────────────────────────────────────────────
    results = []
    kg_sizes = []
    episode_rewards = []

    print(f"[TRAIN] Starting {episodes} episodes...\n")

    # ── Noise symptom pool for training corruption ────────────
    _TRAIN_NOISE_SYMS = [
        "fatigue", "dizziness", "weakness", "loss of appetite", "insomnia",
        "sweating", "chills", "back pain", "joint pain", "sore throat",
        "runny nose", "rash", "anxiety", "palpitations", "dry mouth",
    ]

    for ep in range(1, episodes + 1):

        # 1. Generate clean patient (always start clean for label accuracy)
        obs     = env.reset()
        patient = env._current_patient
        true_dx = patient["disease"]
        true_tx = patient["correct_treatments"][0]
        di, ti  = _action_idx(true_dx, true_tx)

        # 2. 4-phase noise curriculum — corrupt SYMPTOMS before running NEXUS
        #    Phase 1 (ep 1-1000):    50% clean, 30% partial, 20% noisy
        #    Phase 2 (ep 1001-2500): 30% clean, 40% partial, 30% noisy
        #    Phase 3 (ep 2501-4000): 20% clean, 30% partial, 50% noisy
        #    Phase 4 (ep 4001+):     10% clean, 30% partial, 60% noisy
        if ep <= 1000:
            weights = [0.50, 0.30, 0.20]
        elif ep <= 2500:
            weights = [0.30, 0.40, 0.30]
        elif ep <= 4000:
            weights = [0.20, 0.30, 0.50]
        else:
            weights = [0.10, 0.30, 0.60]

        mode = random.choices(["clean", "partial", "noisy"], weights=weights)[0]

        base_syms = list(patient["symptoms"])

        if mode == "clean":
            train_syms = base_syms
            feat_dropout = 0.0

        elif mode == "partial":
            # Drop 1-2 symptoms (NEXUS sees incomplete info)
            n_drop = random.randint(1, min(2, len(base_syms) - 1))
            train_syms = base_syms[:]
            for _ in range(n_drop):
                if len(train_syms) > 1:
                    train_syms.pop(random.randrange(len(train_syms)))
            feat_dropout = 0.1

        elif mode == "noisy":
            # Drop 1-2 real symptoms AND add 1-3 irrelevant ones
            n_drop = random.randint(1, min(2, len(base_syms) - 1))
            train_syms = base_syms[:]
            for _ in range(n_drop):
                if len(train_syms) > 1:
                    train_syms.pop(random.randrange(len(train_syms)))
            n_add = random.randint(1, 3)
            for ns in random.sample(_TRAIN_NOISE_SYMS,
                                    min(n_add, len(_TRAIN_NOISE_SYMS))):
                if ns not in train_syms:
                    train_syms.append(ns)
            feat_dropout = 0.3

        # (hard_neg mode removed — caused systematic pair bias)

        # 3. Run NEXUS on the (possibly corrupted) symptoms
        #    This is the key: the world model features reflect noisy input
        if mode == "clean":
            nexus_result = env._current_nexus_result  # already computed in reset()
            train_obs    = obs
        else:
            nexus_result = env._run_nexus(train_syms)
            train_obs    = {**obs, "symptoms": train_syms,
                            "nexus_result": nexus_result}

        # 4. Build feature vector
        # Phase 3+ (ep>2500): also randomly zero NEXUS disease scores [0:10]
        # This forces the classifier to stop following NEXUS biases and instead
        # trust symptom differentiators for hard overlap pairs (e.g. meningitis vs migraine)
        nexus_dropout = 0.0
        if ep > 2500:
            nexus_dropout = 0.3   # 30% chance to zero NEXUS scores in phase 3
        if ep > 4000:
            nexus_dropout = 0.5   # 50% chance in phase 4

        state_vec = _build_feature_vector(train_obs, symptom_dropout=feat_dropout,
                                          nexus_score_dropout=nexus_dropout)

        # 5. Agent predicts on clean obs for accuracy tracking
        clean_sv  = _build_feature_vector(obs, symptom_dropout=0.0)
        diagnosis, treatment = agent.choose_action(clean_sv)

        # 6. Supervised update on (possibly noisy) state → correct label
        agent._clf.train_step(state_vec, di, ti, dx_weight=1.0, tx_weight=1.0)

        # Differentiator boost removed — fires asymmetrically across diseases
        # (diseases with more differentiators get more boost episodes)
        # Pure CE loss is sufficient with nexus_score_dropout

        agent._episode += 1

        # 7. Store in replay buffer (mix of clean + noisy for robust replay)
        reward = env._compute_reward(diagnosis, treatment, env._current_nexus_result)
        agent.memory.append((state_vec.copy(), di, ti, reward))

        if ep % learn_every == 0:
            agent.replay(batch_size=min(64, len(agent.memory)))
            kg_sizes.append(len(nexus.kg) if hasattr(nexus, "kg") else 0)

        # 8. NEXUS feedback (always use clean result for KG quality)
        try:
            learner.feedback(env._current_nexus_result, round_id=ep)
        except Exception:
            pass

        # 9. Track accuracy on clean observations
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
        correct_dx = env._matches(diagnosis, true_dx)
        correct_tx = any(env._matches(treatment, t) for t in patient["correct_treatments"])
        info = {
            "correct_diagnosis": correct_dx,
            "correct_treatment": correct_tx,
            "nexus_result": env._current_nexus_result,
        }

        env._log_case(patient["symptoms"], diagnosis, treatment,
                      env._current_nexus_result, reward)
        results.append(info)
        episode_rewards.append(reward)

        if verbose and ep % print_every == 0:
            window = results[-print_every:]
            dx_acc = sum(1 for r in window if r["correct_diagnosis"]) / len(window)
            tx_acc = sum(1 for r in window if r["correct_treatment"]) / len(window)
            avg_r  = sum(episode_rewards[-print_every:]) / print_every
            print(
                f"  Ep {ep:4d} | "
                f"Dx acc: {dx_acc:.0%} | "
                f"Tx acc: {tx_acc:.0%} | "
                f"Avg reward: {avg_r:+.2f} | "
                f"ε={agent.epsilon:.2f}"
            )

    # ── Final summary ─────────────────────────────────────────
    total_dx = sum(1 for r in results if r["correct_diagnosis"]) / len(results)
    total_tx = sum(1 for r in results if r["correct_treatment"]) / len(results)
    avg_reward = sum(episode_rewards) / len(episode_rewards)

    # One-time KG update from this run's best cases (reward > 0)
    try:
        learner.learn_from_cases("case_records.jsonl", min_reward=0.3)
        print(f"[TRAIN] KG updated from high-reward cases this run")
    except Exception:
        pass

    print(f"\n{'='*55}")
    print(f"  Training complete ({episodes} episodes)")
    print(f"  Final Dx accuracy :  {total_dx:.1%}")
    print(f"  Final Tx accuracy :  {total_tx:.1%}")
    print(f"  Avg reward        :  {avg_reward:+.3f}")
    print(f"  KG size at end    :  {len(nexus.kg) if hasattr(nexus, 'kg') else 'N/A'} triples")
    print(f"  Q-states learned  :  {len(agent.Q)}")
    print(f"{'='*55}\n")

    return agent, nexus, learner, results, env


# ─────────────────────────────────────────────────────────────
# 4.  QUICK-START
# ─────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────
# 4.  QUICK-START
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys as _sys, os as _os
    _here = _os.path.dirname(_os.path.abspath(__file__))
    _parent = _os.path.dirname(_here)
    # Ensure both nexus_engine/ and project root are importable
    for _p in [_here, _parent]:
        if _p not in _sys.path:
            _sys.path.insert(0, _p)

    # Suppress repeated [ANATOMY] prints
    import builtins as _bi
    _real_print = _bi.print
    _anatomy_seen = [False]
    def _filtered_print(*args, **kwargs):
        msg = " ".join(str(a) for a in args)
        if msg.startswith("[ANATOMY]"):
            if not _anatomy_seen[0]:
                _anatomy_seen[0] = True
                _real_print(*args, **kwargs)
            return
        _real_print(*args, **kwargs)
    _bi.print = _filtered_print

    agent, nexus, learner, results, env = train(
        episodes=5000,
        print_every=50,
        learn_every=200,
        verbose=True,
    )

    # ── DEMO: show final patient with trained classifier ──────
    print("\n[DEMO] Running one final patient through trained agent...")
    _saved_noise = env.noise_p
    env.noise_p = 0.0
    obs = env.reset()
    env.noise_p = _saved_noise

    state_vec = agent.encode_state(obs)
    # Pure exploitation
    saved_eps = agent.epsilon
    agent.epsilon = 0.0
    dx, tx = agent.choose_action(state_vec)
    agent.epsilon = saved_eps

    _, _, _, info = env.step(dx, tx)

    print(f"  Symptoms      : {obs['symptoms']}")
    print(f"  Agent guessed : diagnosis={dx}, treatment={tx}")
    print(f"  True disease  : {info['true_disease']}")
    print(f"  True treatment: {info['true_treatment']}")
    print(f"  Correct Dx    : {info['correct_diagnosis']}")
    print(f"  Correct Tx    : {info['correct_treatment']}")
    print(f"  Reward        : {info['reward']}")

    dx_list = obs["nexus_result"].get("nexus_diagnoses", [])
    if dx_list:
        print(f"\n  NEXUS top diagnoses:")
        for d in dx_list[:3]:
            print(f"    {d.get('disease','?'):25s}  score={d.get('score', 0):.3f}")