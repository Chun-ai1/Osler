"""
NEXUS Self-Learning Medical Environment  —  v2 (Config-Driven)
════════════════════════════════════════════════════════════════
To add diseases, symptoms, mechanisms, or treatments:
  1. Edit nexus_config.json — that's the ONLY file you need to touch.
  2. Run training as normal — everything else adjusts automatically.

The feature vector, network size, and curriculum all auto-scale to
however many diseases/symptoms you configure.

Quick-start:
  python nexus_engine/nexus_learning_env.py               # train with default config
  python nexus_engine/nexus_learning_env.py --config path  # custom config file
"""

from __future__ import annotations
import json, os, random, math
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np


# ─────────────────────────────────────────────────────────────
# CONFIG LOADER  — single source of truth
# ─────────────────────────────────────────────────────────────

def _find_config(explicit: str | None = None) -> Path:
    """Locate nexus_config.json — checks explicit path, then common locations."""
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
        raise FileNotFoundError(f"Config not found: {explicit}")

    here = Path(__file__).parent
    candidates = [
        here / "nexus_config.json",
        here.parent / "nexus_config.json",
        Path("nexus_config.json"),
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "nexus_config.json not found. "
        "Copy it to the project root or nexus_engine/ directory."
    )


class NexusConfig:
    """
    Loaded once at module import.  All hardcoded numbers come from here.

    Usage after adding a new disease to nexus_config.json:
        - No code changes needed.
        - Feature vector auto-expands.
        - Network input size auto-adjusts.
        - Curriculum, dropout schedules unchanged.
    """

    def __init__(self, path: str | None = None):
        cfg_path = _find_config(path)
        with open(cfg_path, encoding="utf-8") as f:
            raw = json.load(f)

        # ── Diseases ──────────────────────────────────────────
        self.diseases: List[dict] = raw["diseases"]
        self.disease_names: List[str] = [d["name"] for d in self.diseases]

        # ── Treatments ────────────────────────────────────────
        self.treatments: List[str] = raw["treatments"]

        # ── Symptom vocabulary (union of all disease symptoms) ─
        all_syms: set = set()
        for d in self.diseases:
            all_syms.update(d["symptoms"])
        # Preserve insertion order, then sort for determinism
        self.symptoms: List[str] = sorted(all_syms)

        # ── Symptom weights ───────────────────────────────────
        self.symptom_weights: Dict[str, float] = raw.get("symptom_weights", {})

        # ── Combo signals (ordered for consistent slot assignment) ──
        self.combo_signals: Dict[str, dict] = raw.get("combo_signals", {})
        # Remove meta key
        self.combo_signals = {k: v for k, v in self.combo_signals.items()
                              if not k.startswith("_")}

        # ── Disease → system map (for soft partial credit in reward) ──
        self.disease_system: Dict[str, str] = {
            d["name"].lower(): d.get("pathogen", "unknown")
            for d in self.diseases
        }
        # Try to enrich from disease_0001.json
        try:
            import json as _j2, os as _o2
            for _dp in ("disease_0001.json", "../disease_0001.json"):
                if _o2.path.exists(_dp):
                    _full = _j2.load(open(_dp, encoding="utf-8"))
                    for _d in (_full if isinstance(_full, list) else []):
                        _name = _d.get("disease_name", "").lower().strip()
                        _sys  = _d.get("system", "")
                        if _name and _sys:
                            self.disease_system[_name] = _sys
                    break
        except Exception:
            pass

        # ── Anatomy ───────────────────────────────────────────
        anat = raw.get("anatomy", {})
        self.critical_organs: List[str] = anat.get(
            "critical_organs",
            ["brain","heart","lungs","meninges","liver","kidney","spleen","brainstem"]
        )
        self.organ_systems: List[str] = anat.get(
            "organ_systems",
            ["respiratory","cardiovascular","neurologic","gi","systemic","immune","unknown"]
        )

        # ── Training hyperparameters ──────────────────────────
        tr = raw.get("training", {})
        self.episodes:            int   = tr.get("episodes", 5000)
        self.learning_rate:       float = tr.get("learning_rate", 5e-3)
        self.momentum:            float = tr.get("momentum", 0.9)
        self.hidden_size:         int   = tr.get("hidden_size", 256)
        self.replay_buffer_size:  int   = tr.get("replay_buffer_size", 5000)
        self.learn_every:         int   = tr.get("learn_every", 200)
        self.env_noise_p:         float = tr.get("env_noise_p", 0.15)
        self.noise_symptoms:      List[str] = tr.get("noise_symptoms", [])
        self.env_noise_symptoms:  List[str] = tr.get("env_noise_symptoms",
            ["fatigue","dizziness","weakness","loss of appetite","insomnia"])

        self.curriculum: List[dict] = tr.get("curriculum", [
            {"until_episode": 1000,  "clean": 0.50, "partial": 0.30, "noisy": 0.20},
            {"until_episode": 2500,  "clean": 0.30, "partial": 0.40, "noisy": 0.30},
            {"until_episode": 4000,  "clean": 0.20, "partial": 0.30, "noisy": 0.50},
            {"until_episode": 99999, "clean": 0.10, "partial": 0.30, "noisy": 0.60},
        ])

        self.nexus_score_dropout_schedule: List[dict] = tr.get("nexus_score_dropout", [
            {"until_episode": 2500,  "rate": 0.0},
            {"until_episode": 4000,  "rate": 0.3},
            {"until_episode": 99999, "rate": 0.5},
        ])

        # ── Derived sizes ─────────────────────────────────────
        self.N_DX      = len(self.disease_names)
        self.N_TX      = len(self.treatments)
        self.N_SYM     = len(self.symptoms)
        self.N_COMBOS  = len(self.combo_signals)
        self.N_ORGANS  = len(self.critical_organs)
        self.N_SYSTEMS = len(self.organ_systems)

        # Feature vector layout (auto-computed):
        #   [0           : N_DX]           NEXUS disease probability scores
        #   [N_DX        : N_DX+10]        Reserved (zero)
        #   [N_DX+10     : N_DX+10+N_SYS]  Organ system distribution
        #   [N_DX+10+N_SYS : +5]           NEXUS mechanism signals
        #   [... : +N_ORGANS]              Anatomy spread risk
        #   [... : +N_SYM]                 Symptom one-hot (weighted)
        #   [... : +N_COMBOS]              Combo signals
        #   Total padded to multiple of 64
        self.FEAT_DX_START    = 0
        self.FEAT_DX_END      = self.N_DX
        self.FEAT_SYS_START   = self.N_DX + 10
        self.FEAT_SYS_END     = self.FEAT_SYS_START + self.N_SYSTEMS
        self.FEAT_MECH_START  = self.FEAT_SYS_END
        self.FEAT_MECH_END    = self.FEAT_MECH_START + 5
        self.FEAT_ANAT_START  = self.FEAT_MECH_END
        self.FEAT_ANAT_END    = self.FEAT_ANAT_START + self.N_ORGANS
        self.FEAT_SYM_START   = self.FEAT_ANAT_END
        self.FEAT_SYM_END     = self.FEAT_SYM_START + self.N_SYM
        self.FEAT_COMBO_START = self.FEAT_SYM_END
        self.FEAT_COMBO_END   = self.FEAT_COMBO_START + self.N_COMBOS

        raw_dim = self.FEAT_COMBO_END
        # Pad to next multiple of 64 for cache-alignment
        self.N_FEATURES = math.ceil(raw_dim / 64) * 64
        self.INPUT_SIZE = self.N_FEATURES

    def curriculum_weights(self, episode: int) -> List[float]:
        """Return [clean, partial, noisy] weights for current episode."""
        for phase in self.curriculum:
            if episode <= phase["until_episode"]:
                return [phase["clean"], phase["partial"], phase["noisy"]]
        return [0.10, 0.30, 0.60]

    def nexus_score_dropout(self, episode: int) -> float:
        """Return nexus_score_dropout rate for current episode."""
        for phase in self.nexus_score_dropout_schedule:
            if episode <= phase["until_episode"]:
                return phase["rate"]
        return 0.5

    def symptom_weight(self, sym: str) -> float:
        return self.symptom_weights.get(sym, 1.0)

    def build_patient_pool(self) -> List[dict]:
        """
        Convert config diseases → PATIENT_POOL format.
        Rare diseases (≤3 defining symptoms) are oversampled 3×
        so the agent trains on long-tail cases more often.
        """
        base = [
            {
                "disease":            d["name"],
                "symptoms":           d["symptoms"],
                "severity":           d.get("severity", "moderate"),
                "pathogen":           d.get("pathogen", "unknown"),
                "infected_organ":     d.get("infected_organ", "unknown"),
                "system":             d.get("infected_organ", "unknown"),
                "correct_treatments": d["treatments"],
            }
            for d in self.diseases
        ]
        # Long-tail oversampling: sparse profiles seen 3× as often
        pool = []
        for p in base:
            pool.append(p)
            if len(p["symptoms"]) <= 3:
                pool.extend([p, p])
        return pool

    def summary(self) -> str:
        lines = [
            f"NexusConfig loaded:",
            f"  Diseases  : {self.N_DX}  ({', '.join(self.disease_names)})",
            f"  Treatments: {self.N_TX}",
            f"  Symptoms  : {self.N_SYM}",
            f"  Combos    : {self.N_COMBOS}",
            f"  Feature dim: {self.N_FEATURES} (raw={self.FEAT_COMBO_END})",
            f"  Network   : {self.INPUT_SIZE} → {self.hidden_size} → {self.N_DX}/{self.N_TX}",
            f"  Episodes  : {self.episodes}",
        ]
        return "\n".join(lines)


# ── Module-level config instance (loaded once) ────────────────
_CFG: NexusConfig | None = None

def get_config(path: str | None = None) -> NexusConfig:
    global _CFG
    if _CFG is None or path is not None:
        _CFG = NexusConfig(path)
    return _CFG

def reload_config(path: str | None = None) -> NexusConfig:
    """Force reload — call after editing nexus_config.json."""
    global _CFG
    _CFG = NexusConfig(path)
    return _CFG


# ─────────────────────────────────────────────────────────────
# PATIENT POOL  (built from config)
# ─────────────────────────────────────────────────────────────

def _build_patient_pool_from_config(cfg: NexusConfig) -> List[dict]:
    # NOTE: After building the base pool, rare diseases (≤2 symptoms) are
    # oversampled 3× so the agent trains on them more often.
    # This directly addresses the long-tail problem.
    return cfg.build_patient_pool()


# ─────────────────────────────────────────────────────────────
# ACTION HELPERS  (config-driven)
# ─────────────────────────────────────────────────────────────

def _action_idx(dx: str, tx: str, cfg: NexusConfig) -> Tuple[int, int]:
    di = cfg.disease_names.index(dx) if dx in cfg.disease_names else 0
    ti = cfg.treatments.index(tx)    if tx in cfg.treatments    else 0
    return di, ti

def _idx_to_action(di: int, ti: int, cfg: NexusConfig) -> Tuple[str, str]:
    return cfg.disease_names[di % cfg.N_DX], cfg.treatments[ti % cfg.N_TX]

def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


# ─────────────────────────────────────────────────────────────
# FEATURE VECTOR  (auto-sized from config)
# ─────────────────────────────────────────────────────────────

def _build_feature_vector(
    obs: dict,
    cfg: NexusConfig,
    sym2sys: dict,
    symptom_dropout: float = 0.0,
    nexus_score_dropout: float = 0.0,
) -> np.ndarray:
    """
    Auto-sized world-model feature vector.

    All slice indices come from cfg — adding diseases/symptoms/combos
    automatically changes the layout without any code edits.
    """
    nr      = obs.get("nexus_result", {})
    dx_list = nr.get("nexus_diagnoses", [])
    vec     = np.zeros(cfg.N_FEATURES, dtype=np.float32)

    # ── NEXUS disease probability scores ─────────────────────
    raw_scores = np.zeros(cfg.N_DX, dtype=np.float32)
    for d in dx_list[:cfg.N_DX * 2]:
        name  = d.get("disease", "").lower()
        score = float(d.get("score", 0))
        for j, known in enumerate(cfg.disease_names):
            known_words = set(known.split())
            name_words  = set(name.replace("-", " ").split())
            if known_words & name_words or known in name or name in known:
                raw_scores[j] = max(raw_scores[j], score)
                break
    total = raw_scores.sum()
    if nexus_score_dropout > 0.0 and random.random() < nexus_score_dropout:
        pass  # zero — force symptom-driven decision
    else:
        vec[cfg.FEAT_DX_START:cfg.FEAT_DX_END] = raw_scores / (total + 1e-9)

    # ── Organ system distribution ─────────────────────────────
    sys_votes = np.zeros(cfg.N_SYSTEMS, dtype=np.float32)
    for sym in obs.get("symptoms", []):
        sys_name = sym2sys.get(sym, "unknown")
        if sys_name in cfg.organ_systems:
            sys_votes[cfg.organ_systems.index(sys_name)] += 1.0
    if sys_votes.sum() > 0:
        vec[cfg.FEAT_SYS_START:cfg.FEAT_SYS_END] = sys_votes / sys_votes.sum()

    # ── NEXUS mechanism signals ───────────────────────────────
    flags = nr.get("nexus_red_flags", [])
    ms = cfg.FEAT_MECH_START
    vec[ms]   = min(len(flags) / 5.0, 1.0)
    vec[ms+1] = nr.get("nexus_consistency", {}).get("consistency_score", 0.5)
    vec[ms+2] = min(len(nr.get("nexus_suggested_questions", [])) / 5.0, 1.0)
    vec[ms+3] = min(len(nr.get("nexus_root_causes", [])) / 3.0, 1.0)
    etio      = obs.get("etiology", {})
    etio_map  = {"virus": 0.33, "bacteria": 0.67, "non-infectious": 1.0}
    vec[ms+4] = etio_map.get(str(etio.get("top_etiology", "")).lower(), 0.5)

    # ── Anatomy spread risk ───────────────────────────────────
    spread     = nr.get("nexus_pathogen_spread", [])
    organ_risk = {s.get("organ", ""): s.get("risk", 0)
                  for s in spread if isinstance(s, dict)}
    for k, organ in enumerate(cfg.critical_organs):
        vec[cfg.FEAT_ANAT_START + k] = float(organ_risk.get(organ, 0.0))

    # ── Symptom one-hot (weighted, dropout-able) ──────────────
    symptoms = obs.get("symptoms", [])
    for k, sym in enumerate(cfg.symptoms):
        if sym in symptoms and random.random() >= symptom_dropout:
            vec[cfg.FEAT_SYM_START + k] = min(cfg.symptom_weight(sym), 2.5)

    # ── Combo signals ─────────────────────────────────────────
    # Gate combos during heavy dropout to avoid overfitting
    if symptom_dropout <= 0.2:
        sym_set = set(symptoms)
    else:
        sym_set = set(s for s in symptoms if random.random() > symptom_dropout * 0.5)

    for k, (name, combo) in enumerate(cfg.combo_signals.items()):
        requires = combo.get("requires", [])
        excludes = combo.get("excludes", [])
        if (all(r in sym_set for r in requires) and
                not any(e in sym_set for e in excludes)):
            vec[cfg.FEAT_COMBO_START + k] = combo.get("value", 1.5)

    return vec


# ─────────────────────────────────────────────────────────────
# ENVIRONMENT
# ─────────────────────────────────────────────────────────────

def _patch_nexus_atlas_cache(nexus_instance):
    import types
    try:
        from nexus_engine.anatomy_atlas import AnatomyAtlas
        from nexus_engine.pathogen_tracker import PathogenTracker
    except ModuleNotFoundError:
        try:
            from anatomy_atlas import AnatomyAtlas
            from pathogen_tracker import PathogenTracker
        except ModuleNotFoundError:
            return

    _cached_atlas   = AnatomyAtlas()
    _cached_tracker = PathogenTracker(_cached_atlas)

    def _fast_predict(self, sym_set, top_diseases, detected_systems):
        gi_syms      = {"diarrhea","vomiting","nausea","abdominal pain","bloating"}
        resp_syms    = {"cough","shortness of breath","wheezing","sore throat"}
        neuro_syms   = {"headache","numbness","dizziness","weakness","confusion"}
        cardiac_syms = {"chest pain","palpitations"}

        infected_organs, pathogen_type = [], "virus"
        if sym_set & gi_syms:
            infected_organs.append("stomach")
            if "diarrhea" in sym_set: infected_organs.append("ileum")
        if sym_set & resp_syms:
            infected_organs.append("lungs")
        if sym_set & neuro_syms:
            infected_organs.append("brain")
        if sym_set & cardiac_syms:
            infected_organs.append("heart")
        if "fever" in sym_set and not infected_organs:
            infected_organs.append("spleen")
        if not infected_organs:
            return []

        infected_organs = list(set(infected_organs))[:3]
        result = _cached_tracker.track_multiple(infected_organs, pathogen_type, max_hops=6)

        predictions = []
        for oar in result.get("combined_risk", [])[:8]:
            predictions.append({"organ": oar["organ"], "risk": oar["risk"],
                                 "via": oar["via"], "path": oar.get("path",""),
                                 "from": oar.get("infected_from","")})
        for organ, sources in result.get("convergence_points", {}).items():
            predictions.append({"organ": organ, "risk": 0.95, "via": "convergence",
                                 "path": f"Reachable from {' and '.join(sources)}",
                                 "from": "multiple"})
        return predictions

    nexus_instance._predict_pathogen_spread = types.MethodType(_fast_predict, nexus_instance)
    print("[PATCH] NexusMedical._predict_pathogen_spread now uses cached AnatomyAtlas ✓")


class MedicalEnv:
    """
    Config-driven Gym-style environment.
    Adding diseases to nexus_config.json automatically expands this env.
    """

    def __init__(self, nexus_medical, cfg: NexusConfig):
        self.nexus    = nexus_medical
        self.cfg      = cfg
        self.noise_p  = cfg.env_noise_p
        self._patient_pool = cfg.build_patient_pool()
        self._current_patient      = None
        self._current_nexus_result = None
        self._episode = 0

        self._atlas      = self._load_atlas()
        self._etiology   = self._load_etiology()

        # Load NEXUS symptom-to-system map
        try:
            from nexus_engine.nexus_medical import SYMPTOM_TO_SYSTEM as s2s
        except ModuleNotFoundError:
            try:
                from nexus_medical import SYMPTOM_TO_SYSTEM as s2s
            except ModuleNotFoundError:
                s2s = {}
        self._sym2sys = s2s

        _patch_nexus_atlas_cache(self.nexus)

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

    def reset(self) -> dict:
        self._episode += 1
        template = random.choice(self._patient_pool)
        symptoms = list(template["symptoms"])
        for ns in self.cfg.env_noise_symptoms:
            if random.random() < self.noise_p and ns not in symptoms:
                symptoms.append(ns)

        self._current_patient = {
            **template,
            "symptoms": symptoms,
            "age":      random.randint(18, 80),
            "episode":  self._episode,
        }

        nexus_result = self._run_nexus(symptoms)
        self._current_nexus_result = nexus_result

        etiology = {}
        if self._etiology:
            try:
                etiology = self._etiology.classify(symptoms)
            except Exception:
                pass

        return {
            "symptoms":        symptoms,
            "nexus_result":    nexus_result,
            "etiology":        etiology,
            "pathogen_spread": nexus_result.get("nexus_pathogen_spread", []),
        }

    def step(self, diagnosis: str, treatment: str) -> Tuple[dict, float, bool, dict]:
        p            = self._current_patient
        nexus_result = self._current_nexus_result or self._run_nexus(p["symptoms"])
        reward       = self._compute_reward(diagnosis, treatment)

        info = {
            "true_disease":     p["disease"],
            "true_treatment":   p["correct_treatments"],
            "nexus_result":     nexus_result,
            "correct_diagnosis": self._matches(diagnosis, p["disease"]),
            "correct_treatment": any(self._matches(treatment, t)
                                     for t in p["correct_treatments"]),
            "reward": reward,
        }
        self._log_case(p["symptoms"], diagnosis, treatment, nexus_result, reward)
        return None, reward, True, info

    def _compute_reward(self, diagnosis: str, treatment: str) -> float:
        p = self._current_patient

        # Try physiology-aware reward first (nexus_reward.py)
        try:
            from nexus_reward import PhysiologyReward
            if not hasattr(self, "_phys_reward"):
                self._phys_reward = PhysiologyReward(self.nexus, self._atlas)
            total, breakdown = self._phys_reward.compute(
                diagnosis,
                treatment,
                self._current_nexus_result or {},   # nexus_result (3rd arg)
                p,                                  # patient (4th arg)
            )
            # Log breakdown for analysis
            if not hasattr(self, "_last_reward_breakdown"):
                self._last_reward_breakdown = {}
            self._last_reward_breakdown = breakdown
            return round(total, 3)
        except Exception as _re:
            pass  # fall through to simple reward

        # Fallback: exact match ±1 with soft partial credit
        exact_dx = self._matches(diagnosis, p["disease"])
        exact_tx = any(self._matches(treatment, t) for t in p["correct_treatments"])

        r = 1.0 if exact_dx else -1.0
        r += 1.0 if exact_tx else -1.0

        # Soft partial credit: same body system as correct disease → +0.3
        # Helps long-tail: agent learns system-level reasoning even for rare diseases
        if not exact_dx:
            true_sys  = p.get("system", "")
            pred_sys  = self.cfg.disease_system.get(diagnosis.lower(), "")
            if true_sys and pred_sys and true_sys == pred_sys:
                r += 0.3  # partial credit — right system, wrong disease

        return round(r, 3)

    def _run_nexus(self, symptoms: List[str]) -> dict:
        try:
            return self.nexus.enhance_pipeline_result({
                "symptoms": symptoms,
                "final_symptoms": symptoms,
                "top_diseases": [],
                "reasoning": "",
            })
        except Exception as e:
            return {"error": str(e), "nexus_diagnoses": [], "nexus_consistency": {}}

    def build_state(self, obs: dict, **kw) -> np.ndarray:
        return _build_feature_vector(obs, self.cfg, self._sym2sys, **kw)

    @staticmethod
    def _matches(a: str, b: str) -> bool:
        a, b = a.lower().strip(), b.lower().strip()
        return a == b or a in b or b in a

    @staticmethod
    def _log_case(symptoms, diagnosis, treatment, nexus_result, reward):
        record = {
            "final_symptoms": symptoms,
            "top_diseases":   nexus_result.get("nexus_diagnoses", [])[:5],
            "agent_diagnosis": diagnosis,
            "agent_treatment": treatment,
            "reward_final":   {"final": reward},
        }
        with open("case_records.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


# ─────────────────────────────────────────────────────────────
# CLASSIFIER  (auto-sizes to config)
# ─────────────────────────────────────────────────────────────

class _Classifier:
    """
    Two-layer softmax classifier.
    Input size and output heads auto-determined from NexusConfig.
    """

    def __init__(self, cfg: NexusConfig):
        self.cfg = cfg
        self.lr  = cfg.learning_rate
        self.mom = cfg.momentum
        rng = np.random.default_rng(42)

        IN  = cfg.INPUT_SIZE
        HID = cfg.hidden_size
        NDX = cfg.N_DX
        NTX = cfg.N_TX

        self.W1  = rng.normal(0, np.sqrt(2/IN),  (IN,  HID)).astype(np.float32)
        self.b1  = np.zeros(HID, dtype=np.float32)
        self.Wdx = rng.normal(0, np.sqrt(2/HID), (HID, NDX)).astype(np.float32)
        self.bdx = np.zeros(NDX, dtype=np.float32)
        self.Wtx = rng.normal(0, np.sqrt(2/HID), (HID, NTX)).astype(np.float32)
        self.btx = np.zeros(NTX, dtype=np.float32)

        self.vW1  = np.zeros_like(self.W1);  self.vb1  = np.zeros_like(self.b1)
        self.vWdx = np.zeros_like(self.Wdx); self.vbdx = np.zeros_like(self.bdx)
        self.vWtx = np.zeros_like(self.Wtx); self.vbtx = np.zeros_like(self.btx)

    def forward(self, x):
        h = np.maximum(0, x @ self.W1 + self.b1)
        return h @ self.Wdx + self.bdx, h @ self.Wtx + self.btx

    def predict(self, x):
        dx_l, tx_l = self.forward(x)
        return int(np.argmax(dx_l)), int(np.argmax(tx_l))

    def train_step(self, x, dx_label, tx_label, dx_weight=1.0, tx_weight=1.0):
        h       = np.maximum(0, x @ self.W1 + self.b1)
        dx_prob = _softmax(h @ self.Wdx + self.bdx)
        tx_prob = _softmax(h @ self.Wtx + self.btx)

        ddx = dx_prob.copy(); ddx[dx_label] -= 1.0; ddx *= dx_weight
        dtx = tx_prob.copy(); dtx[tx_label] -= 1.0; dtx *= tx_weight

        dWdx = h[:, None] * ddx[None, :]; dbdx = ddx
        dWtx = h[:, None] * dtx[None, :]; dbtx = dtx
        dh   = (ddx @ self.Wdx.T + dtx @ self.Wtx.T) * (h > 0)
        dW1  = x[:, None] * dh[None, :]; db1 = dh

        def _step(p, v, g):
            v[:] = self.mom * v - self.lr * g
            p += v

        _step(self.Wdx, self.vWdx, dWdx); _step(self.bdx, self.vbdx, dbdx)
        _step(self.Wtx, self.vWtx, dWtx); _step(self.btx, self.vbtx, dbtx)
        _step(self.W1,  self.vW1,  dW1);  _step(self.b1,  self.vb1,  db1)

        return (float(-np.log(dx_prob[dx_label] + 1e-9)),
                float(-np.log(tx_prob[tx_label] + 1e-9)))


# ─────────────────────────────────────────────────────────────
# AGENT
# ─────────────────────────────────────────────────────────────

class NexusRLAgent:
    """Config-driven supervised classifier agent."""

    def __init__(self, cfg: NexusConfig):
        self.cfg      = cfg
        self._clf     = _Classifier(cfg)
        self._episode = 0
        self.memory   = deque(maxlen=cfg.replay_buffer_size)
        self.epsilon  = 1.0
        self.epsilon_min   = 0.05
        self.epsilon_decay = 0.9994
        self.warmup        = 300
        self.Q: dict = {}   # stub for compatibility

    def encode_state(self, obs: dict, env: MedicalEnv, **kw) -> np.ndarray:
        return env.build_state(obs, **kw)

    def choose_action(self, state_vec: np.ndarray) -> Tuple[str, str]:
        if self._episode < self.warmup or random.random() < self.epsilon:
            return (random.choice(self.cfg.disease_names),
                    random.choice(self.cfg.treatments))
        di, ti = self._clf.predict(state_vec)
        return _idx_to_action(di, ti, self.cfg)

    def replay(self, batch_size: int = 64):
        n     = min(batch_size, len(self.memory))
        if n < 16:
            return
        batch = random.sample(list(self.memory), n)
        for sv, di, ti, rw in batch:
            if rw > 0:
                old_lr = self._clf.lr
                self._clf.lr *= 0.3
                self._clf.train_step(sv, di, ti,
                                     dx_weight=max(rw/2.5, 0.1),
                                     tx_weight=max(rw/2.5, 0.1))
                self._clf.lr = old_lr


# ─────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────

def train(
    cfg:          NexusConfig | None = None,
    episodes:     int | None = None,
    print_every:  int = 50,
    verbose:      bool = True,
):
    """
    Train the agent.  All hyperparameters come from NexusConfig.
    Pass episodes= to override the config value for quick tests.
    """
    if cfg is None:
        cfg = get_config()

    n_eps = episodes or cfg.episodes

    # ── Init NEXUS ────────────────────────────────────────────
    import sys, os as _os
    _here = _os.path.dirname(_os.path.abspath(__file__))
    if _os.path.basename(_here) == "nexus_engine":
        sys.path.insert(0, _os.path.dirname(_here))
    try:
        from nexus_engine.nexus_medical import NexusMedical
        from nexus_engine.nexus_learning_bridge import NexusLearner
    except ModuleNotFoundError:
        from nexus_medical import NexusMedical
        from nexus_learning_bridge import NexusLearner

    print(cfg.summary())
    print("[TRAIN] Loading NEXUS knowledge web...")
    nexus   = NexusMedical()
    nexus.load_knowledge()
    learner = NexusLearner(nexus)

    case_file = "case_records.jsonl"
    if os.path.exists(case_file):
        os.remove(case_file)

    env   = MedicalEnv(nexus, cfg)
    agent = NexusRLAgent(cfg)

    results, episode_rewards = [], []

    print(f"[TRAIN] Starting {n_eps} episodes...\n")

    for ep in range(1, n_eps + 1):

        obs     = env.reset()
        patient = env._current_patient
        true_dx = patient["disease"]
        true_tx = patient["correct_treatments"][0]
        di, ti  = _action_idx(true_dx, true_tx, cfg)

        # ── Curriculum mode ───────────────────────────────────
        weights = cfg.curriculum_weights(ep)
        mode    = random.choices(["clean", "partial", "noisy"], weights=weights)[0]

        base_syms = list(patient["symptoms"])
        train_syms, feat_dropout = base_syms, 0.0

        if mode == "partial":
            n_drop = random.randint(1, min(2, len(base_syms) - 1))
            train_syms = base_syms[:]
            for _ in range(n_drop):
                if len(train_syms) > 1:
                    train_syms.pop(random.randrange(len(train_syms)))
            feat_dropout = 0.1

        elif mode == "noisy":
            n_drop = random.randint(1, min(2, len(base_syms) - 1))
            train_syms = base_syms[:]
            for _ in range(n_drop):
                if len(train_syms) > 1:
                    train_syms.pop(random.randrange(len(train_syms)))
            n_add = random.randint(1, 3)
            for ns in random.sample(cfg.noise_symptoms,
                                    min(n_add, len(cfg.noise_symptoms))):
                if ns not in train_syms:
                    train_syms.append(ns)
            feat_dropout = 0.3

        # ── Run NEXUS on (possibly corrupted) symptoms ─────────
        if mode == "clean":
            nexus_result = env._current_nexus_result
            train_obs    = obs
        else:
            nexus_result = env._run_nexus(train_syms)
            train_obs    = {**obs, "symptoms": train_syms,
                            "nexus_result": nexus_result}

        # ── Build feature vector ───────────────────────────────
        nd = cfg.nexus_score_dropout(ep)
        state_vec = env.build_state(train_obs,
                                    symptom_dropout=feat_dropout,
                                    nexus_score_dropout=nd)

        # ── Supervised update ──────────────────────────────────
        agent._clf.train_step(state_vec, di, ti)
        agent._episode += 1

        # ── Track accuracy on clean obs ────────────────────────
        clean_sv           = env.build_state(obs)
        diagnosis, treatment = agent.choose_action(clean_sv)
        agent.epsilon      = max(agent.epsilon_min,
                                  agent.epsilon * agent.epsilon_decay)

        reward     = env._compute_reward(diagnosis, treatment)
        correct_dx = env._matches(diagnosis, true_dx)
        correct_tx = any(env._matches(treatment, t)
                         for t in patient["correct_treatments"])

        agent.memory.append((state_vec.copy(), di, ti, reward))

        if ep % cfg.learn_every == 0:
            agent.replay(batch_size=min(64, len(agent.memory)))

        try:
            learner.feedback(env._current_nexus_result, round_id=ep)
        except Exception:
            pass

        results.append({"correct_diagnosis": correct_dx,
                         "correct_treatment": correct_tx,
                         "nexus_result": env._current_nexus_result})
        episode_rewards.append(reward)

        breakdown = getattr(env, "_last_reward_breakdown", {})
        env._log_case(patient["symptoms"], diagnosis, treatment,
                      env._current_nexus_result, reward)
        if breakdown and verbose and ep % print_every == 0:
            # Show physiology breakdown on summary lines
            r_sep = breakdown.get("r_sepsis_cascade", 0)
            r_chem= breakdown.get("r_chemistry", 0)
            chem  = breakdown.get("chem_alerts", [])
            if r_sep < -0.5 or r_chem < -0.5:
                print(f"         ↳ sepsis={r_sep:+.2f} chem={r_chem:+.2f}"
                      + (f" alerts={chem[:2]}" if chem else ""))

        if verbose and ep % print_every == 0:
            w = results[-print_every:]
            print(
                f"  Ep {ep:5d} | "
                f"Dx {sum(r['correct_diagnosis'] for r in w)/len(w):.0%} | "
                f"Tx {sum(r['correct_treatment'] for r in w)/len(w):.0%} | "
                f"Reward {sum(episode_rewards[-print_every:])/print_every:+.2f} | "
                f"ε={agent.epsilon:.2f}"
            )

    # ── Final KG update ───────────────────────────────────────
    try:
        learner.learn_from_cases(case_file, min_reward=0.3)
        print("[TRAIN] KG updated from high-reward cases")
    except Exception:
        pass

    total_dx = sum(r["correct_diagnosis"] for r in results) / len(results)
    total_tx = sum(r["correct_treatment"]  for r in results) / len(results)
    avg_r    = sum(episode_rewards) / len(episode_rewards)

    print(f"\n{'='*55}")
    print(f"  Training complete ({n_eps} episodes)")
    print(f"  Final Dx accuracy :  {total_dx:.1%}")
    print(f"  Final Tx accuracy :  {total_tx:.1%}")
    print(f"  Avg reward        :  {avg_r:+.3f}")
    print(f"{'='*55}\n")

    return agent, nexus, learner, results, env, cfg


# ─────────────────────────────────────────────────────────────
# CHECKPOINT  (save/load with config verification)
# ─────────────────────────────────────────────────────────────

def save_checkpoint(agent: NexusRLAgent, path: str = "nexus_checkpoint.npz"):
    clf = agent._clf
    cfg = agent.cfg
    np.savez(path,
             W1=clf.W1, b1=clf.b1,
             Wdx=clf.Wdx, bdx=clf.bdx,
             Wtx=clf.Wtx, btx=clf.btx,
             diseases   = np.array(cfg.disease_names),
             treatments = np.array(cfg.treatments),
             symptoms   = np.array(cfg.symptoms),
             n_features = np.array([cfg.N_FEATURES]),
             epsilon    = np.array([agent.epsilon]),
             episode    = np.array([agent._episode]))
    sz = os.path.getsize(path) // 1024
    print(f"[CKPT] Saved → {path}  ({sz} KB, "
          f"{cfg.N_DX} diseases, {cfg.N_SYM} symptoms, dim={cfg.N_FEATURES})")


def load_checkpoint(path: str, cfg: NexusConfig | None = None) -> NexusRLAgent:
    if cfg is None:
        cfg = get_config()
    d = np.load(path, allow_pickle=True)

    # Verify compatibility
    saved_diseases = list(d["diseases"])
    if saved_diseases != cfg.disease_names:
        raise ValueError(
            f"Checkpoint disease list doesn't match config.\n"
            f"  Checkpoint: {saved_diseases}\n"
            f"  Config:     {cfg.disease_names}\n"
            f"Delete the checkpoint and retrain, or align your config."
        )
    saved_nf = int(d["n_features"][0])
    if saved_nf != cfg.N_FEATURES:
        raise ValueError(
            f"Checkpoint feature dim {saved_nf} ≠ config feature dim {cfg.N_FEATURES}.\n"
            f"Config was changed since this checkpoint was saved. Delete and retrain."
        )

    agent = NexusRLAgent(cfg)
    clf   = agent._clf
    clf.W1[:] = d["W1"]; clf.b1[:] = d["b1"]
    clf.Wdx[:] = d["Wdx"]; clf.bdx[:] = d["bdx"]
    clf.Wtx[:] = d["Wtx"]; clf.btx[:] = d["btx"]
    agent.epsilon  = float(d["epsilon"][0])
    agent._episode = int(d["episode"][0])

    print(f"[CKPT] Loaded ← {path}  "
          f"(ep={agent._episode}, dim={cfg.N_FEATURES}, ε={agent.epsilon:.3f})")
    return agent


# ─────────────────────────────────────────────────────────────
# QUICK-START
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys as _sys, os as _os, argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    default=None, help="Path to nexus_config.json")
    parser.add_argument("--episodes",  type=int,     help="Override episode count")
    parser.add_argument("--checkpoint",default=None, help="Load from checkpoint")
    parser.add_argument("--save-to",   default="nexus_checkpoint.npz")
    args = parser.parse_args()

    _here   = _os.path.dirname(_os.path.abspath(__file__))
    _parent = _os.path.dirname(_here)
    for _p in [_here, _parent]:
        if _p not in _sys.path:
            _sys.path.insert(0, _p)

    # Suppress repeated [ANATOMY] spam
    import builtins as _bi
    _rp = _bi.print
    _seen = [False]
    def _qp(*a, **k):
        if " ".join(str(x) for x in a).startswith("[ANATOMY]"):
            if not _seen[0]: _seen[0] = True; _rp(*a, **k)
            return
        _rp(*a, **k)
    _bi.print = _qp

    cfg = get_config(args.config)

    if args.checkpoint:
        print(f"[INFO] Loading checkpoint: {args.checkpoint}")
        agent = load_checkpoint(args.checkpoint, cfg)
        # Still need env for demo
        import importlib.util
        from nexus_engine.nexus_medical import NexusMedical
        nexus = NexusMedical(); nexus.load_knowledge()
        env   = MedicalEnv(nexus, cfg)
    else:
        agent, nexus, _, _, env, cfg = train(cfg=cfg, episodes=args.episodes)
        save_checkpoint(agent, args.save_to)

    # Demo
    print("\n[DEMO] Running final patient through trained agent...")
    env.noise_p = 0.0
    obs = env.reset()

    saved_eps   = agent.epsilon
    agent.epsilon = 0.0
    sv  = env.build_state(obs)
    dx, tx = agent.choose_action(sv)
    agent.epsilon = saved_eps

    _, _, _, info = env.step(dx, tx)
    print(f"  Symptoms  : {obs['symptoms']}")
    print(f"  Predicted : {dx} → {tx}")
    print(f"  True      : {info['true_disease']} → {info['true_treatment']}")
    print(f"  Correct Dx: {info['correct_diagnosis']} | Tx: {info['correct_treatment']}")
    print(f"  Reward    : {info['reward']}")
    for d in obs["nexus_result"].get("nexus_diagnoses", [])[:3]:
        print(f"    NEXUS: {d.get('disease','?'):25s}  score={d.get('score',0):.3f}")