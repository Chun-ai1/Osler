"""
NEXUS State Model — JSON-driven Physiology State Simulation Engine

Loads organ state models from medical_knowledge/state_models/*.json.
Each organ file declares its state variables, derivation rules, and the
diseases that use perturbations on those variables.

Architecture (Session B+C):
   organ JSON file
        ↓ load
   StateModel object (variables + rules + disease perturbations)
        ↓ apply disease perturbations to a fresh BodyState
   BodyState (organ → variable → value)
        ↓ evaluate derivation rules
   derived symptoms with rationale

Adding a new disease (no Python edits needed):
   1. Add a perturbation entry to medical_knowledge/state_models/<organ>.json
      under "diseases":
        "<Disease Name>": {
          "review_status": "unreviewed",
          "perturbations": [{"variable": "...", "delta": ..., "cause": "..."}]
        }
   2. If the organ doesn't exist yet, create medical_knowledge/state_models/<organ>.json
      following the schema in bladder.json / lung.json / gi.json.
   3. Re-run NEXUS — disease is picked up automatically.
"""

from __future__ import annotations
import json
import os
import glob
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ════════════════════════════════════════════════════════════════════
# 1. BodyState
# ════════════════════════════════════════════════════════════════════

class BodyState:
    """Tracks state variables per organ. All values clamped to [0.0, 1.0]."""

    def __init__(self):
        self.organs: Dict[str, Dict[str, float]] = {}

    def set(self, organ: str, variable: str, value: float):
        v = max(0.0, min(1.0, float(value)))
        self.organs.setdefault(organ, {})[variable] = v

    def get(self, organ: str, variable: str, default: float = 0.0) -> float:
        return self.organs.get(organ, {}).get(variable, default)

    def apply_delta(self, organ: str, variable: str, delta: float):
        old = self.get(organ, variable, 0.0)
        self.set(organ, variable, old + delta)

    def all_active(self, threshold: float = 0.1) -> List[tuple]:
        result = []
        for o, vars_dict in self.organs.items():
            for v, val in vars_dict.items():
                if val > threshold:
                    result.append((o, v, val))
        return result

    def copy(self) -> "BodyState":
        new = BodyState()
        for o, vars_dict in self.organs.items():
            new.organs[o] = dict(vars_dict)
        return new


# ════════════════════════════════════════════════════════════════════
# 2. Condition evaluator — supports {state, op, threshold}, "all", "any"
# ════════════════════════════════════════════════════════════════════

def _evaluate_condition(cond: dict, state: BodyState, organ: str) -> bool:
    """Evaluate a JSON condition against BodyState in the context of an organ."""
    if not cond:
        return False
    if "all" in cond:
        return all(_evaluate_condition(c, state, organ) for c in cond["all"])
    if "any" in cond:
        return any(_evaluate_condition(c, state, organ) for c in cond["any"])
    state_var = cond.get("state")
    op = cond.get("op")
    threshold = cond.get("threshold")
    if state_var is None or op is None or threshold is None:
        return False
    val = state.get(organ, state_var, 0.0)
    if op == ">":  return val > threshold
    if op == ">=": return val >= threshold
    if op == "<":  return val < threshold
    if op == "<=": return val <= threshold
    if op == "==": return abs(val - threshold) < 1e-6
    return False


# ════════════════════════════════════════════════════════════════════
# 3. Data structures
# ════════════════════════════════════════════════════════════════════

@dataclass
class StateVariable:
    name: str
    range: tuple = (0.0, 1.0)
    description: str = ""
    inverse: bool = False
    default: float = None  # initial healthy value; None = auto (0.0 normal, 1.0 inverse)


@dataclass
class DerivationRule:
    symptom: str
    condition: dict
    rationale: str = ""
    confidence: float = 1.0

    def applies(self, state: BodyState, organ: str) -> bool:
        try:
            return _evaluate_condition(self.condition, state, organ)
        except Exception:
            return False


@dataclass
class Perturbation:
    organ: str
    variable: str
    delta: float
    cause: str = ""


@dataclass
class PropagationRule:
    """Rule that propagates state changes either within or across organs.

    Trigger evaluates on a source organ's state; effect applies a delta to
    (same organ, different variable) OR (different organ, some variable).

    Used in 3 contexts:
      - intra-organ: heart.ischemia high → heart.contractility low
      - cross-organ (hand-written): heart.cardiac_output low → kidney.perfusion low
      - cross-organ (atlas-derived): from atlas_connection_semantics + atlas
    """
    trigger_organ: str           # which organ to check
    trigger_var: str             # which state variable
    trigger_op: str              # >, <, >=, <=, ==
    trigger_threshold: float
    effect_organ: str            # which organ to affect (= trigger_organ for intra-)
    effect_var: str              # which variable to change
    effect_delta: float
    rationale: str = ""
    via: str = "intra_organ"
    review_status: str = "unreviewed"
    name: str = ""

    def applies(self, state: BodyState) -> bool:
        val = state.get(self.trigger_organ, self.trigger_var, 0.0)
        op = self.trigger_op
        th = self.trigger_threshold
        if op == ">":  return val > th
        if op == ">=": return val >= th
        if op == "<":  return val < th
        if op == "<=": return val <= th
        if op == "==": return abs(val - th) < 1e-6
        return False

    def fire(self, state: BodyState):
        state.apply_delta(self.effect_organ, self.effect_var, self.effect_delta)


@dataclass
class DiseaseModel:
    name: str
    organ: str
    perturbations: List[Perturbation] = field(default_factory=list)
    description: str = ""
    review_status: str = "unreviewed"


@dataclass
class OrganModel:
    """Container for one organ's variables + rules + diseases + propagation."""
    organ: str
    variables: Dict[str, StateVariable] = field(default_factory=dict)
    rules: List[DerivationRule] = field(default_factory=list)
    diseases: Dict[str, DiseaseModel] = field(default_factory=dict)
    propagation_rules: List[PropagationRule] = field(default_factory=list)  # NEW
    metadata: dict = field(default_factory=dict)


@dataclass
class AtlasConnectionSemantic:
    """How NEXUS interprets one atlas connection_type for state propagation."""
    connection_type: str
    transmits_state: Optional[str]
    decay: float = 0.5
    polarity: str = "positive"  # 'positive' or 'negative'
    threshold: float = 0.5
    rationale: str = ""


# ════════════════════════════════════════════════════════════════════
# 4. Loader
# ════════════════════════════════════════════════════════════════════

class StateModelRegistry:
    """Lazy-loaded registry of all organ state models from JSON files."""

    def __init__(self):
        self.organs: Dict[str, OrganModel] = {}
        self._disease_index: Dict[str, OrganModel] = {}
        # Cross-organ propagation rules (loaded from cross_organ.json)
        self.global_propagation_rules: List[PropagationRule] = []
        # Atlas connection semantics — how each connection_type transmits state
        self.atlas_semantics: Dict[str, AtlasConnectionSemantic] = {}
        # D.4 Neural framework — referred pain pathways, cranial nerves
        self.neural_framework: dict = {}
        self.referred_pain_pathways: list = []
        # D.5 Vascular framework — territories, watershed, embolic sources
        self.vascular_framework: dict = {}
        self.arterial_territories: list = []
        self.watershed_zones: list = []
        self.embolic_sources: list = []
        # Structural framework — comprehensive human body connectivity
        self.structural_framework: dict = {}
        self.organ_systems: dict = {}
        self.connection_type_semantics: dict = {}
        self.axes_and_loops: dict = {}
        self.regional_anatomy: dict = {}
        self.embryologic_origins: dict = {}
        self.barrier_systems: dict = {}
        # Lab integration — bridges layer5_labs to state variables
        self.lab_integration: dict = {}
        self.lab_to_state_mapping: dict = {}
        # Vital signs framework
        self.vital_signs: dict = {}
        self.shock_indices: dict = {}
        self.critical_vital_combinations: dict = {}
        # Pain framework
        self.pain_qualities: dict = {}
        self.pain_patterns: dict = {}
        self.pain_radiation_patterns: dict = {}
        # Time dynamics
        self.time_scales: dict = {}
        self.disease_timecourses: dict = {}
        self.red_flag_temporal_patterns: dict = {}
        # Bone + gland anatomy extensions
        self.bones_framework: dict = {}
        self.exocrine_glands: dict = {}
        # F.7-F.13 + F.14-F.19 — generic medical frameworks
        # Each loaded as a single dict from its JSON; queryable via get_framework().
        self.pharmacology_framework: dict = {}
        self.microbiology_framework: dict = {}
        self.imaging_framework: dict = {}
        self.immunology_framework: dict = {}
        self.oncology_framework: dict = {}
        self.risk_factors_framework: dict = {}
        self.epidemiology_framework: dict = {}
        self.specialty_psychiatry: dict = {}
        self.specialty_pediatrics: dict = {}
        self.specialty_obstetrics: dict = {}
        self.specialty_dermatology: dict = {}
        self.specialty_ophthalmology: dict = {}
        self.specialty_surgery: dict = {}
        # Symptom aliases (G.3)
        self.symptom_aliases: dict = {}
        self._symptom_alias_lookup: dict = {}  # phrase → canonical
        self._loaded = False

    def load_all(self, base_dir: Optional[str] = None) -> None:
        if self._loaded:
            return
        candidates = [
            base_dir,
            "medical_knowledge/state_models",
            "../medical_knowledge/state_models",
            os.path.join(os.path.dirname(__file__), "..",
                         "medical_knowledge", "state_models"),
        ]
        target_dir = None
        for c in candidates:
            if c and os.path.isdir(c):
                target_dir = c
                break
        if not target_dir:
            self._loaded = True
            return
        # Special-case framework files; everything else is organ JSON
        SPECIAL = {"cross_organ.json", "neural_framework.json",
                   "vascular_framework.json", "structural_framework.json",
                   "atlas_coverage.json",
                   "lab_integration.json", "vital_signs_framework.json",
                   "pain_framework.json", "time_dynamics.json",
                   "anatomy_bones.json", "anatomy_glands.json",
                   "pharmacology_framework.json", "microbiology_framework.json",
                   "imaging_framework.json", "immunology_framework.json",
                   "oncology_framework.json", "risk_factors_framework.json",
                   "epidemiology_framework.json",
                   "specialty_psychiatry.json", "specialty_pediatrics.json",
                   "specialty_obstetrics.json", "specialty_dermatology.json",
                   "specialty_ophthalmology.json", "specialty_surgery.json",
                   "symptom_aliases.json", "disease_timecourse.json",
                   "drug_mechanisms.json"}
        for path in sorted(glob.glob(os.path.join(target_dir, "*.json"))):
            if os.path.basename(path) in SPECIAL:
                continue
            try:
                self._load_organ_file(path)
            except Exception as e:
                print(f"[state_model] WARNING: failed to load {path}: {e}")
        # Load cross-organ propagation if present
        cross_path = os.path.join(target_dir, "cross_organ.json")
        if os.path.exists(cross_path):
            try:
                self._load_cross_organ_file(cross_path)
            except Exception as e:
                print(f"[state_model] WARNING: failed to load cross_organ.json: {e}")
        # Load neural framework (D.4)
        neural_path = os.path.join(target_dir, "neural_framework.json")
        if os.path.exists(neural_path):
            try:
                self._load_neural_framework(neural_path)
            except Exception as e:
                print(f"[state_model] WARNING: failed to load neural_framework.json: {e}")
        # Load vascular framework (D.5)
        vasc_path = os.path.join(target_dir, "vascular_framework.json")
        if os.path.exists(vasc_path):
            try:
                self._load_vascular_framework(vasc_path)
            except Exception as e:
                print(f"[state_model] WARNING: failed to load vascular_framework.json: {e}")
        # Load structural framework (E.1) — comprehensive human body skeleton
        struct_path = os.path.join(target_dir, "structural_framework.json")
        if os.path.exists(struct_path):
            try:
                self._load_structural_framework(struct_path)
            except Exception as e:
                print(f"[state_model] WARNING: failed to load structural_framework.json: {e}")
        # Load atlas coverage mapping (E.2) — bridges atlas (259) → state model (16)
        _load_atlas_coverage_file()
        _autocategorize_unmapped_atlas_organs()

        # Load lab integration (F.1)
        lab_path = os.path.join(target_dir, "lab_integration.json")
        if os.path.exists(lab_path):
            try:
                self._load_lab_integration(lab_path)
            except Exception as e:
                print(f"[state_model] WARNING: failed to load lab_integration.json: {e}")
        # Load vital signs (F.2)
        vs_path = os.path.join(target_dir, "vital_signs_framework.json")
        if os.path.exists(vs_path):
            try:
                self._load_vital_signs(vs_path)
            except Exception as e:
                print(f"[state_model] WARNING: failed to load vital_signs_framework.json: {e}")
        # Load pain framework (F.3)
        pain_path = os.path.join(target_dir, "pain_framework.json")
        if os.path.exists(pain_path):
            try:
                self._load_pain_framework(pain_path)
            except Exception as e:
                print(f"[state_model] WARNING: failed to load pain_framework.json: {e}")
        # Load time dynamics (F.4)
        time_path = os.path.join(target_dir, "time_dynamics.json")
        if os.path.exists(time_path):
            try:
                self._load_time_dynamics(time_path)
            except Exception as e:
                print(f"[state_model] WARNING: failed to load time_dynamics.json: {e}")
        # Load bone framework (F.5)
        bone_path = os.path.join(target_dir, "anatomy_bones.json")
        if os.path.exists(bone_path):
            try:
                self._load_bones(bone_path)
            except Exception as e:
                print(f"[state_model] WARNING: failed to load anatomy_bones.json: {e}")
        # Load gland framework (F.6)
        gland_path = os.path.join(target_dir, "anatomy_glands.json")
        if os.path.exists(gland_path):
            try:
                self._load_glands(gland_path)
            except Exception as e:
                print(f"[state_model] WARNING: failed to load anatomy_glands.json: {e}")

        # F.7-F.19 — Generic medical frameworks (pharm, micro, imaging,
        # immuno, onco, risk, epi, psych, peds, OB, derm, ophthal, surgery).
        # Each loaded as a single dict attached to the registry under
        # an attribute matching the filename.
        _generic_frameworks = [
            ("pharmacology_framework.json",   "pharmacology_framework"),
            ("microbiology_framework.json",   "microbiology_framework"),
            ("imaging_framework.json",        "imaging_framework"),
            ("immunology_framework.json",     "immunology_framework"),
            ("oncology_framework.json",       "oncology_framework"),
            ("risk_factors_framework.json",   "risk_factors_framework"),
            ("epidemiology_framework.json",   "epidemiology_framework"),
            ("specialty_psychiatry.json",     "specialty_psychiatry"),
            ("specialty_pediatrics.json",     "specialty_pediatrics"),
            ("specialty_obstetrics.json",     "specialty_obstetrics"),
            ("specialty_dermatology.json",    "specialty_dermatology"),
            ("specialty_ophthalmology.json",  "specialty_ophthalmology"),
            ("specialty_surgery.json",        "specialty_surgery"),
        ]
        for fname, attr in _generic_frameworks:
            path = os.path.join(target_dir, fname)
            if os.path.exists(path):
                try:
                    with open(path, encoding="utf-8") as f:
                        setattr(self, attr, json.load(f))
                except Exception as e:
                    print(f"[state_model] WARNING: failed to load {fname}: {e}")

        # Load symptom aliases (G.3) — used to expand state-derived symptoms
        # to match cascade/user vocabulary
        aliases_path = os.path.join(target_dir, "symptom_aliases.json")
        if os.path.exists(aliases_path):
            try:
                with open(aliases_path, encoding="utf-8") as f:
                    aliases_data = json.load(f)
                self.symptom_aliases = aliases_data.get("aliases", {})
                # Build reverse lookup: any alias → canonical phrase
                # Also: canonical → canonical (self-map)
                self._symptom_alias_lookup = {}
                for canonical, alias_list in self.symptom_aliases.items():
                    norm_canon = canonical.lower().strip()
                    self._symptom_alias_lookup[norm_canon] = norm_canon
                    for alias in alias_list:
                        norm_alias = alias.lower().strip()
                        # Don't overwrite if alias already maps to a different canonical
                        if norm_alias not in self._symptom_alias_lookup:
                            self._symptom_alias_lookup[norm_alias] = norm_canon
            except Exception as e:
                print(f"[state_model] WARNING: failed to load symptom_aliases.json: {e}")

        self._loaded = True
        if self.organs:
            covered = sum(1 for v in _ATLAS_TO_STATE_ORGAN.values() if v)
            n_kinds = len(set(_ATLAS_ORGAN_KIND.values()))
            print(f"[state_model] Loaded {len(self.organs)} organs, "
                  f"{len(self._disease_index)} disease models, "
                  f"{len(self.global_propagation_rules)} cross-organ rules, "
                  f"{len(self.atlas_semantics)} atlas semantics, "
                  f"{len(self.referred_pain_pathways)} referred-pain pathways, "
                  f"{len(self.arterial_territories)} arterial territories, "
                  f"{len(self.organ_systems)} organ systems, "
                  f"{len(self.regional_anatomy)} regions, "
                  f"{len(self.axes_and_loops)} axes, "
                  f"{len(self.lab_to_state_mapping)} lab mappings, "
                  f"{len(self.vital_signs)} vital signs, "
                  f"{len(self.pain_qualities)} pain qualities, "
                  f"{len(self.disease_timecourses)} disease timecourses, "
                  f"{len(self.bones_framework)} bones, "
                  f"{len(self.exocrine_glands)} glands, "
                  f"atlas coverage: {covered} explicit + "
                  f"{len(_ATLAS_ORGAN_KIND)} categorized ({n_kinds} kinds)")

    def _load_organ_file(self, path: str) -> None:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        meta = data.get("_metadata", {})
        organ_name = meta.get("organ") or os.path.splitext(os.path.basename(path))[0]
        organ = OrganModel(organ=organ_name, metadata=meta)
        for var_name, var_meta in (data.get("state_variables") or {}).items():
            organ.variables[var_name] = StateVariable(
                name=var_name,
                range=tuple(var_meta.get("range", [0.0, 1.0])),
                description=var_meta.get("description", ""),
                inverse=bool(var_meta.get("inverse", False)),
                default=var_meta.get("default"),  # may be None
            )
        for rule_data in (data.get("derivation_rules") or []):
            organ.rules.append(DerivationRule(
                symptom=rule_data["symptom"],
                condition=rule_data["condition"],
                rationale=rule_data.get("rationale", ""),
                confidence=rule_data.get("confidence", 1.0),
            ))
        # Parse intra-organ and outgoing propagation rules
        for prule_data in (data.get("propagation_rules") or []):
            trigger = prule_data["trigger"]
            effect = prule_data["effect"]
            # Default effect organ = same as trigger organ (intra-organ rule)
            effect_organ = effect.get("organ", organ_name)
            trigger_organ = trigger.get("organ", organ_name)
            organ.propagation_rules.append(PropagationRule(
                trigger_organ=trigger_organ,
                trigger_var=trigger["state"],
                trigger_op=trigger["op"],
                trigger_threshold=float(trigger["threshold"]),
                effect_organ=effect_organ,
                effect_var=effect["state"],
                effect_delta=float(effect["delta"]),
                rationale=prule_data.get("rationale", ""),
                via=prule_data.get("via", "intra_organ"),
                review_status=prule_data.get("review_status", "unreviewed"),
                name=prule_data.get("name", ""),
            ))
        for disease_name, dz_data in (data.get("diseases") or {}).items():
            perts = []
            for p in dz_data.get("perturbations", []):
                # Allow disease to perturb other organs via optional "organ" field
                # (e.g. URI in lung organ also perturbs upper_airway state)
                pert_organ = p.get("organ", organ_name)
                perts.append(Perturbation(
                    organ=pert_organ,
                    variable=p["variable"],
                    delta=float(p["delta"]),
                    cause=p.get("cause", ""),
                ))
            model = DiseaseModel(
                name=disease_name,
                organ=organ_name,
                perturbations=perts,
                description=dz_data.get("description", ""),
                review_status=dz_data.get("review_status", "unreviewed"),
            )
            organ.diseases[disease_name] = model
            self._disease_index[disease_name] = organ
        self.organs[organ_name] = organ

    def _load_cross_organ_file(self, path: str) -> None:
        """Load cross_organ.json — global propagation rules + atlas semantics."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        # Atlas connection semantics
        for conn_type, sem in (data.get("atlas_connection_semantics") or {}).items():
            if not isinstance(sem, dict):
                continue
            transmits = sem.get("transmits_state")
            if transmits is None:
                continue
            self.atlas_semantics[conn_type] = AtlasConnectionSemantic(
                connection_type=conn_type,
                transmits_state=transmits,
                decay=float(sem.get("decay", 0.5)),
                polarity=sem.get("polarity", "positive"),
                threshold=float(sem.get("threshold", 0.5)),
                rationale=sem.get("rationale", ""),
            )
        # Global propagation rules
        for rule in (data.get("global_propagation_rules") or []):
            trigger = rule["trigger"]
            effect = rule["effect"]
            self.global_propagation_rules.append(PropagationRule(
                trigger_organ=trigger["organ"],
                trigger_var=trigger["state"],
                trigger_op=trigger["op"],
                trigger_threshold=float(trigger["threshold"]),
                effect_organ=effect["organ"],
                effect_var=effect["state"],
                effect_delta=float(effect["delta"]),
                rationale=rule.get("rationale", ""),
                via=rule.get("via", "circulates_through"),
                review_status=rule.get("review_status", "unreviewed"),
                name=rule.get("name", ""),
            ))

    def _load_neural_framework(self, path: str) -> None:
        """Load neural framework (D.4) — referred pain pathways + cranial nerves
        + neural propagation rules."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.neural_framework = data
        self.referred_pain_pathways = data.get("referred_pain_pathways", []) or []
        # Add neural propagation rules to global pool
        for rule in (data.get("neural_propagation_rules") or []):
            trigger = rule["trigger"]
            effect = rule["effect"]
            self.global_propagation_rules.append(PropagationRule(
                trigger_organ=trigger["organ"],
                trigger_var=trigger["state"],
                trigger_op=trigger["op"],
                trigger_threshold=float(trigger["threshold"]),
                effect_organ=effect["organ"],
                effect_var=effect["state"],
                effect_delta=float(effect["delta"]),
                rationale=rule.get("rationale", ""),
                via=rule.get("via", "neural"),
                review_status=rule.get("review_status", "unreviewed"),
                name=rule.get("name", ""),
            ))

    def _load_vascular_framework(self, path: str) -> None:
        """Load vascular framework (D.5) — territories + watershed + embolic
        sources + vascular propagation rules."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.vascular_framework = data
        self.arterial_territories = data.get("arterial_territories", []) or []
        self.watershed_zones = data.get("watershed_zones", []) or []
        self.embolic_sources = data.get("embolic_sources", []) or []
        # Add vascular propagation rules to global pool
        for rule in (data.get("vascular_propagation_rules") or []):
            trigger = rule["trigger"]
            effect = rule["effect"]
            self.global_propagation_rules.append(PropagationRule(
                trigger_organ=trigger["organ"],
                trigger_var=trigger["state"],
                trigger_op=trigger["op"],
                trigger_threshold=float(trigger["threshold"]),
                effect_organ=effect["organ"],
                effect_var=effect["state"],
                effect_delta=float(effect["delta"]),
                rationale=rule.get("rationale", ""),
                via=rule.get("via", "arterial"),
                review_status=rule.get("review_status", "unreviewed"),
                name=rule.get("name", ""),
            ))

    def _load_structural_framework(self, path: str) -> None:
        """Load structural framework (E.1) — comprehensive human body skeleton:
        organ systems, connection type semantics, regulatory axes, regional
        anatomy, embryologic origins, barriers."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.structural_framework = data
        self.organ_systems = data.get("organ_systems", {}) or {}
        self.connection_type_semantics = data.get("connection_type_semantics", {}) or {}
        self.axes_and_loops = data.get("axes_and_loops", {}) or {}
        self.regional_anatomy = data.get("regional_anatomy", {}) or {}
        self.embryologic_origins = data.get("embryologic_origins", {}) or {}
        self.barrier_systems = data.get("barrier_systems", {}) or {}

    def _load_lab_integration(self, path: str) -> None:
        """Load lab→state mapping (F.1)."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.lab_integration = data
        self.lab_to_state_mapping = data.get("lab_to_state_mapping", {}) or {}

    def _load_vital_signs(self, path: str) -> None:
        """Load vital signs framework (F.2)."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.vital_signs = data.get("vital_signs", {}) or {}
        self.shock_indices = data.get("shock_indices", {}) or {}
        self.critical_vital_combinations = data.get("critical_combinations", {}) or {}

    def _load_pain_framework(self, path: str) -> None:
        """Load pain characterization framework (F.3)."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.pain_qualities = data.get("pain_qualities", {}) or {}
        self.pain_patterns = data.get("pain_patterns", {}) or {}
        self.pain_radiation_patterns = data.get("pain_radiation_patterns", {}) or {}

    def _load_time_dynamics(self, path: str) -> None:
        """Load time dynamics framework (F.4)."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.time_scales = data.get("time_scales", {}) or {}
        self.disease_timecourses = data.get("disease_timecourses", {}) or {}
        self.red_flag_temporal_patterns = data.get("red_flag_temporal_patterns", {}) or {}

    def _load_bones(self, path: str) -> None:
        """Load bone anatomy extension (F.5)."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.bones_framework = data.get("bones", {}) or {}

    def _load_glands(self, path: str) -> None:
        """Load exocrine gland anatomy extension (F.6)."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.exocrine_glands = data.get("exocrine_glands", {}) or {}

    def has_disease(self, disease_name: str) -> bool:
        self.load_all()
        return disease_name in self._disease_index

    def __contains__(self, disease_name: str) -> bool:
        return self.has_disease(disease_name)

    def get(self, disease_name: str, default=None):
        """Dict-like get — returns DiseaseModel or default. Used by
        nexus_medical.reason() for backward compat with the old dict API."""
        self.load_all()
        organ = self._disease_index.get(disease_name)
        return organ.diseases.get(disease_name) if organ else default

    def get_disease(self, disease_name: str) -> Optional[DiseaseModel]:
        return self.get(disease_name)

    def get_organ_for_disease(self, disease_name: str) -> Optional[OrganModel]:
        self.load_all()
        return self._disease_index.get(disease_name)

    def all_diseases(self) -> List[str]:
        self.load_all()
        return sorted(self._disease_index.keys())

    def __bool__(self):
        self.load_all()
        return bool(self._disease_index)

    def __len__(self):
        self.load_all()
        return len(self._disease_index)


# Global singleton
_REGISTRY = StateModelRegistry()


# ════════════════════════════════════════════════════════════════════
# 5. Forward pipeline — disease → state → symptoms
# ════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════
# Atlas-organ → state-model-organ resolver
# ════════════════════════════════════════════════════════════════════

# Atlas uses anatomical names (r_kidney, l_kidney, heart, lungs); state model
# uses generic organ names (kidney, heart, lung). This map resolves atlas →
# state-model organ. Multiple atlas organs may map to the same state organ.
# Atlas uses anatomical names (r_kidney, l_kidney, heart, lungs); state model
# uses generic organ names (kidney, heart, lung). This map resolves atlas →
# state-model organ. Loaded from medical_knowledge/state_models/atlas_coverage.json
# at registry load time. Bootstrap fallback covers the most common cases for
# environments where the JSON file is missing.
_ATLAS_TO_STATE_ORGAN: Dict[str, str] = {
    "heart": "heart",
    "r_kidney": "kidney", "l_kidney": "kidney",
    "lungs": "lung", "r_lung": "lung", "l_lung": "lung",
    "bladder": "bladder", "stomach": "gi", "duodenum": "gi",
    "jejunum": "gi", "ileum": "gi", "asc_colon": "gi",
    "desc_colon": "gi", "sigmoid": "gi", "transverse_colon": "gi",
    "esophagus": "gi", "small_intestine": "gi", "blood": "blood",
}

# Atlas organ → kind classification (organ, vessel, nerve, duct, etc.)
# Also loaded from atlas_coverage.json. Affects which atlas entities
# participate in state-derived propagation.
_ATLAS_ORGAN_KIND: Dict[str, str] = {}


# Module-level cached AnatomyAtlas (instantiated lazily, once per process).
# Prevents the ~3000× re-instantiation that caused the validation slowdown.
_CACHED_ATLAS = None
_CACHED_ATLAS_FAILED = False


def _get_cached_atlas():
    """Return module-level singleton AnatomyAtlas. Created on first call."""
    global _CACHED_ATLAS, _CACHED_ATLAS_FAILED
    if _CACHED_ATLAS is None and not _CACHED_ATLAS_FAILED:
        try:
            from nexus_engine.anatomy_atlas import AnatomyAtlas
            _CACHED_ATLAS = AnatomyAtlas()
        except Exception:
            _CACHED_ATLAS_FAILED = True
            return None
    return _CACHED_ATLAS


def _autocategorize_unmapped_atlas_organs() -> None:
    """Apply pattern-based defaults for atlas entities not in the explicit
    mapping. Vessels (anything ending in _a, _v, vein-related names) and
    nerves are categorized as such; everything else stays unmapped."""
    atlas = _get_cached_atlas()
    if atlas is None:
        return
    for organ_id, organ in atlas.organs.items():
        if organ_id in _ATLAS_ORGAN_KIND:
            continue  # already categorized
        # Pattern: arterial endings
        if organ_id.endswith("_a") or organ_id.endswith("_aa") or \
           organ_id in {"aorta", "aortic_arch", "abdominal_aorta",
                        "thoracic_aorta", "celiac", "celiac_trunk"}:
            _ATLAS_ORGAN_KIND[organ_id] = "vessel"
            continue
        # Pattern: venous endings
        if organ_id.endswith("_v") or organ_id.endswith("_vein") or \
           organ_id in {"svc", "ivc", "portal_v", "azygos_v",
                        "coronary_sinus", "cavernous_sinus"}:
            _ATLAS_ORGAN_KIND[organ_id] = "vessel"
            continue
        # Pattern: lymphatic
        if organ_id.endswith("_ln") or "lymphatic" in organ_id or \
           organ_id in {"thoracic_duct", "cisterna_chyli"}:
            _ATLAS_ORGAN_KIND[organ_id] = "lymph_node" if "_ln" in organ_id else "duct"
            continue
        # Pattern: nerves
        if organ_id.endswith("_nerve") or organ_id.endswith("_plexus") or \
           organ_id in {"vagus", "phrenic", "sciatic"}:
            _ATLAS_ORGAN_KIND[organ_id] = "nerve"
            continue
        # By default, system-based heuristic
        if organ.system == "cardiovascular":
            _ATLAS_ORGAN_KIND[organ_id] = "vessel"
        elif organ.system == "neurologic":
            _ATLAS_ORGAN_KIND[organ_id] = "nerve"
        elif organ.system == "lymphatic":
            _ATLAS_ORGAN_KIND[organ_id] = "lymph_node"
        else:
            _ATLAS_ORGAN_KIND[organ_id] = "unknown"


def _load_atlas_coverage_file() -> None:
    """Load atlas_coverage.json (if present) into module-level maps."""
    global _ATLAS_TO_STATE_ORGAN, _ATLAS_ORGAN_KIND
    candidates = [
        "medical_knowledge/state_models/atlas_coverage.json",
        "../medical_knowledge/state_models/atlas_coverage.json",
        os.path.join(os.path.dirname(__file__), "..",
                     "medical_knowledge", "state_models",
                     "atlas_coverage.json"),
    ]
    for p in candidates:
        if not os.path.exists(p):
            continue
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            mapping = data.get("atlas_to_state_organ", {})
            for atlas_id, entry in mapping.items():
                if atlas_id.startswith("_"):
                    continue
                if not isinstance(entry, dict):
                    continue
                state_organ = entry.get("state_organ")
                kind = entry.get("kind")
                if state_organ:
                    _ATLAS_TO_STATE_ORGAN[atlas_id] = state_organ
                if kind:
                    _ATLAS_ORGAN_KIND[atlas_id] = kind
            return
        except Exception as e:
            print(f"[state_model] WARNING: failed to load atlas_coverage.json: {e}")
            return


# Variable aliases: same physiological concept, different organ-specific names.
# When atlas semantic says "adjacent transmits inflammation", we accept any of
# these inflammation-type variables in the receiving organ.
_VARIABLE_ALIASES = {
    "inflammation": [
        "inflammation",
        "alveolar_inflammation",
        "airway_inflammation",
        "parenchymal_inflammation",
        "mucosal_inflammation",
    ],
    "infection_load": [
        "infection_load",
        "luminal_pathogen",
        "luminal_bacteria",
    ],
    "perfusion": [
        "perfusion",
    ],
}


def _resolve_variable_in_organ(organ_name: str, canonical_var: str) -> Optional[str]:
    """Find the specific variable name in this organ matching a canonical concept.
    E.g. canonical 'inflammation' resolves to 'parenchymal_inflammation' in kidney.
    """
    if organ_name not in _REGISTRY.organs:
        return None
    organ = _REGISTRY.organs[organ_name]
    # Direct hit first
    if canonical_var in organ.variables:
        return canonical_var
    # Try aliases
    for alias in _VARIABLE_ALIASES.get(canonical_var, []):
        if alias in organ.variables:
            return alias
    return None


def _organ_for_state_var(atlas_organ: str, state_var: str) -> Optional[str]:
    """Given an atlas organ name and a (possibly canonical) state variable
    name, return the state-model organ that owns this variable (or None).

    Uses _VARIABLE_ALIASES so canonical names like 'inflammation' resolve
    to organ-specific variants like 'parenchymal_inflammation' (kidney) or
    'alveolar_inflammation' (lung)."""
    state_organ = _ATLAS_TO_STATE_ORGAN.get(atlas_organ)
    if not state_organ:
        return None
    if state_organ not in _REGISTRY.organs:
        return None
    # Try direct match first
    if state_var in _REGISTRY.organs[state_organ].variables:
        return state_organ
    # Try aliases
    for alias in _VARIABLE_ALIASES.get(state_var, []):
        if alias in _REGISTRY.organs[state_organ].variables:
            return state_organ
    return None


# ════════════════════════════════════════════════════════════════════
# Forward pipeline — disease → state → symptoms (with propagation)
# ════════════════════════════════════════════════════════════════════

_SIMULATE_DISEASE_CACHE: Dict[str, Optional[Dict]] = {}


def simulate_disease(disease_name: str,
                     initial_state: Optional[BodyState] = None,
                     max_propagation_rounds: int = 3) -> Optional[Dict]:
    """Forward simulation with propagation.

    Pipeline:
      1. Initialize body state (inverse variables to 1.0)
      2. Apply disease perturbations
      3. Run propagation rounds (max_propagation_rounds iterations):
         - For each organ's intra-organ + outgoing rules
         - For each global cross-organ rule
         Stops early if no rule fires in a round (steady state reached)
      4. Derive symptoms from all rules across all organs

    Results are cached: simulate_disease(D) is deterministic when initial_state
    is None and max_propagation_rounds is default. Cache keyed on disease name.

    Returns None if disease has no state model.
    """
    # Cache lookup — only when using default args (no custom initial_state)
    cache_key = None
    if initial_state is None and max_propagation_rounds == 3:
        cache_key = disease_name
        if cache_key in _SIMULATE_DISEASE_CACHE:
            return _SIMULATE_DISEASE_CACHE[cache_key]

    _REGISTRY.load_all()
    model = _REGISTRY.get_disease(disease_name)
    if not model:
        if cache_key is not None:
            _SIMULATE_DISEASE_CACHE[cache_key] = None
        return None
        return None
    organ_model = _REGISTRY.get_organ_for_disease(disease_name)
    if not organ_model:
        return None

    state = initial_state.copy() if initial_state else BodyState()

    # Initialize variables to their healthy default:
    #   - if `default` is explicitly set in JSON, use that (e.g. 0.5 for
    #     bidirectional vars like cortisol_level, transit_time, t3_t4_level)
    #   - else if `inverse=true`, default to 1.0 (healthy = high)
    #   - else 0.0 (healthy = low)
    for organ_name, organ_m in _REGISTRY.organs.items():
        for var_name, var in organ_m.variables.items():
            if var_name in state.organs.get(organ_name, {}):
                continue
            if var.default is not None:
                state.set(organ_name, var_name, float(var.default))
            elif var.inverse:
                state.set(organ_name, var_name, 1.0)
            # else leave at 0.0 (default of BodyState.get)

    # Apply disease perturbations (the seed)
    for p in model.perturbations:
        state.apply_delta(p.organ, p.variable, p.delta)

    # Propagation rounds — apply rules until steady state or max rounds.
    # Each unique rule fires AT MOST ONCE per simulation. Otherwise the
    # same trigger keeps firing across rounds as long as condition holds.
    propagation_trace = []
    fired_rule_ids = set()
    fired_atlas_ids = set()

    # Module-level atlas (cached across all simulate_disease calls)
    _atlas = _get_cached_atlas()
    def _get_atlas():
        return _atlas

    for round_num in range(max_propagation_rounds):
        fired_this_round = []

        # ── Phase 1: Hand-written rules (intra-organ + cross-organ explicit) ──
        all_rules: List[PropagationRule] = []
        for o_model in _REGISTRY.organs.values():
            all_rules.extend(o_model.propagation_rules)
        all_rules.extend(_REGISTRY.global_propagation_rules)

        for prule in all_rules:
            rule_id = (prule.trigger_organ, prule.trigger_var,
                       prule.effect_organ, prule.effect_var)
            if rule_id in fired_rule_ids:
                continue
            if not prule.applies(state):
                continue
            before = state.get(prule.effect_organ, prule.effect_var, 0.0)
            prule.fire(state)
            after = state.get(prule.effect_organ, prule.effect_var, 0.0)
            if abs(after - before) > 1e-6:
                fired_rule_ids.add(rule_id)
                fired_this_round.append({
                    "round":       round_num + 1,
                    "kind":        "explicit",
                    "name":        prule.name or f"{prule.trigger_organ}.{prule.trigger_var}→{prule.effect_organ}.{prule.effect_var}",
                    "trigger":     f"{prule.trigger_organ}.{prule.trigger_var} {prule.trigger_op} {prule.trigger_threshold}",
                    "effect":      f"{prule.effect_organ}.{prule.effect_var} {'+' if prule.effect_delta>=0 else ''}{prule.effect_delta:.2f}",
                    "before":      round(before, 2),
                    "after":       round(after, 2),
                    "via":         prule.via,
                    "rationale":   prule.rationale,
                })

        # ── Phase 2: Atlas-derived auto propagation (D.2) ──
        # For each atlas connection, if its connection_type has a semantic
        # (atlas_connection_semantics), check whether source organ's transmitted
        # state has crossed the trigger threshold. If so, propagate to target.
        #
        # Semantics:
        # - For NORMAL variables (high=bad like inflammation, infection_load):
        #     fire when src_val >= threshold. delta = (src_val - threshold) * decay.
        # - For INVERSE variables (high=good like perfusion, gas_exchange):
        #     fire when src_val <= threshold (i.e. dropped to bad level).
        #     delta = -(threshold - src_val) * decay (reduce target's good var).
        atlas = _get_atlas()
        if atlas and _REGISTRY.atlas_semantics:
            for sem in _REGISTRY.atlas_semantics.values():
                if not sem.transmits_state:
                    continue
                for conn in atlas.connections:
                    if conn.conn_type != sem.connection_type:
                        continue
                    src = conn.source
                    tgt = conn.target
                    src_organ = _organ_for_state_var(src, sem.transmits_state)
                    tgt_organ = _organ_for_state_var(tgt, sem.transmits_state)
                    if not src_organ or not tgt_organ:
                        continue
                    if src_organ == tgt_organ:
                        continue  # don't propagate organ to itself

                    # Resolve canonical 'inflammation' → 'parenchymal_inflammation'
                    # for THIS specific organ. May be different in src vs tgt.
                    src_var = _resolve_variable_in_organ(src_organ, sem.transmits_state)
                    tgt_var = _resolve_variable_in_organ(tgt_organ, sem.transmits_state)
                    if not src_var or not tgt_var:
                        continue

                    src_val = state.get(src_organ, src_var, 0.0)

                    # Determine whether source variable is inverse
                    is_inverse = False
                    src_org_model = _REGISTRY.organs.get(src_organ)
                    if src_org_model and src_var in src_org_model.variables:
                        is_inverse = src_org_model.variables[src_var].inverse

                    if is_inverse:
                        if src_val > sem.threshold:
                            continue
                        delta_magnitude = (sem.threshold - src_val) * sem.decay
                        delta = -delta_magnitude if sem.polarity == "positive" else delta_magnitude
                    else:
                        if src_val < sem.threshold:
                            continue
                        delta_magnitude = (src_val - sem.threshold) * sem.decay
                        delta = delta_magnitude if sem.polarity == "positive" else -delta_magnitude

                    if abs(delta) < 1e-6:
                        continue

                    atlas_id = (sem.connection_type, src, tgt, sem.transmits_state)
                    if atlas_id in fired_atlas_ids:
                        continue
                    before = state.get(tgt_organ, tgt_var, 0.0)
                    state.apply_delta(tgt_organ, tgt_var, delta)
                    after = state.get(tgt_organ, tgt_var, 0.0)
                    if abs(after - before) > 1e-6:
                        fired_atlas_ids.add(atlas_id)
                        fired_this_round.append({
                            "round":     round_num + 1,
                            "kind":      "atlas_derived",
                            "name":      f"atlas[{sem.connection_type}]: {src}→{tgt}",
                            "trigger":   f"{src_organ}.{src_var} {'<=' if is_inverse else '>='} {sem.threshold} (val={src_val:.2f})",
                            "effect":    f"{tgt_organ}.{tgt_var} {'+' if delta>=0 else ''}{delta:.2f}",
                            "before":    round(before, 2),
                            "after":     round(after, 2),
                            "via":       sem.connection_type,
                            "rationale": sem.rationale,
                        })

        propagation_trace.extend(fired_this_round)
        if not fired_this_round:
            break  # steady state — stop early

    # ── D.4: Referred-pain pathway evaluation ──
    # For each loaded referred_pain_pathway, check if its trigger state is
    # above threshold in the body state. If so, derive a "referred pain at <region>"
    # symptom alongside the explicit derivation rules.
    referred_pain_emitted = []
    for pathway in _REGISTRY.referred_pain_pathways:
        organ = pathway.get("visceral_organ")
        var_name = pathway.get("trigger_state")
        threshold = pathway.get("trigger_threshold", 0.5)
        if not organ or not var_name:
            continue
        # Resolve canonical var to organ-specific (e.g., 'inflammation' → 'parenchymal_inflammation')
        resolved_var = _resolve_variable_in_organ(organ, var_name)
        if resolved_var is None:
            # Maybe organ is not in state model (e.g. 'gallbladder' not yet modeled)
            continue
        val = state.get(organ, resolved_var, 0.0)
        # Check direction (inverse-aware)
        organ_model = _REGISTRY.organs.get(organ)
        is_inverse = False
        if organ_model and resolved_var in organ_model.variables:
            is_inverse = organ_model.variables[resolved_var].inverse
        triggered = (val <= threshold) if is_inverse else (val >= threshold)
        if not triggered:
            continue
        for region in pathway.get("somatic_referral", []):
            referred_pain_emitted.append({
                "symptom":    f"referred pain to {region}",
                "rationale":  pathway.get("rationale", ""),
                "confidence": 0.9,
                "from_organ": organ,
                "via":        "neural_referred_pain",
                "pathway":    pathway.get("name", ""),
                "spinal_segments": pathway.get("spinal_segments", []),
            })

    # Derive symptoms from ALL organ rules (state may now span multiple organs)
    derivations = []
    for o_model in _REGISTRY.organs.values():
        for rule in o_model.rules:
            if rule.applies(state, o_model.organ):
                derivations.append({
                    "symptom":    rule.symptom,
                    "rationale":  rule.rationale,
                    "confidence": rule.confidence,
                    "from_organ": o_model.organ,
                })

    # Combine: explicit rules + referred pain
    derivations.extend(referred_pain_emitted)

    # Deduplicate symptoms (multiple organs/pathways might produce same symptom)
    seen_syms = set()
    deduped = []
    for d in derivations:
        if d["symptom"] not in seen_syms:
            deduped.append(d)
            seen_syms.add(d["symptom"])

    result = {
        "disease":           model.name,
        "organ":             model.organ,
        "review_status":     model.review_status,
        "perturbations":     [
            {"organ": p.organ, "variable": p.variable,
             "delta": p.delta, "cause": p.cause}
            for p in model.perturbations
        ],
        "propagation_trace": propagation_trace,
        "final_state":       state,
        "active_state":      state.all_active(0.1),
        "derived_symptoms":  [d["symptom"] for d in deduped],
        "derivations":       deduped,
    }
    if cache_key is not None:
        _SIMULATE_DISEASE_CACHE[cache_key] = result
    return result


# ════════════════════════════════════════════════════════════════════
# 6. Reverse pipeline
# ════════════════════════════════════════════════════════════════════

def reverse_match(user_symptoms: List[str]) -> Dict:
    _REGISTRY.load_all()
    user_norm = {s.lower().replace("_", " ").strip()
                 for s in (user_symptoms or [])}

    scores = []
    for d_name in _REGISTRY.all_diseases():
        sim = simulate_disease(d_name)
        if not sim:
            continue
        sim_norm = {s.lower().replace("_", " ").strip()
                    for s in sim["derived_symptoms"]}
        overlap = user_norm & sim_norm
        missing = user_norm - sim_norm
        extra = sim_norm - user_norm
        score = len(overlap) / max(len(user_norm), 1)
        scores.append({
            "disease":  d_name,
            "score":    round(score, 3),
            "matched":  sorted(overlap),
            "missing":  sorted(missing),
            "extra":    sorted(extra),
        })
    scores.sort(key=lambda x: x["score"], reverse=True)
    return {
        "user_symptoms":  sorted(user_norm),
        "disease_scores": scores,
    }


# ════════════════════════════════════════════════════════════════════
# 7. Public helpers
# ════════════════════════════════════════════════════════════════════

def is_state_modeled(disease_name: str) -> bool:
    return _REGISTRY.has_disease(disease_name)


def simulate(disease_name: str) -> Optional[Dict]:
    return simulate_disease(disease_name)


def list_state_modeled_diseases() -> List[str]:
    return _REGISTRY.all_diseases()


# ════════════════════════════════════════════════════════════════════
# D.4 / D.5 — Neural + Vascular framework query helpers
# ════════════════════════════════════════════════════════════════════

def get_referred_pain_pathways() -> List[dict]:
    """Return all loaded referred-pain pathways."""
    _REGISTRY.load_all()
    return list(_REGISTRY.referred_pain_pathways)


def get_cranial_nerves() -> dict:
    """Return the 12 cranial nerves dict from neural framework."""
    _REGISTRY.load_all()
    return _REGISTRY.neural_framework.get("cranial_nerves", {})


def get_arterial_territories() -> List[dict]:
    """Return all loaded arterial territories."""
    _REGISTRY.load_all()
    return list(_REGISTRY.arterial_territories)


def get_watershed_zones() -> List[dict]:
    """Return all watershed zones (vulnerable to hypoperfusion)."""
    _REGISTRY.load_all()
    return list(_REGISTRY.watershed_zones)


def get_embolic_sources() -> List[dict]:
    """Return embolic source mapping."""
    _REGISTRY.load_all()
    return list(_REGISTRY.embolic_sources)


def find_territory_for_artery(artery_name: str) -> Optional[dict]:
    """Look up which organs/regions an artery supplies."""
    _REGISTRY.load_all()
    a_norm = artery_name.lower().replace("_", "").replace("-", "")
    for t in _REGISTRY.arterial_territories:
        for key in ("artery", "atlas_artery"):
            v = (t.get(key) or "").lower().replace("_", "").replace("-", "")
            if v and (v in a_norm or a_norm in v):
                return t
    return None


# ════════════════════════════════════════════════════════════════════
# E.1 — Structural framework query helpers (comprehensive connectivity)
# ════════════════════════════════════════════════════════════════════

def get_organ_systems() -> dict:
    """Return all 11 organ systems with their core organs + functions."""
    _REGISTRY.load_all()
    return dict(_REGISTRY.organ_systems)


def get_connection_type_semantics() -> dict:
    """Return semantic documentation of all 14 atlas connection types."""
    _REGISTRY.load_all()
    return dict(_REGISTRY.connection_type_semantics)


def get_axes() -> dict:
    """Return all regulatory axes (HPA, HPT, HPG, RAAS, calcium, etc.)."""
    _REGISTRY.load_all()
    return dict(_REGISTRY.axes_and_loops)


def get_regional_anatomy() -> dict:
    """Return body regions (head/neck, thorax, abdomen, pelvis, limbs, back)."""
    _REGISTRY.load_all()
    return dict(_REGISTRY.regional_anatomy)


def get_embryologic_origins() -> dict:
    """Return foregut/midgut/hindgut derivatives + referred pain dermatomes."""
    _REGISTRY.load_all()
    return dict(_REGISTRY.embryologic_origins)


def get_barrier_systems() -> dict:
    """Return BBB, intestinal, alveolar-capillary, and other barriers."""
    _REGISTRY.load_all()
    return dict(_REGISTRY.barrier_systems)


def find_organ_system(organ_name: str) -> Optional[str]:
    """Given an organ name (e.g. 'kidney'), return which organ system it belongs to."""
    _REGISTRY.load_all()
    organ_norm = organ_name.lower().strip()
    for sys_name, sys_data in _REGISTRY.organ_systems.items():
        if not isinstance(sys_data, dict):
            continue
        for key in ("core_organs", "accessory_organs", "vasculature",
                    "male_organs", "female_organs"):
            organs = sys_data.get(key, [])
            for o in organs:
                if o.lower() == organ_norm:
                    return sys_name
    return None


def find_region_for_organ(organ_name: str) -> Optional[str]:
    """Given an organ name, return which body region contains it."""
    _REGISTRY.load_all()
    organ_norm = organ_name.lower().strip()
    for region_name, region_data in _REGISTRY.regional_anatomy.items():
        if not isinstance(region_data, dict):
            continue
        for o in region_data.get("organs", []):
            if o.lower() == organ_norm:
                return region_name
    return None


def find_axis_containing(organ_or_hormone: str) -> List[str]:
    """Find all regulatory axes containing this organ or hormone."""
    _REGISTRY.load_all()
    query = organ_or_hormone.lower().strip()
    matches = []
    for axis_name, axis_data in _REGISTRY.axes_and_loops.items():
        if not isinstance(axis_data, dict):
            continue
        components = axis_data.get("components", [])
        comp_text = " ".join(components).lower()
        if query in comp_text:
            matches.append(axis_name)
    return matches


# ════════════════════════════════════════════════════════════════════
# E.2 — Atlas coverage helpers (bridge atlas 259 → state model 16)
# ════════════════════════════════════════════════════════════════════

def get_atlas_coverage_map() -> Dict[str, str]:
    """Return the complete atlas → state model mapping (atlas_id → state_organ).
    Only includes entities that have a state model."""
    _REGISTRY.load_all()
    return dict(_ATLAS_TO_STATE_ORGAN)


def get_atlas_organ_kinds() -> Dict[str, str]:
    """Return the kind classification for all atlas organs (organ / vessel /
    nerve / duct / lymph_node / barrier / region / structural / unknown)."""
    _REGISTRY.load_all()
    return dict(_ATLAS_ORGAN_KIND)


def get_state_organ_to_atlas_organs() -> Dict[str, List[str]]:
    """Reverse mapping: state organ → list of atlas entities it covers.
    Tells you which atlas entities each state model represents."""
    _REGISTRY.load_all()
    result: Dict[str, List[str]] = {}
    for atlas_id, state_organ in _ATLAS_TO_STATE_ORGAN.items():
        if not state_organ:
            continue
        result.setdefault(state_organ, []).append(atlas_id)
    return {k: sorted(v) for k, v in result.items()}


def get_atlas_kind(atlas_organ: str) -> Optional[str]:
    """Look up the kind classification of a specific atlas entity."""
    _REGISTRY.load_all()
    return _ATLAS_ORGAN_KIND.get(atlas_organ)


def validate_atlas_coverage() -> Dict:
    """Audit the atlas coverage: how many of the 259 atlas entities are
    mapped to state models, by kind. Useful for quality checks."""
    _REGISTRY.load_all()
    try:
        from nexus_engine.anatomy_atlas import AnatomyAtlas
        atlas = AnatomyAtlas()
    except Exception:
        return {"error": "atlas not loadable"}
    total = len(atlas.organs)
    mapped = sum(1 for o in atlas.organs if o in _ATLAS_TO_STATE_ORGAN
                 and _ATLAS_TO_STATE_ORGAN[o])
    unmapped = total - mapped

    from collections import Counter
    kind_counts = Counter(_ATLAS_ORGAN_KIND.get(o, "uncategorized")
                          for o in atlas.organs)

    # Which atlas organs are completely unmapped AND unclassified
    truly_unknown = [
        o for o in atlas.organs
        if o not in _ATLAS_TO_STATE_ORGAN and o not in _ATLAS_ORGAN_KIND
    ]

    return {
        "total_atlas_organs": total,
        "mapped_to_state_organ": mapped,
        "unmapped_but_categorized": unmapped - len(truly_unknown),
        "truly_unknown_count": len(truly_unknown),
        "truly_unknown_samples": truly_unknown[:10],
        "by_kind": dict(kind_counts),
        "coverage_pct": round(100 * mapped / max(total, 1), 1),
    }


# ════════════════════════════════════════════════════════════════════
# F.1 — Lab integration helpers
# ════════════════════════════════════════════════════════════════════

def get_lab_to_state_mapping() -> dict:
    """Return lab → state variable mapping."""
    _REGISTRY.load_all()
    return dict(_REGISTRY.lab_to_state_mapping)


def get_lab_implications(lab_name: str) -> List[dict]:
    """For a given lab name, return list of state variables it informs."""
    _REGISTRY.load_all()
    return _REGISTRY.lab_to_state_mapping.get(lab_name, [])


def get_labs_for_state(organ: str, state_var: str) -> List[str]:
    """Find which labs predict this state variable."""
    _REGISTRY.load_all()
    matches = []
    for lab_name, mappings in _REGISTRY.lab_to_state_mapping.items():
        for m in mappings:
            if not isinstance(m, dict):
                continue
            if m.get("organ") == organ and m.get("state") == state_var:
                matches.append(lab_name)
    return matches


# ════════════════════════════════════════════════════════════════════
# F.2 — Vital signs helpers
# ════════════════════════════════════════════════════════════════════

def get_vital_signs() -> dict:
    _REGISTRY.load_all()
    return dict(_REGISTRY.vital_signs)


def get_critical_vital_combinations() -> dict:
    _REGISTRY.load_all()
    return dict(_REGISTRY.critical_vital_combinations)


# ════════════════════════════════════════════════════════════════════
# F.3 — Pain helpers
# ════════════════════════════════════════════════════════════════════

def get_pain_qualities() -> dict:
    _REGISTRY.load_all()
    return dict(_REGISTRY.pain_qualities)


def get_pain_radiation_patterns() -> dict:
    _REGISTRY.load_all()
    return dict(_REGISTRY.pain_radiation_patterns)


# ════════════════════════════════════════════════════════════════════
# F.4 — Time dynamics helpers
# ════════════════════════════════════════════════════════════════════

def get_time_scales() -> dict:
    _REGISTRY.load_all()
    return dict(_REGISTRY.time_scales)


def get_disease_timecourse(disease_name: str) -> Optional[dict]:
    _REGISTRY.load_all()
    return _REGISTRY.disease_timecourses.get(disease_name)


def get_red_flag_temporal_patterns() -> dict:
    _REGISTRY.load_all()
    return dict(_REGISTRY.red_flag_temporal_patterns)


# ════════════════════════════════════════════════════════════════════
# F.5 / F.6 — Anatomy extensions
# ════════════════════════════════════════════════════════════════════

def get_bones_framework() -> dict:
    _REGISTRY.load_all()
    return dict(_REGISTRY.bones_framework)


def get_exocrine_glands() -> dict:
    _REGISTRY.load_all()
    return dict(_REGISTRY.exocrine_glands)


# ════════════════════════════════════════════════════════════════════
# F.7-F.19 — Generic medical framework accessors
# ════════════════════════════════════════════════════════════════════

def get_pharmacology() -> dict:
    """Drug classes, mechanisms, state effects, side effects, interactions."""
    _REGISTRY.load_all()
    return dict(_REGISTRY.pharmacology_framework)


def get_microbiology() -> dict:
    """Pathogens by class, gram stain, antibiotic susceptibility."""
    _REGISTRY.load_all()
    return dict(_REGISTRY.microbiology_framework)


def get_imaging() -> dict:
    """Imaging modalities + key findings + ECG patterns."""
    _REGISTRY.load_all()
    return dict(_REGISTRY.imaging_framework)


def get_immunology() -> dict:
    """Innate/adaptive immunity, hypersensitivity, autoimmune, immunodef."""
    _REGISTRY.load_all()
    return dict(_REGISTRY.immunology_framework)


def get_oncology() -> dict:
    """Tumor types, TNM staging, metastasis pathways, common cancers."""
    _REGISTRY.load_all()
    return dict(_REGISTRY.oncology_framework)


def get_risk_factors() -> dict:
    """Modifiable/non-modifiable risk factors per disease + risk scores."""
    _REGISTRY.load_all()
    return dict(_REGISTRY.risk_factors_framework)


def get_epidemiology() -> dict:
    """Population epidemiology — age, sex, geography, race/ethnicity."""
    _REGISTRY.load_all()
    return dict(_REGISTRY.epidemiology_framework)


def get_specialty(name: str) -> dict:
    """Access specialty frameworks. name in:
    'psychiatry', 'pediatrics', 'obstetrics', 'dermatology',
    'ophthalmology', 'surgery'."""
    _REGISTRY.load_all()
    attr = f"specialty_{name.lower()}"
    return dict(getattr(_REGISTRY, attr, {}))


def find_drug_class_for(drug_name: str) -> Optional[str]:
    """Look up which drug class a specific drug belongs to."""
    _REGISTRY.load_all()
    classes = _REGISTRY.pharmacology_framework.get("drug_classes", {})
    drug_norm = drug_name.lower()
    for class_name, class_data in classes.items():
        if not isinstance(class_data, dict):
            continue
        examples = class_data.get("examples", [])
        for ex in examples:
            if drug_norm in ex.lower() or ex.lower() in drug_norm:
                return class_name
    return None


def find_pathogen_info(pathogen_name: str) -> Optional[dict]:
    """Look up a pathogen across all microbiology categories."""
    _REGISTRY.load_all()
    micro = _REGISTRY.microbiology_framework
    name_norm = pathogen_name.lower()
    for category, organisms in micro.items():
        if category.startswith("_") or not isinstance(organisms, dict):
            continue
        for org_name, org_data in organisms.items():
            if not isinstance(org_data, dict):
                continue
            if name_norm in org_name.lower():
                return {"category": category, "name": org_name, **org_data}
    return None


# ════════════════════════════════════════════════════════════════════
# LINKAGE INTEGRITY AUDIT — verify every framework cross-references
# the state model + identify gaps in connectivity
# ════════════════════════════════════════════════════════════════════

def audit_framework_linkage() -> dict:
    """Comprehensive audit: are all frameworks loaded? Do they cross-reference
    state models correctly? Returns a detailed report."""
    _REGISTRY.load_all()

    report = {
        "frameworks_loaded": {},
        "linkage": {},
        "warnings": [],
        "summary": {},
    }

    # Check each framework is loaded with content
    framework_checks = [
        ("organs (state models)",      len(_REGISTRY.organs),                     16),
        ("diseases",                    len(_REGISTRY._disease_index),             40),
        ("global propagation rules",    len(_REGISTRY.global_propagation_rules),   5),
        ("atlas semantics",             len(_REGISTRY.atlas_semantics),            10),
        ("referred-pain pathways",      len(_REGISTRY.referred_pain_pathways),     10),
        ("arterial territories",        len(_REGISTRY.arterial_territories),       25),
        ("watershed zones",             len(_REGISTRY.watershed_zones),            3),
        ("organ systems",               len(_REGISTRY.organ_systems),              10),
        ("connection semantics docs",   len(_REGISTRY.connection_type_semantics),  10),
        ("axes",                        len(_REGISTRY.axes_and_loops),             10),
        ("regions",                     len(_REGISTRY.regional_anatomy),           5),
        ("barriers",                    len(_REGISTRY.barrier_systems),            5),
        ("lab→state mappings",          len(_REGISTRY.lab_to_state_mapping),       40),
        ("vital signs",                 len(_REGISTRY.vital_signs),                6),
        ("pain qualities",              len(_REGISTRY.pain_qualities),             8),
        ("disease timecourses",         len(_REGISTRY.disease_timecourses),        10),
        ("bones",                       len(_REGISTRY.bones_framework),            20),
        ("exocrine glands",             len(_REGISTRY.exocrine_glands),            8),
        ("pharmacology drug classes",   len(_REGISTRY.pharmacology_framework.get("drug_classes", {})), 15),
        ("microbiology categories",     sum(1 for k in _REGISTRY.microbiology_framework if not k.startswith("_")), 5),
        ("imaging modalities",          len(_REGISTRY.imaging_framework.get("modalities", {})), 5),
        ("immunology categories",       sum(1 for k in _REGISTRY.immunology_framework if not k.startswith("_")), 4),
        ("oncology categories",         sum(1 for k in _REGISTRY.oncology_framework if not k.startswith("_")), 4),
        ("risk factor categories",      sum(1 for k in _REGISTRY.risk_factors_framework if not k.startswith("_")), 3),
        ("epidemiology categories",     sum(1 for k in _REGISTRY.epidemiology_framework if not k.startswith("_")), 4),
        ("specialty psychiatry",        bool(_REGISTRY.specialty_psychiatry), 1),
        ("specialty pediatrics",        bool(_REGISTRY.specialty_pediatrics), 1),
        ("specialty obstetrics",        bool(_REGISTRY.specialty_obstetrics), 1),
        ("specialty dermatology",       bool(_REGISTRY.specialty_dermatology), 1),
        ("specialty ophthalmology",     bool(_REGISTRY.specialty_ophthalmology), 1),
        ("specialty surgery",           bool(_REGISTRY.specialty_surgery), 1),
    ]
    for name, actual, expected_min in framework_checks:
        ok = actual >= expected_min
        report["frameworks_loaded"][name] = {
            "count": actual, "expected_min": expected_min, "ok": ok
        }
        if not ok:
            report["warnings"].append(
                f"{name}: only {actual} loaded (expected ≥{expected_min})")

    # Linkage check 1: every state organ has ≥1 lab that maps to it
    state_organs_with_labs = set()
    for lab_name, mappings in _REGISTRY.lab_to_state_mapping.items():
        for m in mappings:
            if isinstance(m, dict) and m.get("organ"):
                state_organs_with_labs.add(m["organ"])
    organs_without_labs = (set(_REGISTRY.organs.keys()) - state_organs_with_labs)
    report["linkage"]["organs_with_lab_mapping"] = {
        "covered": sorted(state_organs_with_labs & set(_REGISTRY.organs.keys())),
        "missing": sorted(organs_without_labs),
    }

    # Linkage check 2: every state organ is in some organ system
    if _REGISTRY.organ_systems:
        organs_in_systems = set()
        for sys_data in _REGISTRY.organ_systems.values():
            if not isinstance(sys_data, dict):
                continue
            for key in ("core_organs", "accessory_organs"):
                for o in sys_data.get(key, []):
                    organs_in_systems.add(o.lower())
        # match against state organs (loose)
        organs_unmapped_to_system = []
        for organ in _REGISTRY.organs:
            if not any(organ.lower() in s.lower() or s.lower() in organ.lower()
                       for s in organs_in_systems):
                organs_unmapped_to_system.append(organ)
        report["linkage"]["organs_with_system"] = {
            "missing": organs_unmapped_to_system,
        }

    # Linkage check 3: every state organ is in some region
    organs_in_regions = set()
    for r_data in _REGISTRY.regional_anatomy.values():
        if not isinstance(r_data, dict):
            continue
        for o in r_data.get("organs", []):
            organs_in_regions.add(o.lower())
    organs_unmapped_to_region = []
    for organ in _REGISTRY.organs:
        if not any(organ.lower() in s or s in organ.lower() for s in organs_in_regions):
            organs_unmapped_to_region.append(organ)
    report["linkage"]["organs_with_region"] = {
        "missing": organs_unmapped_to_region,
    }

    # Linkage check 4: pharmacology drugs reference state organs
    pharm_state_effects_organs = set()
    classes = _REGISTRY.pharmacology_framework.get("drug_classes", {})
    for class_data in classes.values():
        if not isinstance(class_data, dict):
            continue
        for effect in class_data.get("state_effects", []):
            if isinstance(effect, dict) and effect.get("organ"):
                pharm_state_effects_organs.add(effect["organ"])
    report["linkage"]["pharmacology_references_organs"] = sorted(
        pharm_state_effects_organs & set(_REGISTRY.organs.keys())
    )

    # Linkage check 5: atlas coverage
    atlas_audit = validate_atlas_coverage()
    report["linkage"]["atlas_coverage"] = {
        "total_atlas_organs": atlas_audit.get("total_atlas_organs"),
        "mapped_to_state": atlas_audit.get("mapped_to_state_organ"),
        "truly_unknown": atlas_audit.get("truly_unknown_count"),
        "coverage_pct": atlas_audit.get("coverage_pct"),
    }

    # Summary
    report["summary"] = {
        "total_frameworks_loaded": sum(1 for f in report["frameworks_loaded"].values() if f["ok"]),
        "total_frameworks_expected": len(framework_checks),
        "total_warnings": len(report["warnings"]),
        "state_organs": len(_REGISTRY.organs),
        "diseases_modeled": len(_REGISTRY._disease_index),
        "atlas_coverage_pct": atlas_audit.get("coverage_pct", 0),
    }
    return report


# ════════════════════════════════════════════════════════════════════
# G.3 — Symptom alias expansion (vocabulary bridging)
# ════════════════════════════════════════════════════════════════════

def get_symptom_aliases() -> dict:
    """Return canonical → [aliases] mapping."""
    _REGISTRY.load_all()
    return dict(_REGISTRY.symptom_aliases)


def canonicalize_symptom(s: str) -> str:
    """Map a free-text symptom to its canonical form via alias lookup.
    If no alias known, returns lowercased original."""
    _REGISTRY.load_all()
    norm = s.lower().strip()
    return _REGISTRY._symptom_alias_lookup.get(norm, norm)


def expand_symptom_aliases(symptoms) -> set:
    """Given a list/set of symptoms, return the set including all aliases.
    Used to bridge state-model vocab (e.g. 'focal weakness') to user vocab
    (e.g. 'unilateral hemiparesis weakness one side')."""
    _REGISTRY.load_all()
    expanded = set()
    for s in symptoms:
        norm = s.lower().strip()
        expanded.add(norm)
        # If this maps to a canonical, add the canonical
        canon = _REGISTRY._symptom_alias_lookup.get(norm)
        if canon and canon != norm:
            expanded.add(canon)
        # Also add all aliases of this canonical (if any)
        if norm in _REGISTRY.symptom_aliases:
            for alias in _REGISTRY.symptom_aliases[norm]:
                expanded.add(alias.lower().strip())
        # If norm IS an alias of some canonical, add all that canonical's aliases
        if canon and canon in _REGISTRY.symptom_aliases:
            for alias in _REGISTRY.symptom_aliases[canon]:
                expanded.add(alias.lower().strip())
    return expanded


def match_symptoms_with_aliases(user_symptoms, candidate_symptoms) -> set:
    """Find overlap between user_symptoms and candidate_symptoms, expanding
    both sides through aliases. Returns the set of canonical matches."""
    user_expanded = expand_symptom_aliases(user_symptoms)
    cand_expanded = expand_symptom_aliases(candidate_symptoms)
    return user_expanded & cand_expanded


# ════════════════════════════════════════════════════════════════════
# G.4.C: STATE EVIDENCE MATCHING
# Reverse-engineer derivation rules to infer what states user symptoms
# probably imply. Then match against disease-predicted states (not just
# disease-predicted symptoms).
#
# This is closer to how clinicians reason:
#   "chest pressure + sweating" → think 'cardiac ischemia + sympathetic'
#   then compare to which diseases produce those states.
# ════════════════════════════════════════════════════════════════════

# Cache for reverse index — built lazily on first call
_SYMPTOM_TO_STATES_INDEX: Optional[dict] = None


def _build_symptom_to_states_index() -> dict:
    """Reverse-engineer the symptom → list-of-(organ, state, op, threshold,
    confidence) index from all organ derivation rules.

    Returns:
        {symptom_lower: [
            {organ, state, op, threshold, confidence, rationale}, ...
        ]}
    """
    _REGISTRY.load_all()
    index: Dict[str, List[Dict]] = {}

    def _extract_state_conditions(condition: dict):
        """Recursively yield (state, op, threshold) from condition trees."""
        if not isinstance(condition, dict):
            return
        if "state" in condition:
            yield (condition["state"],
                   condition.get("op", ">"),
                   condition.get("threshold", 0.5))
        elif "all" in condition:
            for sub in condition["all"]:
                yield from _extract_state_conditions(sub)
        elif "any" in condition:
            for sub in condition["any"]:
                yield from _extract_state_conditions(sub)

    for organ_name, organ_model in _REGISTRY.organs.items():
        for rule in organ_model.rules:
            sym_lower = rule.symptom.lower().strip()
            for state_var, op, threshold in _extract_state_conditions(rule.condition):
                index.setdefault(sym_lower, []).append({
                    "organ": organ_name,
                    "state": state_var,
                    "op": op,
                    "threshold": threshold,
                    "confidence": rule.confidence,
                    "rationale": rule.rationale[:100],
                })
    return index


def get_symptom_to_states_index() -> dict:
    """Get the reverse index (cached)."""
    global _SYMPTOM_TO_STATES_INDEX
    if _SYMPTOM_TO_STATES_INDEX is None:
        _SYMPTOM_TO_STATES_INDEX = _build_symptom_to_states_index()
    return _SYMPTOM_TO_STATES_INDEX


def infer_state_evidence(user_symptoms: list) -> Dict:
    """Given user symptoms, infer what organ states are probably perturbed.

    Returns:
        {
            "state_evidence": {
                "heart.ischemia": {
                    "supporting_symptoms": ["chest pain"],
                    "direction": "high",  # state should be elevated
                    "threshold": 0.5,
                    "weight": 1.0,
                    "specificity": 0.5,  # 1/n_competing_states
                },
                ...
            },
            "unrecognized_symptoms": ["..."],
            "ambiguous_symptoms": ["..."],  # mapped to many states
        }
    """
    index = get_symptom_to_states_index()
    state_evidence = {}
    unrecognized = []
    ambiguous = []

    for raw_sym in user_symptoms:
        sym_lower = raw_sym.lower().strip().replace("_", " ")

        # Look up direct + alias-expanded forms
        candidates = []
        # Direct match
        if sym_lower in index:
            candidates = index[sym_lower]
        else:
            # Try via canonical alias mapping
            canonical = _REGISTRY._symptom_alias_lookup.get(sym_lower)
            if canonical and canonical in index:
                candidates = index[canonical]
            else:
                # Try each alias of this canonical
                for canon, aliases in _REGISTRY.symptom_aliases.items():
                    if sym_lower in [a.lower() for a in aliases]:
                        if canon.lower() in index:
                            candidates = index[canon.lower()]
                            break

        if not candidates:
            unrecognized.append(raw_sym)
            continue

        # Mark as ambiguous if it maps to >3 states (low specificity)
        if len(candidates) > 3:
            ambiguous.append(raw_sym)

        # Compute specificity: rare symptoms (1-state) carry more weight
        specificity = 1.0 / len(candidates)

        for entry in candidates:
            key = f"{entry['organ']}.{entry['state']}"
            # Determine direction: > = "high", < = "low"
            direction = "high" if entry["op"] in (">", ">=") else "low"
            if key not in state_evidence:
                state_evidence[key] = {
                    "supporting_symptoms": [],
                    "direction": direction,
                    "threshold": entry["threshold"],
                    "weight": 0.0,
                    "specificity_total": 0.0,
                    "max_confidence": 0.0,
                    "rationale": entry["rationale"],
                }
            ev = state_evidence[key]
            if raw_sym not in ev["supporting_symptoms"]:
                ev["supporting_symptoms"].append(raw_sym)
            # Weight: confidence × specificity, accumulated across supporting syms
            ev["weight"] += entry["confidence"] * specificity
            ev["specificity_total"] += specificity
            ev["max_confidence"] = max(ev["max_confidence"], entry["confidence"])

    # Normalize weights to 0-1 range
    if state_evidence:
        max_w = max(ev["weight"] for ev in state_evidence.values())
        for ev in state_evidence.values():
            ev["normalized_weight"] = round(ev["weight"] / max_w, 3) if max_w > 0 else 0.0
            ev["weight"] = round(ev["weight"], 3)
            ev["specificity"] = round(1.0 / max(len(ev["supporting_symptoms"]), 1), 3)

    return {
        "state_evidence": state_evidence,
        "unrecognized_symptoms": unrecognized,
        "ambiguous_symptoms": ambiguous,
    }


def state_evidence_match_score(user_symptoms: list, disease_name: str) -> Dict:
    """Score a disease based on STATE OVERLAP, not symptom overlap.

    1. Infer states from user symptoms (via reverse derivation rules).
    2. Get disease-predicted states from simulate_disease().
    3. Compare overlap, weighted by specificity.

    Returns:
        {
            "score": 0.0-1.0,
            "matched_states": [...],  # which states agree
            "user_expected_states": [...],  # what user implied
            "disease_predicted_states": [...],  # what disease predicts
            "weight_matched": float,
            "weight_user_total": float,
        }
    """
    # 1. Infer states from user symptoms
    user_evidence = infer_state_evidence(user_symptoms)
    user_states = user_evidence["state_evidence"]
    if not user_states:
        return {"score": 0.0, "matched_states": [], "user_expected_states": [],
                "disease_predicted_states": [], "reason": "no states inferred"}

    # 2. Get disease's active state from simulation
    sim = simulate_disease(disease_name)
    if not sim:
        return {"score": 0.0, "matched_states": [], "user_expected_states": list(user_states.keys()),
                "disease_predicted_states": [], "reason": "no simulation"}

    # Build disease state dict: {organ.state: value}
    disease_states = {}
    for organ, var, value in sim["active_state"]:
        # Only count states that are perturbed (not default)
        key = f"{organ}.{var}"
        disease_states[key] = value

    # 3. Compare: for each user-expected state, does disease match?
    matched = []
    user_weight_total = sum(ev["weight"] for ev in user_states.values())
    matched_weight = 0.0

    for state_key, user_ev in user_states.items():
        if state_key not in disease_states:
            continue
        disease_val = disease_states[state_key]
        threshold = user_ev["threshold"]
        direction = user_ev["direction"]

        # Does disease's state value satisfy user's implied direction?
        # direction "high" means state > threshold (disease has it elevated)
        # direction "low" means state < threshold (disease has it reduced)
        if direction == "high" and disease_val >= threshold:
            agreement = 1.0
        elif direction == "low" and disease_val <= threshold:
            agreement = 1.0
        else:
            # Partial: how close to threshold?
            agreement = 0.0

        if agreement > 0:
            matched.append({
                "state": state_key,
                "direction": direction,
                "user_threshold": threshold,
                "disease_value": round(disease_val, 2),
                "supporting_symptoms": user_ev["supporting_symptoms"],
                "weight": user_ev["weight"],
            })
            matched_weight += user_ev["weight"] * agreement

    score = matched_weight / user_weight_total if user_weight_total > 0 else 0.0

    return {
        "score": round(score, 3),
        "matched_states": matched,
        "user_expected_states": [{"state": k, "direction": v["direction"],
                                   "weight": v["weight"],
                                   "specificity": v["specificity"]}
                                  for k, v in user_states.items()],
        "disease_predicted_states_count": len(disease_states),
        "weight_matched": round(matched_weight, 3),
        "weight_user_total": round(user_weight_total, 3),
    }


# ════════════════════════════════════════════════════════════════════
# G.4.C3: BAYESIAN STATE INFERENCE
# P(state | symptoms) computed via naive Bayes over derivation rules.
#
# Key improvements over C2 (1/n specificity):
#   - Prior P(state) from how many diseases perturb that state
#   - Likelihood P(symptom | state) from derivation rule confidence
#   - Joint inference: multiple symptoms multiply evidence properly
#   - Disease scoring: P(state assignments | disease) — closer to medical
#     reasoning where doctors think about disease in terms of pathophysiology
#     not symptom checklists.
#
# Honest limitations:
#   - Naive Bayes (state independence) — false but useful simplification
#   - Likelihoods use rule confidence as proxy (not empirically estimated)
#   - Priors from disease registry (not real epidemiology)
# ════════════════════════════════════════════════════════════════════

# Cache: built once per process
_BAYES_STATE_PRIORS: Optional[Dict[str, float]] = None
_BAYES_LIKELIHOODS: Optional[Dict[str, Dict[str, float]]] = None


def _build_bayesian_tables():
    """Build prior P(state perturbed) and likelihood P(symptom | state).

    Priors come from: for each (organ.state, direction), count how many
    diseases in the registry perturb that state in that direction.
    Higher count → higher prior probability of being perturbed.

    Likelihoods come from derivation rule confidence — when a rule says
    "if state X > 0.5 → symptom Y (conf 0.9)", we interpret:
        P(Y | state X is high) = 0.9
        P(Y | state X is not high) = (1 - 0.9) × 0.1 = 0.01 (base rate floor)
    """
    global _BAYES_STATE_PRIORS, _BAYES_LIKELIHOODS
    if _BAYES_STATE_PRIORS is not None:
        return

    _REGISTRY.load_all()

    # ── Step 1: build state priors ──
    # For each (organ.state.direction), count diseases that perturb it
    # Key format: "organ.state.high" or "organ.state.low"
    state_perturbation_counts: Dict[str, int] = {}
    total_diseases = max(len(_REGISTRY._disease_index), 1)

    for disease_name, organ_model in _REGISTRY._disease_index.items():
        organ_name = organ_model.organ
        disease_obj = organ_model.diseases.get(disease_name)
        if disease_obj is None:
            continue
        for pert in disease_obj.perturbations:
            var = pert.variable
            if not var:
                continue
            delta = pert.delta
            # Determine direction based on delta sign + organ's inverse flag
            var_meta = organ_model.variables.get(var)
            is_inverse = var_meta.inverse if var_meta else False
            # For inverse variables (higher=better), a negative delta = state worsens (low direction)
            # For normal variables (higher=worse), a positive delta = state worsens (high direction)
            if is_inverse:
                direction = "low" if delta < 0 else "high"
            else:
                direction = "high" if delta > 0 else "low"
            key = f"{organ_name}.{var}.{direction}"
            state_perturbation_counts[key] = state_perturbation_counts.get(key, 0) + 1

    # Priors with Laplace smoothing: (count + 1) / (total + 2)
    # This keeps even unobserved states with nonzero prior
    _BAYES_STATE_PRIORS = {}
    for key, count in state_perturbation_counts.items():
        _BAYES_STATE_PRIORS[key] = (count + 1) / (total_diseases + 2)

    # ── Step 2: build likelihood table P(symptom | state) ──
    # Use derivation rules: each rule maps state condition → symptom with confidence
    # _BAYES_LIKELIHOODS["organ.state.direction"]["symptom"] = confidence
    _BAYES_LIKELIHOODS = {}

    def _walk_condition(cond):
        """Yield (state_var, direction, threshold) from condition tree."""
        if not isinstance(cond, dict):
            return
        if "state" in cond:
            op = cond.get("op", ">")
            direction = "high" if op in (">", ">=") else "low"
            yield (cond["state"], direction, cond.get("threshold", 0.5))
        elif "all" in cond:
            for sub in cond["all"]:
                yield from _walk_condition(sub)
        elif "any" in cond:
            for sub in cond["any"]:
                yield from _walk_condition(sub)

    for organ_name, organ_model in _REGISTRY.organs.items():
        for rule in organ_model.rules:
            sym_lower = rule.symptom.lower().strip()
            conf = rule.confidence
            for state_var, direction, _thr in _walk_condition(rule.condition):
                key = f"{organ_name}.{state_var}.{direction}"
                # Ensure state in priors (add default if not seen in any disease)
                if key not in _BAYES_STATE_PRIORS:
                    _BAYES_STATE_PRIORS[key] = 1.0 / (total_diseases + 2)
                _BAYES_LIKELIHOODS.setdefault(key, {})
                # Take max if multiple rules give same symptom (rare)
                prev = _BAYES_LIKELIHOODS[key].get(sym_lower, 0.0)
                _BAYES_LIKELIHOODS[key][sym_lower] = max(prev, conf)


def bayesian_state_posteriors(user_symptoms: list,
                                top_k_states: int = 20) -> Dict:
    """Compute P(state perturbed | observed symptoms) via naive Bayes.

    Returns:
        {
            "posteriors": {
                "organ.state.direction": {
                    "posterior": 0.0-1.0 (normalized over states),
                    "log_score": float (unnormalized log posterior),
                    "supporting_symptoms": [...],
                    "prior": float,
                }
            },
            "unrecognized": [...],
            "top_k": [...]  # top-k state keys sorted by posterior
        }
    """
    _build_bayesian_tables()
    if not _BAYES_STATE_PRIORS:
        return {"posteriors": {}, "unrecognized": [], "top_k": []}

    # Normalize user symptoms via alias expansion
    norm_symptoms = []
    unrecognized = []
    seen_symptoms = set()
    for raw in user_symptoms:
        s_lower = raw.lower().strip().replace("_", " ")
        if s_lower in seen_symptoms:
            continue
        seen_symptoms.add(s_lower)

        # Try direct then alias
        canonical = _REGISTRY._symptom_alias_lookup.get(s_lower, s_lower)
        norm_symptoms.append((raw, canonical))

    # Compute log-posterior for each state
    import math
    state_log_posteriors = {}
    state_support = {}  # state → list of supporting symptoms

    for state_key, prior in _BAYES_STATE_PRIORS.items():
        log_post = math.log(prior + 1e-9)
        likelihoods = _BAYES_LIKELIHOODS.get(state_key, {})

        for raw_sym, canon_sym in norm_symptoms:
            lhood = likelihoods.get(canon_sym)
            if lhood is None:
                # Try checking all known aliases of this symptom
                for canon, aliases in _REGISTRY.symptom_aliases.items():
                    if canon_sym == canon.lower() or canon_sym in [a.lower() for a in aliases]:
                        lhood = likelihoods.get(canon.lower())
                        if lhood is None:
                            # Try aliases
                            for alias in aliases:
                                lhood = likelihoods.get(alias.lower())
                                if lhood is not None:
                                    break
                        if lhood is not None:
                            break

            if lhood is not None:
                # State predicts this symptom
                log_post += math.log(lhood + 1e-9)
                state_support.setdefault(state_key, []).append(raw_sym)
            else:
                # State doesn't predict this symptom; small penalty (base rate)
                log_post += math.log(0.05 + 1e-9)

        state_log_posteriors[state_key] = log_post

    # Identify unrecognized symptoms (no state predicts them at all)
    for raw_sym, canon_sym in norm_symptoms:
        found = False
        for likelihoods in _BAYES_LIKELIHOODS.values():
            if canon_sym in likelihoods:
                found = True
                break
        if not found:
            # Try alias forms
            for canon, aliases in _REGISTRY.symptom_aliases.items():
                if canon_sym == canon.lower() or canon_sym in [a.lower() for a in aliases]:
                    for likelihoods in _BAYES_LIKELIHOODS.values():
                        if canon.lower() in likelihoods:
                            found = True
                            break
                        for alias in aliases:
                            if alias.lower() in likelihoods:
                                found = True
                                break
                        if found:
                            break
                if found:
                    break
        if not found:
            unrecognized.append(raw_sym)

    # Normalize posteriors to 0-1 (relative to max log-posterior).
    # CRITICAL: states with no supporting symptoms get posterior=0 — they have
    # only prior, no evidence. This prevents "phantom posteriors" where high-prior
    # states get high posterior even when zero user symptoms support them.
    if not state_log_posteriors:
        return {"posteriors": {}, "unrecognized": unrecognized, "top_k": []}

    # Find max among states that have actual evidence support (>=1 supporting sym)
    states_with_evidence = [
        (k, v) for k, v in state_log_posteriors.items()
        if state_support.get(k)
    ]
    if states_with_evidence:
        max_log = max(v for _, v in states_with_evidence)
    else:
        # No state has any evidence — return empty posteriors
        return {"posteriors": {}, "unrecognized": unrecognized, "top_k": []}

    posteriors = {}
    for state_key, log_p in state_log_posteriors.items():
        supporting = state_support.get(state_key, [])
        if not supporting:
            # Evidence-free state — posterior = 0. We still record it for
            # transparency, but it cannot win ranking.
            rel = 0.0
        else:
            rel = math.exp(log_p - max_log)  # 0 to 1, only among evidence-supported
        posteriors[state_key] = {
            "posterior": round(rel, 4),
            "log_score": round(log_p, 3),
            "prior": round(_BAYES_STATE_PRIORS[state_key], 4),
            "supporting_symptoms": supporting,
            "evidence_supported": bool(supporting),
        }

    # Top-k by posterior (evidence-supported states will dominate)
    top_k = sorted(posteriors.items(),
                    key=lambda x: -x[1]["posterior"])[:top_k_states]

    return {
        "posteriors": posteriors,
        "unrecognized": unrecognized,
        "top_k": [{"state": k, **v} for k, v in top_k],
    }


def bayesian_disease_score(user_symptoms: list, disease_name: str,
                             posteriors: Optional[Dict] = None) -> Dict:
    """Score a disease using Bayesian state posteriors.

    Logic:
      1. Compute P(state | user symptoms) for all states (or use pre-computed).
      2. Get the disease's predicted state perturbations.
      3. For each predicted (state, direction), check posterior probability.
      4. Disease score = blended mean + match_ratio.

    Args:
      user_symptoms: list of user symptom strings
      disease_name: which disease to score
      posteriors: optional pre-computed posteriors dict (from
                  bayesian_state_posteriors). If None, computes here.
                  Pass pre-computed when scoring many diseases for same patient
                  to avoid re-computing.

    Returns:
        {
            "score": 0.0-1.0,
            "matched_states": [...],
            "missed_states": [...],
            "disease_predictions": [...]
        }
    """
    _build_bayesian_tables()

    # 1. State posteriors (use pre-computed if given)
    if posteriors is None:
        state_inf = bayesian_state_posteriors(user_symptoms, top_k_states=50)
        posteriors = state_inf["posteriors"]
    if not posteriors:
        return {"score": 0.0, "matched_states": [], "missed_states": [],
                "disease_predictions": []}

    # 2. Disease predicted perturbations
    if disease_name not in _REGISTRY._disease_index:
        return {"score": 0.0, "matched_states": [], "missed_states": [],
                "disease_predictions": [], "reason": "disease not in state model"}

    organ_model = _REGISTRY._disease_index[disease_name]
    organ_name = organ_model.organ
    disease_obj = organ_model.diseases.get(disease_name)
    if disease_obj is None:
        return {"score": 0.0, "matched_states": [], "missed_states": [],
                "disease_predictions": [], "reason": "disease object missing"}

    matched = []
    missed = []
    score_terms = []

    for pert in disease_obj.perturbations:
        var = pert.variable
        if not var:
            continue
        delta = pert.delta
        var_meta = organ_model.variables.get(var)
        is_inverse = var_meta.inverse if var_meta else False
        if is_inverse:
            direction = "low" if delta < 0 else "high"
        else:
            direction = "high" if delta > 0 else "low"

        state_key = f"{organ_name}.{var}.{direction}"
        post_data = posteriors.get(state_key)
        if post_data:
            post = post_data["posterior"]
            score_terms.append(post)
            entry = {
                "state": state_key,
                "predicted_direction": direction,
                "delta": delta,
                "posterior": post,
                "supporting_symptoms": post_data["supporting_symptoms"],
            }
            if post > 0.5:
                matched.append(entry)
            else:
                missed.append(entry)

    if not score_terms:
        return {"score": 0.0, "matched_states": [], "missed_states": [],
                "disease_predictions": []}

    # Score: blend mean posterior with match ratio.
    # mean_post captures Bayesian probability mass; match_ratio captures
    # "fraction of disease's predicted states that user evidence supports".
    mean_post = sum(score_terms) / len(score_terms)
    n_matched = sum(1 for t in score_terms if t > 0.5)
    match_ratio = n_matched / len(score_terms)

    score = 0.5 * mean_post + 0.5 * match_ratio

    return {
        "score": round(score, 3),
        "matched_states": matched,
        "missed_states": missed,
        "disease_predictions": len(score_terms),
        "mean_posterior": round(mean_post, 3),
        "match_ratio": round(match_ratio, 3),
    }


# ════════════════════════════════════════════════════════════════════
# G.5 #2: TEMPORAL REASONING
# Adjust disease score based on how long user has had symptoms.
# Each disease has a typical onset window [min_hours, max_hours].
# Duration inside window → boost. Outside → penalty.
#
# Loaded from medical_knowledge/state_models/disease_timecourse.json.
# ════════════════════════════════════════════════════════════════════

_DISEASE_TIMECOURSE: Optional[Dict[str, Dict]] = None


def _load_disease_timecourse() -> Dict[str, Dict]:
    """Load disease timecourse data (cached)."""
    global _DISEASE_TIMECOURSE
    if _DISEASE_TIMECOURSE is not None:
        return _DISEASE_TIMECOURSE
    _DISEASE_TIMECOURSE = {}
    candidates = [
        "medical_knowledge/state_models/disease_timecourse.json",
        "../medical_knowledge/state_models/disease_timecourse.json",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                data = json.load(open(path, encoding="utf-8"))
                _DISEASE_TIMECOURSE = data.get("disease_timecourses", {})
                break
            except Exception:
                continue
    return _DISEASE_TIMECOURSE


def temporal_compatibility_score(disease_name: str,
                                   duration_hours: Optional[float]) -> Dict:
    """Score how compatible a disease is with the symptom duration.

    Args:
      disease_name: e.g. "Heart Attack/MI"
      duration_hours: how long symptoms have lasted, in hours. None = unknown.

    Returns:
      {
        "multiplier": 0.3 to 1.4,  # apply to disease score
        "in_window": bool,
        "window": [min, max] or None,
        "rationale": "...",
      }

    Logic:
      - duration_hours is None → multiplier 1.0 (unchanged)
      - disease has no timecourse data → multiplier 1.0
      - inside typical window → ×1.3 (supports diagnosis)
      - slightly outside (within 2× of window) → ×0.7
      - far outside (>2× of window) → ×0.3
    """
    if duration_hours is None or duration_hours <= 0:
        return {"multiplier": 1.0, "in_window": None,
                "window": None, "rationale": "no duration provided"}

    timecourses = _load_disease_timecourse()
    tc = timecourses.get(disease_name)
    if not tc:
        return {"multiplier": 1.0, "in_window": None,
                "window": None, "rationale": "no timecourse data for this disease"}

    window = tc.get("typical_window_hours")
    if not isinstance(window, list) or len(window) != 2:
        return {"multiplier": 1.0, "in_window": None,
                "window": None, "rationale": "malformed window"}

    min_h, max_h = float(window[0]), float(window[1])

    # Inside typical window — supports
    if min_h <= duration_hours <= max_h:
        return {
            "multiplier": 1.3,
            "in_window": True,
            "window": [min_h, max_h],
            "rationale": f"duration {duration_hours}h fits typical window [{min_h}, {max_h}]h",
        }

    # Outside but within 2× factor — partial penalty
    if (duration_hours < min_h and duration_hours >= min_h / 2) or \
       (duration_hours > max_h and duration_hours <= max_h * 2):
        return {
            "multiplier": 0.7,
            "in_window": False,
            "window": [min_h, max_h],
            "rationale": f"duration {duration_hours}h slightly outside typical [{min_h}, {max_h}]h",
        }

    # Far outside — strong penalty
    return {
        "multiplier": 0.3,
        "in_window": False,
        "window": [min_h, max_h],
        "rationale": f"duration {duration_hours}h far outside typical [{min_h}, {max_h}]h",
    }


def symptom_ordering_compatibility(disease_name: str,
                                       symptom_records: List[Dict]) -> Dict:
    """G.5 #2b: Score disease by symptom onset ORDERING, not just total duration.

    User provides per-symptom onset times. The function classifies the
    presentation pattern (acute monophasic / subacute chronic-on-acute /
    chronic only) and checks if the candidate disease's typical timecourse
    fits that pattern.

    Args:
      disease_name: candidate disease
      symptom_records: list of {"name": str, "onset_hours_ago": float | None}

    Returns:
      {
        "multiplier": 0.4 to 1.3,
        "pattern": "monophasic_acute" | "monophasic_subacute" |
                    "chronic_with_acute_decomp" | "chronic_only" | "unknown",
        "rationale": "..."
      }

    Logic — pattern detection from symptom onset spread:
      - All onsets unknown: pattern=unknown, ×1.0 (no info)
      - All onsets ≤ 24h:    monophasic_acute  → favor acute disease windows
      - All onsets 24-168h:  monophasic_subacute → favor subacute (1-7d)
      - All onsets > 168h:   chronic_only → favor chronic disease (>1wk)
      - Mixed (some old, some new <24h): chronic_with_acute_decomp →
            favor diseases that can have BOTH chronic substrate + acute exacerbation
            (COPD exacerbation, asthma exacerbation, CHF decompensation, etc.)
    """
    if not symptom_records:
        return {"multiplier": 1.0, "pattern": "unknown",
                "rationale": "no symptom records"}

    # Extract onset values (None means unknown)
    onsets = [r.get("onset_hours_ago") for r in symptom_records]
    known_onsets = [o for o in onsets if o is not None]

    if not known_onsets:
        return {"multiplier": 1.0, "pattern": "unknown",
                "rationale": "no symptom has temporal info"}

    # Classify the user's presentation pattern
    has_acute     = any(o <= 24 for o in known_onsets)
    has_subacute  = any(24 < o <= 168 for o in known_onsets)
    has_chronic   = any(o > 168 for o in known_onsets)

    if has_acute and (has_subacute or has_chronic):
        pattern = "chronic_with_acute_decomp"
    elif has_chronic and not has_acute and not has_subacute:
        pattern = "chronic_only"
    elif has_subacute and not has_acute and not has_chronic:
        pattern = "monophasic_subacute"
    elif has_acute and not has_subacute and not has_chronic:
        pattern = "monophasic_acute"
    else:
        pattern = "mixed_subacute_chronic"

    # Look up disease timecourse
    timecourses = _load_disease_timecourse()
    tc = timecourses.get(disease_name)
    if not tc:
        return {"multiplier": 1.0, "pattern": pattern,
                "rationale": "no timecourse data for this disease"}

    window = tc.get("typical_window_hours")
    if not isinstance(window, list) or len(window) != 2:
        return {"multiplier": 1.0, "pattern": pattern,
                "rationale": "malformed window"}

    min_h, max_h = float(window[0]), float(window[1])

    # Classify disease's timecourse class
    if max_h < 24:
        disease_class = "acute"
    elif min_h > 168:
        disease_class = "chronic"
    elif max_h <= 168:
        disease_class = "subacute"
    else:
        disease_class = "subacute_to_chronic"  # spans range

    # Pattern × disease_class compatibility matrix
    # Logic: does the disease's typical window fit the user's onset pattern?
    if pattern == "monophasic_acute":
        # All symptoms < 24h → acute disease ideal
        if disease_class == "acute":
            mult = 1.3
            rat = "all symptoms acute onset — matches acute disease"
        elif disease_class == "subacute":
            mult = 0.7
            rat = "all symptoms acute but disease typically subacute"
        elif disease_class == "chronic":
            mult = 0.4
            rat = "all symptoms acute but disease typically chronic (impossible)"
        else:
            mult = 1.0
            rat = "acceptable"

    elif pattern == "monophasic_subacute":
        # All symptoms 1-7 days → subacute disease ideal
        if disease_class == "subacute" or disease_class == "subacute_to_chronic":
            mult = 1.3
            rat = "all symptoms 1-7 days — matches subacute disease"
        elif disease_class == "acute":
            mult = 0.6
            rat = "symptoms subacute but disease typically acute"
        else:
            mult = 0.6
            rat = "subacute presentation vs chronic disease"

    elif pattern == "chronic_only":
        if disease_class == "chronic" or disease_class == "subacute_to_chronic":
            mult = 1.3
            rat = "all symptoms chronic — matches chronic disease"
        else:
            mult = 0.4
            rat = "chronic presentation but disease is not chronic"

    elif pattern == "chronic_with_acute_decomp":
        # SPECIAL CASE: chronic background + acute symptoms.
        # Best fit: disease that's chronic but can exacerbate.
        # (COPD exacerbation, asthma exacerbation, CHF decompensation,
        #  acute-on-chronic kidney injury, etc.)
        dname_lower = disease_name.lower()
        is_exacerbation = any(kw in dname_lower for kw in
                               ["exacerbation", "decompensation", "acute on chronic",
                                "flare", "decomp"])
        if is_exacerbation or disease_class in ("subacute_to_chronic", "chronic"):
            mult = 1.3
            rat = (f"chronic+acute pattern — fits exacerbation/decomp scenario"
                   if is_exacerbation else
                   f"chronic+acute pattern — disease is chronic baseline")
        elif disease_class == "acute":
            mult = 0.7
            rat = "pure acute disease doesn't explain chronic prodrome"
        else:
            mult = 1.0
            rat = "compatible"

    else:  # mixed_subacute_chronic
        mult = 1.0
        rat = "presentation pattern unclear"

    return {
        "multiplier": mult,
        "pattern": pattern,
        "disease_class": disease_class,
        "disease_window": [min_h, max_h],
        "rationale": rat,
    }


# ════════════════════════════════════════════════════════════════════
# G.5 #3: SYMPTOM-DISEASE MISMATCH PENALTY
# When user symptoms strongly contradict a disease's typical presentation,
# penalize that disease's score.
#
# Strategy: compute mismatch_ratio = symptoms user reports that the disease
# does NOT predict. If mismatch_ratio is very high, the disease is unlikely
# to be the right diagnosis EVEN IF it matches some symptoms.
# ════════════════════════════════════════════════════════════════════

def symptom_mismatch_penalty(user_symptoms: list,
                               disease_name: str,
                               threshold: float = 0.7) -> Dict:
    """Compute mismatch penalty for a disease.

    Logic:
      For each user symptom, check if disease's derived_symptoms predicts it
      (via alias expansion). Count unexplained user symptoms.
      If >70% of user symptoms are NOT predicted by this disease → disease
      is incompatible → multiplier 0.6 (strong penalty).
      If 50-70% unexplained → multiplier 0.85 (mild penalty).
      <50% → multiplier 1.0 (no penalty).

    Note: this is NOT exclusion. Even 70% mismatch returns 0.6, not 0.
    Diseases can present atypically.

    Args:
      user_symptoms: list of user-reported symptoms
      disease_name: target disease
      threshold: mismatch ratio above which strong penalty applies

    Returns:
      {
        "multiplier": 0.6 to 1.0,
        "mismatch_ratio": float,
        "unexplained_symptoms": [...],
        "rationale": "...",
      }
    """
    if not user_symptoms:
        return {"multiplier": 1.0, "mismatch_ratio": 0.0,
                "unexplained_symptoms": [], "rationale": "no symptoms"}

    sim = simulate_disease(disease_name)
    if not sim:
        return {"multiplier": 1.0, "mismatch_ratio": 0.0,
                "unexplained_symptoms": [], "rationale": "no state model"}

    user_norm = {s.lower().replace("_", " ").strip() for s in user_symptoms}
    sim_syms_expanded = expand_symptom_aliases(sim["derived_symptoms"])

    matched = user_norm & sim_syms_expanded
    unexplained = sorted(user_norm - sim_syms_expanded)
    mismatch_ratio = len(unexplained) / max(len(user_norm), 1)

    if mismatch_ratio >= threshold:
        return {
            "multiplier": 0.8,
            "mismatch_ratio": round(mismatch_ratio, 2),
            "unexplained_symptoms": unexplained,
            "rationale": f"{len(unexplained)}/{len(user_norm)} symptoms not predicted by this disease",
        }
    if mismatch_ratio >= 0.5:
        return {
            "multiplier": 0.95,
            "mismatch_ratio": round(mismatch_ratio, 2),
            "unexplained_symptoms": unexplained,
            "rationale": f"{len(unexplained)}/{len(user_norm)} symptoms not well explained",
        }
    return {
        "multiplier": 1.0,
        "mismatch_ratio": round(mismatch_ratio, 2),
        "unexplained_symptoms": unexplained,
        "rationale": f"most symptoms explained ({len(matched)}/{len(user_norm)})",
    }


# ════════════════════════════════════════════════════════════════════
# G.5 #4: HISTORY-BASED RISK MODIFIER
# Past medical history + current symptoms → adjust disease probabilities.
# Example: past MI + current chest pain → MI/ACS more likely (×1.5).
# ════════════════════════════════════════════════════════════════════

# History keyword → list of (disease_pattern, multiplier, rationale)
# disease_pattern is a substring match against disease name (lowercase).
_HISTORY_DISEASE_RULES = [
    # Cardiovascular history
    ("prior mi", "heart attack/mi", 1.6,
     "prior MI → re-infarction risk markedly elevated"),
    ("prior mi", "acute coronary syndrome", 1.6,
     "prior MI → recurrent ACS more likely"),
    ("history of mi", "heart attack/mi", 1.6,
     "prior MI → re-infarction risk markedly elevated"),
    ("cad", "heart attack/mi", 1.4,
     "known CAD → MI more likely"),
    ("cad", "acute coronary syndrome", 1.4,
     "known CAD → ACS more likely"),
    ("coronary artery disease", "heart attack/mi", 1.4,
     "known CAD → MI more likely"),

    # Diabetes
    ("type 1 diabetes", "diabetic ketoacidosis", 1.7,
     "T1DM patients at high DKA risk"),
    ("dm1", "diabetic ketoacidosis", 1.7,
     "T1DM patients at high DKA risk"),
    ("diabetes mellitus", "diabetic ketoacidosis", 1.3,
     "DM patients can decompensate to DKA"),

    # Atrial fibrillation history
    ("atrial fibrillation", "ischemic stroke", 1.5,
     "AFib is major cardioembolic stroke source"),
    ("afib", "ischemic stroke", 1.5,
     "AFib is major cardioembolic stroke source"),

    # Thromboembolism history
    ("prior pe", "acute pulmonary embolism", 1.6,
     "prior PE = strongest predictor of recurrent PE"),
    ("prior dvt", "acute pulmonary embolism", 1.5,
     "DVT history strongly increases PE recurrence"),
    ("prior dvt", "deep vein thrombosis", 1.6,
     "DVT recurrence common"),
    ("hypercoagulable", "acute pulmonary embolism", 1.4,
     "thrombophilia increases PE risk"),

    # COPD/asthma history
    ("asthma", "asthma exacerbation", 1.5,
     "known asthma → exacerbation more likely than first-time wheeze"),
    ("copd", "copd", 1.5,
     "known COPD → COPD exacerbation likely"),

    # Allergy history
    ("anaphylaxis", "anaphylaxis", 1.5,
     "prior anaphylaxis raises baseline risk"),
    ("severe allergies", "anaphylaxis", 1.3,
     "known severe allergies"),

    # Kidney stones
    ("kidney stones", "nephrolithiasis", 1.5,
     "prior stones — 50% recurrence within 10 years"),
    ("nephrolithiasis", "nephrolithiasis", 1.5,
     "prior stones — recurrence very common"),

    # UTI history
    ("recurrent uti", "acute uncomplicated cystitis", 1.3,
     "recurrent UTI predisposition"),

    # GERD/PUD
    ("gerd", "gastroesophageal reflux disease", 1.4,
     "known GERD chronic"),

    # Migraine
    ("migraine", "migraine", 1.5,
     "history of migraine"),

    # Hypertension → preeclampsia, hypertensive emergency, stroke
    ("hypertension", "hypertensive emergency", 1.3,
     "known HTN can decompensate"),
    ("hypertension", "ischemic stroke", 1.2,
     "HTN is major stroke risk factor"),
    ("hypertension", "subarachnoid hemorrhage", 1.2,
     "HTN raises SAH risk"),
]


def _extract_past_diseases_from_history(history: List[str]) -> List[str]:
    """Extract disease names mentioned in patient history strings.

    Best-effort fuzzy match: look for any state-model disease name (or its
    short_name) in the lowercased history text. Returns list of canonical
    disease names found.

    Note: This is a string-matching step, but the SCORING after it is
    mechanism-based. Without NER, we can't do better than substring matching
    here; the mechanism logic is in `mechanism_comorbidity_modifier`.
    """
    if not history:
        return []
    _REGISTRY.load_all()  # ensure disease index is built
    found = set()
    history_text = " ".join(str(h).lower() for h in history)

    # Build name index: full name → canonical, plus common short aliases.
    # Sort by length desc so longer matches win (e.g. "type 2 diabetes" beats "diabetes")
    name_index = []
    for dname in _REGISTRY._disease_index.keys():
        name_index.append((dname.lower(), dname))
        # Common short forms
        if "/" in dname:
            for part in dname.split("/"):
                name_index.append((part.strip().lower(), dname))
    # Add hand-curated common abbreviations
    aliases = {
        "mi":              "Heart Attack/MI",
        "heart attack":    "Heart Attack/MI",
        "stemi":           "Heart Attack/MI",
        "nstemi":          "Heart Attack/MI",
        "afib":            "Atrial Fibrillation",
        "a-fib":           "Atrial Fibrillation",
        "atrial fib":      "Atrial Fibrillation",
        "cad":             "Heart Attack/MI",  # CAD → use MI's perturbations as proxy
        "coronary":        "Heart Attack/MI",
        "stroke":          "Ischemic Stroke",
        "cva":             "Ischemic Stroke",
        "ischemic stroke": "Ischemic Stroke",
        "dm1":             "Type 1 Diabetes Mellitus",
        "type 1 diabetes": "Type 1 Diabetes Mellitus",
        "dm2":             "Type 2 Diabetes Mellitus",
        "type 2 diabetes": "Type 2 Diabetes Mellitus",
        "t2dm":            "Type 2 Diabetes Mellitus",
        "htn":             "Hypertensive Emergency",
        "hypertension":    "Hypertensive Emergency",
        "copd":            "COPD",
        "asthma":          "Asthma Exacerbation",
        "pe":              "Acute Pulmonary Embolism",
        "dvt":             "Deep Vein Thrombosis",
        "gerd":            "Gastroesophageal Reflux Disease",
        "ckd":             "Chronic Kidney Disease",
        "hypothyroid":     "Hypothyroidism",
    }
    for alias, canonical in aliases.items():
        if canonical in _REGISTRY._disease_index:
            name_index.append((alias, canonical))

    # Sort by string length desc for longest-match wins
    name_index.sort(key=lambda x: -len(x[0]))

    for needle, canonical in name_index:
        if needle in history_text:
            found.add(canonical)

    return sorted(found)


def mechanism_comorbidity_modifier(disease_name: str,
                                      past_diseases: List[str]) -> Dict:
    """G.5 #4 (rewritten): mechanism-based comorbidity inference.

    Instead of keyword matching (e.g. "if 'prior mi' in history and 'mi' in
    disease → ×1.6"), this function computes a MECHANISM OVERLAP between
    the past diseases' persistent state legacy and the current disease's
    state requirements.

    Logic:
      For each past_disease:
        - Simulate it. Get its perturbations.
        - Each perturbation = a state that's likely persistently affected
          even after acute resolution (e.g. prior MI → contractility ↓ stays).
      For the candidate current disease D:
        - Get D's perturbations.
        - Count states that overlap between past legacy and D's needs.
        - More overlap → D is mechanism-coherent with patient's substrate.

    Args:
      disease_name: candidate disease being scored
      past_diseases: list of past disease names (extracted from history)

    Returns:
      {
        "multiplier": 1.0 to ~2.0,
        "overlapping_states": [list of "organ.state" that overlap],
        "past_diseases_used": [...],
        "rationale": "..."
      }
    """
    if not past_diseases:
        return {"multiplier": 1.0, "overlapping_states": [],
                "past_diseases_used": [],
                "rationale": "no past disease state data"}

    # Get candidate disease's perturbations
    current_sim = simulate_disease(disease_name)
    if not current_sim:
        return {"multiplier": 1.0, "overlapping_states": [],
                "past_diseases_used": [],
                "rationale": "candidate disease has no state model"}

    # Build current disease's state requirements (organ.variable.direction)
    current_states = set()
    for p in current_sim["perturbations"]:
        organ = p["organ"]
        var = p["variable"]
        delta = p["delta"]
        # Direction based on sign of delta (positive = elevated, negative = reduced)
        # This is the simple version — same as what bayesian_state_posteriors uses.
        direction = "high" if delta > 0 else "low"
        current_states.add(f"{organ}.{var}.{direction}")

    if not current_states:
        return {"multiplier": 1.0, "overlapping_states": [],
                "past_diseases_used": [],
                "rationale": "candidate has no perturbations"}

    # Build legacy state from past diseases
    legacy_states = set()  # "organ.var.direction" → set of (past_disease, delta)
    legacy_provenance = {}  # state_key → list of (past_dz, delta)

    for past_dz in past_diseases:
        # Don't penalize/reward the disease for being its own history
        # (re-infarction is a legitimate clinical pattern, but the boost
        # should come from state overlap not name match)
        if past_dz == disease_name:
            continue
        past_sim = simulate_disease(past_dz)
        if not past_sim:
            continue
        for p in past_sim["perturbations"]:
            organ = p["organ"]
            var = p["variable"]
            delta = p["delta"]
            direction = "high" if delta > 0 else "low"
            key = f"{organ}.{var}.{direction}"
            legacy_states.add(key)
            legacy_provenance.setdefault(key, []).append((past_dz, delta))

    if not legacy_states:
        return {"multiplier": 1.0, "overlapping_states": [],
                "past_diseases_used": past_diseases,
                "rationale": "past diseases have no perturbation overlap"}

    # Compute overlap
    overlap = current_states & legacy_states
    if not overlap:
        return {"multiplier": 1.0, "overlapping_states": [],
                "past_diseases_used": past_diseases,
                "rationale": "no mechanism overlap between past + current"}

    # Compute multiplier: more overlap relative to current disease's needs → higher
    # 1 overlapping state: ×1.2
    # 2 overlapping states: ×1.4
    # 3+ overlapping: ×1.6
    # Special case: if past = current (re-incidence), STILL boost but cap at 1.5
    overlap_count = len(overlap)
    if overlap_count >= 3:
        mult = 1.6
    elif overlap_count == 2:
        mult = 1.4
    else:
        mult = 1.2

    # Build rationale
    overlap_with_provenance = []
    for state_key in sorted(overlap):
        provs = legacy_provenance.get(state_key, [])
        prov_names = [p[0] for p in provs]
        overlap_with_provenance.append({
            "state": state_key,
            "from_past_disease": prov_names,
        })

    rationale = (
        f"{overlap_count} state(s) shared with past disease history: "
        f"{', '.join(sorted(overlap))[:120]}"
    )

    return {
        "multiplier": mult,
        "overlapping_states": list(sorted(overlap)),
        "overlap_with_provenance": overlap_with_provenance,
        "past_diseases_used": past_diseases,
        "rationale": rationale,
    }


def history_disease_modifier(disease_name: str,
                               history: List[str]) -> Dict:
    """Apply history-based risk modifier — MECHANISM-BASED with keyword fallback.

    Tries mechanism overlap first (preferred — uses actual state model).
    Falls back to keyword rules ONLY if no past diseases extractable from
    history text. The mechanism path makes NEXUS's reasoning consistent with
    its core philosophy: rank by state perturbation overlap, not name match.

    Args:
      disease_name: e.g. "Heart Attack/MI"
      history: list of past medical history strings

    Returns:
      {
        "multiplier": 1.0 to ~2.0,
        "method": "mechanism" or "keyword_fallback",
        "rationale": "..."
      }
    """
    if not history:
        return {"multiplier": 1.0, "matched_rules": [],
                "method": "none",
                "rationale": "no history provided"}

    # ── Step 1: Try mechanism-based (preferred) ──
    past_diseases = _extract_past_diseases_from_history(history)
    mech_fired = False
    mech_result = None
    if past_diseases:
        mech_result = mechanism_comorbidity_modifier(disease_name, past_diseases)
        if mech_result["multiplier"] != 1.0 or mech_result["overlapping_states"]:
            mech_fired = True
            return {
                "multiplier":      mech_result["multiplier"],
                "method":          "mechanism",
                "overlapping_states": mech_result["overlapping_states"],
                "past_diseases_used": past_diseases,
                "rationale":       mech_result["rationale"],
            }

    # ── Step 2: Keyword fallback ──
    # Used when (a) no past disease extracted, OR (b) past disease extracted
    # but mechanism overlap was 0 (e.g. AFib → stroke is causal, not
    # state-overlap; can only be captured via keyword rule for now).
    history_text = " ".join(str(h).lower() for h in history)
    dz_lower = disease_name.lower()

    multiplier = 1.0
    matched_rules = []
    for hist_kw, dz_pattern, mult, rationale in _HISTORY_DISEASE_RULES:
        if hist_kw in history_text and dz_pattern in dz_lower:
            # No dedup needed — mechanism already returned (above) if it fired.
            multiplier *= mult
            matched_rules.append(rationale)

    return {
        "multiplier":    round(multiplier, 2),
        "method":        "keyword_fallback" if matched_rules else "none",
        "matched_rules": matched_rules,
        "past_diseases_extracted": past_diseases,  # for transparency
        "rationale":     "; ".join(matched_rules) if matched_rules
                          else "no matching history modifier",
    }


# ════════════════════════════════════════════════════════════════════
# G.5 #5: DRUG RESPONSE INFERENCE (skeleton — full system pending)
# When user reports "drug X relieved symptom Y", infer the mechanism
# that's consistent with that drug's action.
# ════════════════════════════════════════════════════════════════════

# Drug → mechanism map. Loaded from JSON (medical_knowledge/state_models/
# drug_mechanisms.json) so drugs can be added by editing JSON, not Python.
_DRUG_MECHANISM_MAP_CACHE: Optional[Dict[str, Dict]] = None


def _load_drug_mechanism_map() -> Dict[str, Dict]:
    """Load drug mechanism map from JSON (cached).

    Returns dict keyed by lowercase drug name. Empty dict on failure
    (treatment/drug-response features degrade gracefully rather than crash).
    """
    global _DRUG_MECHANISM_MAP_CACHE
    if _DRUG_MECHANISM_MAP_CACHE is not None:
        return _DRUG_MECHANISM_MAP_CACHE
    _DRUG_MECHANISM_MAP_CACHE = {}
    candidates = [
        "medical_knowledge/state_models/drug_mechanisms.json",
        "../medical_knowledge/state_models/drug_mechanisms.json",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                data = json.load(open(p, encoding="utf-8"))
                # normalize keys to lowercase for consistent lookup
                _DRUG_MECHANISM_MAP_CACHE = {
                    str(k).lower(): v for k, v in data.get("drugs", {}).items()
                }
                break
            except Exception as e:
                print(f"[state_model] WARNING: failed to load drug_mechanisms.json: {e}")
                continue
    return _DRUG_MECHANISM_MAP_CACHE


class _DrugMechanismMapProxy:
    """Lazy dict-like proxy so existing code using _DRUG_MECHANISM_MAP[...] /
    `in` / .items() / .values() keeps working unchanged after the move to JSON."""
    def __getitem__(self, key):
        return _load_drug_mechanism_map()[str(key).lower()]
    def __contains__(self, key):
        return str(key).lower() in _load_drug_mechanism_map()
    def get(self, key, default=None):
        return _load_drug_mechanism_map().get(str(key).lower(), default)
    def items(self):
        return _load_drug_mechanism_map().items()
    def keys(self):
        return _load_drug_mechanism_map().keys()
    def values(self):
        return _load_drug_mechanism_map().values()
    def __len__(self):
        return len(_load_drug_mechanism_map())
    def __iter__(self):
        return iter(_load_drug_mechanism_map())


_DRUG_MECHANISM_MAP = _DrugMechanismMapProxy()


def drug_response_modifier(disease_name: str,
                             drug_responses: List[Dict]) -> Dict:
    """Apply drug-response modifier to disease score.

    Args:
      disease_name: e.g. "Heart Attack/MI"
      drug_responses: list of dicts like
        [{"drug": "nitroglycerin", "effect": "relieved", "symptom": "chest pain"}]

    Returns:
      {"multiplier": 1.0 to 1.5, "rationale": "..."}
    """
    if not drug_responses:
        return {"multiplier": 1.0, "rationale": "no drug response data"}

    dz_lower = disease_name.lower()
    multiplier = 1.0
    rationales = []

    for dr in drug_responses:
        if not isinstance(dr, dict):
            continue
        drug = str(dr.get("drug", "")).lower().strip()
        effect = str(dr.get("effect", "")).lower().strip()
        sym = str(dr.get("symptom", "")).lower().strip()

        if drug not in _DRUG_MECHANISM_MAP:
            continue
        drug_info = _DRUG_MECHANISM_MAP[drug]

        # "Relieved" → drug supports the disease it targets
        if effect in {"relieved", "improved", "worked", "helped"}:
            for supported_dz in drug_info["supports_diseases"]:
                if supported_dz in dz_lower:
                    multiplier *= 1.4
                    rationales.append(f"{drug} relieved → {drug_info['rationale']}")
                    break

        # "No effect" → drug does NOT support diseases it targets
        elif effect in {"no effect", "didn't help", "didnt help",
                          "ineffective", "didn't work"}:
            for supported_dz in drug_info["supports_diseases"]:
                if supported_dz in dz_lower:
                    multiplier *= 0.7
                    rationales.append(
                        f"{drug} ineffective → less consistent with {supported_dz}")
                    break

    return {
        "multiplier": round(multiplier, 2),
        "rationale": "; ".join(rationales) if rationales else "no drug match",
    }


# ════════════════════════════════════════════════════════════════════
# G.5 #3: NEGATIVE EVIDENCE — handle "patient denies X"
# When patient explicitly denies a symptom, diseases that strongly
# predict that symptom become less likely.
#
# Strength depends on rule confidence in the disease's state model:
#   - High confidence derivation (conf >= 0.9): disease NEEDS this symptom
#     → strong penalty ×0.3 when denied
#   - Moderate (0.7-0.9): expected but not required → ×0.7
#   - Low (<0.7): probabilistic → ×0.9 (minor penalty)
#   - Disease doesn't predict the symptom at all: ×1.0 (no info)
# ════════════════════════════════════════════════════════════════════

def negative_evidence_modifier(disease_name: str,
                                  denied_symptoms: List[str]) -> Dict:
    """Apply negative-evidence penalty for symptoms the patient denies.

    Each denied symptom is checked against the disease's derived_symptoms
    (which come from its state perturbation rules). If the disease would
    derive symptom Y with high confidence, and patient denies Y, that's
    strong evidence AGAINST this disease.

    Args:
      disease_name: candidate disease being scored
      denied_symptoms: list of symptoms patient explicitly says they don't have

    Returns:
      {
        "multiplier": 0.3 to 1.0 (never boost via negation),
        "denied_predicted":  list of symptoms denied that disease predicted,
        "penalties_applied": list of {symptom, confidence, multiplier_factor},
        "rationale": "...",
      }
    """
    if not denied_symptoms:
        return {"multiplier": 1.0, "denied_predicted": [],
                "penalties_applied": [], "rationale": "no denied symptoms"}

    sim = simulate_disease(disease_name)
    if not sim:
        return {"multiplier": 1.0, "denied_predicted": [],
                "penalties_applied": [], "rationale": "no state model"}

    # Normalize denied symptoms (lowercase + space)
    denied_norm = {s.lower().strip().replace("_", " ") for s in denied_symptoms if s}
    if not denied_norm:
        return {"multiplier": 1.0, "denied_predicted": [],
                "penalties_applied": [], "rationale": "empty denied list"}

    # Expand denied via aliases — if user denies "SOB", treat as denying
    # "shortness of breath" too. This makes denial consistent with how
    # affirmation works (which goes through alias expansion).
    _REGISTRY.load_all()
    expanded_denied = set(denied_norm)
    for canonical, alist in _REGISTRY.symptom_aliases.items():
        canon_low = canonical.lower()
        alias_set = {a.lower() for a in alist}
        if canon_low in denied_norm or denied_norm & alias_set:
            expanded_denied.add(canon_low)
            expanded_denied |= alias_set

    # Now find which DENIED symptoms the disease actually predicts.
    # Walk the derivation rules — match by symptom string, get confidence.
    multiplier = 1.0
    penalties = []
    denied_predicted = []

    derivations = sim.get("derivations", [])
    for d in derivations:
        sym = str(d.get("symptom", "")).lower().strip()
        if not sym:
            continue
        if sym in expanded_denied:
            conf = float(d.get("confidence", 0.5))
            # Compute penalty factor: higher confidence → bigger penalty
            if conf >= 0.9:
                factor = 0.3
            elif conf >= 0.7:
                factor = 0.6
            else:
                factor = 0.85
            multiplier *= factor
            penalties.append({
                "symptom": sym,
                "confidence": conf,
                "multiplier_factor": factor,
            })
            denied_predicted.append(sym)

    # Cap floor — never below ×0.1
    multiplier = max(multiplier, 0.1)

    if not penalties:
        rationale = "patient denied symptoms not predicted by this disease (no info)"
    else:
        rationale = (f"patient denies {len(penalties)} symptom(s) that this disease "
                     f"would predict — strongest: "
                     f"{penalties[0]['symptom']!r} (conf={penalties[0]['confidence']:.2f})")

    return {
        "multiplier": round(multiplier, 3),
        "denied_predicted": denied_predicted,
        "penalties_applied": penalties,
        "rationale": rationale,
    }


# ════════════════════════════════════════════════════════════════════
# G.5 #9: LAB RECOMMENDATION FROM MISSING STATE
# Given a candidate disease and current user evidence, infer which labs
# would test the disease's predicted state perturbations that user
# symptoms haven't already confirmed.
#
# Strategy:
#   1. Disease D predicts perturbations P = {(organ, state, direction), ...}
#   2. User evidence supports states E ⊆ all_possible_states
#   3. Gap: states D predicts but E hasn't confirmed
#   4. Reverse-lookup labs: which labs measure these states?
#   5. Rank labs by:
#      - specificity (lab measures one state vs many)
#      - magnitude of state perturbation (delta)
# ════════════════════════════════════════════════════════════════════

# Cache: state → list of labs that measure it (reverse index)
_STATE_TO_LABS_INDEX: Optional[Dict[str, List[Dict]]] = None


def _build_state_to_labs_index() -> Dict[str, List[Dict]]:
    """Reverse-index from organ.state → [labs that measure it].

    Reads lab_integration.json and builds:
      "heart.ischemia": [
        {"lab": "troponin", "direction": "high",
         "weight": 0.9, "rationale": "..."}
      ]

    Returns dict keyed by "organ.state" (without direction suffix —
    each entry records its own direction).
    """
    global _STATE_TO_LABS_INDEX
    if _STATE_TO_LABS_INDEX is not None:
        return _STATE_TO_LABS_INDEX

    _STATE_TO_LABS_INDEX = {}
    candidates = [
        "medical_knowledge/state_models/lab_integration.json",
        "../medical_knowledge/state_models/lab_integration.json",
    ]
    lab_data = None
    for path in candidates:
        if os.path.exists(path):
            try:
                lab_data = json.load(open(path, encoding="utf-8"))
                break
            except Exception:
                continue
    if not lab_data:
        return _STATE_TO_LABS_INDEX

    mapping = lab_data.get("lab_to_state_mapping", {})
    for lab_name, entries in mapping.items():
        if lab_name.startswith("_"):  # skip _comment
            continue
        if not isinstance(entries, list):
            continue
        for e in entries:
            organ = e.get("organ")  # may be None (cross-organ marker)
            state = e.get("state")
            if not state:
                continue

            # high_means_perturb tells us: high lab → state up (direction "high")
            # low_means_perturb tells us: low lab → state up (direction "low")
            for key, direction in [("high_means_perturb", "high"),
                                    ("low_means_perturb", "low")]:
                weight = e.get(key)
                if weight is None:
                    continue
                # Key: organ.state (organ may be None for cross-organ labs like WBC→inflammation)
                organ_str = organ if organ else "*"
                map_key = f"{organ_str}.{state}.{direction}"
                _STATE_TO_LABS_INDEX.setdefault(map_key, []).append({
                    "lab": lab_name,
                    "direction": direction,
                    "weight": float(weight),
                    "rationale": e.get("rationale", ""),
                })

    return _STATE_TO_LABS_INDEX


def recommend_labs_for_diagnosis(disease_name: str,
                                    user_symptoms: List[str],
                                    max_labs: int = 5) -> Dict:
    """Recommend lab tests for a candidate disease based on state gap.

    Args:
      disease_name: candidate disease being investigated
      user_symptoms: symptoms patient has reported (used to compute
                     which states are already evidence-supported)
      max_labs: cap on number of recommendations

    Returns:
      {
        "disease": disease_name,
        "recommended_labs": [
          {
            "lab": "troponin",
            "tests_state": "heart.ischemia.high",
            "delta": 0.85,
            "rationale": "...",
            "priority": "high|medium|low"
          }
        ],
        "rationale_summary": "..."
      }
    """
    sim = simulate_disease(disease_name)
    if not sim:
        return {"disease": disease_name, "recommended_labs": [],
                "rationale_summary": "disease has no state model"}

    # Build set of states user symptoms ALREADY support
    user_state_inference = infer_state_evidence(user_symptoms)
    user_supported_states = set(user_state_inference["state_evidence"].keys())

    # Build disease's predicted perturbation states
    predicted_states = []  # list of (state_key, delta)
    for p in sim["perturbations"]:
        organ = p["organ"]
        var = p["variable"]
        delta = p["delta"]
        direction = "high" if delta > 0 else "low"
        state_key = f"{organ}.{var}.{direction}"
        predicted_states.append((state_key, abs(delta)))

    if not predicted_states:
        return {"disease": disease_name, "recommended_labs": [],
                "rationale_summary": "disease has no perturbations"}

    # GAP: predicted states NOT supported by user symptoms
    state_to_labs_index = _build_state_to_labs_index()
    candidate_labs = []  # list of dicts

    for state_key, delta_magnitude in predicted_states:
        # Already supported? Skip — lab not needed to confirm
        # state_key includes direction; user_supported_states uses different scheme,
        # so do best-effort match by ignoring direction suffix for comparison
        state_core = state_key.rsplit(".", 1)[0]  # "heart.ischemia.high" → "heart.ischemia"
        user_states_core = {k.rsplit(".", 1)[0] if "." in k else k
                              for k in user_supported_states}
        # Note: user state evidence uses different key format
        # (we check both the full state_key and the core)
        if state_key in user_supported_states or state_core in user_states_core:
            continue

        # Try direct match in lab index
        labs = state_to_labs_index.get(state_key, [])
        # Also try cross-organ "*.state.direction" (e.g. WBC for any inflammation)
        organ, var, direction = state_key.split(".", 2)
        wildcard_key = f"*.{var}.{direction}"
        labs = labs + state_to_labs_index.get(wildcard_key, [])

        for lab_entry in labs:
            # Score: lab weight × perturbation magnitude
            score = lab_entry["weight"] * delta_magnitude
            candidate_labs.append({
                "lab":          lab_entry["lab"],
                "tests_state":  state_key,
                "delta":        round(delta_magnitude, 2),
                "rationale":    lab_entry["rationale"],
                "score":        round(score, 3),
            })

    if not candidate_labs:
        return {"disease": disease_name, "recommended_labs": [],
                "rationale_summary":
                f"{len(predicted_states)} predicted states already supported "
                f"by user symptoms, or no labs map to remaining states"}

    # Dedupe by lab name (keep highest-scoring entry per lab)
    by_lab = {}
    for c in candidate_labs:
        existing = by_lab.get(c["lab"])
        if existing is None or c["score"] > existing["score"]:
            by_lab[c["lab"]] = c

    # Sort and label priority
    sorted_labs = sorted(by_lab.values(), key=lambda x: -x["score"])
    for lab in sorted_labs:
        if lab["score"] >= 0.5:
            lab["priority"] = "high"
        elif lab["score"] >= 0.25:
            lab["priority"] = "medium"
        else:
            lab["priority"] = "low"

    final_labs = sorted_labs[:max_labs]

    summary = (f"For '{disease_name}': {len(final_labs)} lab(s) suggested to "
                 f"confirm {len(final_labs)} predicted perturbation(s) not yet "
                 f"supported by symptoms alone.")

    return {
        "disease":           disease_name,
        "recommended_labs":  final_labs,
        "n_predicted_states": len(predicted_states),
        "n_user_supported":  len(user_supported_states),
        "rationale_summary": summary,
    }


def recommend_labs_for_top_diagnoses(diagnoses: List[Dict],
                                         user_symptoms: List[str],
                                         top_k: int = 3,
                                         max_labs_per_disease: int = 3) -> Dict:
    """Top-level: aggregate lab recommendations across top-k diagnoses.

    Args:
      diagnoses: ranked diagnoses (with .disease field)
      user_symptoms: patient symptoms
      top_k: how many top diagnoses to consider
      max_labs_per_disease: cap labs per disease

    Returns:
      {
        "labs_aggregated": [
          {"lab": "troponin", "supports_diagnoses": ["MI"], "max_priority": "high", "score": 0.85},
          ...
        ],
        "per_disease": {disease_name: recommendation_dict, ...}
      }
    """
    per_disease = {}
    lab_aggregator = {}  # lab_name → {supports_diagnoses: [...], max_score: X}

    for d in diagnoses[:top_k]:
        dname = d.get("disease")
        if not dname:
            continue
        rec = recommend_labs_for_diagnosis(dname, user_symptoms,
                                              max_labs=max_labs_per_disease)
        per_disease[dname] = rec

        for lab_rec in rec["recommended_labs"]:
            lab_name = lab_rec["lab"]
            entry = lab_aggregator.setdefault(lab_name, {
                "lab": lab_name,
                "supports_diagnoses": [],
                "max_score": 0.0,
                "max_priority": "low",
                "rationales": [],
            })
            entry["supports_diagnoses"].append(dname)
            if lab_rec["score"] > entry["max_score"]:
                entry["max_score"] = lab_rec["score"]
                entry["max_priority"] = lab_rec["priority"]
            entry["rationales"].append(f"{dname}: {lab_rec['rationale'][:60]}")

    # Sort aggregated by max_score
    labs_aggregated = sorted(lab_aggregator.values(),
                              key=lambda x: -x["max_score"])

    return {
        "labs_aggregated": labs_aggregated,
        "per_disease":     per_disease,
    }


# ════════════════════════════════════════════════════════════════════
# G.5 #10: MECHANISM-DERIVED TREATMENT RECOMMENDATION
# Given a disease, identify treatment targets from its state perturbations,
# then find drugs whose state_effects counteract those perturbations.
# NO hardcoded disease→drug lookup — purely derived from mechanism.
# ════════════════════════════════════════════════════════════════════

# Cache: (organ, variable, direction) → list of drugs that counteract it
_STATE_TO_DRUGS_INDEX: Optional[Dict[str, List[Dict]]] = None


def _build_state_to_drugs_index() -> Dict[str, List[Dict]]:
    """Reverse-index from organ.variable.direction → drugs that PUSH IT THERE.

    Example: drugs that push heart.ischemia → low:
      nitroglycerin (state_effects includes heart.ischemia.low)
      aspirin (state_effects includes heart.ischemia.low)
      heparin (state_effects includes heart.ischemia.low)

    Returns dict keyed by "organ.variable.direction".
    """
    global _STATE_TO_DRUGS_INDEX
    if _STATE_TO_DRUGS_INDEX is not None:
        return _STATE_TO_DRUGS_INDEX

    _STATE_TO_DRUGS_INDEX = {}
    for drug_name, drug_info in _DRUG_MECHANISM_MAP.items():
        effects = drug_info.get("state_effects", [])
        if not effects:
            continue
        for e in effects:
            organ = e.get("organ", "*")
            var   = e.get("variable")
            direction = e.get("direction")
            if not var or not direction:
                continue
            key = f"{organ}.{var}.{direction}"
            _STATE_TO_DRUGS_INDEX.setdefault(key, []).append({
                "drug":       drug_name,
                "drug_class": drug_info.get("drug_class", ""),
                "rationale":  drug_info.get("rationale", ""),
            })
    return _STATE_TO_DRUGS_INDEX


def recommend_treatment_for_diagnosis(disease_name: str,
                                          max_recommendations: int = 6) -> Dict:
    """Mechanism-derived treatment recommendation.

    Logic:
      1. Get disease's perturbations (e.g. MI → heart.ischemia +0.85)
      2. Each perturbation defines a TREATMENT TARGET: push state back toward healthy.
         - perturbation +delta → need to push direction "low"  to counteract
         - perturbation -delta → need to push direction "high" to counteract
      3. Reverse-lookup _DRUG_MECHANISM_MAP: which drugs push that state?
      4. Rank drugs by:
         - magnitude of perturbation they target (higher delta → priority)
         - how many treatment targets they hit (multi-effect drugs prioritized)
         - drug class diversity (recommend different classes first)
      5. Return ranked treatment plan with mechanism rationale per drug.

    NO hardcoded "MI → give nitroglycerin" — entirely derived.
    """
    sim = simulate_disease(disease_name)
    if not sim:
        return {"disease": disease_name, "recommended_drugs": [],
                "treatment_targets": [],
                "rationale_summary": "no state model for this disease"}

    perturbations = sim.get("perturbations", [])
    if not perturbations:
        return {"disease": disease_name, "recommended_drugs": [],
                "treatment_targets": [],
                "rationale_summary": "disease has no perturbations to target"}

    # ── Step 1: Build treatment targets from perturbations ──
    # For each perturbation (organ.var Δ), the TARGET direction is the OPPOSITE
    treatment_targets = []
    for p in perturbations:
        organ = p["organ"]
        var = p["variable"]
        delta = p["delta"]
        # If perturbation pushed state up (delta > 0), we want to push it DOWN
        # If perturbation pushed state down (delta < 0), we want to push it UP
        target_direction = "low" if delta > 0 else "high"
        treatment_targets.append({
            "organ":          organ,
            "variable":       var,
            "target_direction": target_direction,
            "magnitude":      round(abs(delta), 2),
            "rationale":      f"disease perturbed {organ}.{var} → "
                              f"treatment should push it {target_direction}",
        })

    # ── Step 2: For each target, find drugs that push state in target direction ──
    state_to_drugs = _build_state_to_drugs_index()
    drug_candidates = {}  # drug_name → {hits: [...], total_magnitude: float}

    for target in treatment_targets:
        organ = target["organ"]
        var = target["variable"]
        direction = target["target_direction"]
        magnitude = target["magnitude"]

        # Direct match on organ
        key_specific  = f"{organ}.{var}.{direction}"
        key_wildcard  = f"*.{var}.{direction}"   # drugs that affect this state in any organ

        for key in (key_specific, key_wildcard):
            for drug_entry in state_to_drugs.get(key, []):
                drug = drug_entry["drug"]
                entry = drug_candidates.setdefault(drug, {
                    "drug":       drug,
                    "drug_class": drug_entry["drug_class"],
                    "rationale":  drug_entry["rationale"],
                    "hits":       [],
                    "total_magnitude": 0.0,
                })
                # Avoid double-counting same target via specific+wildcard
                hit_key = f"{organ}.{var}"
                if hit_key in [h["organ_var"] for h in entry["hits"]]:
                    continue
                entry["hits"].append({
                    "organ_var": hit_key,
                    "target_direction": direction,
                    "magnitude": magnitude,
                })
                entry["total_magnitude"] += magnitude

    if not drug_candidates:
        return {"disease": disease_name, "recommended_drugs": [],
                "treatment_targets": treatment_targets,
                "rationale_summary":
                "no drugs in drug map counteract this disease's perturbations "
                f"({len(treatment_targets)} targets identified)"}

    # ── Step 3: Rank drugs — prefer multi-hit + high-magnitude ──
    ranked = sorted(drug_candidates.values(),
                     key=lambda x: (-len(x["hits"]), -x["total_magnitude"]))

    # ── Step 4: Class-diverse top-N selection ──
    # Avoid recommending 3 different β-blockers when 1 + diverse classes is better.
    seen_classes = set()
    final_recommendations = []
    for r in ranked:
        cls = r["drug_class"]
        if cls and cls in seen_classes:
            # Already have a drug of this class — skip (but allow if multi-hit)
            if len(r["hits"]) < 2:
                continue
        if cls:
            seen_classes.add(cls)
        # Priority labels
        if len(r["hits"]) >= 3 or r["total_magnitude"] >= 1.5:
            priority = "high"
        elif len(r["hits"]) >= 2 or r["total_magnitude"] >= 0.7:
            priority = "medium"
        else:
            priority = "low"
        r["priority"] = priority
        final_recommendations.append(r)
        if len(final_recommendations) >= max_recommendations:
            break

    summary = (f"{len(treatment_targets)} treatment target(s) identified from "
                 f"{disease_name}'s state perturbations. "
                 f"{len(final_recommendations)} drug(s) match these targets "
                 f"via state-effect mechanism.")

    return {
        "disease":            disease_name,
        "treatment_targets":  treatment_targets,
        "recommended_drugs":  final_recommendations,
        "rationale_summary":  summary,
    }


def recommend_treatment_for_top_diagnoses(diagnoses: List[Dict],
                                                top_k: int = 3,
                                                max_drugs_per_disease: int = 4) -> Dict:
    """Aggregate mechanism-derived treatment recommendations across top-k diagnoses.

    Returns:
      {
        "treatment_per_disease": {disease_name: recommendation_dict},
        "drugs_aggregated": [
          {"drug": "aspirin", "supports_diagnoses": ["MI", "ACS"],
           "max_priority": "high", "total_hits": 4}
        ]
      }
    """
    per_disease = {}
    drug_aggregator = {}

    for d in diagnoses[:top_k]:
        dname = d.get("disease")
        if not dname:
            continue
        rec = recommend_treatment_for_diagnosis(dname,
                                                  max_recommendations=max_drugs_per_disease)
        per_disease[dname] = rec
        for drug_rec in rec.get("recommended_drugs", []):
            dname_drug = drug_rec["drug"]
            entry = drug_aggregator.setdefault(dname_drug, {
                "drug": dname_drug,
                "drug_class": drug_rec.get("drug_class", ""),
                "supports_diagnoses": [],
                "total_hits": 0,
                "max_priority": "low",
                "rationale": drug_rec.get("rationale", ""),
            })
            entry["supports_diagnoses"].append(dname)
            entry["total_hits"] += len(drug_rec.get("hits", []))
            # Promote priority if any disease called it high
            pri = drug_rec.get("priority", "low")
            if pri == "high" or (pri == "medium" and entry["max_priority"] == "low"):
                entry["max_priority"] = pri

    drugs_aggregated = sorted(drug_aggregator.values(),
                                 key=lambda x: (-x["total_hits"],
                                                 -len(x["supports_diagnoses"])))

    return {
        "treatment_per_disease": per_disease,
        "drugs_aggregated":      drugs_aggregated,
    }


def reload_state_model() -> Dict[str, int]:
    """Clear ALL module-level caches and force re-load of state model + frameworks.

    Call this after editing any JSON in medical_knowledge/state_models/.
    Without this, individual caches retain stale data and queries return
    inconsistent results.

    Returns a dict summary of what was reloaded.
    """
    global _CACHED_ATLAS, _CACHED_ATLAS_FAILED
    global _SYMPTOM_TO_STATES_INDEX
    global _BAYES_STATE_PRIORS, _BAYES_LIKELIHOODS
    global _DISEASE_TIMECOURSE
    global _DRUG_MECHANISM_MAP_CACHE, _STATE_TO_DRUGS_INDEX, _STATE_TO_LABS_INDEX

    # Reset module-level caches
    _SIMULATE_DISEASE_CACHE.clear()
    _SYMPTOM_TO_STATES_INDEX = None
    _BAYES_STATE_PRIORS = None
    _BAYES_LIKELIHOODS = None
    _DISEASE_TIMECOURSE = None
    _CACHED_ATLAS = None
    _CACHED_ATLAS_FAILED = False
    _DRUG_MECHANISM_MAP_CACHE = None    # drug map (now JSON-backed)
    _STATE_TO_DRUGS_INDEX = None        # reverse index built from drug map
    _STATE_TO_LABS_INDEX = None         # reverse index built from lab map

    # Reset registry — simplest: clear .organs, ._disease_index, then re-load.
    # Other framework attrs get overwritten during load_all().
    _REGISTRY.organs.clear()
    _REGISTRY._disease_index.clear()
    _REGISTRY._loaded = False
    _REGISTRY.load_all()

    return {
        "organs":          len(_REGISTRY.organs),
        "diseases":        len(_REGISTRY._disease_index),
        "aliases":         len(getattr(_REGISTRY, "symptom_aliases", {}) or {}),
        "drugs":           len(_load_drug_mechanism_map()),
        "atlas_loaded":    _CACHED_ATLAS is not None or _CACHED_ATLAS_FAILED is False,
    }


# Backward compat — the global registry now behaves like a dict (via has_disease + get)
STATE_MODEL_DISEASES = _REGISTRY