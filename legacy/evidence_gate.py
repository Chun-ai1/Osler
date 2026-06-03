"""
evidence_gate.py  —  4-Layer Evidence Sufficiency Gate
═══════════════════════════════════════════════════════════════════════
All sets (VAGUE_STANDALONE, SYSTEM_ANCHORS, VIRAL_ANCHORS, etc.)
are built at runtime from symptom_000N.json files.
No clinical lists hardcoded here.

Vague classification rules (derived from the data):
  • 3+ systems listed          → vague (non-discriminating)
  • gastrointestinal + neurologic   → vague (classic non-specific pair)
  • neurologic + circulatory        → vague (needs laterality/location)
  • musculoskeletal + neurologic    → vague (needs laterality)

Layers:
  1  TRIAGE / SAFETY    always available from red flags + triage level
  2  SYSTEM / SYNDROME  requires ≥1 non-vague system anchor
  3  ETIOLOGY           requires pathogen evidence hits > 0
  4  DISEASE LABEL      anchor + score≥0.72 + margin≥0.15
"""
from __future__ import annotations
import json, os, glob
from dataclasses import dataclass, field
from typing import List, Set, Dict, Optional, Tuple


# ══════════════════════════════════════════════════════════════════════
#  Symptom-file loader + set builder
# ══════════════════════════════════════════════════════════════════════

_VAGUE_SYSTEM_PAIRS: Tuple[frozenset, ...] = (
    frozenset({"gastrointestinal", "neurologic"}),   # nausea, vomiting
    frozenset({"neurologic", "circulatory"}),         # tingling, numbness
    frozenset({"musculoskeletal", "neurologic"}),     # weakness
    frozenset({"immune", "thermoregulatory"}),        # chills (fever onset — not localizing alone)
)

def _is_vague(systems: List[str]) -> bool:
    """Return True if this symptom cannot anchor a disease label alone."""
    sys_set = frozenset(s.lower().strip() for s in systems)
    if len(sys_set) >= 3:
        return True
    for pair in _VAGUE_SYSTEM_PAIRS:
        if pair.issubset(sys_set):
            return True
    return False


def _build_sets_from_files(folder: str) -> dict:
    """
    Read every symptom_*.json in *folder* and return:
      vague_standalone   – symptoms that need location/context before they anchor
      system_anchors     – system → set of non-vague symptom names
      syndrome_labels    – system → human-readable syndrome string
      viral_anchors      – symptoms whose causes include viral illness
      bacterial_anchors  – symptoms whose causes include bacterial illness
      red_flag_syms      – symptoms with ≥3 red-flag entries
      sym_systems        – symptom → list of systems (raw, for reference)
    """
    paths = sorted(glob.glob(os.path.join(folder, "symptom_*.json")))
    data  = []
    for p in paths:
        try:
            obj = json.load(open(p, encoding="utf-8"))
            if obj.get("symptom"):          # skip blank entries
                data.append(obj)
        except Exception:
            pass

    vague_standalone : Set[str] = set()
    system_anchors   : Dict[str, Set[str]] = {}
    viral_anchors    : Set[str] = set()
    bacterial_anchors: Set[str] = set()
    red_flag_syms    : Set[str] = set()
    sym_systems      : Dict[str, List[str]] = {}

    for s in data:
        name     = s["symptom"].lower().strip()
        systems  = [x.lower().strip() for x in (s.get("systems") or [])]
        causes   = " ".join(c.lower() for c in (s.get("common_causes") or []))
        red_flags = s.get("red_flags") or []

        sym_systems[name] = systems

        # ── Vague classification ──────────────────────────────────────
        if _is_vague(systems):
            vague_standalone.add(name)

        # ── System anchors (non-vague symptoms only) ──────────────────
        if name not in vague_standalone:
            for sys in systems:
                system_anchors.setdefault(sys, set()).add(name)

        # ── Viral / bacterial anchors (from causes field) ─────────────
        if any(k in causes for k in ("viral", " flu", "cold", "virus")):
            viral_anchors.add(name)
        if any(k in causes for k in ("bacterial", "bacteria", "food poisoning")):
            bacterial_anchors.add(name)

        # ── Red-flag symptoms (≥3 entries = clinically serious) ───────
        if len(red_flags) >= 3:
            red_flag_syms.add(name)

    # Fixed syndrome labels per system (presentation layer only)
    syndrome_labels: Dict[str, str] = {
        "respiratory":           "respiratory syndrome",
        "cardiac":               "cardiovascular process",
        "cardiovascular":        "cardiovascular process",
        "gastrointestinal":      "GI syndrome",
        "neurologic":            "neurologic syndrome",
        "musculoskeletal":       "musculoskeletal process",
        "cutaneous":             "dermatologic process",
        "immune":                "immune / allergic process",
        "thermoregulatory":      "systemic illness",
        "circulatory":           "circulatory process",
        "reproductive":          "reproductive process",
        "urinary":               "urinary tract process",
        "nervous":               "neurologic process",
        "auditory (vestibular)": "vestibular syndrome",
        "lymphatic":             "lymphatic process",
        "vascular":              "vascular process",
    }

    # ── Disease index from disease_0001.json ─────────────────────────────────
    dis_index = _build_disease_index(folder)

    print(
        f"[evidence_gate] Built from {len(data)} symptom files + "
        f"{len(dis_index)} diseases: "
        f"{len(vague_standalone)} vague, "
        f"{len(system_anchors)} systems, "
        f"{len(viral_anchors)} viral anchors"
    )
    return {
        "vague_standalone" : vague_standalone,
        "system_anchors"   : system_anchors,
        "syndrome_labels"  : syndrome_labels,
        "viral_anchors"    : viral_anchors,
        "bacterial_anchors": bacterial_anchors,
        "red_flag_syms"    : red_flag_syms,
        "sym_systems"      : sym_systems,
        "disease_index"    : dis_index,
    }


_TRIAGE_URGENCY = {
    "emergent": 5, "ed": 4, "urgent_care": 3, "urgent": 3,
    "semi-urgent": 2, "clinic": 1, "routine": 0,
}

def _norm_dis_sym(s: str) -> str:
    """Normalize a disease symptom string for matching against user symptoms."""
    import re
    s = s.lower().strip()
    s = re.sub(r"\([^)]*\)", "", s).strip()   # remove parentheticals
    s = re.split(r"\s+(?:with|and|or|before|after|in|of|due)\s+", s)[0].strip()
    return s


def _build_disease_index(folder: str) -> dict:
    """
    Build disease evidence index.
    
    PHASE B REFACTOR: Now uses cascade-derived data from
    medical_knowledge/derived/disease_index_cascade.json instead of
    reading disease_0001.json common_symptoms (which is hardcoded).
    
    The cascade index has:
      - anchors/support: derived from organ_function.json cascade produces_symptom_*
      - system/triage/onset/must_not_miss/red_flags: supplemental from disease_0001
        (allowed per user spec — these fields cannot be derived from cascade)
    
    Returns: {disease_name → {system, anchors, support, triage_val, onset, must_not_miss}}
    """
    dis_index = {}
    
    # Prefer cascade-derived index (Phase B)
    parent = os.path.dirname(folder)
    cwd    = os.getcwd()
    cascade_paths = [
        os.path.join(cwd, "medical_knowledge", "derived", "disease_index_cascade.json"),
        os.path.join(parent, "derived", "disease_index_cascade.json"),
        os.path.join(folder, "..", "derived", "disease_index_cascade.json"),
    ]
    for p in cascade_paths:
        if os.path.exists(p):
            try:
                raw = json.load(open(p, encoding="utf-8"))
                # raw is {disease_name: {anchors, support, system, ...}}
                for name, entry in raw.items():
                    anchors_list = entry.get("anchors", [])
                    support_list = entry.get("support", [])
                    triage = entry.get("triage", "routine")
                    dis_index[name] = {
                        "system"       : (entry.get("system") or "").lower(),
                        "anchors"      : set(_norm_dis_sym(s) for s in anchors_list),
                        "support"      : set(_norm_dis_sym(s) for s in support_list),
                        "all_syms"     : set(_norm_dis_sym(s) for s in
                                              (anchors_list + support_list)),
                        "triage"       : triage,
                        "triage_val"   : _TRIAGE_URGENCY.get(triage, 0),
                        "onset"        : entry.get("onset"),
                        "must_not_miss": bool(entry.get("must_not_miss")),
                        "red_flags"    : entry.get("red_flags", []),
                        "category"     : entry.get("category", ""),
                    }
                if dis_index:
                    return dis_index   # Phase B: cascade-derived index, done
            except Exception as e:
                print(f"[evidence_gate] cascade index error {p}: {e}")
    
    # Legacy fallback — only if cascade index missing (shouldn't happen in prod)
    search_paths = [
        os.path.join(cwd, "medical_knowledge", "diseases", "disease_0001.json"),
        os.path.join(parent, "diseases", "disease_0001.json"),
        os.path.join(folder, "..", "diseases", "disease_0001.json"),
        os.path.join(cwd, "medical_knowledge", "disease_0001.json"),
        os.path.join(cwd, "disease_0001.json"),
        os.path.join(parent, "disease_0001.json"),
        os.path.join(cwd, "nexus_engine", "disease_0001.json"),
    ]
    paths = []
    for p in search_paths:
        if os.path.exists(p):
            paths.append(p)
            break
    for p in paths:
        try:
            raw = json.load(open(p, encoding="utf-8"))
            diseases = raw if isinstance(raw, list) else [raw]
            for dis in diseases:
                name = (dis.get("disease_name") or "").strip()
                if not name:
                    continue
                raw_syms = dis.get("common_symptoms") or []
                if not isinstance(raw_syms, list):
                    raw_syms = []
                norm_syms = [_norm_dis_sym(s) for s in raw_syms]
                mid = max(1, len(norm_syms) // 2)
                _mnm = dis.get("must_not_miss")
                if isinstance(_mnm, list):
                    _mnm_flag = len(_mnm) > 0
                else:
                    _mnm_flag = bool(_mnm)
                _rfs = dis.get("red_flags") or []
                if not isinstance(_rfs, list):
                    _rfs = []
                dis_index[name] = {
                    "system"       : (dis.get("system") or "").lower(),
                    "anchors"      : set(norm_syms[:mid]),
                    "support"      : set(norm_syms[mid:]),
                    "all_syms"     : set(norm_syms),
                    "triage"       : (dis.get("triage_level") or "routine").lower(),
                    "triage_val"   : _TRIAGE_URGENCY.get(
                                         (dis.get("triage_level") or "routine").lower(), 0),
                    "onset"        : dis.get("onset_pattern"),
                    "must_not_miss": _mnm_flag,
                    "red_flags"    : [(r or "").lower() for r in _rfs if isinstance(r, str)],
                    "category"     : dis.get("category", ""),
                }
        except Exception as e:
            print(f"[evidence_gate] disease file error {p}: {e}")
    return dis_index


# Minimal hard-fallback used only when symptom files are absent
_FALLBACK_SETS: dict = {
    "vague_standalone":  frozenset({
        "nausea", "vomiting", "dizziness", "headache", "weakness",
        "numbness", "tingling", "swelling", "abdominal pain", "chest pain",
    }),
    "system_anchors": {
        "respiratory":      {"cough", "shortness of breath"},
        "gastrointestinal": {"diarrhea", "constipation"},
        "cardiac":          {"palpitations", "shortness of breath"},
        "cutaneous":        {"rash", "itching", "redness"},
        "immune":           {"chills", "joint pain", "rash", "redness"},
        "musculoskeletal":  {"joint pain"},
        "thermoregulatory": {"chills"},
    },
    "syndrome_labels": {
        "respiratory": "respiratory syndrome",
        "gastrointestinal": "GI syndrome",
        "cardiac": "cardiovascular process",
        "cutaneous": "dermatologic process",
        "immune": "immune / allergic process",
    },
    "viral_anchors":    frozenset({"cough", "chills", "rash", "joint pain"}),
    "bacterial_anchors":frozenset({"chills", "diarrhea"}),
    "red_flag_syms":    frozenset({"chest pain", "shortness of breath", "headache"}),
    "sym_systems":      {},
}

# Module-level singleton — built once
_SETS: Optional[dict] = None

def _get_sets() -> dict:
    global _SETS
    if _SETS is None:
        # Search in common locations
        for folder in (
            "medical_knowledge/symptoms",
            "medical_knowledge/symptom",
            "symptoms",
            ".",
        ):
            if glob.glob(os.path.join(folder, "symptom_*.json")):
                _SETS = _build_sets_from_files(folder)
                break
        if _SETS is None:
            print("[evidence_gate] WARNING: symptom files not found — using fallback defaults")
            _SETS = _FALLBACK_SETS
    return _SETS


# ── Registry cache for _review_status lookup ────────────────────────────
# We load disease_mechanism_map.json once and cache it. evidence_gate uses
# this only to check whether a disease is `_review_status="unreviewed"` —
# if so, the gate refuses the disease label regardless of other checks.
# This is the safety mechanism for Session A's unreviewed-cascade workflow.
_REGISTRY: Optional[dict] = None


def _get_registry() -> dict:
    """Lazy-load disease_mechanism_map.json to check _review_status flags."""
    global _REGISTRY
    if _REGISTRY is None:
        candidates = [
            "medical_knowledge/registry/disease_mechanism_map.json",
            "../medical_knowledge/registry/disease_mechanism_map.json",
        ]
        for p in candidates:
            if os.path.exists(p):
                try:
                    _REGISTRY = json.load(open(p, encoding="utf-8")).get("mappings", {})
                    break
                except Exception:
                    continue
        if _REGISTRY is None:
            _REGISTRY = {}
    return _REGISTRY


# ══════════════════════════════════════════════════════════════════════
#  Data classes
# ══════════════════════════════════════════════════════════════════════

@dataclass
class EvidenceAssessment:
    # Layer 1
    triage_level : str  = "moderate"
    has_red_flags: bool = False

    # Layer 2
    output_state    : str       = "triage_only"   # "label"|"syndrome"|"triage_only"
    detected_systems: List[str] = field(default_factory=list)
    syndrome_label  : str       = ""

    # Layer 3
    etiology_allowed  : bool = False
    etiology_evidence : str  = ""
    etiology_level    : str  = "uncertain"   # "uncertain"|"leaning"|"confirmed"
    # "leaning"   = etiology_ok=True, viral/bacterial tendency can be mentioned
    # "confirmed"  = high confidence, enough to state clearly (not just suggest)

    # Layer 4
    disease_label_allowed: bool = False
    disease_gate_details : Dict = field(default_factory=dict)

    # Metadata
    all_vague    : bool  = False
    anchor_count : int   = 0
    support_count: int   = 0
    top_score    : float = 0.0
    top_margin   : float = 0.0


# ══════════════════════════════════════════════════════════════════════
#  Main gate function
# ══════════════════════════════════════════════════════════════════════

def assess_evidence(
    symptoms    : List[str],
    nexus_result: dict,
    thresholds  : dict = None,
) -> EvidenceAssessment:
    """
    Run all 4 evidence layers.

    Parameters
    ----------
    symptoms      current-turn symptom list (strings, any case)
    nexus_result  dict from nexus_medical.enhance_pipeline_result()
    thresholds    optional override from nexus_config.json["thresholds"]
    """
    sets = _get_sets()
    VAGUE   = sets["vague_standalone"]
    SYS_A   = sets["system_anchors"]
    SYN_L   = sets["syndrome_labels"]
    VIRAL   = sets["viral_anchors"]
    BACT    = sets["bacterial_anchors"]

    t  = thresholds or {}
    dr = t.get("disease_ranking", {})
    et = t.get("etiology", {})

    DISEASE_MIN_SCORE  = dr.get("disease_label_min_score",  0.72)
    DISEASE_MIN_MARGIN = dr.get("disease_label_min_margin", 0.15)
    MIN_MECH           = et.get("min_mech_activations",      5)

    sym_set = {s.lower().replace("_", " ").strip() for s in symptoms if s}
    ea = EvidenceAssessment()

    # ── Layer 1: Triage / Safety ──────────────────────────────────────────────
    triage = nexus_result.get("triage") or {}
    ea.triage_level  = (triage.get("level") or "MODERATE").lower()
    ea.has_red_flags = bool(
        (nexus_result.get("red_flag_block") or {}).get("red_flags") or
        nexus_result.get("nexus_red_flags")
    )

    # ── High-risk symptom freeze ──────────────────────────────────────────────
    # When an A-observed red-flag symptom is present, it takes priority over
    # system scoring. Syndrome is overridden to reflect the dominant safety concern,
    # and etiology + disease label are suppressed (not enough specificity when
    # triage is driven by a red-flag symptom).
    _HIGH_RISK_SYMS = {
        "chest pain":           "mixed high-risk chest-pain syndrome",
        "shortness of breath":  "mixed respiratory / cardiopulmonary syndrome",
        "syncope":              "high-risk syncope / loss of consciousness syndrome",
        "fainting":             "high-risk syncope / loss of consciousness syndrome",
        "altered mental status":"high-risk neurologic / altered consciousness syndrome",
        "confusion":            "high-risk neurologic / altered consciousness syndrome",
        "focal weakness":       "high-risk neurologic syndrome — rule out stroke",
        "slurred speech":       "high-risk neurologic syndrome — rule out stroke",
        "worst headache":       "high-risk headache syndrome — rule out SAH",
        "thunderclap headache": "high-risk headache syndrome — rule out SAH",
        "severe abdominal pain":"high-risk abdominal syndrome",
        "hematemesis":          "high-risk GI bleeding syndrome",
        "hemoptysis":           "high-risk respiratory / bleeding syndrome",
        "bloody stool":         "high-risk GI bleeding syndrome",
        "melena":               "high-risk GI bleeding syndrome",
    }

    # Check which (if any) high-risk symptoms are present
    _rfd            = nexus_result.get("red_flag_block") or {}
    _observed_names = {r.get("name","").lower() for r in (_rfd.get("observed_red_flags") or [])}
    _high_risk_present = [s for s in sym_set if s in _HIGH_RISK_SYMS]
    # Also catch if symptom matches an observed_red_flag rule name
    _freeze = bool(_high_risk_present)

    if _freeze:
        # Use the most dangerous symptom's syndrome label
        _priority = ["chest pain","shortness of breath","syncope","fainting",
                     "hematemesis","hemoptysis","bloody stool","melena",
                     "altered mental status","confusion","focal weakness",
                     "slurred speech","worst headache","thunderclap headache",
                     "severe abdominal pain"]
        _dominant = next((s for s in _priority if s in _high_risk_present), _high_risk_present[0])
        ea.output_state     = "syndrome"
        ea.syndrome_label   = _HIGH_RISK_SYMS[_dominant]
        ea.detected_systems = []
        ea.etiology_allowed = False
        ea.etiology_evidence= f"frozen: high-risk symptom '{_dominant}' present — etiology not applicable"
        ea.etiology_level   = "uncertain"
        ea.disease_label_allowed = False
        # Skip Layers 2, 3, 4 — return early after writing result
        ea.disease_gate_details = {
            "gate_passed": False, "frozen_by": _dominant,
            "reason": "high-risk symptom overrides evidence gate",
        }
        return ea

    # ── Layer 2: System / Syndrome ────────────────────────────────────────────
    all_vague  = bool(sym_set) and all(s in VAGUE for s in sym_set)
    ea.all_vague = all_vague

    # System priority for tie-breaking (GI and respiratory most common in acute illness)
    _SYS_PRIORITY = {
        "gastrointestinal": 10, "respiratory": 9, "cardiac": 8,
        "cutaneous": 7, "musculoskeletal": 6, "neurologic": 5,
        "immune": 4, "urinary": 4, "reproductive": 3,
        "thermoregulatory": 2, "circulatory": 2, "vascular": 1,
    }

    # Weighted system scoring:
    #   non-vague anchor in system  → 2 points  (discriminating)
    #   vague symptom mapped to sys → 1 point   (contextual — vomiting supports GI)
    # This correctly weights diarrhea+vomiting > cough for primary system.
    sym_all_sys = sets.get("sym_systems", {})
    scored: Dict[str, tuple] = {}
    for system, anchors in SYS_A.items():
        non_vague_hits   = sym_set & anchors
        all_in_sys       = {s for s in sym_set if system in sym_all_sys.get(s, [])}
        vague_contextual = all_in_sys - non_vague_hits
        total = len(non_vague_hits) * 2 + len(vague_contextual)
        if total > 0:
            scored[system] = (total, non_vague_hits, vague_contextual)

    nexus_systems = [s.lower().strip() for s in (nexus_result.get("nexus_detected_systems") or [])]

    if scored and not all_vague:
        primary = max(scored, key=lambda s: (scored[s][0], _SYS_PRIORITY.get(s, 0)))
        _, _primary_non_vague, _ = scored[primary]

        # ── Multi-system blending ────────────────────────────────────────────
        # When 2+ systems have meaningful scores, don't let one system dominate.
        # Use a blended label that reflects the breadth of the presentation.
        _sorted_sys  = sorted(scored.keys(), key=lambda s: -scored[s][0])
        _top_score   = scored[primary][0]
        _second_sys  = [s for s in _sorted_sys if s != primary]
        _close_sys   = [s for s in _second_sys
                        if scored[s][0] >= max(1, _top_score * 0.5)]

        # Friendly name map for blended labels
        _SYS_SHORT = {
            "gastrointestinal": "GI", "respiratory": "respiratory",
            "cardiovascular": "cardiac", "cutaneous": "skin",
            "neurologic": "neurologic", "musculoskeletal": "musculoskeletal",
            "urinary": "urinary", "thermoregulatory": "systemic",
            "immune": "immune", "circulatory": "circulatory",
        }

        if _close_sys:
            # Multiple systems — build blended label
            _primary_short  = _SYS_SHORT.get(primary, primary)
            _secondary_short = _SYS_SHORT.get(_close_sys[0], _close_sys[0])
            if _top_score == scored[_close_sys[0]][0]:
                # Tied scores — equal blend
                ea.syndrome_label = (f"mixed {_primary_short}/{_secondary_short} "
                                     f"syndrome pattern")
            else:
                # Primary leads but secondary is significant
                ea.syndrome_label = (f"{SYN_L.get(primary, f'{primary} syndrome')} "
                                     f"with {_secondary_short} features")
        else:
            # Single dominant system
            ea.syndrome_label = SYN_L.get(primary, f"{primary} syndrome")

        ea.detected_systems = _sorted_sys[:4]
        ea.output_state     = "syndrome"
        _primary_sys        = primary
    elif nexus_systems and not all_vague:
        primary             = nexus_systems[0]
        ea.syndrome_label   = SYN_L.get(primary, f"{primary} syndrome")
        ea.detected_systems = nexus_systems[:4]
        ea.output_state     = "syndrome"
        _primary_non_vague  = set()
        _primary_sys        = primary
    else:
        ea.output_state     = "triage_only"
        ea.syndrome_label   = ""
        ea.detected_systems = nexus_systems[:4]
        _primary_non_vague  = set()
        _primary_sys        = ""

    # ── Layer 3: Etiology ─────────────────────────────────────────────────────
    # Reviewer spec: require DISCRIMINATING anchors, not broad symptom-level ones.
    # "cough, diarrhea, vomiting" are all viral anchors in symptom files but they
    # don't discriminate viral from food poisoning / medication / noninfectious.
    mechs_activated = (nexus_result.get("nexus_stats") or {}).get("mechanisms_activated", 0)
    etiol           = nexus_result.get("nexus_etiology") or {}
    etiol_type      = etiol.get("etiology", "uncertain")
    etiol_confidence= float(etiol.get("confidence", 0.0))

    # Broad anchors (from symptom files — used for system scoring only)
    viral_hits = len(sym_set & VIRAL)
    bact_hits  = len(sym_set & BACT)

    # Discriminating anchors — load from etiology_anchors.json
    try:
        import json as _eaj, os as _eaos
        _ea_data = {}
        for _ea_path in ("nexus_engine/etiology_anchors.json", "nexus_engine/etiology_anchors.json"):
            if _eaos.path.exists(_ea_path):
                _ea_data = _eaj.load(open(_ea_path, encoding="utf-8"))
                break
        _DISC_VIRAL = set(_ea_data.get("discriminating_viral", [
            "fever", "chills", "myalgia", "body aches",
            "sore throat", "runny nose", "nasal congestion", "headache",
            "sick contact", "pharyngitis",
        ]))
        _DISC_BACT = set(_ea_data.get("discriminating_bacterial", [
            "purulent discharge", "focal pain", "dysuria", "urinary frequency",
            "flank pain", "productive cough", "high fever", "night sweats",
            "localized swelling", "lymphadenopathy",
        ]))
    except Exception:
        _DISC_VIRAL = {"fever", "chills", "myalgia", "body aches", "fatigue",
                       "sore throat", "runny nose", "nasal congestion", "sneezing",
                       "headache", "sick contact", "uri prodrome", "pharyngitis"}
        _DISC_BACT  = {"purulent discharge", "focal pain", "dysuria", "urinary frequency",
                       "flank pain", "productive cough", "high fever", "night sweats",
                       "localized swelling", "lymphadenopathy"}
    viral_specific = len(sym_set & _DISC_VIRAL)
    bact_specific  = len(sym_set & _DISC_BACT)

    # Threshold config
    _thresholds_et = (thresholds or {}).get("etiology", {})
    _et_min        = _thresholds_et.get("viral_min",      0.70)   # "confirmed" floor
    _et_lean       = _thresholds_et.get("uncertain_below", 0.50)  # "leaning" floor
    _disc_required = 2   # discriminating anchors needed to allow etiology output

    # Dominant discriminating anchor count for the classified etiology
    _disc_count = viral_specific if etiol_type == "viral" else bact_specific

    if all_vague:
        ea.etiology_allowed  = False
        ea.etiology_level    = "uncertain"
        ea.etiology_evidence = "all vague — etiology gate blocked"

    elif mechs_activated <= MIN_MECH and viral_specific == 0 and bact_specific == 0:
        ea.etiology_allowed  = False
        ea.etiology_level    = "uncertain"
        ea.etiology_evidence = (
            f"no discriminating pathogen evidence: mechs={mechs_activated}, "
            f"viral_disc={viral_specific}, bact_disc={bact_specific}"
        )

    elif etiol_type == "uncertain" or etiol_confidence < _et_lean:
        ea.etiology_allowed  = False
        ea.etiology_level    = "uncertain"
        ea.etiology_evidence = (
            f"classifier uncertain or confidence too low: "
            f"{etiol_type} confidence={etiol_confidence:.2f} < {_et_lean}"
        )

    elif _disc_count < _disc_required:
        # Broad symptoms present but no specific discriminating anchors
        # → internal lean only, not for output
        ea.etiology_allowed  = False
        ea.etiology_level    = "weak_lean"
        ea.etiology_evidence = (
            f"insufficient discriminating anchors: {etiol_type} "
            f"disc_count={_disc_count} (need ≥{_disc_required}), "
            f"confidence={etiol_confidence:.2f}"
        )

    elif etiol_confidence >= _et_min and _disc_count >= _disc_required and mechs_activated > 10:
        ea.etiology_allowed  = True
        ea.etiology_level    = "confirmed"
        ea.etiology_evidence = (
            f"strong: {etiol_type} conf={etiol_confidence:.2f}, "
            f"disc_anchors={_disc_count}, mechs={mechs_activated}"
        )

    else:
        ea.etiology_allowed  = True
        ea.etiology_level    = "leaning"
        ea.etiology_evidence = (
            f"moderate: {etiol_type} conf={etiol_confidence:.2f}, "
            f"disc_anchors={_disc_count}"
        )

    # ── Layer 4: Disease Label ────────────────────────────────────────────────
    nexus_dx = nexus_result.get("nexus_diagnoses") or []
    if len(nexus_dx) >= 2:
        top_score  = float(nexus_dx[0].get("score", 0))
        top_margin = top_score - float(nexus_dx[1].get("score", 0))
    elif len(nexus_dx) == 1:
        top_score  = float(nexus_dx[0].get("score", 0))
        top_margin = top_score
    else:
        top_score = top_margin = 0.0

    ea.top_score  = top_score
    ea.top_margin = top_margin

    # ── Disease contract check (using disease_0001.json) ────────────────────
    # For the top NEXUS disease, verify its defining symptoms are actually present.
    # This replaces opaque score thresholds with transparent symptom matching.
    DIS_INDEX = sets.get("disease_index", {})

    dis_anchor_count    = 0
    dis_support_count   = 0
    dis_system_match    = False
    dis_onset_ok        = True
    top_dis_name        = nexus_dx[0].get("disease", "") if nexus_dx else ""
    must_not_miss_flag  = False
    triage_upgrade      = None

    if top_dis_name and top_dis_name in DIS_INDEX:
        entry = DIS_INDEX[top_dis_name]
        # Normalize sym_set for disease matching
        _norm_set = {_norm_dis_sym(s) for s in sym_set}

        dis_anchor_count  = len(_norm_set & entry["anchors"])
        dis_support_count = len(_norm_set & entry["support"])

        # ── System match: check membership in ANY detected system, not just primary ──
        # Previous bug: only checked == _primary_sys (exact equality, single system).
        # That failed when:
        #   1. Patient symptoms produce multi-system detection (e.g. flank pain
        #      maps to "musculoskeletal" but disease is "urinary") — both present
        #      but primary was the wrong one.
        #   2. Naming variants between cascade index ("dermatologic", "renal")
        #      and evidence_gate system map ("cutaneous", "urinary").
        # Fix: check entry["system"] against the FULL set of detected systems
        # (primary + nexus_systems), with name normalization.
        _SYS_ALIASES = {
            "dermatologic": {"cutaneous", "dermatologic", "integumentary"},
            "cutaneous":    {"cutaneous", "dermatologic", "integumentary"},
            "integumentary":{"cutaneous", "dermatologic", "integumentary"},
            "urinary":      {"urinary", "renal", "genitourinary"},
            "renal":        {"urinary", "renal", "genitourinary"},
            "gastrointestinal": {"gastrointestinal", "gi", "digestive"},
            "gi":           {"gastrointestinal", "gi", "digestive"},
            "cardiovascular":{"cardiovascular", "cardiac", "circulatory"},
            "neurologic":   {"neurologic", "neurological", "cns"},
            "respiratory":  {"respiratory", "pulmonary"},
            "immune":       {"immune", "hematologic", "lymphatic", "immunologic"},
            "hematologic":  {"immune", "hematologic", "lymphatic"},
            "musculoskeletal":{"musculoskeletal", "msk"},
            "msk":          {"musculoskeletal", "msk"},   # NEW: atlas uses "msk"
            "endocrine":    {"endocrine", "metabolic"},
            "hepatobiliary":{"hepatobiliary", "hepatic"},
            "reproductive": {"reproductive", "gynecologic"},
            "sensory":      {"sensory", "ophthalmologic", "otologic", "special_sense"},  # NEW
        }
        entry_sys = (entry["system"] or "").lower()
        entry_sys_set = _SYS_ALIASES.get(entry_sys, {entry_sys}) if entry_sys else set()
        # All systems detected for this case: primary + multi-system + nexus reasoner
        all_detected = set()
        if _primary_sys:
            all_detected.add(_primary_sys.lower())
            all_detected |= _SYS_ALIASES.get(_primary_sys.lower(), set())
        for s in nexus_systems:
            all_detected.add(s.lower())
            all_detected |= _SYS_ALIASES.get(s.lower(), set())
        # Also include all systems from scored dict (multi-system blending)
        for s in scored.keys():
            all_detected.add(s.lower())
            all_detected |= _SYS_ALIASES.get(s.lower(), set())

        if not entry_sys:
            # Disease has no system declared — fall back to permissive
            dis_system_match = True
        elif not all_detected:
            # No system detected — permissive (only score_gate will guard label)
            dis_system_match = True
        else:
            dis_system_match = bool(entry_sys_set & all_detected)

        # Onset compatibility: chronic disease should not rank for acute presentation
        _user_onset = (nexus_result.get("case_context") or {}).get("onset", "")
        if entry["onset"] == "chronic" and _user_onset in ("acute", "sudden"):
            dis_onset_ok = False

        # Triage upgrade from disease file
        if entry["triage_val"] >= 3:   # urgent or above
            triage_upgrade = entry["triage"]

    # Must-not-miss: check ALL diseases in index against symptoms
    for dis_name, entry in DIS_INDEX.items():
        if entry.get("must_not_miss") and entry.get("triage_val", 0) >= 4:
            _norm_set = {_norm_dis_sym(s) for s in sym_set}
            if len(_norm_set & entry["anchors"]) >= 1:
                must_not_miss_flag = True
                break

    # System-level anchors (from symptom files) — still count for fallback
    anchor_syms  : Set[str] = _primary_non_vague if nexus_dx else set()
    support_syms : Set[str] = (sym_set - anchor_syms - VAGUE) if nexus_dx else set()

    ea.anchor_count  = max(len(anchor_syms), dis_anchor_count)
    ea.support_count = max(len(support_syms), dis_support_count)

    # Disease label gate — now uses BOTH score threshold AND disease contract:
    # Either: score >= 0.72 + margin >= 0.15 + system_anchor_count >= 2
    # Or:     disease contract match (dis_anchor_count >= 2 + system match)
    score_gate = (
        top_score  >= DISEASE_MIN_SCORE
        and top_margin >= DISEASE_MIN_MARGIN
        and len(anchor_syms) >= 2
    )
    contract_gate = (
        dis_anchor_count >= 2
        and dis_system_match
        and dis_onset_ok
        and top_score >= dr.get("contract_floor", 0.5)   # lower floor when contract is solid
        and top_margin >= 0.1
    )
    gate_ok = (
        not all_vague
        and (score_gate or contract_gate)
        and ea.output_state in ("syndrome", "label")
    )

    # Triage upgrade from disease knowledge
    if triage_upgrade and gate_ok:
        _cur_val = {"routine": 0, "moderate": 1, "prompt": 2,
                    "urgent": 3, "emergency": 4}.get(ea.triage_level, 1)
        _new_val = _TRIAGE_URGENCY.get(triage_upgrade, 0)
        if _new_val > _cur_val:
            ea.triage_level = triage_upgrade

    # ── Safety gate: unreviewed disease cascades ─────────────────────────
    # If the top diagnosis's cascade has _review_status="unreviewed" in the
    # registry, REFUSE the disease label regardless of other gates.
    # This safety check runs even when gate_ok is already False, so that
    # the trace explicitly records why the label is blocked.
    review_status_blocked = False
    if top_dis_name:
        reg = _get_registry()
        reg_entry = reg.get(top_dis_name, {})
        if reg_entry.get("_review_status") == "unreviewed":
            review_status_blocked = True
            gate_ok = False   # downgrade — label refused

    ea.disease_label_allowed = gate_ok
    if gate_ok:
        ea.output_state = "label"
    elif review_status_blocked:
        # Make the reason explicit in details so proof block can show it
        if ea.output_state == "triage_only":
            ea.output_state = "syndrome"   # at least give system context
        # Note: keep top_dis_name in details but flag the block
        ea.syndrome_label = (ea.syndrome_label or
                             f"{ (reg_entry.get('organ') or 'process').replace('_', ' ') } process — diagnosis pending medical review")

    ea.disease_gate_details = {
        "top_score"          : round(top_score,  3),
        "top_margin"         : round(top_margin, 3),
        "anchor_count"       : ea.anchor_count,
        "support_count"      : ea.support_count,
        "anchor_syms"        : sorted(anchor_syms),
        "support_syms"       : sorted(support_syms),
        "dis_anchor_count"   : dis_anchor_count,
        "dis_support_count"  : dis_support_count,
        "dis_system_match"   : dis_system_match,
        "dis_onset_ok"       : dis_onset_ok,
        "contract_gate"      : contract_gate if nexus_dx else False,
        "score_gate"         : score_gate if nexus_dx else False,
        "must_not_miss"      : must_not_miss_flag,
        "all_vague"          : all_vague,
        "gate_passed"        : gate_ok,
        "top_disease"        : top_dis_name,
        "min_score_req"      : DISEASE_MIN_SCORE,
        "min_margin_req"     : DISEASE_MIN_MARGIN,
        "review_status_blocked": review_status_blocked,
    }

    return ea


# ══════════════════════════════════════════════════════════════════════
#  Apply gate → result dict
# ══════════════════════════════════════════════════════════════════════

def apply_gate_to_result(result: dict, ea: EvidenceAssessment) -> dict:
    """Write evidence assessment back into result dict for downstream modules."""
    result["evidence_assessment"] = {
        "output_state"          : ea.output_state,
        "syndrome_label"        : ea.syndrome_label,
        "detected_systems"      : ea.detected_systems,
        "etiology_allowed"      : ea.etiology_allowed,
        "etiology_evidence"     : ea.etiology_evidence,
        "etiology_level"        : ea.etiology_level,      # "uncertain"|"leaning"|"confirmed"
        "disease_label_allowed" : ea.disease_label_allowed,
        "disease_gate_details"  : ea.disease_gate_details,
        "all_vague"             : ea.all_vague,
        "triage_level"          : ea.triage_level,
        "has_red_flags"         : ea.has_red_flags,
    }

    # Suppress etiology when gate says no
    if not ea.etiology_allowed:
        etiol = result.get("nexus_etiology") or {}
        if etiol.get("etiology") not in ("uncertain", None):
            result["nexus_etiology"] = {
                **etiol,
                "etiology"   : "uncertain",
                "confidence" : 0.0,
                "scores"     : {"viral": 0.0, "bacterial": 0.0, "non_infectious": 0.0},
                "_suppressed_by": "evidence_gate: " + ea.etiology_evidence,
            }

    # Disease display: patient sees disease name only when label gate passes
    result["nexus_diagnoses_display"] = (
        result.get("nexus_diagnoses", [])[:3] if ea.disease_label_allowed else []
    )

    return result