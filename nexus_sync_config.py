#!/usr/bin/env python3
"""
NEXUS 配置自动同步
════════════════════════════════════════════════════════════════
读取现有的 NEXUS 知识库文件，自动生成用于训练的 nexus_config.json。

工作方式：
  1. 扫描 disease_0001.json            → 疾病列表 + 症状
  2. 扫描 symptom_XXXX.json 文件        → 症状权重 + 红旗症状
  3. 扫描 *mechanism*.json 文件         → 病毒/细菌锚定词 + 噪声症状
  4. 与你的手动配置合并                  → 最终 config

添加新疾病文件后运行一次即可：
    python nexus_engine/nexus_sync_config.py
    python nexus_engine/nexus_sync_config.py --knowledge-dir .
    python nexus_engine/nexus_sync_config.py --dry-run   # 预览，不写入文件

生成的 nexus_config.json 供 nexus_learning_env_v2.py 训练使用，无需修改其他文件。
"""

import argparse, glob, json, os, re, sys
from pathlib import Path
from typing import Dict, List, Set

# ─────────────────────────────────────────────────────────────
# DEFAULTS  (used when NEXUS knowledge can't determine a value)
# ─────────────────────────────────────────────────────────────

# 按疾病名关键词匹配的备用治疗方案
_TREATMENT_HINTS: Dict[str, List[str]] = {
    "pneumonia":        ["antibiotics", "oxygen therapy"],
    "flu":              ["antivirals", "rest", "fluids"],
    "influenza":        ["antivirals", "rest", "fluids"],
    "appendicitis":     ["surgery", "antibiotics"],
    "meningitis":       ["antibiotics", "steroids", "iv fluids"],
    "heart attack":     ["aspirin", "pci", "nitroglycerin"],
    "mi":               ["aspirin", "pci", "nitroglycerin"],
    "myocardial":       ["aspirin", "pci", "nitroglycerin"],
    "gastroenteritis":  ["fluids", "rest", "antiemetics"],
    "sepsis":           ["antibiotics", "iv fluids", "vasopressors"],
    "asthma":           ["inhaler", "bronchodilators", "steroids"],
    "migraine":         ["triptans", "nsaids", "rest"],
    "covid":            ["antivirals", "oxygen therapy", "rest"],
    "stroke":           ["tpa", "aspirin", "rehabilitation"],
    "diabetes":         ["insulin", "metformin", "lifestyle"],
    "hypertension":     ["ace inhibitors", "beta blockers", "lifestyle"],
    "pneumothorax":     ["chest tube", "oxygen therapy", "rest"],
    "pulmonary embolism": ["anticoagulants", "oxygen therapy", "thrombolytics"],
    "dvt":              ["anticoagulants", "compression", "elevation"],
    "anaphylaxis":      ["epinephrine", "antihistamines", "steroids"],
    "cellulitis":       ["antibiotics", "elevation", "fluids"],
    "pericarditis":     ["nsaids", "colchicine", "rest"],
}

# 按病原体/疾病类型推断严重程度
_SEVERITY_HINTS = {
    "non-infectious": "moderate",
    "bacteria": "severe",
    "virus": "moderate",
    "fungal": "moderate",
}

_CRITICAL_DISEASES = {
    "sepsis", "meningitis", "heart attack", "stroke", "anaphylaxis",
    "pulmonary embolism", "pneumothorax", "aortic dissection", "pe",
}

# 基础症状权重（诊断信号强度，来自训练经验）
_BASE_SYMPTOM_WEIGHTS = {
    "left arm pain":        2.5,
    "stiff neck":           2.5,
    "loss of smell":        2.5,
    "confusion":            2.0,
    "wheezing":             2.0,
    "sensitivity to light": 1.8,
    "chest pain":           1.8,
    "abdominal pain":       1.8,
    "diarrhea":             1.8,
    "body aches":           1.8,
    "headache":             1.5,
    "sweating":             1.5,
    "shortness of breath":  1.4,
    "weakness":             1.3,
    "dizziness":            1.3,
    # Additional high-value symptoms
    "facial drooping":      2.5,
    "sudden numbness":      2.5,
    "speech difficulty":    2.5,
    "arm weakness":         2.0,
    "jaw pain":             2.0,
    "loss of taste":        2.0,
    "neck stiffness":       2.5,
    "photophobia":          1.8,
    "altered consciousness": 2.5,
    "rash":                 1.5,
    "hypotension":          2.0,
}

# 基础症状组合规则
_BASE_COMBOS = {
    "mi_definite":      {"requires": ["chest pain", "left arm pain"],    "excludes": [],                              "value": 2.0},
    "mi_probable":      {"requires": ["chest pain", "sweating"],         "excludes": [],                              "value": 1.5},
    "meningitis_full":  {"requires": ["headache", "stiff neck", "fever"],"excludes": [],                              "value": 2.5},
    "meningitis_part":  {"requires": ["headache", "stiff neck"],         "excludes": [],                              "value": 2.0},
    "sepsis_combo":     {"requires": ["fever", "confusion"],             "excludes": [],                              "value": 1.8},
    "asthma_attack":    {"requires": ["wheezing", "shortness of breath"],"excludes": [],                              "value": 2.0},
    "gi_not_cardiac":   {"requires": ["abdominal pain", "nausea"],       "excludes": ["chest pain","left arm pain"],  "value": 1.5},
    "covid_smell":      {"requires": ["loss of smell"],                  "excludes": [],                              "value": 2.5},
    "flu_triad":        {"requires": ["body aches", "headache", "fever"],"excludes": [],                              "value": 1.5},
    "stroke_classic":   {"requires": ["facial drooping", "arm weakness"],"excludes": [],                              "value": 2.5},
}


# ─────────────────────────────────────────────────────────────
# READERS
# ─────────────────────────────────────────────────────────────

def read_disease_files(knowledge_dir: str) -> List[dict]:
    """
    从 disease_0001.json 或 medical_knowledge/diseases/ 读取疾病数据
    and medical_knowledge/diseases_pro/ (matching your NEXUS structure).

    Each file is expected to have:
        disease_name:     str
        common_symptoms:  list[str]
        red_flags:        list[str]       (optional)
        treatments:       list[str]       (optional — we fall back to hints)
        pathogen_type:    str             (optional)
        infected_organ:   str             (optional)
        severity:         str             (optional)
    """
    diseases = []
    seen_names: Set[str] = set()

    folders = ["diseases", "diseases_pro", "symptoms"]
    for folder in folders:
        pattern = os.path.join(knowledge_dir, folder, "*.json")
        for fpath in sorted(glob.glob(pattern)):
            try:
                d = json.load(open(fpath, encoding="utf-8"))
            except Exception as e:
                print(f"  [警告] 无法读取文件 {fpath}: {e}")
                continue

            if not isinstance(d, dict):
                continue

            name = d.get("disease_name", "").strip().lower()
            if not name or name in seen_names:
                continue

            # Get common symptoms
            symptoms = [s.lower().strip() for s in d.get("common_symptoms", [])
                       if isinstance(s, str) and s.strip()]
            if not symptoms:
                continue  # skip entries with no symptoms

            # Severity
            severity = d.get("severity", "").lower()
            if not severity:
                if any(k in name for k in _CRITICAL_DISEASES):
                    severity = "critical"
                else:
                    severity = _SEVERITY_HINTS.get(
                        d.get("pathogen_type", "").lower(), "moderate"
                    )

            # Treatments — prefer explicit, fall back to hints
            treatments = d.get("treatments", d.get("treatment_options", []))
            if not treatments:
                for hint_key, hint_tx in _TREATMENT_HINTS.items():
                    if hint_key in name:
                        treatments = hint_tx
                        break
            if not treatments:
                treatments = ["rest", "fluids"]  # safe fallback

            diseases.append({
                "name":           name,
                "symptoms":       symptoms[:8],   # cap at 8 to keep clean boundaries
                "severity":       severity,
                "pathogen":       d.get("pathogen_type", d.get("pathogen", "unknown")).lower(),
                "infected_organ": d.get("infected_organ", d.get("primary_organ", "unknown")).lower(),
                "treatments":     [t.lower().strip() for t in treatments if isinstance(t, str)],
                "_source_file":   os.path.basename(fpath),
            })
            seen_names.add(name)

    return diseases


def read_symptom_weights_from_nexus(nexus_medical_path: str) -> Dict[str, float]:
    """
    返回基础症状权重。
    完整权重构建由 build_symptom_weights() 处理
    using data from read_symptom_files() and read_mechanism_files().
    """
    return dict(_BASE_SYMPTOM_WEIGHTS)
def collect_all_treatments(diseases: List[dict]) -> List[str]:
    """Collect all unique treatments from all diseases, sorted."""
    base = [
        "antibiotics", "antivirals", "antifungals",
        "oxygen therapy", "iv fluids", "fluids", "rest",
        "aspirin", "steroids", "nsaids", "triptans",
        "inhaler", "bronchodilators", "surgery",
        "pci", "nitroglycerin", "vasopressors",
        "antiemetics", "analgesics",
        # Extended
        "tpa", "anticoagulants", "rehabilitation",
        "insulin", "metformin",
        "epinephrine", "antihistamines",
        "ace inhibitors", "beta blockers",
        "colchicine", "compression",
    ]
    extra = set()
    for d in diseases:
        for t in d["treatments"]:
            if t not in base:
                extra.add(t)
    return base + sorted(extra)


def build_combo_signals(diseases: List[dict]) -> Dict[str, dict]:
    """
    为新疾病自动生成症状组合信号。 based on their
    symptom sets. High-weight symptoms become combo triggers.
    """
    combos = dict(_BASE_COMBOS)

    # Auto-generate combos for diseases with 2+ high-weight unique symptoms
    all_symptom_sets = {d["name"]: set(d["symptoms"]) for d in diseases}

    for disease in diseases:
        name    = disease["name"]
        syms    = set(disease["symptoms"])
        hi_syms = [s for s in syms if _BASE_SYMPTOM_WEIGHTS.get(s, 1.0) >= 2.0]

        if len(hi_syms) >= 2:
            # Find other diseases' symptoms to use as excludes
            others_syms = set()
            for other_name, other_syms in all_symptom_sets.items():
                if other_name != name:
                    others_syms |= (other_syms - syms)

            combo_key = name.replace(" ", "_") + "_hi"
            if combo_key not in combos:
                combos[combo_key] = {
                    "requires": hi_syms[:2],
                    "excludes": [],
                    "value":    2.0,
                    "description": f"Auto: {name} high-signal combo",
                }

    return combos



# ─────────────────────────────────────────────────────────────
# PROJECT ADAPTER  (reads disease_0001.json + symptom_XXXX.json)
# ─────────────────────────────────────────────────────────────

def read_from_project_root(project_dir: str) -> List[dict]:
    """
    Read diseases from disease_0001.json (array format used in this project).
    Falls back to read_disease_files() if diseases/ folder exists.
    """
    import glob as _glob
    diseases = []
    seen: Set[str] = set()

    # Try disease_0001.json first (array of disease objects)
    single_file = os.path.join(project_dir, "disease_0001.json")
    if os.path.exists(single_file):
        raw = json.load(open(single_file, encoding="utf-8"))
        if isinstance(raw, list):
            for d in raw:
                name = d.get("disease_name", "").strip().lower()
                if not name or name in seen:
                    continue
                symptoms = [s.lower().strip() for s in d.get("common_symptoms", [])
                           if isinstance(s, str) and s.strip()]
                if not symptoms:
                    continue
                # Map triage_level to severity
                triage = d.get("triage_level", "").lower()
                severity_map = {
                    "emergent": "critical", "ed": "critical",
                    "urgent_care": "severe", "urgent": "severe",
                    "moderate": "moderate", "routine": "mild",
                }
                severity = severity_map.get(triage, "moderate")
                # Get treatment
                tx_block = d.get("treatment", {})
                treatments = []
                if isinstance(tx_block, dict):
                    treatments = tx_block.get("home_care", [])
                elif isinstance(tx_block, list):
                    treatments = tx_block
                # Fallback to hints
                if not treatments:
                    for hint_key, hint_tx in _TREATMENT_HINTS.items():
                        if hint_key in name:
                            treatments = hint_tx
                            break
                if not treatments:
                    treatments = ["rest", "fluids"]
                diseases.append({
                    "name":           name,
                    "symptoms":       symptoms[:8],
                    "severity":       severity,
                    "pathogen":       d.get("pathogen_type", "unknown").lower(),
                    "infected_organ": d.get("system", "unknown").lower(),
                    "treatments":     [t.lower().strip() for t in treatments
                                      if isinstance(t, str)],
                    "_source_file":   "disease_0001.json",
                })
                seen.add(name)
            print(f"[SYNC] 从 disease_0001.json 读取 {len(diseases)} 个疾病")
    return diseases


# ─────────────────────────────────────────────────────────────
# SYMPTOM FILE READER
# ─────────────────────────────────────────────────────────────

def read_symptom_files(project_dir: str) -> dict:
    """
    Scan symptom_XXXX.json files and extract:
      - symptom → systems mapping
      - symptom → red_flags (boosts weight)
      - symptom → related_symptoms (for combo suggestions)
      - All symptom names (for noise_symptoms list)

    Returns dict with:
      "symptom_systems":   {symptom: [systems]}
      "red_flag_syms":     set of symptom names that have red flags
      "related":           {symptom: [related_symptoms]}
      "all_names":         [all symptom names found]
    """
    import glob as _glob

    result = {
        "symptom_systems": {},
        "red_flag_syms":   set(),
        "related":         {},
        "all_names":       [],
    }

    # Search in project_dir and common subfolders
    patterns = [
        os.path.join(project_dir, "symptom_*.json"),
        os.path.join(project_dir, "symptoms", "symptom_*.json"),
        os.path.join(project_dir, "medical_knowledge", "symptoms", "symptom_*.json"),
    ]
    found = []
    for pat in patterns:
        found.extend(_glob.glob(pat))

    if not found:
        print(f"  [警告] 未找到症状文件 symptom_*.json，路径：{project_dir}")
        return result

    print(f"[SYNC] 读取 {len(found)} 个症状文件，路径：{project_dir}")
    for fpath in sorted(found):
        try:
            d = json.load(open(fpath, encoding="utf-8"))
        except Exception as e:
            print(f"  [WARN] Could not read {fpath}: {e}")
            continue

        name = d.get("symptom", "").lower().strip()
        if not name:
            continue

        result["all_names"].append(name)
        result["symptom_systems"][name] = d.get("systems", [])
        result["related"][name]         = [r.lower().strip()
                                           for r in d.get("related_symptoms", [])]
        if d.get("red_flags"):
            result["red_flag_syms"].add(name)

    print(f"  → {len(result['all_names'])} 个症状，"
          f"{len(result['red_flag_syms'])} 个含红旗警示")
    return result


# ─────────────────────────────────────────────────────────────
# MECHANISM FILE READER
# ─────────────────────────────────────────────────────────────

def read_mechanism_files(project_dir: str) -> dict:
    """
    Scan mechanism JSON files and extract:
      - symptom coverage counts (how many mechanisms mention each symptom)
      - disease→mechanism links
      - viral vs bacterial signal counts per symptom
      - combo candidates (symptom pairs that co-occur in many mechanisms)

    Returns dict with:
      "sym_mech_count":   {symptom: count}   — mechanisms covering this symptom
      "viral_syms":       {symptom: count}   — viral mechanism coverage
      "bact_syms":        {symptom: count}   — bacterial mechanism coverage
      "disease_syms":     {disease: {syms}}  — symptoms linked to each disease
      "total":            int
    """
    import glob as _glob
    from collections import defaultdict, Counter

    result = {
        "sym_mech_count": Counter(),
        "viral_syms":     Counter(),
        "bact_syms":      Counter(),
        "disease_syms":   defaultdict(set),
        "total":          0,
    }

    patterns = [
        os.path.join(project_dir, "*mechanism*.json"),
        os.path.join(project_dir, "mechanisms", "*mechanism*.json"),
        os.path.join(project_dir, "medical_knowledge", "mechanisms", "*mechanism*.json"),
    ]
    found = []
    for pat in patterns:
        found.extend(_glob.glob(pat))
    found = list(set(found))  # deduplicate

    if not found:
        print(f"  [警告] 未找到机制文件，路径：{project_dir}")
        return result

    print(f"[SYNC] 读取 {len(found)} 个机制文件")
    for fpath in found:
        try:
            items = json.load(open(fpath, encoding="utf-8"))
            if not isinstance(items, list):
                continue
        except Exception as e:
            print(f"  [WARN] {fpath}: {e}")
            continue

        domain = None  # detect from items
        for m in items:
            result["total"] += 1
            domain = m.get("domain", "").lower()

            # Normalise symptom names (mechanisms use underscores)
            raw_syms = m.get("typical_symptoms", [])
            syms = [s.lower().replace("_", " ").strip() for s in raw_syms]

            for sym in syms:
                result["sym_mech_count"][sym] += 1
                if domain == "virus":
                    result["viral_syms"][sym] += 1
                elif domain == "bacteria":
                    result["bact_syms"][sym] += 1

            # Disease links
            for dis in m.get("linked_diseases", []) or []:
                dis_norm = dis.lower().replace("_", " ").strip()
                for sym in syms:
                    result["disease_syms"][dis_norm].add(sym)

        fname = os.path.basename(fpath)
        print(f"  {fname}: {len(items)} mechanisms")

    print(f"  → 共 {result['total']} 条机制，"
          f"覆盖 {len(result['sym_mech_count'])} 个独立症状")
    return result


# ─────────────────────────────────────────────────────────────
# WEIGHT BUILDER (uses symptom + mechanism data)
# ─────────────────────────────────────────────────────────────

def build_symptom_weights(
    base_weights: dict,
    sym_data: dict,
    mech_data: dict,
) -> dict:
    """
    Merge base weights with data-derived weights from symptom and mechanism files.

    Rules:
      - Symptoms with red_flags in symptom files get +0.3 boost (min 1.8)
      - Symptoms with high viral mechanism coverage get viral_weight recorded
      - Symptoms with high bacterial mechanism coverage noted
      - Base weights always win if already set (don't downgrade manually tuned values)
    """
    weights = dict(base_weights)

    total_mechs = max(mech_data["total"], 1)

    for sym in sym_data["all_names"]:
        # Red flag boost
        if sym in sym_data["red_flag_syms"]:
            if sym not in weights:
                weights[sym] = 1.8
            elif weights[sym] < 1.8:
                weights[sym] = 1.8  # floor at 1.8 for red-flag symptoms

        # Mechanism coverage boost
        mech_count = mech_data["sym_mech_count"].get(sym, 0)
        mech_frac  = mech_count / total_mechs
        if mech_frac > 0.15 and sym not in weights:
            # High mechanism coverage → moderate signal
            weights[sym] = round(1.0 + min(mech_frac * 3, 0.8), 2)

    return weights


# ─────────────────────────────────────────────────────────────────────────────
# DISEASE RULES AUTO-GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

_SYM_NORM = {
    "burning with urination":        "dysuria",
    "increased frequency":           "urinary frequency",
    "urgency":                       "urinary urgency",
    "low-grade fever":               "fever",
    "productive cough":              "cough",
    "pleuritic chest pain":          "chest pain",
    "chest pressure with exertion":  "chest pain",
    "burning chest pain after meals":"heartburn",
    "very high blood pressure":      "hypertension",
    "increased thirst":              "polydipsia",
    "frequent urination":            "urinary frequency",
    "throbbing headache":            "headache",
    "light sensitivity":             "photophobia",
    "sound sensitivity":             "phonophobia",
    "bilateral pressure-like headache":"headache",
    "neck tightness":                "neck stiffness",
    "anorexia and nausea":           "nausea",
    "right upper quadrant pain":     "abdominal pain",
    "jaundice and scleral icterus":  "jaundice",
    "abdominal cramps":              "abdominal pain",
    "fatigue and malaise":           "fatigue",
    "shortness of breath":           "shortness of breath",
    "runny nose":                    "runny nose",
    "sore throat":                   "sore throat",
}

_SYM_KEYWORDS = {
    "fever": "fever",           "cough": "cough",
    "headache": "headache",     "nausea": "nausea",
    "vomiting": "vomiting",     "diarrhea": "diarrhea",
    "fatigue": "fatigue",       "chills": "chills",
    "rash": "rash",             "swelling": "swelling",
    "dizziness": "dizziness",   "pain": "pain",
    "weakness": "weakness",     "jaundice": "jaundice",
    "wheezing": "wheezing",     "dyspnea": "shortness of breath",
    "hemoptysis": "hemoptysis", "hematuria": "blood in urine",
    "edema": "swelling",        "photophobia": "photophobia",
    "hypertension": "high blood pressure",
}


def _normalize_symptom(s: str) -> str:
    """Map verbose disease symptom string to normalized short form."""
    s = s.lower().strip()
    for verbose, norm in _SYM_NORM.items():
        if s.startswith(verbose):
            return norm
    for kw, norm in _SYM_KEYWORDS.items():
        if kw in s:
            return norm
    return s


def generate_disease_rules(project_dir: str) -> dict:
    """
    Auto-generate disease_rules.json from disease_0001.json.
    Uses common_symptoms as both required_any and coherence.
    Merges with any existing manual overrides in disease_rules.json.
    """
    import glob as _glob

    # Find disease file
    disease_file = None
    for p in [os.path.join(project_dir, "disease_0001.json"),
              os.path.join(os.path.dirname(project_dir), "disease_0001.json"),
              "disease_0001.json"]:
        if os.path.exists(p):
            disease_file = p
            break

    if not disease_file:
        print("  [警告] 未找到 disease_0001.json，跳过规则自动生成")
        return {}

    diseases = json.load(open(disease_file, encoding="utf-8"))
    if not isinstance(diseases, list):
        return {}

    rules = {}
    for d in diseases:
        name = d.get("disease_name", "").strip().lower()
        if not name:
            continue
        raw_syms = d.get("common_symptoms", [])
        normalized = list({_normalize_symptom(s) for s in raw_syms if s.strip()})
        if len(normalized) >= 1:
            rules[name] = {
                "required_any": sorted(normalized),
                "coherence":    sorted(normalized),
                "_source":      "auto-generated from disease_0001.json",
            }

    # Load existing manual overrides and merge (manual wins)
    manual_path = os.path.join(project_dir, "nexus_engine", "disease_rules.json")
    if os.path.exists(manual_path):
        try:
            manual = json.load(open(manual_path, encoding="utf-8")).get("rules", {})
            for name, rule in manual.items():
                if "_source" not in rule or "auto" not in rule.get("_source", ""):
                    rules[name] = rule  # manual overrides auto
        except Exception as e:
            print(f"  [警告] 无法读取手动规则: {e}")

    print(f"[SYNC] 自动生成 {len(rules)} 条疾病证据规则（来自 disease_0001.json）")
    return rules

# ─────────────────────────────────────────────────────────────
# MAIN CONFIG BUILDER
# ─────────────────────────────────────────────────────────────

def build_config(
    knowledge_dir: str,
    nexus_medical_path: str,
    existing_config_path: str | None = None,
    training_overrides: dict | None = None,
) -> dict:
    """
    Build a complete nexus_config.json from your NEXUS knowledge files.

    existing_config_path: if provided, preserve manual overrides from it
                          (symptom_weights you tuned, training hyperparams, etc.)
    """
    print(f"\n[SYNC] 读取疾病数据：{knowledge_dir}")
    # Try disease_0001.json in knowledge_dir itself first (this project's format)
    # Also check parent (in case knowledge_dir is medical_knowledge/ subdirectory)
    diseases = read_from_project_root(knowledge_dir)
    if not diseases:
        project_root = str(Path(knowledge_dir).parent)
        diseases = read_from_project_root(project_root)
    if not diseases:
        diseases = read_disease_files(knowledge_dir)
    print(f"[SYNC] 找到 {len(diseases)} 个疾病（含症状数据）")
    for d in diseases:
        print(f"       {d['name']:25s} {len(d['symptoms'])} symptoms  ({d['_source_file']})")

    print(f"\n[SYNC] 读取症状权重来源：{nexus_medical_path}")
    sym_weights = read_symptom_weights_from_nexus(nexus_medical_path)

    # ── Scan symptom files ─────────────────────────────────────
    sym_data = read_symptom_files(knowledge_dir)

    # ── Scan mechanism files ───────────────────────────────────
    mech_data = read_mechanism_files(knowledge_dir)

    # ── Merge weights from all sources ────────────────────────
    sym_weights = build_symptom_weights(sym_weights, sym_data, mech_data)

    # ── Auto-generate disease rules ──────────────────────────────────────
    disease_rules = generate_disease_rules(knowledge_dir)
    if disease_rules:
        rules_path = os.path.join(knowledge_dir, "nexus_engine", "disease_rules.json")
        _existing = {}
        try:
            _existing = json.load(open(rules_path, encoding="utf-8")) if os.path.exists(rules_path) else {}
        except Exception:
            pass
        _existing["_comment"] = "自动生成自 disease_0001.json。手动规则会覆盖自动规则。"
        _existing["_version"] = "1.0"
        _existing["rules"] = disease_rules
        json.dump(_existing, open(rules_path, "w", encoding="utf-8"),
                  indent=2, ensure_ascii=False)
        print(f"[SYNC] disease_rules.json 已更新 → {rules_path}")

    treatments = collect_all_treatments(diseases)
    combos     = build_combo_signals(diseases)

    # ── Auto noise_symptoms from symptom files (vague/nonspecific) ──
    # Symptoms that appear in many mechanisms but are non-specific
    _total = max(mech_data["total"], 1)
    _auto_noise = [s for s, c in mech_data["sym_mech_count"].most_common(30)
                   if c / _total > 0.12 and s not in {"chest pain", "shortness of breath",
                   "syncope", "confusion", "bloody stool", "hematemesis"}]

    # ── Load existing config for manual overrides ──────────────
    existing = {}
    if existing_config_path and os.path.exists(existing_config_path):
        existing = json.load(open(existing_config_path, encoding="utf-8"))
        print(f"[SYNC] 保留手动配置：{existing_config_path}")

        # Merge symptom weights (keep manually tuned values)
        # Validate keys: skip anything that looks like code, not a symptom name
        if "symptom_weights" in existing:
            for k, v in existing["symptom_weights"].items():
                if k.startswith("_"):
                    continue
                # Only accept clean symptom names (short, no special chars)
                if (isinstance(k, str) and len(k) < 45 and isinstance(v, (int, float)) and
                        not any(c in k for c in ['(', ')', '{', '}', '[', ']', '\n', '\\', ':'])):
                    sym_weights[k] = v

        # Keep manual combo signals
        if "combo_signals" in existing:
            for k, v in existing["combo_signals"].items():
                if not k.startswith("_"):
                    combos[k] = v

    # ── Default training config ────────────────────────────────
    training = {
        "episodes": 5000,
        "learning_rate": 0.005,
        "momentum": 0.9,
        "hidden_size": 256,
        "replay_buffer_size": 5000,
        "learn_every": 200,
        "env_noise_p": 0.15,
        "curriculum": [
            {"until_episode": 1000,  "clean": 0.50, "partial": 0.30, "noisy": 0.20},
            {"until_episode": 2500,  "clean": 0.30, "partial": 0.40, "noisy": 0.30},
            {"until_episode": 4000,  "clean": 0.20, "partial": 0.30, "noisy": 0.50},
            {"until_episode": 99999, "clean": 0.10, "partial": 0.30, "noisy": 0.60},
        ],
        "nexus_score_dropout": [
            {"until_episode": 2500,  "rate": 0.0},
            {"until_episode": 4000,  "rate": 0.3},
            {"until_episode": 99999, "rate": 0.5},
        ],
        "noise_symptoms": _auto_noise[:15] if _auto_noise else [
            "fatigue", "dizziness", "weakness", "loss of appetite", "insomnia",
            "sweating", "chills", "back pain", "joint pain", "sore throat",
            "runny nose", "rash", "anxiety", "palpitations", "dry mouth",
        ],
        "env_noise_symptoms": _auto_noise[:5] if _auto_noise else [
            "fatigue", "dizziness", "weakness", "loss of appetite", "insomnia",
        ],
    }

    # Override from existing config or explicit overrides
    if "training" in existing:
        for k, v in existing["training"].items():
            training[k] = v
    if training_overrides:
        training.update(training_overrides)

    # Scale episodes with disease count (more diseases = more training needed)
    n_diseases = len(diseases)
    if n_diseases > 15:
        training["episodes"] = max(training["episodes"], 7000)
    if n_diseases > 25:
        training["episodes"] = max(training["episodes"], 10000)

    # ── Clean up disease entries (remove internal _source_file) ──
    clean_diseases = [{k: v for k, v in d.items() if not k.startswith("_")}
                      for d in diseases]

    config = {
        "_comment": "Auto-generated by nexus_sync_config.py — edit nexus_config.json for manual overrides",
        "_version": "2.0",
        "_generated_from": knowledge_dir,
        "_disease_count": n_diseases,

        "diseases": clean_diseases,
        "treatments": treatments,
        "_symptom_count":   len(sym_data["all_names"]),
        "_mechanism_count": mech_data["total"],
        "_viral_coverage":  len(mech_data["viral_syms"]),
        "_bact_coverage":   len(mech_data["bact_syms"]),

        "symptom_weights": {
            "_comment": "Diagnostic signal strength. 1.0=neutral, 2.5=pathognomonic. Edit to tune.",
            **{k: v for k, v in sorted(sym_weights.items()) if not k.startswith("_")},
        },

        "combo_signals": {
            "_comment": "Symptom combinations that strongly indicate a disease.",
            **{k: v for k, v in combos.items() if not k.startswith("_")},
        },

        "anatomy": {
            "critical_organs": ["brain","heart","lungs","meninges","liver","kidney","spleen","brainstem"],
            "organ_systems":   ["respiratory","cardiovascular","neurologic","gi","systemic","immune","unknown"],
        },

        "training": training,

        "feature_layout": {
            "_comment": "Auto-computed from config. Do not edit manually.",
            "nexus_disease_scores": "[0 : N_DISEASES]",
            "reserved":             "[N_DISEASES : N_DISEASES+10]",
            "organ_system_dist":    "[N_DISEASES+10 : N_DISEASES+17]",
            "nexus_mechanism":      "[N_DISEASES+17 : N_DISEASES+22]",
            "anatomy_spread":       "[N_DISEASES+22 : N_DISEASES+30]",
            "symptom_onehot":       "[N_DISEASES+30 : N_DISEASES+30+N_SYMPTOMS]",
            "combo_signals":        "[N_DISEASES+30+N_SYMPTOMS : +N_COMBOS]",
            "total":                "auto, padded to next multiple of 64",
        },
    }

    return config


def print_summary(config: dict):
    diseases   = config["diseases"]
    n_diseases  = len(diseases)
    n_syms      = len(set(s for d in diseases for s in d["symptoms"]))
    n_tx        = len(config["treatments"])
    n_sym_files = config.get("_symptom_count", 0)
    n_mechs     = config.get("_mechanism_count", 0)
    n_combos   = len([k for k in config["combo_signals"] if not k.startswith("_")])
    import math
    n_weights  = len([k for k in config["symptom_weights"] if not k.startswith("_")])
    raw_dim    = n_diseases + 10 + 7 + 5 + 8 + n_syms + n_combos
    n_features = math.ceil(raw_dim / 64) * 64
    episodes   = config["training"]["episodes"]

    print(f"""
╔══════════════════════════════════════════════════════╗
║  NEXUS 配置摘要                                      ║
╠══════════════════════════════════════════════════════╣
║  疾病数量     : {n_diseases:>4}                                  ║
║  症状文件     : {n_sym_files:>4}  (symptom_XXXX.json)              ║
║  机制条数     : {n_mechs:>4}  (病毒 + 细菌 + 通用)             ║
║  治疗方案     : {n_tx:>4}                                  ║
║  独立症状     : {n_syms:>4}  (来自疾病模板)                    ║
║  症状权重     : {n_weights:>4}  (来自症状文件 + 机制文件)        ║
║  组合信号     : {n_combos:>4}  (多症状组合规则)                 ║
║  特征维度     : {n_features:>4}  (自动计算)                     ║
║  训练轮次     : {episodes:>4}                                  ║
╚══════════════════════════════════════════════════════╝
""")


# ─────────────────────────────────────────────────────────────
# WATCHER MODE  (auto-resync when knowledge files change)
# ─────────────────────────────────────────────────────────────

def watch_and_sync(knowledge_dir: str, output_path: str,
                   nexus_medical_path: str, interval: int = 30):
    """
    监听知识库目录，文件变更时自动重新生成配置。
    Useful during active knowledge base development.

    Run with: python nexus_sync_config.py --watch
    """
    import time, hashlib

    def dir_hash(path):
        h = hashlib.md5()
        for f in sorted(glob.glob(os.path.join(path, "**/*.json"), recursive=True)):
            try:
                h.update(open(f, "rb").read())
            except Exception:
                pass
        return h.hexdigest()

    print(f"[监听] 监控目录 {knowledge_dir}，每 {interval} 秒检查一次...")
    print(f"[监听] 按 Ctrl+C 停止。\n")

    last_hash = None
    while True:
        try:
            current_hash = dir_hash(knowledge_dir)
            if current_hash != last_hash:
                print(f"\n[监听] 检测到文件变更，正在重新生成配置...")
                config = build_config(knowledge_dir, nexus_medical_path, output_path)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                print(f"[监听] 配置已更新 → {output_path}")
                print_summary(config)
                print(f"[监听] 检查点文件已失效，请删除 nexus_checkpoint.npz")
                print(f"        并重新训练后再运行测试。\n")
                last_hash = current_hash
            time.sleep(interval)
        except KeyboardInterrupt:
            print("\n[WATCH] Stopped.")
            break


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="从 NEXUS 知识库文件自动生成 nexus_config.json。"
    )
    parser.add_argument("--knowledge-dir",  default="medical_knowledge",
                        help="知识库目录路径（默认：./medical_knowledge，或直接用 . 指定项目根目录）")
    parser.add_argument("--nexus-medical",  default=None,
                        help="nexus_medical.py 路径（不填则自动查找）")
    parser.add_argument("--output",         default="nexus_config.json",
                        help="输出配置文件路径（默认：nexus_config.json）")
    parser.add_argument("--dry-run",        action="store_true",
                        help="预览生成结果，不写入文件")
    parser.add_argument("--watch",          action="store_true",
                        help="监听知识库目录，文件变更时自动重新生成")
    parser.add_argument("--watch-interval", type=int, default=30,
                        help="监听间隔秒数（默认：30 秒）")
    parser.add_argument("--episodes",       type=int, default=None,
                        help="覆盖训练轮次")
    args = parser.parse_args()

    # Auto-detect nexus_medical.py
    here = Path(__file__).parent
    if args.nexus_medical:
        nexus_medical_path = args.nexus_medical
    else:
        candidates = [
            here / "nexus_medical.py",
            here.parent / "nexus_medical.py",
            here / "nexus_engine" / "nexus_medical.py",
        ]
        nexus_medical_path = next(
            (str(c) for c in candidates if c.exists()), ""
        )

    # Resolve knowledge dir relative to CWD first, then script location
    knowledge_dir = args.knowledge_dir
    if not os.path.isabs(knowledge_dir):
        cwd = Path.cwd()
        for base in [cwd, cwd.parent, here.parent, here, Path(".")]:
            candidate = base / knowledge_dir
            if candidate.exists():
                knowledge_dir = str(candidate)
                break

    # Also accept project root directly (disease_0001.json in CWD)
    if not os.path.exists(knowledge_dir):
        # Try the current working directory as fallback
        if os.path.exists("disease_0001.json"):
            knowledge_dir = "."
            print(f"[SYNC] 使用项目根目录（已找到 disease_0001.json）")
        else:
            print(f"[错误] 知识库目录不存在：{knowledge_dir}")
            print(f"        请在项目根目录（disease_0001.json 所在位置）运行，")
            print(f"        或指定参数：--knowledge-dir .")
            sys.exit(1)

    if args.watch:
        watch_and_sync(
            knowledge_dir,
            args.output,
            nexus_medical_path,
            interval=args.watch_interval,
        )
        return

    # Single sync run
    overrides = {}
    if args.episodes:
        overrides["episodes"] = args.episodes

    config = build_config(
        knowledge_dir       = knowledge_dir,
        nexus_medical_path  = nexus_medical_path,
        existing_config_path= args.output if os.path.exists(args.output) else None,
        training_overrides  = overrides or None,
    )

    print_summary(config)

    if args.dry_run:
        print("[预览模式] 将写入：", args.output)
        print(json.dumps(config, indent=2)[:2000], "...")
        return

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"[SYNC] 配置已写入 → {args.output}")

    # Warn if checkpoint exists (now invalid)
    checkpoint = Path(args.output).parent / "nexus_checkpoint.npz"
    if checkpoint.exists():
        print(f"\n[警告] nexus_checkpoint.npz 文件已存在，但可能已失效")
        print(f"       （疾病或症状数量有变化时需重新训练）")
        print(f"       执行：rm nexus_checkpoint.npz && python nexus_engine/nexus_learning_env_v2.py")
    else:
        print(f"\n下一步：")
        print(f"  python nexus_engine/nexus_learning_env_v2.py")


if __name__ == "__main__":
    main()