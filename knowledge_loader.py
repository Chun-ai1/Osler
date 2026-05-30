"""
Medical Knowledge Loader
═══════════════════════════════════════════════════════════════
Loads all medical knowledge from JSON files in medical_knowledge/.
Provides single source of truth for anatomy + physiology data.

USAGE:
    from nexus_engine.knowledge_loader import KnowledgeLoader
    kb = KnowledgeLoader()
    organs = kb.organ_geometry              # 208 organs
    sym_map = kb.symptom_organ_map          # symptom → organs
    pain_zones = kb.pain_localization       # pain type → zone
    body_zones = kb.body_zones              # 16 body zones
    mech_kw = kb.mechanism_keywords         # mech keyword → organs
    sym_eff = kb.symptom_effects            # symptom → physiology delta
    dis_anat = kb.disease_anatomy           # disease → 3D fingerprint

If a JSON file is missing, the loader returns an empty dict and prints
a warning. Each consumer module should check for empty data and use
inline fallback if needed.
"""
from __future__ import annotations
import json
import os
from typing import Dict, Any, Optional


class KnowledgeLoader:
    """Single source of truth for medical knowledge files."""

    DEFAULT_BASE_DIR = "medical_knowledge"

    # File registry: attribute_name → (subdir, filename, top_level_key)
    REGISTRY = {
        "organ_geometry":     ("anatomy",    "organ_geometry.json",     "organs"),
        "symptom_organ_map":  ("anatomy",    "symptom_organ_map.json",  "symptoms"),
        "mechanism_keywords": ("anatomy",    "mechanism_keywords.json", "keywords"),
        "body_zones":         ("anatomy",    "body_zones.json",         "zones"),
        "pain_localization":  ("anatomy",    "pain_localization.json",  "pain_to_zone"),
        "disease_anatomy":    ("anatomy",    "disease_anatomy.json",    "diseases"),
        "symptom_effects":    ("physiology", "symptom_effects.json",    "symptom_effects"),
    }

    def __init__(self, base_dir: Optional[str] = None, verbose: bool = True):
        self.base_dir = base_dir or self.DEFAULT_BASE_DIR
        self.verbose = verbose
        self._cache: Dict[str, Any] = {}
        self._warnings: list = []
        self._load_all()

    def _load_all(self):
        """Load every registered file into self._cache."""
        loaded = []
        for attr, (subdir, fname, key) in self.REGISTRY.items():
            data = self._load_file(subdir, fname, key)
            self._cache[attr] = data
            if data:
                loaded.append(f"{attr}={len(data)}")
            else:
                self._warnings.append(f"{subdir}/{fname} not loaded")
        if self.verbose:
            print(f"[KB] Loaded: {', '.join(loaded)}")
            if self._warnings:
                for w in self._warnings:
                    print(f"[KB] WARNING: {w}")

    def _load_file(self, subdir: str, fname: str, top_key: str) -> dict:
        """Read a JSON file and extract the data under `top_key`."""
        # Try multiple base path variants (helps when called from different cwd)
        candidates = [
            os.path.join(self.base_dir, subdir, fname),
            os.path.join("..", self.base_dir, subdir, fname),
            os.path.join("/mnt/user-data/outputs", self.base_dir, subdir, fname),
        ]
        for path in candidates:
            if os.path.exists(path):
                try:
                    raw = json.load(open(path, encoding="utf-8"))
                    return raw.get(top_key, {})
                except (json.JSONDecodeError, OSError) as e:
                    self._warnings.append(f"{path}: {e}")
                    return {}
        return {}

    # ── Convenience accessors ──
    @property
    def organ_geometry(self) -> dict:
        return self._cache.get("organ_geometry", {})

    @property
    def symptom_organ_map(self) -> dict:
        return self._cache.get("symptom_organ_map", {})

    @property
    def mechanism_keywords(self) -> dict:
        return self._cache.get("mechanism_keywords", {})

    @property
    def body_zones(self) -> dict:
        return self._cache.get("body_zones", {})

    @property
    def pain_localization(self) -> dict:
        return self._cache.get("pain_localization", {})

    @property
    def disease_anatomy(self) -> dict:
        return self._cache.get("disease_anatomy", {})

    @property
    def symptom_effects(self) -> dict:
        # Convert any list back to tuple where needed (perfusion_organ field)
        raw = self._cache.get("symptom_effects", {})
        result = {}
        for sym, effects in raw.items():
            cleaned = {}
            for k, v in effects.items():
                if k == "perfusion_organ" and isinstance(v, list):
                    cleaned[k] = tuple(v)
                else:
                    cleaned[k] = v
            result[sym] = cleaned
        return result

    def reload(self):
        """Re-read all JSON files (e.g., after admin updates a file)."""
        self._cache.clear()
        self._warnings.clear()
        self._load_all()


# ── Module-level convenience: a singleton instance ──
_global_kb: Optional[KnowledgeLoader] = None

def get_kb() -> KnowledgeLoader:
    """Get the singleton KnowledgeLoader (auto-initializes on first call)."""
    global _global_kb
    if _global_kb is None:
        _global_kb = KnowledgeLoader(verbose=False)
    return _global_kb


if __name__ == "__main__":
    kb = KnowledgeLoader()
    print(f"\n=== Knowledge Loader Summary ===")
    print(f"  organ_geometry:    {len(kb.organ_geometry)} entries")
    print(f"  symptom_organ_map: {len(kb.symptom_organ_map)} entries")
    print(f"  mechanism_keywords:{len(kb.mechanism_keywords)} entries")
    print(f"  body_zones:        {len(kb.body_zones)} entries")
    print(f"  pain_localization: {len(kb.pain_localization)} entries")
    print(f"  disease_anatomy:   {len(kb.disease_anatomy)} entries")
    print(f"  symptom_effects:   {len(kb.symptom_effects)} entries")