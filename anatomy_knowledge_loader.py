"""
NEXUS Anatomy Knowledge Loader (rewritten for pure-mechanism architecture)
═══════════════════════════════════════════════════════════════
Populates a NEXUS KnowledgeGraph with anatomy-grounded triples.

PHASE B REFACTOR: previous version loaded `bacteria_mechanisms_1200.json`,
`virus_mechanisms_1500.json`, `mechanisms_rag_final.json` (hardcoded mechanism
lists) and stuffed their disease/organ relationships into the KG. Those files
are no longer used in pure-mechanism reasoning. This version sources the
exact same kinds of triples from cascade-derived data instead.

Triples produced (all cascade- or atlas-derived):
  - <disease>    affects_organ   <organ>     conf=0.85  source=cascade_registry
  - <disease>    has_symptom     <symptom>   conf=0.90  source=cascade_produces
  - <symptom>    localizes_to    <organ>     conf=0.75  source=cascade_organ
  - <organ>      part_of         <system>    conf=1.00  source=atlas
  - <disease>    affects_zone    <zone>      conf=0.80  source=cascade_anatomy

Sources used (all cascade/atlas, NO hardcoded mechanism JSONs):
  1. medical_knowledge/registry/disease_mechanism_map.json
  2. medical_knowledge/layer2_physiology/organ_function.json
  3. medical_knowledge/derived/disease_anatomy_cascade.json
  4. medical_knowledge/derived/disease_index_cascade.json
  5. AnatomyAtlas (organ → system mapping)
"""
from __future__ import annotations
import json
import os
import re
from typing import Dict, Set
from collections import defaultdict


class AnatomyKnowledgeLoader:
    """
    Builds an anatomy-grounded NEXUS KnowledgeGraph using cascade-derived data.
    """

    def __init__(self, atlas, kg):
        """
        Args:
            atlas: AnatomyAtlas instance (provides organ → system + 3D positions)
            kg:    KnowledgeGraph instance (will be populated)
        """
        self.atlas = atlas
        self.kg = kg

        # Registry → atlas organ mapping (when registry uses generic names)
        self._registry_to_atlas: Dict[str, list] = {
            "blood":                    [],
            "bone":                     ["bone_marrow"],
            "brachial_plexus":          [],
            "breast":                   [],
            "bronchus":                 ["r_bronchus", "l_bronchus"],
            "cauda_equina":             ["spinal_cord"],
            "colon":                    ["asc_colon", "trans_colon", "desc_colon", "sig_colon"],
            "common_peroneal_nerve":    [],
            "coronary_arteries":        ["coronary_aa", "lad_a", "rca_a", "lcx_a"],
            "eye":                      [],
            "facial_nerve":             [],
            "fascial_compartment_leg":  [],
            "iliotibial_band":          [],
            "joint":                    [],
            "kidney":                   ["r_kidney", "l_kidney"],
            "lymph_node":               ["cerv_ln", "axil_ln", "ing_ln", "med_ln"],
            "median_nerve":             [],
            "middle_ear":               [],
            "nose":                     ["pharynx"],
            "ovary":                    ["uterus"],
            "pancreas_islets":          ["pancreas"],
            "pericardium":              ["heart"],
            "piriformis_muscle":        [],
            "pulmonary_arteries":       ["heart"],
            "retina":                   [],
            "small_intestine":          ["duodenum", "jejunum", "ileum"],
            "spinal_nerves":            ["spinal_cord"],
            "supraspinatus_muscle":     [],
            "systemic_immune":          ["spleen"],
            "throat":                   ["pharynx"],
            "tibial_nerve":             [],
            "trigeminal_nerve":         ["brain"],
            "ulnar_nerve":              [],
            "upper_respiratory_tract":  ["pharynx", "trachea"],
            "vasculature_systemic":     ["aorta", "heart"],
        }

        self.stats = {
            "diseases_loaded": 0,
            "triples_added":   0,
            "cascade_files":   0,
        }

    # ═══════════════════════════════════════════════
    # PUBLIC API
    # ═══════════════════════════════════════════════
    def load_all(self, knowledge_dir: str = "medical_knowledge"):
        """Load all cascade-derived sources into the KG."""

        # 1. Atlas: organ → system + organ → zone (always available)
        self._load_atlas_facts()

        # 2. Registry: disease → organ
        registry_path = os.path.join(knowledge_dir, "registry", "disease_mechanism_map.json")
        if os.path.exists(registry_path):
            self._load_registry(registry_path)

        # 3. Cascade: organ_function.json → disease → symptom mapping
        of_path = os.path.join(knowledge_dir, "layer2_physiology", "organ_function.json")
        if os.path.exists(of_path) and os.path.exists(registry_path):
            self._load_cascade_symptoms(registry_path, of_path)

        # 4. Cascade-derived disease anatomy (zones)
        anat_path = os.path.join(knowledge_dir, "derived", "disease_anatomy_cascade.json")
        if os.path.exists(anat_path):
            self._load_cascade_anatomy(anat_path)

        print(f"[NEXUS-ANATOMY] Cascade-derived KG: "
              f"{self.stats['cascade_files']} files, "
              f"{self.stats['diseases_loaded']} diseases, "
              f"{self.stats['triples_added']} triples")

    # ═══════════════════════════════════════════════
    # LOADERS
    # ═══════════════════════════════════════════════
    def _load_atlas_facts(self):
        """Organ → system facts directly from atlas."""
        for organ_name, organ in self.atlas.organs.items():
            sys_name = (organ.system or "").lower()
            if sys_name:
                self._add(organ_name, "part_of", sys_name, 1.0, "atlas")

    def _load_registry(self, path: str):
        """disease_mechanism_map.json → disease → organ triples."""
        try:
            data = json.load(open(path, encoding="utf-8"))
        except Exception as e:
            print(f"[NEXUS-ANATOMY] registry load error: {e}")
            return
        mappings = data.get("mappings", data) if isinstance(data, dict) else {}
        self.stats["cascade_files"] += 1

        for disease, m in mappings.items():
            organ = m.get("organ", "")
            if not organ:
                continue
            self.stats["diseases_loaded"] += 1
            d_norm = self._norm(disease)

            # Resolve to atlas organs
            atlas_organs = ([organ] if organ in self.atlas.organs
                            else self._registry_to_atlas.get(organ, []))
            for o in atlas_organs:
                if o in self.atlas.organs:
                    self._add(d_norm, "affects_organ", o, 0.85, "cascade_registry")

    def _load_cascade_symptoms(self, registry_path: str, of_path: str):
        """organ_function.json cascades → disease → produces_symptom triples."""
        try:
            registry = json.load(open(registry_path, encoding="utf-8")).get("mappings", {})
            of = json.load(open(of_path, encoding="utf-8")).get("organs", {})
        except Exception as e:
            print(f"[NEXUS-ANATOMY] cascade load error: {e}")
            return
        self.stats["cascade_files"] += 1

        # Regex strip mechanism descriptor suffixes for cleaner symptom names
        STRIP = [r'\bvia\b.*$', r'\bfrom\b.*$', r'\bdue to\b.*$',
                 r'\bthrough\b.*$', r'\bsecondary to\b.*$',
                 r'\bcaused by\b.*$', r'\bsame mechanism\b.*$',
                 r'\b[tlc]\d+[-/]\d+\b', r'\b[tlc]\d+\b']

        def clean(s):
            s = s.lower().strip().replace('_', ' ')
            for pat in STRIP:
                s = re.sub(pat, '', s, flags=re.IGNORECASE)
            return re.sub(r'\s+', ' ', s).strip().rstrip(',;.- ')

        for disease, m in registry.items():
            organ = m.get("organ", "")
            failure_mode = m.get("failure_mode", "")
            if organ not in of:
                continue
            d_norm = self._norm(disease)
            for fm in of[organ].get("failure_modes", []):
                if fm.get("mode") != failure_mode:
                    continue
                for step in fm.get("cascade", []):
                    for k, v in step.items():
                        if k.startswith("produces_symptom") and isinstance(v, str):
                            sym_clean = clean(v)
                            if sym_clean and len(sym_clean) > 1:
                                sym_norm = self._norm(sym_clean)
                                self._add(d_norm, "has_symptom", sym_norm, 0.90,
                                          "cascade_produces")
                                # Transitive: symptom localizes to this disease's organ
                                atlas_organs = ([organ] if organ in self.atlas.organs
                                                else self._registry_to_atlas.get(organ, []))
                                for o in atlas_organs:
                                    if o in self.atlas.organs:
                                        self._add(sym_norm, "localizes_to", o, 0.75,
                                                  "cascade_organ")
                break

    def _load_cascade_anatomy(self, path: str):
        """disease_anatomy_cascade.json → disease → zone triples."""
        try:
            data = json.load(open(path, encoding="utf-8"))
        except Exception as e:
            print(f"[NEXUS-ANATOMY] anatomy load error: {e}")
            return
        diseases = data.get("diseases", {})
        self.stats["cascade_files"] += 1

        for d_norm, info in diseases.items():
            for zone in info.get("zones", []):
                self._add(d_norm, "affects_zone", zone, 0.80, "cascade_anatomy")

    # ═══════════════════════════════════════════════
    # HELPERS
    # ═══════════════════════════════════════════════
    def _add(self, s: str, p: str, o: str, conf: float, source: str):
        s_n = self._norm(s)
        o_n = self._norm(o)
        if s_n and o_n:
            self.kg.add(s_n, p, o_n, conf, source)
            self.stats["triples_added"] += 1

    @staticmethod
    def _norm(s) -> str:
        if not s:
            return ""
        return str(s).strip().lower().replace(" ", "_").replace("-", "_")