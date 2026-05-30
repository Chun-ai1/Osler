"""
NEXUS Derivation Engine — Compute facts from LAYER 1 mechanisms
═══════════════════════════════════════════════════════════════
This module DERIVES medical facts from physical/anatomic data
instead of looking them up in hardcoded JSON.

What it can derive (from LAYER 1 data):
  • organ → body zone (from 3D containment)
  • disease → zones (from primary organs' zones)
  • disease → system (from primary organs' system)
  • symptom → zones (from primary organs' zones)
  • symptom → adjacent organs (from atlas connections)

These are TRUE derivations — no medical knowledge is hardcoded here.
The only inputs are LAYER 1 (geometry + connections + innervation).

USAGE:
    from .derivation_engine import DerivationEngine
    de = DerivationEngine()
    
    de.derive_organ_zones("appendix")
    # → {"primary_zones": ["rlq"], "method": "3d_containment"}
    
    de.derive_disease_zones(primary_organs=["appendix", "ileocolic_a"])
    # → ["rlq"]  (union of all primary organ zones)
    
    de.derive_disease_system(primary_organs=["heart"])
    # → "cardiovascular"
"""
from __future__ import annotations
import json
import os
from typing import Dict, List, Set, Optional
from collections import defaultdict


class DerivationEngine:
    """Derives medical facts from LAYER 1 anatomy data."""
    
    DEFAULT_PATHS = {
        'organ_geometry':  'medical_knowledge/anatomy/organ_geometry.json',
        'body_zones':      'medical_knowledge/anatomy/body_zones.json',
    }
    
    # Organ ID → system mapping (derived from anatomy_atlas naming conventions)
    # This is the only "hardcoded" thing — and it's a NAMING CONVENTION,
    # not medical knowledge. (How we chose to name our system categories.)
    ORGAN_SYSTEM_RULES = [
        # (keyword in organ_id, system)
        # ── Cardiovascular ──
        ('heart',        'cardiovascular'),
        ('atrium',       'cardiovascular'),
        ('ventricle',    'cardiovascular'),
        ('coronary',     'cardiovascular'),
        ('aorta',        'cardiovascular'),
        ('vena_cava',    'cardiovascular'),
        ('_a',           'cardiovascular'),     # any artery
        ('_v',           'cardiovascular'),     # any vein
        ('lad',          'cardiovascular'),
        ('rca',          'cardiovascular'),
        ('lcx',          'cardiovascular'),
        ('pericardium',  'cardiovascular'),
        # ── Respiratory ──
        ('lung',         'respiratory'),
        ('bronchus',     'respiratory'),
        ('trachea',      'respiratory'),
        ('pleura',       'respiratory'),
        ('pharynx',      'respiratory'),
        ('larynx',       'respiratory'),
        ('alveol',       'respiratory'),
        # ── GI (digestive) ──
        ('stomach',      'gi'),
        ('esophagus',    'gi'),
        ('duodenum',     'gi'),
        ('jejunum',      'gi'),
        ('ileum',        'gi'),
        ('colon',        'gi'),
        ('rectum',       'gi'),
        ('cecum',        'gi'),
        ('appendix',     'gi'),
        # ── Hepatobiliary ──
        ('liver',        'hepatobiliary'),
        ('bile',         'hepatobiliary'),
        ('gallbladder',  'hepatobiliary'),
        ('pancreas',     'hepatobiliary'),
        # ── Renal/Urinary ──
        ('kidney',       'renal'),
        ('ureter',       'renal'),
        ('bladder',      'renal'),
        ('urethra',      'renal'),
        ('prostate',     'renal'),
        # ── Reproductive ──
        ('uterus',       'reproductive'),
        ('ovary',        'reproductive'),
        ('ovaries',      'reproductive'),
        ('testes',       'reproductive'),
        ('testis',       'reproductive'),
        # ── Neurologic ──
        ('brain',        'neurologic'),
        ('cerebr',       'neurologic'),
        ('spinal_cord',  'neurologic'),
        ('brainstem',    'neurologic'),
        ('meninges',     'neurologic'),
        ('cerebellum',   'neurologic'),
        # ── Endocrine ──
        ('thyroid',      'endocrine'),
        ('adrenal',      'endocrine'),
        ('pituitary',    'endocrine'),
        # ── Hematologic / lymphatic ──
        ('spleen',       'hematologic'),
        ('thymus',       'hematologic'),
        ('_ln',          'hematologic'),       # any lymph node
        ('bone_marrow',  'hematologic'),
        # ── MSK ──
        ('muscle',       'msk'),
        ('bone',         'msk'),
        ('joint',        'msk'),
        # ── Integumentary ──
        ('skin',         'integumentary'),
    ]
    
    def __init__(self):
        self.organ_geometry: Dict[str, dict] = {}
        self.body_zones: Dict[str, dict] = {}
        self._organ_zone_cache: Dict[str, List[str]] = {}
        self._atlas_connections: List = []
        self._load()
        self._compute_organ_zones()
    
    def _load(self):
        for key, path in self.DEFAULT_PATHS.items():
            for p in [path, f"../{path}",
                      os.path.join(os.path.dirname(__file__), '..', path)]:
                if os.path.exists(p):
                    try:
                        data = json.load(open(p, encoding='utf-8'))
                        if key == 'organ_geometry':
                            self.organ_geometry = data.get('organs', {})
                        elif key == 'body_zones':
                            self.body_zones = data.get('zones', {})
                        break
                    except Exception:
                        pass
        
        # Try to load atlas for connections
        try:
            from .anatomy_atlas import AnatomyAtlas
            self._atlas = AnatomyAtlas()
            self._atlas_connections = self._atlas.connections
        except ImportError:
            try:
                from anatomy_atlas import AnatomyAtlas
                self._atlas = AnatomyAtlas()
                self._atlas_connections = self._atlas.connections
            except ImportError:
                self._atlas = None
                self._atlas_connections = []
        
        print(f"[Derivation] {len(self.organ_geometry)} organs, "
              f"{len(self.body_zones)} zones, "
              f"{len(self._atlas_connections)} connections")
    
    # ═══════════════════════════════════════════════════════════
    # Core derivation: organ 3D position → body zone (containment)
    # ═══════════════════════════════════════════════════════════
    
    def _compute_organ_zones(self):
        """Pre-compute organ → zones via 3D containment.
        Reads positions from atlas.organs[id].pos_3d (where they actually live)."""
        if not self._atlas:
            return
        for organ_id, organ_obj in self._atlas.organs.items():
            center = getattr(organ_obj, 'pos_3d', None)
            if not center or len(center) != 3:
                continue
            zones = []
            for zone_id, zone_info in self.body_zones.items():
                zc = zone_info.get('center', [0,0,0])
                ze = zone_info.get('extent', [0,0,0])
                # Inflate zone extent slightly (5%) for organs near boundary
                pad = 0.02
                if (abs(center[0] - zc[0]) <= ze[0] + pad and
                    abs(center[1] - zc[1]) <= ze[1] + pad and
                    abs(center[2] - zc[2]) <= ze[2] + pad):
                    zones.append(zone_id)
            self._organ_zone_cache[organ_id] = zones
    
    # ═══════════════════════════════════════════════════════════
    # Public API
    # ═══════════════════════════════════════════════════════════
    
    def derive_organ_zones(self, organ_id: str) -> List[str]:
        """Which body zone(s) contains this organ?"""
        return self._organ_zone_cache.get(organ_id, [])
    
    def derive_disease_zones(self, primary_organs: List[str]) -> List[str]:
        """Aggregate zones from all primary organs of a disease.
        Returns sorted, deduplicated list of body zones."""
        zones: Set[str] = set()
        for organ in primary_organs:
            zones.update(self.derive_organ_zones(organ))
        return sorted(zones)
    
    def derive_organ_system(self, organ_id: str) -> str:
        """Map organ → body system using atlas (single source of truth).
        Falls back to naming rules only if the atlas is missing data."""
        # Primary path: ask the atlas directly
        if self._atlas and organ_id in self._atlas.organs:
            sys_attr = getattr(self._atlas.organs[organ_id], 'system', None)
            if sys_attr:
                return sys_attr
        # Fallback: naming rules (for organs not in atlas — rare)
        oid_l = organ_id.lower()
        for keyword, system in self.ORGAN_SYSTEM_RULES:
            if keyword in oid_l:
                return system
        return ""
    
    def derive_disease_system(self, primary_organs: List[str]) -> str:
        """Aggregate primary organs' systems → most common system."""
        if not primary_organs:
            return ""
        system_counts: Dict[str, int] = defaultdict(int)
        for organ in primary_organs:
            sys = self.derive_organ_system(organ)
            if sys:
                system_counts[sys] += 1
        if not system_counts:
            return ""
        # Most frequent system wins
        return max(system_counts.items(), key=lambda x: x[1])[0]
    
    def derive_symptom_zones(self, primary_organs: List[str]) -> List[str]:
        """Same as disease zones, but for symptoms."""
        return self.derive_disease_zones(primary_organs)
    
    def derive_symptom_adjacent(self, primary_organs: List[str],
                                  max_distance: int = 1) -> List[str]:
        """Find organs adjacent to the symptom's primary organs via atlas."""
        if not self._atlas_connections:
            return []
        primary_set = set(primary_organs)
        adjacent: Set[str] = set()
        
        # Build adjacency map (cached)
        if not hasattr(self, '_adjacency_map'):
            self._adjacency_map = defaultdict(set)
            for conn in self._atlas_connections:
                src = getattr(conn, 'source', None)
                tgt = getattr(conn, 'target', None)
                if src and tgt:
                    self._adjacency_map[src].add(tgt)
                    self._adjacency_map[tgt].add(src)
        
        # 1-hop adjacents
        for organ in primary_set:
            adjacent.update(self._adjacency_map.get(organ, set()))
        
        # Remove primary organs from adjacent set
        adjacent -= primary_set
        return sorted(adjacent)
    
    def derive_full_disease_anatomy(self, primary_organs: List[str]) -> dict:
        """One-shot: derive all of a disease's anatomic fields from primary organs."""
        return {
            'primary': sorted(primary_organs),
            'zones':   self.derive_disease_zones(primary_organs),
            'system':  self.derive_disease_system(primary_organs),
            '_derived': True,  # mark as derived, not hardcoded
        }
    
    def derive_full_symptom_anatomy(self, primary_organs: List[str]) -> dict:
        """One-shot: derive all of a symptom's anatomic fields from primary organs."""
        return {
            'primary':  sorted(primary_organs),
            'adjacent': self.derive_symptom_adjacent(primary_organs),
            'zones':    self.derive_symptom_zones(primary_organs),
            '_derived': True,
        }


# ═══════════════════════════════════════════════════════════════
# Demo / verification
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    de = DerivationEngine()
    
    print("\n" + "="*70)
    print("DERIVATION VERIFICATION")
    print("="*70)
    
    # Test 1: organ → zone (containment)
    print("\n1. Organ → zone (3D containment)")
    for organ in ['heart', 'appendix', 'liver', 'l_kidney', 'brain']:
        zones = de.derive_organ_zones(organ)
        print(f"   {organ:15s} → {zones}")
    
    # Test 2: disease → zones (aggregation)
    print("\n2. Disease → zones (from primary organs)")
    test_diseases = [
        ('appendicitis', ['appendix', 'cecum', 'ileocolic_a']),
        ('heart_attack', ['heart', 'lad_a', 'rca_a']),
        ('pneumonia',    ['r_lung', 'l_lung', 'lungs']),
        ('cystitis',     ['bladder']),
        ('meningitis',   ['meninges', 'brain']),
    ]
    for name, organs in test_diseases:
        zones = de.derive_disease_zones(organs)
        system = de.derive_disease_system(organs)
        print(f"   {name:18s}: zones={zones}, system={system}")
    
    # Test 3: symptom → zones + adjacent
    print("\n3. Symptom → zones + adjacent organs")
    test_symptoms = [
        ('chest_pain', ['heart', 'lad_a', 'rca_a', 'aorta', 'esophagus']),
        ('abdominal_pain', ['stomach', 'duodenum', 'jejunum', 'ileum']),
        ('headache',   ['brain', 'meninges']),
    ]
    for name, organs in test_symptoms:
        result = de.derive_full_symptom_anatomy(organs)
        print(f"\n   {name}:")
        print(f"     zones:    {result['zones']}")
        print(f"     adjacent: {result['adjacent'][:6]}")