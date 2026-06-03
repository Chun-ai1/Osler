"""
NeuroPainReasoner v2 — Mechanistic Pain Localization
═══════════════════════════════════════════════════════════════
Derives pain localization from FIRST PRINCIPLES, both directions:

  FORWARD:  organ → predicted pain zones
            "appendix" → "umbilical → rlq" (classic migration)

  REVERSE:  pain zones → likely organs (NEW in v2)
            ["umbilical", "rlq"] → ["appendix" (high), "ovaries", "ileum"]

Used by NEXUS to:
  1. Predict expected pain pattern for a hypothesized disease
  2. Validate diagnoses against patient's reported pain
  3. Suggest organs to consider given pain location

Replaces hardcoded "appendicitis is in RLQ" with derivation from:
  - Organ neural anatomy (innervation.json)
  - Dermatome map (dermatome_zones.json)  
  - Visceral → somatic migration rules
"""
from __future__ import annotations
import json
import os
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict


class NeuroPainReasoner:
    DEFAULT_PATHS = {
        'innervation':       'medical_knowledge/neurology/innervation.json',
        'dermatome_zones':   'medical_knowledge/neurology/dermatome_zones.json',
        'pain_mechanisms':   'medical_knowledge/neurology/pain_mechanisms.json',
    }
    
    def __init__(self):
        self.innervation: Dict[str, dict] = {}
        self.dermatome_to_zone: Dict[str, List[str]] = {}
        self.pain_rules: Dict = {}
        self._zone_to_organs: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        self._load_data()
        self._build_reverse_index()
    
    def _load_data(self):
        for key, path in self.DEFAULT_PATHS.items():
            candidates = [
                path, f"../{path}",
                os.path.join(os.path.dirname(__file__), '..', path),
            ]
            for p in candidates:
                if os.path.exists(p):
                    try:
                        data = json.load(open(p, encoding='utf-8'))
                        if key == 'innervation':
                            self.innervation = data.get('organs', {})
                        elif key == 'dermatome_zones':
                            raw = data.get('dermatomes', {})
                            # Normalize: ensure all values are lists
                            self.dermatome_to_zone = {
                                k: (v if isinstance(v, list) else [v])
                                for k, v in raw.items()
                            }
                        elif key == 'pain_mechanisms':
                            self.pain_rules = data.get('rules', {})
                        break
                    except Exception:
                        continue
        print(f"[NeuroPain] {len(self.innervation)} organs, "
              f"{len(self.dermatome_to_zone)} dermatomes")
    
    # ═══════════════════════════════════════════════════════════
    # Range expansion (T6-T9, T10_back-T12_back, etc.)
    # ═══════════════════════════════════════════════════════════
    
    def expand_spinal_range(self, level_str: str) -> List[str]:
        """Expand 'T1-T4' to ['T1','T2','T3','T4']. Handles _back suffix too."""
        if not level_str:
            return []
        if '-' not in level_str:
            return [level_str.strip()]
        
        parts = level_str.split('-')
        if len(parts) != 2:
            return [level_str.strip()]
        
        start, end = parts[0].strip(), parts[1].strip()
        
        # Detect _back suffix
        suffix_start = '_back' if start.endswith('_back') else ''
        suffix_end = '_back' if end.endswith('_back') else ''
        
        start_clean = start.replace('_back', '')
        end_clean = end.replace('_back', '')
        
        if start_clean[0] != end_clean[0]:
            return self._expand_cross_region(start, end, suffix_start)
        
        prefix = start_clean[0]
        try:
            start_n = int(start_clean[1:])
            end_n = int(end_clean[1:])
            return [f"{prefix}{n}{suffix_start}" for n in range(start_n, end_n + 1)]
        except ValueError:
            return [start, end]
    
    def _expand_cross_region(self, start: str, end: str, suffix: str = '') -> List[str]:
        regions = {'C': range(1,9), 'T': range(1,13), 'L': range(1,6), 'S': range(1,6)}
        order = ['C', 'T', 'L', 'S']
        result = []
        try:
            sp, sn = start[0], int(start[1:].replace('_back',''))
            ep, en = end[0], int(end[1:].replace('_back',''))
        except (ValueError, IndexError):
            return [start, end]
        capture = False
        for region in order:
            for n in regions[region]:
                level = f"{region}{n}{suffix}"
                if level == start: capture = True
                if capture: result.append(level)
                if level == end: return result
        return result
    
    def levels_to_zones(self, levels: List[str]) -> List[str]:
        """Map spinal levels to body zones via dermatome map."""
        zones: Set[str] = set()
        for level in levels:
            for z in self.dermatome_to_zone.get(level, []):
                zones.add(z)
        return sorted(zones)
    
    # ═══════════════════════════════════════════════════════════
    # FORWARD: organ → pain prediction
    # ═══════════════════════════════════════════════════════════
    
    def predict_pain_pattern(self, organ_id: str) -> dict:
        """Given an organ, predict where pain will be felt."""
        innerv = self.innervation.get(organ_id)
        if not innerv:
            return {
                'organ': organ_id,
                'visceral_zones': [], 'somatic_zones': [],
                'reasoning': [f'no innervation data for {organ_id}'],
                'confidence': 0.0,
            }
        
        result = {
            'organ': organ_id,
            'side':  innerv.get('side', 'unknown'),
            'visceral_zones': [],
            'somatic_zones': [],
            'migration': '',
            'reasoning': [],
            'confidence': 1.0,
        }
        
        # Visceral
        v_levels_raw = innerv.get('visceral') or []
        v_levels = []
        for raw in v_levels_raw:
            v_levels.extend(self.expand_spinal_range(raw))
        v_zones = self.levels_to_zones(v_levels)
        result['visceral_zones'] = v_zones
        result['visceral_levels'] = v_levels
        if v_zones:
            result['reasoning'].append(
                f"Visceral afferents via {','.join(v_levels_raw)} → cord → {v_zones}")
        
        # Somatic
        s_levels_raw = innerv.get('somatic') or []
        s_levels = []
        for raw in s_levels_raw:
            if raw == 'all':
                s_levels.append('all')
            else:
                s_levels.extend(self.expand_spinal_range(raw))
        s_zones = self.levels_to_zones(s_levels)
        result['somatic_zones'] = s_zones
        result['somatic_levels'] = s_levels
        if s_zones:
            result['reasoning'].append(
                f"Somatic afferents via {','.join(s_levels_raw)} → directly: {s_zones}")
        
        # Apply LATERALITY filter (keep only side-appropriate zones)
        side = innerv.get('side', 'midline')
        side_zone = innerv.get('side_zone')
        if side_zone:
            # Disease has explicit lateralized zone (e.g., appendix → rlq specifically)
            result['somatic_zones'] = [side_zone] + [
                z for z in result['somatic_zones']
                if z not in ('rlq', 'llq', 'right_flank', 'left_flank')
            ]
        elif side == 'right':
            result['somatic_zones'] = [z for z in s_zones
                                        if 'l' not in z.split('_')[0] or z.startswith('right')]
            result['somatic_zones'] = [
                'rlq' if z == 'llq' else
                'right_flank' if z == 'left_flank' else z
                for z in result['somatic_zones']
            ]
        elif side == 'left':
            result['somatic_zones'] = [
                'llq' if z == 'rlq' else
                'left_flank' if z == 'right_flank' else z
                for z in result['somatic_zones']
            ]
        
        # Migration pattern
        if v_zones and result['somatic_zones']:
            v_main = v_zones[0]
            s_main = result['somatic_zones'][0]
            if v_main != s_main:
                result['migration'] = f"{v_main} → {s_main}"
                result['reasoning'].append(
                    f"PREDICTION: pain begins in {v_main} (visceral, dull), "
                    f"migrates to {s_main} (somatic, sharp) when "
                    f"parietal layer involved")
            else:
                result['migration'] = f"{v_main} (no migration)"
        
        notes = innerv.get('notes')
        if notes:
            result['textbook_notes'] = notes
        
        return result
    
    # ═══════════════════════════════════════════════════════════
    # REVERSE: pain zones → likely organs (the diagnostic direction!)
    # ═══════════════════════════════════════════════════════════
    
    def _build_reverse_index(self):
        """
        Pre-compute zone → organ mapping for fast reverse lookup.
        For each organ, what zones could its pain map to?
        """
        for organ_id in self.innervation:
            pattern = self.predict_pain_pattern(organ_id)
            for z in pattern.get('visceral_zones', []):
                self._zone_to_organs[z].append((organ_id, 'visceral'))
            for z in pattern.get('somatic_zones', []):
                self._zone_to_organs[z].append((organ_id, 'somatic'))
        
        n_zones = len(self._zone_to_organs)
        n_total = sum(len(v) for v in self._zone_to_organs.values())
        print(f"[NeuroPain] Reverse index: {n_zones} zones → {n_total} organ links")
    
    def predict_organs_from_pain(self, pain_zones: List[str],
                                   migration: bool = False) -> List[Tuple[str, float, str]]:
        """
        Reverse reasoning: given where pain is, what organs might be the source?
        
        Scoring algorithm (mechanistic, not heuristic):
          1. SPECIFICITY: organs whose pain zones EXACTLY match patient's zones
             score higher than organs that share zones but have many extra ones
          2. COVERAGE: organs that cover ALL patient zones score higher
             than organs covering only some
          3. MIGRATION: only credit migration when patient has 2+ zones AND 
             organ has DIFFERENT visceral vs somatic zones
          4. LATERALITY: pain on right side reduces score for left-side organs
        """
        if not pain_zones:
            return []
        
        zone_set = set(z.lower().strip() for z in pain_zones)
        n_patient_zones = len(zone_set)
        
        # Detect patient laterality
        patient_right = any('rlq' in z or 'right' in z for z in zone_set)
        patient_left  = any('llq' in z or 'left' in z for z in zone_set)
        
        results = []
        for organ_id in self.innervation:
            pattern = self.predict_pain_pattern(organ_id)
            v_zones = set(pattern.get('visceral_zones', []))
            s_zones = set(pattern.get('somatic_zones', []))
            organ_zones = v_zones | s_zones
            
            if not organ_zones:
                continue
            
            # 1. COVERAGE: how many patient zones does this organ explain?
            covered = zone_set & organ_zones
            if not covered:
                continue
            coverage = len(covered) / n_patient_zones  # 0-1
            
            # 2. SPECIFICITY: how concentrated is the organ on patient zones?
            #    If organ projects to 10 zones and 1 matches → low specificity
            #    If organ projects to 2 zones and both match → high specificity
            specificity = len(covered) / len(organ_zones)
            
            # 3. MIGRATION bonus: only if patient has 2+ zones AND
            #    organ has different visceral and somatic zones AND
            #    BOTH the organ's visceral and somatic match patient zones
            migration_bonus = 0.0
            v_diff_s = bool(v_zones - s_zones) and bool(s_zones - v_zones)
            v_match = bool(zone_set & v_zones)
            s_match = bool(zone_set & s_zones)
            if n_patient_zones >= 2 and v_diff_s and v_match and s_match:
                migration_bonus = 0.5
            
            # 4. LATERALITY mismatch penalty
            organ_side = pattern.get('side', '')
            laterality_penalty = 0.0
            if patient_right and organ_side == 'left':
                laterality_penalty = 0.3
            elif patient_left and organ_side == 'right':
                laterality_penalty = 0.3
            
            score = coverage * 0.5 + specificity * 0.5 + migration_bonus
            score = max(0.0, score - laterality_penalty)
            
            # Build evidence
            evidence_parts = [
                f"covers {len(covered)}/{n_patient_zones} patient zones: {sorted(covered)}",
                f"specificity={specificity:.2f}",
            ]
            if migration_bonus > 0:
                evidence_parts.append("MIGRATION match")
            if laterality_penalty > 0:
                evidence_parts.append(f"side mismatch ({organ_side})")
            
            results.append((organ_id, score, "; ".join(evidence_parts)))
        
        results.sort(key=lambda x: -x[1])
        return results
    
    def explain_pain_migration(self, early_zone: str, late_zone: str) -> List[Tuple[str, float, str]]:
        """
        Patient says: pain started in X, moved to Y.
        Which organ has this exact migration pattern?
        """
        candidates = []
        for organ_id in self.innervation:
            pattern = self.predict_pain_pattern(organ_id)
            v_zones = pattern.get('visceral_zones', [])
            s_zones = pattern.get('somatic_zones', [])
            
            if early_zone in v_zones and late_zone in s_zones:
                # Perfect match
                pattern_str = pattern.get('migration', '')
                candidates.append((organ_id, 1.0, 
                    f"perfect migration match: {pattern_str}"))
            elif early_zone in v_zones:
                candidates.append((organ_id, 0.5,
                    f"matches early visceral zone {early_zone}"))
            elif late_zone in s_zones:
                candidates.append((organ_id, 0.5,
                    f"matches late somatic zone {late_zone}"))
        
        candidates.sort(key=lambda x: -x[1])
        return candidates


# ═══════════════════════════════════════════════════════════════
# Demo / test
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    np = NeuroPainReasoner()
    
    print("\n" + "="*70)
    print("FORWARD: organ → predicted pain pattern")
    print("="*70)
    for organ in ["appendix", "heart", "spleen", "pancreas", "l_kidney", "desc_colon"]:
        p = np.predict_pain_pattern(organ)
        print(f"\n  {organ:20s} (side={p['side']})")
        if p.get('migration'):
            print(f"    📍 {p['migration']}")
        print(f"    Visceral zones: {p['visceral_zones']}")
        print(f"    Somatic zones:  {p['somatic_zones']}")
    
    print("\n" + "="*70)
    print("REVERSE: pain location → likely organ (the diagnostic direction!)")
    print("="*70)
    
    test_complaints = [
        # (input zones, expected top organ, scenario)
        (['umbilical', 'rlq'],     'appendix',     "Migrating periumbilical → RLQ"),
        (['epigastric'],           'stomach',      "Epigastric pain only"),
        (['rlq'],                  'appendix',     "Just RLQ pain"),
        (['llq'],                  'sig_colon',    "Just LLQ pain"),
        (['epigastric', 'jaw', 'arm_medial'], 'heart', "Chest + jaw + L arm"),
        (['epigastric', 'shoulder'], 'gallbladder', "Epigastric + R shoulder"),
        (['flank', 'inguinal'],    'r_ureter',     "Flank → groin (renal colic)"),
    ]
    
    for zones, expected, scenario in test_complaints:
        results = np.predict_organs_from_pain(zones)
        top5 = results[:5]
        print(f"\n  Scenario: {scenario}")
        print(f"  Pain zones: {zones}")
        print(f"  Expected top organ: {expected}")
        print(f"  NEXUS predicts:")
        for organ, score, evidence in top5:
            mark = "✓" if organ == expected else " "
            print(f"    {mark} {organ:20s} score={score:.2f}  {evidence[:70]}")