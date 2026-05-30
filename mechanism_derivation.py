"""
NEXUS Mechanism Derivation Engine — Phase B Implementation
═══════════════════════════════════════════════════════════════
Derives clinical knowledge from LAYER 2-4 mechanistic data.

INPUT (LAYER 1-4):
  • LAYER 1: anatomy, innervation, dermatomes (already complete)
  • LAYER 2: organ_function with cascade chains
  • LAYER 3: pathogens, immune responses
  • LAYER 4: age/sex/variant dependencies

OUTPUT (auto-derived for any disease):
  • common_symptoms: from organ failure cascade + immune response
  • symptom_weights: from specificity in cascade
  • red_flags: from severity_marker tags in cascades  
  • age_groups: from age_dependencies
  • sex_bias: from sex_dependencies
  • complications: from cascade end-states
  • pathophysiology: reconstructed from cascade chain

This is the proof-of-concept that, with proper LAYER 2-4 data, 
NEXUS can DERIVE most hardcoded fields.
"""
from __future__ import annotations
import json
import os
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict


class MechanismDerivationEngine:
    """Walks LAYER 2-4 to derive clinical knowledge for diseases."""
    
    DEFAULT_PATHS = {
        # LAYER 1
        'innervation':       'medical_knowledge/neurology/innervation.json',
        # LAYER 2
        'organ_function':    'medical_knowledge/layer2_physiology/organ_function.json',
        # LAYER 3
        'pathogens':         'medical_knowledge/layer3_pathogen/pathogens.json',
        'immune':            'medical_knowledge/layer3_pathogen/immune_responses.json',
        # LAYER 4
        'age_dependencies':  'medical_knowledge/layer4_development/age_dependencies.json',
        'sex_dependencies':  'medical_knowledge/layer4_development/sex_dependencies.json',
        'anatomic_variants': 'medical_knowledge/layer4_development/anatomic_variants.json',
    }
    
    def __init__(self):
        self.organ_function: Dict = {}
        self.pathogens: Dict = {}
        self.immune: Dict = {}
        self.age_deps: Dict = {}
        self.sex_deps: Dict = {}
        self.variants: Dict = {}
        self._load_data()
    
    def _load_data(self):
        loaded = {}
        for key, path in self.DEFAULT_PATHS.items():
            for p in [path, f"../{path}",
                      os.path.join(os.path.dirname(__file__), '..', path)]:
                if os.path.exists(p):
                    try:
                        data = json.load(open(p, encoding='utf-8'))
                        if key == 'organ_function':
                            self.organ_function = data.get('organs', {})
                        elif key == 'pathogens':
                            self.pathogens = data.get('pathogens', {})
                        elif key == 'immune':
                            self.immune = data.get('responses', {})
                        elif key == 'age_dependencies':
                            self.age_deps = data.get('tissues', {})
                        elif key == 'sex_dependencies':
                            self.sex_deps = data.get('patterns', {})
                        elif key == 'anatomic_variants':
                            self.variants = data.get('variants', {})
                        loaded[key] = True
                        break
                    except Exception:
                        pass
        
        print(f"[MechDerive] Loaded LAYER 2-4: "
              f"organs={len(self.organ_function)}, "
              f"pathogens={len(self.pathogens)}, "
              f"immune={len(self.immune)}, "
              f"age_deps={len(self.age_deps)}, "
              f"sex={len(self.sex_deps)}, "
              f"variants={len(self.variants)}")
    
    # ═══════════════════════════════════════════════════════════
    # Core derivation: walk a cascade chain to extract symptoms
    # ═══════════════════════════════════════════════════════════
    
    def derive_disease(self, primary_organ: str, failure_mode: str,
                        pathogen: Optional[str] = None) -> dict:
        """
        Given:
          • primary organ (e.g., 'appendix')
          • failure mode (e.g., 'luminal_obstruction')
          • optional pathogen (for infectious diseases)
        
        Derive a complete clinical picture from LAYER 2-4 data.
        
        Returns dict with:
          • common_symptoms (list of derived symptoms)
          • symptom_chain (ordered cascade of how symptoms appear)
          • red_flags (severity_marker tagged events)
          • complications (cascade end-states)
          • pathophysiology (reconstructed text)
          • age_groups (from age_dependencies)
          • sex_bias (from sex_dependencies)
          • specificity_estimates (per symptom)
        """
        result = {
            'primary_organ': primary_organ,
            'failure_mode': failure_mode,
            'pathogen': pathogen,
            'common_symptoms': [],
            'symptom_chain': [],
            'red_flags': [],
            'complications': [],
            'pathophysiology': '',
            'age_groups': [],
            'sex_bias': 'any',
            'derivation_evidence': [],
            '_derived': True,
        }
        
        # Look up organ failure mode
        organ_data = self.organ_function.get(primary_organ, {})
        if not organ_data:
            result['error'] = f"No LAYER 2 data for organ '{primary_organ}'"
            return result
        
        fmode = None
        for fm in organ_data.get('failure_modes', []):
            if fm.get('mode') == failure_mode:
                fmode = fm
                break
        
        if not fmode:
            modes_avail = [f.get('mode') for f in organ_data.get('failure_modes', [])]
            result['error'] = f"Failure mode '{failure_mode}' not found. Available: {modes_avail}"
            return result
        
        # ── Walk the cascade ──
        cascade = fmode.get('cascade', [])
        path_text = []
        
        for step in cascade:
            event = step.get('event', '')
            path_text.append(f"Step {step.get('step', '?')}: {event}")
            
            # Extract symptoms produced at this step
            for key, value in step.items():
                if key.startswith('produces_symptom'):
                    symptom = value
                    result['common_symptoms'].append(symptom)
                    result['symptom_chain'].append({
                        'step': step.get('step'),
                        'symptom': symptom,
                        'mechanism': event,
                    })
                    result['derivation_evidence'].append(
                        f"{event} → {symptom}"
                    )
                if key == 'severity_marker' and value == 'red_flag':
                    result['red_flags'].append({
                        'event': event,
                        'time_hours': step.get('time_hours', 'unknown'),
                    })
                if key == 'produces_complication':
                    result['complications'].append(value)
            
            # Extract any "derives_*" hints (cross-references to other layers)
            for key, value in step.items():
                if key.startswith('derives_'):
                    result['derivation_evidence'].append(
                        f"  Inferred {key.replace('derives_','')}: {value}"
                    )
        
        # ── Apply pathogen-specific symptoms ──
        if pathogen and pathogen in self.pathogens:
            pdata = self.pathogens[pathogen]
            # Direct symptoms (e.g., rhinovirus → rhinorrhea)
            if 'produces_symptoms_direct' in pdata:
                for s in pdata['produces_symptoms_direct']:
                    if s not in result['common_symptoms']:
                        result['common_symptoms'].append(s)
                        result['derivation_evidence'].append(
                            f"Pathogen {pathogen} → {s}"
                        )
            
            # Severity comes from pathogen
            if 'typical_severity' in pdata:
                result['pathogen_severity'] = pdata['typical_severity']
        
        # ── Apply immune response (for infectious failure modes) ──
        if 'infection' in failure_mode.lower():
            immune_pattern = 'acute_bacterial_inflammation' if 'bacteri' in str(pathogen).lower() else 'viral_response'
            imm = self.immune.get(immune_pattern, {})
            for effect in imm.get('produces_systemic_effects', []):
                eff = effect.get('effect')
                if eff and eff not in result['common_symptoms']:
                    result['common_symptoms'].append(eff)
                    result['derivation_evidence'].append(
                        f"Immune {immune_pattern} → {eff}: "
                        f"{effect.get('mechanism','')}"
                    )
        
        # ── Apply age dependencies ──
        # Search for any age tissue that maps to this organ or failure mode
        for tissue_id, tissue in self.age_deps.items():
            implications = tissue.get('implications', [])
            for imp in implications:
                # Check if this implication "derives" anything related to our disease
                derives = imp.get('derives_age_distribution') or imp.get('derives', '')
                if isinstance(derives, str):
                    if (primary_organ in tissue_id.lower() or
                        primary_organ in derives.lower() or
                        failure_mode in derives.lower()):
                        if 'peak' in derives.lower() or 'common' in derives.lower():
                            ar = imp.get('age_range', [])
                            if ar:
                                result['age_groups'].append({
                                    'range_years': ar,
                                    'risk': imp.get('obstruction_risk') or imp.get('risk') or 'elevated',
                                    'reason': imp.get('reason', tissue_id),
                                })
                                result['derivation_evidence'].append(
                                    f"Age {ar[0]}-{ar[1]}: {derives}"
                                )
        
        # ── Apply sex dependencies ──
        for pat_id, pat in self.sex_deps.items():
            applies = pat.get('applies_to', [])
            disease_keywords = [primary_organ, failure_mode] + (
                [pathogen] if pathogen else [])
            
            for kw in disease_keywords:
                if any(kw in a or a in kw for a in applies):
                    ratio = pat.get('incidence_ratio_F_to_M', 1.0)
                    if ratio > 1.5:
                        result['sex_bias'] = 'female_predominant'
                        result['sex_ratio_F_to_M'] = ratio
                    elif ratio < 0.67:
                        result['sex_bias'] = 'male_predominant'
                    result['derivation_evidence'].append(
                        f"Sex pattern {pat_id}: {pat.get('mechanism', pat.get('anatomic_factor',''))}"
                    )
                    break
        
        # ── Build pathophysiology text ──
        result['pathophysiology'] = " → ".join(path_text)
        
        # ── Deduplicate ──
        result['common_symptoms'] = list(dict.fromkeys(result['common_symptoms']))
        # Add cleaned versions for human-readable output
        result['common_symptoms_clean'] = [
            self._normalize_symptom(s) for s in result['common_symptoms']
        ]
        
        # ── Compute symptom specificity (rough estimate) ──
        # A symptom is HIGHER specificity if it appears in fewer cascades
        result['symptom_specificity'] = self._compute_specificity(result['common_symptoms'])
        
        return result
    

    @staticmethod
    def _normalize_symptom(s: str) -> str:
        """Strip mechanism explanations from symptom names for cleaner output."""
        # Remove _via_..., _from_..., _due_to_... suffixes
        import re as _re
        clean = _re.split(r'_(?:via|from|due_to|by|using|through)_', s, 1)[0]
        # Replace underscores with spaces
        return clean.replace('_', ' ').strip()

    def _compute_specificity(self, symptoms: List[str]) -> Dict[str, str]:
        """Estimate how specific each symptom is — high specificity if it 
        only appears in cascades for one or few diseases."""
        # Count how often each symptom appears across all organ failure modes
        symptom_counts = defaultdict(int)
        for organ, odata in self.organ_function.items():
            for fmode in odata.get('failure_modes', []):
                for step in fmode.get('cascade', []):
                    for k, v in step.items():
                        if k.startswith('produces_symptom'):
                            symptom_counts[v] += 1
        
        result = {}
        for s in symptoms:
            count = symptom_counts.get(s, 1)
            # Normalize substring matches too
            for tracked, c in symptom_counts.items():
                if tracked != s and (s in tracked or tracked in s):
                    count = max(count, c)
                    break
            
            if count == 1:
                result[s] = 'high'
            elif count <= 3:
                result[s] = 'medium'
            else:
                result[s] = 'low'
        return result


# ═══════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    engine = MechanismDerivationEngine()
    
    print("\n" + "="*70)
    print("PHASE B DEMO: Derive clinical knowledge from LAYER 2-4 alone")
    print("="*70)
    
    test_cases = [
        # (organ, failure_mode, pathogen, expected_disease_name)
        ('appendix',          'luminal_obstruction',   None,                            'Appendicitis'),
        ('meninges',          'infection_inflammation', 'neisseria_meningitidis',       'Bacterial Meningitis'),
        ('lungs',             'alveolar_consolidation', 'streptococcus_pneumoniae',     'Pneumonia'),
        ('heart',             'coronary_obstruction_acute', None,                       'Heart Attack'),
        ('bladder',           'infection_cystitis',     'escherichia_coli_uropathogenic', 'Cystitis'),
        ('brain',             'vascular_dysregulation_migraine', None,                  'Migraine'),
        ('pancreas_islets',   'beta_cell_resistance_T2DM', None,                        'Type 2 Diabetes'),
        ('bronchus',          'bronchospasm_asthma',    None,                           'Asthma'),
    ]
    
    for organ, fmode, pathogen, expected in test_cases:
        print(f"\n{'─'*70}")
        print(f"Disease: {expected}")
        print(f"Input: organ='{organ}', failure='{fmode}', pathogen='{pathogen}'")
        print(f"{'─'*70}")
        
        result = engine.derive_disease(organ, fmode, pathogen)
        
        if result.get('error'):
            print(f"  ⚠ {result['error']}")
            continue
        
        print(f"\n  📋 DERIVED common_symptoms:")
        for s in result['common_symptoms'][:8]:
            specificity = result['symptom_specificity'].get(s, '?')
            print(f"     • {s:50s} [{specificity}]")
        
        if result['red_flags']:
            print(f"\n  🚨 DERIVED red_flags:")
            for rf in result['red_flags']:
                print(f"     • {rf['event']}  (time: {rf.get('time_hours')}h)")
        
        if result['complications']:
            print(f"\n  ⚠️ DERIVED complications: {result['complications']}")
        
        if result['age_groups']:
            print(f"\n  👤 DERIVED age_groups:")
            for ag in result['age_groups'][:2]:
                print(f"     • Age {ag['range_years'][0]}-{ag['range_years'][1]}: {ag.get('risk','?')}")
        
        if result.get('sex_bias') and result['sex_bias'] != 'any':
            print(f"\n  ⚧ DERIVED sex_bias: {result['sex_bias']}")
        
        print(f"\n  💭 Evidence chain ({len(result['derivation_evidence'])} steps)")