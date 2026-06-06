"""
Symptom Weight Deriver — Auto-compute symptom_weights from cascade data
═══════════════════════════════════════════════════════════════
You're right: weights should not be hand-set. They should emerge from
information theory + the cascade library + the RL classifier.

This module derives symptom_weights using THREE complementary sources:

1. CASCADE-BASED SPECIFICITY (information-theoretic, no training needed)
   For each symptom S, count how many disease cascades produce it (DF).
   weight ∝ log(N_diseases / DF) — TF-IDF style
   Rare symptoms (1 disease) → high weight
   Common symptoms (10 diseases) → low weight

2. CO-OCCURRENCE INFORMATION GAIN (when registry is populated)
   For each (symptom, disease) pair, compute mutual information:
   How much does seeing the symptom reduce uncertainty about the disease?

3. RL-LEARNED IMPORTANCE (when nexus_learning_env classifier exists)
   Optional: load the trained classifier weights, extract first-layer
   gradients to estimate symptom importance per disease.

Output: medical_knowledge/derived/symptom_weights.json
Consumers: nexus_medical Step 5e reads this if present, falls back to
           hardcoded weights in disease_0001.json otherwise.
"""
from __future__ import annotations
import json
import os
import math
from typing import Dict, List, Tuple, Set
from collections import defaultdict


class WeightDeriver:
    """Derives symptom_weights from cascade structure + registry + (optional) classifier."""
    
    DEFAULT_PATHS = {
        'organ_function':    'medical_knowledge/layer2_physiology/organ_function.json',
        'mechanism_map':     'medical_knowledge/registry/disease_mechanism_map.json',
        'output':            'medical_knowledge/derived/symptom_weights.json',
        'pathogens':         'medical_knowledge/layer3_pathogen/pathogens.json',
        'immune':            'medical_knowledge/layer3_pathogen/immune_responses.json',
    }
    
    def __init__(self):
        self.organ_function: Dict = {}
        self.mechanism_map: Dict = {}
        self.pathogens: Dict = {}
        self.immune: Dict = {}
        self._load()
    
    def _load(self):
        for key, path in self.DEFAULT_PATHS.items():
            if key == 'output':
                continue
            for p in [path, f"../{path}",
                      os.path.join(os.path.dirname(__file__), '..', path)]:
                if os.path.exists(p):
                    try:
                        data = json.load(open(p, encoding='utf-8'))
                        if key == 'organ_function':
                            self.organ_function = data.get('organs', {})
                        elif key == 'mechanism_map':
                            self.mechanism_map = data.get('mappings', {})
                        elif key == 'pathogens':
                            self.pathogens = data.get('pathogens', {})
                        elif key == 'immune':
                            self.immune = data.get('responses', {})
                        break
                    except Exception:
                        pass
        print(f"[WeightDeriver] Loaded: "
              f"organs={len(self.organ_function)}, "
              f"diseases={len(self.mechanism_map)}, "
              f"pathogens={len(self.pathogens)}")
    
    # ═══════════════════════════════════════════════════════════
    # Method 1: cascade-based specificity (TF-IDF)
    # ═══════════════════════════════════════════════════════════
    
    def _gather_disease_symptoms(self) -> Dict[str, Set[str]]:
        """For each registered disease → set of symptoms its cascade produces."""
        result = {}
        for disease_name, mapping in self.mechanism_map.items():
            organ = mapping.get('organ', '')
            failure_mode = mapping.get('failure_mode', '')
            pathogen = mapping.get('pathogen')
            
            symptoms = set()
            organ_data = self.organ_function.get(organ, {})
            for fm in organ_data.get('failure_modes', []):
                if fm.get('mode') != failure_mode:
                    continue
                for step in fm.get('cascade', []):
                    for k, v in step.items():
                        if k.startswith('produces_symptom'):
                            # Normalize to a comparable form
                            clean = self._normalize(v)
                            symptoms.add(clean)
            
            # Also pathogen direct symptoms
            if pathogen and pathogen in self.pathogens:
                pdata = self.pathogens[pathogen]
                for s in pdata.get('produces_symptoms_direct', []):
                    symptoms.add(self._normalize(s))
            
            # Immune-driven symptoms (for infections)
            if 'infection' in failure_mode.lower():
                imm_pattern = ('acute_bacterial_inflammation' if pathogen and 'bacteri' in str(pathogen).lower()
                                else 'viral_response')
                imm = self.immune.get(imm_pattern, {})
                for effect in imm.get('produces_systemic_effects', []):
                    eff = effect.get('effect')
                    if eff:
                        symptoms.add(self._normalize(eff))
            
            result[disease_name] = symptoms
        return result
    
    @staticmethod
    def _normalize(symptom: str) -> str:
        """Normalize symptom string for cross-disease comparison.
        Strips _via_*, _T*_dermatome, etc., keeps the core symptom name."""
        # Strip mechanism suffix
        core = symptom.split('_via_')[0]
        # Strip dermatome/level annotation
        for marker in ['_T1', '_T2', '_T3', '_T4', '_T5', '_T6', '_T7', '_T8', '_T9',
                        '_T10', '_T11', '_T12', '_L1', '_L2', '_S2', '_C3',
                        '_visceral', '_somatic', '_referral', '_dermatome']:
            core = core.split(marker)[0]
        return core.strip('_').lower()
    
    def derive_specificity_weights(self) -> Dict[str, Dict[str, dict]]:
        """
        Compute TF-IDF-like specificity weights.
        
        Returns: {disease_name: {symptom: {weight, specificity, evidence}}}
        """
        disease_symptoms = self._gather_disease_symptoms()
        n_diseases = len(disease_symptoms)
        
        if n_diseases == 0:
            return {}
        
        # Document frequency for each symptom
        df: Dict[str, int] = defaultdict(int)
        for syms in disease_symptoms.values():
            for s in syms:
                df[s] += 1
        
        # Compute weights per disease per symptom
        result = {}
        for disease, syms in disease_symptoms.items():
            disease_weights = {}
            for s in syms:
                # Inverse Document Frequency
                # If symptom appears in K diseases out of N → idf = log(N/K) + 1
                idf = math.log(max(n_diseases, 1) / max(df[s], 1)) + 1.0
                
                # Map to weight scale 1-5 and specificity tier
                if df[s] == 1:
                    weight = 5
                    specificity = "very_high"
                elif df[s] <= 2:
                    weight = 4
                    specificity = "high"
                elif df[s] <= n_diseases * 0.3:
                    weight = 3
                    specificity = "medium"
                elif df[s] <= n_diseases * 0.6:
                    weight = 2
                    specificity = "low"
                else:
                    weight = 1
                    specificity = "very_low"
                
                disease_weights[s] = {
                    'weight': weight,
                    'specificity': specificity,
                    'idf': round(idf, 2),
                    'document_frequency': df[s],
                    'evidence': f"appears in {df[s]}/{n_diseases} disease cascades",
                }
            result[disease] = disease_weights
        
        return result
    
    # ═══════════════════════════════════════════════════════════
    # Method 2: pairwise mutual information (when more data exists)
    # ═══════════════════════════════════════════════════════════
    
    def derive_mutual_info(self, virtual_patients_per_disease: int = 100) -> Dict[str, Dict[str, float]]:
        """
        Generate virtual patients from each disease's cascade,
        then compute MI(symptom, disease) over the synthetic dataset.
        Higher MI = symptom is more discriminative for that disease.
        """
        # This is more advanced — generate patients with noise, compute conditional probs
        # Stub for now: Phase B.2 work
        pass
    
    # ═══════════════════════════════════════════════════════════
    # Method 3: optional RL-learned weights
    # ═══════════════════════════════════════════════════════════
    
    def load_rl_weights(self, checkpoint_path: str = "nexus_checkpoint.npz") -> Dict[str, Dict[str, float]]:
        """
        If a trained classifier exists, extract symptom→disease importance
        via input-layer weight magnitudes. Optional enhancement.
        """
        if not os.path.exists(checkpoint_path):
            return {}
        try:
            import numpy as np
            ckpt = np.load(checkpoint_path, allow_pickle=True)
            # Extract first-layer weights to estimate symptom importance per disease
            # (full implementation requires knowing feature layout)
            return {}  # stub
        except Exception:
            return {}
    
    # ═══════════════════════════════════════════════════════════
    # Save derived weights
    # ═══════════════════════════════════════════════════════════
    
    def save(self, path: str = None) -> str:
        path = path or self.DEFAULT_PATHS['output']
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        weights = self.derive_specificity_weights()
        
        payload = {
            "_meta": {
                "description": "Auto-derived symptom_weights from cascade information theory",
                "method": "TF-IDF specificity (rare symptoms = high weight)",
                "n_diseases_analyzed": len(weights),
                "regenerate_with": "python3 nexus_engine/weight_deriver.py",
                "consumed_by": "nexus_medical Step 5e (clinical specificity)",
                "version": "1.0",
            },
            "disease_weights": weights,
        }
        
        with open(path, 'w') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return path


# ═══════════════════════════════════════════════════════════════
# CLI / demo
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    deriver = WeightDeriver()
    
    print("\n" + "="*70)
    print("AUTO-DERIVING symptom_weights from cascade information")
    print("="*70)
    
    weights = deriver.derive_specificity_weights()
    
    if not weights:
        print("\nNo diseases found. Make sure disease_mechanism_map.json is populated.")
    else:
        print(f"\nDerived weights for {len(weights)} diseases\n")
        
        for disease, sym_weights in weights.items():
            print(f"\n  {disease}")
            print(f"  {'─' * 60}")
            sorted_syms = sorted(sym_weights.items(), key=lambda x: (-x[1]['weight'], x[0]))
            for sym, info in sorted_syms[:6]:
                print(f"    {sym:35s}  weight={info['weight']}  ({info['specificity']:10s})  "
                      f"in {info['document_frequency']} disease(s)")
        
        # Save
        path = deriver.save()
        print(f"\n✓ Saved: {path}")