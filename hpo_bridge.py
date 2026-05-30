"""
HPO ↔ NEXUS Vocabulary Bridge
═══════════════════════════════════════════════════════════════
HPO uses its own phenotype vocabulary (HP:0002090 = "Pneumonia")
NEXUS uses anatomy + symptom vocabularies (organs, body zones, symptoms)

Without a bridge, when GNN sees:
   disease X → HP:0002090
It learns nothing useful, because HP:0002090 has no other connections.

This module CONNECTS HPO phenotypes to NEXUS organs/symptoms by:
  1. Direct name match (HP:0002090 "Pneumonia" → NEXUS symptom "pneumonia")
  2. Substring match (HP:0010741 "Pneumonia susceptibility" → "pneumonia")  
  3. Anatomy keyword extraction (HP:0001824 "Weight loss" → no anatomy)
  4. HPO is_a hierarchy walk (if direct fails, walk up to abstract parent)

After bridging, the GNN sees HPO phenotypes as nodes connected to BOTH
diseases (from HPO data) AND organs/symptoms (from this bridge),
allowing it to actually learn from the HPO data.
"""
from __future__ import annotations
import json
import os
import re
from typing import Dict, List, Set, Optional


# ═══════════════════════════════════════════════════════════════
# Anatomy keywords for phenotype name parsing
# ═══════════════════════════════════════════════════════════════

# Maps anatomy keywords to NEXUS organ IDs.
# Conservative — only well-established mappings.
ANATOMY_KEYWORDS = {
    # Cardiovascular
    'heart':           ['heart'],
    'cardiac':         ['heart'],
    'myocardial':      ['heart'],
    'pericardial':     ['heart'],
    'coronary':        ['lad_a', 'rca_a', 'lcx_a', 'left_main_coronary'],
    'aortic':          ['aorta'],
    'aorta':           ['aorta'],
    'arterial':        ['aorta'],
    'venous':          [],  # too generic
    
    # Respiratory  
    'pulmonary':       ['r_lung', 'l_lung', 'lungs'],
    'lung':            ['r_lung', 'l_lung', 'lungs'],
    'respiratory':     ['r_lung', 'l_lung', 'lungs'],
    'bronch':          ['r_bronchus', 'l_bronchus'],
    'tracheal':        ['trachea'],
    'pleural':         ['pleura'],
    
    # GI
    'hepatic':         ['liver'],
    'liver':           ['liver'],
    'biliary':         ['bile_duct', 'gallbladder'],
    'gallbladder':     ['gallbladder'],
    'pancreatic':      ['pancreas'],
    'pancreas':        ['pancreas'],
    'gastric':         ['stomach'],
    'stomach':         ['stomach'],
    'esophageal':      ['esophagus'],
    'esophagus':       ['esophagus'],
    'duodenal':        ['duodenum'],
    'jejunal':         ['jejunum'],
    'ileal':           ['ileum'],
    'colonic':         ['asc_colon', 'desc_colon', 'trans_colon', 'sig_colon'],
    'rectal':          ['rectum'],
    'appendiceal':     ['appendix'],
    'appendix':        ['appendix'],
    'intestinal':      ['jejunum', 'ileum', 'duodenum'],
    'bowel':           ['jejunum', 'ileum', 'asc_colon', 'desc_colon'],
    
    # Renal/urinary
    'renal':           ['r_kidney', 'l_kidney'],
    'kidney':          ['r_kidney', 'l_kidney'],
    'ureter':          ['r_ureter', 'l_ureter'],
    'bladder':         ['bladder'],
    'urethr':          ['urethra'],
    'prostat':         ['prostate'],
    
    # Neurologic
    'cerebral':        ['brain'],
    'cerebellar':      ['cerebellum'],
    'brain':           ['brain'],
    'cortical':        ['brain'],
    'subcortical':     ['brain'],
    'meningeal':       ['meninges'],
    'spinal':          ['spinal_cord'],
    'cranial':         ['brain', 'brainstem'],
    'optic':           ['ophthalmic_a'],
    
    # Endocrine
    'thyroid':         ['thyroid'],
    'adrenal':         ['r_adrenal', 'l_adrenal'],
    'pituitary':       ['pituitary'],
    
    # Reproductive
    'uterine':         ['uterus'],
    'ovarian':         ['ovaries'],
    'testicular':      ['testes'],
    
    # Skin/MSK
    'cutaneous':       ['skin'],
    'dermal':          ['skin'],
    'skin':            ['skin'],
    'muscular':        ['muscles'],
    'skeletal':        ['bones'],
    'osseous':         ['bones'],
    'bone':            ['bones'],
}

# Common symptom synonyms — HPO phenotype name → NEXUS symptom name
SYMPTOM_SYNONYMS = {
    'fever':                   'fever',
    'pyrexia':                 'fever',
    'hyperthermia':            'fever',
    'cough':                   'cough',
    'dyspnea':                 'shortness of breath',
    'shortness of breath':     'shortness of breath',
    'breath difficulty':       'shortness of breath',
    'chest pain':              'chest pain',
    'thoracic pain':           'chest pain',
    'abdominal pain':          'abdominal pain',
    'belly pain':              'abdominal pain',
    'stomach pain':            'abdominal pain',
    'headache':                'headache',
    'cephalgia':               'headache',
    'migraine':                'headache',
    'nausea':                  'nausea',
    'vomiting':                'vomiting',
    'emesis':                  'vomiting',
    'diarrhea':                'diarrhea',
    'fatigue':                 'fatigue',
    'weakness':                'weakness',
    'asthenia':                'weakness',
    'dizziness':               'dizziness',
    'vertigo':                 'dizziness',
    'syncope':                 'syncope',
    'fainting':                'syncope',
    'wheezing':                'wheezing',
    'palpitations':            'palpitations',
    'tachycardia':             'palpitations',
    'sweating':                'sweating',
    'diaphoresis':             'sweating',
    'chills':                  'chills',
    'rigor':                   'chills',
    'rash':                    'rash',
    'erythema':                'rash',
    'jaundice':                'jaundice',
    'icterus':                 'jaundice',
    'confusion':               'confusion',
    'altered mental status':   'confusion',
    'lethargy':                'fatigue',
    'malaise':                 'fatigue',
    'weight loss':             'weight loss',
    'cachexia':                'weight loss',
    'anorexia':                'loss of appetite',
}


# ═══════════════════════════════════════════════════════════════
# HPO bridge builder
# ═══════════════════════════════════════════════════════════════

class HPOBridge:
    """
    Builds bridge edges connecting HPO phenotypes to NEXUS vocabulary.
    
    Output: 3 new edge sets that can be added to KnowledgeGraphV2:
      • phenotype_indicates_organ:  HPO phenotype → NEXUS organ
      • phenotype_indicates_symptom: HPO phenotype → NEXUS symptom
      • phenotype_in_zone:          HPO phenotype → NEXUS zone (via organ)
    """
    
    HP_OBO_PATH = "medical_knowledge/external_sources/hp.obo"
    
    def __init__(self, base_graph_path: str = "medical_knowledge/graph/knowledge_graph.json"):
        # Load NEXUS vocabularies for matching
        self.nexus_organs: Set[str] = set()
        self.nexus_symptoms: Set[str] = set()
        self.nexus_zones: Set[str] = set()
        self.organ_to_zones: Dict[str, List[str]] = {}
        
        if os.path.exists(base_graph_path):
            data = json.load(open(base_graph_path))
            edges = data.get('edges', {})
            
            for src, dsts in edges.get('disease_affects_organ', {}).items():
                self.nexus_organs.update(dsts)
            for src, dsts in edges.get('symptom_indicates_organ', {}).items():
                self.nexus_organs.update(dsts)
                self.nexus_symptoms.add(src)
            for src, dsts in edges.get('disease_in_zone', {}).items():
                self.nexus_zones.update(dsts)
            for src, dsts in edges.get('organ_in_zone', {}).items():
                self.organ_to_zones.setdefault(src, []).extend(dsts)
        
        print(f"[HPO Bridge] NEXUS vocabulary: "
              f"{len(self.nexus_organs)} organs, "
              f"{len(self.nexus_symptoms)} symptoms, "
              f"{len(self.nexus_zones)} zones")
    
    def parse_hp_obo(self) -> Dict[str, dict]:
        """Read hp.obo, return {hpo_id: {name, parents}} for all phenotypes."""
        if not os.path.exists(self.HP_OBO_PATH):
            print(f"[HPO Bridge] hp.obo not found at {self.HP_OBO_PATH}")
            return {}
        
        phenotypes = {}
        current = {}
        current_id = None
        
        with open(self.HP_OBO_PATH, encoding='utf-8') as f:
            for line in f:
                line = line.rstrip()
                if line == "[Term]":
                    if current_id:
                        phenotypes[current_id] = current
                    current = {'parents': [], 'synonyms': []}
                    current_id = None
                elif line.startswith("id: HP:"):
                    current_id = line[4:]
                elif line.startswith("name: ") and current_id:
                    current['name'] = line[6:]
                elif line.startswith("is_a: HP:"):
                    parent = line.split('!')[0].strip()[6:]
                    current['parents'].append(parent)
                elif line.startswith("synonym: "):
                    # synonym: "Acute myocardial infarction" EXACT [...]
                    m = re.match(r'synonym:\s*"([^"]+)"', line)
                    if m:
                        current['synonyms'].append(m.group(1).lower())
        
        if current_id:
            phenotypes[current_id] = current
        
        print(f"[HPO Bridge] Parsed {len(phenotypes)} HPO phenotypes")
        return phenotypes
    
    def map_phenotype(self, hpo_id: str, info: dict,
                      hpo_db: Dict[str, dict]) -> dict:
        """
        For a single phenotype, find its NEXUS vocabulary mappings.
        Returns: {organs: [...], symptoms: [...], zones: [...], match_method: ...}
        """
        result = {'organs': set(), 'symptoms': set(), 'zones': set(),
                  'match_method': 'none'}
        
        names_to_try = [info.get('name', '').lower()] + info.get('synonyms', [])
        names_to_try = [n.strip() for n in names_to_try if n.strip()]
        
        # ── Try direct symptom synonym ──
        for name in names_to_try:
            if name in SYMPTOM_SYNONYMS:
                result['symptoms'].add(SYMPTOM_SYNONYMS[name])
                result['match_method'] = 'symptom_synonym'
                # Continue to also check anatomy
        
        # ── Anatomy keyword extraction ──
        for name in names_to_try:
            for keyword, organs in ANATOMY_KEYWORDS.items():
                if keyword in name:
                    valid_organs = [o for o in organs if o in self.nexus_organs]
                    result['organs'].update(valid_organs)
                    if valid_organs:
                        result['match_method'] = 'anatomy_keyword' if not result['organs'] else result['match_method']
        
        # ── Direct match against NEXUS symptom names ──
        for name in names_to_try:
            for sym in self.nexus_symptoms:
                if name == sym or sym in name:
                    result['symptoms'].add(sym)
                    if result['match_method'] == 'none':
                        result['match_method'] = 'direct_symptom'
                    break
        
        # ── Walk up HPO is_a hierarchy if no match ──
        if not result['organs'] and not result['symptoms']:
            visited = {hpo_id}
            stack = list(info.get('parents', []))
            depth = 0
            while stack and depth < 5:  # max 5 levels up
                next_stack = []
                for parent_id in stack:
                    if parent_id in visited or parent_id not in hpo_db:
                        continue
                    visited.add(parent_id)
                    p_info = hpo_db[parent_id]
                    p_names = [p_info.get('name', '').lower()] + p_info.get('synonyms', [])
                    
                    # Check if any parent matches
                    matched = False
                    for pn in p_names:
                        for keyword, organs in ANATOMY_KEYWORDS.items():
                            if keyword in pn:
                                valid_organs = [o for o in organs if o in self.nexus_organs]
                                result['organs'].update(valid_organs)
                                if valid_organs:
                                    result['match_method'] = f'parent_anatomy_d{depth+1}'
                                    matched = True
                        if pn in SYMPTOM_SYNONYMS:
                            result['symptoms'].add(SYMPTOM_SYNONYMS[pn])
                            result['match_method'] = f'parent_symptom_d{depth+1}'
                            matched = True
                    
                    if not matched:
                        next_stack.extend(p_info.get('parents', []))
                
                stack = next_stack
                depth += 1
                if result['organs'] or result['symptoms']:
                    break
        
        # ── Derive zones from organs ──
        for organ in result['organs']:
            if organ in self.organ_to_zones:
                result['zones'].update(self.organ_to_zones[organ])
        
        # Convert sets to sorted lists for JSON serialization
        result['organs'] = sorted(result['organs'])
        result['symptoms'] = sorted(result['symptoms'])
        result['zones'] = sorted(result['zones'])
        return result
    
    def build_bridge(self, output_path: str = "medical_knowledge/graph/hpo_bridge.json"):
        """Build the full HPO bridge and save to JSON."""
        hpo_db = self.parse_hp_obo()
        if not hpo_db:
            print("[HPO Bridge] No HPO data — bridge not built")
            return None
        
        print(f"[HPO Bridge] Building bridge for {len(hpo_db)} phenotypes...")
        
        bridge = {}
        method_counts = {}
        for hpo_id, info in hpo_db.items():
            mapping = self.map_phenotype(hpo_id, info, hpo_db)
            method = mapping.pop('match_method')
            method_counts[method] = method_counts.get(method, 0) + 1
            
            if mapping['organs'] or mapping['symptoms']:
                bridge[hpo_id] = {
                    'name': info.get('name', ''),
                    **mapping,
                    'match_method': method,
                }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump({
                '_meta': {
                    'description': 'HPO phenotype → NEXUS vocabulary bridge',
                    'total_hpo_phenotypes': len(hpo_db),
                    'mapped_phenotypes':    len(bridge),
                    'mapping_methods':      method_counts,
                },
                'phenotypes': bridge,
            }, f, indent=2, ensure_ascii=False)
        
        print(f"[HPO Bridge] Saved: {output_path}")
        print(f"  Mapped: {len(bridge)}/{len(hpo_db)} ({100*len(bridge)/len(hpo_db):.1f}%)")
        print(f"  By method:")
        for method, count in sorted(method_counts.items(), key=lambda x: -x[1]):
            print(f"    {method:30s} {count}")
        
        return bridge


if __name__ == "__main__":
    bridge = HPOBridge()
    bridge.build_bridge()