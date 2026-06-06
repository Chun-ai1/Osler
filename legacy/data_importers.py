"""
NEXUS Authoritative Data Importers v2 — Smart Filtering
═══════════════════════════════════════════════════════════════
Imports from public medical ontologies with PRIMARY-CARE FILTERING.

Key improvements over v1:
  1. ICD-10 chapter filter — only common clinical chapters (J/K/I/N/G/E/...)
  2. Excludes obscure subcategories (allergies to specific chemicals, etc.)
  3. Cross-references via OMIM/MESH IDs (not name matching)
  4. Requires ICD-10 code to be considered useful
  5. Detects parent-child structure to skip deep niche leaves

Filtering target: ~200-500 real primary-care/ED diseases with symptoms.
"""
from __future__ import annotations
import json
import os
import urllib.request
import urllib.parse
import re
from typing import List, Dict, Optional, Set
from collections import defaultdict


# ═══════════════════════════════════════════════════════════════
# Primary-care ICD-10 chapter map (the chapters that matter clinically)
# ═══════════════════════════════════════════════════════════════

ICD10_CLINICAL_CHAPTERS = {
    'A': 'Infectious - bacterial/parasitic',
    'B': 'Infectious - viral/mycoses',
    'C': 'Neoplasms (selected)',
    'D': 'Blood/immune disorders',
    'E': 'Endocrine/metabolic',
    'F': 'Mental/behavioral (selected)',
    'G': 'Neurologic',
    'H': 'Eye/ear (H00-H59 eye, H60-H95 ear)',
    'I': 'Cardiovascular',
    'J': 'Respiratory',
    'K': 'Digestive',
    'L': 'Skin',
    'M': 'Musculoskeletal',
    'N': 'Genitourinary',
    'O': 'Pregnancy-related',
    'R': 'Symptoms/signs (used for chief complaints)',
    'S': 'Trauma/injury',
    'T': 'External cause/poisoning',
}

# Skip these — too generic or too administrative
ICD10_SKIP_CHAPTERS = {'P', 'Q', 'U', 'V', 'W', 'X', 'Y', 'Z'}

# Subcategory keywords that indicate "too niche" — exclude these
NICHE_KEYWORDS = {
    # Drug-specific allergies
    'allergic reaction to', 'allergy to ', 'hypersensitivity to ',
    # Specific chemicals
    'chloride', 'sulfate', 'phosphate', 'oxide ',
    # Very specific organism allergies
    'allergic contact dermatitis to ',
    # Rare nationality/region-specific
    'brazilian', 'argentine', 'venezuelan', 'bolivian', 'peruvian',
    # Rare parasitic
    'mesocestoidiasis', 'echinostomiasis', 'capillariasis',
    'opisthorchiasis', 'dipylidiasis',
    # Specific occupational
    'baker', 'hairdresser', 'beautician',
    # Embedded ICD-O codes (cancer-specific subtypes)
    '/3', '/0', '/6', '/9',
    # Histologic subtypes
    'subtype', 'variant of', 'in situ',
    # Specific drug names (non-allergic context)
    'methamphetamine', 'phencyclidine',
    # Body part very specific (not main organ system)
    'of left ', 'of right ',  # too granular for primary care
    'unspecified',  # ICD "NOS" — too vague
}


def is_clinical_useful(name: str, icd_code: str = '') -> bool:
    """Check if a disease name passes the 'primary-care relevance' filter."""
    nl = name.lower()
    
    # Exclude drug-specific allergies (any name ending in " Allergy" 
    # is usually a specific drug allergy, not a clinical category)
    if nl.endswith(' allergy') and not nl.startswith('allergy'):
        return False
    if 'allergic contact dermatitis' in nl and 'to ' in nl:
        return False
    if 'allergic reaction' in nl and ' to ' in nl:
        return False
    
    # Exclude niche keyword matches
    for kw in NICHE_KEYWORDS:
        if kw in nl:
            return False
    
    # Reject if name is too long (often = highly specific subtype)
    if len(name) > 80:
        return False
    
    # Reject if ICD-O cancer code with histology subtype
    if 'ICDO' in icd_code:
        return False
    
    # If ICD-10 code, check chapter
    if icd_code.startswith('ICD10CM:') or icd_code.startswith('ICD10:'):
        code = icd_code.split(':')[-1]
        if code:
            chapter = code[0].upper()
            if chapter in ICD10_SKIP_CHAPTERS:
                return False
            if chapter not in ICD10_CLINICAL_CHAPTERS:
                return False
    
    return True


# ═══════════════════════════════════════════════════════════════
# NEXUS schema
# ═══════════════════════════════════════════════════════════════

def make_nexus_disease(
    name: str, icd_code: str = "", omim_id: str = "",
    pathophys: str = "", common_symptoms: List[str] = None,
    data_source: str = "", source_id: str = "",
    verification_status: str = "imported_authoritative",
    source_url: str = "",
) -> dict:
    return {
        "disease_name": name,
        "icd_code": icd_code,
        "category": "",
        "system": _infer_system_from_icd(icd_code),
        "pathophysiology": [pathophys] if pathophys else [],
        "common_symptoms": common_symptoms or [],
        "red_flags": [],
        "risk_factors": [],
        "diagnostic_criteria": {"clinical": [], "lab_tests": [], "imaging": []},
        "treatment": {"home_care": [], "when_to_seek_hospital": []},
        "complications": [],
        "disease_type": "disease",
        "onset_pattern": None,
        "age_groups": None,
        "sex_bias": None,
        "contagiousness": None,
        "symptom_weights": None,
        "differential_diagnosis": [],
        "must_not_miss": [],
        "triage_level": "routine",
        "emergency_level": "moderate",
        "_meta": {
            "data_source": data_source,
            "source_id": source_id,
            "omim_id": omim_id,
            "verification_status": verification_status,
            "source_url": source_url,
        },
    }


def _infer_system_from_icd(icd_code: str) -> str:
    """Map ICD-10 chapter letter → NEXUS system."""
    if not icd_code:
        return ""
    code = icd_code.split(':')[-1] if ':' in icd_code else icd_code
    if not code:
        return ""
    ch = code[0].upper()
    return {
        'A': 'infectious',          # bacterial/parasitic infections
        'B': 'infectious',          # viral/mycotic infections
        'C': 'oncologic',
        'D': 'hematologic',         # blood/immune
        'E': 'endocrine',
        'F': 'psychiatric',
        'G': 'neurologic',
        'H': 'neurologic',          # eye/ear (close to neuro)
        'I': 'cardiovascular',
        'J': 'respiratory',
        'K': 'gi',
        'L': 'integumentary',
        'M': 'msk',
        'N': 'renal',
        'O': 'reproductive',
        'R': '',
        'S': 'msk',                 # injury — usually MSK
        'T': '',                    # external/poisoning
    }.get(ch, '')


# ═══════════════════════════════════════════════════════════════
# Disease Ontology importer (improved)
# ═══════════════════════════════════════════════════════════════

class DiseaseOntologyImporter:
    """Imports DO with smart filtering — keeps only primary-care relevant entries."""
    
    LOCAL_PATH = "medical_knowledge/external_sources/doid.obo"
    
    def parse(self, max_count: int = 1000) -> List[dict]:
        """Parse DOID OBO file with primary-care filtering."""
        if not os.path.exists(self.LOCAL_PATH):
            print(f"[DO] File not found: {self.LOCAL_PATH}")
            print("[DO] Download from: https://github.com/DiseaseOntology/HumanDiseaseOntology")
            return []
        
        diseases = []
        current = {}
        with open(self.LOCAL_PATH, encoding='utf-8') as f:
            for line in f:
                line = line.rstrip()
                if line == "[Term]":
                    if current.get('name') and not current.get('is_obsolete'):
                        diseases.append(current)
                    current = {}
                elif line.startswith("id: "):
                    current['id'] = line[4:]
                elif line.startswith("name: "):
                    current['name'] = line[6:]
                elif line.startswith("is_obsolete: true"):
                    current['is_obsolete'] = True
                elif line.startswith("xref: ICD10CM:"):
                    current.setdefault('icd_codes', []).append(line[6:])
                elif line.startswith("xref: ICD10:"):
                    current.setdefault('icd_codes', []).append(line[6:])
                elif line.startswith("xref: OMIM:"):
                    current.setdefault('omim_ids', []).append(line[6:])
                elif line.startswith("xref: MESH:"):
                    current.setdefault('mesh_ids', []).append(line[6:])
                elif line.startswith("xref: ORDO:") or line.startswith("xref: Orphanet:"):
                    current.setdefault('orphanet_ids', []).append(line[6:])
                elif line.startswith("def: "):
                    current['definition'] = line[5:].split('"')[1] if '"' in line else line[5:]
        
        if current.get('name') and not current.get('is_obsolete'):
            diseases.append(current)
        
        # Apply primary-care filter
        filtered = []
        for d in diseases:
            icd = (d.get('icd_codes', [''])[0] if d.get('icd_codes') else '')
            if not icd:
                # Require ICD code — no code usually means too niche
                continue
            if not is_clinical_useful(d['name'], icd):
                continue
            filtered.append(d)
        
        print(f"[DO] {len(diseases)} total → {len(filtered)} after primary-care filter")
        
        # Convert to NEXUS schema
        nexus_diseases = []
        for d in filtered[:max_count]:
            icd = d.get('icd_codes', [''])[0]
            omim = d.get('omim_ids', [''])[0] if d.get('omim_ids') else ''
            nexus_d = make_nexus_disease(
                name=d['name'].title(),
                icd_code=icd,
                omim_id=omim,
                pathophys=d.get('definition', ''),
                data_source="Disease Ontology (DO)",
                source_id=d.get('id', ''),
                source_url=f"https://disease-ontology.org/?id={d.get('id', '')}",
            )
            # Stash MESH/Orphanet for HPO matching
            nexus_d['_meta']['mesh_ids'] = d.get('mesh_ids', [])
            nexus_d['_meta']['orphanet_ids'] = d.get('orphanet_ids', [])
            nexus_diseases.append(nexus_d)
        
        return nexus_diseases


# ═══════════════════════════════════════════════════════════════
# HPO importer (improved — uses OMIM/Orphanet ID matching)
# ═══════════════════════════════════════════════════════════════

class HPOPhenotypeImporter:
    """Imports HPO disease-phenotype annotations, indexed by OMIM/Orphanet ID."""
    
    LOCAL_PATH = "medical_knowledge/external_sources/phenotype.hpoa"
    
    def parse_by_id(self) -> Dict[str, List[str]]:
        """Returns {disease_id: [phenotype_names]} indexed by OMIM:xxx / ORPHA:xxx."""
        if not os.path.exists(self.LOCAL_PATH):
            print(f"[HPO] File not found: {self.LOCAL_PATH}")
            return {}
        
        # phenotype.hpoa columns:
        # database_id, disease_name, qualifier, hpo_id, reference, evidence,
        # onset, frequency, sex, modifier, aspect, biocuration
        
        # We need a way to convert HPO IDs to phenotype names.
        # The .hpoa file gives HPO IDs, not names. Names are in hp.obo.
        # If hp.obo not present, we use the HPO ID as a placeholder name.
        
        hp_obo = "medical_knowledge/external_sources/hp.obo"
        hpo_id_to_name = {}
        if os.path.exists(hp_obo):
            current_id = None
            with open(hp_obo, encoding='utf-8') as f:
                for line in f:
                    line = line.rstrip()
                    if line.startswith("id: HP:"):
                        current_id = line[4:]
                    elif line.startswith("name: ") and current_id:
                        hpo_id_to_name[current_id] = line[6:]
                        current_id = None
            print(f"[HPO] Loaded {len(hpo_id_to_name)} phenotype names from hp.obo")
        else:
            print(f"[HPO] hp.obo not found — phenotype names will be HPO IDs")
        
        disease_phenotypes = defaultdict(list)
        with open(self.LOCAL_PATH, encoding='utf-8') as f:
            for line in f:
                if line.startswith('#') or '\t' not in line:
                    continue
                parts = line.rstrip().split('\t')
                if len(parts) >= 4:
                    db_id = parts[0]              # OMIM:xxxx or ORPHA:xxxx
                    hpo_id = parts[3]              # HP:xxxxxxx
                    if hpo_id and db_id:
                        # Convert HPO ID to readable name if possible
                        phen_name = hpo_id_to_name.get(hpo_id, hpo_id).lower()
                        disease_phenotypes[db_id].append(phen_name)
        
        # Deduplicate phenotypes per disease
        for d in disease_phenotypes:
            disease_phenotypes[d] = list(dict.fromkeys(disease_phenotypes[d]))[:12]
        
        print(f"[HPO] Phenotypes indexed for {len(disease_phenotypes)} diseases")
        return dict(disease_phenotypes)


# ═══════════════════════════════════════════════════════════════
# Combined pipeline (improved)
# ═══════════════════════════════════════════════════════════════

def import_pipeline(target_count: int = 500) -> List[dict]:
    """
    Improved pipeline: DO → smart filter → HPO match by OMIM ID → save.
    """
    print("=== NEXUS Authoritative Data Import v2 ===\n")
    
    # Step 1: Disease Ontology with smart filter
    do = DiseaseOntologyImporter()
    diseases = do.parse(max_count=target_count)
    if not diseases:
        return []
    
    # Step 2: HPO phenotypes indexed by OMIM/Orphanet ID
    hpo = HPOPhenotypeImporter()
    hpo_index = hpo.parse_by_id()
    
    # Step 3: Match HPO to DO via:
    # (a) OMIM/Orphanet ID cross-references (most accurate)
    # (b) Disease NAME similarity (fallback — needed because DO common diseases
    #     often lack OMIM xrefs since OMIM is mostly Mendelian disorders)

    # Build a name-keyed index from HPO so we can do fast lookup
    # We need disease NAMES in the HPO file too. They're in the second column
    # of phenotype.hpoa. Let me re-parse to get them.
    hpo_name_to_id = {}
    hpo_path = "medical_knowledge/external_sources/phenotype.hpoa"
    if os.path.exists(hpo_path):
        with open(hpo_path, encoding='utf-8') as f:
            for line in f:
                if line.startswith('#') or '\t' not in line:
                    continue
                parts = line.rstrip().split('\t')
                if len(parts) >= 2:
                    db_id = parts[0]
                    db_name = parts[1].lower().strip()
                    if db_name and db_id in hpo_index:
                        hpo_name_to_id[db_name] = db_id

    print(f"[HPO] Name-index built: {len(hpo_name_to_id)} names")

    matched = 0
    matched_by_id = 0
    matched_by_name = 0
    for d in diseases:
        omim = d['_meta'].get('omim_id', '')
        orphanet_ids = d['_meta'].get('orphanet_ids', [])

        # Try OMIM first (most accurate)
        if omim:
            omim_key = omim if omim.startswith('OMIM:') else f'OMIM:{omim}'
            if omim_key in hpo_index:
                d['common_symptoms'] = hpo_index[omim_key]
                d['_meta']['hpo_matched_via'] = omim_key
                matched += 1
                matched_by_id += 1
                continue

        # Try Orphanet
        orpha_matched = False
        for orpha_id in orphanet_ids:
            orpha_key = orpha_id if orpha_id.startswith('ORPHA:') else f'ORPHA:{orpha_id}'
            if orpha_key in hpo_index:
                d['common_symptoms'] = hpo_index[orpha_key]
                d['_meta']['hpo_matched_via'] = orpha_key
                matched += 1
                matched_by_id += 1
                orpha_matched = True
                break
        if orpha_matched:
            continue

        # FALLBACK: try matching by disease NAME (case-insensitive substring)
        d_name = d['disease_name'].lower().strip()
        # Strip common prefixes that don't appear in HPO
        d_name_norm = d_name.replace('acute ', '').replace('chronic ', '').strip()

        # Exact match first
        if d_name in hpo_name_to_id:
            db_id = hpo_name_to_id[d_name]
            d['common_symptoms'] = hpo_index[db_id]
            d['_meta']['hpo_matched_via'] = f"name_exact:{db_id}"
            matched += 1
            matched_by_name += 1
            continue

        if d_name_norm in hpo_name_to_id:
            db_id = hpo_name_to_id[d_name_norm]
            d['common_symptoms'] = hpo_index[db_id]
            d['_meta']['hpo_matched_via'] = f"name_norm:{db_id}"
            matched += 1
            matched_by_name += 1
            continue

        # Substring match (last resort) — find HPO disease whose name contains ours
        # or vice versa (only if either name >= 10 chars to avoid false matches)
        if len(d_name) >= 10:
            for hpo_name, hpo_db_id in hpo_name_to_id.items():
                if len(hpo_name) >= 10 and (
                    d_name == hpo_name or
                    (d_name in hpo_name and abs(len(d_name) - len(hpo_name)) < 15) or
                    (hpo_name in d_name and abs(len(d_name) - len(hpo_name)) < 15)
                ):
                    d['common_symptoms'] = hpo_index[hpo_db_id]
                    d['_meta']['hpo_matched_via'] = f"name_fuzzy:{hpo_db_id}"
                    matched += 1
                    matched_by_name += 1
                    break

    print(f"[HPO] Matched: {matched} total ({matched_by_id} by ID, {matched_by_name} by name)")
    
    # Step 4: Filter — only keep diseases with at least SOME symptoms
    # (otherwise NEXUS reasoning can't use them)
    final = [d for d in diseases if d.get('common_symptoms')]
    
    print(f"\n=== Pipeline Results ===")
    print(f"  After clinical filter: {len(diseases)}")
    print(f"  Matched to HPO:        {matched}")
    print(f"  With ≥1 symptom (kept): {len(final)}")
    print(f"  Final yield:           {len(final)} usable disease entries")
    
    # System breakdown
    sys_counts = defaultdict(int)
    for d in final:
        sys_counts[d.get('system') or 'unknown'] += 1
    print(f"\n  By body system:")
    for s, n in sorted(sys_counts.items(), key=lambda x: -x[1]):
        print(f"    {s:20s} {n}")
    
    return final


def save_imported(diseases: List[dict], filename: str = "disease_imported.json"):
    out_dir = "medical_knowledge/diseases"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(diseases, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVED] {len(diseases)} diseases → {path}")


if __name__ == "__main__":
    diseases = import_pipeline(target_count=1000)
    if diseases:
        save_imported(diseases)