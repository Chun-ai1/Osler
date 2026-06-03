"""
External Ontology Importer for NEXUS
=====================================
Importers for ontology files that the user downloads locally.

The sandbox couldn't reach raw.githubusercontent.com or purl.obolibrary.org,
so the user must download these files first:

  UBERON (~50 MB):
    https://github.com/obophenotype/uberon/releases/latest/download/uberon.obo
    
  MONDO Disease Ontology (~30 MB):
    https://github.com/monarch-initiative/mondo/releases/latest/download/mondo.obo
  
  FMA (Foundational Model of Anatomy, ~200+ MB OWL):
    https://bioportal.bioontology.org/ontologies/FMA
    (requires free academic registration)

Save these to: medical_knowledge/external_sources/

USAGE:
    python3 nexus_engine/external_ontology_importer.py \\
        --uberon medical_knowledge/external_sources/uberon.obo \\
        --mondo medical_knowledge/external_sources/mondo.obo

OUTPUT:
    medical_knowledge/external_ontologies/
      uberon_terms.json          (all UBERON anatomical structures)
      uberon_parents.json        (parent-child relationships)
      uberon_to_nexus.json       (auto-mapped to existing NEXUS organs)
      mondo_diseases.json        (disease list with synonyms)
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
from pathlib import Path


# Reuse existing parse_hp_obo logic for OBO format
def parse_obo(obo_path: str, id_prefix: str = "UBERON") -> dict:
    """Generic OBO parser. Returns {term_id: {name, parents, synonyms, xrefs, definition}}."""
    if not os.path.exists(obo_path):
        raise FileNotFoundError(f"{obo_path} not found")
    
    terms = {}
    current = {}
    current_id = None
    in_term_stanza = False
    
    print(f"[OBO Parser] Reading {obo_path}...")
    line_count = 0
    
    with open(obo_path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            line_count += 1
            line = line.rstrip()
            
            if line == "[Term]":
                if current_id:
                    terms[current_id] = current
                current = {'parents': [], 'synonyms': [], 'xrefs': []}
                current_id = None
                in_term_stanza = True
                continue
            
            if line.startswith("[") and line.endswith("]"):
                # Other stanza type (e.g. [Typedef]) — flush current
                if current_id:
                    terms[current_id] = current
                current_id = None
                in_term_stanza = False
                continue
            
            if not in_term_stanza:
                continue
            
            if line.startswith(f"id: {id_prefix}:"):
                current_id = line[4:]
            elif line.startswith("name: ") and current_id:
                current['name'] = line[6:]
            elif line.startswith(f"is_a: {id_prefix}:") and current_id:
                parent = line.split('!')[0].strip()[6:]
                current['parents'].append(parent)
            elif line.startswith("def: ") and current_id:
                m = re.search(r'"([^"]+)"', line)
                if m:
                    current['definition'] = m.group(1)
            elif line.startswith("synonym: ") and current_id:
                m = re.search(r'"([^"]+)"', line)
                if m:
                    current['synonyms'].append(m.group(1))
            elif line.startswith("xref: ") and current_id:
                current['xrefs'].append(line[6:].strip())
            elif line.startswith("is_obsolete: true"):
                current['obsolete'] = True
    
    if current_id:
        terms[current_id] = current
    
    # Filter out obsolete
    active = {k: v for k, v in terms.items() if not v.get('obsolete')}
    print(f"[OBO Parser] {line_count} lines, {len(terms)} terms ({len(active)} active)")
    return active


def import_uberon(obo_path: str, output_dir: str) -> dict:
    """Import UBERON anatomy ontology."""
    print(f"\n[UBERON Importer] Processing {obo_path}")
    
    terms = parse_obo(obo_path, id_prefix="UBERON")
    
    # Filter to human-relevant entries
    # UBERON has many cross-species terms; we want human anatomical structures
    
    # Save full
    with open(os.path.join(output_dir, "uberon_terms.json"), 'w') as f:
        json.dump({
            "_meta": {
                "source": "UBERON OBO",
                "total_terms": len(terms),
            },
            "terms": terms,
        }, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ Saved uberon_terms.json: {len(terms)} terms")
    return terms


def import_mondo(obo_path: str, output_dir: str) -> dict:
    """Import MONDO disease ontology."""
    print(f"\n[MONDO Importer] Processing {obo_path}")
    
    terms = parse_obo(obo_path, id_prefix="MONDO")
    
    with open(os.path.join(output_dir, "mondo_diseases.json"), 'w') as f:
        json.dump({
            "_meta": {
                "source": "MONDO OBO",
                "total_diseases": len(terms),
            },
            "diseases": terms,
        }, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ Saved mondo_diseases.json: {len(terms)} diseases")
    return terms


def auto_map_uberon_to_nexus(uberon_terms: dict, nexus_atlas_path: str,
                              output_dir: str) -> dict:
    """Auto-map UBERON anatomical IDs to existing NEXUS atlas organs."""
    print("\n[UBERON → NEXUS Mapping]")
    
    # Load NEXUS atlas
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from anatomy_atlas import AnatomyAtlas
        atlas = AnatomyAtlas()
        nexus_organs = list(atlas.organs.keys())
    except Exception as e:
        print(f"  Cannot load atlas: {e}")
        return {}
    
    print(f"  NEXUS atlas: {len(nexus_organs)} organs")
    
    mappings = {}
    
    # For each NEXUS organ, search UBERON for matching term
    # Build search index from UBERON
    uberon_by_name = {}
    for uid, info in uberon_terms.items():
        name = info.get('name', '').lower().strip()
        if name:
            uberon_by_name.setdefault(name, []).append(uid)
        for syn in info.get('synonyms', []):
            uberon_by_name.setdefault(syn.lower().strip(), []).append(uid)
    
    matched = 0
    for organ in nexus_organs:
        # Normalize organ name (NEXUS uses underscores)
        normalized = organ.replace('_', ' ').lower()
        
        # Try exact match
        if normalized in uberon_by_name:
            mappings[organ] = {
                "uberon_id": uberon_by_name[normalized][0],
                "match": "exact_name",
            }
            matched += 1
            continue
        
        # Try without anatomical qualifiers
        simple = re.sub(r'^(left_|right_|superior_|inferior_|anterior_|posterior_)', '', organ)
        simple = simple.replace('_', ' ').lower()
        if simple in uberon_by_name and simple != normalized:
            mappings[organ] = {
                "uberon_id": uberon_by_name[simple][0],
                "match": "stripped_prefix",
                "matched_name": simple,
            }
            matched += 1
    
    print(f"  Auto-mapped: {matched}/{len(nexus_organs)} ({100*matched/len(nexus_organs):.1f}%)")
    
    with open(os.path.join(output_dir, "uberon_to_nexus.json"), 'w') as f:
        json.dump({
            "_meta": {"total_nexus_organs": len(nexus_organs), "mapped": matched},
            "mappings": mappings,
        }, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ Saved uberon_to_nexus.json")
    return mappings


def import_anatomy_subtree(uberon_terms: dict, root_id: str, output_dir: str,
                            label: str) -> dict:
    """Extract a subtree of UBERON for a specific body system.
    
    Example: import_anatomy_subtree(terms, "UBERON:0001016", "nervous_system")
    extracts all descendants of "nervous system".
    """
    # Build parent → children index
    children_of = {}
    for tid, info in uberon_terms.items():
        for parent in info.get('parents', []):
            children_of.setdefault(parent, []).append(tid)
    
    # BFS descendants
    subtree = {}
    queue = [root_id]
    while queue:
        current = queue.pop(0)
        if current in subtree:
            continue
        if current in uberon_terms:
            subtree[current] = uberon_terms[current]
            queue.extend(children_of.get(current, []))
    
    output_path = os.path.join(output_dir, f"uberon_{label}.json")
    with open(output_path, 'w') as f:
        json.dump({
            "_meta": {"root": root_id, "label": label, "count": len(subtree)},
            "terms": subtree,
        }, f, indent=2)
    
    print(f"  ✓ Subtree '{label}' ({root_id}): {len(subtree)} terms → {output_path}")
    return subtree


def main():
    parser = argparse.ArgumentParser(description="Import external ontologies into NEXUS")
    parser.add_argument('--uberon', help="Path to uberon.obo")
    parser.add_argument('--mondo', help="Path to mondo.obo")
    parser.add_argument('--output', default="medical_knowledge/external_ontologies",
                        help="Output directory")
    parser.add_argument('--extract-subtree', action='store_true',
                        help="Also extract common anatomy subtrees from UBERON")
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    uberon_terms = None
    
    if args.uberon:
        uberon_terms = import_uberon(args.uberon, args.output)
        auto_map_uberon_to_nexus(uberon_terms, "anatomy_atlas.py", args.output)
        
        if args.extract_subtree and uberon_terms:
            # Common medical subtrees
            print("\n[Extracting body system subtrees]")
            subtrees = {
                "UBERON:0001016": "nervous_system",
                "UBERON:0001017": "central_nervous_system",
                "UBERON:0000010": "peripheral_nervous_system",
                "UBERON:0002416": "integumental_system",
                "UBERON:0001434": "skeletal_system",
                "UBERON:0001015": "musculature_system",
                "UBERON:0004535": "cardiovascular_system",
                "UBERON:0001004": "respiratory_system",
                "UBERON:0001007": "digestive_system",
                "UBERON:0001008": "renal_system",
                "UBERON:0000990": "reproductive_system",
                "UBERON:0000949": "endocrine_system",
                "UBERON:0002405": "immune_system",
                "UBERON:0001846": "inner_ear",
                "UBERON:0000970": "eye",
            }
            for root_id, label in subtrees.items():
                try:
                    import_anatomy_subtree(uberon_terms, root_id, args.output, label)
                except Exception as e:
                    print(f"  ✗ {label}: {e}")
    
    if args.mondo:
        import_mondo(args.mondo, args.output)
    
    if not args.uberon and not args.mondo:
        parser.print_help()
        print("\nTo download the ontology files:")
        print("  UBERON: https://github.com/obophenotype/uberon/releases/latest/download/uberon.obo")
        print("  MONDO:  https://github.com/monarch-initiative/mondo/releases/latest/download/mondo.obo")


if __name__ == "__main__":
    main()