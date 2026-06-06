"""
atlas_extension_loader.py
=========================
Loads LAYER 1.5 anatomy data (peripheral nerves, muscles, microvascular beds,
fascia, lymphatics) and INJECTS it into the existing AnatomyAtlas.

Why this exists:
  AnatomyAtlas covers ~208 organs at organ-level (heart, lungs, kidney).
  LAYER 1.5 adds tissue-level detail (median nerve, supraspinatus, retinal
  capillaries). Without this loader, the two are isolated — diagnoses about
  carpal tunnel can't use atlas's spatial/referred-pain reasoning.

What it does:
  1. Reads layer1_5_anatomy/*.json
  2. Adds each peripheral nerve as Organ(system="neurologic", region=inferred)
  3. Adds each muscle as Organ(system="msk", region=inferred)  
  4. Adds each microvascular bed as Organ(system="cardiovascular")
  5. Adds nerve→muscle innervation as Connection(conn_type="innervation")
  6. Adds nerve→compression_site as Connection(conn_type="compression")
  7. Adds microvascular→organ supply as Connection(conn_type="arterial_micro")
  
Result: a single, unified anatomy graph.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Optional

try:
    from .anatomy_atlas import AnatomyAtlas, Organ, Connection
except ImportError:
    from anatomy_atlas import AnatomyAtlas, Organ, Connection


def _add_organ_safe(atlas, name, system, region, funcs=None):
    """Add organ to atlas AND ensure resolve() finds it (bypasses '' alias bug)."""
    atlas._o(name=name, system=system, region=region,
             pos=(0.0, 0.0, 0.0), p3d=(0.0, 0.0, 0.0),
             funcs=funcs or [])
    # Explicit alias so resolve() finds exact match before fuzzy '' bug
    n = name.strip().lower().replace("_", " ")
    atlas._alias[n] = name
    atlas._alias[name.lower()] = name


def _infer_region(nerve_name: str, trunk_path: list) -> str:
    """Infer body region from nerve trunk path."""
    path_str = " ".join(trunk_path).lower() if trunk_path else nerve_name.lower()
    if any(k in path_str for k in ["face", "trigem", "ophthalmic", "maxillary", "mandibular", "cranial"]):
        return "head"
    if any(k in path_str for k in ["laryng", "phrenic", "neck"]):
        return "neck"
    if any(k in path_str for k in ["brach", "axill", "arm", "forearm", "hand", "wrist", "elbow",
                                   "median", "ulnar", "radial", "thoracic_nerve"]):
        return "limbs"
    if any(k in path_str for k in ["sciatic", "femoral", "tibial", "peroneal", "thigh", 
                                   "leg", "ankle", "foot", "lumbar", "sacral"]):
        return "limbs"
    return "thorax"


def _infer_muscle_region(origin: str, insertion: str) -> str:
    """Infer body region from muscle origin/insertion."""
    text = f"{origin} {insertion}".lower()
    if any(k in text for k in ["humerus", "scapula", "clavicle", "shoulder"]):
        return "limbs"  # upper limb / shoulder girdle
    if any(k in text for k in ["radius", "ulna", "carpal", "metacarpal", "phalanx_thumb",
                                "phalanges", "wrist"]):
        return "limbs"
    if any(k in text for k in ["femur", "tibia", "fibula", "tarsal", "metatarsal", 
                                "calcaneus", "iliac", "ischial", "pubic"]):
        return "limbs"
    if any(k in text for k in ["vertebra", "ribs", "costal", "xiphoid", "sacrum"]):
        return "thorax"
    if "diaphragm" in text or "central_tendon" in text:
        return "thorax"
    return "thorax"


def load_layer_1_5_into_atlas(atlas: AnatomyAtlas,
                              data_dir: str = None,
                              verbose: bool = False) -> dict:
    """
    Inject LAYER 1.5 anatomy data into the AnatomyAtlas.
    
    Returns a stats dict: {nerves_added, muscles_added, microvasc_added,
                            connections_added, ...}
    """
    if data_dir is None:
        # Try common paths
        candidates = [
            "medical_knowledge/layer1_5_anatomy",
            "../medical_knowledge/layer1_5_anatomy",
            os.path.join(os.path.dirname(__file__), "..", "medical_knowledge", "layer1_5_anatomy"),
        ]
        data_dir = next((p for p in candidates if os.path.isdir(p)), None)
        if data_dir is None:
            return {"error": "Cannot locate layer1_5_anatomy directory"}
    
    stats = {"nerves_added": 0, "muscles_added": 0, "microvasc_added": 0,
             "fascia_added": 0, "lymph_added": 0, "connections_added": 0,
             "compressions_added": 0, "innervations_added": 0}
    
    # ─── 1. Peripheral nerves ───
    nerves_path = os.path.join(data_dir, "peripheral_nerves.json")
    if os.path.exists(nerves_path):
        nerves_data = json.load(open(nerves_path))
        for nerve_id, nerve_info in nerves_data.get("nerves", {}).items():
            if nerve_id in atlas.organs:
                continue  # already exists
            region = _infer_region(nerve_id, nerve_info.get("trunk_path", []))
            _add_organ_safe(atlas, name=nerve_id, system="neurologic", region=region, funcs=["peripheral_nerve"] + nerve_info.get("motor_supply", [])[:3])
            stats["nerves_added"] += 1
            
            # Add compression sites as connections
            for cs in nerve_info.get("compression_sites", []):
                site = cs.get("site", "")
                syndrome = cs.get("syndrome", "")
                # We add a self-referential "compression" connection for now
                # (a real compression target would need site as organ)
                atlas._c(src=nerve_id, tgt=nerve_id, ct="compression",
                         desc=f"{site}: {syndrome}", w=0.8)
                stats["compressions_added"] += 1
            
            # Add motor supply connections (nerve → muscle)
            for muscle in nerve_info.get("motor_supply", []):
                if muscle in atlas.organs:
                    atlas._c(src=nerve_id, tgt=muscle, ct="innervation",
                             desc="motor_supply", w=1.0)
                    stats["innervations_added"] += 1
    
    # ─── 2. Muscles ───
    muscles_path = os.path.join(data_dir, "muscles.json")
    if os.path.exists(muscles_path):
        muscles_data = json.load(open(muscles_path))
        for muscle_id, muscle_info in muscles_data.get("muscles", {}).items():
            if muscle_id in atlas.organs:
                continue
            region = _infer_muscle_region(muscle_info.get("origin", ""),
                                            muscle_info.get("insertion", ""))
            _add_organ_safe(atlas, name=muscle_id, system="msk", region=region, funcs=muscle_info.get("action", [])[:3])
            stats["muscles_added"] += 1
            
            # Add innervation connection (muscle ← nerve)
            innerv = muscle_info.get("innervation", "")
            # Extract nerve name (everything before "_nerve")
            for nerve_name in atlas.organs:
                if nerve_name in innerv and atlas.organs[nerve_name].system == "neurologic":
                    atlas._c(src=nerve_name, tgt=muscle_id, ct="innervation",
                             desc="innervates", w=1.0)
                    stats["innervations_added"] += 1
                    break
    
    # ─── 3. Microvascular beds ───
    microvasc_path = os.path.join(data_dir, "microvasculature.json")
    if os.path.exists(microvasc_path):
        mv_data = json.load(open(microvasc_path))
        for mv_id, mv_info in mv_data.get("microvascular_beds", {}).items():
            if mv_id in atlas.organs:
                continue
            # Microvasc bed inherits region from associated organ
            location = mv_info.get("location", "")
            region = "head" if "retina" in location else (
                "abdomen" if any(k in location for k in ["glomer", "splanchnic", "gut"]) else (
                "thorax" if any(k in location for k in ["pulmonary", "myocardial"]) else "limbs"))
            _add_organ_safe(atlas, name=mv_id, system="cardiovascular", region=region, funcs=["microvasculature"])
            stats["microvasc_added"] += 1
            
            # Connect supplier arteries (if they exist in atlas)
            for supplier in mv_info.get("supplied_by", []):
                # supplier names may have suffixes; do partial match
                for atlas_organ in atlas.organs:
                    if supplier.replace("_artery", "") in atlas_organ:
                        atlas._c(src=atlas_organ, tgt=mv_id, ct="arterial",
                                 desc="microvascular_supply", w=0.7)
                        stats["connections_added"] += 1
                        break
    
    # ─── 4. Fascia ───
    fascia_path = os.path.join(data_dir, "fascia.json")
    if os.path.exists(fascia_path):
        fascia_data = json.load(open(fascia_path))
        for f_id, f_info in fascia_data.get("fascia", {}).items():
            if f_id in atlas.organs:
                continue
            _add_organ_safe(atlas, name=f_id, system="msk", region="limbs", funcs=["fascia_or_tunnel"])
            stats["fascia_added"] += 1
    
    # ─── 5. Lymphatics ───
    lymph_path = os.path.join(data_dir, "lymphatics.json")
    if os.path.exists(lymph_path):
        lymph_data = json.load(open(lymph_path))
        for ln_id, ln_info in lymph_data.get("lymph_groups", {}).items():
            if ln_id in atlas.organs:
                continue
            loc = ln_info.get("location", "")
            region = "neck" if "cervical" in loc or "neck" in loc else (
                "abdomen" if "mesenteric" in loc or "abdomin" in loc else (
                "thorax" if "mediastinal" in loc else (
                "limbs" if "axill" in loc or "inguinal" in loc else "thorax")))
            _add_organ_safe(atlas, name=ln_id, system="lymphatic", region=region, funcs=["lymph_drainage"])
            stats["lymph_added"] += 1
    
    if verbose:
        print(f"[LAYER 1.5 LOADER] Injected:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
    
    return stats


def get_extended_atlas(verbose: bool = False) -> AnatomyAtlas:
    """Convenience: get an AnatomyAtlas with LAYER 1.5 already loaded."""
    atlas = AnatomyAtlas()
    load_layer_1_5_into_atlas(atlas, verbose=verbose)
    return atlas


if __name__ == "__main__":
    atlas = get_extended_atlas(verbose=True)
    print(f"\nFinal atlas: {len(atlas.organs)} organs, {len(atlas.connections)} connections")
    
    # Demo: query the extended atlas
    print("\n--- Query: median nerve ---")
    org = atlas.get_organ("median_nerve")
    if org:
        print(f"  System: {org.system}, Region: {org.region}")
    
    neighbors = atlas.get_neighbors("median_nerve", conn_types=["innervation"])
    print(f"  Innervates: {neighbors[:5]}")