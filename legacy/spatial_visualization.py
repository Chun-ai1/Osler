"""
NEXUS 3D Spatial Visualization API
═══════════════════════════════════════════════════════════════
Exposes Three.js-renderable JSON of the body state.
Provides imaging registration and surgical planning endpoints.
"""
from __future__ import annotations
from flask import Blueprint, jsonify, request
from typing import Dict, List, Tuple, Optional
import math

spatial_bp = Blueprint('spatial_3d', __name__, url_prefix='/spatial')

# Will be set by init_spatial()
_spatial_engine = None
_atlas = None


def init_spatial(spatial_engine, atlas):
    """Wire SpatialEngine + AnatomyAtlas into the visualization API."""
    global _spatial_engine, _atlas
    _spatial_engine = spatial_engine
    _atlas = atlas


# ═══════════════════════════════════════════════════════════════
# 1. Full body 3D scene (Three.js consumes this)
# ═══════════════════════════════════════════════════════════════

@spatial_bp.get("/scene")
def get_scene():
    """
    Return all 208 organs as Three.js-renderable boxes.
    Format optimized for direct mesh creation in browser.
    """
    if not _spatial_engine:
        return jsonify({"error": "spatial engine not initialized"}), 503

    scene = {
        "version":  "1.0",
        "units":    "normalized_body (1.0 = ~170cm)",
        "axes":     {"x": "lateral (+R/-L)", "y": "vertical (+up)", "z": "front-back"},
        "organs":   [],
        "zones":    [],
        "vessels":  [],   # vessels rendered as cylinders
    }

    SYSTEM_COLORS = {
        "cardiovascular":   "#dc2626",   # red
        "respiratory":      "#3b82f6",   # blue
        "gi":               "#a16207",   # tan
        "hepatobiliary":    "#92400e",   # brown
        "renal":            "#fbbf24",   # yellow
        "neurologic":       "#7c3aed",   # purple
        "endocrine":        "#10b981",   # emerald
        "lymphatic":        "#06b6d4",   # cyan
        "hematologic":      "#be123c",   # rose
        "reproductive":     "#ec4899",   # pink
        "msk":              "#94a3b8",   # slate
        "integumentary":    "#fde68a",   # tan
        "region":           "#9ca3af",   # gray
    }

    for name, region in _spatial_engine.regions.items():
        is_vessel = any(kw in name for kw in
                        ['_a', '_v', '_aa', '_vv', 'aorta', 'sinus',
                         'duct', 'cisterna', 'arch'])
        organ_data = {
            "id":       name,
            "name":     name.replace('_', ' ').title(),
            "system":   region.system,
            "color":    SYSTEM_COLORS.get(region.system, "#6b7280"),
            "center":   list(region.center),
            "extent":   list(region.extent),
            "volume":   region.volume_ml,
            "surgical_zone": getattr(region, 'surgical_zone', ''),
            "type":     "vessel" if is_vessel else "organ",
            "side":     region.side,
        }
        if is_vessel:
            scene["vessels"].append(organ_data)
        else:
            scene["organs"].append(organ_data)

    for name, zone in _spatial_engine.zones.items():
        scene["zones"].append({
            "id":     name,
            "name":   name.replace('_', ' ').title(),
            "center": list(zone.center),
            "extent": list(zone.extent),
        })

    return jsonify(scene)


# ═══════════════════════════════════════════════════════════════
# 2. Patient state visualization — symptoms → 3D highlights
# ═══════════════════════════════════════════════════════════════

@spatial_bp.post("/visualize_state")
def visualize_state():
    """
    Given patient symptoms (and optional mechanisms), highlight on 3D anatomy:
      • primary organs (red): directly involved by symptoms
      • adjacent organs (orange): physically close, secondary risk
      • mechanism organs (yellow): inferred from active mechanisms
      • zones: body regions to spotlight
    """
    data = request.get_json() or {}
    symptoms = data.get("symptoms", [])
    mechanisms = data.get("mechanisms", [])  # optional list of mech dicts

    if not _spatial_engine:
        return jsonify({"error": "spatial engine not initialized"}), 503

    try:
        from .symptom_organ_map import merge_symptom_organs, get_organs_for_mechanism
    except ImportError:
        from nexus_engine.symptom_organ_map import (
            merge_symptom_organs, get_organs_for_mechanism)

    # Project ALL symptoms onto 3D (not just pain)
    sym_map = merge_symptom_organs(symptoms)
    primary_organs  = set(sym_map["primary"])
    adjacent_organs = set(sym_map["adjacent"])
    zones_active    = set(sym_map["zones"])

    # Project mechanisms onto 3D
    mechanism_organs = set()
    for mech in mechanisms:
        mechanism_organs.update(get_organs_for_mechanism(mech))

    # 3D physical neighbors of primary organs
    spatial_neighbors = set()
    for organ in list(primary_organs)[:6]:
        for n_organ, _ in _spatial_engine.organs_near(organ, radius=0.06)[:5]:
            spatial_neighbors.add(n_organ)
    spatial_neighbors -= primary_organs

    # Volume at risk
    vol = _spatial_engine.volume_at_risk(list(primary_organs))

    return jsonify({
        "zones_active":       sorted(zones_active),
        "organs_primary":     sorted(primary_organs),
        "organs_adjacent":    sorted(adjacent_organs),
        "organs_secondary":   sorted(adjacent_organs | spatial_neighbors),
        "organs_mechanism":   sorted(mechanism_organs - primary_organs - adjacent_organs)[:20],
        "spatial_neighbors":  sorted(spatial_neighbors),
        "volume_at_risk_ml":  vol.get("total_ml", 0),
        "matched_symptoms":   sym_map["matched_symptoms"],
        "unmatched_symptoms": sym_map["unmatched_symptoms"],
    })


# ═══════════════════════════════════════════════════════════════
# 3. Trajectory simulation (stab wound, dissection, embolus)
# ═══════════════════════════════════════════════════════════════

@spatial_bp.post("/trajectory")
def trajectory():
    """
    Compute 3D trajectory from organ A to organ B.
    Returns ordered list of structures the line passes through.

    Use cases:
      • Stab/gunshot wound trajectory
      • Aortic dissection extension
      • Pulmonary embolus path
      • Tumor invasion route
    """
    data = request.get_json() or {}
    start = data.get("start")
    end   = data.get("end")
    step  = data.get("step", 0.02)

    if not start or not end:
        return jsonify({"error": "start and end required"}), 400

    if not _spatial_engine:
        return jsonify({"error": "spatial engine not initialized"}), 503

    traj = _spatial_engine.trajectory_3d(start, end, step=step)
    return jsonify({
        "start":    start,
        "end":      end,
        "passes_through": [{"organ": name, "distance": d} for name, d in traj],
        "n_structures":   len(traj),
    })


# ═══════════════════════════════════════════════════════════════
# 4. Imaging Registration — map CT/MRI lesion to anatomy
# ═══════════════════════════════════════════════════════════════

@spatial_bp.post("/register_lesion")
def register_lesion():
    """
    Given a 3D lesion coordinate (from CT/MRI report or user input),
    identify:
      - which organ contains the lesion
      - which organs are within risk distance
      - what surgical zone it's in
      - estimated tissue volume affected

    Body coordinates: same normalized system as the atlas.
    User can provide raw mm and we auto-normalize.
    """
    data = request.get_json() or {}
    point = data.get("coordinate")  # [x, y, z]
    radius = data.get("lesion_radius", 0.02)  # default 2cm
    units = data.get("units", "normalized")   # or "mm"

    if not point or len(point) != 3:
        return jsonify({"error": "coordinate [x,y,z] required"}), 400

    if not _spatial_engine:
        return jsonify({"error": "spatial engine not initialized"}), 503

    # Convert mm to normalized body units (assume 170cm height = 1.0 unit)
    if units == "mm":
        point = [c / 1700.0 for c in point]
        radius = radius / 1700.0

    # Find containing organ
    containing = []
    nearby = []
    for name, region in _spatial_engine.regions.items():
        d = math.sqrt(sum((point[i] - region.center[i])**2 for i in range(3)))
        if region.contains(tuple(point)):
            containing.append({"organ": name, "distance": round(d, 4),
                               "system": region.system})
        elif d <= radius * 3:
            nearby.append({"organ": name, "distance": round(d, 4),
                           "system": region.system})

    nearby.sort(key=lambda x: x["distance"])
    
    # Compute affected volume (lesion sphere intersecting bounding boxes)
    affected_vol = 0
    for c in containing:
        affected_vol += min(
            (4/3) * math.pi * radius**3 * 1700**3,  # lesion sphere in mm³
            _spatial_engine.regions[c["organ"]].volume_ml * 1000
        )

    return jsonify({
        "lesion_at":         point,
        "lesion_radius":     radius,
        "contained_in":      containing,
        "nearby_at_risk":    nearby[:8],
        "estimated_affected_volume_ml": round(affected_vol / 1000, 2),
        "surgical_zone":     containing[0].get("system") if containing else "unknown",
    })


# ═══════════════════════════════════════════════════════════════
# 5. Surgical Planning — trajectory safety analysis
# ═══════════════════════════════════════════════════════════════

@spatial_bp.post("/surgical_path")
def surgical_path():
    """
    Plan a needle/catheter/access path from skin entry to target organ.
    Returns:
      - structures the path passes through
      - critical structures within safety margin
      - safer alternative entry points
      - injury risk score (0-1)

    Use cases:
      • Liver biopsy entry site
      • Pneumothorax tube placement
      • Lumbar puncture trajectory
      • Tumor ablation needle path
    """
    data = request.get_json() or {}
    entry  = data.get("entry_point")        # [x,y,z] skin entry
    target = data.get("target_organ")
    safety_margin = data.get("safety_margin", 0.01)  # 1cm default

    if not entry or not target:
        return jsonify({"error": "entry_point and target_organ required"}), 400

    if not _spatial_engine:
        return jsonify({"error": "spatial engine not initialized"}), 503

    if target not in _spatial_engine.regions:
        return jsonify({"error": f"target organ '{target}' unknown"}), 404

    target_center = _spatial_engine.regions[target].center

    # Trace the path (sample points along line)
    n_steps = 50
    path_organs = []
    risk_organs = []

    for i in range(n_steps + 1):
        t = i / n_steps
        point = tuple(entry[j] + (target_center[j] - entry[j]) * t for j in range(3))

        # Direct hits
        for name, region in _spatial_engine.regions.items():
            if region.contains(point) and name != target:
                if not any(p["organ"] == name for p in path_organs):
                    path_organs.append({
                        "organ": name,
                        "at_distance": round(t, 3),
                        "system": region.system,
                    })

            # Within safety margin
            d = math.sqrt(sum((point[k] - region.center[k])**2 for k in range(3)))
            min_d_to_box = max(0, d - max(region.extent))
            if 0 < min_d_to_box <= safety_margin and name != target:
                if not any(p["organ"] == name for p in risk_organs):
                    risk_organs.append({
                        "organ": name,
                        "min_distance": round(min_d_to_box, 4),
                        "system": region.system,
                    })

    # Critical structures penalty
    critical_systems = {"cardiovascular", "neurologic"}
    critical_hits = sum(1 for p in path_organs if p["system"] in critical_systems)
    critical_near = sum(1 for p in risk_organs if p["system"] in critical_systems)
    risk_score = min((critical_hits * 0.3 + critical_near * 0.1
                      + len(path_organs) * 0.05), 1.0)

    return jsonify({
        "entry":           entry,
        "target":          target,
        "passes_through":  path_organs,
        "near_path":       risk_organs[:10],
        "risk_score":      round(risk_score, 2),
        "warning":         (
            "HIGH RISK - traverses critical structures" if risk_score > 0.6
            else "MODERATE - review approach" if risk_score > 0.3
            else "LOW RISK - acceptable approach"
        ),
        "n_structures_on_path": len(path_organs),
    })


# ═══════════════════════════════════════════════════════════════
# 6. Spatial Spread — peritonitis, abscess, infection volume
# ═══════════════════════════════════════════════════════════════

@spatial_bp.post("/spread")
def spread():
    """
    Simulate 3D spread from a source (abscess, hematoma, peritonitis).
    Returns affected organs ranked by distance + confidence.
    """
    data = request.get_json() or {}
    source = data.get("source_organ")
    radius = data.get("max_radius", 0.10)   # 10cm default
    
    if not source:
        return jsonify({"error": "source_organ required"}), 400

    if not _spatial_engine:
        return jsonify({"error": "spatial engine not initialized"}), 503

    affected = _spatial_engine.spatial_spread(source, max_radius=radius, max_organs=15)
    return jsonify({
        "source":   source,
        "radius":   radius,
        "affected": affected,
        "n_organs": len(affected),
    })

@spatial_bp.post("/reason_3d")
def reason_3d():
    """
    Full NEXUS reasoning + synchronized 3D visualization.
    Returns both the diagnoses AND the 3D state in one call.

    Frontend: send symptoms, receive {diagnoses, highlights, mechanisms_active}.
    """
    data = request.get_json() or {}
    symptoms = data.get("symptoms", [])

    if not _spatial_engine:
        return jsonify({"error": "spatial engine not initialized"}), 503

    # Try to call the full NEXUS pipeline
    nexus_result = {}
    try:
        from flask import current_app
        nexus_inst = current_app.config.get("nexus_instance")
        if nexus_inst:
            nexus_result = nexus_inst.reason(symptoms)
    except Exception as e:
        nexus_result = {"_error": str(e)}

    # Get the 3D state (from nexus_result if Step 5d ran, else compute now)
    spatial = nexus_result.get("spatial_3d") or {}

    if not spatial:
        # Fallback: compute spatial directly
        try:
            from .symptom_organ_map import merge_symptom_organs
        except ImportError:
            from nexus_engine.symptom_organ_map import merge_symptom_organs
        sm = merge_symptom_organs(symptoms)
        spatial = {
            "primary_organs":  sm["primary"],
            "adjacent_organs": sm["adjacent"],
            "zones_active":    sm["zones"],
        }

    return jsonify({
        "symptoms":           symptoms,
        "diagnoses":          nexus_result.get("diagnoses", [])[:5],
        "consistency":        nexus_result.get("nexus_consistency", {}),
        "active_mechanisms":  nexus_result.get("stats", {}).get("mechanisms_activated", 0),
        "spatial_3d":         spatial,
        "physiology":         nexus_result.get("physiology_state", {}),
    })