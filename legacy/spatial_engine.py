"""
NEXUS Spatial Engine — 3D Anatomical Reasoning
═══════════════════════════════════════════════════════════════
Treats the body as a 3D coordinate space.
Enables reasoning about:
  • Geometric proximity (which organs are physically near each other)
  • Spatial spread (how does inflammation propagate through tissue volume)
  • Pain localization (which 3D region does this pain map to)
  • Trajectory tracing (path of an embolus, dissection, infection)
  • Volume/distance queries (how much brain tissue is at risk)

This is what AI can do that humans cannot — actually compute in 3D space.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
import math
import heapq


# ═══════════════════════════════════════════════════════════════
# 3D Region — bounding box + center + radius
# ═══════════════════════════════════════════════════════════════

@dataclass
class Region3D:
    """A 3D anatomical region (organ or area)."""
    name:       str
    center:     Tuple[float, float, float]   # (x, y, z) center
    extent:     Tuple[float, float, float]   # (dx, dy, dz) bounding box half-sizes
    volume_ml:  float = 0.0                  # approximate volume
    system:     str = ""
    region:     str = ""                     # body region (head/thorax/abdomen/etc.)
    side:       str = "midline"

    def contains(self, point: Tuple[float, float, float]) -> bool:
        """Is a 3D point inside this region's bounding box?"""
        return all(
            self.center[i] - self.extent[i] <= point[i] <= self.center[i] + self.extent[i]
            for i in range(3)
        )

    def distance_to(self, other_center: Tuple[float, float, float]) -> float:
        """Euclidean distance between centers."""
        return math.sqrt(sum(
            (self.center[i] - other_center[i]) ** 2 for i in range(3)
        ))

    def overlaps(self, other: "Region3D") -> bool:
        """Do two bounding boxes intersect?"""
        return all(
            abs(self.center[i] - other.center[i]) <= self.extent[i] + other.extent[i]
            for i in range(3)
        )


# ═══════════════════════════════════════════════════════════════
# Default 3D extents and volumes (mostly literature-based)
# ═══════════════════════════════════════════════════════════════
# Body axes:
#   x = lateral  (negative=left, positive=right)
#   y = vertical (positive=up,    negative=down)
#   z = front-back (positive=front, negative=back)

# Import comprehensive geometry database (covers all 208 organs)
try:
    from .anatomy_geometry_db import ORGAN_GEOMETRY
except ImportError:
    try:
        from anatomy_geometry_db import ORGAN_GEOMETRY
    except ImportError:
        ORGAN_GEOMETRY = {}   # fallback: empty dict, defaults will be used


# ═══════════════════════════════════════════════════════════════
# Body Regions — 3D zones for pain/symptom localization
# ═══════════════════════════════════════════════════════════════

_FALLBACK_BODY_ZONES = {
    "head":            {"center": (0,    0.80,  0),    "extent": (0.10, 0.10, 0.10)},
    "neck":            {"center": (0,    0.60,  0),    "extent": (0.06, 0.05, 0.05)},
    "thorax_central":  {"center": (0,    0.35,  0.05), "extent": (0.10, 0.10, 0.05)},
    "right_thorax":    {"center": (0.10, 0.35,  0.05), "extent": (0.08, 0.10, 0.05)},
    "left_thorax":     {"center": (-0.10,0.35,  0.05), "extent": (0.08, 0.10, 0.05)},
    "epigastric":      {"center": (0,    0.15,  0.10), "extent": (0.05, 0.05, 0.05)},
    "ruq":             {"center": (0.10, 0.12,  0.08), "extent": (0.06, 0.07, 0.06)},
    "luq":             {"center": (-0.10,0.12,  0.08), "extent": (0.06, 0.07, 0.06)},
    "rlq":             {"center": (0.10,-0.10,  0.05), "extent": (0.06, 0.07, 0.06)},
    "llq":             {"center": (-0.10,-0.10, 0.05), "extent": (0.06, 0.07, 0.06)},
    "umbilical":       {"center": (0,    0.00,  0.05), "extent": (0.05, 0.05, 0.05)},
    "suprapubic":      {"center": (0,   -0.18,  0.04), "extent": (0.05, 0.05, 0.04)},
    "right_flank":     {"center": (0.13, 0.05, -0.10), "extent": (0.04, 0.07, 0.05)},
    "left_flank":      {"center": (-0.13,0.05, -0.10), "extent": (0.04, 0.07, 0.05)},
    "lower_back":      {"center": (0,    0.00, -0.15), "extent": (0.10, 0.10, 0.04)},
    "pelvis":          {"center": (0,   -0.20,  0),    "extent": (0.10, 0.08, 0.08)},
}


# Load from JSON via knowledge_loader, fall back to inline data if missing
try:
    from .knowledge_loader import get_kb
    _kb = get_kb()
    BODY_ZONES = _kb.body_zones or _FALLBACK_BODY_ZONES
except (ImportError, Exception):
    try:
        from knowledge_loader import get_kb
        _kb = get_kb()
        BODY_ZONES = _kb.body_zones or _FALLBACK_BODY_ZONES
    except Exception:
        BODY_ZONES = _FALLBACK_BODY_ZONES


# ═══════════════════════════════════════════════════════════════
# Pain Location → 3D Zone Mapping
# ═══════════════════════════════════════════════════════════════

_FALLBACK_PAIN_LOCALIZATION: Dict[str, str] = {
    "headache":              "head",
    "neck_pain":             "neck",
    "stiff_neck":            "neck",
    "chest_pain":            "thorax_central",
    "left_chest_pain":       "left_thorax",
    "right_chest_pain":      "right_thorax",
    "epigastric_pain":       "epigastric",
    "right_upper_quadrant":  "ruq",
    "ruq_pain":              "ruq",
    "left_upper_quadrant":   "luq",
    "luq_pain":              "luq",
    "right_lower_quadrant":  "rlq",
    "rlq_pain":              "rlq",
    "mcburney_point":        "rlq",        # appendicitis classic
    "left_lower_quadrant":   "llq",
    "llq_pain":              "llq",
    "umbilical_pain":        "umbilical",
    "periumbilical_pain":    "umbilical",
    "suprapubic_pain":       "suprapubic",
    "right_flank_pain":      "right_flank",
    "flank_pain":            "right_flank",  # default to right
    "left_flank_pain":       "left_flank",
    "back_pain":             "lower_back",
    "pelvic_pain":           "pelvis",
    "abdominal_pain":        "umbilical",   # vague — defaults to center
}


# Load from JSON via knowledge_loader, fall back to inline data if missing
try:
    from .knowledge_loader import get_kb
    _kb = get_kb()
    PAIN_LOCALIZATION = _kb.pain_localization or _FALLBACK_PAIN_LOCALIZATION
except (ImportError, Exception):
    try:
        from knowledge_loader import get_kb
        _kb = get_kb()
        PAIN_LOCALIZATION = _kb.pain_localization or _FALLBACK_PAIN_LOCALIZATION
    except Exception:
        PAIN_LOCALIZATION = _FALLBACK_PAIN_LOCALIZATION


# ═══════════════════════════════════════════════════════════════
# Spatial Engine — main API
# ═══════════════════════════════════════════════════════════════

class SpatialEngine:
    """
    3D anatomical reasoning engine.
    
    USAGE:
        eng = SpatialEngine(atlas)        # atlas = your AnatomyAtlas
        organs = eng.organs_in_zone("rlq")          # what's in the RLQ?
        nearest = eng.organs_near("appendix", 0.05) # what's within 5cm?
        path = eng.spatial_path("soleal_vv", "lung") # 3D propagation route
    """

    def __init__(self, atlas=None):
        self.atlas = atlas
        self.regions: Dict[str, Region3D] = {}
        self.zones:   Dict[str, Region3D] = {}
        self._build_regions()
        self._build_zones()
        print(f"[SPATIAL] {len(self.regions)} 3D regions, {len(self.zones)} body zones")

    def _build_regions(self):
        """Build 3D Region objects from atlas organs + ORGAN_GEOMETRY."""
        if not self.atlas:
            return
        for name, organ in self.atlas.organs.items():
            geom = ORGAN_GEOMETRY.get(name, {})
            extent = geom.get("extent", (0.02, 0.02, 0.02))   # default small box
            volume = geom.get("volume_ml", 10)
            self.regions[name] = Region3D(
                name=name,
                center=organ.pos_3d,
                extent=extent,
                volume_ml=volume,
                system=organ.system,
                region=organ.region,
                side=organ.position,
            )

    def _build_zones(self):
        """Body zones for symptom localization."""
        for name, geom in BODY_ZONES.items():
            self.zones[name] = Region3D(
                name=name,
                center=geom["center"],
                extent=geom["extent"],
                volume_ml=0,
                region=name,
            )

    # ─────────────────────────────────────────────────────────
    # 3D queries
    # ─────────────────────────────────────────────────────────

    def organs_in_zone(self, zone_name: str) -> List[Tuple[str, float]]:
        """All organs whose bounding box overlaps a body zone, ranked by distance."""
        if zone_name not in self.zones:
            return []
        zone = self.zones[zone_name]
        hits = []
        for name, region in self.regions.items():
            if region.overlaps(zone):
                d = region.distance_to(zone.center)
                hits.append((name, round(d, 3)))
        return sorted(hits, key=lambda x: x[1])

    def organs_near(self, organ_name: str, radius: float = 0.05) -> List[Tuple[str, float]]:
        """All organs within radius (in normalized body units) of given organ."""
        if organ_name not in self.regions:
            return []
        target = self.regions[organ_name]
        hits = []
        for name, region in self.regions.items():
            if name == organ_name:
                continue
            d = region.distance_to(target.center)
            if d <= radius:
                hits.append((name, round(d, 3)))
        return sorted(hits, key=lambda x: x[1])

    def localize_pain(self, pain_terms: List[str]) -> dict:
        """
        Map pain descriptions to 3D zones AND list organs at risk in those zones.
        This is the core of "where is the pain" reasoning.
        """
        zones_hit = set()
        for term in pain_terms:
            t = term.lower().strip().replace(" ", "_")
            zone = PAIN_LOCALIZATION.get(t)
            if zone:
                zones_hit.add(zone)
                continue
            # Fuzzy
            for key, z in PAIN_LOCALIZATION.items():
                if t in key or key in t:
                    zones_hit.add(z)
                    break

        # For each zone, list candidate organs
        zone_organs = {}
        for z in zones_hit:
            organs = self.organs_in_zone(z)
            zone_organs[z] = [o[0] for o in organs[:8]]

        return {
            "zones":         sorted(zones_hit),
            "candidate_organs_per_zone": zone_organs,
            "all_candidate_organs": sorted({o for organs in zone_organs.values()
                                              for o in organs}),
        }

    def spatial_spread(self, source_organ: str, max_radius: float = 0.10,
                       max_organs: int = 12) -> List[dict]:
        """
        3D physical spread — like ink in water. Different from BFS through
        vessel connections (which is in anatomy_bridge); this is pure 3D distance.
        Useful for: tumor invasion, abscess, hematoma expansion, peritonitis.
        """
        if source_organ not in self.regions:
            return []
        source = self.regions[source_organ]
        spread = []
        for name, region in self.regions.items():
            if name == source_organ:
                continue
            d = region.distance_to(source.center)
            if d <= max_radius:
                # Confidence falls off with distance (inverse square-ish)
                confidence = round(max(0.05, 1.0 - (d / max_radius) ** 1.5), 2)
                spread.append({
                    "organ":      name,
                    "distance":   round(d, 3),
                    "confidence": confidence,
                    "system":     region.system,
                })
        spread.sort(key=lambda x: x["distance"])
        return spread[:max_organs]

    def trajectory_3d(self, start: str, end: str,
                      step: float = 0.03) -> List[Tuple[str, float]]:
        """
        Trace a straight-line trajectory between two organs and list every
        organ whose bounding box the line passes through.
        Useful for: bullet/stab wound paths, tumor invasion routes, dissection.
        """
        if start not in self.regions or end not in self.regions:
            return []
        s = self.regions[start].center
        e = self.regions[end].center
        # Sample points along line
        dist = math.sqrt(sum((s[i] - e[i]) ** 2 for i in range(3)))
        n_steps = max(int(dist / step), 5)
        passed = {}
        for i in range(n_steps + 1):
            t = i / n_steps
            point = tuple(s[j] + (e[j] - s[j]) * t for j in range(3))
            for name, region in self.regions.items():
                if region.contains(point):
                    if name not in passed:
                        passed[name] = round(t * dist, 3)
        return sorted(passed.items(), key=lambda x: x[1])

    def volume_at_risk(self, organs: List[str]) -> dict:
        """Sum the volume of multiple organs — useful for 'how much tissue at risk'."""
        total_ml = 0
        per_organ = {}
        for o in organs:
            if o in self.regions:
                v = self.regions[o].volume_ml
                total_ml += v
                per_organ[o] = v
        return {
            "total_ml":  round(total_ml, 0),
            "per_organ": per_organ,
        }

    # ─────────────────────────────────────────────────────────
    # Combined: symptom → 3D analysis
    # ─────────────────────────────────────────────────────────

    def analyze_symptom(self, symptom: str) -> dict:
        """
        Given one symptom term, return its full 3D analysis:
          • Which body zone(s) does it map to?
          • What organs are in those zones?
          • What organs are physically adjacent (3D neighbors)?
          • Estimated volume at risk
        """
        loc = self.localize_pain([symptom])
        candidates = loc.get("all_candidate_organs", [])

        # 3D neighbors of each candidate
        neighbors = {}
        for organ in candidates[:5]:
            near = self.organs_near(organ, radius=0.06)
            neighbors[organ] = [n[0] for n in near[:5]]

        # Volume calculation
        vol = self.volume_at_risk(candidates)

        return {
            "symptom":          symptom,
            "zones":            loc["zones"],
            "candidate_organs": candidates,
            "spatial_neighbors": neighbors,
            "volume_at_risk_ml": vol["total_ml"],
        }


# ═══════════════════════════════════════════════════════════════
# Quick test
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    sys.path.insert(0, "..")
    from anatomy_atlas import AnatomyAtlas

    atlas = AnatomyAtlas()
    eng = SpatialEngine(atlas)

    print("\n=== TEST 1: McBurney's Point Pain (RLQ) ===")
    r = eng.localize_pain(["mcburney_point"])
    print(f"Zones: {r['zones']}")
    print(f"Candidate organs in those zones:")
    for zone, organs in r["candidate_organs_per_zone"].items():
        print(f"  {zone}: {organs}")

    print("\n=== TEST 2: Appendix → 3D spatial neighbors ===")
    near = eng.organs_near("appendix", radius=0.08)
    print(f"Within 8cm of appendix: {[(o,d) for o,d in near[:8]]}")

    print("\n=== TEST 3: Heart → 3D physical neighbors ===")
    near = eng.organs_near("heart", radius=0.10)
    print(f"Within 10cm of heart: {[(o,d) for o,d in near[:8]]}")

    print("\n=== TEST 4: Aortic dissection trajectory (heart → abdominal_aorta) ===")
    traj = eng.trajectory_3d("heart", "abdominal_aorta")
    print("Organs along the path:")
    for name, dist in traj[:10]:
        print(f"  {name} (at {dist})")

    print("\n=== TEST 5: Full symptom analysis: 'mcburney_point' ===")
    a = eng.analyze_symptom("mcburney_point")
    print(f"Zones:              {a['zones']}")
    print(f"Candidate organs:   {a['candidate_organs'][:6]}")
    print(f"Volume at risk:     {a['volume_at_risk_ml']} ml")
    print(f"3D neighbors of top candidate:")
    for organ, nbrs in list(a['spatial_neighbors'].items())[:3]:
        print(f"  {organ} → {nbrs}")

    print("\n=== TEST 6: Stab wound trajectory (skin → liver) ===")
    traj = eng.trajectory_3d("skin", "liver", step=0.02)
    print(f"Organs along stab path: {[name for name,_ in traj]}")