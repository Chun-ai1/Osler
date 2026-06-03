"""
NEXUS Spatial Reasoner — Deep 3D Reasoning Layer
═══════════════════════════════════════════════════════════════
Uses 3D anatomy as a FIRST-CLASS REASONING INPUT, not just visualization.

Six core reasoning capabilities:
  1. ANATOMICAL CONSISTENCY  — Penalize diagnoses inconsistent with 3D location
  2. SPATIAL DIFFERENTIAL    — Limit candidates to organs in the affected region
  3. MULTI-SYMPTOM OVERLAP   — Find diseases that explain ALL spatial signals
  4. ANATOMICAL EXCLUSION    — Detect spatial impossibilities
  5. SPREAD PROPAGATION      — 3D-adjacent organs → secondary differentials
  6. ORGAN-DISEASE MAPPING   — Each disease has a 3D "fingerprint"

This module operates on diagnoses and outputs:
  • Score adjustments (positive or negative)
  • New differential diagnoses (from spatial spread)
  • Anatomical reasoning explanations (for trace)
"""
from __future__ import annotations
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict

# NeuroPainReasoner — mechanism-based pain reasoning (Path C)
try:
    from .neuro_pain_reasoner import NeuroPainReasoner
except ImportError:
    try:
        from neuro_pain_reasoner import NeuroPainReasoner
    except ImportError:
        NeuroPainReasoner = None

# DerivationEngine — derives zones/system/adjacent from LAYER 1 data
try:
    from .derivation_engine import DerivationEngine
except ImportError:
    try:
        from derivation_engine import DerivationEngine
    except ImportError:
        DerivationEngine = None


# ═══════════════════════════════════════════════════════════════
# DISEASE → 3D ORGAN FINGERPRINT
# ═══════════════════════════════════════════════════════════════
# Each disease has expected primary organs + region.
# This is the GROUND TRUTH for anatomical consistency checks.

_FALLBACK_DISEASE_ANATOMY: Dict[str, dict] = {}
# Phase B refactor: hardcoded fallback removed. Disease anatomy is now sourced
# exclusively from medical_knowledge/derived/disease_anatomy_cascade.json,
# which is generated from disease_mechanism_map.json + anatomy_atlas (cascade-derived).
# If that JSON is missing, spatial reasoning gracefully no-ops (no fingerprints
# loaded → no spatial adjustments fire → cascade matching alone determines ranking).


# Load disease anatomy with priority:
#   1. Cascade-derived (medical_knowledge/derived/disease_anatomy_cascade.json) — PREFERRED
#   2. knowledge_loader (legacy JSON path)
#   3. _FALLBACK_DISEASE_ANATOMY (hardcoded 80 entries — last resort)
def _load_cascade_anatomy():
    """Load cascade-derived disease anatomy fingerprints (Phase B refactor)."""
    import json as _j, os as _o
    candidates = [
        "medical_knowledge/derived/disease_anatomy_cascade.json",
        "../medical_knowledge/derived/disease_anatomy_cascade.json",
    ]
    for p in candidates:
        if _o.path.exists(p):
            try:
                data = _j.load(open(p, encoding="utf-8"))
                diseases = data.get("diseases", {})
                # Filter to non-empty entries only — empty ones provide no signal
                return {k: v for k, v in diseases.items()
                        if v.get("primary") or v.get("zones") or v.get("system")}
            except Exception:
                continue
    return None

_cascade = _load_cascade_anatomy()
if _cascade:
    DISEASE_ANATOMY = _cascade
    print(f"[SpatialReasoner] Cascade-derived anatomy: {len(_cascade)} diseases")
else:
    # Fall back to old paths (legacy)
    try:
        from .knowledge_loader import get_kb
        _kb = get_kb()
        DISEASE_ANATOMY = _kb.disease_anatomy or _FALLBACK_DISEASE_ANATOMY
    except (ImportError, Exception):
        try:
            from knowledge_loader import get_kb
            _kb = get_kb()
            DISEASE_ANATOMY = _kb.disease_anatomy or _FALLBACK_DISEASE_ANATOMY
        except Exception:
            DISEASE_ANATOMY = _FALLBACK_DISEASE_ANATOMY
    print(f"[SpatialReasoner] Using fallback (hardcoded) anatomy: {len(DISEASE_ANATOMY)} diseases")


# ═══════════════════════════════════════════════════════════════
# SPATIAL REASONER — main reasoning class
# ═══════════════════════════════════════════════════════════════

class SpatialReasoner:
    """
    Performs deep 3D anatomical reasoning on diagnostic candidates.
    Adjusts scores, generates explanations, suggests new differentials.
    """

    # Score deltas — larger than Step 5d's small boost
    BOOST_PRIMARY_MATCH       = +0.30   # all primary organs match 3D
    BOOST_ZONE_MATCH          = +0.20   # zone matches
    BOOST_SYSTEM_MATCH        = +0.10   # system aligns
    PENALTY_ZONE_MISMATCH     = -0.40   # 3D location contradicts diagnosis
    PENALTY_PRIMARY_NONE      = -0.25   # NO overlap with disease's primary organs
    PENALTY_SYSTEM_MISMATCH   = -0.15   # totally different system

    def __init__(self, spatial_engine=None):
        self.spatial_engine = spatial_engine
        self.disease_anatomy = DISEASE_ANATOMY
        # Build reverse index: organ → diseases that affect it
        self.organ_to_diseases = defaultdict(set)
        for dis, info in DISEASE_ANATOMY.items():
            for organ in info.get("primary", []):
                self.organ_to_diseases[organ].add(dis)

        # ── Mechanism-based pain reasoner (Path C) ──
        # Predicts pain location from organ neuroanatomy, not from
        # hardcoded disease→pain mappings.
        try:
            self.neuro_pain = NeuroPainReasoner() if NeuroPainReasoner else None
        except Exception as _ne:
            print(f"[SpatialReasoner] NeuroPainReasoner unavailable: {_ne}")
            self.neuro_pain = None
        
        # ── DerivationEngine: derive zones/system/adjacent from LAYER 1 data ──
        try:
            self.derivation = DerivationEngine() if DerivationEngine else None
        except Exception as _de:
            print(f"[SpatialReasoner] DerivationEngine unavailable: {_de}")
            self.derivation = None

    # ─────────────────────────────────────────────────────────
    # Capability 1: Anatomical consistency check
    # ─────────────────────────────────────────────────────────
    def consistency_check(self, disease: str, sym_organs: dict) -> Tuple[float, str]:
        """
        Compare disease's expected anatomy with patient's symptom-derived 3D location.
        Returns (score_delta, explanation).

        Uses DOMINANT zones/organs (those hit by ≥50% of symptoms) for stricter
        anatomical signal. Loose union of all symptom organs/zones is too noisy
        because every body symptom maps to multiple organs.
        """
        d_norm = disease.lower().replace(' ', '_').replace('-', '_')
        d_info = self._lookup_disease(d_norm)
        if not d_info:
            return 0.0, "no anatomy fingerprint for this disease"

        d_primary = set(d_info.get("primary", []))
        d_zones   = set(d_info.get("zones", []))

        # Prefer DOMINANT (high-confidence) anatomy — these are zones/organs
        # most symptoms agree on, not just any organ touched by any symptom.
        sym_primary = set(sym_organs.get("primary", []))
        sym_zones   = set(sym_organs.get("zones", []))
        dominant_zones  = set(sym_organs.get("dominant_zones", []))
        dominant_organs = set(sym_organs.get("dominant_organs", []))

        # If disease has no anatomy info, skip
        if not d_primary and not d_zones:
            return 0.0, "disease has no spatial fingerprint"

        # ── ZONE CHECK (strongest signal) ──
        # If patient has dominant zones (consistent localization across symptoms),
        # disease MUST overlap with one of them, otherwise it's anatomically wrong.
        if dominant_zones and d_zones:
            if d_zones & dominant_zones:
                return (self.BOOST_ZONE_MATCH,
                        f"3D zone match: {sorted(d_zones & dominant_zones)}")
            else:
                # Disease zone is NOT in patient's dominant zones — strong mismatch.
                # E.g. patient has dominant epigastric/umbilical → migraine (head) wrong
                return (self.PENALTY_ZONE_MISMATCH * 1.5,
                        f"3D dominant zone mismatch: disease in {sorted(d_zones)[:2]} "
                        f"but symptoms localize to {sorted(dominant_zones)}")

        # ── PRIMARY ORGAN CHECK (secondary signal) ──
        if d_primary and sym_primary:
            primary_overlap = d_primary & sym_primary
            dominant_overlap = d_primary & dominant_organs
            if dominant_overlap:
                # Strong match: dominant organs (multiple symptoms agree)
                coverage = len(dominant_overlap) / len(d_primary)
                delta = self.BOOST_PRIMARY_MATCH * min(1.0, coverage * 1.5)
                return (delta,
                        f"3D dominant organ match: {sorted(dominant_overlap)[:3]} "
                        f"(disease coverage={coverage:.0%})")
            elif primary_overlap:
                # Weak match: only one symptom hits, organ overlap exists
                coverage = len(primary_overlap) / len(d_primary | sym_primary)
                delta = self.BOOST_PRIMARY_MATCH * coverage * 0.5
                return (delta,
                        f"3D weak organ match: {sorted(primary_overlap)[:2]} "
                        f"(coverage={coverage:.0%})")
            else:
                # No primary overlap — anatomical inconsistency
                return (self.PENALTY_PRIMARY_NONE * 1.6,
                        f"3D mismatch: disease expects {sorted(d_primary)[:3]} "
                        f"but symptoms point to {sorted(sym_primary)[:3]}")

        # ── Fallback: loose zone match ──
        if d_zones and sym_zones:
            if d_zones & sym_zones:
                return (self.BOOST_ZONE_MATCH * 0.5,
                        f"3D loose zone match: {sorted(d_zones & sym_zones)}")
            else:
                return (self.PENALTY_ZONE_MISMATCH,
                        f"3D zone mismatch: disease in {sorted(d_zones)} "
                        f"but symptoms in {sorted(sym_zones)}")

        return 0.0, "insufficient spatial signal"
    
    def predict_pain_from_disease(self, disease_id: str) -> dict:
        """
        Path C: Derive expected pain pattern from disease's primary organ
        using NeuroPainReasoner — no hardcoded disease→pain mapping.
        """
        if not getattr(self, 'neuro_pain', None):
            return {}
        d_norm = disease_id.lower().replace(' ', '_').replace('-', '_')
        d_info = self._lookup_disease(d_norm)
        if not d_info:
            return {}
        primary_organs = d_info.get('primary', [])
        if not primary_organs:
            return {}
        return self.neuro_pain.predict_pain_pattern(primary_organs[0])
    
    def reverse_predict_organs_from_pain(self, pain_zones, top_k=10):
        """
        Path C reverse: given patient pain zones, derive likely source organs
        using NeuroPainReasoner.
        Returns list of (organ_id, score, evidence) tuples.
        """
        if not getattr(self, 'neuro_pain', None):
            return []
        return self.neuro_pain.predict_organs_from_pain(pain_zones)[:top_k]

    # ─────────────────────────────────────────────────────────
    # Capability 2: Spatial differential — what diseases match this 3D location?
    # ─────────────────────────────────────────────────────────
    def spatial_differential(self, sym_organs: dict, top_n: int = 8) -> List[Tuple[str, float]]:
        """
        Given a 3D symptom signature, return the top diseases whose anatomy
        matches this location. Used to suggest new candidates not yet in scoring.
        """
        sym_primary = set(sym_organs.get("primary", []))
        sym_zones   = set(sym_organs.get("zones", []))

        if not sym_primary and not sym_zones:
            return []

        scored = []
        for dis, info in self.disease_anatomy.items():
            d_primary = set(info.get("primary", []))
            d_zones   = set(info.get("zones", []))
            if not d_primary and not d_zones:
                continue

            # Jaccard-style score
            organ_overlap = len(d_primary & sym_primary)
            zone_overlap  = len(d_zones & sym_zones)
            organ_total   = len(d_primary | sym_primary) or 1
            zone_total    = len(d_zones | sym_zones) or 1

            score = (0.7 * (organ_overlap / organ_total)
                     + 0.3 * (zone_overlap / zone_total))
            if score > 0:
                scored.append((dis, round(score, 3)))

        scored.sort(key=lambda x: -x[1])
        return scored[:top_n]

    # ─────────────────────────────────────────────────────────
    # Capability 3: Multi-symptom spatial intersection
    # ─────────────────────────────────────────────────────────
    def find_unifying_diagnosis(self, sym_organs: dict) -> List[Tuple[str, str]]:
        """
        Find diseases whose primary organs intersect ALL symptom regions.
        Useful when symptoms span multiple zones — the unifying disease
        affects organs in all of them.
        """
        sym_primary = set(sym_organs.get("primary", []))
        if not sym_primary:
            return []

        # Find diseases whose anatomy contains organs from this set
        candidates = []
        for organ in sym_primary:
            if organ in self.organ_to_diseases:
                candidates.extend(self.organ_to_diseases[organ])

        # Count how many primary organs each disease covers
        coverage = defaultdict(int)
        for dis in candidates:
            d_primary = set(self.disease_anatomy[dis].get("primary", []))
            coverage[dis] = len(d_primary & sym_primary)

        # Return diseases that cover at least 2 of the symptom organs
        unifying = [(d, f"covers {n} symptom organs")
                    for d, n in coverage.items() if n >= 2]
        unifying.sort(key=lambda x: -coverage[x[0]])
        return unifying[:5]

    # ─────────────────────────────────────────────────────────
    # Capability 4: 3D adjacency spread (find secondary differentials)
    # ─────────────────────────────────────────────────────────
    def adjacency_differentials(self, primary_organs: List[str],
                                radius: float = 0.05) -> List[str]:
        """
        Use SpatialEngine to find organs physically adjacent to primary organs.
        Each adjacent organ → diseases that affect it.
        These are SECONDARY differentials (worth considering).
        """
        if not self.spatial_engine:
            return []

        adjacent_organs = set()
        for organ in primary_organs[:5]:
            for n_organ, _ in self.spatial_engine.organs_near(organ, radius)[:5]:
                adjacent_organs.add(n_organ)

        # Get diseases of those adjacent organs
        adjacent_diseases = set()
        for organ in adjacent_organs:
            adjacent_diseases.update(self.organ_to_diseases.get(organ, set()))

        return sorted(adjacent_diseases)

    # ─────────────────────────────────────────────────────────
    # Capability 5: Anatomical exclusion / contradiction
    # ─────────────────────────────────────────────────────────
    def find_contradictions(self, sym_organs: dict, candidates: List[str]) -> List[Tuple[str, str]]:
        """
        Identify diagnoses that are anatomically incompatible with patient's 3D state.
        Returns list of (disease, reason) pairs to be DEMOTED in ranking.
        """
        sym_primary = set(sym_organs.get("primary", []))
        sym_zones   = set(sym_organs.get("zones", []))

        if not sym_primary and not sym_zones:
            return []

        contradictions = []
        for cand in candidates:
            d_norm = cand.lower().replace(' ', '_')
            d_info = self._lookup_disease(d_norm)
            if not d_info:
                continue

            d_primary = set(d_info.get("primary", []))
            d_zones   = set(d_info.get("zones", []))

            # Hard contradiction: zones completely don't match
            if d_zones and sym_zones and not (d_zones & sym_zones):
                # And primary organs also don't match
                if d_primary and sym_primary and not (d_primary & sym_primary):
                    contradictions.append((cand,
                        f"zones {sorted(d_zones)} ≠ {sorted(sym_zones)}, "
                        f"organs {sorted(d_primary)[:2]} ≠ {sorted(sym_primary)[:2]}"))

        return contradictions

    # ─────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────
    def _lookup_disease(self, name: str) -> Optional[dict]:
        """
        Find disease anatomy by:
          1. Exact match in verified data
          2. Normalized match (lowercase, strip punctuation)
          3. Fuzzy substring match
          4. GNN-inferred predictions
        After lookup, MISSING fields (zones, system) are DERIVED from primary organs
        via DerivationEngine — no longer hardcoded in JSON.
        """
        # Tier 1: verified data (exact)
        if name in self.disease_anatomy:
            d = dict(self.disease_anatomy[name])
            d['_source'] = 'verified'
            return self._enrich_with_derived(d)

        # Tier 1b: normalized exact match — mirror build_disease_anatomy_cascade._to_key
        # The build script generates keys via this same normalization, so an exact
        # normalized lookup hits every disease that has a fingerprint.
        norm = (name.lower()
                .replace("'", "")
                .replace("\u2019", "")          # smart quote
                .replace("(", "")
                .replace(")", "")
                .replace("/", "_")
                .replace("-", "_")
                .replace(" ", "_"))
        if norm in self.disease_anatomy:
            d = dict(self.disease_anatomy[norm])
            d['_source'] = 'verified'
            return self._enrich_with_derived(d)

        # Tier 2: verified data (fuzzy)
        for key in self.disease_anatomy:
            if norm in key or key in norm:
                if abs(len(norm) - len(key)) < 10:
                    d = dict(self.disease_anatomy[key])
                    d['_source'] = 'verified_fuzzy'
                    return self._enrich_with_derived(d)
        
        # Tier 3: GNN-inferred (lazy-loaded, confidence-gated)
        if not hasattr(self, '_gnn_inferred'):
            self._gnn_inferred = self._load_gnn_inferred()
        
        if name in self._gnn_inferred:
            pred = self._gnn_inferred[name]
            # Only use if top confidence is high enough
            if pred.get('top_organ_confidence', 0) >= 0.5:
                return {
                    'primary': pred.get('predicted_organs', []),
                    'zones':   pred.get('predicted_zones', []),
                    'system':  '',
                    '_source': 'gnn_inferred',
                    '_confidence': pred.get('top_organ_confidence', 0),
                }
        
        return None
    
    def _load_gnn_inferred(self) -> dict:
        """
        Load GNN-inferred predictions for unknown diseases.
        
        Default: v1 TransE (validated, val_MRR=0.20, predicts correct organs).
        v2 ComplEx + HPO disabled by default (HPO data introduces skin bias).
        
        To enable v2: rename .FAILED_HPO_BIAS.json back to .json
        See HPO_INTEGRATION_GUIDE.md for the issue and remediation options.
        """
        import json, os
        # v1 TransE first (validated working). v2 only if user explicitly re-enables.
        candidates = [
            "medical_knowledge/graph/disease_anatomy_inferred.json",
            "../medical_knowledge/graph/disease_anatomy_inferred.json",
            "medical_knowledge/graph/disease_anatomy_inferred_v2.json",
            "../medical_knowledge/graph/disease_anatomy_inferred_v2.json",
        ]
        for path in candidates:
            if os.path.exists(path):
                try:
                    data = json.load(open(path, encoding='utf-8'))
                    self._gnn_version = 'v2_complex' if 'v2' in path else 'v1_transe'
                    return data.get('predictions', {})
                except Exception:
                    continue
        self._gnn_version = 'none'
        return {}
    
    def _enrich_with_derived(self, d: dict) -> dict:
        """If d is missing 'zones' or 'system', derive them from 'primary' organs."""
        if not d.get('primary'):
            return d
        if not getattr(self, 'derivation', None):
            return d
        primary = d.get('primary', [])
        if 'zones' not in d or not d.get('zones'):
            d['zones'] = self.derivation.derive_disease_zones(primary)
            d['_zones_derived'] = True
        if 'system' not in d or not d.get('system'):
            d['system'] = self.derivation.derive_disease_system(primary)
            d['_system_derived'] = True
        return d

    # ─────────────────────────────────────────────────────────
    # Top-level: full 3D reasoning pass on a list of diagnoses
    # ─────────────────────────────────────────────────────────
    def reason(self, sym_organs: dict, current_scores: Dict[str, float]) -> dict:
        """
        Full 3D reasoning pass.
        Input:
          sym_organs:     output of merge_symptom_organs()
          current_scores: {disease: score} from Steps 1-5c
        Output:
          {
            adjustments:           {disease: delta},
            explanations:          {disease: reason},
            new_candidates:        [disease, ...]   # from spatial differential
            contradictions:        [(disease, reason), ...],
            unifying_diagnosis:    [(disease, reason), ...],
            adjacency_diff:        [disease, ...],
          }
        """
        adjustments = {}
        explanations = {}

        # Apply consistency check to each existing candidate
        for disease in current_scores:
            delta, reason = self.consistency_check(disease, sym_organs)
            if abs(delta) > 0.01:
                adjustments[disease] = delta
                explanations[disease] = reason

        # Spatial differential: new candidates from 3D
        new_candidates = []
        for dis, score in self.spatial_differential(sym_organs, top_n=5):
            if dis not in current_scores:
                new_candidates.append((dis, score))

        # Find contradictions among current candidates
        contradictions = self.find_contradictions(
            sym_organs, list(current_scores.keys()))

        # Unifying diagnosis (multi-organ spans)
        unifying = self.find_unifying_diagnosis(sym_organs)

        # Adjacency differentials
        adj_diff = self.adjacency_differentials(sym_organs.get("primary", []))

        return {
            "adjustments":     adjustments,
            "explanations":    explanations,
            "new_candidates":  new_candidates,
            "contradictions":  contradictions,
            "unifying_diagnosis": unifying,
            "adjacency_differentials": adj_diff[:10],
        }


# ═══════════════════════════════════════════════════════════════
# Standalone test
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    sys.path.insert(0, '..')

    reasoner = SpatialReasoner()

    print("=== TEST 1: Consistency check ===")
    sym_organs = {
        "primary":  ["heart", "lad_a", "rca_a"],
        "adjacent": ["lungs", "aorta"],
        "zones":    ["thorax_central"],
    }
    for dis in ["heart_attack", "appendicitis", "pneumonia", "migraine"]:
        delta, reason = reasoner.consistency_check(dis, sym_organs)
        print(f"  {dis:25s} {delta:+.2f}  — {reason}")

    print("\n=== TEST 2: Spatial differential (RLQ pain → which diseases?) ===")
    sym_organs = {
        "primary":  ["appendix", "cecum", "ileum"],
        "adjacent": [],
        "zones":    ["rlq"],
    }
    for dis, score in reasoner.spatial_differential(sym_organs):
        print(f"  {dis:30s} {score:.3f}")

    print("\n=== TEST 3: Contradictions ===")
    sym_organs = {
        "primary":  ["brain"],
        "adjacent": [],
        "zones":    ["head"],
    }
    contradictions = reasoner.find_contradictions(
        sym_organs, ["heart_attack", "migraine", "appendicitis"])
    for dis, reason in contradictions:
        print(f"  ✗ {dis}: {reason}")

    print("\n=== TEST 4: Full 3D reasoning pass ===")
    sym_organs = {
        "primary":  ["heart", "lad_a", "lcx_a"],
        "adjacent": ["lungs"],
        "zones":    ["thorax_central"],
    }
    current_scores = {
        "heart_attack": 0.6,
        "pneumonia":    0.4,
        "migraine":     0.3,
        "appendicitis": 0.2,
    }
    result = reasoner.reason(sym_organs, current_scores)
    print(f"  Adjustments:")
    for d, delta in result["adjustments"].items():
        print(f"    {d:25s}: {delta:+.2f}  ({result['explanations'].get(d,'?')})")
    print(f"  Contradictions: {result['contradictions']}")
    print(f"  Adjacency differentials: {result['adjacency_differentials']}")