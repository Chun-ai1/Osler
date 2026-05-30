"""
NEXUS Pathogen Spread Tracker

Given an infected organ and pathogen type (virus/bacteria),
traces WHERE the pathogen will likely spread through:
  - Blood vessels (arterial/venous/portal)
  - Direct contact (adjacent organs)
  - Lymphatic drainage
  - GI tract continuity

Uses the anatomy atlas's real vascular connections to predict spread.

Example:
  Input:  virus in stomach
  Output: stomach → portal_vein → liver → hepatic_vv → ivc → heart → lungs → aorta → brain, kidneys, ...

This is pure anatomical reasoning - no LLM, no guessing.
"""
from __future__ import annotations
from typing import Dict, List, Set, Tuple
from collections import deque


class PathogenTracker:

    def __init__(self, atlas):
        self.atlas = atlas

    def track_spread(self, infected_organ: str, pathogen_type: str = "virus",
                     max_hops: int = 8) -> dict:
        """
        Track where a pathogen spreads from an infected organ.

        Args:
            infected_organ: starting organ (e.g. "stomach")
            pathogen_type: "virus", "bacteria", or "fungal"
            max_hops: max steps to trace

        Returns:
            {
                "origin": "stomach",
                "pathogen": "virus",
                "spread_chain": [
                    {"step": 1, "from": "stomach", "to": "portal_vein", 
                     "route": "portal", "reasoning": "venous drainage from GI"},
                    ...
                ],
                "organs_at_risk": ["liver", "lungs", "brain", ...],
                "timeline": [
                    {"phase": "early", "organs": ["portal_vein", "liver"]},
                    {"phase": "intermediate", "organs": ["heart", "lungs"]},
                    {"phase": "late", "organs": ["brain", "kidneys", "spleen"]},
                ],
                "clinical_implications": [...]
            }
        """
        if infected_organ not in self.atlas.organs:
            return {"error": f"Unknown organ: {infected_organ}"}

        # Route preferences by pathogen type
        route_weights = self._get_route_weights(pathogen_type)

        # BFS through vascular system with clinical reasoning
        spread_chain = []
        organs_at_risk = []
        visited = {infected_organ}
        queue = deque([(infected_organ, 0, [])])  # (organ, hop, path_so_far)

        while queue:
            current, hop, path = queue.popleft()
            if hop >= max_hops:
                continue

            connections = self.atlas.get_connections_from(current)
            for conn in connections:
                target = conn.target
                route = conn.conn_type

                # Skip if already visited or route not relevant
                if target in visited:
                    continue
                if route not in route_weights:
                    continue

                weight = route_weights[route]
                if weight <= 0:
                    continue

                visited.add(target)
                new_path = path + [current]
                risk = round(max(0.1, 1.0 - hop * 0.12) * weight, 2)

                reasoning = self._explain_route(current, target, route, pathogen_type)

                spread_chain.append({
                    "step": len(spread_chain) + 1,
                    "from": current,
                    "to": target,
                    "route": route,
                    "hop": hop + 1,
                    "risk": risk,
                    "reasoning": reasoning,
                    "full_path": new_path + [target],
                })

                # Only non-vessel organs are "at risk"
                org = self.atlas.organs.get(target)
                if org and org.system != "cardiovascular":
                    organs_at_risk.append({
                        "organ": target,
                        "system": org.system,
                        "risk": risk,
                        "hop": hop + 1,
                        "via": route,
                        "path": " → ".join(new_path + [target]),
                    })

                queue.append((target, hop + 1, new_path))

        # Build timeline phases
        timeline = self._build_timeline(organs_at_risk)

        # Clinical implications
        implications = self._clinical_implications(
            infected_organ, pathogen_type, organs_at_risk)

        return {
            "origin": infected_organ,
            "pathogen": pathogen_type,
            "spread_chain": spread_chain,
            "organs_at_risk": organs_at_risk,
            "total_organs_at_risk": len(organs_at_risk),
            "timeline": timeline,
            "clinical_implications": implications,
        }

    def track_multiple(self, infected_organs: List[str], pathogen_type: str = "virus",
                       max_hops: int = 6) -> dict:
        """Track spread from multiple infected organs simultaneously."""
        all_results = []
        all_at_risk = {}

        for organ in infected_organs:
            result = self.track_spread(organ, pathogen_type, max_hops)
            all_results.append(result)

            for oar in result.get("organs_at_risk", []):
                name = oar["organ"]
                if name not in all_at_risk or oar["risk"] > all_at_risk[name]["risk"]:
                    all_at_risk[name] = oar
                    all_at_risk[name]["infected_from"] = organ

        # Sort by risk
        combined_risk = sorted(all_at_risk.values(), key=lambda x: -x["risk"])

        # Find convergence points (organs reachable from multiple sources)
        convergence = {}
        for result in all_results:
            for oar in result.get("organs_at_risk", []):
                name = oar["organ"]
                convergence.setdefault(name, set()).add(result["origin"])
        multi_source = {k: sorted(v) for k, v in convergence.items() if len(v) > 1}

        return {
            "infected_organs": infected_organs,
            "pathogen": pathogen_type,
            "individual_results": all_results,
            "combined_risk": combined_risk,
            "convergence_points": multi_source,
            "total_organs_at_risk": len(combined_risk),
        }

    def _get_route_weights(self, pathogen_type: str) -> Dict[str, float]:
        """How easily pathogen spreads via each route type."""
        base = {
            "arterial": 1.0,    # blood-borne spread (most common)
            "venous": 0.95,     # venous return
            "portal": 0.9,      # portal system (GI → liver)
            "cardiac": 1.0,     # through heart chambers
            "gi_tract": 0.7,    # along GI lumen
            "lymphatic": 0.6,   # lymphatic drainage
            "adjacent": 0.3,    # direct tissue invasion
            "airway": 0.5,      # respiratory tract
            "neural": 0.1,      # along nerves (rare for most pathogens)
            "urinary": 0.4,     # urinary tract (ascending)
            "biliary": 0.3,     # biliary system
        }

        if pathogen_type == "bacteria":
            base["lymphatic"] = 0.8   # bacteria love lymph nodes
            base["adjacent"] = 0.5    # abscess formation
            base["urinary"] = 0.7     # UTI ascending
            base["biliary"] = 0.6     # cholangitis
        elif pathogen_type == "virus":
            base["arterial"] = 1.0    # viremia spreads everywhere
            base["neural"] = 0.5      # neurotropic viruses (rabies, HSV)
            base["airway"] = 0.8      # respiratory viruses
        elif pathogen_type == "fungal":
            base["arterial"] = 0.8
            base["lymphatic"] = 0.7
            base["adjacent"] = 0.6

        return base

    def _explain_route(self, source: str, target: str, route: str,
                       pathogen_type: str) -> str:
        """Generate clinical reasoning for each spread step."""
        explanations = {
            "portal": f"{pathogen_type} from {source} drains via portal vein to {target}",
            "venous": f"{pathogen_type} enters venous return from {source} toward {target}",
            "arterial": f"{pathogen_type} in arterial blood flows from {source} to {target}",
            "cardiac": f"{pathogen_type}-laden blood passes through {source} to {target}",
            "gi_tract": f"{pathogen_type} spreads along GI lumen from {source} to {target}",
            "lymphatic": f"{pathogen_type} drains via lymphatics from {source} to {target}",
            "adjacent": f"{pathogen_type} invades adjacent tissue from {source} to {target}",
            "airway": f"{pathogen_type} spreads along airway from {source} to {target}",
            "neural": f"{pathogen_type} travels along nerve pathway from {source} to {target}",
            "urinary": f"{pathogen_type} ascends urinary tract from {source} to {target}",
            "biliary": f"{pathogen_type} spreads through biliary system from {source} to {target}",
        }

        # Special clinical explanations
        if source == "portal_vein" and target == "liver":
            return f"Portal blood carries {pathogen_type} from GI to liver (first-pass filtration)"
        if source == "ivc" and target == "right_atrium":
            return f"{pathogen_type} in venous blood reaches heart via IVC → right atrium"
        if source == "right_ventricle" and "pulm" in target:
            return f"Heart pumps {pathogen_type}-laden blood to lungs (pulmonary circulation)"
        if "lung" in target:
            return f"{pathogen_type} reaches lungs — risk of pneumonia / respiratory seeding"
        if target == "brain":
            return f"{pathogen_type} crosses blood-brain barrier — risk of encephalitis/meningitis"
        if "kidney" in target:
            return f"{pathogen_type} reaches kidney via renal artery — risk of pyelonephritis"
        if target == "spleen":
            return f"{pathogen_type} filtered by spleen (immune response) — risk of splenic involvement"
        if target == "liver":
            return f"{pathogen_type} reaches liver — risk of hepatitis / abscess"

        return explanations.get(route, f"{pathogen_type} spreads from {source} to {target} via {route}")

    def _build_timeline(self, organs_at_risk: list) -> list:
        """Group spread into clinical timeline phases."""
        early = [o for o in organs_at_risk if o["hop"] <= 2]
        mid = [o for o in organs_at_risk if 3 <= o["hop"] <= 4]
        late = [o for o in organs_at_risk if o["hop"] >= 5]

        timeline = []
        if early:
            timeline.append({
                "phase": "early (hours)",
                "organs": [o["organ"] for o in early],
                "description": "Direct drainage and adjacent spread"
            })
        if mid:
            timeline.append({
                "phase": "intermediate (days)",
                "organs": [o["organ"] for o in mid],
                "description": "Hematogenous (blood-borne) spread"
            })
        if late:
            timeline.append({
                "phase": "late (days-weeks)",
                "organs": [o["organ"] for o in late],
                "description": "Secondary seeding to distant organs"
            })
        return timeline

    def _clinical_implications(self, origin: str, pathogen: str,
                               organs_at_risk: list) -> list:
        """Generate clinical implications from spread pattern."""
        implications = []
        organ_names = {o["organ"] for o in organs_at_risk}

        if "liver" in organ_names:
            implications.append(f"Liver involvement: monitor LFTs, risk of hepatitis/abscess")
        if "lungs" in organ_names or "r_lung" in organ_names or "l_lung" in organ_names:
            implications.append(f"Pulmonary seeding: chest X-ray recommended, risk of pneumonia")
        if "brain" in organ_names:
            implications.append(f"CNS involvement: neurologic monitoring, risk of encephalitis/meningitis")
        if "r_kidney" in organ_names or "l_kidney" in organ_names:
            implications.append(f"Renal involvement: monitor creatinine, risk of pyelonephritis")
        if "heart" in organ_names:
            implications.append(f"Cardiac involvement: echo if suspected endocarditis")
        if "spleen" in organ_names:
            implications.append(f"Splenic involvement: imaging if suspected abscess")
        if "meninges" in organ_names:
            implications.append(f"Meningeal involvement: lumbar puncture if meningitis suspected")

        if pathogen == "bacteria":
            implications.append("Blood cultures x2 before antibiotics")
            implications.append("Monitor for sepsis criteria (qSOFA)")
        elif pathogen == "virus":
            implications.append("PCR/serology for viral identification")
            implications.append("Supportive care, monitor for systemic inflammatory response")

        return implications