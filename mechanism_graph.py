"""
Mechanism Graph Engine

Builds a directed reasoning graph from your mechanism data:

    symptom ─triggers→ mechanism ─pathway→ [step1 → step2 → step3] ─produces→ effect ─causes→ disease

Every reasoning step is traceable. No embeddings, no similarity.
The graph answers questions like:
  - "Why does this symptom lead to this disease?" → causal chain
  - "What mechanisms are activated by these symptoms?" → graph traversal
  - "If mechanism X is active, what else should happen?" → forward inference
  - "These 3 symptoms share which mechanism?" → convergence detection

Replaces RAG search with structured graph queries.
"""

import json
import os
import glob
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional


class MechanismNode:
    __slots__ = ['id', 'title', 'domain', 'category', 'pathway_steps',
                 'trigger', 'effects', 'symptoms', 'diseases',
                 'red_flags', 'labs', 'vitals']

    def __init__(self, mech_id, title, domain="general"):
        self.id = mech_id
        self.title = title
        self.domain = domain
        self.category = []
        self.pathway_steps = []
        self.trigger = []
        self.effects = set()
        self.symptoms = set()
        self.diseases = set()
        self.red_flags = []
        self.labs = []
        self.vitals = []


class MechanismGraph:
    """
    Directed graph of mechanisms with full causal chain traversal.
    """

    def __init__(self):
        self.nodes: Dict[str, MechanismNode] = {}

        # Forward indexes (for graph traversal)
        self.symptom_to_mechs: Dict[str, Set[str]] = defaultdict(set)
        self.effect_to_mechs: Dict[str, Set[str]] = defaultdict(set)
        self.disease_to_mechs: Dict[str, Set[str]] = defaultdict(set)
        self.category_to_mechs: Dict[str, Set[str]] = defaultdict(set)
        self.trigger_to_mechs: Dict[str, Set[str]] = defaultdict(set)

        # Reverse indexes
        self.mech_to_symptoms: Dict[str, Set[str]] = defaultdict(set)
        self.mech_to_effects: Dict[str, Set[str]] = defaultdict(set)
        self.mech_to_diseases: Dict[str, Set[str]] = defaultdict(set)

        # Effect chains: effect A can trigger mechanism B
        self.effect_chains: Dict[str, Set[str]] = defaultdict(set)

        self._loaded = False

    def load(self, knowledge_dir: str = "medical_knowledge"):
        """Load all mechanisms and build the graph."""
        if self._loaded:
            return

        for fname in ["mechanisms_rag_final.json", "bacteria_mechanisms_1200.json",
                       "virus_mechanisms_1500.json"]:
            _script_dir = os.path.dirname(os.path.abspath(__file__))
            search_paths = [
                os.path.join(knowledge_dir, "mechanisms", fname),
                os.path.join(knowledge_dir, fname),
                fname,
                os.path.join(_script_dir, fname),
                os.path.join(os.path.dirname(_script_dir), fname),
                os.path.join("nexus_engine", fname),
            ]
            path = next((p for p in search_paths if os.path.exists(p)), None)
            if not path:
                continue
            try:
                data = json.load(open(path, "r", encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(data, list):
                continue

            for m in data:
                self._add_mechanism(m)

        # Build effect chains: if effect A is also a symptom that triggers mechanism B
        self._build_effect_chains()

        self._loaded = True
        print(f"[MECH_GRAPH] {len(self.nodes)} mechanisms, "
              f"{sum(len(v) for v in self.symptom_to_mechs.values())} symptom→mech links, "
              f"{sum(len(v) for v in self.effect_chains.values())} effect chains")

    def _add_mechanism(self, m: dict):
        mech_id = m.get("id") or m.get("title", "")
        if not mech_id or mech_id in self.nodes:
            return

        node = MechanismNode(
            mech_id,
            m.get("title", ""),
            m.get("domain", "general"),
        )

        # Category
        cat = m.get("category", "")
        if isinstance(cat, str) and cat:
            node.category = [cat.lower()]
        elif isinstance(cat, list):
            node.category = [c.lower() for c in cat if isinstance(c, str)]

        # Pathway (causal steps)
        node.pathway_steps = m.get("pathway", m.get("core_mechanism_steps", []))
        if isinstance(node.pathway_steps, str):
            node.pathway_steps = [node.pathway_steps]

        # Trigger
        trigger = m.get("trigger", [])
        if isinstance(trigger, str):
            trigger = [trigger]
        node.trigger = [t.lower() for t in trigger if isinstance(t, str)]

        # Effects
        effects = m.get("effects", m.get("primary_effects", []))
        for eff in effects:
            if isinstance(eff, str) and eff.strip():
                e = eff.lower().strip()
                node.effects.add(e)
                self.effect_to_mechs[e].add(mech_id)
                self.mech_to_effects[mech_id].add(e)

        # Symptoms
        for sym in m.get("typical_symptoms", []):
            if isinstance(sym, str) and sym.strip():
                s = sym.lower().strip()
                node.symptoms.add(s)
                self.symptom_to_mechs[s].add(mech_id)
                self.mech_to_symptoms[mech_id].add(s)

        # Diseases
        diseases = m.get("linked_diseases", m.get("related_diseases", []))
        if isinstance(diseases, str):
            diseases = [diseases]
        for dis in diseases:
            if isinstance(dis, str) and dis.strip():
                d = dis.strip().lower()
                node.diseases.add(d)
                self.disease_to_mechs[d].add(mech_id)
                self.mech_to_diseases[mech_id].add(d)

        # Red flags, labs, vitals
        node.red_flags = m.get("red_flags", m.get("key_red_flags", []))
        node.labs = m.get("typical_labs", [])
        node.vitals = m.get("typical_vitals", [])

        # Category index
        for cat in node.category:
            self.category_to_mechs[cat].add(mech_id)

        # Trigger index
        for t in node.trigger:
            self.trigger_to_mechs[t].add(mech_id)

        self.nodes[mech_id] = node

    def _build_effect_chains(self):
        """If effect A produced by mechanism X is also a symptom that triggers mechanism Y,
        then X → Y is an effect chain (cascade)."""
        all_effects = set()
        for mech_id, effects in self.mech_to_effects.items():
            all_effects.update(effects)

        for effect in all_effects:
            # Does this effect trigger other mechanisms (as a symptom)?
            if effect in self.symptom_to_mechs:
                for downstream_mech in self.symptom_to_mechs[effect]:
                    # Every mechanism that produces this effect chains to mechanisms triggered by it
                    for upstream_mech in self.effect_to_mechs[effect]:
                        if upstream_mech != downstream_mech:
                            self.effect_chains[upstream_mech].add(downstream_mech)

    # ═══════════════════════════════════════
    # GRAPH QUERIES — structured reasoning
    # ═══════════════════════════════════════

    def reason_from_symptoms(self, symptoms: List[str], max_chain_depth: int = 3) -> dict:
        """
        Main reasoning entry point.
        Builds the full causal chain: symptoms → mechanisms → effects → diseases.

        Returns structured reasoning with every step traceable.
        """
        self.load()
        sym_set = set(s.lower().strip().replace(" ", "_") for s in symptoms)

        # Step 1: Find directly activated mechanisms
        activated = set()
        activation_map = {}  # mech_id → which symptoms activated it
        for sym in sym_set:
            for s_key in self.symptom_to_mechs:
                if sym == s_key or sym in s_key or s_key in sym:
                    for mech_id in self.symptom_to_mechs[s_key]:
                        activated.add(mech_id)
                        activation_map.setdefault(mech_id, set()).add(sym)

        # Step 2: Follow effect chains (cascade reasoning)
        cascade_mechs = set()
        cascade_chains = []
        visited = set(activated)
        frontier = set(activated)

        for depth in range(max_chain_depth):
            next_frontier = set()
            for mech_id in frontier:
                for downstream in self.effect_chains.get(mech_id, set()):
                    if downstream not in visited:
                        visited.add(downstream)
                        next_frontier.add(downstream)
                        cascade_mechs.add(downstream)
                        # Find the bridging effect
                        upstream_effects = self.mech_to_effects.get(mech_id, set())
                        downstream_symptoms = self.mech_to_symptoms.get(downstream, set())
                        bridge = upstream_effects & downstream_symptoms
                        cascade_chains.append({
                            "from_mech": self.nodes[mech_id].title[:60] if mech_id in self.nodes else mech_id,
                            "to_mech": self.nodes[downstream].title[:60] if downstream in self.nodes else downstream,
                            "bridge_effect": sorted(bridge)[:3],
                            "depth": depth + 1,
                        })
            frontier = next_frontier
            if not frontier:
                break

        # Step 3: Collect all effects from activated mechanisms
        all_effects = set()
        for mech_id in activated | cascade_mechs:
            all_effects.update(self.mech_to_effects.get(mech_id, set()))

        # Step 4: Predict additional symptoms from effects
        predicted_symptoms = set()
        for mech_id in activated | cascade_mechs:
            for sym in self.mech_to_symptoms.get(mech_id, set()):
                if sym not in sym_set:
                    predicted_symptoms.add(sym)

        # Step 5: Score diseases by mechanism support
        disease_scores = defaultdict(lambda: {"score": 0, "mechanisms": [], "coverage": 0})
        for mech_id in activated | cascade_mechs:
            node = self.nodes.get(mech_id)
            if not node:
                continue
            # How many input symptoms does this mechanism explain?
            explained = len(activation_map.get(mech_id, set()))
            is_cascade = mech_id in cascade_mechs

            for disease in node.diseases:
                ds = disease_scores[disease]
                weight = explained * (0.5 if is_cascade else 1.0)
                ds["score"] += weight
                if len(ds["mechanisms"]) < 5:
                    ds["mechanisms"].append({
                        "id": node.title[:60],
                        "domain": node.domain,
                        "explained_symptoms": sorted(activation_map.get(mech_id, set()))[:3],
                        "cascade": is_cascade,
                    })

        # Calculate coverage: what fraction of input symptoms does each disease explain?
        for disease, ds in disease_scores.items():
            all_explained = set()
            for m_info in ds["mechanisms"]:
                all_explained.update(m_info["explained_symptoms"])
            ds["coverage"] = len(all_explained & sym_set) / max(len(sym_set), 1)
            ds["explained"] = sorted(all_explained & sym_set)
            ds["unexplained"] = sorted(sym_set - all_explained)

        # Sort by score
        ranked = sorted(disease_scores.items(), key=lambda x: -x[1]["score"])

        # Step 6: Collect red flags and labs
        red_flags = set()
        predicted_labs = set()
        for mech_id in activated:
            node = self.nodes.get(mech_id)
            if node:
                for rf in node.red_flags:
                    if isinstance(rf, str):
                        red_flags.add(rf)
                for lab in node.labs:
                    if isinstance(lab, str):
                        predicted_labs.add(lab)

        # Step 7: Build causal chains (the "reasoning trace")
        causal_chains = []
        for disease, ds in ranked[:5]:
            chain = {
                "disease": disease,
                "score": round(ds["score"], 2),
                "coverage": round(ds["coverage"], 2),
                "chain": [],
            }
            for m_info in ds["mechanisms"][:3]:
                mech_title = m_info["id"]
                # Find the node
                node = None
                for n in self.nodes.values():
                    if n.title[:60] == mech_title:
                        node = n
                        break
                if node:
                    chain["chain"].append({
                        "symptoms": m_info["explained_symptoms"],
                        "mechanism": node.title,
                        "pathway": node.pathway_steps[:3],
                        "effects": sorted(node.effects)[:3],
                        "domain": node.domain,
                    })
            causal_chains.append(chain)

        # Step 8: Domain analysis (virus vs bacteria)
        domain_counts = defaultdict(int)
        for mech_id in activated:
            node = self.nodes.get(mech_id)
            if node:
                domain_counts[node.domain] += 1

        return {
            "activated_mechanisms": len(activated),
            "cascade_mechanisms": len(cascade_mechs),
            "total_mechanisms": len(activated) + len(cascade_mechs),
            "cascade_chains": cascade_chains[:10],
            "effects_produced": sorted(all_effects)[:20],
            "predicted_symptoms": sorted(predicted_symptoms)[:15],
            "disease_ranking": [
                {
                    "disease": d,
                    "score": round(ds["score"], 2),
                    "coverage": round(ds["coverage"], 2),
                    "explained": ds["explained"],
                    "unexplained": ds["unexplained"],
                    "mechanism_count": len(ds["mechanisms"]),
                }
                for d, ds in ranked[:10]
            ],
            "causal_chains": causal_chains,
            "red_flags": sorted(red_flags),
            "predicted_labs": sorted(predicted_labs),
            "domain_analysis": dict(domain_counts),
        }

    def explain_why(self, symptom: str, disease: str) -> List[dict]:
        """
        Answer: "Why could this symptom be caused by this disease?"
        Returns all causal chains connecting symptom → ... → disease.
        """
        self.load()
        sym = symptom.lower().strip()
        dis = disease.lower().strip()

        chains = []
        # Find mechanisms that have BOTH this symptom and this disease
        sym_mechs = set()
        for s_key, mechs in self.symptom_to_mechs.items():
            if sym == s_key or sym in s_key:
                sym_mechs.update(mechs)

        dis_mechs = self.disease_to_mechs.get(dis, set())
        shared = sym_mechs & dis_mechs

        for mech_id in shared:
            node = self.nodes.get(mech_id)
            if node:
                chains.append({
                    "mechanism": node.title,
                    "pathway": node.pathway_steps,
                    "domain": node.domain,
                    "effects": sorted(node.effects)[:5],
                    "reasoning": f"{symptom} → [{node.title}] → {disease}",
                })

        return chains

    def find_convergence(self, symptoms: List[str]) -> List[dict]:
        """
        Find mechanisms that explain MULTIPLE symptoms simultaneously.
        These are the strongest diagnostic signals.
        """
        self.load()
        sym_set = set(s.lower().strip() for s in symptoms)

        mech_coverage = {}
        for sym in sym_set:
            for s_key, mechs in self.symptom_to_mechs.items():
                if sym == s_key or sym in s_key:
                    for mech_id in mechs:
                        mech_coverage.setdefault(mech_id, set()).add(sym)

        # Only keep mechanisms that explain 2+ symptoms
        convergent = []
        for mech_id, explained in mech_coverage.items():
            if len(explained) >= 2:
                node = self.nodes.get(mech_id)
                if node:
                    convergent.append({
                        "mechanism": node.title,
                        "explains": sorted(explained),
                        "coverage": len(explained) / len(sym_set),
                        "diseases": sorted(node.diseases)[:5],
                        "domain": node.domain,
                    })

        convergent.sort(key=lambda x: -x["coverage"])
        return convergent[:10]

    def get_stats(self) -> dict:
        self.load()
        return {
            "total_mechanisms": len(self.nodes),
            "unique_symptoms": len(self.symptom_to_mechs),
            "unique_effects": len(self.effect_to_mechs),
            "unique_diseases": len(self.disease_to_mechs),
            "effect_chains": sum(len(v) for v in self.effect_chains.values()),
            "domains": dict(defaultdict(int, {
                n.domain: 1 for n in self.nodes.values()
            })),
        }