"""
NEXUS  —  NEXUS  auto_learning
"""
from __future__ import annotations
import json, os, math
from typing import Dict, List, Any
from collections import defaultdict


class NexusLearner:
    def __init__(self, nexus_medical):
        self.nexus = nexus_medical
        self.kg = getattr(nexus_medical, 'kg', None)
        self.stats = {"cases_learned":0,"traces_learned":0,"interactions_learned":0,"penalties_learned":0,"triples_added":0}
        self._freq = defaultdict(lambda: defaultdict(int))

    def _kg_add(self, *args, **kwargs):
        if self.kg: self.kg.add(*args, **kwargs)

    def _kg_query(self, *args, **kwargs):
        return self.kg.query(*args, **kwargs) if self.kg else []

    def _kg_len(self):
        return len(self.kg) if self.kg else 0

    def _kg_entities(self):
        return self.kg.entities() if self.kg and hasattr(self.kg, "entities") else set()

    def learn_from_all(self, case_path="case_records.jsonl", trace_path="auto_learning/reasoning_trace_log.jsonl",
                       interaction_path="auto_learning/interaction_log.jsonl", penalty_path="auto_learning/penalties.json",
                       coverage_path="auto_learning/disease_coverage.json", weight_path="auto_learning/mechanism_weights.json"):
        self.learn_from_cases(case_path)
        self.learn_from_traces(trace_path)
        self.learn_from_interactions(interaction_path)
        self.learn_from_penalties(penalty_path)
        self.learn_from_coverage(coverage_path)
        self.learn_from_mechanism_weights(weight_path)
        print(f"[NEXUS-LEARN] done: {self.stats} | graph: {self._kg_len()} triples")
        return dict(self.stats)

    def learn_from_cases(self, path="case_records.jsonl", min_reward=0.3, max_n=5000):
        if not os.path.exists(path): return 0
        count = 0
        for row in self._iter_jsonl(path, max_n):
            ri = row.get("reward_final", {})
            reward = float(ri.get("final", ri.get("trace", 0)) if isinstance(ri, dict) else (ri or 0))
            if reward < min_reward: continue
            symptoms, diseases = row.get("final_symptoms", []), row.get("top_diseases", [])
            if not symptoms or not diseases: continue
            count += 1
            for di in diseases:
                dn, sc = self._parse_disease(di)
                if not dn or sc < 0.1: continue
                conf = max(0.3, min(0.92, min(sc/5.0, 1.0) * min(reward, 1.0)))
                for sym in symptoms:
                    s = self._norm(sym)
                    if s:
                        self._kg_add(dn, "has_symptom", s, conf, "case_learning")
                        self._freq[s][dn] += 1; self.stats["triples_added"] += 1
        self.stats["cases_learned"] = count
        if count: print(f"[NEXUS-LEARN] cases: {count}")
        return count

    def learn_from_traces(self, path="auto_learning/reasoning_trace_log.jsonl", min_reward=0.2, max_n=3000):
        if not os.path.exists(path): return 0
        count = 0
        for row in self._iter_jsonl(path, max_n):
            trace = row.get("trace", row)
            ri = row.get("reward", trace.get("reward_path_level", {}))
            reward = float(ri.get("total", ri.get("final", 0)) if isinstance(ri, dict) else (ri or 0))
            if reward < min_reward: continue
            count += 1
            for m in (trace.get("mechanisms", trace.get("mech_hits", [])) or []):
                if not isinstance(m, dict): continue
                title = self._norm(m.get("title") or m.get("mechanism") or "")
                score = float(m.get("score", 0) or 0)
                if not title or score < 0.1: continue
                for d in m.get("related_diseases", []):
                    dn = self._norm(d)
                    if dn: self._kg_add(title, "causes", dn, min(score*min(reward,1.0), 0.9), "trace_learning"); self.stats["triples_added"] += 1
                for s in m.get("matched_symptoms", []):
                    sn = self._norm(s)
                    if sn: self._kg_add(sn, "triggers", title, 0.7, "trace_learning"); self.stats["triples_added"] += 1
        self.stats["traces_learned"] = count
        if count: print(f"[NEXUS-LEARN] traces: {count}")
        return count

    def learn_from_interactions(self, path="auto_learning/interaction_log.jsonl", max_n=2000):
        if not os.path.exists(path): return 0
        count = 0
        for item in self._iter_jsonl(path, max_n):
            if float(item.get("reward", 0) or 0) < 0.2: continue
            for di in (item.get("top_diseases", []) or []):
                dn, _ = self._parse_disease(di)
                if not dn: continue
                for sym in (item.get("parsed_symptoms", item.get("symptoms", [])) or []):
                    s = self._norm(sym if isinstance(sym, str) else (sym.get("symptom","") if isinstance(sym, dict) else ""))
                    if s: self._kg_add(dn, "has_symptom", s, 0.7, "interaction_learning"); self.stats["triples_added"] += 1
            count += 1
        self.stats["interactions_learned"] = count
        if count: print(f"[NEXUS-LEARN] interactions: {count}")
        return count

    def learn_from_penalties(self, path="auto_learning/penalties.json"):
        if not os.path.exists(path): return 0
        try:
            data = json.load(open(path, "r", encoding="utf-8"))
        except: return 0
        count = 0
        for disease, info in (data.get("diseases", {}) or {}).items():
            pc = int(info.get("count", info) if isinstance(info, dict) else (info or 0))
            if pc >= 3:
                d = self._norm(disease)
                if d: self._kg_add(d, "has_penalty", "high_error_rate", 0.8, "penalty"); count += 1
        for mech, info in (data.get("mechanisms", {}) or {}).items():
            pc = int(info.get("count", info) if isinstance(info, dict) else (info or 0))
            if pc >= 3:
                m = self._norm(mech)
                if m: self._kg_add(m, "has_penalty", "unreliable", 0.7, "penalty"); count += 1
        self.stats["penalties_learned"] = count
        return count

    def learn_from_coverage(self, path="auto_learning/disease_coverage.json"):
        if not os.path.exists(path): return 0
        try: data = json.load(open(path, "r", encoding="utf-8"))
        except: return 0
        count = 0
        for system, info in data.items():
            c = int(info.get("count", info) if isinstance(info, dict) else (info or 0))
            if c < 10: continue
            s = self._norm(system)
            if s:
                conf = min(0.5 + math.log(c+1)*0.05, 0.95)
                self._kg_add(s, "coverage_level", "well_studied" if c>100 else "moderate", conf, "coverage"); count += 1
        return count

    def learn_from_mechanism_weights(self, path="auto_learning/mechanism_weights.json"):
        if not os.path.exists(path): return 0
        try: data = json.load(open(path, "r", encoding="utf-8"))
        except: return 0
        w_mech = data.get("w_mech", {})
        count = 0
        for title, weight in w_mech.items():
            w = float(weight or 1.0); m = self._norm(title)
            if not m: continue
            if w > 1.2: self._kg_add(m, "learned_weight", "high", min(w/2.0, 0.95), "weight"); count += 1
            elif w < 0.7: self._kg_add(m, "learned_weight", "low", 0.6, "weight"); count += 1
        return count

    # ═══ : NEXUS → auto_learning ═══
    def feedback(self, enhanced_result, round_id=0):
        self._fb_coverage(enhanced_result, round_id)
        self._fb_trace(enhanced_result)
        self._fb_weights(enhanced_result)

    def _fb_coverage(self, result, round_id):
        try:
            from auto_learning.disease_coverage import load_coverage, update_coverage, save_coverage
        except ImportError: return
        systems = set()
        for dx in result.get("nexus_diagnoses", [])[:5]:
            for t in self._kg_query(dx.get("disease",""), "belongs_to_system"):
                systems.add(t.object)
        if systems:
            cov = load_coverage(); cov = update_coverage(cov, list(systems), inc=1, round_id=round_id); save_coverage(cov)

    def _fb_trace(self, result):
        try:
            from auto_learning.trace_logger import log_reasoning_trace
        except ImportError: return
        ns = result.get("nexus_stats", {})
        if ns:
            log_reasoning_trace({"source":"nexus","mechanisms":ns.get("mechanisms_activated",0),
                "diseases":ns.get("diseases_considered",0),
                "consistency":result.get("nexus_consistency",{}).get("consistency_score",0)})

    def _fb_weights(self, result):
        try:
            from auto_learning.mechanism_weight_learner import update_mechanism_weights
        except ImportError: return
        me = result.get("nexus_mechanism_evidence", {})
        if not me: return
        cons = result.get("nexus_consistency",{}).get("consistency_score", 0.5)
        hits = []
        for disease, evs in me.items():
            for ev in evs: hits.append({"title":f"nexus:{disease}","score":ev.get("confidence",0.5)})
        if hits:
            try: update_mechanism_weights({"mechanism_hits":hits}, cons-0.5, beta=0.02)
            except: pass

    # ═══ Utils ═══
    @staticmethod
    def _norm(s):
        return str(s).strip().lower().replace("_"," ") if s else ""

    @staticmethod
    def _parse_disease(di):
        if isinstance(di, str): return di.strip().lower().replace("_"," "), 0.5
        if isinstance(di, dict):
            n = (di.get("disease") or di.get("disease_name") or di.get("name") or "")
            sc = float(di.get("total_score", di.get("score", 0.5)) or 0.5)
            return n.strip().lower().replace("_"," "), sc
        return "", 0.0

    @staticmethod
    def _iter_jsonl(path, max_n):
        count = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if count >= max_n: break
                line = line.strip()
                if not line: continue
                try: yield json.loads(line); count += 1
                except: continue

    def get_learned_stats(self):
        return {**self.stats, "graph_size": self._kg_len(), "entities": len(self._kg_entities())}