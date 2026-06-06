"""
NEXUS Core v1.0
═══════════════════════════════════════════════════════════════
General-purpose RDF-style knowledge graph + symbolic reasoning engine.

This file is a STANDALONE reasoning toolkit:
  1. KnowledgeGraph    — RDF-style triples + indexing
  2. InferenceRule     — first-order rule definitions
  3. LogicEngine       — forward chain + backward chain
  4. ConstraintSolver  — AC-3 arc consistency + backtracking
  5. AbductiveReasoner — find best explanation for an observation
  6. NexusOutput       — JSON encoding for downstream consumers

It does NOT load disease/symptom/mechanism data. Anyone using this engine is
responsible for populating the KnowledgeGraph. The NEXUS pure-mechanism
pipeline does not currently populate this KG — it is available for future
use if needed (e.g. for prove/explain endpoints over arbitrary KG data).
"""
from __future__ import annotations
import json
import time
from typing import (
    Any, Dict, List, Optional, Set, Tuple, Callable, Union
)
from dataclasses import dataclass, field
from collections import defaultdict


# ═══════════════════════════════════════════════════════════════
# 1. Knowledge Graph
# ═══════════════════════════════════════════════════════════════

@dataclass
class Triple:
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    source: str = "axiom"
    id: int = -1
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "subject":    self.subject,
            "predicate":  self.predicate,
            "object":     self.object,
            "confidence": round(self.confidence, 4),
            "source":     self.source,
            "id":         self.id,
        }

    def __repr__(self):
        return f"({self.subject} —[{self.predicate}]→ {self.object}  conf={self.confidence:.2f})"


class KnowledgeGraph:
    """RDF-style triple store with subject/predicate/object indexes."""

    def __init__(self):
        self.triples: List[Triple] = []
        self._by_subject:   Dict[str, Set[int]] = defaultdict(set)
        self._by_predicate: Dict[str, Set[int]] = defaultdict(set)
        self._by_object:    Dict[str, Set[int]] = defaultdict(set)
        self._type_hierarchy: Dict[str, str] = {}   # child -> parent

    # ── Add ──
    def add(self, subject: str, predicate: str, obj: str,
            confidence: float = 1.0, source: str = "axiom") -> Triple:
        """Add triple; dedupe by exact match (keep max confidence)."""
        existing = self.query(subject, predicate, obj)
        if existing:
            t = existing[0]
            t.confidence = max(t.confidence, confidence)
            return t
        idx = len(self.triples)
        t = Triple(subject=subject, predicate=predicate, object=obj,
                   confidence=confidence, source=source, id=idx)
        self.triples.append(t)
        self._by_subject[subject].add(idx)
        self._by_predicate[predicate].add(idx)
        self._by_object[obj].add(idx)
        return t

    def add_many(self, triples: List[Tuple[str, str, str, float, str]]):
        """Bulk add: (subject, predicate, object[, confidence[, source]])."""
        for item in triples:
            s, p, o = item[0], item[1], item[2]
            conf = item[3] if len(item) > 3 else 1.0
            src  = item[4] if len(item) > 4 else "axiom"
            self.add(s, p, o, conf, src)

    # ── Query ──
    def query(self, subject: Optional[str] = None,
              predicate: Optional[str] = None,
              obj: Optional[str] = None,
              min_confidence: float = 0.0) -> List[Triple]:
        """None = wildcard. Returns all matching triples."""
        candidates: Optional[Set[int]] = None
        if subject is not None:
            candidates = set(self._by_subject.get(subject, set()))
        if predicate is not None:
            p = self._by_predicate.get(predicate, set())
            candidates = candidates & p if candidates is not None else set(p)
        if obj is not None:
            o = self._by_object.get(obj, set())
            candidates = candidates & o if candidates is not None else set(o)

        pool = candidates if candidates is not None else range(len(self.triples))
        results = []
        for i in pool:
            t = self.triples[i]
            if subject   is not None and t.subject   != subject:   continue
            if predicate is not None and t.predicate != predicate: continue
            if obj       is not None and t.object    != obj:       continue
            if t.confidence >= min_confidence:
                results.append(t)
        return results

    # ── Type hierarchy ──
    def define_type_hierarchy(self, child: str, parent: str):
        self._type_hierarchy[child] = parent

    def is_a(self, entity: str, type_name: str) -> bool:
        """Walks 'is_a'/'类型' edges + the explicit hierarchy table."""
        direct = self.query(entity, "is_a", None) + self.query(entity, "类型", None)
        for t in direct:
            current = t.object
            while current:
                if current == type_name:
                    return True
                current = self._type_hierarchy.get(current)
        return False

    # ── Misc ──
    def entities(self) -> Set[str]:
        return set(self._by_subject.keys()) | set(self._by_object.keys())

    def predicates(self) -> Set[str]:
        return set(self._by_predicate.keys())

    def neighbors(self, entity: str) -> List[Triple]:
        indices = self._by_subject.get(entity, set()) | self._by_object.get(entity, set())
        return [self.triples[i] for i in indices]

    def to_json(self) -> list:
        return [t.to_dict() for t in self.triples]

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.triples)

    def __repr__(self):
        return f"KnowledgeGraph({len(self.triples)} triples, {len(self.entities())} entities)"


# ═══════════════════════════════════════════════════════════════
# 2. Inference Rule
# ═══════════════════════════════════════════════════════════════

@dataclass
class InferenceRule:
    """
    conditions:  list of (subject, predicate, object) patterns
                 Variables start with "?" (e.g. "?X")
    conclusion:  one (subject, predicate, object) pattern
    confidence_decay: multiplier applied to derived confidence (default 0.95)
    """
    name: str
    conditions: List[Tuple[str, str, str]]
    conclusion: Tuple[str, str, str]
    confidence_decay: float = 0.95

    def calc_confidence(self, matched_triples: List[Triple]) -> float:
        if not matched_triples:
            return 0.5
        return min(t.confidence for t in matched_triples) * self.confidence_decay


# ═══════════════════════════════════════════════════════════════
# 3. Logic Engine — forward & backward chaining
# ═══════════════════════════════════════════════════════════════

class LogicEngine:
    """Forward chain (data-driven) and backward chain (goal-driven) reasoner."""

    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg
        self.rules: List[InferenceRule] = []
        self.inference_log: List[dict] = []
        self.max_depth: int = 12

    def add_rule(self, rule: InferenceRule):
        self.rules.append(rule)

    def add_rules(self, rules: List[InferenceRule]):
        self.rules.extend(rules)

    # ── Helpers ──
    @staticmethod
    def _is_var(s: str) -> bool:
        return isinstance(s, str) and s.startswith("?")

    def _match_pattern(self, pattern: Tuple[str, str, str], triple: Triple,
                       bindings: Dict[str, str]) -> Optional[Dict[str, str]]:
        """Match pattern against triple under current bindings. Returns new bindings or None."""
        new_bindings = dict(bindings)
        fields = [
            (pattern[0], triple.subject),
            (pattern[1], triple.predicate),
            (pattern[2], triple.object),
        ]
        for pval, tval in fields:
            if pval is None:
                continue
            if self._is_var(pval):
                if pval in new_bindings:
                    if new_bindings[pval] != tval:
                        return None
                else:
                    new_bindings[pval] = tval
            else:
                if pval != tval:
                    return None
        return new_bindings

    def _resolve_pattern(self, pattern: Tuple[str, str, str],
                         bindings: Dict[str, str]) -> Tuple[Optional[str], ...]:
        """Substitute bound variables in pattern."""
        resolved = []
        for p in pattern:
            if p is None:
                resolved.append(None)
            elif self._is_var(p) and p in bindings:
                resolved.append(bindings[p])
            else:
                resolved.append(p)
        return tuple(resolved)

    # ── Find all bindings for a conjunction of patterns ──
    def _find_bindings(self, conditions: List[Tuple[str, str, str]],
                       idx: int = 0,
                       bindings: Dict[str, str] = None) -> List[Dict[str, Any]]:
        if bindings is None:
            bindings = {}
        if idx >= len(conditions):
            return [{"bindings": dict(bindings), "triples": []}]

        pattern = conditions[idx]
        resolved = self._resolve_pattern(pattern, bindings)
        q_args = [None if (r and self._is_var(r)) else r for r in resolved]
        candidates = self.kg.query(*q_args)

        results = []
        for triple in candidates:
            new_bindings = self._match_pattern(pattern, triple, bindings)
            if new_bindings is not None:
                deeper = self._find_bindings(conditions, idx + 1, new_bindings)
                for d in deeper:
                    d["triples"] = [triple] + d.get("triples", [])
                    results.append(d)
        return results

    # ── Forward chaining ──
    def forward_chain(self, max_iterations: int = 50) -> Dict[str, Any]:
        self.inference_log = []
        new_facts = []
        iteration = 0
        while iteration < max_iterations:
            found_new = False
            iteration += 1
            for rule in self.rules:
                binding_results = self._find_bindings(rule.conditions)
                for result in binding_results:
                    bindings = result["bindings"]
                    triples  = result["triples"]
                    conc = self._resolve_pattern(rule.conclusion, bindings)
                    s, p, o = conc
                    if not s or not p or not o:
                        continue
                    if self._is_var(s) or self._is_var(p) or self._is_var(o):
                        continue
                    if self.kg.query(s, p, o):
                        continue
                    conf = rule.calc_confidence(triples)
                    new_triple = self.kg.add(s, p, o, conf, f"inferred:{rule.name}")
                    new_facts.append(new_triple)
                    found_new = True
                    self.inference_log.append({
                        "step":     len(self.inference_log),
                        "rule":     rule.name,
                        "bindings": dict(bindings),
                        "inputs":   [t.to_dict() for t in triples],
                        "output":   new_triple.to_dict(),
                    })
            if not found_new:
                break
        return {
            "new_facts":     [t.to_dict() for t in new_facts],
            "iterations":    iteration,
            "total_triples": len(self.kg),
            "log":           self.inference_log,
        }

    # ── Backward chaining ──
    def backward_chain(self, goal: Tuple[str, str, str],
                       depth: int = 0,
                       visited: Set[str] = None) -> Dict[str, Any]:
        if visited is None:
            visited = set()
        goal_key = f"{goal[0]}|{goal[1]}|{goal[2]}"
        if goal_key in visited or depth > self.max_depth:
            return {"proven": False, "proof": [], "confidence": 0.0}
        visited.add(goal_key)

        # 1. Direct facts
        s = None if self._is_var(goal[0]) else goal[0]
        p = None if self._is_var(goal[1]) else goal[1]
        o = None if self._is_var(goal[2]) else goal[2]
        direct = self.kg.query(s, p, o)
        if direct:
            return {
                "proven":     True,
                "proof":      [{"type": "fact", "triple": direct[0].to_dict()}],
                "confidence": direct[0].confidence,
            }

        # 2. Try rules
        for rule in self.rules:
            dummy = Triple(subject=goal[0], predicate=goal[1], object=goal[2])
            match = self._match_pattern(rule.conclusion, dummy, {})
            if match is None:
                continue
            all_proven = True
            sub_proofs = []
            min_conf   = 1.0
            for condition in rule.conditions:
                resolved = self._resolve_pattern(condition, match)
                sub = self.backward_chain(resolved, depth + 1, set(visited))
                if not sub["proven"]:
                    all_proven = False
                    break
                sub_proofs.append(sub)
                min_conf = min(min_conf, sub.get("confidence", 0.5))
            if all_proven:
                return {
                    "proven":     True,
                    "proof":      [{"type": "rule", "name": rule.name, "sub_proofs": sub_proofs}],
                    "confidence": min_conf * rule.confidence_decay,
                }
        return {"proven": False, "proof": [], "confidence": 0.0}


# ═══════════════════════════════════════════════════════════════
# 4. Constraint Solver (AC-3 + backtracking with MRV)
# ═══════════════════════════════════════════════════════════════

class ConstraintSolver:
    """Generic CSP solver."""

    def __init__(self):
        self.variables: Dict[str, Set] = {}
        self.constraints: List[Dict] = []

    def add_variable(self, name: str, domain: list):
        self.variables[name] = set(domain)

    def add_constraint(self, vars: List[str],
                       check: Callable[[dict], bool], name: str = ""):
        self.constraints.append({"vars": vars, "check": check, "name": name})

    def _revise(self, xi: str, xj: str, constraint: dict) -> list:
        removed = []
        domain_i = self.variables[xi]
        domain_j = self.variables[xj]
        for vi in list(domain_i):
            supported = False
            for vj in domain_j:
                if constraint["check"]({xi: vi, xj: vj}):
                    supported = True
                    break
            if not supported:
                domain_i.discard(vi)
                removed.append(vi)
        return removed

    def arc_consistency(self) -> Dict[str, Any]:
        queue = []
        steps = []
        for c in self.constraints:
            if len(c["vars"]) == 2:
                queue.append((c["vars"][0], c["vars"][1], c))
                queue.append((c["vars"][1], c["vars"][0], c))
        while queue:
            xi, xj, constraint = queue.pop(0)
            removed = self._revise(xi, xj, constraint)
            if removed:
                steps.append({
                    "variable":    xi,
                    "removed":     removed,
                    "domain_size": len(self.variables[xi]),
                })
                if len(self.variables[xi]) == 0:
                    return {"consistent": False, "steps": steps}
                for c in self.constraints:
                    if xi in c["vars"]:
                        for xk in c["vars"]:
                            if xk != xi and xk != xj:
                                queue.append((xk, xi, c))
        return {"consistent": True, "steps": steps}

    def solve(self) -> Dict[str, Any]:
        ac = self.arc_consistency()
        if not ac["consistent"]:
            return {"solved": False, "assignment": None, "steps": ac["steps"]}
        steps = list(ac["steps"])
        result = self._backtrack({}, steps)
        return {"solved": result is not None, "assignment": result, "steps": steps}

    def _backtrack(self, assignment: dict, steps: list) -> Optional[dict]:
        if len(assignment) == len(self.variables):
            for c in self.constraints:
                if not c["check"](assignment):
                    return None
            return dict(assignment)
        unassigned = [(name, dom) for name, dom in self.variables.items()
                      if name not in assignment]
        if not unassigned:
            return None
        # MRV: pick variable with smallest remaining domain
        var_name, domain = min(unassigned, key=lambda x: len(x[1]))
        for value in list(domain):
            assignment[var_name] = value
            valid = True
            for c in self.constraints:
                if all(v in assignment for v in c["vars"]):
                    if not c["check"](assignment):
                        valid = False
                        break
            if valid:
                result = self._backtrack(assignment, steps)
                if result is not None:
                    return result
            del assignment[var_name]
        return None


# ═══════════════════════════════════════════════════════════════
# 5. Abductive Reasoner
# ═══════════════════════════════════════════════════════════════

class AbductiveReasoner:
    """Find best explanation for an observation by partial-rule matching."""

    def __init__(self, kg: KnowledgeGraph, engine: LogicEngine):
        self.kg = kg
        self.engine = engine

    def find_best_explanation(self, observation: Tuple[str, str, str],
                              max_hypotheses: int = 5) -> List[Dict]:
        obs_triple = Triple(subject=observation[0], predicate=observation[1],
                            object=observation[2])
        hypotheses = []
        for rule in self.engine.rules:
            match = self.engine._match_pattern(rule.conclusion, obs_triple, {})
            if match is None:
                continue
            satisfied = 0
            total = len(rule.conditions)
            missing = []
            for cond in rule.conditions:
                resolved = self.engine._resolve_pattern(cond, match)
                q_args = [None if (r and self.engine._is_var(r)) else r for r in resolved]
                found = self.kg.query(*q_args)
                if found:
                    satisfied += 1
                else:
                    missing.append({
                        "subject":   resolved[0],
                        "predicate": resolved[1],
                        "object":    resolved[2],
                    })
            score = satisfied / total if total > 0 else 0.0
            hypotheses.append({
                "rule":               rule.name,
                "plausibility":       round(score, 3),
                "evidence_found":     satisfied,
                "evidence_needed":    total,
                "missing_conditions": missing,
                "bindings":           match,
            })
        hypotheses.sort(key=lambda h: h["plausibility"], reverse=True)
        return hypotheses[:max_hypotheses]


# ═══════════════════════════════════════════════════════════════
# 6. Output encoding
# ═══════════════════════════════════════════════════════════════

class NexusOutput:
    """JSON encoders (nexus-reasoning-v1 schema)."""

    SCHEMA = "nexus-reasoning-v1"

    @classmethod
    def _base(cls) -> dict:
        return {
            "schema":    cls.SCHEMA,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    @classmethod
    def encode_inference(cls, result: dict) -> dict:
        out = cls._base()
        out["type"]          = "forward_inference"
        out["new_facts"]     = result.get("new_facts", [])
        out["iterations"]    = result.get("iterations", 0)
        out["total_triples"] = result.get("total_triples", 0)
        return out

    @classmethod
    def encode_proof(cls, result: dict) -> dict:
        out = cls._base()
        out["type"]       = "backward_proof"
        out["proven"]     = result.get("proven", False)
        out["confidence"] = round(result.get("confidence", 0.0), 4)
        out["proof_tree"] = result.get("proof", [])
        return out

    @classmethod
    def encode_abduction(cls, hypotheses: list) -> dict:
        out = cls._base()
        out["type"]       = "abduction"
        out["hypotheses"] = hypotheses
        return out

    @classmethod
    def encode_constraint(cls, result: dict) -> dict:
        out = cls._base()
        out["type"]         = "constraint_solution"
        out["solved"]       = result.get("solved", False)
        out["assignment"]   = result.get("assignment")
        out["search_steps"] = len(result.get("steps", []))
        return out

    @classmethod
    def encode_full(cls, **kwargs) -> dict:
        out = cls._base()
        out["type"] = "full_reasoning"
        out.update(kwargs)
        return out