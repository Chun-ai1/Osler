"""
Medical Knowledge Graph + GNN-style Learning
═══════════════════════════════════════════════════════════════
Pure NumPy implementation of TransE-style knowledge graph embedding.
Learns from existing manually-curated relationships and predicts
NEW relationships for diseases/symptoms not yet annotated.

How it works:
  1. Load knowledge graph (disease ↔ organ ↔ zone ↔ symptom)
  2. Embed every node into 64-dim vector
  3. Each edge type r has its own translation vector
  4. Train so that: embedding(head) + embedding(r) ≈ embedding(tail)
  5. To predict missing edges: compute h+r and find closest tail

Why this matters:
  • The 13 verified diseases have full anatomy/symptom data
  • Importing 1000 new diseases from Disease Ontology = no anatomy data
  • GNN can BOOTSTRAP missing data with confidence scores
  • Predictions marked as "gnn_inferred" — not used for clinical decisions
    until manually verified

Math (TransE):
  Loss: max(0, margin + ||h+r-t||² - ||h+r-t'||²)
  where t' is a corrupted (negative) tail
  
This is a ~50-line implementation. Real GNNs (R-GCN, CompGCN) work
better but require PyTorch. For this graph size, TransE is sufficient.
"""
from __future__ import annotations
import json
import os
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


class KnowledgeGraph:
    """Loads the medical knowledge graph from JSON and provides query interface."""
    
    DEFAULT_PATH = "medical_knowledge/graph/knowledge_graph.json"
    
    def __init__(self, path: Optional[str] = None):
        self.path = path or self.DEFAULT_PATH
        self.edges: Dict[str, Dict[str, List[str]]] = {}
        self.nodes: List[str] = []
        self.node_to_id: Dict[str, int] = {}
        self.edge_types: List[str] = []
        self.triples: List[Tuple[int, int, int]] = []  # (head, relation, tail)
        self._load()
    
    def _load(self):
        if not os.path.exists(self.path):
            print(f"[KG] Graph file not found: {self.path}")
            return
        
        data = json.load(open(self.path, encoding='utf-8'))
        self.edges = data.get('edges', {})
        self.edge_types = list(self.edges.keys())
        
        # Collect all nodes
        node_set = set()
        for et, edges in self.edges.items():
            for src, dsts in edges.items():
                node_set.add(src)
                node_set.update(dsts)
        
        self.nodes = sorted(node_set)
        self.node_to_id = {n: i for i, n in enumerate(self.nodes)}
        
        # Build triples
        for r_id, et in enumerate(self.edge_types):
            for src, dsts in self.edges[et].items():
                src_id = self.node_to_id[src]
                for dst in dsts:
                    self.triples.append((src_id, r_id, self.node_to_id[dst]))
        
        print(f"[KG] Loaded: {len(self.nodes)} nodes, "
              f"{len(self.edge_types)} edge types, "
              f"{len(self.triples)} triples")
    
    def neighbors(self, node: str, edge_type: Optional[str] = None) -> List[str]:
        """Find all neighbors of a node, optionally filtered by edge type."""
        if edge_type:
            return self.edges.get(edge_type, {}).get(node, [])
        result = set()
        for et, edges in self.edges.items():
            result.update(edges.get(node, []))
        return sorted(result)
    
    def has_edge(self, src: str, edge_type: str, dst: str) -> bool:
        return dst in self.edges.get(edge_type, {}).get(src, [])


class TransEEmbedder:
    """
    TransE-style knowledge graph embedding.
    Learns vector representations such that h + r ≈ t for valid triples.
    """
    
    def __init__(self, kg: KnowledgeGraph, dim: int = 64, margin: float = 1.0,
                 learning_rate: float = 0.01, seed: int = 42):
        self.kg = kg
        self.dim = dim
        self.margin = margin
        self.lr = learning_rate
        
        rng = np.random.RandomState(seed)
        n_nodes = len(kg.nodes)
        n_relations = len(kg.edge_types)
        
        # Initialize embeddings (Xavier-ish)
        scale = 6.0 / np.sqrt(dim)
        self.E = rng.uniform(-scale, scale, (n_nodes, dim))
        self.R = rng.uniform(-scale, scale, (n_relations, dim))
        
        # Normalize
        self.E /= np.linalg.norm(self.E, axis=1, keepdims=True) + 1e-9
        self.R /= np.linalg.norm(self.R, axis=1, keepdims=True) + 1e-9
        
        self.training_loss_history = []
    
    def _score(self, h_id, r_id, t_id):
        """L2 distance: lower = more plausible."""
        return np.linalg.norm(self.E[h_id] + self.R[r_id] - self.E[t_id])
    
    def _corrupt(self, triple, rng):
        """Replace head or tail randomly to create negative sample."""
        h, r, t = triple
        if rng.random() < 0.5:
            # corrupt head
            h_neg = rng.randint(len(self.kg.nodes))
            while self.kg.has_edge(self.kg.nodes[h_neg], 
                                    self.kg.edge_types[r],
                                    self.kg.nodes[t]):
                h_neg = rng.randint(len(self.kg.nodes))
            return (h_neg, r, t)
        else:
            t_neg = rng.randint(len(self.kg.nodes))
            while self.kg.has_edge(self.kg.nodes[h],
                                    self.kg.edge_types[r],
                                    self.kg.nodes[t_neg]):
                t_neg = rng.randint(len(self.kg.nodes))
            return (h, r, t_neg)
    
    def train(self, epochs: int = 100, verbose: bool = True):
        """Standard TransE training loop with margin loss."""
        rng = np.random.RandomState(0)
        triples = self.kg.triples
        
        for epoch in range(epochs):
            rng.shuffle(triples)
            total_loss = 0.0
            
            for triple in triples:
                neg = self._corrupt(triple, rng)
                
                pos_score = self._score(*triple)
                neg_score = self._score(*neg)
                
                loss = max(0, self.margin + pos_score - neg_score)
                if loss > 0:
                    total_loss += loss
                    
                    # Gradient update (simplified)
                    h, r, t = triple
                    h_n, _, t_n = neg
                    
                    diff_pos = self.E[h] + self.R[r] - self.E[t]
                    diff_neg = self.E[h_n] + self.R[r] - self.E[t_n]
                    
                    norm_pos = np.linalg.norm(diff_pos) + 1e-9
                    norm_neg = np.linalg.norm(diff_neg) + 1e-9
                    
                    grad_pos = diff_pos / norm_pos
                    grad_neg = diff_neg / norm_neg
                    
                    self.E[h] -= self.lr * grad_pos
                    self.R[r] -= self.lr * (grad_pos - grad_neg)
                    self.E[t] += self.lr * grad_pos
                    self.E[h_n] += self.lr * grad_neg
                    self.E[t_n] -= self.lr * grad_neg
                    
                    # Re-normalize
                    self.E[h] /= np.linalg.norm(self.E[h]) + 1e-9
                    self.E[t] /= np.linalg.norm(self.E[t]) + 1e-9
                    self.E[h_n] /= np.linalg.norm(self.E[h_n]) + 1e-9
                    self.E[t_n] /= np.linalg.norm(self.E[t_n]) + 1e-9
            
            avg_loss = total_loss / max(len(triples), 1)
            self.training_loss_history.append(avg_loss)
            
            if verbose and (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: avg loss = {avg_loss:.4f}")
        
        if verbose:
            print(f"  Final loss: {self.training_loss_history[-1]:.4f}")
    
    def predict_tail(self, head: str, edge_type: str, 
                     top_k: int = 10, only_type: Optional[str] = None) -> List[Tuple[str, float]]:
        """
        Given (head, edge_type, ?), predict the most likely tails.
        Returns ranked list of (tail_node, confidence) pairs.
        
        only_type: optional — restrict tails to nodes that appear as tails of this edge type
                  (e.g., only return organs when predicting disease_affects_organ)
        """
        if head not in self.kg.node_to_id:
            return []
        if edge_type not in self.kg.edge_types:
            return []
        
        h_id = self.kg.node_to_id[head]
        r_id = self.kg.edge_types.index(edge_type)
        
        target = self.E[h_id] + self.R[r_id]
        
        # Restrict candidate tails by type if specified
        candidate_ids = list(range(len(self.kg.nodes)))
        if only_type:
            valid_tails = set()
            for src, dsts in self.kg.edges.get(only_type, {}).items():
                valid_tails.update(dsts)
            candidate_ids = [self.kg.node_to_id[n] for n in valid_tails 
                             if n in self.kg.node_to_id]
        
        # Score each candidate
        scores = []
        for cand_id in candidate_ids:
            dist = np.linalg.norm(target - self.E[cand_id])
            # Convert distance to confidence (lower dist = higher conf)
            confidence = np.exp(-dist)
            scores.append((self.kg.nodes[cand_id], float(confidence)))
        
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]
    
    def find_similar_nodes(self, node: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Find nodes with similar embeddings — useful for finding analogous diseases."""
        if node not in self.kg.node_to_id:
            return []
        
        nid = self.kg.node_to_id[node]
        node_emb = self.E[nid]
        
        sims = self.E @ node_emb / (
            np.linalg.norm(self.E, axis=1) * np.linalg.norm(node_emb) + 1e-9)
        
        ranked = [(self.kg.nodes[i], float(sims[i])) 
                  for i in range(len(self.kg.nodes)) if i != nid]
        ranked.sort(key=lambda x: -x[1])
        return ranked[:top_k]
    
    def save(self, path: str):
        """Save trained embeddings."""
        np.savez(path,
                 E=self.E, R=self.R,
                 nodes=np.array(self.kg.nodes),
                 edge_types=np.array(self.kg.edge_types))
    
    def load(self, path: str):
        """Load pre-trained embeddings."""
        data = np.load(path, allow_pickle=True)
        self.E = data['E']
        self.R = data['R']


# ═══════════════════════════════════════════════════════════════
# Knowledge augmentation pipeline
# ═══════════════════════════════════════════════════════════════

def augment_disease_anatomy(kg: KnowledgeGraph, embedder: TransEEmbedder,
                             confidence_threshold: float = 0.5,
                             only_unknown: bool = True) -> dict:
    """
    For every disease in kg, predict its primary organs and zones.
    Returns predictions for diseases that DON'T already have anatomy data.
    """
    predictions = {}
    
    # Find all disease nodes (heads of disease_has_symptom or disease_affects_organ)
    diseases_with_anatomy = set(kg.edges.get('disease_affects_organ', {}).keys())
    diseases_with_symptoms = set(kg.edges.get('disease_has_symptom', {}).keys())
    all_diseases = diseases_with_anatomy | diseases_with_symptoms
    
    for disease in all_diseases:
        already_has = disease in diseases_with_anatomy
        if only_unknown and already_has:
            continue
        
        # Predict organs
        organ_preds = embedder.predict_tail(
            disease, 'disease_affects_organ', top_k=10,
            only_type='disease_affects_organ')
        
        # Predict zones
        zone_preds = embedder.predict_tail(
            disease, 'disease_in_zone', top_k=5,
            only_type='disease_in_zone')
        
        # Filter by confidence
        organs = [o for o, c in organ_preds if c >= confidence_threshold]
        zones  = [z for z, c in zone_preds if c >= confidence_threshold]
        
        if organs or zones:
            predictions[disease] = {
                'predicted_organs':     organs[:5],
                'predicted_zones':      zones[:3],
                'top_organ_confidence': organ_preds[0][1] if organ_preds else 0,
                'top_zone_confidence':  zone_preds[0][1] if zone_preds else 0,
                'verification_status':  'gnn_inferred',
                'already_known':        already_has,
            }
    
    return predictions


if __name__ == "__main__":
    print("=" * 70)
    print("NEXUS Knowledge Graph + GNN Learning")
    print("=" * 70)
    
    # Step 1: Load graph
    kg = KnowledgeGraph()
    
    # Step 2: Train embedder
    print(f"\nTraining TransE embedder (64-dim, 100 epochs)...")
    embedder = TransEEmbedder(kg, dim=64)
    embedder.train(epochs=100, verbose=True)
    
    # Step 3: Test predictions on KNOWN data (validation)
    print(f"\n=== Test 1: Predict organs for KNOWN disease 'appendicitis' ===")
    preds = embedder.predict_tail('appendicitis', 'disease_affects_organ', 
                                    top_k=8, only_type='disease_affects_organ')
    for organ, conf in preds:
        actual = kg.has_edge('appendicitis', 'disease_affects_organ', organ)
        marker = "✓" if actual else "?"
        print(f"  {marker} {organ:25s} confidence={conf:.3f}")
    
    print(f"\n=== Test 2: Find diseases similar to 'appendicitis' ===")
    sims = embedder.find_similar_nodes('appendicitis', top_k=8)
    for node, sim in sims:
        # Filter to only show disease-like nodes
        is_disease = node in kg.edges.get('disease_affects_organ', {})
        if is_disease:
            print(f"  {node:30s} similarity={sim:.3f}")
    
    print(f"\n=== Test 3: Predict zone for 'appendicitis' ===")
    zone_preds = embedder.predict_tail('appendicitis', 'disease_in_zone',
                                         top_k=5, only_type='disease_in_zone')
    for zone, conf in zone_preds:
        actual = kg.has_edge('appendicitis', 'disease_in_zone', zone)
        marker = "✓" if actual else "?"
        print(f"  {marker} {zone:20s} confidence={conf:.3f}")