"""
NEXUS Integration Guide (v7)
═══════════════════════════════════════════════════

NEXUS is a pure symbolic reasoning engine.
Zero LLM dependency. Zero RAG/FAISS. Zero embeddings.

Architecture:
  symptoms → NEXUS 16-step reasoning → diagnoses + evidence chain

Modules:
  nexus_medical.py          — Core 12-step reasoning engine
  symptom_expander.py       — System inference, danger bias, predicted symptoms
  combination_reasoner.py   — Symptom pair analysis, red flag combos
  disease_ranker_v2.py      — Multi-factor scoring (sym + mech + coverage + urgency)
  etiology_classifier.py    — Virus vs bacteria vs non-infectious (4-layer)
  pathogen_tracker.py       — Vascular spread prediction (BFS on anatomy)
  physiology_engine.py      — Sepsis cascade, cancer metastasis, organ failure, chemistry
  anatomy_atlas.py          — 127 organs, 300 connections, full vascular tree
  anatomy_bridge.py         — Organ identification from KG queries
  anatomy_knowledge_loader.py — Loads disease/symptom → organ triples into KG
  nexus_core.py             — KnowledgeGraph class (used by anatomy)
  nexus_trace.py            — Full reasoning trace with terminal output
  nexus_learning_bridge.py  — Auto-learning feedback loop
  nexus_routes.py           — Flask API endpoints
  deterministic_response.py — Patient-facing response (no LLM)

Usage in app.py:
    from nexus_engine.nexus_routes import nexus_bp, init_nexus
    nexus_instance = init_nexus()
    app.register_blueprint(nexus_bp)

    # In chat_stream:
    result = nexus_instance.enhance_pipeline_result(result, user_input=user_input)

    # Deterministic response (no LLM):
    from deterministic_response import generate_response
    answer = generate_response(result, user_input)

    # Etiology classification:
    from etiology_classifier import EtiologyClassifier
    ec = EtiologyClassifier()
    etiology = ec.classify(symptoms, labs=patient_labs)

Knowledge sources (33,000+ connections):
    - 13 disease profiles (symptoms, red_flags, complications)
    - 21 symptom profiles (systems, causes, related)
    - 3,647 mechanisms (effects, symptoms, diseases)
    - 1,200 bacteria mechanisms
    - 1,500 virus mechanisms
    - symptom_graph (841 co-occurrence edges)
    - 127 organs, 300 vascular connections
"""