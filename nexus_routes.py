"""
NEXUS Flask API  + 
═══════════════════════════════════════════════════════════════
API:
    POST /nexus/reason       — 
    POST /nexus/prove        — 
    POST /nexus/explain      — 
    GET  /nexus/stats        —  + 
    GET  /nexus/query        — 
    POST /nexus/learn        — 
"""

from flask import Blueprint, request, jsonify, render_template
import os
from nexus_engine.nexus_medical import NexusMedical
from nexus_engine.nexus_learning_bridge import NexusLearner

nexus_bp = Blueprint(
    "nexus", __name__, url_prefix="/nexus",
    template_folder=os.path.join(os.path.dirname(__file__), "templates"),
)

_nexus: NexusMedical = None
_learner: NexusLearner = None


def init_nexus(
    symptom_dir="medical_knowledge/symptoms",
    mechanism_path="medical_knowledge/mechanisms/mechanisms_rag_final.json",
    disease_dir=None,
    auto_learn=True,
):
    """
    app  +  auto_learning 

    :
        from nexus_engine.nexus_routes import nexus_bp, init_nexus
        nexus = init_nexus()
        app.register_blueprint(nexus_bp)
    """
    global _nexus, _learner
    _nexus = NexusMedical()

    print("[NEXUS] ...")
    _nexus.load_knowledge()

    _learner = NexusLearner(_nexus) if NexusLearner else None
    if auto_learn and _learner:
        try:
            _learner.learn_from_all()
        except Exception as e:
            print(f"[NEXUS] auto_learn failed (non-blocking): {e}")

    print(f"[NEXUS] : {len(_nexus.diseases)} diseases, "
          f"{len(_nexus.symptom_info)} symptoms, {len(_nexus.mech_info)} mechanisms")
    return _nexus


def get_nexus() -> NexusMedical:
    global _nexus
    if _nexus is None:
        _nexus = NexusMedical()
        _nexus.load_knowledge()
    return _nexus


def get_learner() -> NexusLearner:
    global _learner
    if _learner is None:
        _learner = NexusLearner(get_nexus())
    return _learner


# ═══════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════

@nexus_bp.post("/reason")
def nexus_reason():
    """"""
    data = request.get_json(silent=True) or {}
    symptoms = data.get("symptoms", [])
    if not symptoms:
        return jsonify({"error": " symptoms "}), 400

    nexus = get_nexus()
    result = nexus.enhance_pipeline_result(
        {"symptoms": symptoms, "top_diseases": [], "reasoning": ""}
    )

    #  auto_learning
    try:
        get_learner().feedback(result)
    except Exception:
        pass

    return jsonify({
        "status": "ok",
        "diagnoses": result.get("nexus_diagnoses", []),
        "mechanism_evidence": result.get("nexus_mechanism_evidence", {}),
        "red_flags": result.get("nexus_red_flags", []),
        "suggested_questions": result.get("nexus_suggested_questions", []),
        "root_causes": result.get("nexus_root_causes", []),
        "abduction": result.get("nexus_abduction", {}),
        "consistency": result.get("nexus_consistency", {}),
        "stats": result.get("nexus_stats", {}),
    })


@nexus_bp.post("/prove")
def nexus_prove():
    """NEXUS """
    data = request.get_json(silent=True) or {}
    symptoms = data.get("symptoms", [])
    if not symptoms:
        return jsonify({"error": " symptoms"}), 400

    nexus = get_nexus()
    result = nexus.reason(symptoms)
    return jsonify(result)


@nexus_bp.post("/explain")
def nexus_explain():
    """NEXUS """
    data = request.get_json(silent=True) or {}
    symptoms = data.get("symptoms", [])
    nexus = get_nexus()
    result = nexus.reason(symptoms)
    return jsonify({"thinking": result["thinking"], "diagnoses": result["diagnoses"]})


@nexus_bp.get("/stats")
def nexus_stats():
    """NEXUS """
    nexus = get_nexus()
    return jsonify({
        "diseases": len(nexus.diseases),
        "symptom_profiles": len(nexus.symptom_info),
        "mechanisms": len(nexus.mech_info),
    })


@nexus_bp.get("/query")
def nexus_query():
    """"""
    s = request.args.get("subject")
    p = request.args.get("predicate")
    o = request.args.get("object")
    nexus = get_nexus()
    # New NEXUS doesn't have a query API - use reason instead
    symptoms = [x for x in [s, p, o] if x]
    if symptoms:
        result = nexus.reason(symptoms)
        return jsonify({"count": len(result["diagnoses"]), "diagnoses": result["diagnoses"][:10]})
    return jsonify({"count": 0, "diagnoses": []})


@nexus_bp.post("/learn")
def nexus_learn():
    """"""
    nexus = get_nexus()
    nexus._loaded = False
    nexus.load_knowledge()
    return jsonify({"status": "ok", "diseases": len(nexus.diseases)})


@nexus_bp.get("/")
@nexus_bp.get("/view")
def nexus_page():
    """NEXUS """
    return render_template("nexus.html")


# ═══════════════════════════════════════════════════════════════
#  API
# ═══════════════════════════════════════════════════════════════

_anatomy = None

def get_anatomy():
    global _anatomy
    if _anatomy is None:
        from nexus_engine.anatomy_bridge import AnatomyBridge
        _anatomy = AnatomyBridge(get_nexus())
    return _anatomy


@nexus_bp.get("/anatomy")
def anatomy_page():
    """3D """
    return render_template("anatomy3d.html")


@nexus_bp.get("/anatomy/data")
def anatomy_data():
    """ ( 3D )"""
    return jsonify(get_anatomy().get_atlas_json())


@nexus_bp.post("/anatomy/spread")
def anatomy_spread():
    """"""
    data = request.get_json(silent=True) or {}
    organ = data.get("organ", "")
    spread_type = data.get("type", "infection")
    max_hops = int(data.get("max_hops", 5))
    if not organ:
        return jsonify({"error": " organ "}), 400
    results = get_anatomy().query_spread(organ, spread_type, max_hops)
    return jsonify({"origin": organ, "type": spread_type, "paths": results})


@nexus_bp.post("/anatomy/path")
def anatomy_path():
    """"""
    data = request.get_json(silent=True) or {}
    origin = data.get("origin", "")
    target = data.get("target", "")
    spread_type = data.get("type", "infection")
    if not origin or not target:
        return jsonify({"error": " origin  target"}), 400
    result = get_anatomy().query_path(origin, target, spread_type)
    return jsonify({"path": result})


@nexus_bp.post("/anatomy/referred_pain")
def anatomy_referred():
    """"""
    data = request.get_json(silent=True) or {}
    location = data.get("location", "")
    if not location:
        return jsonify({"error": " location "}), 400
    results = get_anatomy().query_referred_pain(location)
    return jsonify({"location": location, "sources": results})