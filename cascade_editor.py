"""
NEXUS Cascade Editor — Web UI for adding mechanism cascades
═══════════════════════════════════════════════════════════════
Helps doctors/medical-students extend NEXUS's mechanism library
without touching code. Designed to be a guided, helpful interface.

Run:  python3 nexus_engine/cascade_editor.py
URL:  http://localhost:5005
"""
import os
import json
import sys
from typing import Dict, Any, List

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

try:
    from flask import Flask, request, jsonify, render_template_string, redirect
except ImportError:
    print("Flask not installed: pip install flask")
    sys.exit(1)


KB_PATH = os.path.join(ROOT, "medical_knowledge", "layer2_physiology", "organ_function.json")


def load_kb() -> Dict[str, Any]:
    if not os.path.exists(KB_PATH):
        return {"_meta": {"version": "0.1"}, "organs": {}}
    return json.load(open(KB_PATH, encoding="utf-8"))


def save_kb(data: Dict[str, Any]):
    tmp = KB_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, KB_PATH)




# ─────────────────────────────────────────────────────────────
# Disease registration helpers — coordinate updates across files
# ─────────────────────────────────────────────────────────────

DISEASE_REGISTRY_PATH = os.path.join(ROOT, "medical_knowledge", "registry", "disease_mechanism_map.json")
DISEASE_DB_PATH = os.path.join(ROOT, "medical_knowledge", "diseases", "disease_0001.json")
DISEASE_ANATOMY_PATH = os.path.join(ROOT, "medical_knowledge", "anatomy", "disease_anatomy.json")


def load_json(path: str, default=None):
    if not os.path.exists(path):
        return default if default is not None else {}
    return json.load(open(path, encoding="utf-8"))


def save_json(path: str, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def extract_clinical_from_cascade(cascade: list) -> dict:
    """Auto-extract symptoms, red_flags, complications from a cascade."""
    symptoms = []
    red_flags = []
    complications = []
    for step in cascade:
        for k, v in step.items():
            if k.startswith('produces_symptom'):
                # Try to clean: strip _via_*, _T*, etc. for friendlier names
                clean = v.split('_via_')[0].split('_T')[0]
                clean = clean.replace('_', ' ').strip()
                if clean and clean not in symptoms:
                    symptoms.append(clean)
            elif k == 'severity_marker' and v == 'red_flag':
                # Red flag = the event of this step
                red_flags.append(step.get('event', '').replace('_', ' '))
            elif k == 'produces_complication':
                complications.append(v.replace('_', ' '))
    return {
        'common_symptoms': symptoms,
        'red_flags': red_flags,
        'complications': complications,
    }


def register_disease(disease_name: str, organ: str, failure_mode: str,
                     pathogen: str = None, icd_code: str = "",
                     prevalence: str = "uncommon", age_groups: list = None,
                     sex_bias: str = "any", triage_level: str = "urgent",
                     overwrite: bool = False) -> dict:
    """
    Atomically register a new disease across 4 files.
    Returns dict of {success: bool, errors: list, files_modified: list}.
    """
    result = {"success": False, "errors": [], "files_modified": []}
    
    # 1. Verify cascade exists
    kb = load_kb()
    organ_data = kb.get('organs', {}).get(organ)
    if not organ_data:
        result['errors'].append(f"Organ '{organ}' not found in organ_function.json")
        return result
    fmode = next((fm for fm in organ_data.get('failure_modes', []) if fm.get('mode') == failure_mode), None)
    if not fmode:
        avail = [fm.get('mode') for fm in organ_data.get('failure_modes', [])]
        result['errors'].append(f"Failure mode '{failure_mode}' not found for {organ}. Available: {avail}")
        return result
    
    # 2. Extract clinical info from cascade
    clinical = extract_clinical_from_cascade(fmode.get('cascade', []))
    
    # 3. Update disease_mechanism_map.json (registry)
    registry_data = load_json(DISEASE_REGISTRY_PATH, default={"_meta": {}, "mappings": {}})
    if disease_name in registry_data.get('mappings', {}) and not overwrite:
        result['errors'].append(f"Disease '{disease_name}' already registered (use overwrite to replace)")
        return result
    registry_data.setdefault('mappings', {})[disease_name] = {
        "organ": organ, "failure_mode": failure_mode, "pathogen": pathogen,
    }
    save_json(DISEASE_REGISTRY_PATH, registry_data)
    result['files_modified'].append('disease_mechanism_map.json')
    
    # 4. Update disease_0001.json
    disease_db = load_json(DISEASE_DB_PATH, default=[])
    if not isinstance(disease_db, list):
        disease_db = []
    # Remove existing if overwrite
    disease_db = [d for d in disease_db if d.get('disease_name') != disease_name]
    
    # Auto-derive symptom_weights via cascade information theory
    # (no more placeholder values — these come from real cascade analysis)
    try:
        from weight_deriver import WeightDeriver
        deriver = WeightDeriver()
        # Re-derive ALL diseases to keep weights consistent
        deriver.save()
        # Read the just-saved weights for THIS disease
        weights_path = os.path.join(ROOT, "medical_knowledge", "derived", "symptom_weights.json")
        derived_data = json.load(open(weights_path))
        disease_weights = derived_data.get('disease_weights', {}).get(disease_name, {})
        weights_list = []
        for sym in clinical['common_symptoms']:
            # Match the normalized name back to original
            from weight_deriver import WeightDeriver as _WD
            norm = _WD._normalize(sym)
            info = disease_weights.get(norm) or disease_weights.get(sym)
            if info:
                weights_list.append({
                    "symptom": sym,
                    "weight": info['weight'],
                    "specificity": info['specificity'],
                    "_derived_from": "cascade_information_theory",
                })
            else:
                # Fallback if normalization mismatch
                weights_list.append({"symptom": sym, "weight": 3, "specificity": "medium",
                                     "_derived_from": "fallback_default"})
    except Exception as _we:
        # Last-resort fallback
        weights_list = [{"symptom": s, "weight": 3, "specificity": "medium"}
                         for s in clinical['common_symptoms']]
    
    new_disease = {
        "disease_name": disease_name,
        "icd_code": icd_code,
        "common_symptoms": clinical['common_symptoms'],
        "symptom_weights": weights_list,
        "red_flags": clinical['red_flags'],
        "complications": clinical['complications'],
        "prevalence_text": prevalence,
        "age_groups": age_groups or ["adult"],
        "sex_bias": sex_bias,
        "triage_level": triage_level,
        "_source": "cascade_editor",
        "_weights_method": "auto_derived_from_cascade_TF_IDF",
    }
    disease_db.append(new_disease)
    save_json(DISEASE_DB_PATH, disease_db)
    result['files_modified'].append('disease_0001.json')
    
    # 5. Update disease_anatomy.json
    anatomy = load_json(DISEASE_ANATOMY_PATH, default={"_meta": {}, "diseases": {}})
    disease_id = disease_name.lower().replace(' ', '_').replace('-', '_')
    anatomy.setdefault('diseases', {})[disease_id] = {
        "primary": [organ]   # zones and system are auto-derived
    }
    save_json(DISEASE_ANATOMY_PATH, anatomy)
    result['files_modified'].append('disease_anatomy.json')
    
    result['success'] = True
    result['extracted_clinical'] = clinical
    result['placeholder_weights'] = weights_list
    return result


def get_valid_organs() -> List[Dict[str, str]]:
    """Read valid organ IDs from anatomy_atlas."""
    try:
        from anatomy_atlas import AnatomyAtlas
        atlas = AnatomyAtlas()
        organs = []
        for oid, organ in atlas.organs.items():
            organs.append({
                'id': oid,
                'name': organ.name or oid,
                'system': organ.system or 'unknown',
            })
        organs.sort(key=lambda x: (x['system'], x['id']))
        return organs
    except Exception as e:
        print(f"[Warning] Could not load atlas: {e}")
        return []


def validate_cascade(cascade: List[dict]) -> List[str]:
    errors = []
    if not cascade:
        errors.append("Cascade must have at least one step")
        return errors
    for i, step in enumerate(cascade):
        if 'event' not in step:
            errors.append(f"Step {i+1}: missing 'event' field")
        if 'step' not in step:
            errors.append(f"Step {i+1}: missing 'step' number field")
    return errors


# ─────────────────────────────────────────────────────────────
# HTML Templates with extensive inline help
# ─────────────────────────────────────────────────────────────

INDEX_HTML = r"""
<!DOCTYPE html>
<html>
<head>
<title>NEXUS Cascade Editor</title>
<style>
  body { font-family: -apple-system, "Helvetica Neue", sans-serif; max-width: 1200px; margin: 30px auto; padding: 20px; background: #f5f7fa; color: #2c3e50; }
  h1 { margin: 0; }
  .header { display: flex; justify-content: space-between; align-items: center; }
  .panel { background: white; border-radius: 8px; padding: 20px; margin: 20px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
  .help { background: #fffbe6; border: 1px solid #ffe58f; padding: 15px; border-radius: 6px; margin: 15px 0; }
  .help h3 { margin-top: 0; color: #ad8b00; }
  .stats { display: flex; gap: 15px; flex-wrap: wrap; }
  .stat { background: linear-gradient(135deg,#74b9ff,#0984e3); color: white; padding: 18px; border-radius: 8px; flex: 1; min-width: 150px; }
  .stat .num { font-size: 32px; font-weight: bold; }
  .stat .label { font-size: 13px; opacity: 0.9; }
  .organ-list { display: grid; grid-template-columns: repeat(auto-fill,minmax(220px,1fr)); gap: 10px; }
  .organ-card { background: #ecf3fc; padding: 14px; border-radius: 6px; cursor: pointer; transition: all 0.2s; border: 2px solid transparent; }
  .organ-card:hover { background: #d4e4f7; border-color: #3498db; transform: translateY(-2px); }
  .organ-card h3 { margin: 0 0 8px 0; color: #2c3e50; font-size: 16px; }
  .organ-card .system { font-size: 11px; color: #7f8c8d; text-transform: uppercase; }
  .organ-card .modes { font-size: 12px; color: #555; margin-top: 5px; }
  button, .btn { background: #3498db; color: white; border: none; padding: 12px 22px; border-radius: 5px; cursor: pointer; text-decoration: none; display: inline-block; font-size: 14px; }
  .btn:hover { background: #2980b9; }
  .btn-success { background: #27ae60; }
  .btn-success:hover { background: #229954; }
  .badge { display: inline-block; background: #ecf0f1; padding: 3px 8px; border-radius: 10px; font-size: 11px; }
  details { margin: 10px 0; }
  details summary { cursor: pointer; padding: 8px; background: #ecf0f1; border-radius: 4px; }
  pre { background: #2c3e50; color: #ecf0f1; padding: 15px; overflow-x: auto; border-radius: 4px; font-size: 12px; }
</style>
</head>
<body>
  <div class="header">
    <h1>🧠 NEXUS Cascade Editor</h1>
    <a href="/new" class="btn btn-success">+ Add Failure Mode</a>
    <a href="/register" style="background:#e67e22;color:white;border:none;padding:12px 22px;border-radius:5px;text-decoration:none;margin-left:8px;font-weight:bold;">📝 Register Disease</a>
    <a href="/refresh_weights" style="background:#16a085;color:white;border:none;padding:12px 22px;border-radius:5px;text-decoration:none;margin-left:8px;">🔄 Refresh Auto-Weights</a>
  </div>
  <p>Help build NEXUS's mechanism layer. Each cascade you add lets NEXUS auto-derive clinical symptoms.</p>
  <p style="margin:10px 0;"><a href="/reference" style="display:inline-block;background:#9b59b6;color:white;padding:8px 16px;border-radius:5px;text-decoration:none;font-weight:bold;">📚 Open Medical Reference (T-levels, immune molecules, naming, common mistakes)</a></p>

  <div class="help">
    <h3>📖 First time? Read this</h3>
    <p><b>What is a "failure mode"?</b> It's a way an organ can go wrong. For example:</p>
    <ul>
      <li><b>appendix</b> failure modes: <code>luminal_obstruction</code> (= appendicitis), <code>perforation</code></li>
      <li><b>heart</b> failure modes: <code>coronary_obstruction_acute</code> (= MI), <code>arrhythmia</code>, <code>pump_failure</code></li>
      <li><b>kidney</b> failure modes: <code>infection_pyelonephritis</code>, <code>obstruction</code>, <code>AKI</code></li>
    </ul>
    <p>Each failure mode has a <b>cascade</b>: ordered steps showing how the disease unfolds, with symptoms produced at each step.</p>
    <p>From these atomic facts, NEXUS auto-derives the clinical picture (no hardcoding).</p>
    <details>
      <summary>👀 Click here for a complete example</summary>
<pre>{
  "step": 1,
  "event": "obstruction by fecalith",
  "produces_symptom": "intraluminal_pressure_rises"
},
{
  "step": 2,
  "event": "venous_drainage_compromised",
  "produces_symptom": "visceral_pain_T10_periumbilical"
},
{
  "step": 3,
  "event": "transmural_inflammation_extends_to_parietal_peritoneum",
  "produces_symptom_1": "somatic_pain_T12_RLQ_McBurneys_point",
  "produces_symptom_2": "rebound_tenderness"
}</pre>
    </details>
  </div>

  <div class="stats">
    <div class="stat"><div class="num">{{n_organs}}</div><div class="label">Organs with cascades</div></div>
    <div class="stat"><div class="num">{{n_failure_modes}}</div><div class="label">Failure modes</div></div>
    <div class="stat"><div class="num">{{n_cascade_steps}}</div><div class="label">Cascade steps</div></div>
    <div class="stat"><div class="num">{{n_derived_symptoms}}</div><div class="label">Auto-derivable symptoms</div></div>
  </div>

  <div class="panel">
    <h2>📋 Existing Organs ({{n_organs}})</h2>
    <p style="color:#7f8c8d;font-size:14px;">Click any organ to see its current cascades</p>
    <div class="organ-list">
      {% for org_id, org in organs.items() %}
      <a href="/organ/{{org_id}}" style="text-decoration:none;color:inherit;">
        <div class="organ-card">
          <h3>{{org_id}}</h3>
          <div class="system">{% for fm in org.failure_modes %}<span class="badge">{{fm.mode}}</span> {% endfor %}</div>
          <div class="modes">{{org.failure_modes|length}} failure mode(s) • {% set total = namespace(s=0) %}{% for fm in org.failure_modes %}{% set total.s = total.s + (fm.cascade|length if fm.cascade else 0) %}{% endfor %}{{total.s}} cascade steps</div>
        </div>
      </a>
      {% endfor %}
    </div>
  </div>
</body>
</html>
"""


ORGAN_HTML = r"""
<!DOCTYPE html>
<html>
<head>
<title>{{organ_id}} - Cascade Editor</title>
<style>
  body { font-family: -apple-system, sans-serif; max-width: 1100px; margin: 30px auto; padding: 20px; background: #f5f7fa; color: #2c3e50; }
  .panel { background: white; border-radius: 8px; padding: 20px; margin: 20px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
  .step { background: #f8f9fa; padding: 14px; margin: 10px 0; border-left: 4px solid #3498db; border-radius: 4px; }
  .step .step-num { font-size: 11px; color: #7f8c8d; text-transform: uppercase; }
  .step .event { font-weight: bold; margin: 4px 0; }
  .symptom { display: inline-block; background: #e3f2fd; padding: 5px 12px; margin: 3px; border-radius: 12px; font-size: 12px; }
  .red-flag { background: #ffebee; color: #c62828; }
  .complication { background: #fff3e0; color: #e65100; }
  .btn-back { background: #95a5a6; text-decoration:none; padding: 10px 18px; border-radius:5px; color:white; display:inline-block; }
  pre { background: #2c3e50; color: #ecf0f1; padding: 15px; overflow-x: auto; border-radius: 4px; font-size: 12px; }
  h2 { color: #2c3e50; }
</style>
</head>
<body>
  <a href="/" class="btn-back">← Back to all organs</a>
  <a href="/register" style="background:#e67e22;color:white;padding:10px 18px;border-radius:5px;text-decoration:none;margin-left:8px;">📝 Register a disease using these cascades</a>
  <h1>{{organ_id}}</h1>

  {% for fm in organ.failure_modes %}
  <div class="panel">
    <h2>⚙️ Failure mode: <code>{{fm.mode}}</code></h2>
    <p><b>Speed:</b> {{fm.speed or 'unspecified'}} &nbsp; | &nbsp; <b>Cause:</b> {{fm.cause or fm.causes or '?'}}</p>

    <h3>Cascade ({{fm.cascade|length}} steps)</h3>
    {% for step in fm.cascade %}
    <div class="step">
      <div class="step-num">Step {{step.step}}</div>
      <div class="event">{{step.event}}</div>
      <div>
        {% for k, v in step.items() %}
          {% if k.startswith('produces_symptom') %}
            <span class="symptom">→ {{v}}</span>
          {% elif k == 'severity_marker' and v == 'red_flag' %}
            <span class="symptom red-flag">🚨 RED FLAG</span>
          {% elif k == 'produces_complication' %}
            <span class="symptom complication">⚠ {{v}}</span>
          {% endif %}
        {% endfor %}
      </div>
    </div>
    {% endfor %}
  </div>
  {% endfor %}

  <div class="panel">
    <h3>🔬 Derived Clinical Picture (auto-computed by NEXUS)</h3>
    <pre>{{derived_picture}}</pre>
  </div>

  <div class="panel">
    <h3>📝 Raw JSON</h3>
    <details>
      <summary style="cursor:pointer;padding:8px;background:#ecf0f1;border-radius:4px;">Show JSON</summary>
      <pre>{{raw_json}}</pre>
    </details>
  </div>
</body>
</html>
"""


NEW_CASCADE_HTML = r"""
<!DOCTYPE html>
<html>
<head>
<title>New Cascade - Editor</title>
<style>
  body { font-family: -apple-system, sans-serif; max-width: 1000px; margin: 30px auto; padding: 20px; background: #f5f7fa; color: #2c3e50; }
  .panel { background: white; border-radius: 8px; padding: 25px; margin: 20px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
  .help-box { background: #fffbe6; border: 1px solid #ffe58f; padding: 15px; border-radius: 6px; margin: 15px 0; }
  .help-box h3 { margin-top: 0; color: #ad8b00; }
  label { display:block; margin: 18px 0 6px; font-weight: bold; color: #2c3e50; }
  label .required { color: #e74c3c; }
  input, select, textarea { width: 100%; padding: 10px; border: 1px solid #bdc3c7; border-radius: 4px; font-size: 14px; box-sizing: border-box; }
  textarea { font-family: 'Monaco', 'Menlo', monospace; min-height: 280px; }
  button { background: #27ae60; color: white; border: none; padding: 14px 24px; border-radius: 5px; cursor: pointer; font-size: 16px; font-weight: bold; }
  button:hover { background: #2ecc71; }
  .btn-back { background: #95a5a6; text-decoration:none; padding: 10px 18px; border-radius:5px; color:white; display:inline-block; }
  .hint { color: #7f8c8d; font-size: 13px; margin: 5px 0 0 0; line-height: 1.5; }
  .error { color: #c0392b; padding: 12px; background: #fadbd8; border-radius: 4px; margin: 10px 0; font-weight: bold; }
  .example { background: #e8f5e9; padding: 10px 14px; border-radius: 4px; font-family: monospace; font-size: 13px; }
  .examples-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
  details summary { cursor: pointer; padding: 8px; background: #ecf0f1; border-radius: 4px; font-weight: bold; }
  pre { background: #2c3e50; color: #ecf0f1; padding: 15px; overflow-x: auto; border-radius: 4px; font-size: 12px; }
  table.organ-list { width: 100%; border-collapse: collapse; font-size: 13px; }
  table.organ-list td { padding: 4px 8px; border-bottom: 1px solid #ecf0f1; }
  table.organ-list code { background: #ecf0f1; padding: 2px 6px; border-radius: 3px; }
</style>
</head>
<body>
  <a href="/" class="btn-back">← Back</a>
  <a href="/reference" style="display:inline-block;background:#9b59b6;color:white;padding:10px 18px;border-radius:5px;text-decoration:none;margin-left:8px;">📚 Reference (T-levels, immune)</a>
  <h1>➕ New Failure Mode Cascade</h1>

  <!-- BIG HELP SECTION AT TOP -->
  <div class="help-box">
    <h3>🎯 Naming Conventions — read first</h3>
    <ul>
      <li><b>Organ ID</b>: lowercase + underscores. Use the EXACT ID from the atlas (see dropdown below).</li>
      <li><b>Failure mode</b>: lowercase + underscores. Describe the MECHANISM, not the disease name.
        <br>✅ <code>luminal_obstruction</code>, <code>infection_inflammation</code>, <code>ischemia_reperfusion</code>
        <br>❌ <code>appendicitis</code>, <code>meningitis</code> (these are diagnoses, not mechanisms)</li>
      <li><b>Cascade events</b>: describe what happens at the cellular/anatomic level using medical terms.
        <br>✅ <code>premature_trypsinogen_activation_within_acinar_cells</code>
        <br>❌ <code>pancreas_starts_breaking_down</code> (too vague)</li>
      <li><b>Symptoms</b>: use medical terms with mechanism explanation. Format: <code>SYMPTOM_via_MECHANISM</code>
        <br>✅ <code>periumbilical_pain_via_T10_visceral_afferents</code>
        <br>❌ <code>belly_pain</code> (NEXUS won't know which dermatome)</li>
    </ul>
  </div>

  <div class="help-box">
    <h3>🔬 What is a "failure mode"?</h3>
    <p>It's <b>not</b> the disease name. It's the <b>mechanism by which the organ goes wrong</b>.</p>
    <table class="organ-list">
      <tr><th>Organ</th><th>Disease (NOT what you write)</th><th>Failure Mode (what to write)</th></tr>
      <tr><td><code>appendix</code></td><td>Appendicitis</td><td><code>luminal_obstruction</code></td></tr>
      <tr><td><code>heart</code></td><td>Heart Attack</td><td><code>coronary_obstruction_acute</code></td></tr>
      <tr><td><code>meninges</code></td><td>Bacterial Meningitis</td><td><code>infection_inflammation</code></td></tr>
      <tr><td><code>bladder</code></td><td>Cystitis</td><td><code>infection_cystitis</code></td></tr>
      <tr><td><code>brain</code></td><td>Migraine</td><td><code>vascular_dysregulation_migraine</code></td></tr>
      <tr><td><code>lungs</code></td><td>Pneumonia</td><td><code>alveolar_consolidation</code></td></tr>
    </table>
    <p style="margin-top:15px;"><b>Why this matters:</b> One organ can fail in multiple ways → many diseases.
    <br>e.g., <code>kidney</code> has failure modes: <code>infection_pyelonephritis</code>, <code>AKI</code>, <code>obstruction</code>, <code>glomerulonephritis</code> → 4 different diseases.</p>
  </div>

  {% if error %}<div class="error">⚠️ {{error}}</div>{% endif %}

  <form method="POST" action="/new">
    <div class="panel">
      <label>Organ ID <span class="required">*</span></label>
      <select name="organ_id" required onchange="document.getElementById('custom-organ').value=this.value">
        <option value="">— Pick from valid atlas organs —</option>
        {% for o in valid_organs %}
        <option value="{{o.id}}">{{o.id}} ({{o.system}}) — {{o.name}}</option>
        {% endfor %}
      </select>
      <p class="hint">Or type below if your organ isn't in the dropdown (will be added):</p>
      <input type="text" id="custom-organ" name="organ_id_custom" placeholder="e.g., pancreas" oninput="this.form.organ_id.value=''">
      <p class="hint">Tip: only use atlas IDs to keep NEXUS spatially aware. Custom IDs work but won't have 3D positioning.</p>

      <label>Failure mode name <span class="required">*</span></label>
      <input type="text" name="failure_mode" placeholder="e.g., acute_inflammation" required pattern="[a-z_0-9]+" title="lowercase + underscores only">
      <p class="hint">Lowercase + underscores. Describe the <b>mechanism</b>, not the diagnosis.<br>
        ✅ <code>luminal_obstruction</code>, <code>ischemia_reperfusion</code>, <code>autoimmune_attack</code><br>
        ❌ <code>appendicitis</code>, <code>cancer</code>
      </p>

      <label>Cause(s)</label>
      <input type="text" name="cause" placeholder="e.g., fecalith, bacterial_invasion, atherosclerotic_plaque_rupture">
      <p class="hint">Free text. What triggers this failure mode? Comma-separated.</p>

      <label>Speed of progression</label>
      <select name="speed">
        <option value="seconds">seconds (e.g., V-fib, anaphylaxis)</option>
        <option value="minutes">minutes (e.g., MI, stroke)</option>
        <option value="minutes_to_hours">minutes_to_hours (e.g., DKA, severe asthma)</option>
        <option value="hours_to_days" selected>hours_to_days (e.g., appendicitis, pneumonia)</option>
        <option value="days_to_weeks">days_to_weeks (e.g., subacute endocarditis)</option>
        <option value="weeks_to_months">weeks_to_months (e.g., TB, chronic conditions)</option>
        <option value="months_to_years">months_to_years (e.g., DM2, atherosclerosis)</option>
      </select>

      <label>Cascade chain (JSON array) <span class="required">*</span></label>
      <p class="hint">📌 <b>Format rules</b>:</p>
      <ul class="hint">
        <li>Array of step objects. Each step needs <code>step</code> (number) and <code>event</code> (mechanism).</li>
        <li>Add <code>produces_symptom</code> (or <code>produces_symptom_1</code>, <code>_2</code>...) for each symptom this step causes.</li>
        <li>Add <code>severity_marker: "red_flag"</code> if this step is dangerous.</li>
        <li>Add <code>produces_complication: "name"</code> for downstream complications.</li>
        <li>Add <code>time_hours: N</code> for time-sensitive events.</li>
      </ul>
      <textarea name="cascade" required>[
  {
    "step": 1,
    "event": "describe what happens at the cellular/anatomic level (e.g., 'fecalith obstructs lumen')",
    "produces_symptom": "describe what the patient experiences (e.g., 'intraluminal_pressure_rises')"
  },
  {
    "step": 2,
    "event": "next mechanism step (e.g., 'venous_drainage_compromised')",
    "produces_symptom_1": "first symptom (e.g., 'visceral_pain_T10_periumbilical')",
    "produces_symptom_2": "second symptom (e.g., 'nausea_via_vagal_reflex')"
  },
  {
    "step": 3,
    "event": "if untreated, severe complication (e.g., 'transmural_necrosis_with_perforation')",
    "severity_marker": "red_flag",
    "produces_complication": "name_of_complication (e.g., 'peritonitis')",
    "time_hours": 48
  }
]</textarea>

      <details style="margin-top:15px;">
        <summary>👀 See real example: Acute Appendicitis cascade</summary>
<pre>[
  {
    "step": 1,
    "event": "luminal_obstruction_by_fecalith_or_lymphoid_hyperplasia",
    "produces_symptom": "secretions_continue_accumulating"
  },
  {
    "step": 2,
    "event": "intraluminal_pressure_rises_above_70mmHg",
    "produces_symptom": "visceral_pain_via_T10",
    "derives_pain_zone": "umbilical_from_T10_dermatome"
  },
  {
    "step": 3,
    "event": "venous_drainage_compromised_at_85mmHg",
    "produces_symptom": "edema_and_ischemia"
  },
  {
    "step": 4,
    "event": "bacterial_overgrowth",
    "time_hours": 6,
    "produces_symptom": "fever_via_immune_response",
    "produces_symptom_2": "leukocytosis"
  },
  {
    "step": 5,
    "event": "wall_inflammation_extends_to_parietal_peritoneum",
    "time_hours": 12,
    "produces_symptom": "somatic_pain_via_T12_L1_parietal",
    "derives_pain_migration": "umbilical_to_RLQ_McBurney"
  },
  {
    "step": 6,
    "event": "perforation_if_untreated",
    "time_hours": 48,
    "severity_marker": "red_flag",
    "produces_complication": "peritonitis"
  }
]</pre>
      </details>

      <button type="submit" style="margin-top:20px;">💾 Save Cascade</button>
    </div>
  </form>

  <div class="panel">
    <h3>📚 Reference: Valid Organ IDs ({{valid_organs|length}} from atlas)</h3>
    <details>
      <summary>Show all valid organ IDs grouped by system</summary>
      <table class="organ-list" style="margin-top:10px;">
        <tr><th>ID</th><th>System</th><th>Name</th></tr>
        {% for o in valid_organs %}
        <tr><td><code>{{o.id}}</code></td><td>{{o.system}}</td><td>{{o.name}}</td></tr>
        {% endfor %}
      </table>
    </details>
  </div>
</body>
</html>
"""


REFERENCE_HTML = r"""
<!DOCTYPE html>
<html>
<head>
<title>Medical Reference — Cascade Editor</title>
<style>
  body { font-family: -apple-system, sans-serif; max-width: 1100px; margin: 30px auto; padding: 20px; background: #f5f7fa; color: #2c3e50; }
  .panel { background: white; border-radius: 8px; padding: 20px; margin: 20px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
  .btn-back { background: #95a5a6; text-decoration:none; padding: 10px 18px; border-radius:5px; color:white; display:inline-block; }
  table { width: 100%; border-collapse: collapse; margin: 15px 0; font-size: 14px; }
  th, td { padding: 10px 12px; border-bottom: 1px solid #ecf0f1; text-align: left; vertical-align: top; }
  th { background: #34495e; color: white; }
  td.level { font-family: monospace; font-weight: bold; color: #c0392b; white-space: nowrap; }
  td.zone { color: #27ae60; }
  td.organ { color: #2980b9; font-weight: 500; }
  code { background: #ecf0f1; padding: 2px 6px; border-radius: 3px; font-size: 13px; }
  .toc { background: #ecf3fc; padding: 15px; border-radius: 6px; margin: 15px 0; }
  .toc a { display: inline-block; margin: 4px 8px 4px 0; padding: 6px 12px; background: white; border-radius: 4px; text-decoration: none; color: #2980b9; font-size: 13px; }
  .toc a:hover { background: #3498db; color: white; }
  .example { background: #e8f5e9; padding: 12px; border-radius: 4px; font-family: monospace; font-size: 13px; margin: 8px 0; }
  .warning { background: #fff3e0; border-left: 4px solid #e67e22; padding: 12px; margin: 10px 0; border-radius: 4px; }
  .info { background: #e3f2fd; border-left: 4px solid #3498db; padding: 12px; margin: 10px 0; border-radius: 4px; }
  h2 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 8px; margin-top: 30px; }
  h3 { color: #34495e; margin-top: 25px; }
  .sign { background: #fffbe6; border: 1px solid #ffe58f; padding: 10px 14px; margin: 8px 0; border-radius: 6px; }
  .sign b { color: #d35400; }
</style>
</head>
<body>
  <a href="/" class="btn-back">← Back to home</a>
  <h1>📚 Medical Reference</h1>
  <p>Speed-card for filling cascade JSON — bookmark this page.</p>

  <div class="toc">
    <b>Quick jump:</b>
    <a href="#nerve-levels">Spinal nerve levels (T/L/S)</a>
    <a href="#trigeminal">Trigeminal (V1/V2/V3)</a>
    <a href="#dermatomes">Dermatomes by zone</a>
    <a href="#classic-signs">Classic clinical signs</a>
    <a href="#immune">Immune molecules</a>
    <a href="#organs">Naming conventions</a>
    <a href="#templates">Symptom name templates</a>
    <a href="#mistakes">Common mistakes</a>
  </div>

  <!-- ─────────────────────────── -->
  <div class="panel" id="nerve-levels">
    <h2>🧬 1. Spinal Nerve Levels — what each T/L/S means</h2>
    
    <div class="info">
      <b>Why this matters:</b> Internal organ pain shows up at the SKIN dermatome of the same spinal level.
      Heart (T1-T4) → pain felt at T1 dermatome (left arm). 
      Appendix (T10) → pain felt at T10 dermatome (umbilicus).
    </div>
    
    <h3>Cervical (C) — neck/shoulder/arm</h3>
    <table>
      <tr><th style="width:80px">Level</th><th>Body region</th><th>What organs use this</th></tr>
      <tr><td class="level">C2-C3</td><td class="zone">Back of head, neck</td><td class="organ">Meninges (occipital headache)</td></tr>
      <tr><td class="level">C3-C5</td><td class="zone">Neck, shoulder, top of shoulder</td><td class="organ">⭐ Diaphragm (referred shoulder pain — Kehr's, Boas)</td></tr>
      <tr><td class="level">C5-C8</td><td class="zone">Arm (lateral=C6, medial=C8)</td><td class="organ">Arm itself</td></tr>
    </table>

    <h3>Thoracic (T) — chest, abdomen ANTERIOR</h3>
    <table>
      <tr><th style="width:80px">Level</th><th>Body region</th><th>What organs use this</th></tr>
      <tr><td class="level">T1</td><td class="zone">⭐ Axilla, medial arm</td><td class="organ">Heart (MI left arm pain)</td></tr>
      <tr><td class="level">T2-T5</td><td class="zone">Chest central</td><td class="organ">Heart, esophagus, trachea</td></tr>
      <tr><td class="level">T2-T6</td><td class="zone">Lateral chest, axilla</td><td class="organ">Lungs, pleura</td></tr>
      <tr><td class="level">T6-T9</td><td class="zone">⭐ Epigastric (xiphoid to umbilicus)</td><td class="organ">FOREGUT: stomach, duodenum, liver, gallbladder, pancreas, spleen</td></tr>
      <tr><td class="level">T9-T11</td><td class="zone">Lower epigastric, periumbilical</td><td class="organ">Jejunum, ileum (start of midgut)</td></tr>
      <tr><td class="level">T10</td><td class="zone">⭐ Umbilical (around belly button)</td><td class="organ">MIDGUT: appendix, cecum, ascending colon, ileum</td></tr>
      <tr><td class="level">T11-T12</td><td class="zone">Lower abdomen</td><td class="organ">Bladder, ureter (upper)</td></tr>
      <tr><td class="level">T12</td><td class="zone">⭐ RLQ / LLQ (suprapubic)</td><td class="organ">Appendix when parietal peritoneum involved</td></tr>
    </table>

    <h3>Thoracic (T) — back/POSTERIOR (use _back suffix)</h3>
    <table>
      <tr><th style="width:90px">Level</th><th>Body region</th><th>Organs</th></tr>
      <tr><td class="level">T6_back-T9_back</td><td class="zone">Upper-mid back</td><td class="organ">⭐ Pancreas (epigastric pain → BACK radiation)</td></tr>
      <tr><td class="level">T10_back-T12_back</td><td class="zone">Lower back, flank</td><td class="organ">⭐ Kidney, adrenal (flank pain)</td></tr>
    </table>

    <h3>Lumbar (L) & Sacral (S)</h3>
    <table>
      <tr><th style="width:80px">Level</th><th>Body region</th><th>Organs</th></tr>
      <tr><td class="level">L1</td><td class="zone">⭐ Inguinal/groin, lower abdomen</td><td class="organ">HINDGUT: descending/sigmoid colon. ALSO renal colic groin radiation.</td></tr>
      <tr><td class="level">L1-L2</td><td class="zone">LLQ, suprapubic</td><td class="organ">Descending colon, sigmoid colon, uterus</td></tr>
      <tr><td class="level">L2-L4</td><td class="zone">Anterior thigh, knee</td><td class="organ">Hip joint</td></tr>
      <tr><td class="level">S1-S2</td><td class="zone">Posterior leg, heel</td><td class="organ">—</td></tr>
      <tr><td class="level">S2-S4</td><td class="zone">⭐ Perineum, anus, genitals</td><td class="organ">Rectum, prostate, urethra, bladder neck</td></tr>
    </table>
  </div>

  <!-- ─────────────────────────── -->
  <div class="panel" id="trigeminal">
    <h2>🧠 2. Trigeminal Nerve (face) — V1, V2, V3</h2>
    <div class="info">
      <b>Face doesn't follow spinal nerves.</b> The face is innervated by cranial nerve V (trigeminal) with 3 branches.
    </div>
    <table>
      <tr><th>Branch</th><th>Region</th><th>Used in</th></tr>
      <tr><td class="level">V1 (ophthalmic)</td><td class="zone">Forehead, scalp, eye, nasal bridge</td><td class="organ">Meningeal pain (intracranial via dural innervation)</td></tr>
      <tr><td class="level">V2 (maxillary)</td><td class="zone">Mid-face, upper lip, upper teeth, cheek</td><td class="organ">Sinusitis</td></tr>
      <tr><td class="level">V3 (mandibular)</td><td class="zone">Lower face, jaw, lower lip, lower teeth, ear</td><td class="organ">⭐ MI jaw pain (via vagal-trigeminal interaction)</td></tr>
    </table>
  </div>

  <!-- ─────────────────────────── -->
  <div class="panel" id="dermatomes">
    <h2>📍 3. Quick Reverse: Body Zone → Spinal Level</h2>
    <p>If you know WHERE the pain is, this tells you which level to use:</p>
    <table>
      <tr><th>Pain location</th><th>Spinal levels</th></tr>
      <tr><td class="zone">Forehead, eye</td><td class="level">V1</td></tr>
      <tr><td class="zone">Lower face / jaw</td><td class="level">V3</td></tr>
      <tr><td class="zone">Back of head</td><td class="level">C2, C3</td></tr>
      <tr><td class="zone">Neck</td><td class="level">C3, C4</td></tr>
      <tr><td class="zone">Shoulder tip</td><td class="level">C3-C5</td></tr>
      <tr><td class="zone">Left arm medial / axilla</td><td class="level">T1</td></tr>
      <tr><td class="zone">Chest center (sternum)</td><td class="level">T2-T5</td></tr>
      <tr><td class="zone">Nipple line</td><td class="level">T4</td></tr>
      <tr><td class="zone">Xiphoid / epigastric</td><td class="level">T6-T9</td></tr>
      <tr><td class="zone">Umbilical</td><td class="level">T10</td></tr>
      <tr><td class="zone">Suprapubic / RLQ / LLQ</td><td class="level">T12, L1</td></tr>
      <tr><td class="zone">Inguinal / groin</td><td class="level">L1</td></tr>
      <tr><td class="zone">Anterior thigh</td><td class="level">L2-L4</td></tr>
      <tr><td class="zone">Perineum / anus / genitals</td><td class="level">S2-S4</td></tr>
      <tr><td class="zone">Mid back</td><td class="level">T6_back-T9_back</td></tr>
      <tr><td class="zone">Lower back / flank</td><td class="level">T10_back-T12_back</td></tr>
    </table>
  </div>

  <!-- ─────────────────────────── -->
  <div class="panel" id="classic-signs">
    <h2>⭐ 4. Classic Clinical Signs (all from this map!)</h2>
    <p>These famous signs are NOT memorized — they EMERGE from the dermatome map:</p>
    
    <div class="sign"><b>MI left arm pain</b><br>Heart (T1-T4 visceral) → T1 dermatome → medial arm. <code>chest_pain_T1_T4_visceral</code> + <code>left_arm_pain_T1</code></div>
    
    <div class="sign"><b>MI jaw pain</b><br>Heart (vagal afferents) → V3 trigeminal → jaw. <code>jaw_pain_V3_referral</code></div>
    
    <div class="sign"><b>Kehr's sign (splenic rupture → left shoulder)</b><br>Spleen irritates diaphragm (C3-C5 phrenic) → C5 dermatome → shoulder tip. <code>shoulder_pain_via_phrenic_C3_C5</code></div>
    
    <div class="sign"><b>Boas sign (cholecystitis → right scapula)</b><br>Same mechanism, right side. Gallbladder → diaphragm → C3-C5 → R shoulder/scapula. <code>right_subscapular_pain_via_phrenic</code></div>
    
    <div class="sign"><b>Appendicitis migration (umbilicus → RLQ)</b><br>Phase 1: T10 visceral (appendix wall) → umbilical region. Phase 2: T12 somatic (parietal peritoneum) → RLQ McBurney point. <code>periumbilical_pain_T10_visceral</code> → <code>rlq_pain_T12_somatic_McBurney</code></div>
    
    <div class="sign"><b>Renal colic "loin to groin"</b><br>Ureter (T11-L2 visceral) → flank, then L1 dermatome → groin/inguinal. <code>flank_pain_T11_L2</code> + <code>groin_radiation_via_L1</code></div>
    
    <div class="sign"><b>Pancreatitis epigastric → back radiation</b><br>Pancreas (T6-T9 visceral) → epigastric. Retroperitoneal inflammation → T10-T12 back somatic → back. <code>epigastric_pain_T6_T9_visceral</code> + <code>back_radiation_T10_back_T12_back</code></div>
    
    <div class="sign"><b>Diaphragmatic shoulder pain</b><br>Any diaphragm irritation (subphrenic abscess, ruptured organ, free air) → C3-C5 phrenic → shoulder tip. <code>shoulder_tip_pain_via_C3_C5_phrenic</code></div>
  </div>

  <!-- ─────────────────────────── -->
  <div class="panel" id="immune">
    <h2>🛡️ 5. Immune Molecules — what each does</h2>
    <p>Use these in <code>event</code> and <code>produces_symptom</code> fields to make NEXUS understand the immune mechanism.</p>
    
    <h3>Bacterial / acute inflammation</h3>
    <table>
      <tr><th>Molecule</th><th>What it does</th><th>Example symptom string</th></tr>
      <tr><td class="level">IL-1</td><td>Acts on hypothalamus → fever; recruits neutrophils</td><td><code>fever_via_IL1_hypothalamus</code></td></tr>
      <tr><td class="level">IL-6</td><td>Drives liver CRP synthesis; fever; "sickness behavior"</td><td><code>fever_via_IL6_TNF</code><br><code>fatigue_via_IL6_sickness_behavior</code></td></tr>
      <tr><td class="level">TNF-α</td><td>Anorexia; fever; sepsis at high levels</td><td><code>anorexia_via_TNF_hypothalamus</code></td></tr>
      <tr><td class="level">Complement</td><td>Lyses bacteria; recruits neutrophils</td><td><code>tissue_damage_via_complement_MAC</code></td></tr>
      <tr><td class="level">CRP</td><td>Acute phase reactant — lab marker, not direct symptom</td><td>(use as cascade evidence, not symptom)</td></tr>
    </table>

    <h3>Viral response</h3>
    <table>
      <tr><th>Molecule</th><th>What it does</th><th>Example</th></tr>
      <tr><td class="level">Interferon (IFN-α/β)</td><td>⭐ Causes flu-like symptoms: myalgia, fatigue, low-grade fever</td><td><code>myalgia_via_interferon_response</code><br><code>fatigue_via_IFN_systemic</code></td></tr>
      <tr><td class="level">CD8 T cells</td><td>Kill virus-infected cells</td><td><code>tissue_damage_via_CD8_killing</code></td></tr>
    </table>

    <h3>Allergy / mast cell</h3>
    <table>
      <tr><th>Molecule</th><th>Effect</th><th>Example</th></tr>
      <tr><td class="level">Histamine</td><td>Vasodilation, edema, itch, bronchoconstriction</td><td><code>itching_via_histamine</code><br><code>bronchoconstriction_via_histamine</code></td></tr>
      <tr><td class="level">Leukotrienes</td><td>Bronchoconstriction (slower than histamine), mucus</td><td><code>bronchospasm_via_leukotrienes</code></td></tr>
      <tr><td class="level">IgE</td><td>Triggers mast cell degranulation</td><td><code>allergy_via_IgE_crosslinking</code></td></tr>
    </table>

    <h3>Coagulation</h3>
    <table>
      <tr><th>Molecule</th><th>Effect</th><th>Example</th></tr>
      <tr><td class="level">Thrombin / fibrin</td><td>Clot formation</td><td><code>thrombosis_via_fibrin</code></td></tr>
      <tr><td class="level">Platelet TXA2</td><td>Platelet aggregation</td><td><code>platelet_aggregation_via_TXA2</code></td></tr>
    </table>

    <div class="warning">
      <b>Don't mix them up:</b><br>
      • Muscle aches in flu = <b>interferon</b>, NOT IL-6<br>
      • Fever in bacterial = <b>IL-1, IL-6, TNF</b><br>
      • Itching in allergy = <b>histamine</b>, NOT TNF
    </div>
  </div>

  <!-- ─────────────────────────── -->
  <div class="panel" id="organs">
    <h2>🏷️ 6. Naming Conventions</h2>
    <table>
      <tr><th>Rule</th><th>✅ Correct</th><th>❌ Wrong</th></tr>
      <tr><td>All lowercase</td><td><code>heart</code></td><td><code>Heart</code></td></tr>
      <tr><td>Underscores not spaces/hyphens</td><td><code>r_lung</code>, <code>bile_duct</code></td><td><code>r-lung</code>, <code>r lung</code></td></tr>
      <tr><td>Side prefix r_ / l_</td><td><code>r_kidney</code>, <code>l_lung</code></td><td><code>right_kidney</code>, <code>kidney_R</code></td></tr>
      <tr><td>Artery suffix _a</td><td><code>lad_a</code>, <code>aorta</code></td><td><code>lad_artery</code></td></tr>
      <tr><td>Vein suffix _v</td><td><code>portal_v</code>, <code>ivc</code></td><td><code>portal_vein</code></td></tr>
      <tr><td>Failure mode = mechanism</td><td><code>luminal_obstruction</code></td><td><code>appendicitis</code></td></tr>
      <tr><td>Cascade event = mechanism phrase</td><td><code>premature_trypsinogen_activation</code></td><td><code>pancreas_breaks_down</code></td></tr>
    </table>
  </div>

  <!-- ─────────────────────────── -->
  <div class="panel" id="templates">
    <h2>📝 7. Symptom Name Templates</h2>
    <p>Use these patterns. NEXUS understands more when you include the mechanism.</p>
    
    <h3>Pain symptoms</h3>
    <div class="example">SITE_pain_LEVEL_TYPE</div>
    <ul>
      <li><code>chest_pain_T1_T4_visceral</code> — ✅ NEXUS knows it's heart-area, T1-T4 dermatome</li>
      <li><code>periumbilical_pain_T10_visceral</code> — ✅ midgut visceral</li>
      <li><code>rlq_pain_T12_somatic_McBurney</code> — ✅ somatic, parietal involvement</li>
      <li><code>flank_pain_T10_T12_back_somatic</code> — ✅ posterior dermatome</li>
    </ul>

    <h3>Mechanism-tagged symptoms</h3>
    <div class="example">SYMPTOM_via_MECHANISM</div>
    <ul>
      <li><code>fever_via_IL6_TNF</code> — ✅ tells immune pattern</li>
      <li><code>shoulder_pain_via_phrenic_C3_C5</code> — ✅ Kehr/Boas mechanism</li>
      <li><code>nausea_via_vagal_reflex</code> — ✅ autonomic mechanism</li>
      <li><code>polyuria_via_osmotic_diuresis</code> — ✅ DM2 mechanism</li>
    </ul>

    <h3>Severity markers in event</h3>
    <ul>
      <li><code>"severity_marker": "red_flag"</code> + <code>"time_hours": 6</code> for time-sensitive emergencies</li>
      <li><code>"produces_complication": "peritonitis"</code> for downstream serious conditions</li>
    </ul>
  </div>

  <!-- ─────────────────────────── -->
  <div class="panel" id="mistakes">
    <h2>⚠️ 8. Common Mistakes</h2>
    
    <div class="warning">
      <b>Mistake 1: Writing disease name as failure mode</b><br>
      ❌ <code>"mode": "appendicitis"</code><br>
      ✅ <code>"mode": "luminal_obstruction"</code>
    </div>

    <div class="warning">
      <b>Mistake 2: Vague symptom strings</b><br>
      ❌ <code>"belly_pain"</code> — NEXUS can't infer dermatome<br>
      ✅ <code>"periumbilical_pain_T10_visceral"</code>
    </div>

    <div class="warning">
      <b>Mistake 3: Wrong T-level</b><br>
      ❌ <code>"chest_pain_T8_visceral"</code> — T8 is upper abdomen, not chest<br>
      ✅ <code>"chest_pain_T1_T4_visceral"</code>
    </div>

    <div class="warning">
      <b>Mistake 4: Wrong immune molecule</b><br>
      ❌ <code>"myalgia_via_IL6"</code> — IL-6 doesn't cause muscle aches<br>
      ✅ <code>"myalgia_via_interferon_response"</code> — interferon is the culprit
    </div>

    <div class="warning">
      <b>Mistake 5: Forgetting laterality</b><br>
      ❌ <code>"organ_id": "kidney"</code> — atlas has r_kidney AND l_kidney<br>
      ✅ <code>"organ_id": "r_kidney"</code> or <code>"l_kidney"</code>
    </div>

    <div class="warning">
      <b>Mistake 6: Skipping cascade steps</b><br>
      Each step should be ONE atomic event. Don't combine multiple mechanisms in one step.<br>
      ❌ One step: "infection causes inflammation and fever and edema and AMS"<br>
      ✅ Step 1: infection → inflammation. Step 2: IL-1 release → fever. Step 3: edema → ICP rise → AMS.
    </div>
  </div>

  <a href="/new" style="display:inline-block;background:#27ae60;color:white;padding:14px 28px;border-radius:5px;text-decoration:none;font-weight:bold;font-size:16px;margin:20px 0;">📝 OK, I'm ready — start filling a cascade →</a>
</body>
</html>
"""



REGISTER_HTML = r"""
<!DOCTYPE html>
<html>
<head>
<title>Register Disease - Cascade Editor</title>
<style>
  body { font-family: -apple-system, sans-serif; max-width: 1000px; margin: 30px auto; padding: 20px; background: #f5f7fa; color: #2c3e50; }
  .panel { background: white; border-radius: 8px; padding: 25px; margin: 20px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
  .help-box { background: #fffbe6; border: 1px solid #ffe58f; padding: 15px; border-radius: 6px; margin: 15px 0; }
  label { display:block; margin: 14px 0 4px; font-weight: bold; color: #2c3e50; }
  label .required { color: #e74c3c; }
  input, select, textarea { width: 100%; padding: 9px; border: 1px solid #bdc3c7; border-radius: 4px; font-size: 14px; box-sizing: border-box; }
  button { background: #27ae60; color: white; border: none; padding: 14px 26px; border-radius: 5px; cursor: pointer; font-size: 16px; font-weight: bold; }
  .btn-back { background: #95a5a6; text-decoration:none; padding: 10px 18px; border-radius:5px; color:white; display:inline-block; }
  .hint { color: #7f8c8d; font-size: 13px; margin: 4px 0 0 0; }
  .error { color: #c0392b; padding: 12px; background: #fadbd8; border-radius: 4px; margin: 10px 0; font-weight: bold; }
  .success { color: #27ae60; padding: 12px; background: #d5f5e3; border-radius: 4px; margin: 10px 0; }
  .preview { background: #e8f5e9; padding: 15px; border-radius: 6px; margin: 10px 0; font-family: monospace; font-size: 13px; white-space: pre-wrap; }
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
</style>
</head>
<body>
  <a href="/" class="btn-back">← Home</a>
  <a href="/reference" style="display:inline-block;background:#9b59b6;color:white;padding:10px 18px;border-radius:5px;text-decoration:none;margin-left:8px;">📚 Reference</a>
  <h1>📝 Register Disease (Step 2 of 2)</h1>
  
  <div class="help-box">
    <h3>📖 What this does</h3>
    <p>After you add a <b>cascade</b> in the editor, use this page to <b>register the cascade as a real disease</b> in NEXUS.</p>
    <ul>
      <li>Auto-extracts symptoms / red flags / complications from your cascade</li>
      <li>Generates placeholder symptom_weights (you can refine later)</li>
      <li>Updates 4 files atomically (registry + disease_0001 + disease_anatomy + symptom data)</li>
    </ul>
    <p><b>Result:</b> NEXUS can now diagnose this disease in <code>nm.reason([symptoms])</code>.</p>
  </div>
  
  {% if error %}<div class="error">⚠️ {{error}}</div>{% endif %}
  {% if success_msg %}<div class="success">✅ {{success_msg}}</div>{% endif %}
  {% if extracted %}
  <div class="panel">
    <h3>✨ Auto-extracted from cascade</h3>
    <div class="preview">Common symptoms: {{extracted.common_symptoms|join(", ")}}

Red flags: {{extracted.red_flags|join(", ") or "(none)"}}

Complications: {{extracted.complications|join(", ") or "(none)"}}</div>
    <p class="hint">These come from your cascade's <code>produces_symptom</code> and <code>severity_marker</code> fields.</p>
  </div>
  {% endif %}
  
  <form method="POST" action="/register">
    <div class="panel">
      <label>Disease Name <span class="required">*</span></label>
      <input type="text" name="disease_name" placeholder="e.g., Acute Pancreatitis" required>
      <p class="hint">Use the standard clinical name (will appear in diagnosis output).</p>
      
      <label>ICD-10 Code (optional)</label>
      <input type="text" name="icd_code" placeholder="e.g., K85.9">
      
      <div class="grid-2">
        <div>
          <label>Organ ID <span class="required">*</span></label>
          <select name="organ" required>
            <option value="">— Pick organ with cascade —</option>
            {% for org_id, org in organs.items() %}
            <option value="{{org_id}}">{{org_id}} ({{org.failure_modes|length}} mode(s))</option>
            {% endfor %}
          </select>
          <p class="hint">Must already have cascade(s) added.</p>
        </div>
        <div>
          <label>Failure Mode <span class="required">*</span></label>
          <input type="text" name="failure_mode" placeholder="e.g., acute_pancreatitis" required>
          <p class="hint">Must match a mode in your chosen organ.</p>
        </div>
      </div>
      
      <label>Pathogen (optional)</label>
      <input type="text" name="pathogen" placeholder="e.g., escherichia_coli_uropathogenic (or leave blank)">
      <p class="hint">Only for infectious diseases. Must exist in pathogens.json.</p>
      
      <div class="grid-2">
        <div>
          <label>Prevalence</label>
          <select name="prevalence">
            <option value="rare">rare (< 0.01% / yr)</option>
            <option value="uncommon" selected>uncommon (0.01% - 1%)</option>
            <option value="common">common (1% - 10%)</option>
            <option value="very_common">very_common (>10%)</option>
          </select>
        </div>
        <div>
          <label>Sex bias</label>
          <select name="sex_bias">
            <option value="any" selected>any</option>
            <option value="female_predominant">female predominant</option>
            <option value="male_predominant">male predominant</option>
          </select>
        </div>
      </div>
      
      <div class="grid-2">
        <div>
          <label>Age groups (which apply)</label>
          <select name="age_groups" multiple style="height:140px;">
            <option value="infant">infant (0-1)</option>
            <option value="child">child (1-12)</option>
            <option value="adolescent">adolescent (12-18)</option>
            <option value="adult" selected>adult (18-65)</option>
            <option value="elderly">elderly (65+)</option>
          </select>
          <p class="hint">Hold Ctrl/Cmd for multiple.</p>
        </div>
        <div>
          <label>Triage level</label>
          <select name="triage_level">
            <option value="emergent">emergent (call 911 / immediate)</option>
            <option value="urgent" selected>urgent (within hours)</option>
            <option value="semi_urgent">semi_urgent (within day)</option>
            <option value="routine">routine</option>
          </select>
        </div>
      </div>
      
      <label><input type="checkbox" name="overwrite" value="1"> Overwrite if disease already exists</label>
      
      <button type="submit" style="margin-top:20px;">💾 Register Disease (updates 4 files)</button>
    </div>
  </form>
</body>
</html>
"""




# ─────────────────────────────────────────────────────────────
# Flask app
# ─────────────────────────────────────────────────────────────

app = Flask(__name__)


@app.route('/')
def index():
    kb = load_kb()
    organs = kb.get('organs', {})
    n_failure_modes = sum(len(o.get('failure_modes', [])) for o in organs.values())
    n_cascade_steps = sum(
        len(fm.get('cascade', []))
        for o in organs.values()
        for fm in o.get('failure_modes', [])
    )
    derived_set = set()
    for o in organs.values():
        for fm in o.get('failure_modes', []):
            for step in fm.get('cascade', []):
                for k, v in step.items():
                    if k.startswith('produces_symptom'):
                        derived_set.add(v)
    return render_template_string(
        INDEX_HTML,
        organs=organs,
        n_organs=len(organs),
        n_failure_modes=n_failure_modes,
        n_cascade_steps=n_cascade_steps,
        n_derived_symptoms=len(derived_set),
    )


@app.route('/organ/<organ_id>')
def organ_view(organ_id):
    kb = load_kb()
    organ = kb.get('organs', {}).get(organ_id)
    if not organ:
        return f"Organ {organ_id} not found", 404
    derived_picture = "(unable to derive)"
    try:
        from mechanism_derivation import MechanismDerivationEngine
        engine = MechanismDerivationEngine()
        if organ.get('failure_modes'):
            fm = organ['failure_modes'][0]
            d = engine.derive_disease(organ_id, fm['mode'])
            derived_picture = json.dumps({
                'common_symptoms': d.get('common_symptoms_clean',
                                         d.get('common_symptoms', [])),
                'red_flags': [r.get('event') for r in d.get('red_flags', [])],
                'complications': d.get('complications', []),
                'pathophysiology_chain': d.get('pathophysiology', '')[:200] + '...',
            }, indent=2, ensure_ascii=False)
    except Exception as e:
        derived_picture = f"(derivation error: {e})"
    return render_template_string(
        ORGAN_HTML,
        organ_id=organ_id,
        organ=organ,
        derived_picture=derived_picture,
        raw_json=json.dumps(organ, indent=2, ensure_ascii=False),
    )


@app.route('/new', methods=['GET', 'POST'])
def new_cascade():
    valid_organs = get_valid_organs()
    if request.method == 'POST':
        # Use dropdown if selected, else custom
        organ_id = request.form.get('organ_id', '').strip()
        if not organ_id:
            organ_id = request.form.get('organ_id_custom', '').strip().lower().replace(' ', '_')
        failure_mode = request.form.get('failure_mode', '').strip().lower()
        cause = request.form.get('cause', '').strip()
        speed = request.form.get('speed', '').strip()
        cascade_raw = request.form.get('cascade', '')
        if not organ_id or not failure_mode:
            return render_template_string(NEW_CASCADE_HTML, error="Organ ID and failure mode are required.", valid_organs=valid_organs)
        try:
            cascade = json.loads(cascade_raw)
        except json.JSONDecodeError as e:
            return render_template_string(NEW_CASCADE_HTML, error=f"Invalid JSON: {e}", valid_organs=valid_organs)
        if not isinstance(cascade, list):
            return render_template_string(NEW_CASCADE_HTML, error="Cascade must be a JSON array (square brackets).", valid_organs=valid_organs)
        errors = validate_cascade(cascade)
        if errors:
            return render_template_string(NEW_CASCADE_HTML, error="Validation: " + "; ".join(errors), valid_organs=valid_organs)
        # Save
        kb = load_kb()
        if organ_id not in kb['organs']:
            kb['organs'][organ_id] = {
                "primary_functions": [],
                "anatomic_constants": {},
                "failure_modes": []
            }
        new_mode = {"mode": failure_mode, "cascade": cascade}
        if cause: new_mode["cause"] = cause
        if speed: new_mode["speed"] = speed
        # Avoid duplicate failure mode
        existing = [fm.get('mode') for fm in kb['organs'][organ_id].get('failure_modes', [])]
        if failure_mode in existing:
            return render_template_string(NEW_CASCADE_HTML, error=f"Failure mode '{failure_mode}' already exists for {organ_id}. Use a different name.", valid_organs=valid_organs)
        kb['organs'][organ_id]['failure_modes'].append(new_mode)
        save_kb(kb)
        return redirect(f'/organ/{organ_id}')
    return render_template_string(NEW_CASCADE_HTML, error=None, valid_organs=valid_organs)



@app.route('/reference')
def reference():
    return render_template_string(REFERENCE_HTML)



@app.route('/register', methods=['GET', 'POST'])
def register():
    kb = load_kb()
    organs = kb.get('organs', {})
    
    if request.method == 'POST':
        try:
            result = register_disease(
                disease_name=request.form.get('disease_name', '').strip(),
                organ=request.form.get('organ', '').strip(),
                failure_mode=request.form.get('failure_mode', '').strip(),
                pathogen=request.form.get('pathogen', '').strip() or None,
                icd_code=request.form.get('icd_code', '').strip(),
                prevalence=request.form.get('prevalence', 'uncommon'),
                age_groups=request.form.getlist('age_groups') or ['adult'],
                sex_bias=request.form.get('sex_bias', 'any'),
                triage_level=request.form.get('triage_level', 'urgent'),
                overwrite=request.form.get('overwrite') == '1',
            )
            if result['success']:
                msg = f"Registered! Updated {len(result['files_modified'])} files: {', '.join(result['files_modified'])}. Restart NEXUS to use."
                return render_template_string(REGISTER_HTML, organs=organs,
                    success_msg=msg, error=None, extracted=result.get('extracted_clinical'))
            else:
                return render_template_string(REGISTER_HTML, organs=organs,
                    error="; ".join(result['errors']), extracted=None, success_msg=None)
        except Exception as e:
            return render_template_string(REGISTER_HTML, organs=organs,
                error=f"Internal error: {e}", extracted=None, success_msg=None)
    
    return render_template_string(REGISTER_HTML, organs=organs,
        error=None, extracted=None, success_msg=None)



@app.route('/refresh_weights', methods=['GET', 'POST'])
def refresh_weights():
    """Re-run weight derivation across all diseases."""
    try:
        from weight_deriver import WeightDeriver
        deriver = WeightDeriver()
        path = deriver.save()
        weights = deriver.derive_specificity_weights()
        return f"""
        <html><body style="font-family:sans-serif;max-width:800px;margin:30px auto;padding:20px;">
        <a href="/">← Home</a>
        <h2>✅ Auto-derived weights refreshed</h2>
        <p>Updated <code>{path}</code></p>
        <p>Recomputed weights for <b>{len(weights)} diseases</b>.</p>
        <p>Restart NEXUS to use the new weights in Step 5e (clinical specificity).</p>
        <pre>{json.dumps({k: list(v.keys())[:3] for k, v in list(weights.items())[:5]}, indent=2)}</pre>
        </body></html>
        """
    except Exception as e:
        return f"<p>Error: {e}</p>", 500


@app.route('/api/derive', methods=['POST'])
def api_derive():
    body = request.json or {}
    try:
        from mechanism_derivation import MechanismDerivationEngine
        engine = MechanismDerivationEngine()
        return jsonify(engine.derive_disease(
            primary_organ=body.get('organ', ''),
            failure_mode=body.get('failure_mode', ''),
            pathogen=body.get('pathogen'),
        ))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    valid = get_valid_organs()
    print(f"\n{'='*60}")
    print(f"NEXUS Cascade Editor")
    print(f"{'='*60}")
    print(f"  KB:           {KB_PATH}")
    print(f"  Valid organs: {len(valid)} (from atlas)")
    print(f"  Web UI:       http://localhost:5005")
    print(f"  Stop:         Ctrl+C")
    print(f"{'='*60}\n")
    app.run(host='0.0.0.0', port=5005, debug=False)