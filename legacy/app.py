"""
app.py — web frontend for the Pharmacology Reasoning Engine.

Two modes (reasoning stays server-side; the browser only shows the cited report):
  • Drug profile (default)  — pick a drug → monograph: use, dose, adverse effects,
    body-state effects, FAERS record, who-should-not-use — all cited & hyperlinked.
  • Scenario                — disease/target → ranked candidate drugs (the original flow).

Run:  pip install flask ; python3 app.py ; open http://127.0.0.1:5000
Live labels first:  python3 clinical_data.py albuterol epinephrine nitroglycerin aspirin heparin --out drug_clinical_data.json
"""
from __future__ import annotations
from flask import Flask, request, jsonify, render_template_string

import reasoning_engine as RE
import drug_profile as DP
from patient_profile import PatientProfile
from render_html import to_html, drug_profile_html

app = Flask(__name__)
DRUGS_PKPD = RE._load("drugs_pkpd.json")["drugs"]
CLINICAL = RE._load("drug_clinical_data.json")["drugs"]
# every drug we can profile: those with a mechanism entry OR a fetched FDA label
DRUG_LIST = sorted(set(DP.available_drugs(DRUGS_PKPD)) | set(CLINICAL.keys()))
DRUG_CLASSES = DP.drug_classes(CLINICAL)


@app.route("/api/browse", methods=["POST"])
def api_browse():
    d = request.get_json(force=True) or {}
    rows = DP.browse(CLINICAL, d.get("class_filter", ""), d.get("indication", ""))
    return jsonify({"rows": rows, "count": len(rows)})

SCENARIOS = {
    "acs": ([{"organ": "heart", "variable": "myocardial_oxygen_demand", "direction": "low"},
             {"organ": "blood", "variable": "platelet_aggregation", "direction": "low"},
             {"organ": "blood", "variable": "clot_propagation_risk", "direction": "low"}],
            "acute coronary syndrome"),
    "asthma": ([{"organ": "lung", "variable": "bronchospasm", "direction": "low"}], "asthma"),
    "gerd": ([{"organ": "gi", "variable": "gastric_acid", "direction": "low"}], "gastroesophageal reflux disease"),
}
PRESETS = [("Acute coronary syndrome", "acs"), ("Asthma / bronchospasm", "asthma"), ("GERD", "gerd")]


@app.route("/")
def index():
    return render_template_string(PAGE, drugs=DRUG_LIST, presets=PRESETS, classes=DRUG_CLASSES)


@app.route("/api/drug_profile", methods=["POST"])
def api_drug_profile():
    drug = (request.get_json(force=True) or {}).get("drug", "")
    prof = DP.build_profile(drug, DRUGS_PKPD, CLINICAL)
    return jsonify({"html": drug_profile_html(prof)})


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    d = request.get_json(force=True)
    targets, indication = SCENARIOS.get(d.get("scenario"), SCENARIOS["acs"])
    patient = PatientProfile(
        age=_num(d.get("age")), weight_kg=_num(d.get("weight_kg")), egfr=_num(d.get("egfr")),
        hepatic_status=d.get("hepatic_status") or "normal", pregnancy=bool(d.get("pregnancy")),
        allergies=_list(d.get("allergies")), current_medications=_list(d.get("current_medications")),
        symptoms=_list(d.get("symptoms")),
        vitals={k: _num(d.get(k)) for k in ("sbp", "heart_rate", "spo2") if _num(d.get(k)) is not None},
        labs={"potassium": _num(d.get("potassium"))} if _num(d.get("potassium")) is not None else {})
    out = RE.recommend(patient, targets, indication, DRUGS_PKPD, CLINICAL, scenario=d.get("scenario"))
    return jsonify({"html": to_html(out, DRUGS_PKPD)})


def _num(v):
    try:
        return float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _list(v):
    return [x.strip().lower() for x in (v or "").split(",") if x.strip()]


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Pharmacology Reasoning Engine</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,900&family=Newsreader:opsz,wght@6..72,400;6..72,500&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
:root{--paper:#f7f4ec;--ink:#1c1a17;--muted:#6b6356;--rule:#d8d0c0;--accent:#7c1d2b}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font-family:"Newsreader",Georgia,serif;
background-image:radial-gradient(circle at 1px 1px,rgba(0,0,0,.025) 1px,transparent 0);background-size:22px 22px}
.wrap{display:grid;grid-template-columns:360px 1fr;min-height:100vh}
.panel{padding:28px 26px;border-right:3px double var(--ink);position:sticky;top:0;height:100vh;overflow:auto}
.kicker{font-family:"JetBrains Mono",monospace;font-size:11px;letter-spacing:.22em;text-transform:uppercase;color:var(--accent);margin:0 0 8px}
h1{font-family:"Fraunces",serif;font-weight:900;font-size:26px;line-height:1.05;margin:0 0 16px}
.tabs{display:flex;gap:6px;margin-bottom:18px}
.tab{flex:1;font-family:"JetBrains Mono",monospace;font-size:11px;letter-spacing:.06em;text-transform:uppercase;
padding:9px;text-align:center;border:1px solid var(--ink);background:var(--paper);cursor:pointer}
.tab.on{background:var(--ink);color:var(--paper)}
label{display:block;font-family:"JetBrains Mono",monospace;font-size:10px;letter-spacing:.13em;text-transform:uppercase;color:var(--muted);margin:14px 0 4px}
input,select{width:100%;font-family:"Newsreader",serif;font-size:16px;padding:8px 10px;background:#fffdf8;border:1px solid var(--rule);border-radius:2px;color:var(--ink)}
.row{display:grid;grid-template-columns:1fr 1fr;gap:10px}.row3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px}
.chk{display:flex;align-items:center;gap:8px;margin-top:14px;font-size:15px}.chk input{width:auto}
button{margin-top:22px;width:100%;font-family:"JetBrains Mono",monospace;font-weight:600;font-size:13px;letter-spacing:.08em;padding:13px;background:var(--ink);color:var(--paper);border:none;border-radius:2px;cursor:pointer}
button:hover{background:var(--accent)}
.demo{font-family:"JetBrains Mono",monospace;font-size:10.5px;color:var(--muted);margin-top:16px;line-height:1.5;border-top:1px solid var(--rule);padding-top:12px}
.browse{margin-top:18px;border-top:1px solid var(--rule);padding-top:14px}
.kicker2{font-family:"JetBrains Mono",monospace;font-size:10px;letter-spacing:.18em;text-transform:uppercase;color:var(--accent);margin:0 0 6px}
#results{margin-top:12px;max-height:38vh;overflow:auto}
.rmeta{font-family:"JetBrains Mono",monospace;font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:0 0 6px}
.ritem{display:block;width:100%;text-align:left;margin:0 0 4px;padding:7px 9px;background:#fffdf8;border:1px solid var(--rule);border-radius:2px;cursor:pointer;font-family:"Newsreader",serif}
.ritem:hover{background:var(--ink)}.ritem:hover b,.ritem:hover span{color:var(--paper)}
.ritem b{font-weight:600;font-size:14px}.ritem span{display:block;color:var(--muted);font-size:11px;margin-top:2px}
.combo{position:relative}
.acpanel{position:absolute;left:0;right:0;top:calc(100% + 2px);z-index:30;background:#fffdf8;border:1px solid var(--ink);border-radius:2px;max-height:300px;overflow:auto;box-shadow:0 10px 28px rgba(0,0,0,.14)}
.acmeta{padding:6px 10px;font-family:"JetBrains Mono",monospace;font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);border-bottom:1px solid var(--rule);position:sticky;top:0;background:#fffdf8}
.acitem{padding:8px 10px;font-family:"Newsreader",serif;font-size:14px;line-height:1.25;cursor:pointer;border-bottom:1px solid var(--rule);word-break:break-word}
.acitem:last-child{border-bottom:0}
.acitem.active,.acitem:hover{background:var(--ink);color:var(--paper)}
.result{height:100vh;overflow:auto}iframe{width:100%;height:100%;border:0;background:var(--paper)}
.placeholder{display:flex;align-items:center;justify-content:center;height:100%;color:var(--muted);font-family:"Fraunces",serif;font-style:italic;font-size:20px;padding:40px;text-align:center}
.hidden{display:none}
@media(max-width:760px){.wrap{grid-template-columns:1fr}.panel{position:static;height:auto;border-right:0;border-bottom:3px double var(--ink)}.result{height:80vh}}
</style></head><body>
<div class="wrap">
  <div class="panel">
    <p class="kicker">Pharmacology Reasoning</p><h1>Drug & Scenario Explorer</h1>
    <div class="tabs">
      <div class="tab on" id="tab-drug" onclick="mode('drug')">Drug profile</div>
      <div class="tab" id="tab-scn" onclick="mode('scn')">Scenario</div>
    </div>

    <form id="fdrug" onsubmit="runDrug(event)">
      <label>Search a drug ({{drugs|length}} loaded)</label>
      <div class="combo">
        <input id="drugq" name="drug" autocomplete="off" placeholder="type a drug name…" required
               oninput="acFilter()" onfocus="acFilter()" onkeydown="acKey(event)">
        <div id="acpanel" class="acpanel hidden"></div>
      </div>
      <button type="submit">Show drug profile →</button>
    </form>
    <div class="browse">
      <p class="kicker2">or browse the labels</p>
      <label>By drug class</label>
      <select id="classf" onchange="runBrowse()">
        <option value="">— all classes —</option>
        {% for cls,n in classes %}<option value="{{cls}}">{{cls}} ({{n}})</option>{% endfor %}
      </select>
      <label>By indication (what it treats)</label>
      <input id="indq" placeholder="e.g. hypertension, asthma" onkeydown="if(event.key==='Enter'){event.preventDefault();runBrowse();}">
      <button type="button" onclick="runBrowse()">Find drugs →</button>
      <div id="results"></div>
    </div>

    <form id="fscn" class="hidden" onsubmit="runScn(event)">
      <label>Clinical scenario</label>
      <select name="scenario">{% for name,val in presets %}<option value="{{val}}">{{name}}</option>{% endfor %}</select>
      <div class="row"><div><label>Age</label><input name="age" type="number" value="50"></div>
        <div><label>Weight (kg)</label><input name="weight_kg" type="number" value="80"></div></div>
      <div class="row"><div><label>eGFR</label><input name="egfr" type="number" value="84"></div>
        <div><label>Hepatic</label><select name="hepatic_status"><option>normal</option><option>mild</option><option>impaired</option><option>severe</option></select></div></div>
      <label>Allergies</label><input name="allergies" placeholder="e.g., aspirin, penicillin">
      <label>Current medications</label><input name="current_medications" placeholder="e.g., warfarin">
      <label>Symptoms</label><input name="symptoms" placeholder="e.g., wheezing, hives">
      <div class="row3"><div><label>SBP</label><input name="sbp" type="number" value="112"></div>
        <div><label>HR</label><input name="heart_rate" type="number" value="118"></div>
        <div><label>SpO2</label><input name="spo2" type="number" placeholder="98"></div></div>
      <button type="submit">Run reasoning →</button>
    </form>
  </div>
  <div class="result"><div class="placeholder" id="ph">Pick a drug to see its full profile.</div>
    <iframe id="out" class="hidden"></iframe></div>
</div>
<script>
const DRUGS = {{ drugs|tojson }};
let acMatches=[], acIdx=-1;
function acFilter(){
  const q=document.getElementById('drugq').value.trim().toLowerCase();
  const panel=document.getElementById('acpanel');
  if(!q){panel.classList.add('hidden');panel.innerHTML='';acMatches=[];acIdx=-1;return;}
  const pre=[],sub=[];
  for(const d of DRUGS){const i=d.indexOf(q);if(i===0)pre.push(d);else if(i>0)sub.push(d);}
  const all=pre.concat(sub);acMatches=all.slice(0,80);acIdx=-1;
  if(!all.length){panel.innerHTML='<div class="acmeta">no matches</div>';panel.classList.remove('hidden');return;}
  let h=`<div class="acmeta">${all.length} match${all.length==1?'':'es'}${all.length>80?' · showing 80':''}</div>`;
  acMatches.forEach((d,i)=>{h+=`<div class="acitem" data-i="${i}" onmousedown="acPick(event,${i})">${d}</div>`;});
  panel.innerHTML=h;panel.classList.remove('hidden');
}
function acPick(e,i){if(e)e.preventDefault();const d=acMatches[i];if(d===undefined)return;
  document.getElementById('drugq').value=d;document.getElementById('acpanel').classList.add('hidden');
  render('/api/drug_profile',{drug:d});}
function acKey(e){
  const panel=document.getElementById('acpanel');
  if(panel.classList.contains('hidden'))return;
  if(e.key==='ArrowDown'){e.preventDefault();acIdx=Math.min(acIdx+1,acMatches.length-1);acHi();}
  else if(e.key==='ArrowUp'){e.preventDefault();acIdx=Math.max(acIdx-1,0);acHi();}
  else if(e.key==='Enter'){if(acMatches.length){e.preventDefault();acPick(null,acIdx>=0?acIdx:0);}}
  else if(e.key==='Escape'){panel.classList.add('hidden');}
}
function acHi(){const items=document.querySelectorAll('#acpanel .acitem');
  items.forEach((el,i)=>el.classList.toggle('active',i===acIdx));
  if(acIdx>=0&&items[acIdx])items[acIdx].scrollIntoView({block:'nearest'});}
document.addEventListener('click',e=>{const c=document.getElementById('acpanel');
  if(c&&!e.target.closest('.combo'))c.classList.add('hidden');});
function mode(m){
  document.getElementById('tab-drug').classList.toggle('on',m=='drug');
  document.getElementById('tab-scn').classList.toggle('on',m=='scn');
  document.getElementById('fdrug').classList.toggle('hidden',m!='drug');
  document.getElementById('fscn').classList.toggle('hidden',m!='scn');
}
async function render(url,payload){
  const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  const j=await r.json();const ifr=document.getElementById('out');
  document.getElementById('ph').classList.add('hidden');ifr.classList.remove('hidden');ifr.srcdoc=j.html;
}
function runDrug(e){e.preventDefault();render('/api/drug_profile',{drug:e.target.drug.value});}
function openDrug(name){document.querySelector('#fdrug [name=drug]').value=name;render('/api/drug_profile',{drug:name});}
async function runBrowse(){
  const cls=document.getElementById('classf').value, ind=document.getElementById('indq').value;
  const box=document.getElementById('results');box.innerHTML='<p class="rmeta">searching…</p>';
  const r=await fetch('/api/browse',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({class_filter:cls,indication:ind})});
  const j=await r.json();
  if(!j.rows.length){box.innerHTML='<p class="rmeta">no matches</p>';return;}
  let h=`<p class="rmeta">${j.count} match${j.count==1?'':'es'}${j.count>=300?' (showing first 300)':''}</p>`;
  for(const row of j.rows){
    h+=`<button type="button" class="ritem" onclick="openDrug('${row.drug.replace(/'/g,"\\'")}')">`+
       `<b>${row.drug}</b>${row.indications?'<span>'+row.indications+'</span>':''}</button>`;
  }
  box.innerHTML=h;
}
function runScn(e){e.preventDefault();const d=Object.fromEntries(new FormData(e.target).entries());render('/api/recommend',d);}
</script></body></html>"""

if __name__ == "__main__":
    app.run(debug=True, port=5000)