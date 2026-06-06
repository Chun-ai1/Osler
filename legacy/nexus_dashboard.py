"""
NEXUS Live Training Dashboard
════════════════════════════════════════════════════════════════
Flask app that:
  1. Runs the NEXUS self-learning loop in a background thread
  2. Streams real-time episode data via Server-Sent Events (SSE)
  3. Serves a live HTML dashboard at http://localhost:5050

Usage:
    pip install flask
    python nexus_dashboard.py

Then open http://localhost:5050 in your browser.

File layout expected:
    nexus_engine/          ← your existing modules
    nexus_learning_env.py  ← the env + agent from previous step
    nexus_reward.py        ← the physiology reward (this file's companion)
    nexus_dashboard.py     ← this file
"""

from __future__ import annotations
import json
import os
import queue
import random
import threading
import time
from collections import deque
from flask import Flask, Response, render_template_string

app = Flask(__name__)

# ── shared state (thread-safe) ────────────────────────────────
_event_queue: queue.Queue = queue.Queue(maxsize=500)
_train_state = {
    "running":    False,
    "episodes":   0,
    "dx_window":  deque(maxlen=50),
    "tx_window":  deque(maxlen=50),
    "rw_window":  deque(maxlen=50),
    "kg_size":    0,
    "epsilon":    1.0,
    "last_patient": {},
    "last_breakdown": {},
    "recent_log": deque(maxlen=30),
    "curve":      [],           # [{ep, dx, tx, reward}]
}
_train_lock = threading.Lock()


# ─────────────────────────────────────────────────────────────
# Training thread
# ─────────────────────────────────────────────────────────────

def _training_worker(n_episodes: int = 400, learn_every: int = 50):
    """Runs in a background thread; pushes SSE events into _event_queue."""
    try:
        _run_training(n_episodes, learn_every)
    except Exception as e:
        import traceback
        _push({"type": "error", "msg": f"{type(e).__name__}: {e}\n{traceback.format_exc()[-400:]}"})
        _train_state["running"] = False


def _run_training(n_episodes: int = 400, learn_every: int = 50):
    """Actual training logic — called by _training_worker inside a try/except."""
    import sys as _sys, os as _os

    # ── Path setup: ensure BOTH project root AND nexus_engine are importable ──
    _here = _os.path.dirname(_os.path.abspath(__file__))
    _parent = _os.path.dirname(_here)
    # Always add both — harmless if already present
    for _p in [_here, _parent]:
        if _p not in _sys.path:
            _sys.path.insert(0, _p)

    # ── Import NEXUS core (handles root vs inside-nexus_engine layout) ────────
    try:
        from nexus_engine.nexus_medical import NexusMedical
        from nexus_engine.nexus_learning_bridge import NexusLearner
    except ModuleNotFoundError:
        from nexus_medical import NexusMedical          # running from inside nexus_engine/
        from nexus_learning_bridge import NexusLearner

    # ── Import env + reward (these live at project root) ──────────────────────
    try:
        from nexus_learning_env import MedicalEnv, NexusRLAgent
    except ModuleNotFoundError as e:
        _push({"type": "error", "msg": f"Cannot find nexus_learning_env.py — make sure it is in the project root.\n{e}"})
        return
    try:
        from nexus_reward import PhysiologyReward
    except ModuleNotFoundError as e:
        _push({"type": "error", "msg": f"Cannot find nexus_reward.py — make sure it is in the project root.\n{e}"})
        return

    _push({"type": "status", "msg": "Loading NEXUS knowledge web…"})
    nexus = NexusMedical()
    nexus.load_knowledge()
    learner  = NexusLearner(nexus)
    env      = MedicalEnv(nexus, noise_p=0.15)
    agent    = NexusRLAgent(epsilon=1.0, epsilon_decay=0.97)
    # Pass the already-built atlas from env so PhysiologyReward doesn't build a second one
    reward_fn = PhysiologyReward(nexus, atlas=getattr(env, '_atlas', None))

    _push({"type": "status", "msg": "NEXUS ready — starting training loop"})
    _train_state["running"] = True

    for ep in range(1, n_episodes + 1):
        if not _train_state["running"]:
            break

        # ── episode ───────────────────────────────────────────
        obs        = env.reset()
        state_key  = agent.encode_state(obs)
        diagnosis, treatment = agent.choose_action(state_key)

        # physiology-aware reward
        patient    = env._current_patient
        nr         = env._run_nexus(patient["symptoms"])
        reward, breakdown = reward_fn.compute(diagnosis, treatment, nr, patient)

        # store episode
        env._log_case(patient["symptoms"], diagnosis, treatment, nr, reward)

        # learner feedback
        try:
            learner.feedback(nr, round_id=ep)
        except Exception:
            pass

        # Q-update — reuse current state_key as next-state approximation
        # (avoids a second env.reset() which re-instantiates AnatomyAtlas each ep)
        agent.update(state_key, diagnosis, treatment, reward, state_key)

        if ep % learn_every == 0:
            agent.replay(batch_size=min(64, len(agent.memory)))
            try:
                learner.learn_from_cases("case_records.jsonl")
            except Exception:
                pass

        # ── update shared state ───────────────────────────────
        with _train_lock:
            _train_state["episodes"]  = ep
            _train_state["epsilon"]   = round(agent.epsilon, 3)
            _train_state["dx_window"].append(1 if breakdown["correct_dx"] else 0)
            _train_state["tx_window"].append(1 if breakdown["correct_tx"] else 0)
            _train_state["rw_window"].append(reward)

            kg_sz = len(nexus.kg) if hasattr(nexus, "kg") else 0
            _train_state["kg_size"] = kg_sz

            dx_acc = sum(_train_state["dx_window"]) / len(_train_state["dx_window"])
            tx_acc = sum(_train_state["tx_window"]) / len(_train_state["tx_window"])
            avg_rw = sum(_train_state["rw_window"]) / len(_train_state["rw_window"])

            _train_state["last_patient"]   = {
                "disease":   patient["disease"],
                "symptoms":  patient["symptoms"][:4],
                "severity":  patient["severity"],
                "pathogen":  patient.get("pathogen", "?"),
                "organ":     patient.get("infected_organ", "?"),
            }
            _train_state["last_breakdown"] = breakdown

            log_entry = {
                "ep":       ep,
                "disease":  patient["disease"],
                "dx":       diagnosis,
                "tx":       treatment,
                "correct":  breakdown["correct_dx"] and breakdown["correct_tx"],
                "reward":   reward,
                "cascade":  breakdown.get("cascade_info", {}).get("outcome", ""),
                "alerts":   breakdown.get("chem_alerts", []),
            }
            _train_state["recent_log"].appendleft(log_entry)

            if ep % 5 == 0:
                _train_state["curve"].append({
                    "ep":     ep,
                    "dx":     round(dx_acc * 100, 1),
                    "tx":     round(tx_acc * 100, 1),
                    "reward": round(avg_rw, 2),
                })

        # ── push SSE event ────────────────────────────────────
        _push({
            "type":     "episode",
            "ep":       ep,
            "total":    n_episodes,
            "dx_acc":   round(dx_acc * 100, 1),
            "tx_acc":   round(tx_acc * 100, 1),
            "avg_rw":   round(avg_rw, 2),
            "epsilon":  round(agent.epsilon, 3),
            "kg_size":  kg_sz,
            "r_sepsis": round(breakdown.get("r_sepsis_cascade", 0), 2),
            "r_chem":   round(breakdown.get("r_chemistry", 0), 2),
            "cascade":  breakdown.get("cascade_info", {}).get("outcome", ""),
            "correct":  breakdown["correct_dx"] and breakdown["correct_tx"],
            "disease":  patient["disease"],
            "severity": patient["severity"],
            "alerts":   breakdown.get("chem_alerts", []),
            "curve":    list(_train_state["curve"])[-60:],
        })

        time.sleep(0.05)   # ~20 eps/s max (adjust to taste)

    _train_state["running"] = False
    _push({"type": "done", "msg": f"Training complete — {n_episodes} episodes"})


def _push(data: dict):
    try:
        _event_queue.put_nowait(data)
    except queue.Full:
        pass


# ─────────────────────────────────────────────────────────────
# Flask routes
# ─────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return render_template_string(DASHBOARD_HTML)


@app.get("/stream")
def stream():
    """SSE endpoint — browser connects here to receive live updates."""
    def generate():
        # Immediately confirm connection so browser knows SSE is alive
        yield f"data: {json.dumps({'type': 'connected', 'msg': 'Stream connected'})}\n\n"
        while True:
            try:
                data = _event_queue.get(timeout=2)
                yield f"data: {json.dumps(data)}\n\n"
                if data.get("type") == "done":
                    break
            except queue.Empty:
                # Heartbeat every 2s — keeps connection alive during slow NEXUS load
                yield f"data: {json.dumps({'type': 'ping', 'running': _train_state['running']})}\n\n"
    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no",
                             "Connection": "keep-alive"})


@app.post("/start")
def start_training():
    if _train_state["running"]:
        return {"status": "already running"}
    # Clear queue and reset state for a fresh run
    while not _event_queue.empty():
        try: _event_queue.get_nowait()
        except: break
    _train_state["episodes"] = 0
    _train_state["curve"] = []
    _train_state["recent_log"].clear()
    _train_state["dx_window"].clear()
    _train_state["tx_window"].clear()
    _train_state["rw_window"].clear()
    t = threading.Thread(target=_training_worker,
                         kwargs={"n_episodes": 400, "learn_every": 50},
                         daemon=True)
    t.start()
    return {"status": "started"}


@app.post("/stop")
def stop_training():
    _train_state["running"] = False
    return {"status": "stopped"}


@app.get("/state")
def get_state():
    with _train_lock:
        return {
            "episodes": _train_state["episodes"],
            "epsilon":  _train_state["epsilon"],
            "kg_size":  _train_state["kg_size"],
            "curve":    list(_train_state["curve"])[-60:],
            "log":      list(_train_state["recent_log"])[:15],
        }


# ─────────────────────────────────────────────────────────────
# HTML dashboard (single-file, no external build step)
# ─────────────────────────────────────────────────────────────

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NEXUS Live Training Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
       background:#0f1117;color:#e2e8f0;min-height:100vh;padding:20px}
  h1{font-size:18px;font-weight:600;color:#f8fafc;margin-bottom:16px;
     display:flex;align-items:center;gap:10px}
  .dot{width:10px;height:10px;border-radius:50%;background:#22c55e;
       box-shadow:0 0 6px #22c55e;animation:pulse 1.5s infinite}
  .dot.idle{background:#6b7280;box-shadow:none;animation:none}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
  .grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:16px}
  .grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:16px}
  .card{background:#1e2330;border:1px solid #2d3748;border-radius:10px;padding:14px}
  .stat-label{font-size:11px;color:#94a3b8;margin-bottom:4px;text-transform:uppercase;letter-spacing:.05em}
  .stat-val{font-size:26px;font-weight:600;color:#f8fafc}
  .stat-sub{font-size:11px;color:#64748b;margin-top:3px}
  .chart-wrap{position:relative;height:180px}
  .btn{padding:8px 20px;border-radius:6px;border:none;cursor:pointer;
       font-size:13px;font-weight:500;transition:opacity .15s}
  .btn-start{background:#22c55e;color:#052e16}
  .btn-stop{background:#ef4444;color:#fff}
  .btn:hover{opacity:.85}
  .btn:disabled{opacity:.4;cursor:not-allowed}
  .log-wrap{max-height:320px;overflow-y:auto}
  .log-row{display:grid;grid-template-columns:40px 130px 1fr 1fr 70px 70px 80px;
           gap:6px;padding:6px 8px;border-bottom:1px solid #1e2330;font-size:12px;
           align-items:center}
  .log-row:hover{background:#1e2330}
  .log-head{font-size:11px;color:#64748b;font-weight:500;background:#161b27;
            position:sticky;top:0}
  .tag{display:inline-block;padding:2px 7px;border-radius:4px;font-size:11px;font-weight:500}
  .tag-ok{background:#14532d;color:#86efac}
  .tag-fail{background:#450a0a;color:#fca5a5}
  .tag-mod{background:#1c1917;color:#a8a29e}
  .tag-sev{background:#431407;color:#fdba74}
  .tag-crit{background:#3b0764;color:#d8b4fe}
  .chem-alert{font-size:11px;color:#fb923c}
  .section-title{font-size:12px;color:#64748b;font-weight:500;margin-bottom:8px;
                 text-transform:uppercase;letter-spacing:.05em}
  .progress-bar{background:#1e2330;border-radius:4px;height:6px;overflow:hidden;margin-top:6px}
  .progress-fill{height:6px;border-radius:4px;background:#6366f1;transition:width .3s}
  .patient-card{background:#161b27;border-radius:8px;padding:12px;font-size:13px}
  .patient-card .sym{display:flex;flex-wrap:wrap;gap:5px;margin-top:6px}
  .sym-tag{background:#1e2330;border:1px solid #2d3748;border-radius:4px;
           padding:2px 8px;font-size:11px;color:#94a3b8}
  .reward-bar{display:flex;gap:6px;flex-direction:column;margin-top:8px}
  .rb-row{display:flex;align-items:center;gap:8px;font-size:12px}
  .rb-label{min-width:120px;color:#94a3b8}
  .rb-bg{flex:1;background:#2d3748;border-radius:3px;height:5px}
  .rb-fill{height:5px;border-radius:3px;transition:width .4s}
  .rb-val{min-width:40px;text-align:right;color:#f8fafc}
  #status-msg{font-size:12px;color:#64748b;margin-left:8px}
</style>
</head>
<body>

<h1>
  <div class="dot idle" id="dot"></div>
  NEXUS Self-Learning Medical Environment
  <span id="status-msg">Idle — press Start to begin training</span>
  <div style="margin-left:auto;display:flex;gap:8px">
    <button class="btn btn-start" id="btn-start" onclick="startTraining()">Start training</button>
    <button class="btn btn-stop"  id="btn-stop"  onclick="stopTraining()" disabled>Stop</button>
  </div>
</h1>

<!-- Top stats -->
<div class="grid">
  <div class="card">
    <div class="stat-label">Episodes</div>
    <div class="stat-val" id="v-ep">0</div>
    <div class="progress-bar"><div class="progress-fill" id="ep-bar" style="width:0%"></div></div>
  </div>
  <div class="card">
    <div class="stat-label">Dx accuracy (50-ep)</div>
    <div class="stat-val" id="v-dx">—</div>
    <div class="stat-sub">Correct diagnosis</div>
  </div>
  <div class="card">
    <div class="stat-label">Tx accuracy (50-ep)</div>
    <div class="stat-val" id="v-tx">—</div>
    <div class="stat-sub">Correct treatment</div>
  </div>
  <div class="card">
    <div class="stat-label">KG triples learned</div>
    <div class="stat-val" id="v-kg">—</div>
    <div class="stat-sub" id="v-eps">ε = 1.00</div>
  </div>
</div>

<!-- Charts -->
<div class="grid2">
  <div class="card">
    <div class="section-title">Learning curve</div>
    <div class="chart-wrap"><canvas id="curve-chart"></canvas></div>
  </div>
  <div class="card">
    <div class="section-title">Reward breakdown (last episode)</div>
    <div id="reward-breakdown" class="reward-bar">
      <div style="color:#64748b;font-size:12px">Waiting for first episode…</div>
    </div>
  </div>
</div>

<!-- Patient + log -->
<div class="grid2">
  <div class="card">
    <div class="section-title">Current patient</div>
    <div id="patient-display" class="patient-card">
      <div style="color:#64748b">No patient yet</div>
    </div>
    <div style="margin-top:10px">
      <div class="section-title">Sepsis cascade outcome</div>
      <div id="cascade-display" style="font-size:13px;color:#94a3b8">—</div>
      <div id="chem-alerts" style="margin-top:4px"></div>
    </div>
  </div>

  <div class="card">
    <div class="section-title">Episode log</div>
    <div class="log-wrap">
      <div class="log-row log-head">
        <span>Ep</span><span>Disease</span><span>Dx guess</span>
        <span>Treatment</span><span>Reward</span><span>Cascade</span><span>Result</span>
      </div>
      <div id="log-body"></div>
    </div>
  </div>
</div>

<script>
// ── Chart.js setup ──────────────────────────────────────────
const ctx = document.getElementById('curve-chart').getContext('2d');
const chart = new Chart(ctx, {
  type: 'line',
  data: {
    labels: [],
    datasets: [
      { label: 'Dx %',    data: [], borderColor: '#6366f1', tension: 0.4,
        pointRadius: 0, borderWidth: 2, fill: false },
      { label: 'Tx %',    data: [], borderColor: '#22c55e', tension: 0.4,
        pointRadius: 0, borderWidth: 2, fill: false },
      { label: 'Avg rw',  data: [], borderColor: '#f59e0b', tension: 0.4,
        pointRadius: 0, borderWidth: 2, fill: false, yAxisID: 'y2' },
    ]
  },
  options: {
    animation: false,
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { labels: { color: '#94a3b8', font: { size: 11 } } } },
    scales: {
      x: { ticks: { color: '#64748b', maxTicksLimit: 8 }, grid: { color: '#2d3748' } },
      y: { ticks: { color: '#64748b' }, grid: { color: '#2d3748' },
           min: 0, max: 100,
           title: { display: true, text: 'Accuracy %', color: '#64748b', font: { size: 10 } } },
      y2: { position: 'right', ticks: { color: '#f59e0b' }, grid: { display: false },
            title: { display: true, text: 'Reward', color: '#f59e0b', font: { size: 10 } } },
    }
  }
});

function updateChart(curve) {
  chart.data.labels         = curve.map(c => c.ep);
  chart.data.datasets[0].data = curve.map(c => c.dx);
  chart.data.datasets[1].data = curve.map(c => c.tx);
  chart.data.datasets[2].data = curve.map(c => c.reward);
  chart.update('none');
}

// ── Reward breakdown bars ───────────────────────────────────
const RW_FIELDS = [
  { key: 'r_diagnosis',    label: 'Diagnosis',       max: 4,   color: '#6366f1' },
  { key: 'r_treatment',    label: 'Treatment',       max: 3,   color: '#22c55e' },
  { key: 'r_nexus_consistency', label: 'NEXUS consistency', max: 1.2, color: '#38bdf8' },
  { key: 'r_sepsis_cascade', label: 'Sepsis cascade', max: 4, color: '#f59e0b' },
  { key: 'r_chemistry',    label: 'Lab chemistry',   max: 3,   color: '#f87171' },
  { key: 'r_red_flags',    label: 'Red flags',       max: 0.4, color: '#a78bfa' },
  { key: 'r_spread',       label: 'Pathogen spread', max: 1,   color: '#fb923c' },
];

function updateBreakdown(bd) {
  if (!bd || !bd.total) return;
  const html = RW_FIELDS.map(f => {
    const v = bd[f.key] || 0;
    const pct = Math.max(0, Math.min(100, ((v + f.max) / (2 * f.max)) * 100));
    const sign = v >= 0 ? '+' : '';
    return `<div class="rb-row">
      <span class="rb-label">${f.label}</span>
      <div class="rb-bg"><div class="rb-fill" style="width:${pct}%;background:${f.color}"></div></div>
      <span class="rb-val">${sign}${v.toFixed(2)}</span>
    </div>`;
  }).join('') + `<div class="rb-row" style="margin-top:6px;border-top:1px solid #2d3748;padding-top:6px">
      <span class="rb-label" style="color:#f8fafc;font-weight:500">Total reward</span>
      <div class="rb-bg"><div class="rb-fill" style="width:${Math.max(0,Math.min(100,((bd.total+10)/20)*100))}%;background:#f8fafc"></div></div>
      <span class="rb-val" style="color:#f8fafc;font-weight:500">${bd.total >= 0 ? '+' : ''}${bd.total.toFixed(2)}</span>
    </div>`;
  document.getElementById('reward-breakdown').innerHTML = html;
}

// ── Patient display ─────────────────────────────────────────
function updatePatient(d) {
  const sevTag = { moderate: 'tag-mod', severe: 'tag-sev', critical: 'tag-crit' }[d.severity] || 'tag-mod';
  const syms = (d.symptoms || []).map(s => `<span class="sym-tag">${s}</span>`).join('');
  document.getElementById('patient-display').innerHTML = `
    <div style="display:flex;gap:8px;align-items:center;margin-bottom:6px">
      <span class="tag ${sevTag}">${d.severity}</span>
      <span style="font-weight:500">${d.disease}</span>
      <span style="color:#64748b;font-size:11px">${d.pathogen} · organ: ${d.organ}</span>
    </div>
    <div class="sym">${syms}</div>`;
}

// ── Cascade display ─────────────────────────────────────────
const CASCADE_COLOR = { contained: '#22c55e', serious: '#f59e0b', critical: '#f97316', death: '#ef4444' };
function updateCascade(outcome, alerts) {
  const col = CASCADE_COLOR[outcome] || '#94a3b8';
  document.getElementById('cascade-display').innerHTML =
    `<span style="color:${col};font-weight:500">${outcome || '—'}</span>`;
  const alertHtml = (alerts || []).map(a =>
    `<div class="chem-alert">⚠ ${a}</div>`).join('');
  document.getElementById('chem-alerts').innerHTML = alertHtml;
}

// ── Log ─────────────────────────────────────────────────────
const MAX_LOG = 60;
let logRows = [];
function appendLog(d) {
  const ok = d.correct;
  const rSign = d.reward >= 0 ? '+' : '';
  const cascadeCol = CASCADE_COLOR[d.cascade] || '#64748b';
  logRows.unshift(`<div class="log-row">
    <span style="color:#64748b">${d.ep}</span>
    <span>${d.disease}</span>
    <span style="color:#94a3b8">${d.dx}</span>
    <span style="color:#94a3b8">${d.tx}</span>
    <span style="color:${d.reward>=0?'#22c55e':'#ef4444'}">${rSign}${d.reward.toFixed(2)}</span>
    <span style="color:${cascadeCol}">${d.cascade || '—'}</span>
    <span class="tag ${ok?'tag-ok':'tag-fail'}">${ok?'✓ OK':'✗ Fail'}</span>
  </div>`);
  if (logRows.length > MAX_LOG) logRows.pop();
  document.getElementById('log-body').innerHTML = logRows.join('');
}

// ── SSE ─────────────────────────────────────────────────────
let es;
function connectSSE() {
  es = new EventSource('/stream');
  es.onerror = () => {
    document.getElementById('status-msg').textContent = 'SSE connection lost — reload page';
    document.getElementById('dot').classList.add('idle');
  };
  es.onmessage = (e) => {
    const d = JSON.parse(e.data);
    if (d.type === 'ping') {
      if (d.running) document.getElementById('status-msg').textContent = 'Loading NEXUS knowledge… (this takes ~10s)';
      return;
    }
    if (d.type === 'connected') {
      document.getElementById('status-msg').textContent = 'Connected — waiting for NEXUS to load…';
      return;
    }
    if (d.type === 'status') {
      document.getElementById('status-msg').textContent = d.msg;
      return;
    }
    if (d.type === 'error') {
      document.getElementById('status-msg').textContent = '⚠ ERROR — check terminal';
      // Show full error in the patient panel so it's visible
      document.getElementById('patient-display').innerHTML =
        '<pre style="color:#f87171;font-size:11px;white-space:pre-wrap">' + d.msg + '</pre>';
      document.getElementById('dot').classList.add('idle');
      document.getElementById('btn-stop').disabled = true;
      document.getElementById('btn-start').disabled = false;
      return;
    }
    if (d.type === 'done') {
      document.getElementById('status-msg').textContent = d.msg;
      document.getElementById('dot').classList.add('idle');
      document.getElementById('btn-stop').disabled = true;
      document.getElementById('btn-start').disabled = false;
      return;
    }
    if (d.type !== 'episode') return;

    // Update stats
    document.getElementById('v-ep').textContent = d.ep;
    document.getElementById('v-dx').textContent = d.dx_acc + '%';
    document.getElementById('v-tx').textContent = d.tx_acc + '%';
    document.getElementById('v-kg').textContent = d.kg_size.toLocaleString();
    document.getElementById('v-eps').textContent = 'ε = ' + d.epsilon;
    document.getElementById('ep-bar').style.width = (d.ep / d.total * 100) + '%';

    if (d.curve)   updateChart(d.curve);
    appendLog(d);
    updateCascade(d.cascade, d.alerts);

    // patient + breakdown come on the event
    if (d.disease) {
      updatePatient({ disease: d.disease, severity: d.severity,
                      symptoms: [], pathogen: '', organ: '' });
    }
  };
}

function startTraining() {
  fetch('/start', { method: 'POST' }).then(() => {
    document.getElementById('dot').classList.remove('idle');
    document.getElementById('btn-start').disabled = true;
    document.getElementById('btn-stop').disabled = false;
    document.getElementById('status-msg').textContent = 'Starting…';
    connectSSE();
  });
}

function stopTraining() {
  fetch('/stop', { method: 'POST' }).then(() => {
    document.getElementById('dot').classList.add('idle');
    document.getElementById('btn-stop').disabled = true;
    document.getElementById('btn-start').disabled = false;
    document.getElementById('status-msg').textContent = 'Stopped';
    if (es) es.close();
  });
}
</script>
</body>
</html>"""


if __name__ == "__main__":
    import sys as _sys, os as _os
    _here = _os.path.dirname(_os.path.abspath(__file__))
    if _os.path.basename(_here) == "nexus_engine":
        _sys.path.insert(0, _os.path.dirname(_here))
    print("=" * 55)
    print("  NEXUS Live Training Dashboard")
    print("  http://localhost:5050")
    print("=" * 55)
    app.run(host="0.0.0.0", port=5050, debug=False, threaded=True)