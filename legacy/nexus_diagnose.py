"""
NEXUS Self-Contained Diagnostic
================================
Run: python nexus_engine/nexus_diagnose.py

Zero imports from nexus_learning_env internals.
"""
import sys, os, random
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(_here) == "nexus_engine":
    sys.path.insert(0, os.path.dirname(_here))

print("Loading NEXUS...")
try:
    from nexus_engine.nexus_medical import NexusMedical
except ModuleNotFoundError:
    from nexus_medical import NexusMedical

nexus = NexusMedical()
nexus.load_knowledge()
SEP = "=" * 60

PATIENT_POOL = [
    {"disease":"pneumonia",      "symptoms":["cough","fever","shortness of breath","chest pain","fatigue"],       "correct_treatments":["antibiotics","oxygen therapy"]},
    {"disease":"flu",            "symptoms":["fever","body aches","fatigue","cough","headache"],                   "correct_treatments":["antivirals","rest","fluids"]},
    {"disease":"appendicitis",   "symptoms":["abdominal pain","nausea","fever","vomiting"],                        "correct_treatments":["surgery","antibiotics"]},
    {"disease":"meningitis",     "symptoms":["headache","stiff neck","fever","confusion","nausea"],                "correct_treatments":["antibiotics","steroids","iv fluids"]},
    {"disease":"heart attack",   "symptoms":["chest pain","left arm pain","shortness of breath","nausea","sweating"],"correct_treatments":["aspirin","pci","nitroglycerin"]},
    {"disease":"gastroenteritis","symptoms":["diarrhea","vomiting","abdominal pain","nausea","fever"],             "correct_treatments":["fluids","rest","antiemetics"]},
    {"disease":"sepsis",         "symptoms":["fever","confusion","shortness of breath","body aches","weakness"],   "correct_treatments":["antibiotics","iv fluids","vasopressors"]},
    {"disease":"asthma",         "symptoms":["wheezing","shortness of breath","chest tightness","cough"],          "correct_treatments":["inhaler","bronchodilators","steroids"]},
    {"disease":"migraine",       "symptoms":["headache","nausea","dizziness","sensitivity to light"],              "correct_treatments":["triptans","nsaids","rest"]},
    {"disease":"covid",          "symptoms":["fever","cough","fatigue","shortness of breath","loss of smell"],     "correct_treatments":["antivirals","oxygen therapy","rest"]},
]
TREATMENTS = ["antibiotics","antivirals","antifungals","oxygen therapy","iv fluids",
              "fluids","rest","aspirin","steroids","nsaids","triptans","inhaler",
              "bronchodilators","surgery","pci","nitroglycerin","vasopressors","antiemetics","analgesics"]
DISEASES = [p["disease"] for p in PATIENT_POOL]
N_DX, N_FEATURES = len(DISEASES), 64
COMMON = ["fever","cough","headache","chest pain","abdominal pain",
          "nausea","shortness of breath","fatigue","diarrhea","vomiting"]
SYSTEMS = ["respiratory","cardiovascular","neurologic","gi","systemic","immune","unknown"]

def matches(a,b): a,b=a.lower().strip(),b.lower().strip(); return a==b or a in b or b in a

def run_nexus(symptoms):
    try: return nexus.enhance_pipeline_result({"symptoms":symptoms,"final_symptoms":symptoms,"top_diseases":[],"reasoning":""})
    except Exception as e: return {"nexus_diagnoses":[],"nexus_consistency":{},"error":str(e)}

def build_vec(symptoms, nr):
    dx_list = nr.get("nexus_diagnoses",[])
    v = np.zeros(N_FEATURES, np.float32)
    for d in dx_list[:10]:
        name=d.get("disease","").lower(); score=min(float(d.get("score",0))/2,1)
        for j,known in enumerate(DISEASES):
            if set(known.split())&set(name.replace("-"," ").split()) or known in name or name in known:
                v[j]=max(v[j],score); break
    off=N_DX
    for k,sym in enumerate(COMMON):
        if sym in symptoms: v[off+k]=1.0
    off+=len(COMMON)
    v[off]=min(len(nr.get("nexus_red_flags",[]))/5,1); off+=1
    v[off]=nr.get("nexus_consistency",{}).get("consistency_score",0.5); off+=1
    try:
        from nexus_engine.nexus_medical import SYMPTOM_TO_SYSTEM
    except: SYMPTOM_TO_SYSTEM={}
    sf="unknown"
    for s in symptoms:
        if s in SYMPTOM_TO_SYSTEM: sf=SYMPTOM_TO_SYSTEM[s]; break
    if sf in SYSTEMS: v[off+SYSTEMS.index(sf)]=1.0
    return v

def reward(dx, tx, p, nr):
    r  = 1.0 if matches(dx,p["disease"]) else -1.0
    r += 1.0 if any(matches(tx,t) for t in p["correct_treatments"]) else -1.0
    top = nr.get("nexus_diagnoses",[{}])[0].get("disease","").lower()
    if matches(dx, top): r += 0.5
    return round(r,3)

# ── TEST 1: Oracle reward ──────────────────────────────────────
print(f"\n{SEP}\nTEST 1: Oracle reward\n{SEP}")
oracles, randoms = [], []
for p in PATIENT_POOL:
    nr = run_nexus(p["symptoms"])
    true_dx=p["disease"]; true_tx=p["correct_treatments"][0]
    wrong_dx=next(d for d in DISEASES if d!=true_dx)
    wrong_tx=next(t for t in TREATMENTS if t not in p["correct_treatments"])
    o=reward(true_dx,true_tx,p,nr); r_=reward(wrong_dx,wrong_tx,p,nr)
    oracles.append(o); randoms.append(r_)
    top=nr.get("nexus_diagnoses",[{}])[0].get("disease","none")[:20]
    print(f"  {'OK' if o>0 else 'BUG'} {true_dx:18s} oracle={o:+.2f} random={r_:+.2f} nexus='{top}'")
print(f"\n  avg oracle={sum(oracles)/len(oracles):+.3f}  avg random={sum(randoms)/len(randoms):+.3f}")
print(f"  RESULT: {'OK - reward correct' if sum(oracles)/len(oracles)>0 else 'BUG - oracle negative!'}")

# ── TEST 2: Action mapping ────────────────────────────────────
print(f"\n{SEP}\nTEST 2: Action mapping\n{SEP}")
bugs=[]
for p in PATIENT_POOL:
    dx,tx=p["disease"],p["correct_treatments"][0]
    in_d=dx in DISEASES; in_t=tx in TREATMENTS
    di=DISEASES.index(dx) if in_d else -1
    ok=in_d and in_t and DISEASES[di]==dx
    print(f"  {'OK' if ok else 'BUG'} {dx:18s} di={di:2d} in_tx={in_t}")
    if not ok: bugs.append(dx)
print(f"\n  NEXUS name matching:")
for nexus_n,pool_n in [("Community-acquired Pneumonia","pneumonia"),("Acute Gastroenteritis","gastroenteritis"),("Influenza","flu"),("COVID-19","covid"),("Bacterial Meningitis","meningitis")]:
    m=matches(nexus_n.lower(),pool_n)
    w=bool(set(pool_n.split())&set(nexus_n.lower().replace("-"," ").split()))
    print(f"  {'OK' if m or w else 'NO MATCH'} '{nexus_n}' -> '{pool_n}'  matches={m} words={w}")
print(f"\n  RESULT: {'OK' if not bugs else f'BUG: {bugs}'}")

# ── TEST 3: State diversity ───────────────────────────────────
print(f"\n{SEP}\nTEST 3: State diversity\n{SEP}")
vecs=[]
for p in PATIENT_POOL:
    nr=run_nexus(p["symptoms"]); v=build_vec(p["symptoms"],nr); vecs.append(v)
    top=nr.get("nexus_diagnoses",[{}])[0].get("disease","none")[:22]
    print(f"  {p['disease']:18s} nz={np.count_nonzero(v):2d}/64 nexus='{top}' v[:3]={v[:3].round(3)}")
dists=[np.linalg.norm(vecs[i]-vecs[j]) for i in range(len(vecs)) for j in range(i+1,len(vecs))]
print(f"\n  dist min={min(dists):.3f} avg={sum(dists)/len(dists):.3f} max={max(dists):.3f}")
print(f"  RESULT: {'OK - distinct' if min(dists)>0.01 else 'BUG - states too similar!'}")

# ── TEST 4: Overfit 5 cases ───────────────────────────────────
print(f"\n{SEP}\nTEST 4: Overfit test - network memorise 5 cases?\n{SEP}")
samples=[]
for p in PATIENT_POOL[:5]:
    nr=run_nexus(p["symptoms"]); v=build_vec(p["symptoms"],nr)
    di=DISEASES.index(p["disease"]); r_=reward(p["disease"],p["correct_treatments"][0],p,nr)
    samples.append((v,di,r_,p["disease"]))
    print(f"  {p['disease']:18s} di={di} reward={r_:+.2f} nz={np.count_nonzero(v)}")

rng=np.random.default_rng(0)
W1=rng.normal(0,np.sqrt(2/64),(64,32)).astype(np.float32); b1=np.zeros(32,np.float32)
W2=rng.normal(0,np.sqrt(2/32),(32,N_DX)).astype(np.float32); b2=np.zeros(N_DX,np.float32)
mW1=np.zeros_like(W1);vW1=np.zeros_like(W1);mb1=np.zeros_like(b1);vb1=np.zeros_like(b1)
mW2=np.zeros_like(W2);vW2=np.zeros_like(W2);mb2=np.zeros_like(b2);vb2=np.zeros_like(b2)
t=0

def adam(p,m,v,g,t,lr=1e-3):
    m[:]=0.9*m+0.1*g; v[:]=0.999*v+0.001*g**2
    p-=lr*(m/(1-0.9**t))/(np.sqrt(v/(1-0.999**t))+1e-8)

print("\n  Training 3000 steps...")
for _ in range(3000):
    v,di,r_,_=random.choice(samples); t+=1
    h=np.maximum(0,v@W1+b1); q=h@W2+b2
    e=np.zeros(N_DX); e[di]=-(r_-q[di])*2
    adam(W2,mW2,vW2,h[:,None]*e[None,:],t); adam(b2,mb2,vb2,e,t)
    dh=(e@W2.T)*(h>0)
    adam(W1,mW1,vW1,v[:,None]*dh[None,:],t); adam(b1,mb1,vb1,dh,t)

print("\n  Results:")
correct=0
for v,di,r_,disease in samples:
    h=np.maximum(0,v@W1+b1); q=h@W2+b2; pred=int(np.argmax(q)); ok=pred==di; correct+=ok
    print(f"  {'OK' if ok else 'FAIL'} {disease:18s} true={di} pred={pred}({DISEASES[pred]}) Q[true]={q[di]:+.3f}")
pct=correct/len(samples)*100
print(f"\n  {correct}/{len(samples)} ({pct:.0f}%)")
print(f"  RESULT: {'OK - network learns, bug is in training loop' if pct>=80 else 'BUG - network cannot memorise even 5 cases'}")
print(f"\n{SEP}\nDIAGNOSTIC COMPLETE\n{SEP}")