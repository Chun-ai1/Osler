"""
render_html.py — HTML frontend (English). Renders reasoning_engine output as an
editorial clinical dossier with clickable superscript citations + references.

UI reflects: KDIGO eGFR display (no false "renal mild"), multiple target states,
per-drug mechanism chains, alias-merged cards, result levels (mechanism-only /
label-not-loaded / dose-blocked-pending-validation / validated), and the rule that
patient-specific dosing is blocked until a label dose rule is human-validated.
"""
from __future__ import annotations
import html, re
from typing import Any, Dict, List, Optional

_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
_DEC = {"ok": ("OK", "ok"), "caution": ("Caution", "caution"), "adjust": ("Adjust", "adjust"),
        "avoid": ("Do not recommend", "avoid"), "emergency": ("Emergency", "emergency"),
        "insufficient_data": ("Insufficient data", "insufficient")}


def _dailymed(s): return f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={s}" if s and _UUID.match(s) else None


class _Cites:
    def __init__(self): self.items, self._idx, self._bym, self._ext, self._int = [], {}, {}, 0, 0
    def add(self, label, section=None, set_id=None, note=None, url=None, external=True):
        key = (label, section, set_id, note, url, external)
        if key in self._idx: return self._idx[key]
        if external:
            self._ext += 1; marker = str(self._ext)
        else:
            self._int += 1; marker = f"M{self._int}"
        self._idx[key] = marker
        it = {"marker": marker, "external": external, "label": label, "section": section,
              "set_id": set_id, "note": note, "url": url or _dailymed(set_id)}
        self.items.append(it); self._bym[marker] = it
        return marker
    def sup(self, marker):
        it = self._bym.get(marker, {})
        if it.get("external") and it.get("url"):
            return (f'<sup class="cite cite-ext"><a href="{it["url"]}" target="_blank" rel="noopener" '
                    f'title="{html.escape(str(it.get("label","")))}">{marker}</a></sup>')
        tip = html.escape(f'{it.get("label","")} — {it.get("note","")}'.strip(" —"))
        return f'<sup class="cite cite-int" title="{tip}">{marker}</sup>'
    def _row(self, it):
        parts = [f'<span class="src">{html.escape(it["label"])}</span>']
        if it["section"]: parts.append(f'<span class="sec">{html.escape(it["section"])}</span>')
        if it["set_id"] and _UUID.match(it["set_id"]): parts.append(f'<code>set_id {html.escape(it["set_id"])}</code>')
        line = " · ".join(parts)
        if it["url"]:
            lt = "view label ↗" if "dailymed" in it["url"] else "view source ↗"
            line += f' &nbsp;<a class="ext" href="{it["url"]}" target="_blank" rel="noopener">{lt}</a>'
        if it["note"]: line += f' <span class="note">— {html.escape(it["note"])}</span>'
        return f'<li><span class="rn">{it["marker"]}</span>{line}</li>'
    def html(self):
        ext = [it for it in self.items if it["external"]]
        intn = [it for it in self.items if not it["external"]]
        out = []
        if ext:
            out.append('<p class="ref-group">External sources (cited)</p><ol class="refs">')
            out += [self._row(it) for it in ext]; out.append('</ol>')
        if intn:
            out.append('<p class="ref-group">Internal model — NOT an external source</p><ul class="refs internal">')
            out += [self._row(it) for it in intn]; out.append('</ul>')
        return "\n".join(out)


def _label_cite(c, section, ev):
    set_id = ev[0].get("set_id") if ev else None
    ret = ev[0].get("retrieved_at") if ev else None
    note = None
    if set_id and not _UUID.match(set_id): note, set_id = "awaiting live fetch — run clinical_data.populate()", None
    elif ret: note = f"retrieved {ret}"
    return c.sup(c.add("Official clinical source — DailyMed FDA SPL", section, set_id, note))


def _display_status(r):
    """Honest badge: reflects decision + off-label + dose gating, not just safety."""
    dec = r["safety"]["decision"]; lvl = r.get("result_level", ""); ind = r.get("indication_support", "")
    if dec == "avoid": return ("DO NOT RECOMMEND", "avoid")
    if dec == "emergency": return ("EMERGENCY", "emergency")
    if "not loaded" in lvl: return ("NO LABEL", "insufficient")
    if ind == "not in label / off-label": return ("OFF-LABEL", "caution")
    if r.get("dose", {}).get("patient_specific_allowed"):
        return ("DOSE OK", "ok") if dec == "ok" else (dec.upper(), "caution")
    if "blocked pending validation" in lvl: return ("DOSE BLOCKED", "caution")
    return (dec.upper(), "insufficient")


def to_html(out, drugs_pkpd=None, title="Pharmacology Reasoning Report"):
    c = _Cites(); drugs_pkpd = drugs_pkpd or {}; pt = out["patient"]; body = []
    flags = ", ".join(pt.get("flags", [])) or "none"
    meds = ", ".join(pt.get("meds", [])) or "none"
    allg = ", ".join(pt.get("allergies", [])) or "none"
    targets = "".join(f'<li>{html.escape(t)}</li>' for t in out.get("target_states", []))
    body.append(f'''<header class="masthead">
      <p class="kicker">Decision Support · Mechanism → Label → Patient → Gate</p>
      <h1>{html.escape(title)}</h1>
      <div class="meta">
        <div class="full"><span>Target states</span><ul class="targets">{targets}</ul></div>
        <div><span>Indication</span><b>{html.escape(str(out.get("indication") or "unspecified"))}</b></div>
        <div><span>Patient age</span><b>{html.escape(str(pt.get("age")))}</b></div>
        <div class="full"><span>Renal function</span><b>{html.escape(pt.get("renal_label",""))}</b></div>
        <div><span>Allergies</span><b>{html.escape(allg)}</b></div>
        <div><span>Current meds</span><b>{html.escape(meds)}</b></div>
        <div class="full"><span>Patient flags</span><b>{html.escape(flags)}</b></div>
      </div>
    </header>''')

    for r in out["candidates"]:
        dec = r["safety"]["decision"]; dec_label, cls = _DEC.get(dec, (dec, "insufficient"))
        badge_label, badge_cls = _display_status(r)
        ev = r.get("evidence", []); lines = []
        alias = f' <span class="alias">aliases: {html.escape(", ".join(r["aliases"]))}</span>' if r.get("aliases") else ""

        rationale = (drugs_pkpd.get(r["drug"], {}).get("rationale") or "").strip() or None
        mn = c.add("Internal mechanism map (drug_mechanisms / state_effects)", section=rationale,
                   note="estimated mechanism, not a clinical dosing source", external=False)
        mt = ", ".join(m["target"] + f" ({m['effect_type']})" for m in r.get("matched_targets", []))
        lines.append(f'<li><b>Mechanism.</b> {html.escape(r["mechanism_chain"])}{c.sup(mn)}<br>'
                     f'<span class="muted">matches target: {html.escape(mt)}</span></li>')

        sup = r["indication_support"]
        if sup == "label-supported":
            lines.append(f'<li><b>Indication.</b> Listed as an approved use in the label.{_label_cite(c, "Indications and Usage", ev)}</li>')
        elif sup == "not in label / off-label":
            ptype = r.get("loaded_product_type")
            ptxt = f' (loaded product: {html.escape(str(ptype))})' if ptype else ""
            lines.append(f'<li><b>Indication.</b> Not listed in the loaded label{ptxt} — this indication is off-label here, '
                         f'or the loaded product/formulation is wrong for this scenario; guideline/clinician review.{_label_cite(c, "Indications and Usage", ev)}</li>')
        else:
            lines.append('<li><b>Indication.</b> No label loaded; approval status unverified.</li>')

        d = r["dose"]
        if d.get("patient_specific_allowed") and d.get("verbatim"):
            s = d.get("structured", {}) or {}
            struct_bits = " ".join(str(x) for x in [s.get("dose_value"), s.get("dose_unit")] if x)
            freq = f', {html.escape(str(s.get("frequency")))}' if s.get("frequency") else ""
            mx = f' (max {html.escape(str(s.get("maximum")))})' if s.get("maximum") else ""
            struct = f'<span class="dose">{html.escape(struct_bits)}{freq}{mx}</span> ' if struct_bits else ""
            sig = ""
            if d.get("validated_by"):
                sig = f' <span class="muted">— validated by {html.escape(str(d["validated_by"]))} on {html.escape(str(d.get("validated_at","")))}</span>'
            lines.append(f'<li><b>Dose.</b> {struct}{sig}'
                         f'<div class="muted-label">label text{_label_cite(c, "Dosage and Administration", ev)}:</div>'
                         f'{_collapsible(str(d["verbatim"]), lead_chars=150)}</li>')
        elif d.get("verbatim"):
            lines.append(f'<li><b>Dose.</b> <span class="block">Patient-specific dosing blocked pending human validation.</span>'
                         f'<div class="muted-label">label text{_label_cite(c, "Dosage and Administration", ev)}:</div>'
                         f'{_collapsible(str(d["verbatim"]), lead_chars=150)}</li>')
        else:
            lines.append('<li><b>Dose.</b> <span class="block">No label-sourced dose loaded — the engine does not generate a number.</span></li>')

        for rs in r["safety"].get("reasons", []):
            if rs.get("type") == "none":
                lines.append('<li class="risk"><b>Safety status.</b> No blocking factor matched in the label.</li>'); continue
            if rs.get("type") == "no_clinical_data":
                lines.append(f'<li class="risk"><b>Safety status.</b> {html.escape(rs["message"])}.</li>'); continue
            src = rs.get("source"); msg = html.escape(rs["message"])
            if src and "label" in src.lower():
                section = src.split("\u2014")[-1].strip()
                lines.append(f'<li class="risk"><b>Risk.</b> {msg}.{_label_cite(c, section, ev)}</li>')
            else:
                lines.append(f'<li class="risk"><b>Risk.</b> {msg}.</li>')

        sig = r.get("faers_signals", [])
        if sig:
            top = ", ".join(f'{html.escape(s["event"])} ({s.get("report_count","?")})' for s in sig[:5])
            fn = c.add("openFDA drug/event (FAERS)", section="real-world reports",
                       note="signal only, not causation; data may lag months")
            lines.append(f'<li class="faers"><b>Adverse-event signal.</b> {top}.{c.sup(fn)}</li>')

        missing = r["safety"].get("missing_patient_data")
        miss = f'<p class="missing">Missing patient data: {html.escape(", ".join(missing))} — cannot personalize.</p>' if missing else ""
        role = r.get("clinical_role", {})
        role_label = role.get("label", "MECHANISM CANDIDATE")
        role_cls = role.get("role", "mechanism_candidate")
        if role_cls not in ("primary_candidate", "emergency_context_candidate"):
            lines.append(f'<li class="risk"><b>Clinical-context gate.</b> {html.escape(role.get("reason",""))}</li>')
        body.append(f'''<section class="card {badge_cls}">
          <div class="card-head"><h2>{html.escape(r["drug"])}{alias}</h2>
            <span class="badge {badge_cls}">{html.escape(badge_label)}</span></div>
          <p class="level"><span class="role role-{html.escape(role_cls)}">{html.escape(role_label)}</span> · {html.escape(r["result_level"])}</p>
          <ol class="claims">{"".join(lines)}</ol>
          <p class="verdict"><b>Conclusion.</b> {html.escape(r["final_answer"])}</p>{miss}
        </section>''')

    return _HEAD + _CSS + _MID + "".join(body) + f'''
      <section class="references"><h2>References</h2>{c.html()}</section>
      <footer class="disclaimer"><p>{html.escape(out.get("_disclaimer",""))}</p></footer>
    </main>{_JS}</body></html>'''


_HEAD = '<!doctype html><html lang="en"><head><meta charset="utf-8">' \
    '<meta name="viewport" content="width=device-width, initial-scale=1">' \
    '<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>' \
    '<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,900&family=Newsreader:opsz,wght@6..72,400;6..72,500&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">'

_CSS = """<style>
:root{--paper:#f7f4ec;--ink:#1c1a17;--muted:#6b6356;--rule:#d8d0c0;--accent:#7c1d2b;
--ok:#1f5d4c;--caution:#9a6a12;--adjust:#27496b;--avoid:#7c1d2b;--insufficient:#5b5750;}
*{box-sizing:border-box}html{scroll-behavior:smooth}
body{margin:0;background:var(--paper);color:var(--ink);font-family:"Newsreader",Georgia,serif;font-size:18px;line-height:1.55;
background-image:radial-gradient(circle at 1px 1px,rgba(0,0,0,.025) 1px,transparent 0);background-size:22px 22px}
main{max-width:840px;margin:0 auto;padding:52px 28px 92px}
.masthead{border-bottom:3px double var(--ink);padding-bottom:24px}
.kicker{font-family:"JetBrains Mono",monospace;font-size:11px;letter-spacing:.22em;text-transform:uppercase;color:var(--accent);margin:0 0 10px}
h1{font-family:"Fraunces",serif;font-weight:900;font-size:clamp(32px,5.5vw,48px);line-height:1.03;margin:0 0 20px;letter-spacing:-.01em}
.meta{display:grid;grid-template-columns:repeat(2,1fr);gap:2px 28px}
.meta>div{display:flex;flex-direction:column;padding:7px 0;border-bottom:1px solid var(--rule)}
.meta .full{grid-column:1/-1}
.meta span{font-family:"JetBrains Mono",monospace;font-size:10px;letter-spacing:.15em;text-transform:uppercase;color:var(--muted)}
.meta b{font-weight:500}
.targets{margin:2px 0 0;padding-left:18px}.targets li{font-weight:500}
.card{margin:28px 0;padding:22px 26px;background:#fffdf8;border:1px solid var(--rule);border-left:5px solid var(--insufficient);box-shadow:5px 6px 0 rgba(0,0,0,.04)}
.card.ok{border-left-color:var(--ok)}.card.caution{border-left-color:var(--caution)}.card.adjust{border-left-color:var(--adjust)}
.card.avoid{border-left-color:var(--avoid)}.card.emergency{border-left-color:#5a0f17}
.card-head{display:flex;align-items:baseline;justify-content:space-between;gap:14px}
.card-head h2{font-family:"Fraunces",serif;font-weight:600;font-size:28px;margin:0;text-transform:capitalize}
.alias{font-family:"JetBrains Mono",monospace;font-size:11px;color:var(--muted);text-transform:none;font-weight:400}
.badge{font-family:"JetBrains Mono",monospace;font-size:11px;font-weight:600;padding:5px 11px;border-radius:2px;white-space:nowrap;color:#fffdf8;background:var(--insufficient)}
.badge.ok{background:var(--ok)}.badge.caution{background:var(--caution)}.badge.adjust{background:var(--adjust)}.badge.avoid{background:var(--avoid)}.badge.emergency{background:#5a0f17}
.level{font-family:"JetBrains Mono",monospace;font-size:11px;letter-spacing:.04em;color:var(--muted);margin:4px 0 0;text-transform:uppercase}
.role{font-weight:600}
.role-primary_candidate{color:var(--ok)}
.role-emergency_context_candidate{color:var(--avoid)}
.role-conditional_candidate{color:var(--caution)}
.role-mechanism_candidate{color:var(--muted)}
.claims{list-style:none;margin:14px 0 6px;padding:0}
.claims li{position:relative;padding:8px 0 8px 22px;border-bottom:1px dotted var(--rule)}
.claims li:before{content:"";position:absolute;left:2px;top:16px;width:6px;height:6px;background:var(--muted);transform:rotate(45deg)}
.claims li.risk:before{background:var(--accent)}.claims li.faers:before{background:var(--adjust)}
.claims b{font-family:"Fraunces",serif;font-weight:600}
.muted{color:var(--muted);font-size:14.5px}
.dose{font-family:"JetBrains Mono",monospace;font-size:14px;background:#efe9da;padding:1px 6px}
.block{color:var(--accent);font-style:italic}
.verdict{margin:16px 0 0;padding:12px 16px;background:#efe9da;border-left:3px solid var(--ink)}
.missing{font-family:"JetBrains Mono",monospace;font-size:12px;color:var(--muted);margin:8px 0 0}
sup.cite{font-family:"JetBrains Mono",monospace;font-size:.62em;line-height:0;margin-left:1px}
sup.cite a{text-decoration:none;color:var(--accent);font-weight:600;padding:0 2px;border-radius:2px}
sup.cite a:hover{background:var(--accent);color:#fffdf8}
.references{margin-top:50px;border-top:3px double var(--ink);padding-top:18px}
.references h2{font-family:"Fraunces",serif;font-weight:900;font-size:24px;margin:0 0 14px}
.refs{list-style:none;margin:0;padding:0}
.refs li{position:relative;padding:10px 0 10px 34px;border-bottom:1px solid var(--rule);font-size:14.5px;line-height:1.5}
.refs li.flash{background:#fbeec3;transition:background .25s}
.rn{position:absolute;left:0;top:10px;font-family:"JetBrains Mono",monospace;font-size:12px;font-weight:600;color:var(--accent)}
.refs .src{font-weight:500}.refs .sec{font-style:italic;color:#444}
.refs code{font-family:"JetBrains Mono",monospace;font-size:11.5px;color:var(--muted)}
.refs .ext{font-family:"JetBrains Mono",monospace;font-size:11.5px;color:var(--adjust);text-decoration:none}
.refs .ext:hover{text-decoration:underline}.refs .note{color:var(--muted);font-style:italic}
.disclaimer{margin-top:28px;padding:16px 20px;border:1px solid var(--accent);background:#fbf1ef}
.disclaimer p{font-family:"Fraunces",serif;font-size:14px;color:var(--accent);margin:0;font-weight:500}
.ref-group{font-family:"JetBrains Mono",monospace;font-size:10px;letter-spacing:.15em;text-transform:uppercase;color:var(--muted);margin:16px 0 6px}
.muted-label{font-family:"JetBrains Mono",monospace;font-size:10px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);margin:6px 0 2px}
ul.lbl{margin:4px 0 0;padding-left:0;list-style:none}
ul.lbl li{padding:3px 0 3px 16px;line-height:1.55;position:relative}
ul.lbl li:before{content:"–";position:absolute;left:0;color:var(--muted)}
details.more{margin:4px 0 0}
details.more>summary{font-family:"JetBrains Mono",monospace;font-size:11px;letter-spacing:.05em;color:var(--accent);cursor:pointer;list-style:none;padding:2px 0}
details.more>summary::-webkit-details-marker{display:none}
details.more>summary:before{content:"▸ ";font-size:10px}
details.more[open]>summary:before{content:"▾ "}
details.more>summary:hover{text-decoration:underline}
details.more .full{margin:6px 0 0;padding:10px 12px;background:#fffdf8;border-left:2px solid var(--rule);font-size:15px;line-height:1.55}
.refs.internal .rn{color:var(--muted)}
sup.cite-int a{color:var(--muted)}
sup.cite-int a:hover{background:var(--muted);color:#fffdf8}
@media(max-width:560px){.meta{grid-template-columns:1fr}}
</style>"""
_MID = "</head><body><main>"
_JS = """<script>
document.querySelectorAll('sup.cite a').forEach(function(a){a.addEventListener('click',function(){
var el=document.getElementById('ref-'+this.dataset.n);if(!el)return;el.classList.remove('flash');void el.offsetWidth;el.classList.add('flash');
setTimeout(function(){el.classList.remove('flash')},1400);});});
</script>"""


if __name__ == "__main__":
    from pathlib import Path
    import reasoning_engine as RE
    from patient_profile import PatientProfile
    drugs_pkpd = RE._load("drugs_pkpd.json")["drugs"]
    clinical = RE._load("drug_clinical_data.json")["drugs"]
    asthma = [{"organ": "lung", "variable": "bronchospasm", "direction": "low"},
              {"organ": "lung", "variable": "airflow_resistance", "direction": "low"}]
    # asthma + tachycardia, NO anaphylaxis features → albuterol primary, epinephrine conditional
    patient = PatientProfile(age=50, weight_kg=80, egfr=84,
                             vitals={"sbp": 112, "heart_rate": 118}, symptoms=["wheezing"])
    out = RE.recommend(patient, asthma, "asthma", drugs_pkpd, clinical, scenario="asthma")
    Path("pharm_report.html").write_text(to_html(out, drugs_pkpd), encoding="utf-8")
    print("wrote pharm_report.html")


# ════════════════════════════════════════════════════════════════════
# Drug-centric monograph (drug → everything)
# ════════════════════════════════════════════════════════════════════

def _faers_url(drug):
    import urllib.parse
    q = urllib.parse.urlencode({"search": f'patient.drug.openfda.generic_name:"{drug}"',
                                "count": "patient.reaction.reactionmeddrapt.exact"})
    return f"https://api.fda.gov/drug/event.json?{q}"


# Human-readable official FDA page for FAERS (the openFDA API only returns raw JSON).
_FAERS_PAGE = ("https://www.fda.gov/drugs/fda-adverse-event-monitoring-system-aems/"
               "fda-adverse-event-monitoring-system-aems-public-dashboard")

# FDA SPL labels embed standardized PLR section numbers in the raw text:
#   "2 DOSAGE AND ADMINISTRATION", "2.1 Acute ...", "[see Warnings (5.1)]", "(6.1)".
# These are index numbers, not data — strip them. The hard part is NOT deleting dose
# values that look identical ("2.5 mg", "1.2 mg/kg", "5.5 mg"): we only strip a number
# when it is followed by a SECTION HEADING word, sits in a "(...)"/"[see ...]" reference,
# or is a clause-initial integer before a known section title.
_HEADWORD = r'(?=[A-Z][A-Za-z]{2,})'                   # Title-Case OR ALL-CAPS heading word
_SEC_WORDS = (r'INDICATIONS|DOSAGE|CONTRAINDICATIONS|WARNINGS|ADVERSE|DRUG|USE|OVERDOSAGE|'
              r'DESCRIPTION|CLINICAL|NONCLINICAL|REFERENCES|HOW|PATIENT|BOXED|STORAGE')
_XREF = re.compile(r'[\(\[]\s*see[^)\]]*[)\]]', re.I)                       # [see Warnings (5.1)]
_SUBSEC = re.compile(r'\b\d{1,2}(?:\.\d{1,2})+\b\s+' + _HEADWORD)           # 2.1 Acute / 6.1 ADVERSE
_BARE_SEC = re.compile(r'(?:^|(?<=[.;:]\s))\d{1,2}\s+(?=(?:' + _SEC_WORDS + r')\b)', re.I)  # 2 DOSAGE
_SECREF = re.compile(r'\(\s*\d{1,2}(?:\.\d{1,2})*(?:\s*,\s*\d{1,2}(?:\.\d{1,2})*)*\s*\)')   # (6.1) (5.2, 5.3)
# A run of 4+ consecutive numeric tokens = a flattened table (dosimetry/PK/trial data).
# Doses survive because units interleave ("162 mg to 325 mg", "2.5 mg/kg up to 5.5 mg").
_NUMRUN = re.compile(r'(?:[-+]?\d[\d.,/]*%?\s+){3,}[-+]?\d[\d.,/]*%?')


def _clean_label(t):
    if not t:
        return t
    t = _XREF.sub('', t)
    t = _SECREF.sub('', t)
    t = _BARE_SEC.sub('', t)
    t = _SUBSEC.sub('', t)
    t = re.sub(r'^\s*\d{1,2}(?:\.\d{1,2})+\s+', '', t)   # leading bare subsection number
    t = _NUMRUN.sub(' ', t)                              # collapse flattened numeric tables
    t = re.sub(r'[\[\]]', '', t)                          # stray brackets from cross-ref removal
    t = re.sub(r'\(\s*\)', '', t)                         # empty parens
    t = re.sub(r'\s+', ' ', t)
    t = re.sub(r'\s+([.,;:)])', r'\1', t)
    return t.strip(' .;:')


def _is_tabular(t):
    """True if the text is mostly numbers — a flattened label table (dosimetry, PK, trial data)."""
    toks = t.split()
    if len(toks) < 5:
        return False
    nums = sum(1 for w in toks if re.fullmatch(r'[-+]?\d[\d.,]*%?', w))
    return nums / len(toks) > 0.4


def _lead(text, lead_chars=190):
    """First one or two whole sentences (never breaks mid-sentence)."""
    sents = re.split(r'(?<=[.;])\s+', text)
    out = ""
    for s in sents:
        if out and len(out) + len(s) > lead_chars:
            break
        out += (" " if out else "") + s
    return out.strip()


def _tidy_chunk(ch):
    """Replace a flattened numeric table with a short label instead of a wall of numbers."""
    if _is_tabular(ch):
        words = [w for w in ch.split()[:5] if not re.fullmatch(r'[-+]?\d[\d.,]*%?', w)]
        lead = " ".join(words[:4])
        return (f"{lead} — " if lead else "") + "numeric table (see full label)"
    return ch


def _split_subsections(text):
    """Split only at real subsection headings (number + heading word), never at dose decimals."""
    raw = _XREF.sub('', re.sub(r'\s+', ' ', text or '')).strip()
    parts = re.split(r'\s*\b\d{1,2}(?:\.\d{1,2})+\b\s+' + _HEADWORD, raw)
    parts = [_clean_label(p) for p in parts]
    return [p for p in parts if p and len(p) > 8]


def _collapsible(text, lead_chars=190):
    """Clean text → short lead by default, full text in a collapsible <details>.
    Subsections go on their own line; flattened numeric tables are summarized, not dumped."""
    clean = _clean_label(text)
    if not clean:
        return '<p class="block">—</p>'
    chunks = [_tidy_chunk(ch) for ch in _split_subsections(text)]
    if len(chunks) > 1:
        # lead with the first chunk that is real prose, not a table summary
        lead_src = next((ch for ch in chunks if "numeric table" not in ch), chunks[0])
        lead = lead_src if len(lead_src) <= lead_chars * 1.4 else _lead(lead_src, lead_chars)
        full = "".join(f'<li>{html.escape(ch)}</li>' for ch in chunks)
        return (f'<p>{html.escape(lead)}</p>'
                f'<details class="more"><summary>show full label text ({len(chunks)} sections)</summary>'
                f'<ul class="lbl">{full}</ul></details>')
    if _is_tabular(clean):
        return f'<p>{html.escape(_tidy_chunk(clean))}</p>'
    lead = _lead(clean, lead_chars)
    if len(clean) <= len(lead) + 5:
        return f'<p>{html.escape(clean)}</p>'
    return (f'<p>{html.escape(lead)}</p>'
            f'<details class="more"><summary>show full label text</summary>'
            f'<p class="full">{html.escape(clean)}</p></details>')


def _label_ref(c, section, p):
    """One citation for the whole DailyMed label — every section links to the same page."""
    set_id = p["source"].get("set_id")
    ret = p["source"].get("retrieved_at")
    note, sid = (None, set_id)
    if not (set_id and _UUID.match(set_id)):
        note, sid = "awaiting live fetch — run clinical_data.py", None
    elif ret:
        note = f"retrieved {ret}"
    return c.sup(c.add("FDA label — openFDA / DailyMed SPL", None, sid, note))


def drug_profile_html(p, title=None):
    c = _Cites()
    drug = p["drug"]
    title = title or f"{drug.capitalize()} — Drug Profile"
    alias = f' <span class="alias">aliases: {html.escape(", ".join(p["aliases"]))}</span>' if p.get("aliases") else ""
    ident = p["identity"]
    routes = ", ".join(ident.get("routes", [])) or "—"
    ptype = p["source"].get("product_type") or "—"
    rxcui = ident.get("rxnorm_cui") or "—"
    sec = []

    sec.append(f'''<header class="masthead">
      <p class="kicker">Drug Profile · Mechanism · Label · FAERS</p>
      <h1>{html.escape(drug.capitalize())}{alias}</h1>
      <div class="meta">
        <div><span>Drug class</span><b>{html.escape(str(p.get("drug_class") or "—"))}</b></div>
        <div><span>Routes</span><b>{html.escape(routes)}</b></div>
        <div><span>Product type</span><b>{html.escape(str(ptype))}</b></div>
        <div><span>RxNorm CUI</span><b>{html.escape(str(rxcui))}</b></div>
        <div class="full"><span>Approved indications (label)</span><b>{html.escape(", ".join(p.get("indications", [])) or "—")}</b></div>
      </div>
    </header>''')

    def block(heading, zh, inner):
        return f'<section class="card"><div class="card-head"><h2>{heading} <span class="zh">{zh}</span></h2></div>{inner}</section>'

    # 使用方法 — what it is used FOR (indications), not how much (that's Dosage)
    raw_inds = p.get("indications") or []
    inds, seen = [], set()
    for x in raw_inds:
        if _is_tabular(x):                               # drop a flattened-table row
            continue
        cx = _clean_label(x)
        if cx and cx.lower() not in seen:
            seen.add(cx.lower()); inds.append(cx)
    inds = inds[:8]
    ind_txt = p.get("indications_text")
    if inds or ind_txt:
        bits = [f'<p class="muted">Routes: {html.escape(routes)}</p>']
        if inds:
            bits.append(f'<p><b>Approved for:</b> {html.escape(", ".join(inds))}.</p>')
        if ind_txt:
            bits.append(_collapsible(ind_txt))
        inner = "".join(bits) + _label_ref(c, "Indications and Usage", p)
    else:
        inner = f'<p class="muted">Routes: {html.escape(routes)}</p>' \
                '<p class="block">Official label not loaded — indications unavailable.</p>'
    sec.append(block("What it is used for", "使用方法", inner))

    # 劑量 — dosage (label text, with validation state)
    rules = p.get("dose_rules", [])
    if rules:
        items = []
        for r in rules:
            raw = r.get("dose_text_verbatim") or ""
            ind = r.get("indication") or "general"
            validated = r.get("validation", {}).get("validated") or r.get("requires_human_validation") is False
            s = r.get("structured_extraction", {}) or {}
            sline = ""
            if validated and (s.get("dose_value") or s.get("frequency")):
                bits = " ".join(str(x) for x in [s.get("dose_value"), s.get("dose_unit")] if x)
                fr = f', {s.get("frequency")}' if s.get("frequency") else ""
                mx = f' (max {s.get("maximum")})' if s.get("maximum") else ""
                sline = f'<span class="dose">{html.escape(bits)}{html.escape(fr)}{html.escape(mx)}</span> '
            tag = '<span class="ok-tag">human-validated</span>' if validated \
                else '<span class="warn-tag">label text only — not validated for patient-specific use</span>'
            label_block = _collapsible(raw, lead_chars=150) if raw else '<p class="block">—</p>'
            items.append(f'<li><b>{html.escape(ind)}.</b> {sline}{tag}'
                         f'<div class="muted-label">label text{_label_ref(c, "Dosage and Administration", p)}:</div>'
                         f'{label_block}</li>')
        inner = f'<ol class="claims">{"".join(items)}</ol>'
    else:
        inner = '<p class="block">No label dose rule loaded.</p>'
    sec.append(block("Dosage", "劑量", inner))

    # 副作用 — adverse effects (label)
    if p.get("adverse_reactions_text"):
        inner = _collapsible(p["adverse_reactions_text"]) + _label_ref(c, "Adverse Reactions", p)
    else:
        inner = '<p class="block">Official label not loaded — adverse reactions unavailable.</p>'
    sec.append(block("Adverse effects", "副作用", inner))

    # 會改變哪些人體數據 — body-state effects (internal mechanism map, estimated)
    effs = p.get("body_state_changes", [])
    if effs:
        mn = c.add("Internal mechanism map (drug_mechanisms / state_effects)",
                   section=(p.get("rationale") or None), note="estimated mechanism, not a clinical source", external=False)
        rows = "".join(
            f'<li>{html.escape(e["organ"])}.{html.escape(e["variable"])} '
            f'{"↓" if (e.get("direction")=="low" or e.get("max_delta",0)<0) else "↑"} '
            f'<span class="muted">({html.escape(e.get("effect_type","primary"))})</span></li>'
            for e in effs)
        inner = f'<ul class="targets">{rows}</ul><p class="muted">Direction of change this drug exerts on modeled organ-state variables{c.sup(mn)}.</p>'
    else:
        inner = '<p class="block">No modeled body-state effects.</p>'
    sec.append(block("Body-state effects", "會改變哪些人體數據", inner))

    # 不好的紀錄 — FAERS signals
    sig = p.get("faers_signals", [])
    if sig:
        fn = c.add("FDA FAERS — Adverse Event Reporting System (data via openFDA)",
                   section="post-marketing spontaneous reports",
                   note="official FDA data; signal only — a report does NOT mean the drug caused the event; may lag months",
                   url=_FAERS_PAGE)
        rows = ", ".join(f'{html.escape(s["event"])} ({s.get("report_count","?")})' for s in sig[:8])
        inner = f'<p>Most-reported events: {rows}.{c.sup(fn)}</p>'
    else:
        inner = '<p class="block">No FAERS signal loaded (run the fetch to populate).</p>'
    sec.append(block("Bad real-world record", "不好的紀錄", inner))

    # 哪些病人不建議使用 — who should not use it
    items = []
    def _row(label, text, section):
        return (f'<li><b>{label}.</b> {_collapsible(text, lead_chars=170)}'
                f'{_label_ref(c, section, p)}</li>')
    if p.get("contraindications_text"):
        items.append(_row("Contraindications", p["contraindications_text"], "Contraindications"))
    if p.get("warnings_text"):
        items.append(_row("Warnings &amp; precautions", p["warnings_text"], "Warnings and Precautions"))
    if p.get("special_populations_text"):
        items.append(_row("Specific populations", p["special_populations_text"], "Use in Specific Populations"))
    if p.get("interactions_text"):
        items.append(_row("Drug interactions", p["interactions_text"], "Drug Interactions"))
    inner = f'<ol class="claims">{"".join(items)}</ol>' if items \
        else '<p class="block">Official label not loaded — contraindications/cautions cannot be shown.</p>'
    sec.append(block("Who should not use it", "哪些病人不建議使用", inner))

    disclaimer = ("Reference information only. Label text is shown verbatim with source; doses are not "
                  "individualized for any patient. Mechanism data is estimated. FAERS is signal, not "
                  "causation. Clinical decisions require a licensed clinician.")
    return _HEAD + _CSS + _PROFILE_CSS + _MID + "".join(sec) + f'''
      <section class="references"><h2>References</h2>{c.html()}</section>
      <footer class="disclaimer"><p>{disclaimer}</p></footer>
    </main>{_JS}</body></html>'''


_PROFILE_CSS = """<style>
.card-head h2 .zh{font-family:"Newsreader",serif;font-weight:400;font-size:18px;color:var(--muted);margin-left:8px}
.zh{font-size:16px}
.ok-tag{font-family:"JetBrains Mono",monospace;font-size:10px;color:#fffdf8;background:var(--ok);padding:2px 7px;border-radius:2px;white-space:nowrap}
.warn-tag{font-family:"JetBrains Mono",monospace;font-size:10px;color:#fffdf8;background:var(--caution);padding:2px 7px;border-radius:2px}
.card .targets{margin:4px 0 0;padding-left:18px}
.card .targets li{font-family:"JetBrains Mono",monospace;font-size:14px;padding:2px 0}
</style>"""
