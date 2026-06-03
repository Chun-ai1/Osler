"""
render_report.py — Frontend (text) layer of the Pharmacology Reasoning Engine.

Takes the structured output of reasoning_engine.recommend() and renders it as a
human-readable narrative with inline citation markers [n] and a References list.
Every factual claim is tied to its source:

  • dose / contraindication / interaction / warning  → FDA label section + set_id,
    rendered as a clickable DailyMed link  (drugInfo.cfm?setid=...)
  • adverse-event signal                              → openFDA drug/event (FAERS),
    tagged "signal, not causation"
  • mechanism                                         → internal mechanism map
    (drug_mechanisms / state_effects), flagged "estimated, not clinical"

The renderer NEVER invents a dose: if no label-sourced dose is loaded it states
that plainly and cites nothing for the (absent) number.
"""
from __future__ import annotations
import re
from typing import Any, Dict, List, Optional

_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)

_DECISION_ZH = {
    "ok": "可顯示劑量", "caution": "謹慎（需警示）", "adjust": "需依病人調整",
    "avoid": "不建議使用", "emergency": "緊急（不給劑量）",
    "insufficient_data": "資料不足（不給劑量）",
}


def _dailymed(set_id: Optional[str]) -> Optional[str]:
    if set_id and _UUID.match(set_id):
        return f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={set_id}"
    return None


class Citations:
    """Dedup citation registry → inline [n] markers + a references list."""
    def __init__(self):
        self.items: List[Dict[str, Any]] = []
        self._idx: Dict[tuple, int] = {}

    def cite(self, label, section=None, set_id=None, url=None, note=None) -> int:
        key = (label, section, set_id, url, note)
        if key in self._idx:
            return self._idx[key]
        n = len(self.items) + 1
        self._idx[key] = n
        self.items.append({"n": n, "label": label, "section": section,
                           "set_id": set_id, "url": url, "note": note})
        return n

    def render(self) -> List[str]:
        out = []
        for it in self.items:
            parts = [it["label"]]
            if it["section"]:
                parts.append(it["section"])
            if it["set_id"] and _UUID.match(it["set_id"]):
                parts.append(f"set_id={it['set_id']}")
            ref = " — ".join(parts)
            if it["url"]:
                ref += f"  {it['url']}"
            if it["note"]:
                ref += f"  （{it['note']}）"
            out.append(f"[{it['n']}] {ref}")
        return out


def _label_cite(cites: Citations, section: str, evidence: List[Dict[str, Any]]) -> str:
    """Build a citation to an FDA label section using the candidate's set_id."""
    set_id = evidence[0].get("set_id") if evidence else None
    eff = evidence[0].get("effective_date") if evidence else None
    url = _dailymed(set_id)
    note = None
    if set_id and not _UUID.match(set_id):
        note = "尚未抓取仿單，run clinical_data.build_entry()"
    elif eff:
        note = f"label effective {eff}"
    n = cites.cite("FDA 仿單 (openFDA/DailyMed SPL)", section, set_id, url, note)
    return f"[{n}]"


def render(engine_output: Dict[str, Any], drugs_pkpd: Optional[Dict[str, Any]] = None) -> str:
    cites = Citations()
    L: List[str] = []
    drugs_pkpd = drugs_pkpd or {}

    t = engine_output["target_state"]
    pt = engine_output["patient"]
    L.append(f"# 用藥推理報告")
    L.append(f"**目標狀態**：{t['organ']}.{t['variable']} 需往 `{t['direction']}` 推"
             f"　|　**適應症**：{engine_output.get('indication') or '未指定'}")
    flags = "、".join(pt.get("flags", [])) or "無"
    meds = "、".join(pt.get("meds", [])) or "無"
    allg = "、".join(pt.get("allergies", [])) or "無"
    L.append(f"**病人**：年齡 {pt.get('age')}，腎功能 {pt.get('renal')}，"
             f"過敏 {allg}，現有用藥 {meds}，臨床旗標 {flags}")
    L.append("")

    for r in engine_output["candidates"]:
        drug = r["drug"]
        dec = r["safety"]["decision"]
        L.append(f"## {drug} — {_DECISION_ZH.get(dec, dec)}（risk: {r['safety']['risk_level']}）")

        # mechanism (internal, estimated)
        mech_note = "內部機制圖，估計值，非臨床劑量依據"
        rationale = (drugs_pkpd.get(drug, {}).get("rationale") or "").strip()
        mn = cites.cite("內部機制圖 (drug_mechanisms / state_effects)",
                        section=(rationale or None), note=mech_note)
        L.append(f"- **機制**：{r['mechanism'].split('  [')[0]}，機制上與目標狀態相關 [{mn}]。")

        # indication
        ev = r.get("evidence", [])
        sup = r["indication_support"]
        if sup == "label-supported":
            L.append(f"- **適應症**：此用途在仿單中列為核可適應症 {_label_cite(cites, 'Indications and Usage', ev)}。")
        elif sup == "not in label / off-label":
            L.append(f"- **適應症**：仿單未列此用途 → 屬 off-label，需指引／臨床判斷 {_label_cite(cites, 'Indications and Usage', ev)}。")
        else:
            L.append(f"- **適應症**：尚未載入仿單，無法確認是否核可。")

        # dose
        d = r["dose"]
        if r["recommendation_type"] == "label_based_dose_range" and d.get("value"):
            extra = "，".join(x for x in [d.get('route'), d.get('frequency')] if x)
            L.append(f"- **劑量**：仿單 Dosage and Administration 段所載：{d['value']}"
                     f"{('（' + extra + '）') if extra else ''} {_label_cite(cites, 'Dosage and Administration', ev)}。")
        else:
            L.append(f"- **劑量**：無來源劑量，引擎**不生成數字**——需先抓取仿單後才會顯示。")

        # safety reasons
        for rs in r["safety"].get("reasons", []):
            if rs.get("type") == "none":
                continue
            src = rs.get("source")
            if src and "label" in src.lower():
                section = src.split("—")[-1].strip()
                marker = _label_cite(cites, section, ev)
                L.append(f"- **風險**：{rs['message']} {marker}。")
            else:
                L.append(f"- **風險**：{rs['message']}。")

        # FAERS
        sig = r.get("faers_signals", [])
        if sig:
            top = "、".join(f"{s['event']}（{s.get('report_count','?')} 筆）" for s in sig[:5])
            fn = cites.cite("openFDA drug/event (FAERS)", section="真實世界通報",
                            note="僅為關聯訊號，非因果；資料可能延遲數月")
            L.append(f"- **不良事件訊號 (FAERS)**：{top} [{fn}]。")

        L.append(f"- **結論**：{r['final_answer']}")
        if r["safety"].get("missing_patient_data"):
            L.append(f"  （缺少病人資料：{', '.join(r['safety']['missing_patient_data'])}，無法個人化）")
        L.append("")

    L.append("---")
    L.append("## 參考來源")
    L.extend(cites.render())
    L.append("")
    L.append(f"> {engine_output.get('_disclaimer', '')}")
    return "\n".join(L)


if __name__ == "__main__":
    import json
    from pathlib import Path
    import reasoning_engine as RE
    from patient_profile import PatientProfile

    drugs_pkpd = RE._load("drugs_pkpd.json")["drugs"]
    clinical = RE._load("drug_clinical_data.json")["drugs"]
    target = {"organ": "lung", "variable": "bronchospasm", "direction": "low"}
    patient = PatientProfile(age=55, weight_kg=80, egfr=90,
                             vitals={"heart_rate": 135}, current_medications=["metoprolol"])

    print("══════════ A. scaffold (尚未抓取仿單) ══════════\n")
    out = RE.recommend(patient, target, "asthma", drugs_pkpd, clinical)
    rep_a = render(out, drugs_pkpd)
    print(rep_a)

    print("\n\n══════════ B. 模擬已抓取 albuterol 仿單後 ══════════\n")
    clinical["albuterol"]["source"]["set_id"] = "01c77e6a-6381-4005-b647-481bbcd442aa"  # 範例 UUID
    clinical["albuterol"]["source"]["effective_date"] = "2024-01-15"
    clinical["albuterol"]["dosing"][0].update({
        "dose_text": "（仿單 Dosage and Administration 原文）", "route": "inhalation",
        "frequency": "（仿單原文）", "confidence": "label"})
    clinical["albuterol"]["faers_signals"] = [
        {"event": "tremor", "report_count": 1234, "source": "FAERS", "confidence": "signal_only"},
        {"event": "tachycardia", "report_count": 890, "source": "FAERS", "confidence": "signal_only"}]
    out2 = RE.recommend(patient, target, "asthma", drugs_pkpd, clinical)
    rep_b = render(out2, drugs_pkpd)
    print(rep_b)

    Path("report_sample.md").write_text(rep_a + "\n\n\n" + rep_b, encoding="utf-8")
