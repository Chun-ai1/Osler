"""
case_parser.py — turn a free-text clinical case into structured fields the
symbolic engine can consume: PatientProfile fields + a resolved indication.

Two paths:
  • parse_llm(text, key) — when an API key is present, the agent asks the LLM to
    return strict JSON. Highest quality on messy notes.
  • parse_rules(text)    — deterministic regex/keyword extraction, no key needed.
    Good enough for the preset cases and typical short notes; always the fallback.

The output dict is intentionally the same shape demo_app._build_patient() expects.
"""
from __future__ import annotations
import json
import re
from typing import Dict, Optional

import case_targets
import llm_client

_FIELDS = ["age", "sex", "weight_kg", "egfr", "hepatic_status", "allergies",
           "current_medications", "symptoms", "indication", "vitals", "labs"]

_LLM_SYSTEM = (
    "You extract structured data from a clinical case note. Return ONLY minified JSON "
    "with these keys: age (int|null), sex ('M'|'F'|null), weight_kg (number|null), "
    "egfr (number|null), hepatic_status ('normal'|'mild'|'moderate'|'severe'), "
    "allergies (string[]), current_medications (string[]), symptoms (string[]), "
    "vitals (object with optional sbp, heart_rate, spo2 numbers), "
    "labs (object with optional potassium number), "
    "indication (string: the primary clinical problem to treat). "
    "Use lowercase for drug/allergy/symptom names. No prose, no code fences."
)


def parse_llm(text: str, api_key: Optional[str] = None,
              provider: Optional[str] = None) -> Optional[Dict]:
    ok, raw = llm_client.complete_json(_LLM_SYSTEM, text, api_key, provider)
    if not ok:
        return None
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw
    try:
        d = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
    except Exception:
        return None
    # Normalize indication through the same alias map the engine uses.
    canon = case_targets.resolve(d.get("indication") or "")
    if canon:
        d["indication"] = canon
    return _coerce(d)


# ── deterministic fallback ─────────────────────────────────────────────────
_AGE_SEX = re.compile(r"\b(\d{1,3})\s*[- ]?\s*(?:y/?o|yo|year[s]?[- ]?old|岁)?\s*([MFmf])?\b")
_AGE = re.compile(r"\b(\d{1,3})\s*(?:y/?o|yo|year[s]?[- ]?old|岁)\b", re.I)
_BP = re.compile(r"\b(?:bp|blood pressure)\s*[:=]?\s*(\d{2,3})\s*/\s*(\d{2,3})", re.I)
_SBP = re.compile(r"\bsbp\s*[:=]?\s*(\d{2,3})", re.I)
_HR = re.compile(r"\b(?:hr|heart rate|pulse)\s*[:=]?\s*(\d{2,3})", re.I)
_SPO2 = re.compile(r"\b(?:spo2|sao2|o2 sat|sat)\s*[:=]?\s*(\d{2,3})", re.I)
_EGFR = re.compile(r"\b(?:egfr|gfr)\s*[:=]?\s*(\d{1,3})", re.I)
_WT = re.compile(r"\b(\d{2,3})\s*kg\b", re.I)
_K = re.compile(r"\b(?:k\+?|potassium)\s*[:=]?\s*(\d(?:\.\d)?)", re.I)


def _after(text: str, *keywords: str) -> list:
    """Grab the comma/space list following any keyword up to sentence end."""
    for kw in keywords:
        m = re.search(rf"{kw}\s*[:\-]?\s*([^.\n;]+)", text, re.I)
        if m:
            chunk = m.group(1)
            parts = re.split(r"[,/]| and ", chunk)
            out = [p.strip().lower() for p in parts if p.strip() and len(p.strip()) < 40]
            if out:
                return out[:8]
    return []


def parse_rules(text: str) -> Dict:
    t = text or ""
    d: Dict = {"hepatic_status": "normal", "allergies": [], "current_medications": [],
               "symptoms": [], "vitals": {}, "labs": {}}

    m = _AGE_SEX.search(t)
    if m:
        d["age"] = int(m.group(1))
        if m.group(2):
            d["sex"] = m.group(2).upper()
    elif _AGE.search(t):
        d["age"] = int(_AGE.search(t).group(1))
    if "sex" not in d:
        if re.search(r"\bmale\b", t, re.I):
            d["sex"] = "M"
        elif re.search(r"\bfemale\b", t, re.I):
            d["sex"] = "F"

    if _WT.search(t):
        d["weight_kg"] = float(_WT.search(t).group(1))
    if _EGFR.search(t):
        d["egfr"] = float(_EGFR.search(t).group(1))
    if _BP.search(t):
        d["vitals"]["sbp"] = float(_BP.search(t).group(1))
    elif _SBP.search(t):
        d["vitals"]["sbp"] = float(_SBP.search(t).group(1))
    if _HR.search(t):
        d["vitals"]["heart_rate"] = float(_HR.search(t).group(1))
    if _SPO2.search(t):
        d["vitals"]["spo2"] = float(_SPO2.search(t).group(1))
    if _K.search(t):
        d["labs"]["potassium"] = float(_K.search(t).group(1))

    d["allergies"] = _after(t, "allergies", "allergic to", "allergy")
    d["current_medications"] = _after(t, "current medications", "medications", "meds",
                                      "taking", "on home")
    d["symptoms"] = _after(t, "symptoms", "presents with", "complains of", "c/o", "reports")

    # Indication: scan for any known alias, else look for an explicit "for <x>".
    d["indication"] = _detect_indication(t)
    return _coerce(d)


def _detect_indication(text: str) -> str:
    low = text.lower()
    best = None
    for alias, canon in case_targets._ALIAS_TO_CANON.items():
        if re.search(r"\b" + re.escape(alias) + r"\b", low):
            # Prefer the longest alias match (more specific).
            if best is None or len(alias) > best[0]:
                best = (len(alias), canon)
    return best[1] if best else ""


def _coerce(d: Dict) -> Dict:
    """Ensure types/keys are clean and lists are lowercased."""
    out = {k: d.get(k) for k in _FIELDS}
    out["hepatic_status"] = d.get("hepatic_status") or "normal"
    for lk in ("allergies", "current_medications", "symptoms"):
        v = d.get(lk) or []
        out[lk] = [str(x).strip().lower() for x in v if str(x).strip()]
    out["vitals"] = {k: float(v) for k, v in (d.get("vitals") or {}).items()
                     if v is not None}
    out["labs"] = {k: float(v) for k, v in (d.get("labs") or {}).items() if v is not None}
    out["indication"] = d.get("indication") or ""
    return out


def parse(text: str, api_key: Optional[str] = None, provider: Optional[str] = None) -> Dict:
    """Agent entry: LLM if a key is available, else deterministic rules."""
    if llm_client.available(api_key, provider):
        got = parse_llm(text, api_key, provider)
        if got and (got.get("indication") or got.get("symptoms")):
            got["_parser"] = "llm"
            return got
    out = parse_rules(text)
    out["_parser"] = "rules"
    return out


if __name__ == "__main__":
    demo = ("64 yo M with crushing chest pain and diaphoresis, known acute coronary "
            "syndrome. BP 88/54, HR 112, SpO2 94. eGFR 72. No known allergies. "
            "Taking aspirin at home.")
    print(json.dumps(parse_rules(demo), indent=2))
