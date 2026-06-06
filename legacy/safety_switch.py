# safety_switch.py
import re
from typing import Dict, Any, Tuple, List

# unified modes: strict / normal / dev / off
DEFAULT_SAFE_MODE = "strict"

_DOSE_PATTERNS = [
    r"\b\d+(\.\d+)?\s?(mg|g|mcg|µg|ml|mL|units|IU)\b",
    r"\b\d+\s?(times|x)\s?(a|per)?\s?(day|daily)\b",
    r"\b(bid|tid|qid|q\d+h|qhs|hs)\b",
    r"\b(for|x)\s?\d+\s?(days?|weeks?)\b",
]

_RX_TRIGGER_WORDS = [
    "amoxicillin", "azithromycin", "doxycycline", "ciprofloxacin",
    "antibiotic", "steroid", "prednisone",
    "prescription", "rx",
]


def normalize_mode(safe_mode: str) -> str:
    m = (safe_mode or "").strip().lower()
    aliases = {
        "dev": "dev",
        "debug": "dev",

        "off": "off",
        "none": "off",
        "0": "off",
        "false": "off",

        "strict": "strict",
        "prod": "strict",
        "production": "strict",

        "normal": "normal",
        "balanced": "normal",
        "standard": "normal",   # ✅ treat "standard" as "normal"
    }
    return aliases.get(m, "strict")  # unknown -> strict


def build_safety_event(result: Dict[str, Any], safe_mode: str) -> Dict[str, Any]:
    triage = (result or {}).get("triage") or {}
    red_flags = []

    if isinstance(triage, dict):
        red_flags = (triage.get("red_flags") or [])

    if "red_flag_block" in (result or {}):
        red_flags = (result.get("red_flag_block") or {}).get("red_flags", red_flags) or red_flags

    allow_pills = bool((result or {}).get("allow_pills", False))
    mech_risk = triage.get("mechanism_risk") if isinstance(triage, dict) else None

    return {
        "safe_mode": normalize_mode(safe_mode),
        "triage_level": (triage.get("level") if isinstance(triage, dict) else None),
        "mechanism_risk": mech_risk,
        "red_flags": red_flags,
        "allow_pills": allow_pills,
        "rx_possible": bool((result or {}).get("rx_possible", False)),
        "safety_block_reason": (result or {}).get("safety_block_reason", ""),
    }


def hard_block_decision(result: Dict[str, Any], safe_mode: str) -> Tuple[bool, str]:
    """
    True = should hard block (clear pills / force urgent guidance)
    """
    m = normalize_mode(safe_mode)

    triage = (result or {}).get("triage") or {}
    level = (triage.get("level") or "").upper() if isinstance(triage, dict) else ""
    mech_risk = triage.get("mechanism_risk") if isinstance(triage, dict) else None

    # red flags
    if "red_flag_block" in (result or {}):
        red_flags = (result.get("red_flag_block") or {}).get("red_flags", []) or []
    else:
        red_flags = (triage.get("red_flags") or []) if isinstance(triage, dict) else []

    # 1) Always hard-block if clearly urgent
    if level in ("EMERGENCY", "URGENT"):
        return True, f"triage={level}"
    if red_flags:
        return True, "red_flags_present"

    # 2) Mechanism risk threshold differs by mode
    try:
        if mech_risk is not None:
            r = float(mech_risk)
            if m == "strict" and r >= 0.7:
                return True, "high_mechanism_risk(strict>=0.7)"
            if m == "normal" and r >= 0.8:
                return True, "high_mechanism_risk(normal>=0.8)"
            if m == "dev" and r >= 0.9:
                return True, "high_mechanism_risk(dev>=0.9)"
    except Exception:
        pass

    # 3) strict is conservative: if pipeline says no pills, block
    if m == "strict" and not bool((result or {}).get("allow_pills", False)):
        return True, "strict_mode_no_pills"

    # normal/dev: do NOT block just because allow_pills is false
    # (they are meant to be less strict)
    return False, ""


def sanitize_answer(text: str, result: Dict[str, Any], safe_mode: str) -> Tuple[str, List[str]]:
    """
    Post-LLM: remove dosing patterns & RX directives depending on mode.
    """
    m = normalize_mode(safe_mode)

    if m == "dev":
        return text, []

    if not text:
        return text, []

    # off: no changes
    if m == "off":
        return text, []

    removed: List[str] = []
    out = text

    # dev/normal/strict: redact dosing patterns (still a good idea)
    # (If you truly want dev to keep dosing, you can skip this block when m=="dev")
    for pat in _DOSE_PATTERNS:
        new_out, n = re.subn(pat, "[REDACTED]", out, flags=re.IGNORECASE)
        if n > 0:
            removed.append(f"dose_pattern:{pat}")
        out = new_out

    # strict: replace Rx keywords more aggressively
    if m == "strict":
        lowered = out.lower()
        if any(w in lowered for w in _RX_TRIGGER_WORDS):
            out = re.sub(r"(amoxicillin|azithromycin|doxycycline|ciprofloxacin)", "[PRESCRIPTION_MED]", out, flags=re.I)
            out = re.sub(r"\bantibiotics?\b", "prescription medication", out, flags=re.I)
            removed.append("rx_terms_replaced(strict)")

    # normal/dev: only soften explicit antibiotic names (less strict)
    if m in ("normal", "dev"):
        out2 = re.sub(r"(amoxicillin|azithromycin|doxycycline|ciprofloxacin)", "[PRESCRIPTION_MED]", out, flags=re.I)
        if out2 != out:
            removed.append(f"rx_terms_replaced({m})")
            out = out2

    # hard block => append safety tail
    blocked, reason = hard_block_decision(result, safe_mode)
    if blocked:
        safety_tail = (
            "\n\nSafety note: If symptoms worsen, you have severe pain, trouble breathing, confusion, "
            "fainting, or persistent high fever, seek urgent medical care."
        )
        out = out.strip() + safety_tail
        removed.append(f"hard_block_append:{reason}")

    return out, removed


def enforce_medication_gate(result: Dict[str, Any], safe_mode: str) -> Dict[str, Any]:
    """
    Pre-LLM: gate medication list based on risk.
    """
    m = normalize_mode(safe_mode)
    result = dict(result or {})

    if m == "dev":
        result["suppress_new_medications"] = False
        result["allow_pills"] = True
        return result

    # off: do nothing
    if m == "off":
        result["suppress_new_medications"] = False
        return result

    # dev: do not clear pills unless truly hard-block
    # normal/strict: clear pills when blocked
    blocked, reason = hard_block_decision(result, safe_mode)
    if blocked:
        result["pills"] = ""
        result["pill_list"] = []
        result["allow_pills"] = False
        result["suppress_new_medications"] = True
        result["safety_block_reason"] = reason
        return result

    # not blocked
    if m in ("dev", "normal"):
        # allow pills if pipeline provided any; don't force-clear
        result["suppress_new_medications"] = bool(result.get("suppress_new_medications", True))
        return result

    # strict and not blocked: still respect existing allow_pills flag
    return result
