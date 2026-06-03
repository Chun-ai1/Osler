"""
validate_doses.py — human validation of label-extracted dose rules.

Fetched dose rules are stored verbatim with validation_status=extracted_unverified,
which the engine treats as "label text only, patient-specific dosing blocked". This
tool lets a qualified human read each verbatim dose, enter the structured values
(value / unit / frequency / maximum) for a specific indication, and sign it off.
Only then does the engine unlock a patient-specific dose for that indication.

Modes:
  python3 validate_doses.py                      # interactive
  python3 validate_doses.py --list               # list pending rules
  python3 validate_doses.py --validator "Dr X"   # interactive, signed
  python3 validate_doses.py --batch v.json        # apply scripted validations
  python3 validate_doses.py --reset aspirin       # revert a drug to unverified

A timestamped .bak of the data file is written before any change.
"""
from __future__ import annotations
import argparse
import json
import shutil
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def load(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def save(path: str, data: Dict[str, Any], backup: bool = True):
    p = Path(path)
    if backup and p.exists():
        shutil.copy(p, p.with_suffix(p.suffix + f".{datetime.now():%Y%m%d%H%M%S}.bak"))
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def _rule_validated(rule: Dict[str, Any]) -> bool:
    return bool(rule.get("validation", {}).get("validated")
                or rule.get("requires_human_validation") is False)


def pending_rules(data: Dict[str, Any], drug: Optional[str] = None) -> List[Tuple[str, int, Dict]]:
    out = []
    for name, entry in data.get("drugs", {}).items():
        if drug and name != drug.lower():
            continue
        for i, rule in enumerate(entry.get("dose_rules", [])):
            if rule.get("dose_text_verbatim") and not _rule_validated(rule):
                out.append((name, i, rule))
    return out


def validate_rule(data: Dict[str, Any], drug: str, rule_index: int, structured: Dict[str, Any],
                  indication: Optional[str] = None, route: Optional[str] = None,
                  validator: str = "", notes: str = "") -> None:
    entry = data["drugs"][drug]
    rule = entry["dose_rules"][rule_index]
    rule["structured_extraction"] = {
        "dose_value": structured.get("dose_value"),
        "dose_unit": structured.get("dose_unit"),
        "frequency": structured.get("frequency"),
        "maximum": structured.get("maximum"),
    }
    if indication:
        rule["indication"] = indication
    if route:
        rule["route"] = route
    rule["requires_human_validation"] = False
    rule["validation"] = {"validated": True, "validated_by": validator or "unspecified",
                          "validated_at": date.today().isoformat(), "notes": notes}
    # entry is human_validated once at least one rule is signed off
    if any(_rule_validated(r) for r in entry.get("dose_rules", [])):
        entry["validation_status"] = "human_validated"


def reset_drug(data: Dict[str, Any], drug: str) -> None:
    entry = data["drugs"].get(drug.lower())
    if not entry:
        return
    for rule in entry.get("dose_rules", []):
        rule["requires_human_validation"] = True
        rule.pop("validation", None)
        rule["structured_extraction"] = {"dose_value": None, "dose_unit": None,
                                         "frequency": None, "maximum": None}
    entry["validation_status"] = "extracted_unverified"


def apply_batch(data: Dict[str, Any], items: List[Dict[str, Any]]) -> int:
    """items: [{drug, indication?, dose_value, dose_unit, frequency, maximum, route?, validator?, notes?}]"""
    n = 0
    for it in items:
        drug = it["drug"].lower()
        entry = data.get("drugs", {}).get(drug)
        if not entry or not entry.get("dose_rules"):
            print(f"  ! {drug}: no dose rule to validate"); continue
        # match rule by indication if given, else first rule
        idx = 0
        if it.get("indication"):
            for i, r in enumerate(entry["dose_rules"]):
                if (r.get("indication") or "").lower() == it["indication"].lower():
                    idx = i; break
        validate_rule(data, drug, idx,
                      structured={k: it.get(k) for k in ("dose_value", "dose_unit", "frequency", "maximum")},
                      indication=it.get("indication"), route=it.get("route"),
                      validator=it.get("validator", ""), notes=it.get("notes", ""))
        n += 1
        print(f"  ✓ validated {drug} (rule {idx}) for '{it.get('indication') or entry['dose_rules'][idx].get('indication')}'")
    return n


def _ask(prompt: str, default: Optional[str] = None) -> Optional[str]:
    d = f" [{default}]" if default else ""
    v = input(f"  {prompt}{d}: ").strip()
    return v or default


def run_interactive(path: str, validator: str, drug: Optional[str] = None):
    data = load(path)
    todo = pending_rules(data, drug)
    if not todo:
        print("Nothing pending. (All loaded dose rules are validated.)"); return
    print(f"{len(todo)} dose rule(s) pending validation.\n")
    changed = 0
    for name, idx, rule in todo:
        entry = data["drugs"][name]
        ident = entry.get("identifiers", {})
        src = entry.get("source", {})
        print("=" * 72)
        print(f"DRUG: {name}  ({ident.get('product_type') or 'product type unknown'})")
        print(f"  set_id: {src.get('set_id')}   route: {rule.get('route')}")
        print(f"  indication (rule): {rule.get('indication')}")
        print(f"  VERBATIM LABEL DOSE TEXT:\n    {rule.get('dose_text_verbatim')}\n")
        action = _ask("validate this rule? (y/skip/quit)", "skip")
        if action == "quit":
            break
        if action != "y":
            continue
        indication = _ask("indication this dose applies to", rule.get("indication") or "")
        dose_value = _ask("dose_value (number; blank if N/A)")
        dose_unit = _ask("dose_unit (e.g. mg, mcg, tablet)")
        frequency = _ask("frequency (e.g. q4-6h)")
        maximum = _ask("maximum (e.g. 8 inhalations/day)")
        route = _ask("route", rule.get("route") or "")
        notes = _ask("notes (optional)")
        validate_rule(data, name, idx,
                      structured={"dose_value": dose_value, "dose_unit": dose_unit,
                                  "frequency": frequency, "maximum": maximum},
                      indication=indication, route=route, validator=validator, notes=notes)
        changed += 1
        print("  ✓ signed off as human_validated\n")
    if changed:
        save(path, data)
        print(f"Saved {changed} validation(s) to {path} (backup written).")
    else:
        print("No changes.")


def main():
    ap = argparse.ArgumentParser(description="Validate label-extracted dose rules.")
    ap.add_argument("--path", default="drug_clinical_data.json")
    ap.add_argument("--validator", default="")
    ap.add_argument("--drug", default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--batch", default=None, help="JSON file with a list of validations")
    ap.add_argument("--reset", default=None, help="revert a drug to extracted_unverified")
    a = ap.parse_args()

    if a.list:
        for name, idx, rule in pending_rules(load(a.path), a.drug):
            print(f"PENDING  {name} [rule {idx}] indication={rule.get('indication')}  "
                  f"text={(rule.get('dose_text_verbatim') or '')[:60]!r}")
        return
    if a.reset:
        data = load(a.path); reset_drug(data, a.reset); save(a.path, data)
        print(f"reset {a.reset} → extracted_unverified"); return
    if a.batch:
        data = load(a.path)
        n = apply_batch(data, json.loads(Path(a.batch).read_text()))
        if n: save(a.path, data)
        print(f"applied {n} validation(s)"); return
    run_interactive(a.path, a.validator, a.drug)


if __name__ == "__main__":
    main()
