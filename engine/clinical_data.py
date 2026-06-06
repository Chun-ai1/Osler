"""
clinical_data.py — Label layer. Official sources only, with provenance.

build_entry() pipeline:  canonicalize → fetch label → select route/form →
extract sections → structured (unvalidated) dose rules → mark extracted_unverified.

Dose is stored as VERBATIM label text inside a dose_rule whose structured fields
(value/unit/frequency) start null and require_human_validation=True. The engine
will not produce a patient-specific dose until validation_status == "human_validated".

  openFDA label  api.fda.gov/drug/label.json   |  FAERS  api.fda.gov/drug/event.json (signal only)
  RxNorm/RxNav   rxnav.nlm.nih.gov/REST        (name → RxCUI)
No API key required. parse_label_record() is pure / offline-testable.
"""
from __future__ import annotations
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from typing import Any, Dict, List, Optional

from drug_identity import canonicalize_drug_name, CANONICAL

LABEL_URL = "https://api.fda.gov/drug/label.json"
EVENT_URL = "https://api.fda.gov/drug/event.json"
RXNORM_URL = "https://rxnav.nlm.nih.gov/REST/rxcui.json"

_SECTIONS = {
    "indications_and_usage": "indications_and_usage",
    "dosage_and_administration": "dosage_and_administration",
    "contraindications": "contraindications",
    "warnings_and_cautions": "warnings_and_precautions",
    "warnings_and_precautions": "warnings_and_precautions",
    "boxed_warning": "boxed_warning",
    "adverse_reactions": "adverse_reactions",
    "drug_interactions": "drug_interactions",
    "use_in_specific_populations": "use_in_specific_populations",
    "pregnancy": "pregnancy",
}


def _get(url, timeout=6, retries=2):
    import time as _t
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "pharm-reasoning-engine"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return {}                       # no match for this query
            if e.code in (429, 500, 502, 503) and attempt < retries - 1:
                _t.sleep(2 ** attempt)          # backoff on rate-limit / transient
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            if attempt < retries - 1:
                _t.sleep(2 ** attempt); continue
            raise
    return {}


def _get_with_next(url, timeout=30, retries=4):
    """Like _get, but also returns the rel="next" URL from the Link header (openFDA
    search_after deep pagination). Returns (data_dict, next_url_or_None)."""
    import time as _t, re as _re
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "pharm-reasoning-engine"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read().decode("utf-8"))
                link = r.headers.get("Link", "") or ""
            nxt = None
            for part in link.split(","):
                if 'rel="next"' in part.lower():
                    m = _re.search(r'<([^>]+)>', part)
                    if m:
                        nxt = m.group(1)
            return data, nxt
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return {}, None
            if e.code in (429, 500, 502, 503) and attempt < retries - 1:
                _t.sleep(2 ** attempt); continue
            raise
        except (urllib.error.URLError, TimeoutError):
            if attempt < retries - 1:
                _t.sleep(2 ** attempt); continue
            raise
    return {}, None


_SALT_WORDS = {
    "hydrochloride", "hcl", "sulfate", "sulphate", "sodium", "potassium", "calcium",
    "tartrate", "succinate", "besylate", "besilate", "mesylate", "mesilate", "tosylate",
    "maleate", "fumarate", "citrate", "phosphate", "acetate", "hydrobromide", "bromide",
    "bitartrate", "dihydrochloride", "monohydrate", "valerate", "propionate", "decanoate",
    "palmitate", "nitrate", "gluconate", "lactate", "hyclate", "disodium", "dipropionate",
    "xinafoate", "fumarate", "sodium succinate",
}


def base_name(name: str) -> str:
    """Collapse salt/ester variants to the base ingredient: 'metoprolol succinate' -> 'metoprolol'."""
    n = re.sub(r'[^a-z0-9; ]', ' ', (name or "").lower())
    n = re.sub(r'\s+', ' ', n).strip()
    toks = n.split()
    while len(toks) > 1 and toks[-1] in _SALT_WORDS:   # strip trailing salt words
        toks.pop()
    return " ".join(toks)


# words that can legitimately follow an ingredient in a single-drug name (salt/chemical/form)
_MODIFIERS = _SALT_WORDS | {
    "acid", "oxide", "dioxide", "hydroxide", "carbonate", "bicarbonate", "chloride", "peroxide",
    "human", "recombinant", "regular", "glargine", "lispro", "aspart", "detemir", "degludec",
}


def clean_drug_name(name: str) -> str:
    """Strip leading strength numbers and tidy: '5 lidocaine' -> 'lidocaine', '20 benzocaine' -> 'benzocaine'."""
    n = (name or "").lower()
    n = re.sub(r'^\s*(?:\d[\d.,%/]*\s+)+', '', n)       # leading numeric/strength tokens
    n = re.sub(r'[^a-z0-9 ]', ' ', n)
    return re.sub(r'\s+', ' ', n).strip()


def is_single_ingredient(name: str) -> bool:
    """A real single-drug name: one word, or two words where the second is a salt/chemical modifier."""
    toks = name.split()
    if len(toks) == 1:
        return True
    if len(toks) == 2 and toks[1] in _MODIFIERS:
        return True
    return False


# cosmetic / OTC product-form words that signal a non-drug product (not a combination drug)
_JUNK_WORDS = {
    "spray", "kit", "deodorant", "antiperspirant", "sunscreen", "sunblock", "spf", "wash",
    "wipe", "wipes", "scrub", "shampoo", "soap", "cleanser", "moisturizer", "serum", "balm",
    "foam", "rinse", "sanitizer", "remedy", "lotion", "powder", "mask", "toner", "conditioner",
}


def is_noise(name: str) -> bool:
    """Homeopathic ingredient lists (very long), cosmetic/OTC product names, or garbage
    name fragments — not real drugs. Does NOT flag legitimate 2-5 word combination drugs."""
    toks = name.split()
    if len(toks) >= 6:                                   # homeopathic ingredient lists
        return True
    if any(t in _JUNK_WORDS for t in toks):              # cosmetic/OTC product words
        return True
    if any(t.isdigit() for t in toks):                   # stray strength fragments: "as 3", "burn 5"
        return True
    if len(name.replace(" ", "")) <= 2:                  # too short to be a real drug name
        return True
    return False


# High-precision homeopathic markers: exclusive terms (HPUS, homeopathic, globule, succussion)
# plus homeopathic dose-form phrases ("Mix N drops in water", "Take N Pellets by mouth").
# Deliberately NOT matching "conditions listed above" alone (it appears inside real labels) or
# bare potencies like "30C/6X" (collide with storage temps "store at 25 C").
_HOMEO = re.compile(
    r'\bHPUS\b|homeopath|\bglobules?\b|succuss|'
    r'(?:mix|take|use)\s+\d[\d\s./-]*drops\s+(?:in|under|by|on)|'
    r'take\s+\d[\d\s or-]*pellets|\d+\s+pellets\s+(?:by\s+mouth|under)',
    re.I)


def is_homeopathic(entry) -> bool:
    ls = entry.get("label_sections") or {}
    text = " ".join(str(v) for v in ls.values())[:6000]
    return bool(_HOMEO.search(text))


_HEADER_ECHOES = {
    "indications and usage", "dosage and administration", "dosage", "indications", "purpose",
    "uses", "dosage dosage and administration", "indications and usage indications and usage",
    "warnings", "description", "contraindications", "adverse reactions",
}


def has_real_label(entry) -> bool:
    """False ONLY for header-echo junk — entries whose every non-empty section is just its own
    section title (e.g. 'indications and usage'). Empty/missing labels are KEPT (real drugs that
    simply lack fetched label text, e.g. oxygen, ketotifen)."""
    ls = entry.get("label_sections") or {}
    nonempty = [re.sub(r'\s+', ' ', str(v)).strip().lower() for v in ls.values()
                if re.sub(r'\s+', ' ', str(v)).strip()]
    if not nonempty:
        return True                                      # empty/missing → keep
    return any(t not in _HEADER_ECHOES for t in nonempty)  # junk only if ALL sections are echoes


def is_combination(name: str) -> bool:
    return ";" in (name or "")


def _label_page(api_key, skip, limit=1000):
    params = {"search": "_exists_:openfda.generic_name", "limit": limit, "skip": skip}
    if api_key:
        params["api_key"] = api_key
    return _get(f"{LABEL_URL}?{urllib.parse.urlencode(params)}").get("results", [])


def discover_drug_names(n=1000, api_key=None, include_combos=False, exclude=None) -> List[str]:
    """Discover up to N distinct base drugs from the FDA label dataset, skipping any in `exclude`.
    Uses the count endpoint first (most-common), then paginates label records to go past the
    1,000-bucket cap (so we can reach 2,000+ distinct drugs)."""
    import time as _t
    excl = {base_name(x) for x in (exclude or [])}
    names, seen = [], set(excl)

    def add(term):
        if not term or (not include_combos and is_combination(term)):
            return
        b = base_name(term)
        cb = clean_drug_name(b)
        if is_noise(cb):                                  # homeopathic/cosmetic junk: always skip
            return
        if not include_combos and not is_single_ingredient(cb):
            return
        if b and b not in seen:
            seen.add(b); names.append(b)

    # phase 1 — the most-common generic names (one count call, max 1000 buckets)
    params = {"count": "openfda.generic_name.exact", "limit": 1000}
    if api_key:
        params["api_key"] = api_key
    for row in _get(f"{LABEL_URL}?{urllib.parse.urlencode(params)}").get("results", []):
        add((row.get("term") or "").strip())
        if len(names) >= n:
            break

    # phase 2 — scroll the whole label dataset via search_after (Link header, no 25k cap).
    # NOTE: search_after requires sort and must NOT use skip.
    params = {"search": "_exists_:openfda.generic_name", "limit": 1000, "sort": "effective_time:asc"}
    if api_key:
        params["api_key"] = api_key
    url = f"{LABEL_URL}?{urllib.parse.urlencode(params)}"
    pages, max_pages = 0, 400              # safety cap (~400k records) so a stalled scroll can't loop forever
    while url and len(names) < n and pages < max_pages:
        data, url = _get_with_next(url)
        recs = data.get("results", [])
        if not recs:
            break
        for r in recs:
            for g in (r.get("openfda", {}).get("generic_name") or []):
                add((g or "").strip())
            if len(names) >= n:
                break
        pages += 1
        if pages % 25 == 0:
            print(f"    … scrolled {pages} pages, {len(names)} new so far")
        _t.sleep(0.2 if api_key else 1.5)
    exhausted = "dataset exhausted" if not url else f"stopped at {pages} pages"
    print(f"  discovered {len(names)} new base drugs (count + {pages} search_after pages, {exhausted}; "
          f"excluded {len(excl)} already present)")
    return names[:n]


def fetch_label_candidates(generic_name, api_key=None) -> List[Dict[str, Any]]:
    params = {"search": f'openfda.generic_name:"{generic_name}"', "limit": 5}
    if api_key:
        params["api_key"] = api_key
    return _get(f"{LABEL_URL}?{urllib.parse.urlencode(params)}").get("results", [])


# Bias product selection toward the clinically intended product (a drug can have
# many SPLs: e.g. OTC analgesic aspirin vs cardiac antiplatelet aspirin).
INDICATION_HINTS = {
    "aspirin": "myocardial infarction",
    "nitroglycerin": "angina",
    "heparin": "thrombosis",
    "epinephrine": "anaphylaxis",
    "albuterol": "bronchospasm",
}


def select_label(candidates, required_route=None, indication=None, prefer_rx=True):
    """Pick the most clinically relevant SPL and return (record, selection_meta)."""
    if not candidates:
        return None, {}

    def score(c):
        of = c.get("openfda", {})
        ind_text = " ".join(c.get("indications_and_usage", [])).lower()
        routes = " ".join(of.get("route", [])).lower()
        ptype = " ".join(of.get("product_type", [])).lower()
        s = 0
        if indication and indication.lower() in ind_text:
            s += 5
        if required_route and required_route.lower() in routes:
            s += 2
        if prefer_rx and "prescription" in ptype:
            s += 1
        if c.get("dosage_and_administration"):
            s += 1
        return s

    best = max(candidates, key=score)
    of = best.get("openfda", {})
    ind_text = " ".join(best.get("indications_and_usage", [])).lower()
    meta = {"product_type": (of.get("product_type") or [None])[0],
            "candidates_considered": len(candidates),
            "indication_in_selected_label": bool(indication and indication.lower() in ind_text),
            "indication_hint_used": indication}
    return best, meta


def fetch_faers_signal(generic_name, limit=15, api_key=None) -> List[Dict[str, Any]]:
    params = {"search": f'patient.drug.openfda.generic_name:"{generic_name}"',
              "count": "patient.reaction.reactionmeddrapt.exact"}
    if api_key:
        params["api_key"] = api_key
    try:
        data = _get(f"{EVENT_URL}?{urllib.parse.urlencode(params)}")
    except Exception:
        return []
    return [{"event": r["term"].lower(), "report_count": r["count"],
             "source": "FAERS", "confidence": "signal_only"} for r in data.get("results", [])[:limit]]


def normalize_rxcui(name) -> Optional[str]:
    try:
        ids = _get(f"{RXNORM_URL}?{urllib.parse.urlencode({'name': name})}").get("idGroup", {}).get("rxnormId", [])
        return ids[0] if ids else None
    except Exception:
        return None


def parse_label_record(rec: Dict[str, Any], canonical: Optional[str] = None) -> Dict[str, Any]:
    openfda = rec.get("openfda", {})
    first = lambda k: (openfda.get(k) or [None])[0]
    canonical = canonical or canonicalize_drug_name(first("generic_name") or "")
    ident = CANONICAL.get(canonical, {})

    sections: Dict[str, str] = {}
    for raw_key, our_key in _SECTIONS.items():
        v = rec.get(raw_key)
        if v:
            sections.setdefault(our_key, (" ".join(v) if isinstance(v, list) else str(v)).strip())

    eff = rec.get("effective_time")
    dose_text = sections.get("dosage_and_administration")
    dose_rules = [{
        "indication": None, "population": "adult", "route": first("route"),
        "dose_text_verbatim": dose_text,
        "structured_extraction": {"dose_value": None, "dose_unit": None, "frequency": None, "maximum": None},
        "requires_human_validation": True, "source_section": "Dosage and Administration",
    }] if dose_text else []

    return {
        "source": {"primary": "openFDA / DailyMed (FDA SPL)", "database": "DailyMed",
                   "label_type": "FDA SPL", "set_id": rec.get("set_id"), "spl_id": rec.get("id"),
                   "effective_date": (f"{eff[:4]}-{eff[4:6]}-{eff[6:8]}" if eff and len(eff) >= 8 else eff),
                   "retrieved_at": date.today().isoformat()},
        "identifiers": {"canonical_id": ident.get("canonical_id", canonical),
                        "rxnorm_cui": (openfda.get("rxcui") or [None])[0],
                        "generic_name": canonical, "aliases": ident.get("aliases", []),
                        "substance_name": openfda.get("substance_name", []),
                        "product_type": first("product_type"),
                        "pharm_class": openfda.get("pharm_class_epc", []), "route": openfda.get("route", [])},
        "indications": ident.get("indications", []),
        "label_sections": sections,
        "dose_rules": dose_rules,
        "validation_status": "extracted_unverified",
        "has_boxed_warning": "boxed_warning" in sections,
    }


def _label_identity(k, e):
    """A key identifying the underlying label, so different names for the same SPL collapse."""
    sid = (e.get("source") or {}).get("set_id")
    if sid:
        return ("sid", sid)
    ls = e.get("label_sections") or {}
    da, iu = (ls.get("dosage_and_administration") or ""), (ls.get("indications_and_usage") or "")
    if da or iu:
        return ("txt", da[:300], iu[:300])
    return ("uniq", k)            # nothing to compare on → keep as its own entry


def dedupe_file(path, drop_noise=False, single_only=False):
    """Collapse entries that are the same underlying label (same set_id, or identical dosage/
    indication text) to one best-named entry; leading strength numbers are always stripped.
      drop_noise=True   → also drop homeopathic/cosmetic product names + no-label entries (KEEPS combos)
      single_only=True  → also drop combination products (keep single-ingredient drugs only)"""
    from pathlib import Path
    from collections import defaultdict
    data = json.loads(Path(path).read_text())
    drugs = data.get("drugs", {})

    def score(e):
        ls = e.get("label_sections", {}) or {}
        return ((e.get("validation_status") not in (None, "not_loaded")) * 10
                + bool((e.get("source") or {}).get("set_id")) * 4
                + ("prescription" in str((e.get("identifiers") or {}).get("product_type", "")).lower()) * 2
                + len(ls) + len(e.get("dose_rules") or []))

    items, dropped = [], 0
    for k, e in drugs.items():
        nk = clean_drug_name(k)                          # always strip leading strength numbers
        if not nk:
            dropped += 1; continue
        if (drop_noise or single_only) and e.get("validation_status") in (None, "not_loaded"):
            dropped += 1; continue
        if drop_noise and is_noise(nk):                  # homeopathic / cosmetic product
            dropped += 1; continue
        if drop_noise and is_homeopathic(e):             # homeopathic by label markers (HPUS, "mix N drops")
            dropped += 1; continue
        if drop_noise and not has_real_label(e):         # header-only / empty label
            dropped += 1; continue
        if single_only and not is_single_ingredient(nk):  # combination product
            dropped += 1; continue
        items.append((nk, e))

    groups = defaultdict(list)
    for k, e in items:
        groups[_label_identity(k, e)].append((k, e))
    keep = {}
    for members in groups.values():
        k, e = min(members, key=lambda ke: (len(ke[0].split()), len(ke[0]), -score(ke[1])))
        if k not in keep or score(e) > score(keep[k]):
            keep[k] = e

    data["drugs"] = keep
    data.setdefault("_metadata", {})["drug_count"] = len(keep)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False))
    same_label = len(items) - len(keep)
    extra = (f", dropped {dropped} noise/no-label" if (drop_noise or single_only) else "")
    print(f"deduped {len(drugs)} → {len(keep)} drugs (merged {same_label} same-label duplicates{extra})")
    return keep


def build_entry(drug_name, api_key=None, required_route=None, indication=None) -> Dict[str, Any]:
    canonical = canonicalize_drug_name(drug_name)
    required_route = required_route or (CANONICAL.get(canonical, {}).get("routes") or [None])[0]
    indication = indication or INDICATION_HINTS.get(canonical)
    rec, meta = select_label(fetch_label_candidates(canonical, api_key),
                             required_route=required_route, indication=indication)
    if not rec:
        return {"identifiers": {"generic_name": canonical}, "source": {"error": "no label found"},
                "dose_rules": [], "validation_status": "not_loaded"}
    entry = parse_label_record(rec, canonical)
    entry["source"]["selection"] = meta
    if not entry["identifiers"]["rxnorm_cui"]:
        entry["identifiers"]["rxnorm_cui"] = normalize_rxcui(canonical)
    entry["faers_signals"] = fetch_faers_signal(canonical, api_key=api_key)
    return entry


def populate(generic_names, out_path="drug_clinical_data.json", api_key=None,
             merge=False, resume=False, save_every=25):
    import time
    from pathlib import Path
    drugs = {}
    if (merge or resume) and Path(out_path).exists():
        try:
            drugs = json.loads(Path(out_path).read_text()).get("drugs", {})
            print(f"  loaded existing {out_path} ({len(drugs)} drugs already present)")
        except Exception:
            drugs = {}

    def _loaded(entry):
        return bool(entry) and entry.get("validation_status") not in (None, "not_loaded") \
            and entry.get("source", {}).get("set_id")

    def _save():
        out = {"_metadata": {"schema_version": "2.0-clinical-doserules", "generated": date.today().isoformat(),
                             "source": "openFDA label + FAERS + RxNav", "drug_count": len(drugs),
                             "note": "dose_rules are extracted_unverified until human validation"},
               "drugs": drugs}
        Path(out_path).write_text(json.dumps(out, indent=2, ensure_ascii=False))

    total = len(generic_names)
    done = skipped = failed = 0
    for i, name in enumerate(generic_names, 1):
        c = canonicalize_drug_name(name)
        if resume and _loaded(drugs.get(c)):
            skipped += 1
            continue
        print(f"  [{i}/{total}] {c} ...", end=" ", flush=True)
        try:
            e = build_entry(c, api_key=api_key)
        except Exception as err:
            failed += 1
            print(f"FAILED ({type(err).__name__}); skipping")
            continue
        drugs[c] = e
        done += 1
        print(f"set_id={e.get('source', {}).get('set_id')} status={e.get('validation_status')}")
        if done % save_every == 0:
            _save(); print(f"  … checkpoint saved ({len(drugs)} drugs)")
        time.sleep(0.3 if api_key else 1.6)   # 240/min with key; ~37/min without (1000/day cap)
    _save()
    print(f"wrote {out_path}: {len(drugs)} drugs total (fetched {done}, skipped {skipped}, failed {failed})")


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    ap = argparse.ArgumentParser(description="Fetch official labels + FAERS into clinical data.")
    ap.add_argument("drugs", nargs="*")
    ap.add_argument("--all", action="store_true", help="fetch every (canonical) drug in drugs_pkpd.json")
    ap.add_argument("--discover", type=int, default=0,
                    help="fetch the top-N most-common base drugs from the FDA label dataset (max 1000)")
    ap.add_argument("--combos", action="store_true", help="include combination products in --discover")
    ap.add_argument("--dedupe", action="store_true",
                    help="collapse same-label duplicates in an existing --out file, then exit")
    ap.add_argument("--drop-noise", action="store_true",
                    help="with --dedupe, drop homeopathic/cosmetic product names + no-label (KEEPS combos)")
    ap.add_argument("--single-only", action="store_true",
                    help="with --dedupe, also drop combination products (single-ingredient drugs only)")
    ap.add_argument("--merge", action="store_true", help="merge into existing --out file instead of overwriting")
    ap.add_argument("--resume", action="store_true", help="skip drugs already fetched in --out (safe to re-run)")
    ap.add_argument("--save-every", type=int, default=25, help="checkpoint the output file every N drugs")
    ap.add_argument("--pkpd", default="drugs_pkpd.json", help="mechanism file to read the drug list from")
    ap.add_argument("--out", default="drug_clinical_data.json")
    ap.add_argument("--api-key", default=None)
    a = ap.parse_args()
    if a.dedupe:
        dedupe_file(a.out, drop_noise=a.drop_noise or a.single_only, single_only=a.single_only)
        raise SystemExit
    names = list(a.drugs)
    if a.discover:
        if not a.api_key:
            print("WARNING: discovering/fetching this many drugs without --api-key will hit openFDA's "
                  "1,000/day cap. Get a free key at https://open.fda.gov/apis/authentication/")
        existing = {}
        if Path(a.out).exists():
            try:
                existing = json.loads(Path(a.out).read_text()).get("drugs", {})
            except Exception:
                existing = {}
        print(f"--discover: finding {a.discover} NEW base drugs from openFDA "
              f"(excluding {len(existing)} already in {a.out}) …")
        names = discover_drug_names(a.discover, a.api_key, include_combos=a.combos,
                                    exclude=list(existing.keys()))
        print(f"  got {len(names)} names from the FDA label dataset")
    if a.all:
        from drug_identity import dedupe_to_canonical
        pk = json.loads(Path(a.pkpd).read_text())["drugs"]
        names = sorted(dedupe_to_canonical(pk).keys()) + names
    if names:
        populate(names, a.out, a.api_key, merge=a.merge, resume=a.resume, save_every=a.save_every)
    else:
        print("usage:\n"
              "  python3 clinical_data.py --discover 1000 --api-key KEY --resume --out drug_clinical_data.json\n"
              "  python3 clinical_data.py --all --out drug_clinical_data.json          # mechanism-map drugs\n"
              "  python3 clinical_data.py rivaroxaban --merge                          # add one, keep the rest")
