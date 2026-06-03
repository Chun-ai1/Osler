# mechanism_engine/variant_layer.py
import json
import os
from typing import Dict, List

DNA_VARIANT_DB_PATH = "medical_knowledge/dna/dna_variant_db.json"
_DNA_VARIANT_DB = None


def load_dna_variant_db() -> Dict:
    global _DNA_VARIANT_DB
    if _DNA_VARIANT_DB is not None:
        return _DNA_VARIANT_DB
    if not os.path.exists(DNA_VARIANT_DB_PATH):
        _DNA_VARIANT_DB = {}
        return _DNA_VARIANT_DB
    with open(DNA_VARIANT_DB_PATH, "r", encoding="utf-8") as f:
        _DNA_VARIANT_DB = json.load(f) or {}
    return _DNA_VARIANT_DB


def parse_23andme_txt(path: str) -> Dict[str, str]:
    """
    Expects lines like:
    rsid    chrom   pos   genotype
    rs429358 19 45411941 CT
    """
    variants = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            rsid = parts[0].strip()
            if rsid.lower() == "rsid":   # ✅ skip header
                continue
            genotype = parts[3].strip().upper()
            if rsid.startswith("rs") and genotype and genotype != "--":
                variants[rsid] = genotype
    return variants


def parse_vcf(path: str) -> Dict[str, str]:
    """
    Simplified VCF parser:
    - Uses ID column as rsid when available
    - Genotype derived from sample field (GT) if present
    Note: Real VCF needs ref/alt allele mapping; this is minimal.
    """
    variants = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 10:
                continue
            rsid = parts[2].strip()
            if not rsid.startswith("rs"):
                continue
            fmt = parts[8].split(":")
            sample = parts[9].split(":")
            if "GT" not in fmt:
                continue
            gt = sample[fmt.index("GT")]
            # store GT raw e.g. 0/1 for now
            variants[rsid] = gt
    return variants


def match_variants_to_mechanisms(dna_variants: Dict[str, str]) -> List[Dict]:
    """
    Produces variant-driven hits:
    [{rsid,gene,effect,evidence_level,mechanisms,genotype}]
    """
    db = load_dna_variant_db()
    hits = []

    for rsid, genotype in (dna_variants or {}).items():
        entry = db.get(rsid)
        if not entry:
            continue

        risk = (entry.get("risk_allele") or "").upper()
        prot = (entry.get("protective_allele") or "").upper()
        evidence = entry.get("evidence_level", "weak")
        mechanisms = entry.get("mechanisms", []) or []

        effect = None
        # 23andMe genotype is letters; VCF GT is 0/1 — we handle letter mode here
        if any(ch.isalpha() for ch in genotype):
            if risk and risk in genotype:
                effect = "risk_increase"
            elif prot and prot in genotype:
                effect = "protective"
        else:
            # if genotype is GT like 0/1, we can't map without ref/alt; skip safely
            continue

        if not effect:
            continue

        hits.append({
            "rsid": rsid,
            "gene": entry.get("gene"),
            "genotype": genotype,
            "effect": effect,
            "evidence_level": evidence,
            "mechanisms": mechanisms
        })

    return hits

if __name__ == "__main__":
    print(match_variants_to_mechanisms({
        "rs429358": "CT",
        "rs7412": "CC"
    }))
