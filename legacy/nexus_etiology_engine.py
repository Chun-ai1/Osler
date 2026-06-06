"""
NEXUS Etiology Engine
════════════════════════════════════════════════════════════════
真正让 NEXUS 理解 virus vs bacteria vs non-infectious，
而不是靠 RAG 搜索关键词。

核心思想：
  NEXUS 已经推理出"最可能的诊断是 pneumonia"。
  NEXUS 已经知道 pneumonia 是 bacteria。
  所以 etiology = bacteria。

这个引擎把 NEXUS 的推理结论直接转化成 etiology 判断，
同时用机制库的 domain 字段做交叉验证。

取代：
  - search_mechanisms_bacteria() 的 RAG 调用
  - search_mechanisms_virus() 的 RAG 调用
  - EtiologyClassifier 里纯症状模式匹配（作为兜底保留）

整合方式：
  在 nexus_medical.py 的 enhance_pipeline_result() Layer 5 里，
  把 EtiologyClassifier 替换成 NexusEtiologyEngine。
  结果格式完全兼容，不需要改 app.py。
"""

from __future__ import annotations
from collections import defaultdict, Counter
from typing import Dict, List, Set, Optional, Tuple
import json
import os


# ─────────────────────────────────────────────────────────────
# 疾病病因知识库
# 从你的疾病 JSON 文件里读到，这里作为兜底硬编码
# 格式：disease_name → (pathogen_type, confidence)
# ─────────────────────────────────────────────────────────────
_DISEASE_PATHOGEN_MAP: Dict[str, Tuple[str, float]] = {
    # Bacterial
    "pneumonia":          ("bacterial", 0.90),
    "bacterial pneumonia":("bacterial", 0.95),
    "strep throat":       ("bacterial", 0.92),
    "urinary tract infection": ("bacterial", 0.90),
    "uti":                ("bacterial", 0.90),
    "cellulitis":         ("bacterial", 0.88),
    "meningitis":         ("bacterial", 0.85),
    "bacterial meningitis":("bacterial", 0.95),
    "sepsis":             ("bacterial", 0.85),
    "sinusitis":          ("bacterial", 0.75),
    "otitis media":       ("bacterial", 0.78),
    "pyelonephritis":     ("bacterial", 0.90),
    "endocarditis":       ("bacterial", 0.88),
    "osteomyelitis":      ("bacterial", 0.88),
    "appendicitis":       ("bacterial", 0.85),
    "peritonitis":        ("bacterial", 0.90),
    "cholecystitis":      ("bacterial", 0.75),
    "lyme disease":       ("bacterial", 0.90),
    "tuberculosis":       ("bacterial", 0.90),
    "typhoid":            ("bacterial", 0.90),

    # Viral
    "influenza":          ("viral", 0.92),
    "flu":                ("viral", 0.90),
    "covid":              ("viral", 0.92),
    "covid-19":           ("viral", 0.95),
    "common cold":        ("viral", 0.90),
    "viral gastroenteritis":("viral", 0.88),
    "gastroenteritis":    ("viral", 0.70),
    "mononucleosis":      ("viral", 0.92),
    "hiv":                ("viral", 0.95),
    "hepatitis":          ("viral", 0.85),
    "hepatitis a":        ("viral", 0.90),
    "hepatitis b":        ("viral", 0.90),
    "hepatitis c":        ("viral", 0.90),
    "herpes":             ("viral", 0.90),
    "dengue":             ("viral", 0.92),
    "rsv":                ("viral", 0.92),
    "bronchiolitis":      ("viral", 0.82),
    "croup":              ("viral", 0.85),
    "viral meningitis":   ("viral", 0.90),

    # Non-infectious
    "heart attack":       ("non_infectious", 0.95),
    "myocardial infarction":("non_infectious", 0.95),
    "stroke":             ("non_infectious", 0.95),
    "asthma":             ("non_infectious", 0.88),
    "migraine":           ("non_infectious", 0.92),
    "anxiety":            ("non_infectious", 0.90),
    "panic attack":       ("non_infectious", 0.90),
    "gerd":               ("non_infectious", 0.85),
    "acid reflux":        ("non_infectious", 0.85),
    "kidney stones":      ("non_infectious", 0.90),
    "pulmonary embolism": ("non_infectious", 0.88),
    "dvt":                ("non_infectious", 0.85),
    "rheumatoid arthritis":("non_infectious", 0.90),
    "lupus":              ("non_infectious", 0.90),
    "diabetes":           ("non_infectious", 0.92),
    "hypertension":       ("non_infectious", 0.90),
    "anemia":             ("non_infectious", 0.85),
    "hypothyroidism":     ("non_infectious", 0.88),
    "hyperthyroidism":    ("non_infectious", 0.88),
    "copd":               ("non_infectious", 0.88),
    "heart failure":      ("non_infectious", 0.90),
}

# 机制 domain → etiology 映射
_DOMAIN_TO_ETIOLOGY = {
    "bacteria":      "bacterial",
    "bacterial":     "bacterial",
    "virus":         "viral",
    "viral":         "viral",
    "general":       None,    # neutral
    "non-infectious":"non_infectious",
    "autoimmune":    "non_infectious",
    "metabolic":     "non_infectious",
    "trauma":        "non_infectious",
    "vascular":      "non_infectious",
}


class NexusEtiologyEngine:
    """
    用 NEXUS 推理结论判断病因，而不是 RAG 关键词搜索。

    三层推理：
      Layer A (主要) — NEXUS 已推断的 top 诊断 × 疾病病因知识库
                       例：top1=pneumonia → bacterial 0.90
      Layer B (交叉) — 激活的机制里 domain=bacteria/virus 的比例
                       例：7/10 个激活机制是 bacteria domain → bacterial signal
      Layer C (兜底) — 原有的症状模式分析（轻权重，不依赖 RAG）

    结果格式与 EtiologyClassifier 完全兼容：
      {
        "etiology": "bacterial" | "viral" | "non_infectious" | "uncertain",
        "confidence": 0.0-1.0,
        "scores": {"bacterial": 0.7, "viral": 0.2, "non_infectious": 0.1},
        "reasoning": [...],
        "nexus_evidence": {...},   ← 新增：NEXUS 推理来源
        "lab_predictions": {...},
        "recommended_tests": [...],
      }
    """

    def __init__(self, nexus_medical=None):
        """
        nexus_medical: NexusMedical 实例（可选）
                       如果传入，可以访问 mech_info 的 domain 字段
        """
        self.nexus = nexus_medical
        # 从 NEXUS 加载的疾病病因映射（比硬编码更准确）
        self._disease_map: Dict[str, Tuple[str, float]] = dict(_DISEASE_PATHOGEN_MAP)
        self._loaded = False

    def load_from_nexus(self):
        """
        从 NEXUS 的 self.diseases 里读取 pathogen_type 字段，
        补充/覆盖硬编码知识库。
        """
        if self._loaded or not self.nexus:
            return
        try:
            for disease_name, disease_data in self.nexus.diseases.items():
                pt = (disease_data.get("pathogen_type") or
                      disease_data.get("pathogen") or "").lower().strip()
                if pt in ("bacteria", "bacterial"):
                    self._disease_map[disease_name.lower()] = ("bacterial", 0.88)
                elif pt in ("virus", "viral"):
                    self._disease_map[disease_name.lower()] = ("viral", 0.88)
                elif pt in ("non-infectious", "non_infectious"):
                    self._disease_map[disease_name.lower()] = ("non_infectious", 0.88)
            self._loaded = True
            print(f"[NEXUS-ETIOLOGY] Loaded {len(self._disease_map)} disease→pathogen mappings")
        except Exception as e:
            print(f"[NEXUS-ETIOLOGY] load_from_nexus failed (non-blocking): {e}")

    def classify(self,
                 symptoms: List[str],
                 nexus_result: dict = None,
                 vitals: dict = None,
                 labs: dict = None) -> dict:
        """
        主接口 — 与 EtiologyClassifier.classify() 完全兼容。

        nexus_result: nexus_instance.enhance_pipeline_result() 的输出
                      包含 nexus_diagnoses, nexus_thinking 等字段
        """
        self.load_from_nexus()
        reasoning = []
        scores = {"bacterial": 0.0, "viral": 0.0, "non_infectious": 0.0}
        nexus_evidence = {}

        # ══════════════════════════════════════════════════
        # Layer A: NEXUS 诊断结论 × 疾病病因知识库
        # ══════════════════════════════════════════════════
        layer_a = self._from_nexus_diagnoses(nexus_result or {})
        for k in scores:
            scores[k] += layer_a["scores"].get(k, 0.0) * 0.55   # 主权重 55%
        reasoning.extend(layer_a["reasoning"])
        nexus_evidence["diagnosis_evidence"] = layer_a.get("evidence", [])

        # ══════════════════════════════════════════════════
        # Layer B: 激活机制的 domain 分布
        # ══════════════════════════════════════════════════
        layer_b = self._from_mechanism_domains(nexus_result or {})
        for k in scores:
            scores[k] += layer_b["scores"].get(k, 0.0) * 0.30   # 机制权重 30%
        reasoning.extend(layer_b["reasoning"])
        nexus_evidence["mechanism_evidence"] = layer_b.get("evidence", {})

        # ══════════════════════════════════════════════════
        # Layer C: 症状模式兜底（不用 RAG，纯规则）
        # ══════════════════════════════════════════════════
        layer_c = self._from_symptom_pattern(symptoms)
        for k in scores:
            scores[k] += layer_c["scores"].get(k, 0.0) * 0.15   # 兜底权重 15%
        reasoning.extend(layer_c["reasoning"])

        # ══════════════════════════════════════════════════
        # Lab 加权（如果有）
        # ══════════════════════════════════════════════════
        if labs:
            layer_lab = self._from_labs(labs)
            # labs 是强信号，单独叠加（不归入百分比）
            for k in scores:
                scores[k] += layer_lab["scores"].get(k, 0.0) * 0.40
            reasoning.extend(layer_lab["reasoning"])

        # ══════════════════════════════════════════════════
        # Normalize + 判定
        # ══════════════════════════════════════════════════
        total = max(sum(scores.values()), 0.01)
        norm = {k: round(v / total, 3) for k, v in scores.items()}

        best = max(norm, key=norm.get)
        confidence = norm[best]

        # 如果最高分 < 0.45，判定为 uncertain
        if confidence < 0.45:
            etiology = "uncertain"
        else:
            etiology = best

        return {
            "etiology":         etiology,
            "confidence":       round(confidence, 3),
            "scores":           norm,
            "reasoning":        reasoning,
            "nexus_evidence":   nexus_evidence,
            "lab_predictions":  self._predict_labs(etiology),
            "recommended_tests":self._recommend_tests(etiology, confidence, labs is not None),
            "top_etiology":     etiology,   # 兼容 app.py 里的字段名
        }

    # ──────────────────────────────────────────────────────────
    # Layer A: NEXUS 诊断 → 病因
    # ──────────────────────────────────────────────────────────

    def _from_nexus_diagnoses(self, nexus_result: dict) -> dict:
        """
        NEXUS 推断出 top diseases，直接查病因映射表。
        加权平均，越靠前的诊断权重越高。
        """
        scores = {"bacterial": 0.0, "viral": 0.0, "non_infectious": 0.0}
        reasoning = []
        evidence = []

        dx_list = nexus_result.get("nexus_diagnoses", [])
        if not dx_list:
            reasoning.append("NEXUS diagnosis list empty — using symptom pattern only")
            return {"scores": scores, "reasoning": reasoning, "evidence": evidence}

        # 权重递减：top1=0.50, top2=0.25, top3=0.15, ...
        weights = [0.50, 0.25, 0.15, 0.07, 0.03]

        total_weight = 0.0
        for i, dx in enumerate(dx_list[:5]):
            disease_name = (dx.get("disease") or dx.get("name") or "").lower().strip()
            nexus_score  = float(dx.get("score", 0) or 0)
            w = weights[i] if i < len(weights) else 0.02

            # 查病因映射
            pathogen, conf = self._lookup_pathogen(disease_name)
            if pathogen:
                contribution = w * conf * nexus_score
                scores[pathogen] = scores.get(pathogen, 0.0) + contribution
                total_weight += w
                evidence.append({
                    "disease":  disease_name,
                    "pathogen": pathogen,
                    "conf":     conf,
                    "nexus_score": nexus_score,
                    "rank":     i + 1,
                })
                if i < 3:
                    reasoning.append(
                        f"NEXUS top{i+1}: {disease_name} "
                        f"(score={nexus_score:.2f}) → {pathogen} "
                        f"(disease confidence={conf:.0%})"
                    )

        if total_weight > 0:
            for k in scores:
                scores[k] /= max(total_weight, 0.01)

        if not evidence:
            reasoning.append("No disease-pathogen mapping found — using other layers")

        return {"scores": scores, "reasoning": reasoning, "evidence": evidence}

    def _lookup_pathogen(self, disease_name: str) -> Tuple[Optional[str], float]:
        """
        查疾病病因映射表。
        支持部分匹配（pneumonia bacterial → bacterial）。
        """
        name = disease_name.lower().strip()

        # 精确匹配
        if name in self._disease_map:
            return self._disease_map[name]

        # 部分匹配
        for key, (pathogen, conf) in self._disease_map.items():
            if key in name or name in key:
                return pathogen, conf * 0.85   # 部分匹配降低置信度

        return None, 0.0

    # ──────────────────────────────────────────────────────────
    # Layer B: 激活机制的 domain 分布
    # ──────────────────────────────────────────────────────────

    def _from_mechanism_domains(self, nexus_result: dict) -> dict:
        """
        NEXUS 的推理思维链里包含激活的机制。
        通过 mech_info 的 domain 字段判断哪些机制是 bacteria/virus domain。
        这是真正"NEXUS 懂病因"的核心。
        """
        scores = {"bacterial": 0.0, "viral": 0.0, "non_infectious": 0.0}
        reasoning = []
        evidence = {"bacterial": [], "viral": [], "non_infectious": []}

        if not self.nexus:
            return {"scores": scores, "reasoning": reasoning, "evidence": evidence}

        # 从 NEXUS thinking 里提取激活的机制
        thinking = nexus_result.get("nexus_thinking", [])
        active_mech_ids = set()

        for step in thinking:
            if step.get("step") in ("MECHANISM_CHAIN", "MECHANISM_MATCH", "MECHANISMS"):
                mechs = step.get("mechanisms", step.get("matched", []))
                for m in (mechs or []):
                    mid = m if isinstance(m, str) else m.get("id") or m.get("title", "")
                    if mid:
                        active_mech_ids.add(mid)

        # 如果 thinking 里没有，从 nexus_diagnoses 的 evidence 里提取
        if not active_mech_ids:
            for dx in nexus_result.get("nexus_diagnoses", []):
                for ev in dx.get("evidence", []):
                    active_mech_ids.add(str(ev))

        # 统计每个 domain 的激活数量
        domain_counts = Counter()
        domain_scores = Counter()

        for mech_id, mech_info in self.nexus.mech_info.items():
            domain = (mech_info.get("domain") or "general").lower()
            etiology = _DOMAIN_TO_ETIOLOGY.get(domain)

            # 检查这个机制是否与当前症状/诊断有关
            is_active = (
                mech_id in active_mech_ids or
                any(mech_id in active_mech_ids for _ in [True])
            )

            if etiology and is_active:
                domain_counts[etiology] += 1
                domain_scores[etiology] += 1.0

        # 如果没找到激活的机制，扫描全部机制但降权
        if not any(domain_counts.values()) and self.nexus.mech_info:
            # 通过 top 诊断的 disease_to_mechs 找关联机制
            for dx in nexus_result.get("nexus_diagnoses", [])[:3]:
                disease = (dx.get("disease") or "").strip()
                mech_ids = self.nexus.disease_to_mechs.get(disease, set())
                for mid in list(mech_ids)[:15]:
                    info = self.nexus.mech_info.get(mid, {})
                    domain = (info.get("domain") or "general").lower()
                    etiology = _DOMAIN_TO_ETIOLOGY.get(domain)
                    if etiology:
                        domain_counts[etiology] += 1
                        domain_scores[etiology] += float(dx.get("score", 1.0))

        total_mechs = sum(domain_counts.values())
        if total_mechs > 0:
            for etiology, count in domain_counts.items():
                scores[etiology] = count / total_mechs

            bac = domain_counts.get("bacterial", 0)
            vir = domain_counts.get("viral", 0)
            ni  = domain_counts.get("non_infectious", 0)
            reasoning.append(
                f"NEXUS mechanism domains: "
                f"bacterial={bac}, viral={vir}, non-infectious={ni} "
                f"(total {total_mechs} mechanisms analyzed)"
            )

            # Record top mechanisms per domain
            for k in ("bacterial", "viral", "non_infectious"):
                top = [mid for mid, info in self.nexus.mech_info.items()
                       if _DOMAIN_TO_ETIOLOGY.get(
                           (info.get("domain") or "general").lower()
                       ) == k][:3]
                evidence[k] = [self.nexus.mech_info[m].get("title", m)
                               for m in top if m in self.nexus.mech_info]
        else:
            reasoning.append("No mechanism domain data available from NEXUS")

        return {"scores": scores, "reasoning": reasoning, "evidence": evidence}

    # ──────────────────────────────────────────────────────────
    # Layer C: 症状模式（纯规则，不依赖 RAG）
    # ──────────────────────────────────────────────────────────

    def _from_symptom_pattern(self, symptoms: List[str]) -> dict:
        """
        基于临床知识的症状模式判断。
        这是原 EtiologyClassifier 的精简版，作为兜底。
        不调用任何 RAG 或外部数据库。
        """
        scores = {"bacterial": 0.0, "viral": 0.0, "non_infectious": 0.0}
        reasoning = []

        sym_set = set(s.lower().strip().replace(" ", "_") for s in symptoms)

        # 强烈的细菌信号
        strong_bacterial = {
            "purulent_discharge", "productive_cough", "dysuria",
            "pleuritic_chest_pain", "photophobia", "neck_stiffness",
            "high_fever", "rigors", "erythema", "warmth",
        }
        bac_hits = sym_set & strong_bacterial
        if bac_hits:
            scores["bacterial"] += len(bac_hits) * 0.20
            reasoning.append(f"Bacterial signals: {', '.join(bac_hits)}")

        # 强烈的病毒信号
        strong_viral = {
            "myalgia", "body_aches", "sore_throat", "runny_nose",
            "congestion", "hoarseness", "loss_of_smell", "loss_of_taste",
            "rash", "watery_eyes",
        }
        vir_hits = sym_set & strong_viral
        if vir_hits:
            scores["viral"] += len(vir_hits) * 0.18
            reasoning.append(f"Viral signals: {', '.join(vir_hits)}")

        # 非感染性信号
        noninf_indicators = {
            "chest_pain", "left_arm_pain", "sweating",      # cardiac
            "stiff_neck", "sensitivity_to_light",            # migraine/neuro
            "wheezing", "chest_tightness",                   # asthma
            "joint_pain", "morning_stiffness",               # autoimmune
        }
        ni_hits = sym_set & noninf_indicators
        if ni_hits:
            scores["non_infectious"] += len(ni_hits) * 0.15
            reasoning.append(f"Non-infectious signals: {', '.join(ni_hits)}")

        # 高热 → 偏细菌
        if "high_fever" in sym_set or "fever" in sym_set:
            scores["bacterial"] += 0.05
            scores["viral"]     += 0.04

        # 多系统受累 → 偏病毒
        systems_hit = 0
        if sym_set & {"cough", "runny_nose", "sore_throat"}:
            systems_hit += 1
        if sym_set & {"nausea", "vomiting", "diarrhea"}:
            systems_hit += 1
        if sym_set & {"headache", "body_aches", "fatigue"}:
            systems_hit += 1
        if systems_hit >= 2:
            scores["viral"] += 0.10
            reasoning.append(f"Multi-system involvement ({systems_hit} systems) — viral pattern")

        if not (bac_hits or vir_hits or ni_hits):
            reasoning.append("Symptom pattern: no strong etiology signal")

        return {"scores": scores, "reasoning": reasoning}

    # ──────────────────────────────────────────────────────────
    # Lab analysis (保持与原 EtiologyClassifier 一致)
    # ──────────────────────────────────────────────────────────

    def _from_labs(self, labs: dict) -> dict:
        scores = {"bacterial": 0.0, "viral": 0.0, "non_infectious": 0.0}
        reasoning = []

        wbc      = labs.get("wbc")
        neut_pct = labs.get("neutrophils_pct")
        lymph_pct= labs.get("lymphocytes_pct")
        crp      = labs.get("crp")
        pct      = labs.get("pct") or labs.get("procalcitonin")

        if wbc is not None:
            if wbc > 12:
                scores["bacterial"] += 0.25
                reasoning.append(f"WBC {wbc:.1f} K/uL (elevated) — bacterial indicator")
            elif wbc < 4:
                scores["viral"] += 0.20
                reasoning.append(f"WBC {wbc:.1f} K/uL (low) — viral/severe infection")

        if neut_pct is not None:
            if neut_pct > 80:
                scores["bacterial"] += 0.30
                reasoning.append(f"Neutrophils {neut_pct}% — strong bacterial marker")
            elif lymph_pct and lymph_pct > 40:
                scores["viral"] += 0.25
                reasoning.append(f"Lymphocyte predominance {lymph_pct}% — viral pattern")

        if crp is not None:
            if crp > 100:
                scores["bacterial"] += 0.30
                reasoning.append(f"CRP {crp} mg/L (very high) — bacterial infection")
            elif crp > 40:
                scores["bacterial"] += 0.15
                reasoning.append(f"CRP {crp} mg/L (elevated) — bacterial more likely")
            elif crp > 10:
                scores["viral"] += 0.10
                reasoning.append(f"CRP {crp} mg/L (mild) — common in viral infections")

        if pct is not None:
            if pct > 2.0:
                scores["bacterial"] += 0.50
                reasoning.append(f"Procalcitonin {pct} ng/mL (very high) — highly specific for bacterial")
            elif pct > 0.5:
                scores["bacterial"] += 0.30
                reasoning.append(f"Procalcitonin {pct} ng/mL — bacterial likely")
            else:
                scores["viral"] += 0.20
                scores["non_infectious"] += 0.15
                reasoning.append(f"Procalcitonin {pct} ng/mL (low) — bacterial unlikely")

        return {"scores": scores, "reasoning": reasoning}

    # ──────────────────────────────────────────────────────────
    # Lab predictions & test recommendations
    # ──────────────────────────────────────────────────────────

    def _predict_labs(self, etiology: str) -> dict:
        if etiology == "viral":
            return {
                "expected_wbc":          "normal or low (4-10 K/uL)",
                "expected_differential": "lymphocyte predominance (>40%)",
                "expected_crp":          "normal or mildly elevated (<40 mg/L)",
                "expected_pct":          "low (<0.1 ng/mL)",
            }
        elif etiology == "bacterial":
            return {
                "expected_wbc":          "elevated (>12 K/uL) with left shift",
                "expected_differential": "neutrophil predominance (>80%)",
                "expected_crp":          "significantly elevated (>100 mg/L)",
                "expected_pct":          "elevated (>0.5 ng/mL)",
            }
        elif etiology == "non_infectious":
            return {
                "expected_wbc":          "normal or mildly elevated",
                "expected_differential": "varies by condition",
                "expected_crp":          "may be elevated in autoimmune flares",
                "expected_pct":          "low (<0.1 ng/mL)",
            }
        return {}

    def _recommend_tests(self, etiology: str, confidence: float, has_labs: bool) -> list:
        tests = []
        if not has_labs:
            tests.append("Complete blood count (CBC) with differential")
            tests.append("C-reactive protein (CRP)")
        if confidence < 0.60:
            tests.append("Procalcitonin (PCT) — best discriminator for bacterial vs viral")
        if etiology == "bacterial":
            tests.append("Blood cultures (x2 sets) before antibiotics")
            tests.append("Site-specific culture (urine/sputum/wound as appropriate)")
        elif etiology == "viral":
            tests.append("Rapid viral panel / PCR if available")
            tests.append("Consider specific serology based on clinical picture")
        elif etiology == "non_infectious":
            tests.append("ESR (erythrocyte sedimentation rate)")
            tests.append("ANA panel if autoimmune suspected")
        return tests


# ─────────────────────────────────────────────────────────────
# 便捷函数：在 nexus_medical.py 的 enhance_pipeline_result 里调用
# ─────────────────────────────────────────────────────────────

_ENGINE_INSTANCE: Optional[NexusEtiologyEngine] = None

def get_etiology_engine(nexus_medical=None) -> NexusEtiologyEngine:
    """单例模式，避免重复实例化。"""
    global _ENGINE_INSTANCE
    if _ENGINE_INSTANCE is None:
        _ENGINE_INSTANCE = NexusEtiologyEngine(nexus_medical)
    elif nexus_medical and _ENGINE_INSTANCE.nexus is None:
        _ENGINE_INSTANCE.nexus = nexus_medical
        _ENGINE_INSTANCE._loaded = False
    return _ENGINE_INSTANCE


def classify_etiology(symptoms: List[str],
                      nexus_result: dict,
                      nexus_medical=None,
                      labs: dict = None) -> dict:
    """
    一行调用接口。

    在 nexus_medical.py enhance_pipeline_result() Layer 5 里替换：

    旧代码：
        from nexus_engine.etiology_classifier import EtiologyClassifier
        ec = EtiologyClassifier()
        etiology = ec.classify(symptoms, labs=labs)

    新代码：
        from nexus_engine.nexus_etiology_engine import classify_etiology
        etiology = classify_etiology(symptoms, result, nexus_medical=self, labs=labs)

    结果格式完全兼容，不需要改 app.py。
    """
    engine = get_etiology_engine(nexus_medical)
    return engine.classify(symptoms, nexus_result=nexus_result, labs=labs)