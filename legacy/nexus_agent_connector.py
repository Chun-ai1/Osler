"""
NEXUS ML Agent Connector
════════════════════════════════════════════════════════════════
把训练好的模型接入你现有的 Flask 应用。

集成方式（两步）：

  Step 1 — 在 app.py 里注册：
    from nexus_engine.nexus_agent_connector import init_agent, agent_bp
    app.register_blueprint(agent_bp)
    init_agent()                        # 自动加载 checkpoint

  Step 2 — 在你的 chat_stream / reason 流程里调用：
    from nexus_engine.nexus_agent_connector import agent_diagnose
    agent_result = agent_diagnose(symptoms, nexus_result)
    # → 返回 {"diagnosis": "pneumonia", "confidence": 0.87, "treatment": "antibiotics", ...}

新增 API 端点：
    POST /agent/diagnose   — 输入症状，返回 ML 诊断 + NEXUS 推理融合结果
    GET  /agent/status     — 查看模型状态（是否加载、准确率等）
    POST /agent/reload     — 热重载 checkpoint（不重启服务）
"""

from __future__ import annotations
import json, os, random, math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from flask import Blueprint, request, jsonify

# ─────────────────────────────────────────────────────────────
# 全局实例（单例）
# ─────────────────────────────────────────────────────────────

_agent_instance: "NexusAgentConnector | None" = None

agent_bp = Blueprint("agent", __name__, url_prefix="/agent")


# ─────────────────────────────────────────────────────────────
# CONNECTOR 核心类
# ─────────────────────────────────────────────────────────────

class NexusAgentConnector:
    """
    把训练好的 numpy 分类器接入 NEXUS 系统。

    职责：
      1. 加载 nexus_checkpoint.npz（训练好的权重）
      2. 加载 nexus_config.json（疾病/症状/治疗词表）
      3. 接收症状 + NEXUS 推理结果 → 输出 ML 诊断
      4. 融合 ML 置信度 和 NEXUS 证据链 → 最终建议
    """

    def __init__(self,
                 checkpoint_path: str = "nexus_checkpoint.npz",
                 config_path: str | None = None):

        self.checkpoint_path = checkpoint_path
        self.config_path     = config_path
        self.loaded          = False
        self.load_error      = None

        # 配置
        self.disease_names:  List[str] = []
        self.treatments:     List[str] = []
        self.symptoms:       List[str] = []
        self.symptom_weights: Dict[str, float] = {}
        self.combo_signals:  Dict[str, dict] = {}
        self.organ_systems:  List[str] = []
        self.critical_organs:List[str] = []
        self.n_features:     int = 128

        # 模型权重
        self.W1 = self.b1 = None
        self.Wdx = self.bdx = None
        self.Wtx = self.btx = None

        # 统计
        self.stats = {
            "checkpoint":     checkpoint_path,
            "episode":        0,
            "disease_count":  0,
            "symptom_count":  0,
            "n_features":     0,
        }

        self._load()

    # ──────────────────────────────────────────────────────────
    # 加载
    # ──────────────────────────────────────────────────────────

    def _load(self):
        """加载 checkpoint + config，失败时优雅降级（NEXUS 独立继续工作）。"""
        try:
            self._load_config()
            self._load_checkpoint()
            self.loaded = True
            print(f"[AGENT] model loaded successfully")
            print(f"        diseases: {self.stats['disease_count']}  "
                  f"症状: {self.stats['symptom_count']}  "
                  f"特征维度: {self.stats['n_features']}")
        except Exception as e:
            self.loaded    = False
            self.load_error = str(e)
            print(f"[AGENT] model load failed (NEXUS still works): {e}")

    def _load_config(self):
        """从 nexus_config.json 加载词表和配置。"""
        # 自动搜索 config 文件
        here = Path(__file__).parent
        candidates = [
            self.config_path,
            here / "nexus_config.json",
            here.parent / "nexus_config.json",
            Path("nexus_config.json"),
        ]
        cfg_path = next((str(c) for c in candidates
                         if c and Path(c).exists()), None)

        if not cfg_path:
            raise FileNotFoundError(
                "找不到 nexus_config.json。"
                "请先运行: python nexus_engine/nexus_sync_config.py"
            )

        cfg = json.load(open(cfg_path, encoding="utf-8"))

        self.disease_names   = [d["name"] for d in cfg["diseases"]]
        self.treatments      = cfg["treatments"]
        self.organ_systems   = cfg.get("anatomy", {}).get(
            "organ_systems",
            ["respiratory","cardiovascular","neurologic","gi","systemic","immune","unknown"]
        )
        self.critical_organs = cfg.get("anatomy", {}).get(
            "critical_organs",
            ["brain","heart","lungs","meninges","liver","kidney","spleen","brainstem"]
        )
        self.symptom_weights = {
            k: v for k, v in cfg.get("symptom_weights", {}).items()
            if not k.startswith("_")
        }
        self.combo_signals = {
            k: v for k, v in cfg.get("combo_signals", {}).items()
            if not k.startswith("_")
        }

        # 症状词表 = 所有疾病的症状并集
        all_syms = set()
        for d in cfg["diseases"]:
            all_syms.update(d["symptoms"])
        self.symptoms = sorted(all_syms)

    def _load_checkpoint(self):
        """从 .npz 文件加载权重，验证兼容性。"""
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(
                f"找不到 checkpoint: {self.checkpoint_path}\n"
                f"请先训练：python nexus_engine/nexus_learning_env_v2.py"
            )

        d = np.load(self.checkpoint_path, allow_pickle=True)

        # 兼容性检查
        saved_diseases = list(d.get("diseases", []))
        if saved_diseases and saved_diseases != self.disease_names:
            raise ValueError(
                f"Checkpoint 的疾病列表与 config 不匹配。\n"
                f"  Checkpoint: {saved_diseases}\n"
                f"  Config:     {self.disease_names}\n"
                f"请重新训练: rm nexus_checkpoint.npz && python nexus_engine/nexus_learning_env_v2.py"
            )

        saved_nf = int(d["n_features"][0]) if "n_features" in d else d["W1"].shape[0]
        self.n_features = saved_nf

        self.W1  = d["W1"].astype(np.float32)
        self.b1  = d["b1"].astype(np.float32)
        self.Wdx = d["Wdx"].astype(np.float32)
        self.bdx = d["bdx"].astype(np.float32)
        self.Wtx = d["Wtx"].astype(np.float32)
        self.btx = d["btx"].astype(np.float32)

        self.stats.update({
            "episode":       int(d["episode"][0]) if "episode" in d else 0,
            "disease_count": len(self.disease_names),
            "symptom_count": len(self.symptoms),
            "n_features":    self.n_features,
        })

    def reload(self):
        """热重载 checkpoint（无需重启服务）。"""
        self._load()
        return self.loaded

    # ──────────────────────────────────────────────────────────
    # 推理
    # ──────────────────────────────────────────────────────────

    def _build_feature_vector(self, symptoms: List[str],
                               nexus_result: dict,
                               sym2sys: dict) -> np.ndarray:
        """把症状 + NEXUS 结果转成特征向量（和训练时完全一致）。"""
        vec     = np.zeros(self.n_features, dtype=np.float32)
        dx_list = nexus_result.get("nexus_diagnoses", [])
        N_DX    = len(self.disease_names)

        # [0:N_DX] NEXUS 疾病概率
        raw = np.zeros(N_DX, dtype=np.float32)
        for d in dx_list[:N_DX * 2]:
            name  = d.get("disease", "").lower()
            score = float(d.get("score", 0))
            for j, known in enumerate(self.disease_names):
                if set(known.split()) & set(name.split()) or known in name or name in known:
                    raw[j] = max(raw[j], score)
                    break
        total = raw.sum()
        if total > 0:
            vec[0:N_DX] = raw / total

        # 器官系统分布
        sys_s = N_DX + 10
        votes = np.zeros(len(self.organ_systems), dtype=np.float32)
        for sym in symptoms:
            sys_name = sym2sys.get(sym, "unknown")
            if sys_name in self.organ_systems:
                votes[self.organ_systems.index(sys_name)] += 1.0
        if votes.sum() > 0:
            vec[sys_s:sys_s + len(self.organ_systems)] = votes / votes.sum()

        # NEXUS 机制信号
        mech_s = sys_s + len(self.organ_systems)
        flags  = nexus_result.get("nexus_red_flags", [])
        vec[mech_s]   = min(len(flags) / 5.0, 1.0)
        vec[mech_s+1] = nexus_result.get("nexus_consistency", {}).get("consistency_score", 0.5)
        vec[mech_s+2] = min(len(nexus_result.get("nexus_suggested_questions", [])) / 5.0, 1.0)
        vec[mech_s+3] = min(len(nexus_result.get("nexus_root_causes", [])) / 3.0, 1.0)

        # 解剖扩散风险
        anat_s = mech_s + 5
        spread = nexus_result.get("nexus_pathogen_spread", [])
        organ_risk = {s.get("organ", ""): s.get("risk", 0)
                      for s in spread if isinstance(s, dict)}
        for k, organ in enumerate(self.critical_organs):
            vec[anat_s + k] = float(organ_risk.get(organ, 0.0))

        # 症状 one-hot（带权重）
        sym_s = anat_s + len(self.critical_organs)
        sym_set = set(symptoms)
        for k, sym in enumerate(self.symptoms):
            if sym_s + k >= self.n_features:
                break
            if sym in sym_set:
                vec[sym_s + k] = min(self.symptom_weights.get(sym, 1.0), 2.5)

        # 组合信号
        combo_s = sym_s + len(self.symptoms)
        for k, (name, combo) in enumerate(self.combo_signals.items()):
            if combo_s + k >= self.n_features:
                break
            req = combo.get("requires", [])
            exc = combo.get("excludes", [])
            if all(r in sym_set for r in req) and not any(e in sym_set for e in exc):
                vec[combo_s + k] = combo.get("value", 1.5)

        return vec

    def _forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h = np.maximum(0, x @ self.W1 + self.b1)
        return h @ self.Wdx + self.bdx, h @ self.Wtx + self.btx

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()

    def predict(self,
                symptoms: List[str],
                nexus_result: dict,
                sym2sys: dict | None = None) -> dict:
        """
        主推理接口。

        返回：
        {
          "diagnosis":   "pneumonia",          # ML 预测疾病
          "confidence":  0.87,                 # softmax 概率
          "treatment":   "antibiotics",        # 推荐治疗
          "top3": [                            # 前3候选
            {"disease": "pneumonia",  "prob": 0.87},
            {"disease": "sepsis",     "prob": 0.08},
            {"disease": "flu",        "prob": 0.05},
          ],
          "source": "ml_agent",
          "model_episode": 5000,
        }
        """
        if not self.loaded:
            return {"error": "模型未加载", "source": "nexus_only"}

        if sym2sys is None:
            sym2sys = {}

        try:
            vec       = self._build_feature_vector(symptoms, nexus_result, sym2sys)
            dx_logits, tx_logits = self._forward(vec)
            dx_probs  = self._softmax(dx_logits)
            tx_probs  = self._softmax(tx_logits)

            dx_idx    = int(np.argmax(dx_probs))
            tx_idx    = int(np.argmax(tx_probs))

            # Top-3 候选
            top3_idx  = np.argsort(dx_probs)[::-1][:3]
            top3      = [{"disease": self.disease_names[i],
                          "probability": round(float(dx_probs[i]), 4)}
                         for i in top3_idx]

            return {
                "diagnosis":    self.disease_names[dx_idx],
                "confidence":   round(float(dx_probs[dx_idx]), 4),
                "treatment":    self.treatments[tx_idx] if tx_idx < len(self.treatments) else "rest",
                "top3":         top3,
                "source":       "ml_agent",
                "model_episode": self.stats["episode"],
            }
        except Exception as e:
            return {"error": str(e), "source": "ml_agent_error"}

    def fuse_with_nexus(self,
                        symptoms: List[str],
                        nexus_result: dict,
                        sym2sys: dict | None = None) -> dict:
        """
        融合 ML 结果 + NEXUS 推理，输出最终建议。

        融合规则：
          - ML 高置信度(>0.7)  且 NEXUS 同意   → ML 为主，置信度高
          - ML 高置信度        但 NEXUS 不同意 → 展示两者，标注分歧
          - ML 低置信度(<0.5)                 → NEXUS 为主，ML 作参考
          - ML 未加载                          → 纯 NEXUS
        """
        nexus_top = nexus_result.get("nexus_diagnoses", [{}])[0]
        nexus_dx  = nexus_top.get("disease", "").lower()
        nexus_score = float(nexus_top.get("score", 0))

        ml_result = self.predict(symptoms, nexus_result, sym2sys)

        if "error" in ml_result:
            return {
                "final_diagnosis": nexus_dx,
                "final_treatment": None,
                "confidence":      nexus_score,
                "method":          "nexus_only",
                "ml_result":       ml_result,
                "nexus_result":    nexus_result,
            }

        ml_dx    = ml_result["diagnosis"]
        ml_conf  = ml_result["confidence"]
        ml_tx    = ml_result["treatment"]
        ml_agrees = (ml_dx.lower() in nexus_dx or nexus_dx in ml_dx.lower())

        if ml_conf >= 0.70 and ml_agrees:
            method          = "ml_high_confidence_nexus_agree"
            final_diagnosis = ml_dx
            final_treatment = ml_tx
            confidence      = ml_conf

        elif ml_conf >= 0.70 and not ml_agrees:
            method          = "ml_nexus_disagree"
            final_diagnosis = ml_dx          # ML 更可靠（训练数据支持）
            final_treatment = ml_tx
            confidence      = ml_conf * 0.85  # 降低置信度，标注分歧

        else:
            method          = "nexus_primary_ml_reference"
            final_diagnosis = nexus_dx
            final_treatment = ml_tx if ml_conf >= 0.40 else None
            confidence      = nexus_score

        return {
            "final_diagnosis": final_diagnosis,
            "final_treatment": final_treatment,
            "confidence":      round(confidence, 4),
            "method":          method,
            "ml_agrees_nexus": ml_agrees,
            "ml_result":       ml_result,
            "nexus_top":       {
                "disease": nexus_dx,
                "score":   nexus_score,
            },
            "red_flags":            nexus_result.get("nexus_red_flags", []),
            "suggested_questions":  nexus_result.get("nexus_suggested_questions", []),
        }


# ─────────────────────────────────────────────────────────────
# 公共 API 函数（供你的 app.py 调用）
# ─────────────────────────────────────────────────────────────

def init_agent(checkpoint_path: str = "nexus_checkpoint.npz",
               config_path: str | None = None) -> NexusAgentConnector:
    """
    在 app.py 启动时调用一次：
        from nexus_engine.nexus_agent_connector import init_agent
        init_agent()
    """
    global _agent_instance
    _agent_instance = NexusAgentConnector(checkpoint_path, config_path)
    return _agent_instance


def get_agent() -> "NexusAgentConnector | None":
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = NexusAgentConnector()
    return _agent_instance


def agent_diagnose(symptoms: List[str],
                   nexus_result: dict,
                   sym2sys: dict | None = None) -> dict:
    """
    在你的 chat_stream 或 reason 流程里调用：

        from nexus_engine.nexus_agent_connector import agent_diagnose
        agent_result = agent_diagnose(symptoms, nexus_result)

    返回融合后的最终诊断建议。
    """
    agent = get_agent()
    if agent is None:
        return {"error": "Agent 未初始化", "method": "nexus_only"}
    return agent.fuse_with_nexus(symptoms, nexus_result, sym2sys)


# ─────────────────────────────────────────────────────────────
# Flask 路由（自动注册到你的 app）
# ─────────────────────────────────────────────────────────────

@agent_bp.post("/diagnose")
def api_diagnose():
    """
    POST /agent/diagnose
    Body: {"symptoms": ["cough", "fever", ...]}
    返回：ML + NEXUS 融合诊断结果
    """
    from nexus_engine.nexus_routes import get_nexus
    try:
        from nexus_engine.nexus_medical import SYMPTOM_TO_SYSTEM as sym2sys
    except Exception:
        sym2sys = {}

    data     = request.get_json(silent=True) or {}
    symptoms = data.get("symptoms", [])

    if not symptoms:
        return jsonify({"error": "请提供 symptoms 字段"}), 400

    # Step 1: NEXUS 推理
    nexus  = get_nexus()
    n_result = nexus.enhance_pipeline_result({
        "symptoms": symptoms,
        "top_diseases": [],
        "reasoning": "",
    })

    # Step 2: ML Agent 融合
    agent  = get_agent()
    result = agent.fuse_with_nexus(symptoms, n_result, sym2sys)

    return jsonify({
        "status":           "ok",
        "symptoms":         symptoms,
        "final_diagnosis":  result.get("final_diagnosis"),
        "final_treatment":  result.get("final_treatment"),
        "confidence":       result.get("confidence"),
        "method":           result.get("method"),
        "ml_agrees_nexus":  result.get("ml_agrees_nexus"),
        "top3":             result.get("ml_result", {}).get("top3", []),
        "red_flags":        result.get("red_flags", []),
        "suggested_questions": result.get("suggested_questions", []),
        "nexus_diagnoses":  n_result.get("nexus_diagnoses", [])[:5],
    })


@agent_bp.get("/status")
def api_status():
    """GET /agent/status — 查看模型加载状态"""
    agent = get_agent()
    return jsonify({
        "loaded":      agent.loaded if agent else False,
        "error":       agent.load_error if agent else "未初始化",
        "stats":       agent.stats if agent else {},
        "checkpoint":  agent.checkpoint_path if agent else None,
    })


@agent_bp.post("/reload")
def api_reload():
    """POST /agent/reload — 热重载模型（训练完新版本后调用）"""
    agent = get_agent()
    if not agent:
        return jsonify({"error": "Agent 未初始化"}), 500
    success = agent.reload()
    return jsonify({
        "status":  "ok" if success else "failed",
        "loaded":  agent.loaded,
        "stats":   agent.stats,
        "error":   agent.load_error,
    })