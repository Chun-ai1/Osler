# Osler — 算法架构 Review

> 一份对代码库现状的梳理。结论：仓库其实是**两套独立系统**松散堆在一起，外加大量死代码/实验脚手架。新的药物推荐 Demo（`demo/demo_app.py`）只构建在干净的系统 B 之上。

> **📁 仓库已重构（2026-06）**：根目录原本 ~100 个文件已整理为 4 个文件夹。本文档正文按*算法/系统*描述，文件现在的物理位置如下：
> - **`demo/`** — 药物推荐产品（UI + agent）。
> - **`engine/`** — 系统 B 的 7 个药理引擎模块（`reasoning_engine.py`、`patient_profile.py`、`drug_safety_gate.py`、`drug_identity.py`、`clinical_role.py`、`drug_profile.py`、`clinical_data.py`）。
> - **`data/`** — 所有 `.json` / `.npz`（`drugs_pkpd.json`、器官状态模型 `heart.json` 等、以及全部遗留知识库）。
> - **`legacy/`** — 系统 A 诊断系统（`nexus_*`、`state_model.py`、`anatomy_*`、`spatial_*` 等）、旧 web（`app.py`、`chat.html`、`render_*`）、RL/GNN 实验、测试等**未被 demo 使用**的模块。下文提到的这些文件名现在都在 `legacy/` 下。

## 运行环境（重要）
- **用 `py`（Python 3.12）运行**。`python` / `python3` 在本机是损坏的 Windows Store stub。
- 安装依赖：`py -m pip install -r requirements.txt`（仅 `flask` 必需；`openai` 仅聊天用）。
- 多个 JSON 是 GBK/非 UTF-8，读取请用 `encoding='utf-8', errors='replace'`。

---

## 系统 A — 疾病诊断（`nexus_*`，本次 Demo 不使用）

把身体建成一个**状态变量坐标系**，疾病=对状态变量的一组扰动 delta，症状由阈值规则派生。

**调用链**
```
nexus_medical.reason(symptoms, context)
  → state_model.simulate_disease()      # 各器官状态仿真，三分数加权(alias/evidence/bayes)
  → mechanism_derivation                # 机制级联：器官→失效模式→症状
  → physiology_engine.simulate_body_state()  # BP/HR/乳酸/灌注数值仿真
  → spatial_reasoner / anatomy_atlas    # 解剖一致性、牵涉痛
  → evidence_gate + high_risk_gate      # 分诊门控（triage/syndrome/label）
  → nexus_trace.NexusTrace              # 推理 trace（steps[]）
  → deterministic_response.generate_response()  # 无 LLM 的 Markdown 渲染
```

**孤儿前端**：`chat.html`（43KB）是这套诊断系统的前端，它 fetch `/chat_stream`、`/api/symptoms`、`/reset_followup`、`/save_history`——**这些后端端点在仓库里都不存在**，所以 `chat.html` 当前无法独立工作。

**“训练”层（实验性外挂，非生产路径）**
- `knowledge_gnn.py`：纯 NumPy 的 TransE 知识图谱嵌入，学 `h+r≈t`，用于**预测缺失的图谱边**（给未标注疾病补解剖），结果标 `gnn_inferred`、人工核验前不用于临床。
- `nexus_learning_env_*.py` + `nexus_reward.py` + `nexus_learning_bridge.py`：强化学习环境，奖励=诊断/治疗正确性 + 脓毒症级联仿真惩罚，`learning_bridge` 把学到的规则写回图谱。
- `.npz` 检查点（`kg_embeddings.npz`、`nexus_checkpoint_final_v1.npz`）属于这一层，**与药物推荐无关**。

---

## 系统 B — 药物推荐（`reasoning_engine.py` + `app.py`，干净可用，Demo 的基座）

**与系统 A 同一套“状态变量坐标”**：药物=把状态变量推回平衡的一组 delta（`state_effects`）。用药推理 = 在同一坐标系做“向量相消”，找最能抵消病理 delta 的药。

**核心调用链**（`reasoning_engine.recommend()`）
```
recommend(patient, targets, indication, drugs_pkpd, clinical, scenario)
  1. mechanism_candidates(targets, drugs_pkpd)
       # 逐药匹配 state_effects：同变量 + 兼容器官(药的organ或'*') + 反方向
       # 打分 = Σ |max_delta| × effect_type权重(primary1.0/derived0.5/symptom_relief0.3)
  2. drug_safety_gate.evaluate(patient, entry, indication)
       # 过敏/相互作用/患者flag/肾功能 → decision(ok/caution/adjust/avoid/insufficient_data)
       # 剂量门控：仅在 label 已加载 + 人工验证 + 安全不阻断时才显示患者特定剂量
  3. clinical_role.assign_clinical_role(drug, scenario, flags, symptoms)
       # 临床角色 + rank_priority（如哮喘→albuterol primary、epinephrine 不被提升）
  4. 排序 = (clinical_role.rank_priority, -mechanism_score)
```

**输出本身就是结构化推理链**（每个候选药）：`mechanism_chain`、`matched_targets`、`clinical_role`、`safety.{decision,reasons}`、`result_level`、`dose`、`faers_signals`、`final_answer`。→ Demo 右侧侧边栏直接渲染它。

**`PatientProfile`**（`patient_profile.py`）：age/sex/weight/egfr/hepatic/allergies/current_medications/conditions/symptoms/vitals/labs；`.flags()` 产生 hypotension/tachycardia/hypoxia/hyperkalemia/renal_dose_review/hepatic/pregnancy 等告警；`.renal_label()` KDIGO 分级。

### ⚠️ 已知脆弱点
- **词表未严格对齐**：`drugs_pkpd.json` 用 `myocardial_oxygen_demand`，而疾病 JSON（`heart.json`）用 `ischemia/perfusion`。`recommend()` **不读疾病 JSON**，只匹配手写 targets；匹配是**字符串相等**，名字对不上会**静默不命中**。
- **安全门是关键词匹配**，易误触：例如标签里 “renal artery stenosis” 的 “renal” 会被肾功能 flag 命中而误判 avoid（Demo 数据已规避）。
- **delta 数值人工估计、`review_status: unreviewed`、不可用于临床**：方向(↑/↓)稳健、幅度软。引擎适合**可解释的相对排序**，不适合绝对剂量声明。
- `app.py` line 22 加载 `drug_clinical_data.json`（仓库中**不存在**）→ **现有 `app.py` 启动即崩溃**。Demo 改用可选的 `demo_clinical_data.json`，缺失时降级为“仅机制”模式。

---

## 死代码 / 实验脚手架（建议归档）
| 文件 | 性质 |
|---|---|
| `nexus_learning_env_*.py`, `nexus_learning_bridge.py`, `nexus_reward.py` | RL 训练环境 |
| `nexus_harder_tests*.py`, `nexus_evaluate.py`, `nexus_diagnose.py` | 测试/评估 |
| `ai_doctor_pipeline.py` | 指向不存在的 `nexus_runner.py` 的兼容 shim |
| `cascade_editor.py`, `nexus_sync_config.py` | 开发工具 |
| `memory_vectors.index.py` | 空 stub |
| `syndrome_config copy.json` | `syndrome_config.json` 的重复副本 |

---

## 数据文件依赖（生产路径实际加载）
- **系统 B / Demo**：`drugs_pkpd.json`（26 药，driver）、`demo_clinical_data.json`（可选标签，Demo 用）、`sample_cases.json`（预置病例）。
- **系统 A**：`state_models/*.json`、`organ_function.json`、`disease_mechanism_map.json`、`knowledge_graph.json`、`anatomy/*`、`symptom_*`（多数期望位于 `medical_knowledge/` 目录树下，仓库根目录散落同名文件）。

---

## 新增 Demo 的文件（不改动任何现有逻辑文件）
| 文件 | 作用 |
|---|---|
| `case_targets.py` | 病例适应症 → 生理 targets 的桥接（15 个适应症，含别名归一与自检）|
| `sample_cases.json` | 5 个预置病例 |
| `demo_clinical_data.json` | **示意性**标签数据（非真实 FDA、非临床），让安全门+验证剂量在 Demo 中真正触发 |
| `llm_client.py` | provider 无关的聊天封装（OpenAI 默认 / Gemini 可选 / 无 key 优雅降级）|
| `demo_app.py` | 新 Flask 后端：`/`、`/api/cases`、`/api/recommend_case`、`/api/chat`（仅机制或机制+标签）|
| `case_demo.html` | 三栏 UI：左=病例导入、中=聊天、右=推理链侧边栏（复用 `chat.html` 视觉风格）|

**运行**：`py demo_app.py` → http://127.0.0.1:5000 。聊天可选：设 `OPENAI_API_KEY`（或 `GEMINI_API_KEY` + `LLM_PROVIDER=gemini`）。
