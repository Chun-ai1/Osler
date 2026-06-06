# 用藥推理報告
**目標狀態**：lung.bronchospasm 需往 `low` 推　|　**適應症**：asthma
**病人**：年齡 55，腎功能 normal，過敏 無，現有用藥 metoprolol，臨床旗標 tachycardia

## epinephrine — 資料不足（不給劑量）（risk: unknown）
- **機制**：lung.bronchospasm → low (max_delta -0.5)，機制上與目標狀態相關 [1]。
- **適應症**：尚未載入仿單，無法確認是否核可。
- **劑量**：無來源劑量，引擎**不生成數字**——需先抓取仿單後才會顯示。
- **風險**：no label entry; run clinical_data.build_entry()。
- **結論**：Cannot recommend a dose — no label-sourced dose loaded; run clinical_data.build_entry().

## epi pen — 資料不足（不給劑量）（risk: unknown）
- **機制**：lung.bronchospasm → low (max_delta -0.3)，機制上與目標狀態相關 [2]。
- **適應症**：尚未載入仿單，無法確認是否核可。
- **劑量**：無來源劑量，引擎**不生成數字**——需先抓取仿單後才會顯示。
- **風險**：no label entry; run clinical_data.build_entry()。
- **結論**：Cannot recommend a dose — no label-sourced dose loaded; run clinical_data.build_entry().

## albuterol — 資料不足（不給劑量）（risk: unknown）
- **機制**：lung.bronchospasm → low (max_delta -0.5)，機制上與目標狀態相關 [3]。
- **適應症**：此用途在仿單中列為核可適應症 [4]。
- **劑量**：無來源劑量，引擎**不生成數字**——需先抓取仿單後才會顯示。
- **風險**：tachycardia (HR 135 > 100) referenced in label safety text [5]。
- **風險**：no label-sourced dose available; run clinical_data.build_entry()。
- **結論**：Cannot recommend a dose — no label-sourced dose loaded; run clinical_data.build_entry().

## ventolin — 資料不足（不給劑量）（risk: unknown）
- **機制**：lung.bronchospasm → low (max_delta -0.3)，機制上與目標狀態相關 [6]。
- **適應症**：尚未載入仿單，無法確認是否核可。
- **劑量**：無來源劑量，引擎**不生成數字**——需先抓取仿單後才會顯示。
- **風險**：no label entry; run clinical_data.build_entry()。
- **結論**：Cannot recommend a dose — no label-sourced dose loaded; run clinical_data.build_entry().

---
## 參考來源
[1] 內部機制圖 (drug_mechanisms / state_effects) — epi reverses anaphylactic vasodilation/bronchospasm  （內部機制圖，估計值，非臨床劑量依據）
[2] 內部機制圖 (drug_mechanisms / state_effects) — EpiPen → anaphylaxis-specific  （內部機制圖，估計值，非臨床劑量依據）
[3] 內部機制圖 (drug_mechanisms / state_effects) — beta-2 agonist relaxes bronchial smooth muscle  （內部機制圖，估計值，非臨床劑量依據）
[4] FDA 仿單 (openFDA/DailyMed SPL) — Indications and Usage  （尚未抓取仿單，run clinical_data.build_entry()）
[5] FDA 仿單 (openFDA/DailyMed SPL) — Contraindications/Warnings/Specific Populations  （尚未抓取仿單，run clinical_data.build_entry()）
[6] 內部機制圖 (drug_mechanisms / state_effects) — albuterol brand  （內部機制圖，估計值，非臨床劑量依據）

> DECISION SUPPORT ONLY. Doses are label-extracted, never generated. FAERS data is signal, not causation. Final clinical decision requires a licensed clinician.


# 用藥推理報告
**目標狀態**：lung.bronchospasm 需往 `low` 推　|　**適應症**：asthma
**病人**：年齡 55，腎功能 normal，過敏 無，現有用藥 metoprolol，臨床旗標 tachycardia

## albuterol — 謹慎（需警示）（risk: moderate）
- **機制**：lung.bronchospasm → low (max_delta -0.5)，機制上與目標狀態相關 [1]。
- **適應症**：此用途在仿單中列為核可適應症 [2]。
- **劑量**：仿單 Dosage and Administration 段所載：（仿單 Dosage and Administration 原文）（inhalation，（仿單原文）） [3]。
- **風險**：tachycardia (HR 135 > 100) referenced in label safety text [4]。
- **不良事件訊號 (FAERS)**：tremor（1234 筆）、tachycardia（890 筆） [5]。
- **結論**：Label dose may be shown WITH warning; clinician confirms.

## epinephrine — 資料不足（不給劑量）（risk: unknown）
- **機制**：lung.bronchospasm → low (max_delta -0.5)，機制上與目標狀態相關 [6]。
- **適應症**：尚未載入仿單，無法確認是否核可。
- **劑量**：無來源劑量，引擎**不生成數字**——需先抓取仿單後才會顯示。
- **風險**：no label entry; run clinical_data.build_entry()。
- **結論**：Cannot recommend a dose — no label-sourced dose loaded; run clinical_data.build_entry().

## epi pen — 資料不足（不給劑量）（risk: unknown）
- **機制**：lung.bronchospasm → low (max_delta -0.3)，機制上與目標狀態相關 [7]。
- **適應症**：尚未載入仿單，無法確認是否核可。
- **劑量**：無來源劑量，引擎**不生成數字**——需先抓取仿單後才會顯示。
- **風險**：no label entry; run clinical_data.build_entry()。
- **結論**：Cannot recommend a dose — no label-sourced dose loaded; run clinical_data.build_entry().

## ventolin — 資料不足（不給劑量）（risk: unknown）
- **機制**：lung.bronchospasm → low (max_delta -0.3)，機制上與目標狀態相關 [8]。
- **適應症**：尚未載入仿單，無法確認是否核可。
- **劑量**：無來源劑量，引擎**不生成數字**——需先抓取仿單後才會顯示。
- **風險**：no label entry; run clinical_data.build_entry()。
- **結論**：Cannot recommend a dose — no label-sourced dose loaded; run clinical_data.build_entry().

---
## 參考來源
[1] 內部機制圖 (drug_mechanisms / state_effects) — beta-2 agonist relaxes bronchial smooth muscle  （內部機制圖，估計值，非臨床劑量依據）
[2] FDA 仿單 (openFDA/DailyMed SPL) — Indications and Usage — set_id=01c77e6a-6381-4005-b647-481bbcd442aa  https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=01c77e6a-6381-4005-b647-481bbcd442aa  （label effective 2024-01-15）
[3] FDA 仿單 (openFDA/DailyMed SPL) — Dosage and Administration — set_id=01c77e6a-6381-4005-b647-481bbcd442aa  https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=01c77e6a-6381-4005-b647-481bbcd442aa  （label effective 2024-01-15）
[4] FDA 仿單 (openFDA/DailyMed SPL) — Contraindications/Warnings/Specific Populations — set_id=01c77e6a-6381-4005-b647-481bbcd442aa  https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=01c77e6a-6381-4005-b647-481bbcd442aa  （label effective 2024-01-15）
[5] openFDA drug/event (FAERS) — 真實世界通報  （僅為關聯訊號，非因果；資料可能延遲數月）
[6] 內部機制圖 (drug_mechanisms / state_effects) — epi reverses anaphylactic vasodilation/bronchospasm  （內部機制圖，估計值，非臨床劑量依據）
[7] 內部機制圖 (drug_mechanisms / state_effects) — EpiPen → anaphylaxis-specific  （內部機制圖，估計值，非臨床劑量依據）
[8] 內部機制圖 (drug_mechanisms / state_effects) — albuterol brand  （內部機制圖，估計值，非臨床劑量依據）

> DECISION SUPPORT ONLY. Doses are label-extracted, never generated. FAERS data is signal, not causation. Final clinical decision requires a licensed clinician.