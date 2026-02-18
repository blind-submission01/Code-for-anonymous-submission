# ESS / RAS / ESRA

本目录提供三个“关键单元对齐”指标：

- **RAS**：Repair Action Score（对齐 VulAdvisor 思路），衡量 Repair Suggestion 中关键“动作单元”的覆盖情况。
- **ESS**：Error State Score（你的创新），衡量 Root Cause 中关键“缺陷概念”的覆盖情况。
- **ESRA**：`(ESS + RAS) / 2` 的综合分数（如果你需要）。

## 与 VulAdvisor 的关系

- RAS 的抽取与评分对齐 VulAdvisor：基于依存句法抽取 `(verb_lemma, object_phrase)`，然后用 **generalized Jaccard（Counter 多重集）** 评分。
- ESS 不来自 VulAdvisor，是你在 Root Cause 侧的扩展：抽取缺陷概念后用 **set Jaccard** 评分。

## 依赖

```bash
pip install spacy
python -m spacy download en_core_web_sm
```

## 推荐入口

使用 `ESS_and_RAS/esra_score.py`：

```python
from ESS_and_RAS.esra_score import compute_ess, compute_ras, compute_esra

rc_ref = "Root cause: missing bounds check when copying user-controlled length."
rc_pred = "The bug is due to a missing bounds check on the copy length."

rs_ref = "Add a bounds check before copying and validate the length."
rs_pred = "Add checks to validate length before copying data."

ess = compute_ess(rc_ref, rc_pred)  # 0~1，空集默认 0.0
ras = compute_ras(rs_ref, rs_pred)  # 0~1，空集默认 1.0（对齐 VulAdvisor）
esra = compute_esra(rc_ref, rc_pred, rs_ref, rs_pred)

print(ess, ras, esra)
```

## 抽取函数（可用于调试）

`ESS_and_RAS/extraction_phrase.py` 提供抽取逻辑：

- `extract_repair_action_tuples(...)`：返回 `[(verb, object), ...]`（RAS 用）
- `extract_root_cause_concepts(...)`：返回 `["concept", ...]`（ESS 用）
- `extract_phrases(sentence, strategy=...)`：统一入口（用于快速查看抽取结果）

