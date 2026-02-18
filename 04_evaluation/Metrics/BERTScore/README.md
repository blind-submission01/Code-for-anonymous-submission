# BERTScore

本目录提供 `BERTScore` 的最小可复现实用实现，用于对比**生成文本**与**参考文本**的语义相似度。

## 与参考工作（VulAdvisor）对齐说明

- 本实现默认使用 `model_type="bert-base-uncased"`，对齐 VulAdvisor 仓库中的配置（`BERTScorer(model_type='bert-base-uncased')`）。
- 返回值为 0~1 的浮点数（`P/R/F1`）。论文表格若以百分制展示，通常是 `score * 100`。

## 依赖

```bash
pip install bert-score
```

## 用法（Python）

```python
from BERTScore.bertscore_calc import compute_bertscore

candidate = "Add a bounds check before copying."
reference = "Add bounds checking before memcpy."

p, r, f1 = compute_bertscore(candidate, reference)  # 默认 bert-base-uncased
print(p, r, f1)
```

## 可选参数

- `model_type`：BERTScore backbone（默认 `"bert-base-uncased"`）。

