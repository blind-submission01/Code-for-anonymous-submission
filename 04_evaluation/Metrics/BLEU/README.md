# BLEU (sacrebleu)

本目录提供 BLEU-4 的计算实现，**对齐 VulAdvisor 使用的 sacrebleu 口径**。

## 与参考工作（VulAdvisor）对齐说明

- VulAdvisor 使用 `evaluate.load("sacrebleu")` 计算 BLEU（返回 **0~100** 分数）。
- 本实现优先复用同样的 `evaluate` 调用；若环境没有 `evaluate`，则 fallback 到 `sacrebleu` 原生实现。

## 依赖

推荐（与 VulAdvisor 写法一致）：

```bash
pip install evaluate sacrebleu
```

## 用法（Python）

单条：

```python
from BLEU.bleu_score import compute_bleu

pred = "Add a bounds check before copying."
ref = "Add bounds checking before memcpy."

bleu4 = compute_bleu(pred, ref)  # 默认 BLEU-4，返回 0~100
print(bleu4)
```

批量：

```python
from BLEU.bleu_score import compute_bleu

preds = ["a b c", "hello world"]
refs = ["a b d", "hello world!"]

bleu4 = compute_bleu(preds, refs)
print(bleu4)
```

## 参数

- `max_n`：n-gram 阶数（默认 4；若要严格复现 VulAdvisor/论文中的 BLEU-4，不要改）。

