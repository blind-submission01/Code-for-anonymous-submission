# """
# BLEU 计算（对齐 VulAdvisor 的 sacrebleu 口径）。
# """
# from __future__ import annotations

# from typing import Iterable, List, Sequence, Union


# StrOrStrs = Union[str, Sequence[str]]


# def compute_bleu(
#     predictions: StrOrStrs,
#     references: StrOrStrs,
#     max_n: int = 4,
# ) -> float:
#     """
#     返回 sacrebleu 的 BLEU 分数（0-100）。

#     参考实现（VulAdvisor）:
#       evaluate.load(\"sacrebleu\").compute(predictions=preds, references=[[ref] for ref in refs])[\"score\"]
#     """
#     if max_n <= 0:
#         raise ValueError("max_n 必须大于 0")

#     preds = _ensure_list(predictions)
#     refs = _ensure_list(references)
#     if len(preds) != len(refs):
#         raise ValueError(f"predictions 与 references 长度不一致: {len(preds)} vs {len(refs)}")

#     try:
#         from evaluate import load as load_metric  # type: ignore
#     except Exception:
#         load_metric = None

#     if load_metric is not None and max_n == 4:
#         bleu = load_metric("sacrebleu")
#         return float(
#             bleu.compute(predictions=preds, references=[[r] for r in refs])["score"]
#         )

#     from sacrebleu.metrics import BLEU  # type: ignore

#     bleu = BLEU(n_gram_order=max_n)
#     score = bleu.corpus_score(preds, [refs]).score
#     return float(score)
"""
BLEU 计算（对齐 VulAdvisor 的 sacrebleu 口径）。
"""
from __future__ import annotations
from typing import List, Sequence, Union
from sacrebleu.metrics import BLEU

StrOrStrs = Union[str, Sequence[str]]

# 全局缓存
_BLEU = BLEU()


def compute_bleu(
    predictions: StrOrStrs,
    references: StrOrStrs,
    max_n: int = 4,
) -> float:
    """
    返回 sacrebleu 的 BLEU 分数（0-100）。
    """
    preds = _ensure_list(predictions)
    refs = _ensure_list(references)
    
    return float(_BLEU.corpus_score(preds, [refs]).score)


def _ensure_list(value: StrOrStrs) -> List[str]:
    if isinstance(value, str):
        return [value]
    return list(value)


if __name__ == "__main__":
    gen = "the cat is on the mat"
    ref = "there is a cat on the mat"
    score = compute_bleu(gen, ref)
    print(f"BLEU-4(sacrebleu): {score:.4f}")
