"""
BERTScore 计算函数：输入生成句子与参考句子，输出 P/R/F1.
"""
from __future__ import annotations

from typing import Dict, Tuple

from bert_score import BERTScorer


def compute_bertscore(
    candidate: str,
    reference: str,
    model_type: str = "bert-base-uncased",
) -> Tuple[float, float, float]:
    """
    计算单个句子的 BERTScore.

    返回 (precision, recall, f1).
    """
    scorer = _get_scorer(model_type=model_type)
    precision, recall, f1 = scorer.score([candidate], [reference])
    return (float(precision.mean()), float(recall.mean()), float(f1.mean()))


_SCORER_CACHE: Dict[str, BERTScorer] = {}


def _get_scorer(model_type: str) -> BERTScorer:
    """
    VulAdvisor 参考配置:
      - BERTScorer(model_type='bert-base-uncased')
    """
    scorer = _SCORER_CACHE.get(model_type)
    if scorer is None:
        scorer = BERTScorer(model_type=model_type)
        _SCORER_CACHE[model_type] = scorer
    return scorer


if __name__ == "__main__":
    gen = "The cat is on the mat."
    ref = "There is a cat on the mat."
    p, r, f = compute_bertscore(gen, ref)
    print(f"P: {p:.4f}, R: {r:.4f}, F1: {f:.4f}")
